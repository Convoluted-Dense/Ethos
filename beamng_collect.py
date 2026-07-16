"""
beamng_collect.py
=================
Captures raw frames from the BeamNG.drive window (via PrintWindow, works across
virtual desktops) at ~10 FPS and simultaneously receives MotionSim UDP telemetry.
Every captured frame is saved to  dataset/img/<timestamp>.jpg  and every matching
telemetry packet is saved as one row in  dataset/telemetry.csv.

Usage
-----
    python beamng_collect.py               # save to ./dataset at 10 FPS
    python beamng_collect.py --view        # same + live HUD overlay window
    python beamng_collect.py --out mydata  # custom output directory
    python beamng_collect.py --fps 5       # slower capture rate

BeamNG setup
------------
  Options -> Other -> MotionSim UDP  ->  enabled, IP=127.0.0.1, Port=4444
"""

import argparse
import ctypes
import csv
import math
import socket
import struct
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import cv2
import numpy as np

# Optional vJoy for camera-shift loop
try:
    import pyvjoy as _pyvjoy
    _VJOY_OK = True
except ImportError:
    _VJOY_OK = False

# ---------------------------------------------------------------------------
# Camera-shift / steering-offset constants
# ---------------------------------------------------------------------------
BUTTON_B      = 2      # centre camera
BUTTON_X      = 3      # left drift
BUTTON_Y      = 4      # right drift
MAX_DRIFT     = 0.25   # max camera offset correction (25%)
CENTRE_HOLD   = 10.0   # seconds to hold centre before drifting
DRIFT_HOLD    = 3.0    # seconds to hold X or Y
DRIFT_RATE    = MAX_DRIFT / DRIFT_HOLD  # offset units per second while X or Y is held

# ---------------------------------------------------------------------------
# MotionSim UDP packet layout  (BNG1 format, 88 bytes: 4 tag + 21 floats × 4)
# ---------------------------------------------------------------------------
BNG1_FORMAT = "4s" + "f" * 21   # 4-byte tag  +  21 floats (7 groups × 3)
BNG1_SIZE   = struct.calcsize(BNG1_FORMAT)  # = 88 bytes
BNG1_FIELDS = [
    "posX",  "posY",  "posZ",
    "velX",  "velY",  "velZ",
    "accX",  "accY",  "accZ",
    "upX",   "upY",   "upZ",
    "rollPos",  "pitchPos",  "yawPos",
    "rollVel",  "pitchVel",  "yawVel",
    "rollAcc",  "pitchAcc",  "yawAcc",
]
CSV_HEADER = ["frame", "capture_time", "steering", "steering_offset", "steering_combined"] + BNG1_FIELDS

UDP_IP   = "0.0.0.0"
UDP_PORT = 4444
ROI_FILE = "steering_roi.json"


# ---------------------------------------------------------------------------
# Window helpers (same PrintWindow approach as beamng_yolo.py)
# ---------------------------------------------------------------------------
class WindowInfo:
    def __init__(self, hwnd, title):
        self.hwnd  = hwnd
        self.title = title


def find_beamng_window():
    """Return the first visible window whose title contains 'beamng.drive'."""
    user32 = ctypes.windll.user32
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    found = []

    def callback(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n > 0:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if "beamng.drive" in buf.value.lower():
                    found.append(WindowInfo(hwnd, buf.value))
                    return False
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)

    if found:
        win = found[0]
        print(f"[window] Found: '{win.title}'  HWND={win.hwnd}")
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        return win
    return None


def capture_printwindow(hwnd):
    """
    Capture window contents via PrintWindow with PW_RENDERFULLCONTENT=2.
    Works even when the window is on a different virtual desktop.
    Returns a BGR numpy array or None if the window has zero client area.
    """
    user32 = ctypes.windll.user32
    gdi32  = ctypes.windll.gdi32

    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    w = rect.right  - rect.left
    h = rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None

    hwndDC     = user32.GetWindowDC(hwnd)
    mfcDC      = gdi32.CreateCompatibleDC(hwndDC)
    saveBitmap = gdi32.CreateCompatibleBitmap(hwndDC, w, h)
    gdi32.SelectObject(mfcDC, saveBitmap)
    user32.PrintWindow(hwnd, mfcDC, 2)   # PW_RENDERFULLCONTENT

    bmi = bytearray(40)
    bmi[0:4]   = (40).to_bytes(4, "little")
    bmi[4:8]   = w.to_bytes(4, "little", signed=True)
    bmi[8:12]  = (-h).to_bytes(4, "little", signed=True)
    bmi[12:14] = (1).to_bytes(2, "little")
    bmi[14:16] = (32).to_bytes(2, "little")

    buf = bytearray(w * h * 4)
    gdi32.GetDIBits(mfcDC, saveBitmap, 0, h,
                    ctypes.byref(ctypes.c_char.from_buffer(buf)),
                    ctypes.byref(ctypes.c_char.from_buffer(bmi)), 0)

    gdi32.DeleteObject(saveBitmap)
    gdi32.DeleteDC(mfcDC)
    user32.ReleaseDC(hwnd, hwndDC)

    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    return img[:, :, :3].copy()   # BGRA -> BGR


def load_steering_roi():
    import json
    path = Path(ROI_FILE)
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
                roi = tuple(data["roi"]) if "roi" in data and data["roi"] else None
                map_roi = tuple(data["map_roi"]) if "map_roi" in data and data["map_roi"] else None
                return roi, map_roi
        except Exception as e:
            print(f"[warn] Failed to load {ROI_FILE}: {e}")
    return None, None


def extract_steering(img, roi):
    if roi is None or img is None:
        return 0.0
        
    x, y, w, h = roi
    # ensure bounds
    img_h, img_w = img.shape[:2]
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        return 0.0
        
    crop = img[y:y+h, x:x+w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    mid = w // 2
    left_half = mask[:, :mid]
    right_half = mask[:, mid:]
    
    left_orange = np.count_nonzero(left_half)
    right_orange = np.count_nonzero(right_half)
    
    # FIX (Bug 3): use the full crop half-area as the denominator instead of the
    # bounding-box height of detected orange pixels.  If stray orange HUD elements
    # appear at very different Y positions the bounding rect height can jump wildly,
    # making the normalized value meaningless.
    max_area = mid * h  # h is the full crop height (consistent, never jumps)
    if max_area > 0 and (left_orange > 0 or right_orange > 0):
        left_val  = left_orange  / max_area
        right_val = right_orange / max_area
        steering  = right_val - left_val
        return max(-1.0, min(1.0, steering))
            
    return 0.0


# ---------------------------------------------------------------------------
# UDP MotionSim telemetry receiver
# ---------------------------------------------------------------------------
class TelemetryReceiver:
    """
    Daemon thread that reads BNG1 UDP packets and stores the latest one.
    Call .get() from the main thread to obtain a copy.
    """

    def __init__(self, ip=UDP_IP, port=UDP_PORT):
        self._lock      = threading.Lock()
        self._latest    = None
        self.pkt_total  = 0   # any UDP packet received
        self.pkt_valid  = 0   # correctly parsed BNG1 packets

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Increase receive buffer so fast bursts are not dropped
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._sock.bind((ip, port))
        self._sock.settimeout(0.5)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[udp]    Listening on {ip}:{port}  (BNG1_SIZE={BNG1_SIZE} bytes expected)")

    def _loop(self):
        first_packet = True
        while True:
            try:
                data, addr = self._sock.recvfrom(512)
            except socket.timeout:
                continue
            except Exception:
                break

            self.pkt_total += 1

            if first_packet:
                print(f"\n[udp]    First packet: {len(data)} bytes from {addr}  tag={data[:4]}")
                first_packet = False

            if len(data) < BNG1_SIZE:
                # packet too small — print once to help diagnose
                if self.pkt_total <= 3:
                    print(f"[udp]    Short packet: {len(data)} bytes (need {BNG1_SIZE}) — skipping")
                continue

            parts = struct.unpack_from(BNG1_FORMAT, data)
            if parts[0][:4] != b"BNG1":
                if self.pkt_total <= 3:
                    print(f"[udp]    Unknown tag: {parts[0][:4]} — skipping")
                continue

            row = {}
            for i, name in enumerate(BNG1_FIELDS):
                row[name] = parts[1 + i]
            self.pkt_valid += 1
            with self._lock:
                self._latest = row

    def get(self):
        """Return a copy of the latest telemetry dict, or None."""
        with self._lock:
            return dict(self._latest) if self._latest else None

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Camera shift controller  (background thread)
# ---------------------------------------------------------------------------
class CameraShiftController:
    """
    Runs an infinite loop that presses vJoy buttons to shift the in-game
    camera left (X) and right (Y) with a centre reset (B) in between.
    Exposes a thread-safe `offset` float and a `phase` string.
    """

    def __init__(self):
        self._lock   = threading.Lock()
        self._offset = 0.0
        self._phase  = "disabled"
        self._joy    = None
        self._running = True

        if not _VJOY_OK:
            print("[shift]  pyvjoy not available — camera-shift loop disabled.")
            return

        try:
            self._joy = _pyvjoy.VJoyDevice(1)
            # Release all buttons on startup
            for btn in (BUTTON_B, BUTTON_X, BUTTON_Y):
                self._joy.set_button(btn, 0)
            print("[shift]  vJoy Device 1 acquired — camera-shift loop enabled.")
        except Exception as e:
            print(f"[shift]  Failed to acquire vJoy: {e} — camera-shift loop disabled.")
            self._joy = None

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ── thread-safe accessors ────────────────────────────────
    @property
    def offset(self) -> float:
        with self._lock:
            return self._offset

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    def _set(self, offset, phase):
        with self._lock:
            self._offset = offset
            self._phase  = phase

    # ── button helpers ───────────────────────────────────────
    def _press(self, btn):
        if self._joy:
            self._joy.set_button(btn, 1)

    def _release(self, btn):
        if self._joy:
            self._joy.set_button(btn, 0)

    def _press_release(self, btn, hold=0.1):
        """Tap a button for `hold` seconds then release."""
        self._press(btn)
        time.sleep(hold)
        self._release(btn)

    # ── main shift loop ──────────────────────────────────────
    def _loop(self):
        if self._joy is None:
            return

        while self._running:
            # ── CENTRE phase ─────────────────────────────────
            self._press_release(BUTTON_B, hold=0.15)
            self._set(0.0, "centre")
            time.sleep(CENTRE_HOLD)

            # ── LEFT drift (X) phase ─────────────────────────
            self._press(BUTTON_X)
            t0 = time.perf_counter()
            while True:
                elapsed = time.perf_counter() - t0
                if elapsed >= DRIFT_HOLD:
                    break
                with self._lock:
                    self._offset = min(MAX_DRIFT, DRIFT_RATE * elapsed)
                    self._phase  = "left"
                time.sleep(0.05)
            self._release(BUTTON_X)

            # ── CENTRE reset ─────────────────────────────────
            self._press_release(BUTTON_B, hold=0.15)
            self._set(0.0, "centre")
            time.sleep(0.5)   # brief pause so game registers centre

            # ── RIGHT drift (Y) phase ────────────────────────
            self._press(BUTTON_Y)
            t0 = time.perf_counter()
            while True:
                elapsed = time.perf_counter() - t0
                if elapsed >= DRIFT_HOLD:
                    break
                with self._lock:
                    self._offset = max(-MAX_DRIFT, -DRIFT_RATE * elapsed)
                    self._phase  = "right"
                time.sleep(0.05)
            self._release(BUTTON_Y)

            # ── CENTRE reset ─────────────────────────────────
            self._press_release(BUTTON_B, hold=0.15)
            self._set(0.0, "centre")
            time.sleep(0.5)

    def close(self):
        self._running = False
        if self._joy:
            for btn in (BUTTON_B, BUTTON_X, BUTTON_Y):
                try:
                    self._joy.set_button(btn, 0)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Dataset writer  (JPEG + CSV)
# ---------------------------------------------------------------------------
class DatasetWriter:
    def __init__(self, out_dir: Path):
        self.img_dir = out_dir / "img"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        
        self.map_dir = out_dir / "map"
        self.map_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "telemetry.csv"
        file_exists = csv_path.exists()
        self._csv_fh = open(csv_path, "a", newline="", encoding="utf-8")
        self._writer  = csv.DictWriter(self._csv_fh, fieldnames=CSV_HEADER)
        if not file_exists:
            self._writer.writeheader()
        self._csv_fh.flush()

        print(f"[writer] Images  -> {self.img_dir}")
        print(f"[writer] Maps    -> {self.map_dir}")
        print(f"[writer] CSV     -> {csv_path}")

    def write(self, frame_name: str, img: np.ndarray, map_img: np.ndarray | None, capture_time: float, telem, steering: float, steering_offset: float = 0.0, steering_combined: float = 0.0):
        # --- save JPEG ---
        cv2.imwrite(str(self.img_dir / frame_name), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        # --- save MAP if exists ---
        if map_img is not None:
            cv2.imwrite(str(self.map_dir / frame_name), map_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            
        # --- write CSV row (empty strings where telemetry unavailable) ---
        row = {"frame": frame_name, "capture_time": f"{capture_time:.6f}",
               "steering": f"{steering:.4f}",
               "steering_offset": f"{steering_offset:.4f}",
               "steering_combined": f"{steering_combined:.4f}"}
        for f in BNG1_FIELDS:
            row[f] = f"{telem[f]:.6f}" if telem and f in telem else ""
        self._writer.writerow(row)
        self._csv_fh.flush()

    def close(self):
        self._csv_fh.close()


# ---------------------------------------------------------------------------
# Live HUD rendering
# ---------------------------------------------------------------------------
_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_CLR_HUD   = (0, 255, 180)
_CLR_WARN  = (0, 120, 255)
_CLR_SHD   = (0, 0, 0)
_FS        = 0.60
_FT        = 2


def _put(img, text, x, y, color=_CLR_HUD):
    cv2.putText(img, text, (x+1, y+1), _FONT, _FS, _CLR_SHD, _FT+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y  ), _FONT, _FS, color,     _FT,   cv2.LINE_AA)


def draw_hud(img: np.ndarray, telem, fps: float, saved: int, pkt_total: int = 0, pkt_valid: int = 0, steering: float = 0.0, offset: float = 0.0, phase: str = "disabled", steering_combined: float = 0.0) -> np.ndarray:
    vis = img.copy()
    ph, pw = 325, 500

    # semi-transparent dark panel
    panel = vis[:ph, :pw].copy()
    cv2.rectangle(panel, (0, 0), (pw, ph), (8, 8, 8), -1)
    cv2.addWeighted(panel, 0.55, vis[:ph, :pw], 0.45, 0, vis[:ph, :pw])

    y = 25
    _put(vis, f"Capture  {fps:5.1f} fps    saved: {saved}", 10, y)
    y += 22
    _put(vis, f"UDP  rcvd: {pkt_total}  valid BNG1: {pkt_valid}", 10, y, (180, 180, 180))
    
    y += 22
    # Draw visual steering bar (Raw)
    _put(vis, f"Steer (Raw) {steering:+.2f}", 10, y)
    bar_w = 200
    bar_x = 180
    cv2.rectangle(vis, (bar_x, y-12), (bar_x + bar_w, y+2), (60, 60, 60), -1)
    mid_x = bar_x + bar_w // 2
    if steering > 0:
        cv2.rectangle(vis, (mid_x, y-12), (mid_x + int(steering * bar_w/2), y+2), (0, 120, 255), -1)
    elif steering < 0:
        cv2.rectangle(vis, (mid_x + int(steering * bar_w/2), y-12), (mid_x, y+2), (0, 120, 255), -1)
    cv2.line(vis, (mid_x, y-14), (mid_x, y+4), (255, 255, 255), 1)

    y += 22
    # Draw visual steering bar (Comb)
    _put(vis, f"Steer (Comb) {steering_combined:+.2f}", 10, y)
    cv2.rectangle(vis, (bar_x, y-12), (bar_x + bar_w, y+2), (60, 60, 60), -1)
    mid_x = bar_x + bar_w // 2
    if steering_combined > 0:
        cv2.rectangle(vis, (mid_x, y-12), (mid_x + int(steering_combined * bar_w/2), y+2), (0, 120, 255), -1)
    elif steering_combined < 0:
        cv2.rectangle(vis, (mid_x + int(steering_combined * bar_w/2), y-12), (mid_x, y+2), (0, 120, 255), -1)
    cv2.line(vis, (mid_x, y-14), (mid_x, y+4), (255, 255, 255), 1)

    # Offset and phase
    phase_color = {"left": (0, 200, 255), "right": (255, 150, 0), "centre": (0, 255, 120)}.get(phase, (160, 160, 160))
    y += 22
    _put(vis, f"Offset   {offset:+.3f}   [{phase}]", 10, y, phase_color)

    if telem is None:
        y += 28
        if pkt_total == 0:
            _put(vis, "No UDP packets — check BeamNG MotionSim settings", 10, y, _CLR_WARN)
        else:
            _put(vis, f"Packets arriving but no valid BNG1 (got {pkt_total})", 10, y, _CLR_WARN)
        return vis

    # Speed
    speed = math.sqrt(telem["velX"]**2 + telem["velY"]**2 + telem["velZ"]**2) * 3.6
    y += 30; _put(vis, f"Speed    {speed:8.2f} km/h", 10, y)

    # Longitudinal / lateral acceleration (in vehicle frame approx.)
    ax, ay, az = telem["accX"], telem["accY"], telem["accZ"]
    acc_total   = math.sqrt(ax**2 + ay**2 + az**2) / 9.81
    y += 26; _put(vis, f"Accel    {acc_total:8.3f} g   ({ax:.2f},{ay:.2f},{az:.2f} m/s^2)", 10, y)

    # Attitude
    roll  = math.degrees(telem["rollPos"])
    pitch = math.degrees(telem["pitchPos"])
    yaw   = math.degrees(telem["yawPos"])
    y += 26; _put(vis, f"Roll  {roll:+7.2f} deg   Pitch {pitch:+7.2f} deg   Yaw {yaw:+7.2f} deg", 10, y)

    # Angular velocity
    rv, pv, yv = telem["rollVel"], telem["pitchVel"], telem["yawVel"]
    y += 26; _put(vis, f"AngVel   r={rv:+.3f}  p={pv:+.3f}  y={yv:+.3f}  rad/s", 10, y)

    # Angular acceleration
    ra, pa, ya = telem["rollAcc"], telem["pitchAcc"], telem["yawAcc"]
    y += 26; _put(vis, f"AngAcc   r={ra:+.3f}  p={pa:+.3f}  y={ya:+.3f}  rad/s^2", 10, y)

    # World position
    y += 26; _put(vis, f"Pos ({telem['posX']:.1f}, {telem['posY']:.1f}, {telem['posZ']:.1f})", 10, y)

    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def make_parser():
    p = argparse.ArgumentParser(
        description="BeamNG data collector — PrintWindow frames + MotionSim UDP telemetry")
    p.add_argument("--out",    default="dataset",
                   help="root output directory  (default: dataset)")
    p.add_argument("--fps",    type=float, default=10.0,
                   help="target capture FPS  (default: 10)")
    p.add_argument("--port",   type=int,   default=4444,
                   help="UDP port for MotionSim  (default: 4444)")
    p.add_argument("--width",  type=int,   default=1280,
                   help="resize captured frames to this width  (default: 1280)")
    p.add_argument("--height", type=int,   default=720,
                   help="resize captured frames to this height (default: 720)")
    p.add_argument("--view",   action="store_true",
                   help="show live OpenCV window with telemetry HUD")
    return p


def main():
    opt = make_parser().parse_args()
    frame_interval = 1.0 / opt.fps
    out_dir = Path(opt.out)

    # ── locate game window ──────────────────────────────────
    print("[startup] Searching for BeamNG.drive window …")
    win = find_beamng_window()
    if win is None:
        print("ERROR: BeamNG.drive window not found. Make sure the game is running.")
        sys.exit(1)

    # ── load ROIs ───────────────────────────────────────────
    roi, map_roi = load_steering_roi()
    if roi:
        print(f"[startup] Loaded steering ROI: {roi}")
    else:
        print("[startup] No steering_roi.json found. Steering will be 0.0.")
    if map_roi:
        print(f"[startup] Loaded map ROI: {map_roi}")
    else:
        print("[startup] No map ROI found. Map will not be extracted.")
    print("          Run `python calibrate_steering.py` to set them up.")

    # ── telemetry thread ────────────────────────────────────
    telem_rx = TelemetryReceiver(port=opt.port)

    # ── dataset writer ──────────────────────────────────────
    writer = DatasetWriter(out_dir)

    # ── camera shift controller ─────────────────────────────
    shift_ctrl = CameraShiftController()

    print(f"\n[collect] Running at {opt.fps} FPS  ->  {out_dir}/")
    if opt.view:
        print("[collect] Live view active — press  q  to stop.")
    else:
        print("[collect] Press Ctrl-C to stop.")
    print()

    saved       = 0
    fps_display = 0.0
    fps_frames  = 0
    t_fps_ref   = time.perf_counter()

    # ── failsafe state ──────────────────────────────────────
    failsafe_ref_pos = None
    failsafe_ref_time = time.time()

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Capture
            raw = capture_printwindow(win.hwnd)
            if raw is None:
                time.sleep(0.05)
                continue

            # 2. Extract steering and map before resize
            steering = extract_steering(raw, roi)
            
            map_img = None
            if map_roi:
                mx, my, mw, mh = map_roi
                if mx >= 0 and my >= 0 and mx + mw <= raw.shape[1] and my + mh <= raw.shape[0]:
                    m_crop = raw[my:my+mh, mx:mx+mw]
                    # Convert to grayscale
                    m_gray = cv2.cvtColor(m_crop, cv2.COLOR_BGR2GRAY)
                    # Convert to binary (white pixels on black background)
                    # Usually roads on the minimap are light colored, background is dark
                    _, map_img = cv2.threshold(m_gray, 180, 255, cv2.THRESH_BINARY)
            
            # 3. Resize
            if raw.shape[1] != opt.width or raw.shape[0] != opt.height:
                raw = cv2.resize(raw, (opt.width, opt.height),
                                 interpolation=cv2.INTER_LINEAR)

            # 4. Snapshot of latest telemetry
            capture_time = time.time()
            telem = telem_rx.get()

            # ── failsafe check ──────────────────────────────────────
            if telem:
                curr_pos = (telem["posX"], telem["posY"], telem["posZ"])
                if failsafe_ref_pos is None:
                    failsafe_ref_pos = curr_pos
                    failsafe_ref_time = capture_time
                else:
                    dist = math.sqrt((curr_pos[0] - failsafe_ref_pos[0])**2 + 
                                     (curr_pos[1] - failsafe_ref_pos[1])**2 + 
                                     (curr_pos[2] - failsafe_ref_pos[2])**2)
                    if dist > 50.0:
                        # Vehicle moved out of the 50m radius, reset reference point
                        failsafe_ref_pos = curr_pos
                        failsafe_ref_time = capture_time
                    else:
                        # Vehicle is still within 50m radius
                        if capture_time - failsafe_ref_time > 120.0:
                            print(f"\n[failsafe] Vehicle stuck in 50m radius for 2 mins (dist={dist:.1f}m). Pressing HOME to reset!")
                            
                            # Send HOME key (VK_HOME = 0x24) to the background window
                            WM_KEYDOWN = 0x0100
                            WM_KEYUP = 0x0101
                            VK_HOME = 0x24
                            ctypes.windll.user32.PostMessageW(win.hwnd, WM_KEYDOWN, VK_HOME, 0)
                            time.sleep(0.1)
                            ctypes.windll.user32.PostMessageW(win.hwnd, WM_KEYUP, VK_HOME, 0)
                            
                            # Reset reference so we don't spam the key
                            failsafe_ref_pos = curr_pos
                            failsafe_ref_time = capture_time

            # 5. Frame filename = high-precision UNIX timestamp (microseconds)
            frame_name = f"{capture_time:.6f}.jpg"

            # 6. Compute final steering label with offset
            offset = shift_ctrl.offset
            final_steering = max(-1.0, min(1.0, steering + offset))

            # 7. Write to disk (steering = raw steering, steering_combined = raw + offset)
            writer.write(frame_name, raw, map_img, capture_time, telem, steering=steering, steering_offset=offset, steering_combined=final_steering)
            saved += 1
            if saved >= 50000:
                print(f"\n[collect] Limit of 50,000 images reached! Stopping collection...")
                break

            # 7. FPS counter
            fps_frames += 1
            now = time.perf_counter()
            dt  = now - t_fps_ref
            if dt >= 1.0:
                fps_display = fps_frames / dt
                fps_frames  = 0
                t_fps_ref   = now
                status = f"  FPS: {fps_display:.1f}   saved: {saved}   UDP pkts: {telem_rx.pkt_total} / valid: {telem_rx.pkt_valid}   Steer(Raw): {steering:+.2f}  Comb: {final_steering:+.2f}  Offset: {offset:+.3f} [{shift_ctrl.phase}]"
                if telem:
                    spd = math.sqrt(telem["velX"]**2 +
                                    telem["velY"]**2 +
                                    telem["velZ"]**2) * 3.6
                    status += f"   speed: {spd:.1f} km/h"
                print(status, end="\r")

            # 8. Optional live view
            if opt.view:
                vis = draw_hud(raw, telem, fps_display, saved,
                               telem_rx.pkt_total, telem_rx.pkt_valid, steering,
                               offset=offset, phase=shift_ctrl.phase, steering_combined=final_steering)
                dh, dw = vis.shape[:2]
                sc = min(960 / dw, 540 / dh)
                if sc < 1.0:
                    vis = cv2.resize(vis, (int(dw * sc), int(dh * sc)))
                cv2.imshow("BeamNG Collector", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 8. Pace to target FPS
            elapsed = time.perf_counter() - t0
            slack   = frame_interval - elapsed
            if slack > 0:
                time.sleep(slack)

    except KeyboardInterrupt:
        print("\nStopped by Ctrl-C.")
    finally:
        shift_ctrl.close()
        telem_rx.close()
        writer.close()
        cv2.destroyAllWindows()
        print(f"\nDone.  {saved} frames saved to {out_dir}/")


if __name__ == "__main__":
    main()
