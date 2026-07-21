"""
test_trans.py
=============
Real-time inference script for the Transformer model (ViT + TransformerEncoder).
Maintains a 4-frame temporal context buffer and sends steering & speed predictions 
to vJoy for controlling BeamNG.drive.
"""

import argparse
import ctypes
import json
import math
import os
import sys
import threading
import time
import socket
import struct
from ctypes import wintypes
from collections import deque

import cv2
import numpy as np
from PIL import Image
import mss
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ---------------------------------------------------------------------------
# vJoy constants
# ---------------------------------------------------------------------------
VJOY_STEER_MIN = 0x0
VJOY_STEER_MAX = 0x8000  # 32768
VJOY_STEER_MID = (VJOY_STEER_MAX + VJOY_STEER_MIN) // 2

VJOY_SPEED_MIN = 0
VJOY_SPEED_MAX = 32768
VJOY_SPEED_MID = 16384

HID_USAGE_X = 0x30  # Steering
HID_USAGE_Y = 0x31  # Throttle / Speed


# ---------------------------------------------------------------------------
# Configurable Transformer Model Definition
# ---------------------------------------------------------------------------
class SteeringModelTrans(nn.Module):
    def __init__(self, version="trans_vit"):
        super(SteeringModelTrans, self).__init__()
        self.version = version

        if version == "trans_vit":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            weights = ViT_B_16_Weights.DEFAULT
            self.spatial_backbone = vit_b_16(weights=weights)
            self.spatial_backbone.heads = nn.Identity()
            self.embed_dim = 768
            self.temp_heads = 8
            self.temp_ff = 1024
        elif version == "trans_vit_tiny_sim":
            from torchvision.models import VisionTransformer
            self.spatial_backbone = VisionTransformer(
                image_size=112,
                patch_size=16,
                num_layers=4,
                num_heads=3,
                hidden_dim=192,
                mlp_dim=768,
                num_classes=3
            )
            self.spatial_backbone.heads = nn.Identity()
            self.embed_dim = 192
            self.temp_heads = 4
            self.temp_ff = 512
        else:
            raise ValueError(f"Unknown model version: {version}")

        # Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=self.temp_heads, 
            dim_feedforward=self.temp_ff, 
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Learnable temporal positional embedding (up to 4 frames)
        self.temp_pos_embed = nn.Parameter(torch.zeros(1, 4, self.embed_dim))

        # -- Output head
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.embed_dim + 1, 256 if version == "trans_vit" else 128),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256 if version == "trans_vit" else 128, 3)
        )

    def forward(self, x, condition):
        B, seq_len, C, H, W = x.size()
        x_flat = x.view(B * seq_len, C, H, W)
        
        spatial_features = self.spatial_backbone(x_flat)
        spatial_features = spatial_features.view(B, seq_len, -1)
        
        spatial_features = spatial_features + self.temp_pos_embed[:, :seq_len, :]
        
        temporal_features = self.temporal_encoder(spatial_features)
        last_hidden = temporal_features[:, -1, :]
        
        last_hidden = torch.cat((last_hidden, condition), dim=1)
        return self.head(last_hidden)


# ---------------------------------------------------------------------------
# Window capture
# ---------------------------------------------------------------------------
def find_beamng_window():
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
                    found.append(hwnd)
                    return False
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    if found:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        return found[0]
    return None



import mss
from ctypes import wintypes

def get_window_rect(hwnd):
    user32  = ctypes.windll.user32
    cr = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(cr))
    w = cr.right  - cr.left
    h = cr.bottom - cr.top
    if w <= 0 or h <= 0:
        return None
    pt = ctypes.wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return {'left': pt.x, 'top': pt.y, 'width': w, 'height': h}

def capture_mss(sct, monitor):
    frame = sct.grab(monitor)
    img = np.frombuffer(frame.raw, dtype=np.uint8).reshape((frame.height, frame.width, 4))
    return img[:, :, :3].copy()


def capture_printwindow(hwnd):
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None

    hwndDC = user32.GetWindowDC(hwnd)
    mfcDC = gdi32.CreateCompatibleDC(hwndDC)
    saveBitmap = gdi32.CreateCompatibleBitmap(hwndDC, w, h)
    gdi32.SelectObject(mfcDC, saveBitmap)
    user32.PrintWindow(hwnd, mfcDC, 2)  # PW_RENDERFULLCONTENT

    bmi = bytearray(40)
    bmi[0:4] = (40).to_bytes(4, "little")
    bmi[4:8] = w.to_bytes(4, "little", signed=True)
    bmi[8:12] = (-h).to_bytes(4, "little", signed=True)
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
    return img[:, :, :3].copy()  # BGRA -> BGR


# ---------------------------------------------------------------------------
# OpenCV Accelerated Preprocessing
# ---------------------------------------------------------------------------
_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def preprocess_frame(frame_bgr: np.ndarray, target_size=(224, 224)) -> torch.Tensor:
    if frame_bgr.shape[1] != 1280 or frame_bgr.shape[0] != 720:
        frame_bgr = cv2.resize(frame_bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)
    
    # Fast NumPy slice crop: top=231, height=264 -> [231:495]
    crop_img = frame_bgr[231:495, 0:1280]
    
    # Fast OpenCV resize
    resized = cv2.resize(crop_img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # BGR to RGB
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Quick tensor conversion and division
    tensor = torch.from_numpy(resized_rgb.transpose(2, 0, 1)).float() / 255.0
    tensor = _NORMALIZE(tensor)
    return tensor


# ---------------------------------------------------------------------------
# UDP MotionSim telemetry receiver
# ---------------------------------------------------------------------------
BNG1_FORMAT = "4s" + "f" * 21
BNG1_SIZE   = struct.calcsize(BNG1_FORMAT)
BNG1_FIELDS = [
    "posX",  "posY",  "posZ",
    "velX",  "velY",  "velZ",
    "accX",  "accY",  "accZ",
    "upX",   "upY",   "upZ",
    "rollPos",  "pitchPos",  "yawPos",
    "rollVel",  "pitchVel",  "yawVel",
    "rollAcc",  "pitchAcc",  "yawAcc",
]

class TelemetryReceiver:
    def __init__(self, ip="0.0.0.0", port=4444):
        self._lock = threading.Lock()
        self._latest = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._sock.bind((ip, port))
        self._sock.settimeout(0.5)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            try:
                data, _ = self._sock.recvfrom(512)
                if len(data) >= BNG1_SIZE:
                    parts = struct.unpack_from(BNG1_FORMAT, data)
                    if parts[0][:4] == b"BNG1":
                        row = {name: parts[1 + i] for i, name in enumerate(BNG1_FIELDS)}
                        with self._lock:
                            self._latest = row
            except Exception:
                continue

    def get_speed_ms(self) -> float:
        with self._lock:
            if not self._latest:
                return 0.0
            return math.sqrt(self._latest["velX"]**2 + self._latest["velY"]**2 + self._latest["velZ"]**2)


# ---------------------------------------------------------------------------
# vJoy Cruise Control and steering controls
# ---------------------------------------------------------------------------
def _steer_to_vjoy(steering: float) -> int:
    clamped = max(-1.0, min(1.0, steering))
    return int(VJOY_STEER_MID + clamped * (VJOY_STEER_MAX - VJOY_STEER_MID))

def _speed_to_vjoy(scaled_speed: float) -> int:
    clamped = max(0.0, min(1.0, scaled_speed))
    return int(VJOY_SPEED_MIN + clamped * (VJOY_SPEED_MAX - VJOY_SPEED_MIN))


class VJoySender:
    SEND_HZ = 60
    def __init__(self, vjoy_device, telemetry: TelemetryReceiver, max_speed: float, disable_throttle: bool = False):
        self._vjoy = vjoy_device
        self._telemetry = telemetry
        self._max_speed = max_speed
        self._disable_throttle = disable_throttle
        self._lock = threading.Lock()
        
        self._steer = VJOY_STEER_MID
        self._target_speed_ms = 0.0
        
        self._integral = 0.0
        self._smoothed_output = 0.0
        
        self.current_speed_ms = 0.0
        self.pid_output = 0.0

        self._active = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set(self, steering_raw: float, scaled_speed: float):
        with self._lock:
            self._steer = _steer_to_vjoy(steering_raw)
            self._target_speed_ms = scaled_speed * self._max_speed

    def _loop(self):
        interval = 1.0 / self.SEND_HZ
        while self._active:
            t0 = time.perf_counter()
            
            cur_speed = self._telemetry.get_speed_ms()
            
            with self._lock:
                self.current_speed_ms = cur_speed
                target = self._target_speed_ms
                
                # --- Advanced Cruise Control ---
                error = target - cur_speed
                
                # 1. Base Throttle (Feedforward)
                base_throttle = 0.15 * (target / max(1.0, self._max_speed)) if target > 0.5 else 0.0
                
                # 2. Proportional & Integral
                if error > 0:
                    Kp = 0.10
                    p_term = Kp * error
                    self._integral += error * interval * 0.05
                    self._integral = min(0.60, max(0.0, self._integral))
                else:
                    Kp = 0.05 
                    p_term = Kp * error
                    base_throttle = 0.0
                    self._integral = 0.0
                
                # 3. Raw Output
                raw_output = base_throttle + p_term + self._integral
                
                # Deadzone for tiny errors
                if abs(error) < 0.2 and target > 0:
                    raw_output = base_throttle
                
                raw_output = max(-1.0, min(0.75, raw_output))
                
                # 4. Exponential Moving Average (EMA) Filter
                self._smoothed_output = (0.98 * self._smoothed_output) + (0.02 * raw_output)
                
                output = self._smoothed_output
                self.pid_output = output
                
                # Map PID output to Y axis
                if output >= 0:
                    y_val = VJOY_SPEED_MID + int(output * (VJOY_SPEED_MAX - VJOY_SPEED_MID))
                else:
                    y_val = VJOY_SPEED_MID - int(output * (VJOY_SPEED_MIN - VJOY_SPEED_MID))
                
                y_val = max(VJOY_SPEED_MIN, min(VJOY_SPEED_MAX, y_val))
                
                if not getattr(self, 'ai_enabled', True):
                    self._vjoy.data.wAxisX = VJOY_STEER_MID
                    self._vjoy.data.wAxisY = VJOY_SPEED_MID
                elif self._disable_throttle:
                    self._vjoy.data.wAxisX = self._steer
                else:
                    self._vjoy.data.wAxisX = self._steer
                    self._vjoy.data.wAxisY = y_val

            try:
                self._vjoy.update()
            except Exception:
                pass
            elapsed = time.perf_counter() - t0
            slack = interval - elapsed
            if slack > 0:
                time.sleep(slack)

    def stop(self, centre=True):
        self._active = False
        if centre:
            try:
                self._vjoy.data.wAxisX = VJOY_STEER_MID
                self._vjoy.data.wAxisY = VJOY_SPEED_MID
                self._vjoy.update()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# HUD overlay drawing
# ---------------------------------------------------------------------------
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_CLR_GREEN = (0, 255, 128)
_CLR_WHITE = (255, 255, 255)
_CLR_RED   = (0, 80, 255)
_CLR_BLACK = (0, 0, 0)

def _put(img, text, x, y, color=_CLR_GREEN, scale=0.6, thickness=2):
    cv2.putText(img, text, (x + 1, y + 1), _FONT, scale, _CLR_BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x,     y    ), _FONT, scale, color,       thickness,     cv2.LINE_AA)

def draw_hud(img: np.ndarray, steering: float, speed_kmh: float, offset: float, fps: float, vjoy_active: bool, 
             cur_speed_kmh: float = 0.0, pid_out: float = 0.0, condition: float = 0.0,
             model_see: np.ndarray = None, saliency: np.ndarray = None) -> np.ndarray:
    vis = img.copy()
    ph, pw = 270, 460
    panel = vis[:ph, :pw].copy()
    cv2.rectangle(panel, (0, 0), (pw, ph), (8, 8, 8), -1)
    cv2.addWeighted(panel, 0.60, vis[:ph, :pw], 0.40, 0, vis[:ph, :pw])

    y = 28
    vjoy_txt = "vJoy: ACTIVE" if vjoy_active else "vJoy: DRY RUN (--no-vjoy)"
    vjoy_clr = _CLR_GREEN if vjoy_active else _CLR_RED
    _put(vis, vjoy_txt, 10, y, vjoy_clr)

    y += 26
    _put(vis, f"FPS: {fps:5.1f}", 10, y, _CLR_WHITE)

    # Condition Display
    y += 26
    cond_txt = "STRAIGHT"
    cond_clr = _CLR_WHITE
    if condition == -1.0:
        cond_txt = "TURN LEFT"
        cond_clr = (0, 180, 255) # Orange-ish
    elif condition == 1.0:
        cond_txt = "TURN RIGHT"
        cond_clr = (255, 180, 0) # Cyan-ish
    _put(vis, f"Condition: {cond_txt} ({condition:+.0f})", 10, y, cond_clr)

    # Steering
    y += 26
    _put(vis, f"Steering:  {steering:+.3f}", 10, y)
    bar_w = 200
    bar_x = 200
    cv2.rectangle(vis, (bar_x, y - 14), (bar_x + bar_w, y), (50, 50, 50), -1)
    mid_x = bar_x + bar_w // 2
    if steering > 0:
        cv2.rectangle(vis, (mid_x, y - 14), (mid_x + int(steering * bar_w / 2), y), (0, 140, 255), -1)
    elif steering < 0:
        cv2.rectangle(vis, (mid_x + int(steering * bar_w / 2), y - 14), (mid_x, y), (0, 140, 255), -1)
    cv2.line(vis, (mid_x, y - 16), (mid_x, y + 2), _CLR_WHITE, 1)

    # Offset
    y += 26
    _put(vis, f"Offset:    {offset:+.3f}", 10, y)
    cv2.rectangle(vis, (bar_x, y - 14), (bar_x + bar_w, y), (50, 50, 50), -1)
    mid_x = bar_x + bar_w // 2
    display_offset = max(-1.0, min(1.0, offset * 4.0))
    if display_offset > 0:
        cv2.rectangle(vis, (mid_x, y - 14), (mid_x + int(display_offset * bar_w / 2), y), (255, 100, 100), -1)
    elif display_offset < 0:
        cv2.rectangle(vis, (mid_x + int(display_offset * bar_w / 2), y - 14), (mid_x, y), (255, 100, 100), -1)
    cv2.line(vis, (mid_x, y - 16), (mid_x, y + 2), _CLR_WHITE, 1)

    # Speed
    y += 28
    _put(vis, f"Target Spd:  {speed_kmh:5.1f} km/h", 10, y, (0, 255, 255))
    
    y += 26
    _put(vis, f"Actual Spd:  {cur_speed_kmh:5.1f} km/h", 10, y, _CLR_GREEN)
    
    y += 26
    _put(vis, f"PID T/B:   {pid_out:+.2f}", 10, y, _CLR_WHITE)
    cv2.rectangle(vis, (bar_x, y - 14), (bar_x + bar_w, y), (50, 50, 50), -1)
    mid_x = bar_x + bar_w // 2
    if pid_out > 0:
        cv2.rectangle(vis, (mid_x, y - 14), (mid_x + int(pid_out * bar_w / 2), y), (0, 255, 0), -1)
    elif pid_out < 0:
        cv2.rectangle(vis, (mid_x + int(pid_out * bar_w / 2), y - 14), (mid_x, y), (0, 0, 255), -1)
    cv2.line(vis, (mid_x, y - 16), (mid_x, y + 2), _CLR_WHITE, 1)

    y += 26
    _put(vis, "Q: Quit | 1: Left | 2: Straight | 3: Right", 10, y, (120, 120, 120), scale=0.45, thickness=1)

    # -- Overlays: what the model sees & steering attention map
    vh, vw = vis.shape[:2]
    if model_see is not None and saliency is not None:
        dsz = 180
        ms_disp = cv2.resize(model_see, (dsz, dsz))
        sa_disp = cv2.resize(saliency, (dsz, dsz))
        
        y_off = 10
        x_ms = vw - 2 * dsz - 20
        x_sa = vw - dsz - 10
        
        # Draw dark panel backgrounds
        cv2.rectangle(vis, (x_ms - 5, y_off - 5), (x_ms + dsz + 5, y_off + dsz + 25), (15, 15, 15), -1)
        cv2.rectangle(vis, (x_sa - 5, y_off - 5), (x_sa + dsz + 5, y_off + dsz + 25), (15, 15, 15), -1)
        
        # Blend overlay frames
        vis[y_off:y_off+dsz, x_ms:x_ms+dsz] = ms_disp
        vis[y_off:y_off+dsz, x_sa:x_sa+dsz] = sa_disp
        
        # Add overlay text labels
        _put(vis, "MODEL INPUT", x_ms + 25, y_off + dsz + 18, _CLR_WHITE, scale=0.45, thickness=1)
        _put(vis, "STEER ATTN", x_sa + 30, y_off + dsz + 18, _CLR_GREEN, scale=0.45, thickness=1)

    return vis


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------
def make_parser():
    p = argparse.ArgumentParser(description="BeamNG Transformer real-time inference script")
    p.add_argument("--model",     default="best_steering_trans_tiny_sim_model.pth",
                   help="path to trained .pth file")
    p.add_argument("--max-speed", type=float, default=None,
                   help="speed scaling factor")
    p.add_argument("--headless",  action="store_true",
                   help="run without displaying the OpenCV window")
    p.add_argument("--no-vjoy",   action="store_true",
                   help="dry-run mode")
    p.add_argument("--width",     type=int, default=960,
                   help="display window width (default: 960)")
    p.add_argument("--height",    type=int, default=540,
                   help="display window height (default: 540)")
    p.add_argument("--steer-gain", type=float, default=1.0,
                   help="multiplier applied to the predicted steering")
    p.add_argument("--disable-throttle", action="store_true",
                   help="disable Cruise Control output so you can manually control pedals")
    return p


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    opt = make_parser().parse_args()

    try:
        if ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080):
            print("[init] Process priority set to HIGH")
    except Exception as e:
        print(f"[warn] Could not set process priority: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[init] Using device: {device}")

    # Load version and max_speed dynamically from metadata
    version = "trans_vit" # Default
    max_speed = opt.max_speed
    meta_path = os.path.splitext(opt.model)[0] + '_meta.json'
    
    if os.path.exists(meta_path):
        with open(meta_path) as mf:
            meta = json.load(mf)
        max_speed = meta.get('max_speed', max_speed)
        version = meta.get('version', version)
        print(f"[init] Loaded version={version}, max_speed={max_speed:.4f} m/s from {meta_path}")
    elif max_speed is None:
        max_speed = 27.0
        print(f"[warn] No meta JSON found and --max-speed not set. Defaulting to {max_speed}.")
    else:
        print(f"[init] Using --max-speed={max_speed:.4f} m/s (no meta JSON found)")

    target_size = (112, 112) if version == "trans_vit_tiny_sim" else (224, 224)

    print(f"[init] Loading Transformer model from: {opt.model}")
    model = SteeringModelTrans(version=version).to(device)
    model.load_state_dict(torch.load(opt.model, map_location=device))
    model.eval()
    print("[init] Model loaded OK")

    print("[init] Searching for BeamNG.drive window ...")
    hwnd = find_beamng_window()
    if hwnd is None:
        print("ERROR: BeamNG.drive window not found. Is the game running?")
        sys.exit(1)
    print(f"[init] Found window  HWND={hwnd}")

    telemetry = TelemetryReceiver()
    sct = mss.mss()


    vjoy = None
    if not opt.no_vjoy:
        try:
            import pyvjoy
            vjoy = pyvjoy.VJoyDevice(1)
            vjoy.data.wAxisX  = VJOY_STEER_MID
            vjoy.data.wAxisY  = VJOY_SPEED_MID
            vjoy.update()
            print("[init] vJoy device 1 acquired and centred")
        except Exception as e:
            print(f"[warn] Could not open vJoy device 1: {e}")
            vjoy = None

    vjoy_active = vjoy is not None
    vjoy_sender = None
    if vjoy_active:
        vjoy_sender = VJoySender(vjoy, telemetry, max_speed, opt.disable_throttle)
        print(f"[init] vJoy sender thread started at {VJoySender.SEND_HZ} Hz")

    if not opt.headless:
        cv2.namedWindow("BeamNG Transformer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("BeamNG Transformer", opt.width, opt.height)

    # 4-frame ring buffer for temporal sequence context
    frame_buffer = deque(maxlen=4)

    fps_display = 0.0
    fps_frames  = 0
    t_fps_ref   = time.perf_counter()

    current_condition = 0.0

    if opt.headless:
        print("\n[inference] Running in HEADLESS mode -- press Ctrl-C to stop.\n")
    else:
        print("\n[inference] Running -- press Q or ESC to stop. Use keys 1, 2, 3 to steer routing.\n")

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Capture BeamNG
            monitor = get_window_rect(hwnd)
            if monitor is None:
                time.sleep(0.01)
                continue
            raw = capture_mss(sct, monitor)
            if raw is None:
                time.sleep(0.01)
                continue

            # 2. Preprocess with accelerated NumPy/OpenCV
            curr_tensor = preprocess_frame(raw, target_size=target_size).to(device)
            frame_buffer.append(curr_tensor)

            # Pad sequence buffer if startup context is incomplete
            frames = list(frame_buffer)
            while len(frames) < 4:
                frames.insert(0, frames[0])

            # Build (1, 4, 3, 224, 224) input tensor
            tensor = torch.stack(frames, dim=0).unsqueeze(0)
            
            # Build condition input (1, 1) tensor
            condition_tensor = torch.tensor([[current_condition]], dtype=torch.float32).to(device)

            # 3. Model Inference (utilizing AMP autocast, with grad enabled for saliency mapping)
            with torch.enable_grad():
                tensor.requires_grad_()
                if device.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        out = model(tensor, condition_tensor)
                else:
                    out = model(tensor, condition_tensor)
            
                pred_steering      = out[0, 0].item() * opt.steer_gain
                pred_scaled_speed  = out[0, 1].item()
                pred_offset        = out[0, 2].item()
                
                # Compute steering saliency map (gradients of steering output w.r.t input)
                model.zero_grad()
                out[0, 0].backward()
            
            # Saliency map processing
            grad = tensor.grad[0, -1] # (3, H, W)
            saliency, _ = torch.max(torch.abs(grad), dim=0)
            saliency = saliency.cpu().numpy()
            
            sali_max = saliency.max()
            if sali_max > 1e-8:
                saliency = (saliency / sali_max * 255).astype(np.uint8)
            else:
                saliency = np.zeros_like(saliency, dtype=np.uint8)
                
            saliency_heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
            
            # Clean gradient graphs to prevent memory leak
            model.zero_grad()
            tensor.grad = None

            # Un-normalize current frame to visualize model input
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
            unnorm = curr_tensor * std + mean
            unnorm = torch.clamp(unnorm * 255.0, 0.0, 255.0).byte()
            model_see_np = unnorm.cpu().numpy().transpose(1, 2, 0)
            model_see_bgr = cv2.cvtColor(model_see_np, cv2.COLOR_RGB2BGR)

            pred_speed_kmh    = pred_scaled_speed * max_speed * 3.6

            # Clip target speed to safe driving range (0 to 30 km/h)
            pred_speed_kmh    = max(0.0, min(30.0, pred_speed_kmh))
            pred_scaled_speed = pred_speed_kmh / (max_speed * 3.6)

            pred_steering     = max(-1.0, min(1.0, pred_steering))

            # 4. Update vJoy controls
            if vjoy_sender is not None:
                vjoy_sender.set(pred_steering, pred_scaled_speed)

            # 5. Display / FPS
            fps_frames += 1
            now = time.perf_counter()
            if now - t_fps_ref >= 1.0:
                fps_display = fps_frames / (now - t_fps_ref)
                fps_frames = 0
                t_fps_ref = now

            if not opt.headless:
                cur_speed_kmh = telemetry.get_speed_ms() * 3.6
                pid_out = vjoy_sender.pid_output if vjoy_sender else 0.0
                
                # Debug output to terminal
                if pid_out > 0:
                    print(f"\r[PID] Throttle: {pid_out * 100:5.1f}% | Brake:   0.0%  (Target: {pred_speed_kmh:4.1f} km/h, Cur: {cur_speed_kmh:4.1f} km/h)      ", end="", flush=True)
                elif pid_out < 0:
                    print(f"\r[PID] Throttle:   0.0% | Brake: {-pid_out * 100:5.1f}%  (Target: {pred_speed_kmh:4.1f} km/h, Cur: {cur_speed_kmh:4.1f} km/h)      ", end="", flush=True)
                else:
                    print(f"\r[PID] Throttle:   0.0% | Brake:   0.0%  (Target: {pred_speed_kmh:4.1f} km/h, Cur: {cur_speed_kmh:4.1f} km/h)      ", end="", flush=True)

                vis = draw_hud(raw, pred_steering, pred_speed_kmh, pred_offset, fps_display, vjoy_active, 
                               cur_speed_kmh, pid_out, current_condition,
                               model_see=model_see_bgr, saliency=saliency_heatmap)
                dh, dw = vis.shape[:2]
                scale = min(opt.width / dw, opt.height / dh)
                if scale != 1.0:
                    vis = cv2.resize(vis, (int(dw * scale), int(dh * scale)), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("BeamNG Transformer", vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("0"):
                    if vjoy_sender:
                        vjoy_sender.ai_enabled = not getattr(vjoy_sender, "ai_enabled", True)
                        state = "ON" if vjoy_sender.ai_enabled else "OFF"
                        print(f"\n[input] AI Control toggled to: {state}")
                elif key == ord("1"):
                    current_condition = -1.0
                    print(f"\n[input] Routing condition changed to LEFT (-1)")
                elif key == ord("2"):
                    current_condition = 0.0
                    print(f"\n[input] Routing condition changed to STRAIGHT (0)")
                elif key == ord("3"):
                    current_condition = 1.0
                    print(f"\n[input] Routing condition changed to RIGHT (1)")
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopped by Ctrl-C.")
    finally:
        if vjoy_sender is not None:
            vjoy_sender.stop(centre=True)
            print("[exit] vJoy axes reset to safe position.")
        cv2.destroyAllWindows()
        print("[exit] Done.")


if __name__ == "__main__":
    main()
