"""
test_cnn_v3.py
==============
Real-time inference script for the V3 model (Dual-Stream Two-Tower, 3 outputs).
Maintains a 1-frame ring buffer to provide temporal context to the model.
"""

import argparse
import ctypes
import json
import math
import os
import sys
import threading
import time
from ctypes import wintypes
from collections import deque

import cv2
import numpy as np
import socket
import struct
import math
from PIL import Image
import mss
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False

class RegressionTarget:
    def __init__(self, target_idx=0):
        self.target_idx = target_idx
    def __call__(self, model_output):
        return model_output[self.target_idx]

class ActivationTracker:
    def __init__(self, model):
        self.activations = {}
        self.hooks = []
        self.stages = {
            "Stage 1 (Early Edges)": model.backbone[0][1],
            "Stage 3 (Textures)": model.backbone[0][3],
            "Stage 5 (Mid-Late Shapes)": model.backbone[0][5],
            "Stage 7 (High Semantics)": model.backbone[0][7]
        }
        for name, layer in self.stages.items():
            self.hooks.append(layer.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()

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
# Steering model definition V2 (must match train_steering_v3.py exactly)
# ---------------------------------------------------------------------------
class SteeringModelV3(nn.Module):
    FEAT_DIM = 1280
    def __init__(self):
        super(SteeringModelV3, self).__init__()
        base = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            base.features,
            base.avgpool,
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(self.FEAT_DIM * 2, 512),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 3)
        )

    def forward(self, prev, curr):
        feat_curr = self.backbone(curr)
        feat_prev = self.backbone(prev)
        delta = feat_curr - feat_prev
        fused = torch.cat([feat_curr, delta], dim=1)
        return self.head(fused)


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
# Preprocessing
# ---------------------------------------------------------------------------
_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    if frame_bgr.shape[1] != 1280 or frame_bgr.shape[0] != 720:
        frame_bgr = cv2.resize(frame_bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img = TF.crop(img, top=231, left=0, height=264, width=1280)
    img = TF.resize(img, (240, 240))
    tensor = TF.to_tensor(img)
    tensor = _NORMALIZE(tensor)
    return tensor



# ---------------------------------------------------------------------------
# UDP MotionSim telemetry receiver (BNG1)
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
# vJoy helpers
# ---------------------------------------------------------------------------
def _steer_to_vjoy(steering: float) -> int:
    clamped = max(-1.0, min(1.0, steering))
    return int(VJOY_STEER_MID + clamped * (VJOY_STEER_MAX - VJOY_STEER_MID))

def _speed_to_vjoy(scaled_speed: float) -> int:
    clamped = max(0.0, min(1.0, scaled_speed))
    return int(VJOY_AXIS_MIN + clamped * (VJOY_AXIS_MAX - VJOY_AXIS_MIN))


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
        
        # Advanced Cruise Control variables
        self._integral = 0.0
        self._smoothed_output = 0.0
        
        # Public stats for HUD
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
            
            # Read real speed from UDP
            cur_speed = self._telemetry.get_speed_ms()
            
            with self._lock:
                self.current_speed_ms = cur_speed
                target = self._target_speed_ms
                
                # --- Advanced Cruise Control ---
                error = target - cur_speed
                
                # 1. Base Throttle (Feedforward)
                # To maintain speed against friction/drag, we need some throttle.
                base_throttle = 0.15 * (target / max(1.0, self._max_speed)) if target > 0.5 else 0.0
                
                # 2. Proportional & Integral
                if error > 0:
                    # Accelerating: Respond more aggressively to large speed gaps
                    # Kp = 0.1 means if we are 5 m/s too slow, add 50% throttle.
                    Kp = 0.10
                    p_term = Kp * error
                    # Integral helps climb hills over time
                    self._integral += error * interval * 0.05
                    self._integral = min(0.60, max(0.0, self._integral)) # Allow up to 60% extra throttle memory for hills!
                else:
                    # Braking: 
                    # Kp = 0.05 means if we are 5 m/s too fast, apply 25% brakes.
                    Kp = 0.05 
                    p_term = Kp * error
                    # Cut base throttle and reset integral immediately when braking
                    base_throttle = 0.0
                    self._integral = 0.0
                
                # 3. Raw Output
                raw_output = base_throttle + p_term + self._integral
                
                # Deadzone for tiny errors (prevents brake fluttering when coasting exactly at target speed)
                if abs(error) < 0.2 and target > 0:
                    raw_output = base_throttle
                
                raw_output = max(-1.0, min(0.75, raw_output)) # Cap throttle at 75%
                
                # 4. Exponential Moving Average (EMA) Filter for buttery smooth pedals
                # 0.02 weight on new value = smooth transition over ~50 frames (almost 1 full second!)
                self._smoothed_output = (0.98 * self._smoothed_output) + (0.02 * raw_output)
                
                # Final output to vJoy
                output = self._smoothed_output
                self.pid_output = output
                
                # Map PID output to Y axis
                # VJOY_SPEED_MID is 0 (coast). 
                # Throttle (output > 0) -> Sweeps from MID (0) to MAX (32767)
                # Brake (output < 0) -> Sweeps from MID (0) to MIN (-32768)
                if output >= 0:
                    y_val = VJOY_SPEED_MID + int(output * (VJOY_SPEED_MAX - VJOY_SPEED_MID))
                else:
                    y_val = VJOY_SPEED_MID - int(output * (VJOY_SPEED_MIN - VJOY_SPEED_MID))
                
                y_val = max(VJOY_SPEED_MIN, min(VJOY_SPEED_MAX, y_val))
                
                self._vjoy.data.wAxisX = self._steer
                if not self._disable_throttle:
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
# HUD overlay
# ---------------------------------------------------------------------------
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_CLR_GREEN = (0, 255, 128)
_CLR_WHITE = (255, 255, 255)
_CLR_RED   = (0, 80, 255)
_CLR_BLACK = (0, 0, 0)

def _put(img, text, x, y, color=_CLR_GREEN, scale=0.6, thickness=2):
    cv2.putText(img, text, (x + 1, y + 1), _FONT, scale, _CLR_BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x,     y    ), _FONT, scale, color,       thickness,     cv2.LINE_AA)

def draw_hud(img: np.ndarray, steering: float, speed_kmh: float, offset: float, fps: float, vjoy_active: bool, cur_speed_kmh: float = 0.0, pid_out: float = 0.0) -> np.ndarray:
    vis = img.copy()
    ph, pw = 230, 460
    panel = vis[:ph, :pw].copy()
    cv2.rectangle(panel, (0, 0), (pw, ph), (8, 8, 8), -1)
    cv2.addWeighted(panel, 0.60, vis[:ph, :pw], 0.40, 0, vis[:ph, :pw])

    y = 28
    vjoy_txt = "vJoy: ACTIVE" if vjoy_active else "vJoy: DRY RUN (--no-vjoy)"
    vjoy_clr = _CLR_GREEN if vjoy_active else _CLR_RED
    _put(vis, vjoy_txt, 10, y, vjoy_clr)

    y += 26
    _put(vis, f"FPS: {fps:5.1f}", 10, y, _CLR_WHITE)

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
    _put(vis, "Q / ESC to quit", 10, y, (100, 100, 100), scale=0.50, thickness=1)

    return vis


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def make_parser():
    p = argparse.ArgumentParser(description="BeamNG CNN real-time inference (V2)")
    p.add_argument("--model",     default="best_steering_v3_model.pth",
                   help="path to trained .pth file")
    p.add_argument("--max-speed", type=float, default=None,
                   help="speed scaling factor")
    p.add_argument("--cam",       action="store_true",
                   help="enable Grad-CAM visualization for debugging")
    p.add_argument("--activations", action="store_true",
                   help="enable activation map visualization")
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

    print(f"[init] Loading V2 model from: {opt.model}")
    model = SteeringModelV3().to(device)
    model.load_state_dict(torch.load(opt.model, map_location=device))
    model.eval()
    print("[init] Model loaded OK")

    max_speed = opt.max_speed
    meta_path = os.path.splitext(opt.model)[0] + '_meta.json'
    if os.path.exists(meta_path):
        with open(meta_path) as mf:
            meta = json.load(mf)
        max_speed = meta.get('max_speed', max_speed)
        print(f"[init] Loaded max_speed={max_speed:.4f} m/s from {meta_path}")
    elif max_speed is None:
        max_speed = 28.61
        print(f"[warn] No meta JSON found and --max-speed not set. Defaulting to {max_speed}.")
    else:
        print(f"[init] Using --max-speed={max_speed:.4f} m/s (no meta JSON found)")

    print("[init] Searching for BeamNG.drive window ...")
    hwnd = find_beamng_window()
    if hwnd is None:
        print("ERROR: BeamNG.drive window not found. Is the game running?")
        sys.exit(1)
    print(f"[init] Found window  HWND={hwnd}")

    cam = None
    if opt.cam:
        print("[warn] Grad-CAM is temporarily disabled for V3 dual-stream.")
        cam = None

    act_tracker = None
    if opt.activations:
        act_tracker = ActivationTracker(model)
        print("[init] Activation map visualization enabled.")

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
        cv2.namedWindow("BeamNG CNN", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("BeamNG CNN", opt.width, opt.height)

    # 1-frame buffer for temporal context
    frame_buffer = deque(maxlen=1)

    fps_display = 0.0
    fps_frames  = 0
    t_fps_ref   = time.perf_counter()

    if opt.headless:
        print("\n[inference] Running in HEADLESS mode -- press Ctrl-C to stop.\n")
    else:
        print("\n[inference] Running -- press Q or ESC in the HUD window to stop.\n")

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Capture
            monitor = get_window_rect(hwnd)
            if monitor is None:
                time.sleep(0.01)
                continue
            raw = capture_mss(sct, monitor)
            if raw is None:
                time.sleep(0.01)
                continue

            # 2. Preprocess
            curr_tensor = preprocess_frame(raw).to(device)

            # Retrieve prev frame (duplicate if buffer empty)
            prev_tensor = frame_buffer[0] if frame_buffer else curr_tensor
            frame_buffer.append(curr_tensor)

            curr_batch = curr_tensor.unsqueeze(0)
            prev_batch = prev_tensor.unsqueeze(0)

            # 3. Inference
            if act_tracker is not None:
                out = model(prev_batch, curr_batch)
                stages_to_show = ["Stage 1 (Early Edges)", "Stage 3 (Textures)", "Stage 5 (Mid-Late Shapes)", "Stage 7 (High Semantics)"]
                activation_maps = []
                for stage_name in stages_to_show:
                    feat = act_tracker.activations.get(stage_name)
                    if feat is not None:
                        act = torch.norm(feat[0], p=2, dim=0).cpu().numpy()
                        act_min, act_max = act.min(), act.max()
                        if act_max > act_min:
                            act = (act - act_min) / (act_max - act_min)
                        else:
                            act = np.zeros_like(act)
                        act = (act * 255).astype(np.uint8)
                        color_act = cv2.applyColorMap(act, cv2.COLORMAP_JET)
                        color_act = cv2.resize(color_act, (240, 240))
                        cv2.putText(color_act, stage_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                        activation_maps.append(color_act)
                
                if activation_maps:
                    combined_strip = np.hstack(activation_maps)
                    cv2.imshow("Model Activations", combined_strip)
            else:
                with torch.inference_mode():
                    if device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            out = model(prev_batch, curr_batch)
                    else:
                        out = model(prev_batch, curr_batch)

            pred_steering     = out[0, 0].item() * opt.steer_gain
            pred_scaled_speed = (out[0, 1].item()) *1.3
            pred_offset       = out[0, 2].item()
            
            pred_speed_kmh    = pred_scaled_speed * max_speed * 3.6

            # Hard limit speed from 0 to 30 km/h
            pred_speed_kmh    = max(0.0, min(30.0, pred_speed_kmh))
            pred_scaled_speed = pred_speed_kmh / (max_speed * 3.6)

            pred_steering     = max(-1.0, min(1.0, pred_steering))

            # 4. vJoy
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
                
                # Debug print to terminal
                if pid_out > 0:
                    print(f"\r[PID] Throttle: {pid_out * 100:5.1f}% | Brake:   0.0%  (Target: {pred_speed_kmh:4.1f} km/h, Cur: {cur_speed_kmh:4.1f} km/h)      ", end="", flush=True)
                elif pid_out < 0:
                    print(f"\r[PID] Throttle:   0.0% | Brake: {-pid_out * 100:5.1f}%  (Target: {pred_speed_kmh:4.1f} km/h, Cur: {cur_speed_kmh:4.1f} km/h)      ", end="", flush=True)
                else:
                    print(f"\r[PID] Throttle:   0.0% | Brake:   0.0%  (Target: {pred_speed_kmh:4.1f} km/h, Cur: {cur_speed_kmh:4.1f} km/h)      ", end="", flush=True)

                vis = draw_hud(raw, pred_steering, pred_speed_kmh, pred_offset, fps_display, vjoy_active, cur_speed_kmh, pid_out)
                dh, dw = vis.shape[:2]
                scale = min(opt.width / dw, opt.height / dh)
                if scale != 1.0:
                    vis = cv2.resize(vis, (int(dw * scale), int(dh * scale)), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("BeamNG CNN", vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("0"):
                    if vjoy_sender:
                        vjoy_sender.ai_enabled = not getattr(vjoy_sender, "ai_enabled", True)
                        state = "ON" if vjoy_sender.ai_enabled else "OFF"
                        print(f"\n[input] AI Control toggled to: {state}")
            else:
                if cam is not None or act_tracker is not None:
                    cv2.waitKey(1)
                else:
                    time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopped by Ctrl-C.")
    finally:
        if act_tracker is not None:
            act_tracker.remove()
        if vjoy_sender is not None:
            vjoy_sender.stop(centre=True)
            print("[exit] vJoy axes reset to safe position.")
        cv2.destroyAllWindows()
        print("[exit] Done.")


if __name__ == "__main__":
    main()

