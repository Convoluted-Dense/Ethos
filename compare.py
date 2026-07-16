import time
import sys
import ctypes
from ctypes import wintypes
import socket
import struct
import threading
import json
import math
from collections import deque
import os

import cv2
import numpy as np
import mss
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision import transforms

# ---------------------------------------------------------------------------
# Telemetry (UDP OutGauge)
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
# Window Capture
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
        try: ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception: 
            try: ctypes.windll.user32.SetProcessDPIAware()
            except Exception: pass
        return found[0]
    return None

def get_window_rect(hwnd):
    rect = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
    pt = wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return {
        "top": pt.y,
        "left": pt.x,
        "width": rect.right - rect.left,
        "height": rect.bottom - rect.top
    }


# ---------------------------------------------------------------------------
# Ground Truth Steering Extraction
# ---------------------------------------------------------------------------
def load_steering_roi():
    try:
        with open("steering_roi.json", "r") as f:
            data = json.load(f)
            return data.get("roi"), data.get("map_roi")
    except Exception:
        return None, None

def extract_steering(img, roi):
    if not roi:
        return 0.0
    x, y, w, h = roi
    if y + h > img.shape[0] or x + w > img.shape[1]:
        return 0.0
    bar_img = img[y:y+h, x:x+w]
    hsv = cv2.cvtColor(bar_img, cv2.COLOR_BGR2HSV if bar_img.shape[2] == 3 else cv2.COLOR_BGRA2HSV)
    mask = cv2.inRange(hsv, np.array([10, 150, 150]), np.array([25, 255, 255]))
    cols = np.any(mask, axis=0)
    indices = np.where(cols)[0]
    if len(indices) == 0:
        return 0.0
    bar_center = w / 2.0
    left_val = max(0, bar_center - indices[0]) / bar_center
    right_val = max(0, indices[-1] - bar_center) / bar_center
    steering = right_val - left_val
    return max(-1.0, min(1.0, steering))


# ---------------------------------------------------------------------------
# Model 1: V3 Dual-Stream
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
# Model 2: V1 Single-Stream
# ---------------------------------------------------------------------------
class SteeringModel(nn.Module):
    def __init__(self):
        super(SteeringModel, self).__init__()
        weights = EfficientNet_B1_Weights.DEFAULT
        self.model = efficientnet_b1(weights=weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 2)
        )
        
    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _put(img, text, x, y, color=(255, 255, 255), size=0.5):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

def draw_hud(img, actual_steer, actual_speed, m1_steer, m1_speed, m2_steer, m2_speed, fps):
    vis = img.copy()
    h, w = vis.shape[:2]
    
    # Header
    _put(vis, f"COMPARE SCRIPT - FPS: {fps:.1f}", 10, 30, (0, 255, 255), 0.7)
    
    # Steering Bars
    mid_x = w // 2
    bar_w = 400
    y_start = 80
    
    colors = [
        ("ACTUAL", actual_steer, actual_speed, (0, 255, 0)),
        ("Model 1 (V3)", m1_steer, m1_speed, (0, 165, 255)),
        ("Model 2 (V1)", m2_steer, m2_speed, (255, 0, 255))
    ]
    
    for i, (name, steer, speed, color) in enumerate(colors):
        y = y_start + i * 50
        
        # Text
        _put(vis, f"{name}: Steer {steer:+.2f} | Speed {speed:5.1f} km/h", 10, y-15, color, 0.5)
        
        # Steering Bar
        cv2.rectangle(vis, (mid_x - bar_w//2, y - 5), (mid_x + bar_w//2, y + 5), (50, 50, 50), -1)
        cv2.line(vis, (mid_x, y - 8), (mid_x, y + 8), (255,255,255), 2)
        if steer > 0:
            cv2.rectangle(vis, (mid_x, y-5), (mid_x + int(steer * bar_w/2), y+5), color, -1)
        elif steer < 0:
            cv2.rectangle(vis, (mid_x + int(steer * bar_w/2), y-5), (mid_x, y+5), color, -1)
            
    return vis

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080):
        print("[init] Process priority set to HIGH.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] Using device: {device}")
    
    # Load Models
    print("[init] Loading Model 1 (V3)...")
    m1 = SteeringModelV3().to(device)
    m1.load_state_dict(torch.load("best_steering_v3_model.pth", map_location=device, weights_only=True))
    m1.eval()
    
    try:
        with open("best_steering_v3_model_meta.json", "r") as f:
            m1_max_speed = json.load(f)["max_speed"]
    except:
        m1_max_speed = 22.6830
    
    print("[init] Loading Model 2 (V1)...")
    m2 = SteeringModel().to(device)
    m2.load_state_dict(torch.load("best_steering_velocity_model.pth", map_location=device, weights_only=True))
    m2.eval()
    
    try:
        with open("best_steering_velocity_model_meta.json", "r") as f:
            m2_max_speed = json.load(f)["max_speed"]
    except:
        m2_max_speed = 28.61 # Fallback
        
    print(f"[init] Models Loaded. M1 Max Speed: {m1_max_speed:.2f} m/s | M2 Max Speed: {m2_max_speed:.2f} m/s")

    # Image Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Game Capture
    hwnd = find_beamng_window()
    if not hwnd:
        print("ERROR: BeamNG not found.")
        sys.exit(1)
        
    roi, _ = load_steering_roi()
    telemetry = TelemetryReceiver()
    sct = mss.MSS()
    
    # Tracking
    m1_steer_err_sum = 0.0
    m1_speed_err_sum = 0.0
    m2_steer_err_sum = 0.0
    m2_speed_err_sum = 0.0
    frames_counted = 0
    
    frame_buffer = deque(maxlen=1)
    
    cv2.namedWindow("Model Comparison", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Model Comparison", 1280, 720)
    
    t_fps_ref = time.perf_counter()
    fps_frames = 0
    fps_display = 0.0
    
    print("\n--- STARTING COMPARISON (Press Ctrl+C to stop and print results) ---")
    
    try:
        while True:
            rect = get_window_rect(hwnd)
            img = np.array(sct.grab(rect))
            
            raw = img[:, :, :3].copy()
            
            # Ground Truth
            actual_steer = extract_steering(raw, roi)
            actual_speed_kmh = telemetry.get_speed_ms() * 3.6
            
            # Preprocess
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            tensor_img = transform(rgb).unsqueeze(0).to(device)
            
            # Model 2 Inference (Single Frame)
            with torch.inference_mode():
                out2 = m2(tensor_img)
            
            m2_pred_steer = out2[0, 0].item()
            m2_pred_speed_kmh = out2[0, 1].item() * m2_max_speed * 3.6
            m2_pred_steer = max(-1.0, min(1.0, m2_pred_steer))
            m2_pred_speed_kmh = max(0.0, m2_pred_speed_kmh)
            
            # Model 1 Inference (Dual Frame)
            m1_pred_steer = 0.0
            m1_pred_speed_kmh = 0.0
            if len(frame_buffer) > 0:
                prev_tensor = frame_buffer[0]
                with torch.inference_mode():
                    out1 = m1(prev_tensor, tensor_img)
                m1_pred_steer = out1[0, 0].item()
                # Applying the * 1.3 that the user added to their test script manually
                m1_pred_speed_kmh = (out1[0, 1].item() * 1.3) * m1_max_speed * 3.6
                
                m1_pred_steer = max(-1.0, min(1.0, m1_pred_steer))
                m1_pred_speed_kmh = max(0.0, m1_pred_speed_kmh)
                
                # Accumulate Errors (only when both models ran)
                m1_steer_err_sum += abs(m1_pred_steer - actual_steer)
                m1_speed_err_sum += abs(m1_pred_speed_kmh - actual_speed_kmh)
                
                m2_steer_err_sum += abs(m2_pred_steer - actual_steer)
                m2_speed_err_sum += abs(m2_pred_speed_kmh - actual_speed_kmh)
                
                frames_counted += 1
                
            frame_buffer.append(tensor_img)
            
            # Draw HUD
            vis = draw_hud(raw, actual_steer, actual_speed_kmh, m1_pred_steer, m1_pred_speed_kmh, m2_pred_steer, m2_pred_speed_kmh, fps_display)
            cv2.imshow("Model Comparison", vis)
            
            fps_frames += 1
            now = time.perf_counter()
            if now - t_fps_ref >= 1.0:
                fps_display = fps_frames / (now - t_fps_ref)
                fps_frames = 0
                t_fps_ref = now
                
            if cv2.waitKey(1) in (27, ord('q')):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("\n\n" + "="*50)
        print("          COMPARISON RESULTS")
        print("="*50)
        if frames_counted > 0:
            print(f"Total Frames Analyzed: {frames_counted}")
            print("\n--- MODEL 1 (V3 DUAL-STREAM) ---")
            print(f"Average Steering Error: {m1_steer_err_sum / frames_counted:.4f}")
            print(f"Average Speed Error:    {m1_speed_err_sum / frames_counted:.2f} km/h")
            print("\n--- MODEL 2 (V1 SINGLE-STREAM) ---")
            print(f"Average Steering Error: {m2_steer_err_sum / frames_counted:.4f}")
            print(f"Average Speed Error:    {m2_speed_err_sum / frames_counted:.2f} km/h")
        else:
            print("No frames were fully processed.")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
