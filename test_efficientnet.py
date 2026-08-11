"""
test_efficientnet.py
====================
Live inference script for the EfficientNet-B0 steering predictor with 3D Waypoint Overlay.

  - Captures BeamNG.drive window with PrintWindow (same method as beamng_collect.py)
  - Keeps a rolling 2-frame buffer, preprocesses each pair identically to training
  - Runs inference at ~10 FPS and outputs steering on vJoy Device 1 X-axis
  - Displays an Autonomous Navigation HUD showing:
      * 10 future steering predictions projected as WAYPOINTS on the live road ahead
      * Connected path trajectory curve with step labels
      * Current vJoy output bar & steering gauge
      * Live FPS counter & latency statistics

Usage
-----
    python test_efficientnet.py                          # auto-finds best checkpoint
    python test_efficientnet.py --model models/steering_efficientnet_best.pth
    python test_efficientnet.py --fps 10 --smooth 0.35

Controls
--------
    Q or ESC  : quit
    SPACE     : toggle vJoy output on/off (pause mode)
"""

import argparse
import ctypes
import json
import math
import sys
import time
from collections import deque
from ctypes import wintypes
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------------------------------------------------------------------------
# Try to import vJoy
# ---------------------------------------------------------------------------
try:
    import pyvjoy as _pyvjoy
    _VJOY_OK = True
except ImportError:
    _VJOY_OK = False

# ---------------------------------------------------------------------------
# Preprocessing constants — MUST match train_efficientnet.py exactly
# ---------------------------------------------------------------------------
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
TOP_CROP_FRAC  = 0.30
INPUT_H        = 224
INPUT_W        = 224
FUTURE_STEPS   = 1
VJOY_AXIS_MAX  = 0x8000    # 32768 — vJoy full-scale
VJOY_CENTRE    = VJOY_AXIS_MAX // 2   # 16384 — neutral / centre

# ---------------------------------------------------------------------------
# BeamNG window helpers (copied verbatim from beamng_collect.py)
# ---------------------------------------------------------------------------
class WindowInfo:
    def __init__(self, hwnd, title):
        self.hwnd  = hwnd
        self.title = title


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
    user32.PrintWindow(hwnd, mfcDC, 2)  # PW_RENDERFULLCONTENT

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
    return img[:, :, :3].copy()  # BGRA -> BGR


# ---------------------------------------------------------------------------
# EfficientNet-B0 model import
# ---------------------------------------------------------------------------
try:
    from train_efficientnet import SteeringNet
except ImportError:
    class SteeringNet(nn.Module):
        def __init__(self, dropout: float = 0.30):
            super().__init__()
            backbone = models.efficientnet_b0(weights=None)
            old_conv = backbone.features[0][0]
            new_conv = nn.Conv2d(6, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None))
            backbone.features[0][0] = new_conv
            self.features = backbone.features
            self.avgpool  = nn.AdaptiveAvgPool2d(1)
            in_features = 1280
            self.steering_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
            self.speed_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        def forward(self, x):
            feat = torch.flatten(self.avgpool(self.features(x)), 1)
            return self.steering_head(feat), self.speed_head(feat)


# ---------------------------------------------------------------------------
# Preprocessing — identical to SteeringDataset in training
# ---------------------------------------------------------------------------
_mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def preprocess_frame(bgr: np.ndarray, crop_frac: float = TOP_CROP_FRAC) -> torch.Tensor:
    """BGR ndarray -> (3, 224, 224) normalised tensor (CPU)."""
    h = bgr.shape[0]
    y_start = int(h * crop_frac)
    cropped = bgr[y_start:, :, :]
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = (t - _mean) / _std
    return t


def make_input(frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
    """Stack two (3,H,W) tensors into (1,6,H,W) batch."""
    return torch.cat([frame_a, frame_b], dim=0).unsqueeze(0)


# ---------------------------------------------------------------------------
# vJoy interface
# ---------------------------------------------------------------------------
class VJoyOutput:
    """Send steering value [-1, +1] to vJoy Device 1 X axis."""

    def __init__(self):
        self._joy = None
        if not _VJOY_OK:
            print("[vjoy]  pyvjoy not installed — joystick output disabled.")
            return
        try:
            self._joy = _pyvjoy.VJoyDevice(1)
            self._joy.reset()
            print("[vjoy]  Device 1 acquired — steering -> X axis.")
        except Exception as e:
            print(f"[vjoy]  Failed to acquire Device 1: {e}")
            self._joy = None

    def send(self, steering: float):
        """steering in [-1.0, +1.0]; maps to vJoy X axis [0x0000, 0x8000]."""
        if self._joy is None:
            return
        clamped = max(-1.0, min(1.0, steering))
        axis_val = int(VJOY_CENTRE + clamped * VJOY_CENTRE)
        axis_val = max(0, min(VJOY_AXIS_MAX, axis_val))
        self._joy.set_axis(_pyvjoy.HID_USAGE_X, axis_val)

    def centre(self):
        if self._joy is not None:
            self._joy.set_axis(_pyvjoy.HID_USAGE_X, VJOY_CENTRE)

    def close(self):
        self.centre()


# ---------------------------------------------------------------------------
# BeamNG ROI Steering Extractor (extracts true steering from on-screen HUD)
# ---------------------------------------------------------------------------
def load_steering_roi(roi_file: str = "steering_roi.json"):
    path = Path(roi_file)
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return tuple(data["roi"]) if "roi" in data and data["roi"] else None
        except Exception:
            pass
    return None


def extract_steering_from_roi(img: np.ndarray, roi: tuple | None) -> float | None:
    """Extract true steering angle [-1.0, +1.0] from BeamNG HUD orange steering bar."""
    if roi is None or img is None:
        return None
    x, y, w, h = roi
    img_h, img_w = img.shape[:2]
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        return None
    crop = img[y:y+h, x:x+w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mid = w // 2
    left_orange  = np.count_nonzero(mask[:, :mid])
    right_orange = np.count_nonzero(mask[:, mid:])
    max_area = mid * h
    if max_area > 0 and (left_orange > 0 or right_orange > 0):
        left_val  = left_orange  / max_area
        right_val = right_orange / max_area
        return max(-1.0, min(1.0, right_val - left_val))
    return 0.0


# ---------------------------------------------------------------------------
# Active Learning Miner — saves hard / uncertain frames during inference
# ---------------------------------------------------------------------------
class ActiveLearningMiner:
    """
    Monitors live inference predictions and automatically saves frames where
    the model is uncertain or struggling.  Saved frames go to
    ``dataset/active_learning/pending/`` with a JSON sidecar containing
    metadata (uncertainty score, predictions, reason for flagging, and
    true ground-truth steering angle if available via ROI/telemetry).
    """

    def __init__(self,
                 output_dir: str = "dataset/active_learning/pending",
                 jitter_thresh: float = 0.04,
                 correction_thresh: float = 0.15,
                 error_thresh: float = 0.05,
                 cooldown_sec: float = 0.5,
                 buffer_size: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jitter_thresh = jitter_thresh
        self.correction_thresh = correction_thresh
        self.error_thresh = error_thresh
        self.cooldown_sec = cooldown_sec
        self.roi = load_steering_roi()
        if self.roi is not None:
            print(f"[mining] Loaded steering ROI: {self.roi} (HUD ground-truth capture ENABLED)")

        self._pred_buffer: deque[float] = deque(maxlen=buffer_size)
        self._last_save_time: float = 0.0
        self.enabled: bool = False  # toggled by M key
        self.frames_mined: int = 0
        self.session_jitter_flags: int = 0
        self.session_correction_flags: int = 0
        self.session_manual_flags: int = 0
        self.session_error_flags: int = 0

    # ── core check ────────────────────────────────────────────
    def check_and_save(self, bgr_frame: np.ndarray,
                       raw_steer: float, smoothed_steer: float,
                       speed: float,
                       true_steer: float | None = None) -> str | None:
        """Evaluate current prediction and save if uncertain.

        Returns the reason string if a frame was saved, else None.
        """
        if not self.enabled:
            return None

        self._pred_buffer.append(raw_steer)

        if len(self._pred_buffer) < 3:
            return None

        now = time.time()
        if (now - self._last_save_time) < self.cooldown_sec:
            return None

        reason = None
        jitter = float(np.std(list(self._pred_buffer)))
        correction = abs(raw_steer - smoothed_steer)

        # Try capturing true steering from BeamNG HUD ROI if not explicitly provided
        if true_steer is None and self.roi is not None:
            true_steer = extract_steering_from_roi(bgr_frame, self.roi)

        error = abs(raw_steer - true_steer) if true_steer is not None else 0.0

        if jitter > self.jitter_thresh:
            reason = f"jitter={jitter:.4f}"
            self.session_jitter_flags += 1
        elif correction > self.correction_thresh:
            reason = f"correction={correction:.4f}"
            self.session_correction_flags += 1
        elif error > self.error_thresh:
            reason = f"pred_error={error:.4f}"
            self.session_error_flags += 1

        if reason is not None:
            self._save_frame(bgr_frame, raw_steer, smoothed_steer,
                             speed, jitter, correction, reason, true_steer)
            return reason
        return None

    def force_save(self, bgr_frame: np.ndarray,
                   raw_steer: float, smoothed_steer: float,
                   speed: float,
                   true_steer: float | None = None) -> str:
        """Manually flag and save the current frame (F key)."""
        jitter = float(np.std(list(self._pred_buffer))) if len(self._pred_buffer) >= 2 else 0.0
        correction = abs(raw_steer - smoothed_steer)
        reason = "manual"
        self.session_manual_flags += 1

        if true_steer is None and self.roi is not None:
            true_steer = extract_steering_from_roi(bgr_frame, self.roi)

        self._save_frame(bgr_frame, raw_steer, smoothed_steer,
                         speed, jitter, correction, reason, true_steer)
        return reason

    # ── internal save ─────────────────────────────────────────
    def _save_frame(self, bgr_frame: np.ndarray,
                    raw_steer: float, smoothed_steer: float,
                    speed: float, jitter: float,
                    correction: float, reason: str,
                    true_steer: float | None = None):
        ts = time.time()
        stem = f"{ts:.3f}"
        img_path = self.output_dir / f"{stem}.jpg"
        meta_path = self.output_dir / f"{stem}.json"

        cv2.imwrite(str(img_path), bgr_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        label_source = "HUD_ROI" if true_steer is not None else "MODEL_PREDICTION"
        target_steer = true_steer if true_steer is not None else raw_steer

        meta = {
            "timestamp": ts,
            "target_steer": round(target_steer, 6),
            "true_steer": round(true_steer, 6) if true_steer is not None else None,
            "raw_steer": round(raw_steer, 6),
            "smoothed_steer": round(smoothed_steer, 6),
            "label_source": label_source,
            "speed_kmh": round(speed, 2),
            "jitter": round(jitter, 6),
            "correction": round(correction, 6),
            "reason": reason,
            "prediction_buffer": [round(v, 5) for v in self._pred_buffer],
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self._last_save_time = ts
        self.frames_mined += 1

    @property
    def status_text(self) -> str:
        if not self.enabled:
            return "MINING: OFF"
        return f"MINING: ON  ({self.frames_mined} saved)"


# ---------------------------------------------------------------------------
# Multi-Layer Grad-CAM Extractor (Early -> Mid -> Deep)
# ---------------------------------------------------------------------------
class MultiLayerGradCAM:
    """
    Extracts real-time Grad-CAM heatmaps for key stages of EfficientNet-B0:
      - Early Stage  (features[2]): Low-level lane lines & edges
      - Mid Stage    (features[4]): Intermediate road geometry & curvature
      - Deep Stage   (features[8]): High-level semantic steering decisions
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = {}
        self.gradients = {}
        
        base_features = getattr(model, "features", None)
        if base_features is None and hasattr(model, "module"):
            base_features = getattr(model.module, "features", None)
            
        self.target_layers = {}
        if base_features is not None:
            if len(base_features) > 2:
                self.target_layers["L2 (Early: Edges & Lines)"] = base_features[2]
            if len(base_features) > 4:
                self.target_layers["L4 (Mid: Road Geometry)"] = base_features[4]
            if len(base_features) > 8:
                self.target_layers["L8 (Deep: Steering Focus)"] = base_features[8]
            elif len(base_features) > 0:
                self.target_layers["Final Stage"] = base_features[-1]

        self.hooks = []
        for name, layer in self.target_layers.items():
            self.hooks.append(layer.register_forward_hook(self._make_act_hook(name)))
            self.hooks.append(layer.register_full_backward_hook(self._make_grad_hook(name)))

    def _make_act_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output
        return hook

    def _make_grad_hook(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0]
        return hook

    def compute_heatmaps(self, inp_tensor: torch.Tensor, bgr_raw: np.ndarray, crop_frac: float = TOP_CROP_FRAC) -> dict[str, np.ndarray]:
        """Runs forward + backward pass to generate layer Grad-CAM overlays."""
        self.activations.clear()
        self.gradients.clear()

        inp_var = inp_tensor.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        
        with torch.enable_grad():
            preds_out = self.model(inp_var)
            if isinstance(preds_out, tuple):
                steer_pred = preds_out[0]
            else:
                steer_pred = preds_out

            steer_target = steer_pred[0, 0]
            steer_target.backward()

        heatmaps = {}
        h, w = bgr_raw.shape[:2]
        y_start = int(h * crop_frac)
        crop_h = max(1, h - y_start)

        for name in self.target_layers.keys():
            if name in self.activations and name in self.gradients:
                act = self.activations[name].detach()
                grad = self.gradients[name].detach()

                weights = torch.mean(grad, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * act, dim=1, keepdim=True)
                cam = torch.relu(cam)

                cam_np = cam[0, 0].cpu().numpy()
                max_val, min_val = cam_np.max(), cam_np.min()
                if max_val > min_val:
                    cam_np = (cam_np - min_val) / (max_val - min_val + 1e-8)
                else:
                    cam_np = np.zeros_like(cam_np)

                cam_uint8 = (cam_np * 255).astype(np.uint8)
                heatmap_resized = cv2.resize(cam_uint8, (w, crop_h), interpolation=cv2.INTER_LINEAR)
                heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

                overlay = bgr_raw.copy()
                overlay[y_start:, :] = cv2.addWeighted(bgr_raw[y_start:, :], 0.45, heatmap_color, 0.55, 0)
                
                # Draw top crop boundary line on heatmap thumbnail
                cv2.line(overlay, (0, y_start), (w, y_start), (0, 200, 255), 1, cv2.LINE_AA)
                heatmaps[name] = overlay
            else:
                heatmaps[name] = bgr_raw.copy()

        return heatmaps

    def remove_hooks(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# HUD & Waypoint Rendering
# ---------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
_CLR_BG    = (15, 15, 20)
_CLR_TITLE = (0, 220, 255)
_CLR_GOOD  = (0, 230, 130)
_CLR_WARN  = (0, 130, 255)
_CLR_OUT   = (0, 220, 255)
_CLR_PAUSE = (0, 100, 255)
_CLR_WHITE = (240, 240, 240)
_CLR_GREY  = (130, 130, 130)


def _text(img, txt, x, y, color=_CLR_WHITE, scale=0.55, thick=1):
    cv2.putText(img, txt, (x+1, y+1), FONT, scale, (0,0,0), thick+1, cv2.LINE_AA)
    cv2.putText(img, txt, (x,   y  ), FONT, scale, color,   thick,   cv2.LINE_AA)


def draw_hud_waypoints(bgr_raw: np.ndarray,
                       predictions: list[float],
                       smoothed_steer: float,
                       fps: float,
                       vjoy_ok: bool,
                       paused: bool,
                       inference_ms: float,
                       speed_val: float = 0.0,
                       physics_vals: list[float] = None,
                       traj_vals: list[float] = None,
                       gradcam_heatmaps: dict[str, np.ndarray] = None,
                       show_gradcam: bool = True,
                       crop_frac: float = TOP_CROP_FRAC,
                       mining_stats: dict = None) -> np.ndarray:
    """
    Build a 1150x780 HUD panel with:
      - Left panel (760x500): Live BeamNG video with 3D Waypoint Path overlay on the road!
      - Right panel (340x500): Multi-Task telemetry (Speed, Steering bar, stats)
      - Bottom panel (1110x200): Real-Time Multi-Layer Grad-CAM Heatmaps (L2 Early, L4 Mid, L8 Deep)
    """
    if physics_vals is None:
        physics_vals = [0.0, 0.0, 0.0]

    HUD_W = 1150
    HUD_H = 780 if (gradcam_heatmaps and show_gradcam) else 620
    canvas = np.zeros((HUD_H, HUD_W, 3), dtype=np.uint8)
    canvas[:] = _CLR_BG

    # ── 1. MAIN VIDEO WITH WAYPOINT OVERLAY (LEFT PANEL) ──────────────────
    VID_W, VID_H = 760, 520
    video_view = cv2.resize(bgr_raw, (VID_W, VID_H), interpolation=cv2.INTER_LINEAR)

    # Calculate 3D-like waypoint curve projecting out onto the road ahead
    # Car origin is at bottom center of video view
    origin_x = VID_W // 2
    origin_y = VID_H - 15
    
    points = [(origin_x, origin_y)]
    cur_x = float(origin_x)
    cur_y = float(origin_y)
    heading = 0.0

    # Step size upwards towards horizon (horizon is roughly at y=180 in 520h view)
    step_y = (origin_y - 180) / max(1, FUTURE_STEPS)

    for i, steer in enumerate(predictions):
        # Accumulate steering angle into visual heading curve
        heading += steer * 0.32
        # Move forward and laterally
        cur_y -= step_y
        cur_x += step_y * math.sin(heading) * 1.6
        points.append((int(cur_x), int(cur_y)))

    # Draw path connecting waypoints
    for i in range(len(points) - 1):
        pt1 = points[i]
        pt2 = points[i + 1]
        
        # Color gradient: bright cyan at car hood (t+1), fading to yellow-green far ahead
        alpha = 0.0 if FUTURE_STEPS <= 1 else i / (FUTURE_STEPS - 1)
        r = int(0   * (1 - alpha) + 50  * alpha)
        g = int(255 * (1 - alpha) + 200 * alpha)
        b = int(220 * (1 - alpha) + 50  * alpha)
        color = (b, g, r)

        # Draw line glow & shadow
        cv2.line(video_view, pt1, pt2, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(video_view, pt1, pt2, color, 3, cv2.LINE_AA)

    # Draw waypoint circles and labels
    for i in range(1, len(points)):
        pt = points[i]
        steer_val = predictions[i - 1]
        
        alpha = 0.0 if FUTURE_STEPS <= 1 else (i - 1) / (FUTURE_STEPS - 1)
        r = int(0   * (1 - alpha) + 50  * alpha)
        g = int(255 * (1 - alpha) + 200 * alpha)
        b = int(220 * (1 - alpha) + 50  * alpha)
        color = (b, g, r)

        # Highlight t+1 waypoint with outer ring
        if i == 1:
            cv2.circle(video_view, pt, 10, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(video_view, pt, 6, _CLR_GOOD, -1, cv2.LINE_AA)
        else:
            cv2.circle(video_view, pt, 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(video_view, pt, 4, color, -1, cv2.LINE_AA)

        # Text label near waypoint (show every even step or t+1 to avoid clutter)
        if i == 1 or i % 2 == 0:
            lbl = f"t{i}: {steer_val:+.2f}"
            cv2.putText(video_view, lbl, (pt[0] + 10, pt[1] + 4),
                        FONT, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(video_view, lbl, (pt[0] + 10, pt[1] + 4),
                        FONT, 0.42, color if i > 1 else (255, 255, 255), 1, cv2.LINE_AA)

    # Draw top crop boundary line faintly on video view
    crop_y = int(VID_H * crop_frac)
    cv2.line(video_view, (0, crop_y), (VID_W, crop_y), (0, 150, 200), 1, cv2.LINE_AA)
    _text(video_view, f"Top {crop_frac*100:.0f}% Cropped for AI", 10, max(14, crop_y - 6), (0, 200, 255), 0.40)

    # Place video view on canvas
    canvas[40:40+VID_H, 20:20+VID_W] = video_view
    cv2.rectangle(canvas, (19, 39), (20+VID_W, 40+VID_H), _CLR_TITLE, 1)

    # ── 2. RIGHT SIDEBAR (MULTI-TASK STATS & CONTROLS) ────────────────────
    SB_X = 790
    SB_W = 340
    
    _text(canvas, "MULTI-TASK AI TELEMETRY", SB_X, 50, _CLR_TITLE, 0.62, 2)

    # Output Steering Bar
    sy = 78
    _text(canvas, f"vJoy Steer: {smoothed_steer:+.4f}", SB_X, sy, _CLR_OUT, 0.55, 2)
    sy += 14
    cv2.rectangle(canvas, (SB_X, sy), (SB_X + SB_W, sy + 26), (35, 35, 45), -1)
    mid_x = SB_X + SB_W // 2
    fill = int(abs(smoothed_steer) * SB_W // 2)
    bar_clr = _CLR_GOOD if not paused else _CLR_PAUSE
    if smoothed_steer >= 0:
        cv2.rectangle(canvas, (mid_x, sy + 2), (mid_x + fill, sy + 24), bar_clr, -1)
    else:
        cv2.rectangle(canvas, (mid_x - fill, sy + 2), (mid_x, sy + 24), bar_clr, -1)
    cv2.line(canvas, (mid_x, sy), (mid_x, sy + 26), (200, 200, 200), 1)

    if paused:
        _text(canvas, "[ vJoy PAUSED - SPACE to resume ]", SB_X, sy + 40, _CLR_PAUSE, 0.46)

    # AI Predicted Speed & Physics (NEW Multi-Task Display!)
    sy = 138
    _text(canvas, "AI PREDICTED SPEED & PHYSICS", SB_X, sy, _CLR_TITLE, 0.52, 2)
    sy += 22
    _text(canvas, f"Speed     : {speed_val:5.1f} km/h", SB_X, sy, _CLR_GOOD, 0.52, 2)
    # Mini speed bar (0 to 120 km/h)
    cv2.rectangle(canvas, (SB_X + 160, sy - 12), (SB_X + SB_W, sy + 2), (35, 35, 45), -1)
    spd_fill = int(min(120.0, max(0.0, speed_val)) / 120.0 * (SB_W - 160))
    cv2.rectangle(canvas, (SB_X + 160, sy - 12), (SB_X + 160 + spd_fill, sy + 2), (0, 255, 200), -1)
    cv2.rectangle(canvas, (SB_X + 160, sy - 12), (SB_X + SB_W, sy + 2), (80, 80, 100), 1)
    
    sy += 24
    _text(canvas, f"Yaw Rate  : {physics_vals[0]:+.3f} rad/s", SB_X, sy, _CLR_WHITE, 0.48)
    sy += 20
    _text(canvas, f"Lat G (X) : {physics_vals[1]:+.2f} m/s²", SB_X, sy, _CLR_WHITE, 0.48)
    sy += 20
    _text(canvas, f"Lon G (Y) : {physics_vals[2]:+.2f} m/s²", SB_X, sy, _CLR_WHITE, 0.48)

    # Performance Stats
    sy = 250
    _text(canvas, "SYSTEM PERFORMANCE", SB_X, sy, _CLR_WHITE, 0.52)
    sy += 22
    _text(canvas, f"FPS: {fps:5.1f}   Latency: {inference_ms:.1f} ms", SB_X, sy, _CLR_GOOD, 0.50)
    sy += 20
    status_color = _CLR_GOOD if vjoy_ok else _CLR_WARN
    _text(canvas, f"vJoy Axis: {'ACTIVE (Dev 1 X)' if vjoy_ok else 'DISABLED'}", SB_X, sy, status_color, 0.50)

    # Active Learning Mining Status
    if mining_stats is not None:
        sy += 20
        enabled = mining_stats.get("enabled", False)
        mine_clr = _CLR_GOOD if enabled else _CLR_GREY
        status_txt = f"MINING: {'ON' if enabled else 'OFF'}  ({mining_stats.get('total', 0)} saved)"
        _text(canvas, status_txt, SB_X, sy, mine_clr, 0.50)
        
        if enabled:
            sy += 16
            _text(canvas, f"Jitter : {mining_stats.get('jitter', 0)}", SB_X + 10, sy, _CLR_WHITE, 0.44)
            sy += 16
            _text(canvas, f"Correct: {mining_stats.get('correction', 0)}", SB_X + 10, sy, _CLR_WHITE, 0.44)
            sy += 16
            _text(canvas, f"Error  : {mining_stats.get('error', 0)}", SB_X + 10, sy, _CLR_WHITE, 0.44)
            sy += 16
            _text(canvas, f"Manual : {mining_stats.get('manual', 0)}", SB_X + 10, sy, _CLR_WHITE, 0.44)

    # Waypoint Numeric Table
    sy = 315
    _text(canvas, "TCP WAYPOINTS: STEER | LAT (m)", SB_X, sy, _CLR_WHITE, 0.50)
    sy += 8
    cv2.rectangle(canvas, (SB_X, sy), (SB_X + SB_W, sy + 185), (25, 25, 35), -1)
    cv2.rectangle(canvas, (SB_X, sy), (SB_X + SB_W, sy + 185), (60, 60, 80), 1)

    col1_x = SB_X + 8
    for i in range(min(5, FUTURE_STEPS)):
        # Left column t+1 to t+5
        t_l = i
        steer_l = predictions[t_l]
        lat_l_str = f"|{traj_vals[t_l]:+.2f}m" if traj_vals is not None and len(traj_vals) > t_l else ""
        clr_l = _CLR_GOOD if t_l == 0 else _CLR_WHITE
        _text(canvas, f"t+{t_l+1:2d}: {steer_l:+.3f}{lat_l_str}", col1_x, sy + 25 + i * 32, clr_l, 0.44)

    # ── 3. BOTTOM PANEL: GRAD-CAM MULTI-LAYER ATTENTION HEATMAPS ────────
    if gradcam_heatmaps and show_gradcam:
        _text(canvas, "GRAD-CAM LAYER ATTENTION HEATMAPS (EARLY -> MID -> DEEP FEATURE STAGES)", 20, 562, _CLR_TITLE, 0.50, 2)
        panel_y = 574
        box_w, box_h = 356, 180
        spacing = 16

        for idx, (layer_name, heatmap_img) in enumerate(gradcam_heatmaps.items()):
            x_offset = 20 + idx * (box_w + spacing)
            if x_offset + box_w > HUD_W:
                break
            resized_hm = cv2.resize(heatmap_img, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
            canvas[panel_y:panel_y + box_h, x_offset:x_offset + box_w] = resized_hm
            cv2.rectangle(canvas, (x_offset, panel_y), (x_offset + box_w, panel_y + box_h), (0, 180, 220), 1)
            
            # Header label badge on heatmap thumbnail
            cv2.rectangle(canvas, (x_offset, panel_y), (x_offset + box_w, panel_y + 22), (20, 20, 30), -1)
            clr_hdr = _CLR_GOOD if "Deep" in layer_name or "L8" in layer_name else _CLR_WHITE
            _text(canvas, layer_name, x_offset + 8, panel_y + 15, clr_hdr, 0.42, 1)

    # Bottom hints
    _text(canvas, "Controls: Q/ESC: Quit | SPACE: Pause | A/D: Step Frame | H: Heatmaps | Sliders: Crop % & Frame",
          20, HUD_H - 12, _CLR_GREY, 0.44)
    
    # Title bar
    cv2.rectangle(canvas, (0, 0), (HUD_W, 26), (0, 160, 220), -1)
    _text(canvas, "EfficientNet-B0  |  3D Waypoint Navigation & Multi-Layer Grad-CAM HUD  |  BeamNG.drive AI",
          10, 18, (255, 255, 255), 0.52, 2)

    return canvas


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="EfficientNet-B0 live steering inference for BeamNG.drive")
    parser.add_argument("--model",  default="models/best_val_model.pth",
                        help="Path to best_val_model.pth state dict or checkpoint")
    parser.add_argument("--video",  default=None,
                        help="Path to input video file (e.g. woxsen.mp4) instead of window capture")
    parser.add_argument("--fps",    type=float, default=10.0,
                        help="Target inference FPS (default: 10)")
    parser.add_argument("--smooth", type=float, default=0.35,
                        help="EMA smoothing for vJoy output (0=raw, 1=static)")
    parser.add_argument("--device", default="cuda",
                        help="Torch device: cuda or cpu")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    frame_interval = 1.0 / args.fps
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  EfficientNet-B0 - Live Waypoint Navigation Inference")
    print(f"{'='*60}")
    print(f"  Target FPS : {args.fps}")
    print(f"  Smoothing  : {args.smooth}")
    print(f"  Device     : {device}")
    if args.video:
        print(f"  Input Video: {args.video}")
    print()

    # ── Robust Model Search ─────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        fallbacks = [
            Path("models/steering_efficientnet_best.pth"),
            Path("models/best_val_model.pth"),
            Path("steering_efficientnet_best.pth"),
            Path("best_val_model.pth"),
        ]
        for fb in fallbacks:
            if fb.exists():
                print(f"[model] '{args.model}' not found — automatically falling back to '{fb}'")
                model_path = fb
                break

    if not model_path.exists():
        print(f"ERROR: No model checkpoints found at '{model_path}' or fallback locations!")
        print("       Run train_efficientnet.py first to generate a checkpoint.")
        sys.exit(1)

    print(f"[model] Loading SteeringNet from {model_path} ...")
    model = SteeringNet().to(device)
    state = torch.load(str(model_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    print("[model] Loaded OK")

    # ── Video Source vs BeamNG window ───────────────────────────
    cap = None
    win = None
    if args.video:
        vid_path = Path(args.video)
        if not vid_path.exists():
            print(f"ERROR: Video file '{args.video}' not found.")
            sys.exit(1)
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"ERROR: Could not open video file '{args.video}'.")
            sys.exit(1)
        print(f"[input] Playing video file: '{args.video}'")
    else:
        print("[window] Searching for BeamNG.drive ...")
        win = find_beamng_window()
        if win is None:
            print("ERROR: BeamNG.drive window not found. Start the game or pass --video <path.mp4>.")
            sys.exit(1)

    # ── vJoy ────────────────────────────────────────────────────
    vjoy = VJoyOutput()
    vjoy_active = vjoy._joy is not None

    # ── Frame buffer (2 consecutive frames) ─────────────────────
    frame_buf: deque[torch.Tensor] = deque(maxlen=2)

    # ── Grad-CAM Extractor ──────────────────────────────────────
    gradcam = MultiLayerGradCAM(model)
    show_gradcam = True
    gradcam_heatmaps = None

    # ── Active Learning Miner ───────────────────────────────────
    miner = ActiveLearningMiner()
    print(f"[mining] Active Learning Miner ready  (output: {miner.output_dir})")
    print(f"         Press M to toggle mining, F to force-flag a frame.")

    WIN_NAME = "BeamNG AI - 3D Waypoint Navigation"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Crop %", WIN_NAME, int(TOP_CROP_FRAC * 100), 60, lambda x: None)

    total_frames = 0
    if cap is not None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            cv2.createTrackbar("Frame", WIN_NAME, 0, total_frames - 1, lambda x: None)

    print("\n[inference] Starting — press Q or ESC in the UI window to quit.")
    print("            SPACE toggles Pause / Resume.")
    print("            A / D or Left / Right arrows step backward / forward 1 frame when paused.")
    print("            H toggles Multi-Layer Grad-CAM Heatmaps on/off.")
    print("            M toggles Active Learning mining on/off.")
    print("            F force-flags the current frame for active learning.")
    print("            Use top trackbars to adjust Crop % and scrub Video Frames.\n")

    fps_display   = 0.0
    fps_frames    = 0
    t_fps_ref     = time.perf_counter()
    smoothed      = 0.0
    predictions   = [0.0] * FUTURE_STEPS
    speed_val     = 0.0
    physics_vals  = [0.0, 0.0, 0.0]
    traj_vals     = [0.0] * FUTURE_STEPS
    paused        = False
    inference_ms  = 0.0

    try:
        while True:
            t0 = time.perf_counter()

            # Read dynamic crop percentage slider
            crop_val = cv2.getTrackbarPos("Crop %", WIN_NAME)
            if crop_val < 0: crop_val = int(TOP_CROP_FRAC * 100)
            crop_frac = max(0.0, min(0.60, crop_val / 100.0))

            # Handle Frame trackbar seeking
            if cap is not None and total_frames > 0:
                user_frame_pos = cv2.getTrackbarPos("Frame", WIN_NAME)
                current_cap_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                # User manually dragged the frame trackbar
                if user_frame_pos >= 0 and abs(user_frame_pos - current_cap_pos) > 2:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, user_frame_pos)
                    frame_buf.clear()

            # 1. Capture (from Video file or BeamNG window)
            if cap is not None:
                if not paused:
                    ret, raw = cap.read()
                    if not ret or raw is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, raw = cap.read()
                    if not ret or raw is None:
                        time.sleep(0.05)
                        continue
                    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if total_frames > 0 and current_pos >= 0:
                        cv2.setTrackbarPos("Frame", WIN_NAME, min(total_frames - 1, current_pos))
                else:
                    # Paused mode: read current frame without advancing
                    pos = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ret, raw = cap.read()
                    if not ret or raw is None:
                        time.sleep(0.05)
                        continue
            else:
                raw = capture_printwindow(win.hwnd)
                if raw is None:
                    time.sleep(0.05)
                    continue

            # 2. Resize to fixed resolution
            if raw.shape[1] != args.width or raw.shape[0] != args.height:
                raw = cv2.resize(raw, (args.width, args.height),
                                 interpolation=cv2.INTER_LINEAR)

            # 3. Preprocess frame and push to buffer
            tensor = preprocess_frame(raw, crop_frac=crop_frac)
            frame_buf.append(tensor)

            # 4. Inference — only when we have 2 frames
            if len(frame_buf) == 2:
                inp = make_input(frame_buf[0], frame_buf[1]).to(device)

                t_inf = time.perf_counter()
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    preds_out = model(inp)
                inference_ms = (time.perf_counter() - t_inf) * 1000

                # Compute Grad-CAM heatmaps across EfficientNet layers using dynamic crop_frac
                if show_gradcam:
                    gradcam_heatmaps = gradcam.compute_heatmaps(inp, raw, crop_frac=crop_frac)

                if isinstance(preds_out, tuple):
                    steer_preds = preds_out[0]
                    if len(preds_out) > 1 and preds_out[1] is not None:
                        speed_val = float(preds_out[1][0, 0].cpu().numpy())
                    if len(preds_out) > 2 and preds_out[2] is not None:
                        physics_vals = preds_out[2][0].cpu().numpy().tolist()
                    if len(preds_out) > 3 and preds_out[3] is not None:
                        traj_vals = preds_out[3][0].cpu().numpy().tolist()
                else:
                    steer_preds  = preds_out
                    speed_val    = 0.0
                    physics_vals = [0.0, 0.0, 0.0]

                preds_np    = steer_preds[0].cpu().numpy().tolist()
                predictions = preds_np

                # t+1 prediction is what we output to joystick
                raw_steer = float(preds_np[0])

                # EMA smoothing
                alpha    = 1.0 - args.smooth
                smoothed = alpha * raw_steer + (1.0 - alpha) * smoothed
                smoothed = max(-1.0, min(1.0, smoothed))

                # Send to vJoy
                if vjoy_active and not paused:
                    vjoy.send(smoothed)

                # Active Learning: check if this frame should be mined
                al_reason = miner.check_and_save(raw, raw_steer, smoothed, speed_val)
                if al_reason:
                    tqdm_msg = f"[mining] Saved frame #{miner.frames_mined}  ({al_reason})"
                    print(tqdm_msg)

            # 5. FPS counter
            fps_frames += 1
            now = time.perf_counter()
            dt  = now - t_fps_ref
            if dt >= 1.0:
                fps_display = fps_frames / dt
                fps_frames  = 0
                t_fps_ref   = now

            # 6. Draw Waypoints & Multi-Task HUD with Grad-CAM Heatmaps
            mining_stats = {
                "enabled": miner.enabled,
                "total": miner.frames_mined,
                "jitter": miner.session_jitter_flags,
                "correction": miner.session_correction_flags,
                "error": miner.session_error_flags,
                "manual": miner.session_manual_flags,
            }
            
            hud = draw_hud_waypoints(
                raw, predictions, smoothed,
                fps_display, vjoy_active and not paused,
                paused, inference_ms, speed_val, physics_vals, traj_vals,
                gradcam_heatmaps=gradcam_heatmaps, show_gradcam=show_gradcam,
                crop_frac=crop_frac, mining_stats=mining_stats)

            cv2.imshow(WIN_NAME, hud)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # Q or ESC
                break
            elif key == ord(" "):       # SPACE — toggle pause
                paused = not paused
                if paused:
                    vjoy.centre()
                    print("\n[input] Playback PAUSED")
                else:
                    print("\n[input] Playback RESUMED")
            elif key in (ord("a"), ord("A"), 81):  # A or Left arrow — step frame back
                if cap is not None:
                    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    new_pos = max(0, pos - 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    frame_buf.clear()
                    if total_frames > 0:
                        cv2.setTrackbarPos("Frame", WIN_NAME, new_pos)
            elif key in (ord("d"), ord("D"), 83):  # D or Right arrow — step frame forward
                if cap is not None:
                    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    new_pos = min(max(0, total_frames - 1), pos)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    frame_buf.clear()
                    if total_frames > 0:
                        cv2.setTrackbarPos("Frame", WIN_NAME, new_pos)
            elif key in (ord("h"), ord("H")):  # H — toggle Grad-CAM heatmaps
                show_gradcam = not show_gradcam
                print(f"\n[gradcam] Heatmaps {'ENABLED' if show_gradcam else 'DISABLED'}")
            elif key in (ord("m"), ord("M")):  # M — toggle Active Learning mining
                miner.enabled = not miner.enabled
                print(f"\n[mining] Active Learning mining {'ENABLED' if miner.enabled else 'DISABLED'}")
            elif key in (ord("f"), ord("F")):  # F — force-flag current frame
                reason = miner.force_save(raw, raw_steer if 'raw_steer' in dir() else 0.0,
                                          smoothed, speed_val)
                print(f"\n[mining] Force-flagged frame #{miner.frames_mined}  ({reason})")

            # 7. Pace to target FPS
            elapsed = time.perf_counter() - t0
            slack   = frame_interval - elapsed
            if slack > 0:
                time.sleep(slack)

    except KeyboardInterrupt:
        print("\nStopped by Ctrl-C.")
    finally:
        gradcam.remove_hooks()
        if cap is not None:
            cap.release()
        vjoy.close()
        cv2.destroyAllWindows()

        # Active Learning session summary
        if miner.frames_mined > 0:
            print(f"\n{'='*60}")
            print(f"  Active Learning Mining Session Summary")
            print(f"{'='*60}")
            print(f"  Total frames mined : {miner.frames_mined}")
            print(f"  Jitter flags       : {miner.session_jitter_flags}")
            print(f"  Correction flags   : {miner.session_correction_flags}")
            print(f"  Pred error flags   : {miner.session_error_flags}")
            print(f"  Manual flags       : {miner.session_manual_flags}")
            print(f"  Output directory   : {miner.output_dir}")
            print(f"{'='*60}")
            print(f"  Next step: run  python active_learning_curator.py")
            print(f"{'='*60}")

        print("\nDone.")


if __name__ == "__main__":
    main()
