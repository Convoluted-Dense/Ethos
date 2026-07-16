"""
test_cnn_v2.py
==============
Real-time inference script for the V2 model (2-frame stacked, 3 outputs).
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
            "Stage 1 (Early Edges)": model.model.features[1],
            "Stage 3 (Textures)": model.model.features[3],
            "Stage 5 (Mid-Late Shapes)": model.model.features[5],
            "Stage 7 (High Semantics)": model.model.features[7]
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
VJOY_AXIS_MIN = 0x0
VJOY_AXIS_MAX = 0x8000  # 32768
VJOY_AXIS_MID = (VJOY_AXIS_MAX + VJOY_AXIS_MIN) // 2

HID_USAGE_X = 0x30  # Steering
HID_USAGE_Y = 0x31  # Throttle / Speed

# ---------------------------------------------------------------------------
# Steering model definition V2 (must match train_steering_v2.py exactly)
# ---------------------------------------------------------------------------
class SteeringModelV2(nn.Module):
    def __init__(self):
        super(SteeringModelV2, self).__init__()

        weights = EfficientNet_B1_Weights.DEFAULT
        self.model = efficientnet_b1(weights=weights)

        old_conv   = self.model.features[0][0]
        old_weight = old_conv.weight.data.clone()

        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        new_conv.weight.data[:, :3, :, :] = old_weight
        new_conv.weight.data[:, 3:, :, :] = old_weight * 0.5
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()

        self.model.features[0][0] = new_conv

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 3)  # [steering, scaled_speed, lateral_offset]
        )

    def forward(self, x):
        return self.model(x)


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
# vJoy helpers
# ---------------------------------------------------------------------------
def _steer_to_vjoy(steering: float) -> int:
    clamped = max(-1.0, min(1.0, steering))
    return int(VJOY_AXIS_MID + clamped * (VJOY_AXIS_MAX - VJOY_AXIS_MID))

def _speed_to_vjoy(scaled_speed: float) -> int:
    clamped = max(0.0, min(1.0, scaled_speed))
    return int(VJOY_AXIS_MIN + clamped * (VJOY_AXIS_MAX - VJOY_AXIS_MIN))


class VJoySender:
    SEND_HZ = 60
    def __init__(self, vjoy_device):
        self._vjoy   = vjoy_device
        self._lock   = threading.Lock()
        self._steer  = VJOY_AXIS_MID
        self._speed  = VJOY_AXIS_MIN
        self._active = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set(self, steering_raw: float, scaled_speed: float):
        with self._lock:
            self._steer = _steer_to_vjoy(steering_raw)
            self._speed = _speed_to_vjoy(scaled_speed)

    def _loop(self):
        interval = 1.0 / self.SEND_HZ
        while self._active:
            t0 = time.perf_counter()
            with self._lock:
                self._vjoy.data.wAxisX = self._steer
                self._vjoy.data.wAxisY = self._speed
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
                self._vjoy.data.wAxisX = VJOY_AXIS_MID
                self._vjoy.data.wAxisY = VJOY_AXIS_MIN
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

def draw_hud(img: np.ndarray, steering: float, speed_kmh: float, offset: float, fps: float, vjoy_active: bool) -> np.ndarray:
    vis = img.copy()
    ph, pw = 186, 460
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
    bar_x = 175
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
    # Offset is typically [-0.25, 0.25]. We scale it by 4 so it fills the bar nicely.
    display_offset = max(-1.0, min(1.0, offset * 4.0))
    if display_offset > 0:
        cv2.rectangle(vis, (mid_x, y - 14), (mid_x + int(display_offset * bar_w / 2), y), (255, 100, 100), -1)
    elif display_offset < 0:
        cv2.rectangle(vis, (mid_x + int(display_offset * bar_w / 2), y - 14), (mid_x, y), (255, 100, 100), -1)
    cv2.line(vis, (mid_x, y - 16), (mid_x, y + 2), _CLR_WHITE, 1)

    # Speed
    y += 28
    _put(vis, f"Speed:  {speed_kmh:6.1f} km/h", 10, y)
    MAX_DISPLAY_KMH = 150.0
    sv = min(speed_kmh / MAX_DISPLAY_KMH, 1.0)
    cv2.rectangle(vis, (bar_x, y - 14), (bar_x + bar_w, y), (50, 50, 50), -1)
    cv2.rectangle(vis, (bar_x, y - 14), (bar_x + int(sv * bar_w), y), _CLR_GREEN, -1)

    y += 26
    _put(vis, "Q / ESC to quit", 10, y, (100, 100, 100), scale=0.50, thickness=1)

    return vis


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def make_parser():
    p = argparse.ArgumentParser(description="BeamNG CNN real-time inference (V2)")
    p.add_argument("--model",     default="best_steering_v2_model.pth",
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
    model = SteeringModelV2().to(device)
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
        if not HAS_GRAD_CAM:
            print("[warn] --cam passed but pytorch-grad-cam not installed.")
        else:
            target_layers = [model.model.features[-1]]
            cam = GradCAM(model=model, target_layers=target_layers)
            print("[init] Grad-CAM enabled for steering prediction.")

    act_tracker = None
    if opt.activations:
        act_tracker = ActivationTracker(model)
        print("[init] Activation map visualization enabled.")

    vjoy = None
    if not opt.no_vjoy:
        try:
            import pyvjoy
            vjoy = pyvjoy.VJoyDevice(1)
            vjoy.data.wAxisX  = VJOY_AXIS_MID
            vjoy.data.wAxisY  = VJOY_AXIS_MIN
            vjoy.update()
            print("[init] vJoy device 1 acquired and centred")
        except Exception as e:
            print(f"[warn] Could not open vJoy device 1: {e}")
            vjoy = None

    vjoy_active = vjoy is not None
    vjoy_sender = None
    if vjoy_active:
        vjoy_sender = VJoySender(vjoy)
        print(f"[init] vJoy sender thread started at {VJoySender.SEND_HZ} Hz")

    if not opt.headless:
        cv2.namedWindow("BeamNG CNN", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("BeamNG CNN", opt.width, opt.height)

    # 1-frame buffer for temporal context
    frame_buffer = deque(maxlen=5)

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
            raw = capture_printwindow(hwnd)
            if raw is None:
                time.sleep(0.01)
                continue

            # 2. Preprocess
            curr_tensor = preprocess_frame(raw).to(device)

            frame_buffer.append(curr_tensor)

            frames = list(frame_buffer)
            while len(frames) < 5:
                frames.insert(0, frames[0])

            # Build (1, 5, 3, 240, 240) tensor
            tensor = torch.stack(frames, dim=0).unsqueeze(0)

            # 3. Inference
            if cam is not None:
                tensor.requires_grad_(True)
                out = model(tensor)
                targets = [RegressionTarget(target_idx=0)]
                grayscale_cam = cam(input_tensor=tensor, targets=targets)[0, :]
                
                # Reconstruct current frame (channels 3:6) for display
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                curr_t_unnorm = tensor[0:1, -1, :, :, :] * std + mean
                img_float = curr_t_unnorm[0].permute(1, 2, 0).detach().cpu().numpy()
                img_float = np.clip(img_float, 0, 1)
                img_bgr = img_float[:, :, ::-1]
                
                cam_view = show_cam_on_image(img_bgr, grayscale_cam, use_rgb=False)
                cam_view_large = cv2.resize(cam_view, (960, 198))
                cv2.imshow("Model View (Grad-CAM)", cam_view_large)
            elif act_tracker is not None:
                out = model(tensor)
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
                            out = model(tensor)
                    else:
                        out = model(tensor)

            pred_steering     = out[0, 0].item() * opt.steer_gain
            pred_scaled_speed = out[0, 1].item()
            pred_offset       = out[0, 2].item()
            
            pred_speed_kmh    = pred_scaled_speed * max_speed * 3.6

            pred_steering     = max(-1.0, min(1.0, pred_steering))
            pred_scaled_speed = max(0.0,  min(1.0, pred_scaled_speed))

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
                vis = draw_hud(raw, pred_steering, pred_speed_kmh, pred_offset, fps_display, vjoy_active)
                dh, dw = vis.shape[:2]
                scale = min(opt.width / dw, opt.height / dh)
                if scale != 1.0:
                    vis = cv2.resize(vis, (int(dw * scale), int(dh * scale)), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("BeamNG CNN", vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
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
