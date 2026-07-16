import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

# ---------------------------------------------------------------------------
# Steering model definition (must match train_steering.py exactly)
# ---------------------------------------------------------------------------
class SteeringModel(nn.Module):
    def __init__(self):
        super(SteeringModel, self).__init__()
        
        weights = EfficientNet_B1_Weights.DEFAULT
        self.model = efficientnet_b1(weights=weights)
        
        # Replace the final classification head
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 2) # Output: [steering, scaled_speed]
        )
        
    def forward(self, x):
        return self.model(x)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def preprocess(frame_bgr: np.ndarray) -> torch.Tensor:
    if frame_bgr.shape[1] != 1280 or frame_bgr.shape[0] != 720:
        frame_bgr = cv2.resize(frame_bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)

    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img = TF.crop(img, top=231, left=0, height=264, width=1280)
    img = TF.resize(img, (240, 240))

    tensor = TF.to_tensor(img)
    tensor = _NORMALIZE(tensor)

    return tensor.unsqueeze(0)  # add batch dim

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

def draw_hud(img: np.ndarray, steering: float, speed_kmh: float, fps: float) -> np.ndarray:
    vis = img.copy()
    ph, pw = 160, 460
    panel = vis[:ph, :pw].copy()
    cv2.rectangle(panel, (0, 0), (pw, ph), (8, 8, 8), -1)
    cv2.addWeighted(panel, 0.60, vis[:ph, :pw], 0.40, 0, vis[:ph, :pw])

    y = 28
    _put(vis, "vJoy: VIDEO TEST MODE", 10, y, (200, 200, 50))

    y += 26
    _put(vis, f"FPS: {fps:5.1f}", 10, y, _CLR_WHITE)

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
    p = argparse.ArgumentParser(description="BeamNG CNN video inference")
    p.add_argument("--video",     default="test.mp4",
                   help="path to input video file")
    p.add_argument("--model",     default="best_steering_velocity_model.pth",
                   help="path to trained .pth file")
    p.add_argument("--max-speed", type=float, default=None,
                   help="speed scaling factor")
    p.add_argument("--width",     type=int, default=1280,
                   help="display window width (default: 1280)")
    p.add_argument("--height",    type=int, default=720,
                   help="display window height (default: 720)")
    p.add_argument("--steer-gain", type=float, default=1.0,
                   help="multiplier applied to the predicted steering")
    return p

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    opt = make_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[init] Using device: {device}")

    print(f"[init] Loading model from: {opt.model}")
    if not os.path.exists(opt.model):
        print(f"ERROR: Model file not found at {opt.model}")
        sys.exit(1)
        
    model = SteeringModel().to(device)
    model.load_state_dict(torch.load(opt.model, map_location=device, weights_only=True))
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

    print(f"[init] Opening video: {opt.video}")
    cap = cv2.VideoCapture(opt.video)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file {opt.video}")
        sys.exit(1)

    cv2.namedWindow("Video Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Inference", opt.width, opt.height)

    fps_display = 0.0
    fps_frames  = 0
    t_fps_ref   = time.perf_counter()

    print("\n[inference] Running -- press Q or ESC to stop.\n")

    try:
        while True:
            ret, raw = cap.read()
            if not ret:
                print("[info] End of video.")
                break

            # 2. Preprocess
            tensor = preprocess(raw).to(device)

            # 3. Inference
            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = model(tensor)
                else:
                    out = model(tensor)

            pred_steering     = out[0, 0].item() * opt.steer_gain
            pred_scaled_speed = out[0, 1].item()
            
            pred_speed_kmh    = pred_scaled_speed * max_speed * 3.6

            pred_steering     = max(-1.0, min(1.0, pred_steering))
            pred_scaled_speed = max(0.0,  min(1.0, pred_scaled_speed))

            # 5. Display / FPS
            fps_frames += 1
            now = time.perf_counter()
            if now - t_fps_ref >= 1.0:
                fps_display = fps_frames / (now - t_fps_ref)
                fps_frames = 0
                t_fps_ref = now

            vis = draw_hud(raw, pred_steering, pred_speed_kmh, fps_display)
            dh, dw = vis.shape[:2]
            scale = min(opt.width / dw, opt.height / dh)
            if scale != 1.0:
                vis = cv2.resize(vis, (int(dw * scale), int(dh * scale)), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Video Inference", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        print("\nStopped by Ctrl-C.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[exit] Done.")

if __name__ == "__main__":
    main()
