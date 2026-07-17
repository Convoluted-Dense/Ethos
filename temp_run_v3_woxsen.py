"""
temp_run_v3_woxsen.py
=====================
Runs the V3 model on woxsen.mp4 asynchronously at 30 FPS.
Does not slow down the video playback. Inference runs in a background thread,
and the video display uses the latest predictions.
"""

import os
import sys
import cv2
import csv
import time
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from collections import deque
import threading

# ---------------------------------------------------------------------------
# Activation Tracker for Visualizing Feature Maps
# ---------------------------------------------------------------------------
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
# V3 Model Definition
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
# Preprocessing (returns CPU tensor to avoid blocking the main thread)
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
# Global Asynchronous Communication Variables
# ---------------------------------------------------------------------------
pred_steering = 0.0
pred_speed = 0.0
pred_offset = 0.0

latest_activation_strip = None
predictions_log = []  # shared list to collect predictions
predictions_lock = threading.Lock()

frame_lock = threading.Lock()
latest_tensors = None  # (prev_tensor, curr_tensor)
new_frame_event = threading.Event()
running = True

# ---------------------------------------------------------------------------
# Background Inference Worker
# ---------------------------------------------------------------------------
def inference_worker(model, act_tracker, device):
    global pred_steering, pred_speed, pred_offset, latest_activation_strip, running
    
    print("[worker] Inference thread started.")
    while running:
        # Wait for a new frame pair to process
        got_event = new_frame_event.wait(timeout=0.1)
        if not got_event:
            continue
        new_frame_event.clear()
        
        with frame_lock:
            if latest_tensors is None:
                continue
            prev_cpu, curr_cpu, current_frame_idx = latest_tensors
        
        # Move tensors to GPU/device inside background thread
        prev_tensor = prev_cpu.to(device).unsqueeze(0)
        curr_tensor = curr_cpu.to(device).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(prev_tensor, curr_tensor)
            else:
                out = model(prev_tensor, curr_tensor)
        
        # Get predictions
        p_steer = out[0, 0].item()
        p_speed = out[0, 1].item()
        p_offset = out[0, 2].item()
        
        # Save atomically
        pred_steering = p_steer
        pred_speed = p_speed
        pred_offset = p_offset
        
        with predictions_lock:
            predictions_log.append([current_frame_idx, p_steer, p_speed, p_offset])
            
        # Get Activation Maps
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
            latest_activation_strip = np.hstack(activation_maps)

# ---------------------------------------------------------------------------
# Main Video Playback Loop (runs at 30 FPS)
# ---------------------------------------------------------------------------
def main():
    global latest_tensors, running
    
    model_path = "best_steering_v3_model.pth"
    video_path = "woxsen.mp4"
    output_csv = "woxsen_predictions.csv"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading V3 model...")
    model = SteeringModelV3().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # Setup Activation Tracker
    act_tracker = ActivationTracker(model)

    # Start background inference worker thread
    worker_thread = threading.Thread(target=inference_worker, args=(model, act_tracker, device), daemon=True)
    worker_thread.start()

    # Open video
    print(f"Opening video {video_path}...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total video frames: {total_frames}")

    # Set up live window
    window_name = "Woxsen V3 Live Inference (30 FPS Async)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    frame_buffer = deque(maxlen=1)
    paused = False

    frame_idx = 0
    
    # 30 FPS target interval
    target_interval = 1.0 / 30.0

    try:
        while cap.isOpened():
            t_start = time.perf_counter()
            
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                display_frame = frame.copy()
                
                # Preprocess frame on CPU (very fast, doesn't block GUI)
                curr_tensor = preprocess_frame(frame)
                prev_tensor = frame_buffer[0] if frame_buffer else curr_tensor
                frame_buffer.append(curr_tensor)

                # Feed background thread
                with frame_lock:
                    latest_tensors = (prev_tensor, curr_tensor, frame_idx)
                new_frame_event.set()
                
                frame_idx += 1
            else:
                # If paused, keep displaying the current frame
                pass

            # --- Draw Live UI Overlay using latest prediction values ---
            h, w = display_frame.shape[:2]
            
            # semi-transparent panel for stats
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (380, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

            # Print stats
            cv2.putText(display_frame, f"Frame: {frame_idx}/{total_frames}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Pred Steering: {pred_steering:.4f}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 150, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Pred Speed: {pred_speed:.4f}", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 120), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Pred Offset: {pred_offset:.4f}", (20, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 80, 80), 2, cv2.LINE_AA)

            # Steering Graphic at bottom center
            cx, cy = w // 2, h - 60
            cv2.line(display_frame, (cx - 150, cy), (cx + 150, cy), (200, 200, 200), 2)
            cv2.circle(display_frame, (cx, cy), 5, (255, 255, 255), -1)
            
            # Predicted steer dot (orange)
            steer_x = int(cx + pred_steering * 150)
            cv2.circle(display_frame, (steer_x, cy), 12, (0, 140, 255), -1)
            cv2.putText(display_frame, "Steer", (cx - 20, cy - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if paused:
                cv2.putText(display_frame, "PAUSED (Space to resume)", (w // 2 - 150, h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

            # Show windows on main thread
            cv2.imshow(window_name, display_frame)
            
            # Show activations if updated
            if latest_activation_strip is not None:
                cv2.imshow("Model Activations", latest_activation_strip)

            # Enforce exactly 30 FPS playback speed
            t_end = time.perf_counter()
            elapsed = t_end - t_start
            delay_ms = max(1, int((target_interval - elapsed) * 1000)) if not paused else 100
            
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord(' '):
                paused = not paused
            elif key == ord('q') or key == 27: # ESC or Q
                print("Exiting live playback early...")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        # Cleanup
        running = False
        cap.release()
        act_tracker.remove()
        cv2.destroyAllWindows()

    # Write predictions to CSV
    print(f"Writing predictions to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "steering", "speed", "offset"])
        with predictions_lock:
            # Sort predictions by frame_idx in case thread logging got slightly out of order
            predictions_log.sort(key=lambda x: x[0])
            writer.writerows(predictions_log)

    print("Done! Predictions printed below (first 10):")
    for row in predictions_log[:10]:
        print(f"Frame {row[0]}: Steering={row[1]:.4f}, Speed={row[2]:.4f}, Offset={row[3]:.4f}")

if __name__ == "__main__":
    main()
