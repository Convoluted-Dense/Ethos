"""
test_woxsen.py
==============
Runs the V2, V3, or both models on woxsen.mp4 asynchronously at 30 FPS.
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
    def __init__(self, model, is_v2=False):
        self.activations = {}
        self.hooks = []
        
        if is_v2:
            self.stages = {
                "Stage 1 (Early Edges)": model.backbone[0][1],
                "Stage 3 (Textures)": model.backbone[0][3],
                "Stage 5 (Mid-Late Shapes)": model.backbone[0][5],
                "Stage 7 (High Semantics)": model.backbone[0][7]
            }
        else:
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
# V2 Model Definition (LSTM Sequence)
# ---------------------------------------------------------------------------
class SteeringModelV2(nn.Module):
    def __init__(self):
        super(SteeringModelV2, self).__init__()
        weights = EfficientNet_B1_Weights.DEFAULT
        base = efficientnet_b1(weights=weights)
        
        self.backbone = nn.Sequential(
            base.features,
            base.avgpool,
            nn.Flatten()
        )
        
        for i in range(1, 5):
            for param in self.backbone[0][i].parameters():
                param.requires_grad = False
                
        self.lstm = nn.LSTM(input_size=1280, hidden_size=256, num_layers=1, batch_first=True)
        
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        B, seq_len, C, H, W = x.size()
        x_flat = x.view(B * seq_len, C, H, W)
        features = self.backbone(x_flat)
        features = features.view(B, seq_len, -1)
        lstm_out, (hn, cn) = self.lstm(features)
        last_hidden = hn[-1]
        return self.head(last_hidden)

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
# Global Asynchronous Communication Variables
# ---------------------------------------------------------------------------
pred_steering_v2 = 0.0
pred_speed_v2 = 0.0
pred_offset_v2 = 0.0

pred_steering_v3 = 0.0
pred_speed_v3 = 0.0
pred_offset_v3 = 0.0

latest_activation_strip_v2 = None
latest_activation_strip_v3 = None

predictions_log = []
predictions_lock = threading.Lock()

frame_lock = threading.Lock()
latest_tensors = None  # (seq_5_tensor, prev_tensor, curr_tensor, frame_idx)
new_frame_event = threading.Event()
running = True

# ---------------------------------------------------------------------------
# Background Inference Worker
# ---------------------------------------------------------------------------
def inference_worker(model_v2, act_tracker_v2, model_v3, act_tracker_v3, device, version):
    global pred_steering_v2, pred_speed_v2, pred_offset_v2, latest_activation_strip_v2
    global pred_steering_v3, pred_speed_v3, pred_offset_v3, latest_activation_strip_v3
    global running
    
    print(f"[worker] Inference thread started (version={version}).")
    stages_to_show = ["Stage 1 (Early Edges)", "Stage 3 (Textures)", "Stage 5 (Mid-Late Shapes)", "Stage 7 (High Semantics)"]

    while running:
        got_event = new_frame_event.wait(timeout=0.1)
        if not got_event:
            continue
        new_frame_event.clear()
        
        with frame_lock:
            if latest_tensors is None:
                continue
            seq_5_cpu, prev_cpu, curr_cpu, current_frame_idx = latest_tensors
        
        seq_5_tensor = seq_5_cpu.to(device).unsqueeze(0)
        prev_tensor = prev_cpu.to(device).unsqueeze(0)
        curr_tensor = curr_cpu.to(device).unsqueeze(0)
        
        # --- V2 Inference ---
        if version in ['v2', 'both'] and model_v2:
            with torch.no_grad():
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out_v2 = model_v2(seq_5_tensor)
                else:
                    out_v2 = model_v2(seq_5_tensor)
            
            pred_steering_v2 = out_v2[0, 0].item()
            pred_speed_v2 = out_v2[0, 1].item()
            pred_offset_v2 = out_v2[0, 2].item()
            
            activation_maps = []
            for stage_name in stages_to_show:
                feat_full = act_tracker_v2.activations.get(stage_name)
                if feat_full is not None:
                    # feat_full shape is (seq_len, C, H, W)
                    feat = feat_full[-1:] # grab only the last frame in sequence
                    act = torch.norm(feat[0], p=2, dim=0).cpu().numpy()
                    act_min, act_max = act.min(), act.max()
                    if act_max > act_min: act = (act - act_min) / (act_max - act_min)
                    else: act = np.zeros_like(act)
                    act = (act * 255).astype(np.uint8)
                    color_act = cv2.applyColorMap(act, cv2.COLORMAP_JET)
                    color_act = cv2.resize(color_act, (240, 240))
                    cv2.putText(color_act, stage_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    activation_maps.append(color_act)
            if activation_maps:
                latest_activation_strip_v2 = np.hstack(activation_maps)

        # --- V3 Inference ---
        if version in ['v3', 'both'] and model_v3:
            with torch.no_grad():
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out_v3 = model_v3(prev_tensor, curr_tensor)
                else:
                    out_v3 = model_v3(prev_tensor, curr_tensor)
            
            pred_steering_v3 = out_v3[0, 0].item()
            pred_speed_v3 = out_v3[0, 1].item()
            pred_offset_v3 = out_v3[0, 2].item()
            
            activation_maps = []
            for stage_name in stages_to_show:
                feat = act_tracker_v3.activations.get(stage_name)
                if feat is not None:
                    act = torch.norm(feat[0], p=2, dim=0).cpu().numpy()
                    act_min, act_max = act.min(), act.max()
                    if act_max > act_min: act = (act - act_min) / (act_max - act_min)
                    else: act = np.zeros_like(act)
                    act = (act * 255).astype(np.uint8)
                    color_act = cv2.applyColorMap(act, cv2.COLORMAP_JET)
                    color_act = cv2.resize(color_act, (240, 240))
                    cv2.putText(color_act, stage_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    activation_maps.append(color_act)
            if activation_maps:
                latest_activation_strip_v3 = np.hstack(activation_maps)

        with predictions_lock:
            p_log = [current_frame_idx]
            if version in ['v2', 'both']:
                p_log.extend([pred_steering_v2, pred_speed_v2, pred_offset_v2])
            else:
                p_log.extend([None, None, None])
                
            if version in ['v3', 'both']:
                p_log.extend([pred_steering_v3, pred_speed_v3, pred_offset_v3])
            else:
                p_log.extend([None, None, None])
                
            predictions_log.append(p_log)


# ---------------------------------------------------------------------------
# Main Video Playback Loop
# ---------------------------------------------------------------------------
def main():
    global latest_tensors, running
    
    version = input("Choose model version (v2/v3/both): ").strip().lower()
    if version not in ['v2', 'v3', 'both']:
        print("Invalid version, defaulting to both")
        version = 'both'
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_v2, act_tracker_v2 = None, None
    model_v3, act_tracker_v3 = None, None

    if version in ['v2', 'both']:
        path_v2 = "best_steering_v2_model.pth"
        if os.path.exists(path_v2):
            print(f"Loading V2 model from {path_v2}...")
            model_v2 = SteeringModelV2().to(device)
            model_v2.load_state_dict(torch.load(path_v2, map_location=device))
            model_v2.eval()
            act_tracker_v2 = ActivationTracker(model_v2, is_v2=True)
        else:
            print(f"Error: {path_v2} not found.")

    if version in ['v3', 'both']:
        path_v3 = "best_steering_v3_model.pth"
        if os.path.exists(path_v3):
            print(f"Loading V3 model from {path_v3}...")
            model_v3 = SteeringModelV3().to(device)
            model_v3.load_state_dict(torch.load(path_v3, map_location=device))
            model_v3.eval()
            act_tracker_v3 = ActivationTracker(model_v3, is_v2=False)
        else:
            print(f"Error: {path_v3} not found.")

    worker_thread = threading.Thread(target=inference_worker, 
                                     args=(model_v2, act_tracker_v2, model_v3, act_tracker_v3, device, version), 
                                     daemon=True)
    worker_thread.start()

    video_path = "woxsen.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        sys.exit(1)

    print(f"Opening video {video_path}...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total video frames: {total_frames}")

    window_name = f"Woxsen Live Inference ({version.upper()})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    frame_buffer = deque(maxlen=5)
    paused = False
    frame_idx = 0
    target_interval = 1.0 / 30.0

    try:
        while cap.isOpened():
            t_start = time.perf_counter()
            
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                display_frame = frame.copy()
                
                curr_tensor = preprocess_frame(frame)
                frame_buffer.append(curr_tensor)

                tensor_list = list(frame_buffer)
                while len(tensor_list) < 5:
                    tensor_list.insert(0, tensor_list[0])
                seq_5 = torch.stack(tensor_list, dim=0)
                
                prev_tensor = tensor_list[-2]

                with frame_lock:
                    latest_tensors = (seq_5, prev_tensor, curr_tensor, frame_idx)
                new_frame_event.set()
                
                frame_idx += 1
            else:
                pass

            h, w = display_frame.shape[:2]
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

            cv2.putText(display_frame, f"Frame: {frame_idx}/{total_frames}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            y_offset = 70
            if version in ['v2', 'both']:
                cv2.putText(display_frame, f"V2 Steer: {pred_steering_v2:.4f} | Speed: {pred_speed_v2:.2f}", 
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 150, 255), 2)
                y_offset += 30
            
            if version in ['v3', 'both']:
                cv2.putText(display_frame, f"V3 Steer: {pred_steering_v3:.4f} | Speed: {pred_speed_v3:.2f}", 
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

            cx, cy = w // 2, h - 80
            cv2.line(display_frame, (cx - 150, cy), (cx + 150, cy), (200, 200, 200), 2)
            cv2.line(display_frame, (cx - 150, cy+20), (cx + 150, cy+20), (200, 200, 200), 2)
            cv2.circle(display_frame, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(display_frame, (cx, cy+20), 5, (255, 255, 255), -1)
            
            if version in ['v2', 'both']:
                steer_x_v2 = int(cx + pred_steering_v2 * 150)
                cv2.circle(display_frame, (steer_x_v2, cy), 10, (80, 150, 255), -1)
                cv2.putText(display_frame, "V2", (cx - 190, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 150, 255), 2)
            
            if version in ['v3', 'both']:
                steer_x_v3 = int(cx + pred_steering_v3 * 150)
                cv2.circle(display_frame, (steer_x_v3, cy + 20), 10, (0, 140, 255), -1)
                cv2.putText(display_frame, "V3", (cx - 190, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

            if paused:
                cv2.putText(display_frame, "PAUSED (Space to resume)", (w // 2 - 150, h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.imshow(window_name, display_frame)
            if version in ['v2', 'both'] and latest_activation_strip_v2 is not None:
                cv2.imshow("Model Activations V2", latest_activation_strip_v2)
            if version in ['v3', 'both'] and latest_activation_strip_v3 is not None:
                cv2.imshow("Model Activations V3", latest_activation_strip_v3)

            t_end = time.perf_counter()
            elapsed = t_end - t_start
            delay_ms = max(1, int((target_interval - elapsed) * 1000)) if not paused else 100
            
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord(' '):
                paused = not paused
            elif key == ord('q') or key == 27:
                print("Exiting live playback early...")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        running = False
        cap.release()
        if act_tracker_v2: act_tracker_v2.remove()
        if act_tracker_v3: act_tracker_v3.remove()
        cv2.destroyAllWindows()

    output_csv = "woxsen_predictions_compare.csv"
    print(f"Writing predictions to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "v2_steering", "v2_speed", "v2_offset", "v3_steering", "v3_speed", "v3_offset"])
        with predictions_lock:
            predictions_log.sort(key=lambda x: x[0])
            writer.writerows(predictions_log)
    print("Done!")

if __name__ == "__main__":
    main()
