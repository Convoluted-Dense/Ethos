"""
train_efficientnet.py
=====================
Trains an EfficientNet-B0 backbone to predict steering and speed for the next
frame (n+1) from a stack of 2 consecutive frames.  The model is domain-agnostic:
each sample randomly draws from either the real (img/) or diffused (img_diffused/)
domain, and the steering labels are shared between both.

Key features
------------
  - 2 consecutive frames stacked as 6-channel input (same domain, same augmentation)
  - Top 30% of each frame cropped (removes sky / HUD)
  - 30% chance of webcam-shake augmentation (from sim_webcam_shake.py)
  - Weighted sampling to fix severe steering-angle imbalance
  - Huber loss (robust to outlier labels) for both steering and speed
  - Rich color / weather augmentation (HSV jitter, gamma, blur, shadows)
  - Batch-level Mixup augmentation (alpha=0.2, 30% probability)
  - Cosine annealing with warm restarts + linear warm-up
  - EMA (Exponential Moving Average) model for better generalization
  - Gradient accumulation for larger effective batch sizes
  - Chronological train / val split (no temporal leakage)
  - Mixed-precision (fp16) training on CUDA
  - Best-val-loss checkpointing (EMA model)

Usage
-----
    python train_efficientnet.py                       # default settings
    python train_efficientnet.py --epochs 100 --bs 8   # custom
    python train_efficientnet.py --dataset dataset2     # different dataset folder
"""

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn
from torchvision import models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import webcam-shake augmentation helpers from sim_webcam_shake.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_webcam_shake import (
    apply_chromatic_aberration,
    apply_directional_blur,
    apply_rolling_shutter_and_shake,
    add_sensor_noise,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
FUTURE_STEPS  = 1           # predict steering & speed at next frame (n+1)
TOP_CROP_FRAC = 0.30        # remove top 30 % of the image
INPUT_H       = 224
INPUT_W       = 224
MIN_SPEED_KMH = 5.0         # skip near-stationary frames

# Steering bins for weighted sampling
STEER_BINS = [-1.0, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1.01]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SteeringDataset(Dataset):
    """
    Yields (stacked_tensor, steer_target, speed_target) where:
      - stacked_tensor: (6, H, W) 2 consecutive frames (n, n+1)
      - steer_target  : (1,) steering at frame n+1
      - speed_target  : (1,) speed (km/h) at frame n+1
    """

    def __init__(self, indices: list[int], rows: list[dict],
                 img_dir: Path, diffused_dir: Path,
                 is_train: bool = True,
                 webcam_shake_prob: float = 0.30):
        self.indices       = indices
        self.rows          = rows
        self.img_dir       = img_dir
        self.diffused_dir  = diffused_dir
        self.is_train      = is_train
        self.shake_prob    = webcam_shake_prob if is_train else 0.0

        self._diffused_names = set(f.name for f in diffused_dir.glob("*.jpg"))
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.indices)

    def _load_bgr(self, path: Path) -> np.ndarray | None:
        try:
            if not path.is_file() or path.stat().st_size == 0:
                return None
            img = cv2.imread(str(path))
            if img is not None and img.ndim == 3 and img.shape[0] > 10 and img.shape[1] > 10:
                return img
        except Exception:
            pass
        return None

    def _crop_top(self, img: np.ndarray) -> np.ndarray:
        h = img.shape[0]
        if h <= 10:
            return img
        y_start = int(h * TOP_CROP_FRAC)
        return img[y_start:, :, :]

    def _apply_webcam_shake(self, img1: np.ndarray, img2: np.ndarray, speed_kmh: float):
        vib_amp   = 1.2 + speed_kmh / 20.0
        phase     = random.uniform(0, 2 * math.pi)
        freq      = 3.5 + speed_kmh / 30.0

        dx  = random.gauss(0, vib_amp * 1.5)
        dy  = random.gauss(0, vib_amp * 2.0)
        rot = random.gauss(0, vib_amp * 0.12)

        shake_vx = random.gauss(0, 2.0)
        shake_vy = random.gauss(0, 2.0)

        results = []
        for img in (img1, img2):
            out = apply_rolling_shutter_and_shake(
                img, base_dx=dx, base_dy=dy, base_rot=rot,
                vib_amp=vib_amp, freq_jello=freq, phase=phase)
            out = apply_directional_blur(out, vx=shake_vx, vy=shake_vy)
            out = apply_chromatic_aberration(out, shift_x=2, shift_y=1)
            out = add_sensor_noise(out, noise_level=4.0 + speed_kmh / 25.0)
            results.append(out)

        return results[0], results[1]

    def _apply_color_augmentation(self, img: np.ndarray) -> np.ndarray:
        """Rich color / weather augmentation — simulates dawn, dusk, fog, rain."""
        # --- HSV hue / saturation jitter ---
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180  # hue shift ±10°
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)  # sat ±30%
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # --- Random gamma correction (bright / dark) ---
        if random.random() < 0.4:
            gamma = random.uniform(0.7, 1.4)
            table = np.array([((i / 255.0) ** gamma) * 255
                              for i in range(256)]).astype(np.uint8)
            img = cv2.LUT(img, table)

        # --- Gaussian blur (simulates lens fog / slight defocus) ---
        if random.random() < 0.15:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # --- Random shadow rectangle (simulates tree / overpass shadows) ---
        if random.random() < 0.2:
            h, w = img.shape[:2]
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            x2 = random.randint(x1 + w // 4, w)
            y2 = random.randint(y1 + h // 4, h)
            shadow = img.copy().astype(np.float32)
            shadow[y1:y2, x1:x2] *= random.uniform(0.4, 0.7)
            img = np.clip(shadow, 0, 255).astype(np.uint8)

        return img

    def _to_tensor_normalised(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = (t - self._mean) / self._std
        return t

    def _get_item_impl(self, idx: int):
        i = self.indices[idx]
        row_n = self.rows[i]
        row_n1 = self.rows[i + 1]

        fname_n  = row_n.get("frame")
        fname_n1 = row_n1.get("frame")

        if not fname_n or not fname_n1:
            raise RuntimeError(f"Missing frame filename at index {i}")

        can_diffuse = (fname_n in self._diffused_names and fname_n1 in self._diffused_names)
        use_diffused = can_diffuse and (random.random() < 0.5) if self.is_train else False

        if use_diffused:
            img_n  = self._load_bgr(self.diffused_dir / fname_n)
            img_n1 = self._load_bgr(self.diffused_dir / fname_n1)
            if img_n is None or img_n1 is None:
                img_n  = self._load_bgr(self.img_dir / fname_n)
                img_n1 = self._load_bgr(self.img_dir / fname_n1)
        else:
            img_n  = self._load_bgr(self.img_dir / fname_n)
            img_n1 = self._load_bgr(self.img_dir / fname_n1)

        if img_n is None or img_n1 is None:
            raise RuntimeError(f"Could not load image pair: {fname_n}, {fname_n1}")

        img_n  = self._crop_top(img_n)
        img_n1 = self._crop_top(img_n1)

        try:
            vx = float(row_n.get("velX", 0))
            vy = float(row_n.get("velY", 0))
            vz = float(row_n.get("velZ", 0))
            speed_kmh_n = math.sqrt(vx**2 + vy**2 + vz**2) * 3.6
            if math.isnan(speed_kmh_n) or math.isinf(speed_kmh_n): speed_kmh_n = 30.0
        except (ValueError, TypeError):
            speed_kmh_n = 30.0

        if random.random() < self.shake_prob:
            try:
                img_n, img_n1 = self._apply_webcam_shake(img_n, img_n1, speed_kmh_n)
            except Exception:
                pass

        if self.is_train:
            # Brightness / contrast jitter (original)
            brightness = random.uniform(0.85, 1.15)
            contrast   = random.uniform(0.85, 1.15)
            for arr in (img_n, img_n1):
                tmp = arr.astype(np.float32)
                tmp = tmp * contrast + (brightness - 1.0) * 128
                np.clip(tmp, 0, 255, out=tmp)
                arr[:] = tmp.astype(np.uint8)

            # Rich color / weather augmentation (NEW)
            if random.random() < 0.5:
                img_n  = self._apply_color_augmentation(img_n)
                img_n1 = self._apply_color_augmentation(img_n1)

        t_n  = self._to_tensor_normalised(img_n)
        t_n1 = self._to_tensor_normalised(img_n1)
        stacked = torch.cat([t_n, t_n1], dim=0)  # (6, H, W)

        # Labels for frame n+1
        steer_n1 = float(row_n1.get("steering_combined", row_n1.get("steering", 0.0)))
        if math.isnan(steer_n1) or math.isinf(steer_n1): steer_n1 = 0.0
        steer_t = torch.nan_to_num(torch.tensor([steer_n1], dtype=torch.float32), nan=0.0)

        try:
            vx1 = float(row_n1.get("velX", 0))
            vy1 = float(row_n1.get("velY", 0))
            vz1 = float(row_n1.get("velZ", 0))
            speed_kmh_n1 = math.sqrt(vx1**2 + vy1**2 + vz1**2) * 3.6
            if math.isnan(speed_kmh_n1) or math.isinf(speed_kmh_n1): speed_kmh_n1 = speed_kmh_n
        except (ValueError, TypeError):
            speed_kmh_n1 = speed_kmh_n
        speed_t = torch.nan_to_num(torch.tensor([speed_kmh_n1], dtype=torch.float32), nan=0.0)

        return stacked, steer_t, speed_t

    def __getitem__(self, idx: int):
        for attempt in range(10):
            try:
                return self._get_item_impl(idx)
            except Exception:
                idx = random.randint(0, len(self.indices) - 1)
        return (torch.zeros(6, INPUT_H, INPUT_W, dtype=torch.float32),
                torch.zeros(1, dtype=torch.float32),
                torch.zeros(1, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SteeringNet(nn.Module):
    """
    EfficientNet-B0 backbone predicting Steering and Speed for frame n+1:
      - Input: 6 channels (stacked frames n and n+1)
      - Head 1: Steering prediction (1 output)
      - Head 2: Speed prediction (km/h, 1 output)
    """

    def __init__(self, dropout: float = 0.30):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            6, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = old_conv.weight
            new_conv.weight *= 0.5
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        backbone.features[0][0] = new_conv

        self.features = backbone.features
        self.avgpool  = nn.AdaptiveAvgPool2d(1)

        for i, block in enumerate(self.features):
            if i <= 6:
                for p in block.parameters():
                    p.requires_grad = False

        in_features = 1280
        self.steering_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        self.speed_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        steer = self.steering_head(feat)
        speed = self.speed_head(feat)
        return steer, speed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_telemetry(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _is_valid_float(val) -> bool:
    if val is None:
        return False
    try:
        f = float(val)
        return not (math.isnan(f) or math.isinf(f))
    except (ValueError, TypeError):
        return False


def _is_valid_image(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def build_valid_indices(rows: list[dict], img_dir: Path, diffused_dir: Path,
                        min_speed_kmh: float = MIN_SPEED_KMH) -> list[int]:
    n = len(rows)
    valid = []
    skipped_missing_img = 0
    skipped_bad_label = 0
    skipped_bad_speed = 0

    for i in range(n - 1):
        row_n  = rows[i]
        row_n1 = rows[i + 1]
        fname_n  = row_n.get("frame")
        fname_n1 = row_n1.get("frame")

        if not fname_n or not fname_n1:
            skipped_missing_img += 1
            continue

        path_n  = img_dir / fname_n
        path_n1 = img_dir / fname_n1

        if not (_is_valid_image(path_n) and _is_valid_image(path_n1)):
            skipped_missing_img += 1
            continue

        steer_n1 = row_n1.get("steering_combined", row_n1.get("steering"))
        if not _is_valid_float(steer_n1):
            skipped_bad_label += 1
            continue

        vx = row_n.get("velX")
        vy = row_n.get("velY")
        vz = row_n.get("velZ")
        if not (_is_valid_float(vx) and _is_valid_float(vy) and _is_valid_float(vz)):
            skipped_bad_speed += 1
            continue

        speed_kmh = math.sqrt(float(vx)**2 + float(vy)**2 + float(vz)**2) * 3.6
        if speed_kmh < min_speed_kmh:
            skipped_bad_speed += 1
            continue

        valid.append(i)

    print(f"[data] Filtering stats: Skipped {skipped_missing_img} missing/corrupted images, "
          f"{skipped_bad_label} bad steering labels, {skipped_bad_speed} low/bad speed rows.")
    return valid


def build_sample_weights(rows: list[dict], indices: list[int]) -> list[float]:
    """
    Compute per-sample weights based on inverse frequency of steering bins.
    This ensures the sampler oversamples rare turn angles (sharp left/right)
    and undersamples the dominant straight-driving data.
    """
    steers = []
    for i in indices:
        row_n1 = rows[i + 1]
        s = float(row_n1.get("steering_combined", row_n1.get("steering", 0.0)))
        steers.append(s)

    # Digitise into bins
    bin_ids = np.digitize(steers, STEER_BINS[1:])  # bins 0..6
    bin_counts = np.bincount(bin_ids, minlength=len(STEER_BINS) - 1).astype(np.float64)
    bin_counts = np.maximum(bin_counts, 1.0)  # avoid div-by-zero

    # Inverse frequency weight
    bin_weights = 1.0 / bin_counts
    bin_weights /= bin_weights.sum()  # normalise so weights sum to 1

    sample_weights = [float(bin_weights[b]) for b in bin_ids]

    # Print distribution info
    bin_labels = [f"[{STEER_BINS[i]:+.2f}, {STEER_BINS[i+1]:+.2f})"
                  for i in range(len(STEER_BINS) - 1)]
    print("[data] Steering bin distribution & weights:")
    for label, cnt, w in zip(bin_labels, bin_counts, bin_weights):
        print(f"       {label}:  {int(cnt):6d} samples  ->  weight {w:.4f}")

    return sample_weights


def mixup_batch(imgs: torch.Tensor, steer: torch.Tensor, speed: torch.Tensor,
                alpha: float = 0.2):
    """
    Apply Mixup augmentation to a batch.  Linearly interpolates pairs of
    images AND labels, encouraging smoother regression boundaries.
    """
    if alpha <= 0:
        return imgs, steer, speed

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5 for stability

    batch_size = imgs.size(0)
    perm = torch.randperm(batch_size, device=imgs.device)

    mixed_imgs  = lam * imgs  + (1 - lam) * imgs[perm]
    mixed_steer = lam * steer + (1 - lam) * steer[perm]
    mixed_speed = lam * speed + (1 - lam) * speed[perm]

    return mixed_imgs, mixed_steer, mixed_speed


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def worker_init_fn(worker_id: int):
    """Called inside each DataLoader worker process on Windows (spawn).
    cv2.setNumThreads(0) in the main process has no effect on spawned workers
    because Windows uses 'spawn' (fresh Python interpreter per worker).
    Setting it here ensures OpenCV never tries to multi-thread inside a worker,
    which prevents the queue deadlock that otherwise occurs after epoch 1.
    """
    import cv2 as _cv2
    _cv2.setNumThreads(0)
    import random as _random
    import numpy as _np
    _random.seed(worker_id)
    _np.random.seed(worker_id)


class LinearWarmup:
    """Wraps a scheduler with a linear warm-up for the first `warmup_epochs` epochs."""
    def __init__(self, optimizer, scheduler, warmup_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self._step_count = 0

    def step(self):
        self._step_count += 1
        if self._step_count <= self.warmup_epochs:
            # Linear ramp from near-zero to base_lr
            warmup_lr = self.base_lr * (self._step_count / self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = warmup_lr
        else:
            self.scheduler.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ---------------------------------------------------------------------------
# Training & Validation Loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, ema_model, loader, optimizer, scaler, device,
                    accum_steps: int = 1, mixup_prob: float = 0.3):
    model.train()
    running_loss = 0.0
    n_batches = 0
    is_cuda = (device.type == "cuda")
    pbar = tqdm(loader, desc="  Training", leave=False, dynamic_ncols=True)

    optimizer.zero_grad(set_to_none=True)

    for step_i, (imgs, steer_targets, speed_targets) in enumerate(pbar):
        imgs          = imgs.to(device, non_blocking=is_cuda)
        steer_targets = steer_targets.to(device, non_blocking=is_cuda)
        speed_targets = speed_targets.to(device, non_blocking=is_cuda)

        # --- Mixup augmentation (batch-level, 30% probability) ---
        if random.random() < mixup_prob:
            imgs, steer_targets, speed_targets = mixup_batch(
                imgs, steer_targets, speed_targets, alpha=0.2)

        with torch.amp.autocast(device_type=device.type, enabled=is_cuda):
            steer_preds, speed_preds = model(imgs)
            # Huber loss: quadratic near zero, linear for outliers
            loss_steer = F.huber_loss(steer_preds, steer_targets, delta=0.5)
            loss_speed = F.huber_loss(speed_preds, speed_targets, delta=5.0)
            loss = loss_steer + 0.1 * loss_speed
            loss = loss / accum_steps  # scale for gradient accumulation

        loss_val = loss.item() * accum_steps  # un-scaled for logging
        if not math.isfinite(loss_val):
            tqdm.write(f"\n[!] Warning: Non-finite loss ({loss_val}) encountered. Skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if is_cuda:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer every `accum_steps` batches
        if (step_i + 1) % accum_steps == 0 or (step_i + 1) == len(loader):
            if is_cuda:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            # Update EMA model after each optimizer step
            if ema_model is not None:
                ema_model.update_parameters(model)

        running_loss += loss_val
        n_batches += 1
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    return running_loss / max(1, n_batches)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    n_batches = 0
    steer_maes = []
    speed_maes = []
    is_cuda = (device.type == "cuda")
    pbar = tqdm(loader, desc="  Validating", leave=False, dynamic_ncols=True)

    for imgs, steer_targets, speed_targets in pbar:
        imgs          = imgs.to(device, non_blocking=is_cuda)
        steer_targets = steer_targets.to(device, non_blocking=is_cuda)
        speed_targets = speed_targets.to(device, non_blocking=is_cuda)

        with torch.amp.autocast(device_type=device.type, enabled=is_cuda):
            steer_preds, speed_preds = model(imgs)
            loss_steer = F.huber_loss(steer_preds, steer_targets, delta=0.5)
            loss_speed = F.huber_loss(speed_preds, speed_targets, delta=5.0)
            loss = loss_steer + 0.1 * loss_speed

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            continue

        running_loss += loss_val
        n_batches += 1
        steer_maes.append((steer_preds - steer_targets).abs().mean().item())
        speed_maes.append((speed_preds - speed_targets).abs().mean().item())
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    avg_loss = running_loss / max(1, n_batches)
    avg_steer_mae = sum(steer_maes) / max(1, len(steer_maes))
    avg_speed_mae = sum(speed_maes) / max(1, len(speed_maes))

    return avg_loss, avg_steer_mae, avg_speed_mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 to predict steering and speed for frame n+1")
    parser.add_argument("--dataset", default="dataset",
                        help="Dataset root directory (default: dataset)")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--bs",      type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--wd",      type=float, default=1e-3,
                        help="Weight decay (default: 1e-3)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="Fraction of data for validation (default: 0.15)")
    parser.add_argument("--shake-prob", type=float, default=0.30,
                        help="Webcam shake augmentation probability (default: 0.30)")
    parser.add_argument("--save-dir", default="models",
                        help="Directory for checkpoints (default: models)")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--accum-steps", type=int, default=2,
                        help="Gradient accumulation steps (default: 2, effective batch = bs × accum)")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Linear LR warm-up epochs (default: 3)")
    parser.add_argument("--mixup-prob", type=float, default=0.3,
                        help="Probability of applying Mixup per batch (default: 0.3)")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate (default: 0.999)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap total samples for quick smoke tests (e.g. --max-samples 100)")
    parser.add_argument("--active-learning", action="store_true",
                        help="Include approved Active Learning frames with priority sampling")
    parser.add_argument("--al-only", action="store_true",
                        help="Train EXCLUSIVELY on Active Learning frames (saves to a separate model file)")
    parser.add_argument("--al-weight", type=float, default=3.0,
                        help="Sampling weight multiplier for Active Learning frames (default: 3.0)")
    args = parser.parse_args()

    if args.al_only:
        args.active_learning = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eff_batch = args.bs * args.accum_steps
    print(f"\n{'='*70}")
    print(f"  EfficientNet-B0 Steering & Speed Predictor (n+1 frame)")
    print(f"  >>> Robust Training Mode <<<")
    print(f"{'='*70}")
    print(f"  Device         : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Dataset        : {args.dataset}/")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Batch size     : {args.bs}  (effective: {eff_batch} with {args.accum_steps}x accum)")
    print(f"  Learning rate  : {args.lr}  (warm-up: {args.warmup_epochs} epochs)")
    print(f"  Loss           : Huber (delta_steer=0.5, delta_speed=5.0)")
    print(f"  EMA decay      : {args.ema_decay}")
    print(f"  Mixup prob     : {args.mixup_prob}")
    print(f"  Top crop       : {TOP_CROP_FRAC*100:.0f}%")
    print()

    root         = Path(args.dataset)
    img_dir      = root / "img"
    diffused_dir = root / "img_diffused"
    csv_path     = root / "telemetry.csv"
    save_dir     = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)

    rows = load_telemetry(csv_path)
    print(f"[data] Loaded {len(rows)} telemetry rows")

    all_indices = build_valid_indices(rows, img_dir, diffused_dir)
    print(f"[data] Valid training samples: {len(all_indices)} "
          f"(after speed filter >{MIN_SPEED_KMH} km/h)")

    if len(all_indices) < 20:
        print("ERROR: Not enough valid samples to train.")
        sys.exit(1)

    split_idx   = int(len(all_indices) * (1.0 - args.val_frac))
    train_idx   = all_indices[:split_idx]
    val_idx     = all_indices[split_idx:]

    if args.max_samples is not None:
        cap_train = max(2, int(args.max_samples * (1.0 - args.val_frac)))
        cap_val   = max(1, args.max_samples - cap_train)
        train_idx = train_idx[:cap_train]
        val_idx   = val_idx[:cap_val]
        print(f"[data] *** max-samples={args.max_samples}: capped to "
              f"train={len(train_idx)}, val={len(val_idx)} ***")

    print(f"[data] Train: {len(train_idx)}  |  Val: {len(val_idx)}  "
          f"(chronological split at index {split_idx})")

    # --- Active Learning: merge approved hard frames ---
    al_rows = []
    al_train_idx = []
    al_img_dir = Path("dataset/active_learning/approved")
    al_csv_path = al_img_dir / "telemetry_al.csv"

    if args.active_learning and al_csv_path.exists():
        al_rows = load_telemetry(al_csv_path)
        if al_rows:
            # Build valid indices for AL data — images are directly in al_img_dir
            # AL frames are single-frame entries (not pairs), so we create
            # synthetic consecutive pairs by pairing each frame with itself.
            # The trainer loads (frame_n, frame_n+1), so adjacent AL rows work
            # as long as consecutive rows exist.
            n_al = len(al_rows)
            for i in range(n_al - 1):
                fname_n = al_rows[i].get("frame")
                fname_n1 = al_rows[i + 1].get("frame")
                if not fname_n or not fname_n1:
                    continue
                if (al_img_dir / fname_n).is_file() and (al_img_dir / fname_n1).is_file():
                    steer_n1 = al_rows[i + 1].get("steering_combined",
                                                   al_rows[i + 1].get("steering"))
                    if steer_n1 is not None:
                        try:
                            float(steer_n1)
                            al_train_idx.append(i)
                        except (ValueError, TypeError):
                            pass
            print(f"[data] Active Learning: loaded {len(al_train_idx)} "
                  f"valid frame pairs from {al_csv_path}")
    elif args.active_learning:
        print(f"[data] Active Learning: no approved data found at {al_csv_path}")
        print(f"       Run active_learning_curator.py first.")

    # --- Weighted sampling to fix steering imbalance ---
    print()
    sample_weights = build_sample_weights(rows, train_idx)

    # Merge AL weights with priority multiplier
    al_sample_weights = []
    if al_train_idx:
        al_base_weights = build_sample_weights(al_rows, al_train_idx)
        al_sample_weights = [w * args.al_weight for w in al_base_weights]
        print(f"[data] Active Learning frames get {args.al_weight}x sampling priority")

    # Combine or isolate
    if args.al_only and al_train_idx:
        combined_weights = al_sample_weights
        combined_count = len(al_train_idx)
        print(f"[data] AL-ONLY Mode: Training on {combined_count} AL frames ONLY")
    else:
        combined_weights = sample_weights + al_sample_weights
        combined_count = len(train_idx) + len(al_train_idx)

    train_sampler = WeightedRandomSampler(
        weights=combined_weights,
        num_samples=combined_count,
        replacement=True,
    )

    train_ds = SteeringDataset(
        train_idx, rows, img_dir, diffused_dir,
        is_train=True, webcam_shake_prob=args.shake_prob)

    # Merge Active Learning dataset if available
    if al_train_idx:
        al_ds = SteeringDataset(
            al_train_idx, al_rows, al_img_dir, al_img_dir,
            is_train=True, webcam_shake_prob=args.shake_prob)
        if args.al_only:
            train_ds = al_ds
        else:
            from torch.utils.data import ConcatDataset
            train_ds = ConcatDataset([train_ds, al_ds])
            print(f"[data] Combined dataset: {len(train_ds)} samples "
                  f"(base: {len(train_idx)}, AL: {len(al_train_idx)})")

    val_ds = SteeringDataset(
        val_idx, rows, img_dir, diffused_dir,
        is_train=False)

    drop_last = combined_count >= args.bs
    _use_persistent = (args.workers > 0)
    train_loader = DataLoader(
        train_ds, batch_size=args.bs, sampler=train_sampler,
        num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=drop_last,
        persistent_workers=_use_persistent,
        prefetch_factor=2 if _use_persistent else None,
        worker_init_fn=worker_init_fn if args.workers > 0 else None)
    val_loader = DataLoader(
        val_ds, batch_size=args.bs, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=_use_persistent,
        prefetch_factor=2 if _use_persistent else None,
        worker_init_fn=worker_init_fn if args.workers > 0 else None)

    print(f"\n[data] Train batches/epoch: {len(train_loader)}  |  "
          f"Val batches: {len(val_loader)}")

    model = SteeringNet().to(device)
    total_p, train_p = count_params(model)
    print(f"\n[model] EfficientNet-B0  -  Total params: {total_p:,}  |  "
          f"Trainable: {train_p:,}  ({train_p/total_p*100:.1f}%)")

    # --- EMA model ---
    ema_avg_fn = get_ema_avg_fn(args.ema_decay)
    ema_model = AveragedModel(model, avg_fn=ema_avg_fn)
    print(f"[model] EMA shadow model created (decay={args.ema_decay})")

    # --- torch.compile ---
    try:
        print(f"[model] Attempting to compile model with torch.compile...")
        model = torch.compile(model)
        print(f"[model] Successfully applied torch.compile!")
    except Exception as e:
        print(f"[model] torch.compile is not supported or failed: {e}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.wd)

    # --- Cosine annealing with warm restarts ---
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scheduler = LinearWarmup(
        optimizer, cosine_scheduler,
        warmup_epochs=args.warmup_epochs, base_lr=args.lr)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 0
    best_val_loss = float("inf")
    best_path = save_dir / "steering_efficientnet_best.pth"
    if args.al_only:
        best_path = save_dir / "steering_efficientnet_al_only.pth"

    # Auto-resume from best checkpoint when doing active learning fine-tuning
    if (args.active_learning or args.al_only) and args.resume is None and (save_dir / "steering_efficientnet_best.pth").exists():
        args.resume = str(save_dir / "steering_efficientnet_best.pth")
        print(f"[ckpt] Active Learning mode: auto-resuming from {args.resume}")

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if "ema_model" in ckpt:
            ema_model.load_state_dict(ckpt["ema_model"])
        print(f"[ckpt] Resumed from epoch {start_epoch}  "
              f"(best val loss: {best_val_loss:.6f})")

    print(f"\n{'-'*78}")
    print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>12}  "
          f"{'Steer MAE':>12}  {'Speed MAE (km/h)':>18}  {'LR':>10}  {'Time':>6}")
    print(f"{'-'*78}")

    target_epochs = args.epochs
    if (args.active_learning or args.al_only) and start_epoch > 0:
        # In AL fine-tuning, treat args.epochs as 'additional epochs to train'
        target_epochs = start_epoch + args.epochs
        print(f"\n[info] Active Learning: fine-tuning for {args.epochs} additional epochs (stopping at {target_epochs})")

    for epoch in range(start_epoch, target_epochs):
        t0 = time.perf_counter()

        train_loss = train_one_epoch(
            model, ema_model, train_loader, optimizer, scaler, device,
            accum_steps=args.accum_steps, mixup_prob=args.mixup_prob)

        # Update BN stats on EMA model before validation
        # NOTE: We skip torch.optim.swa_utils.update_bn() because it
        # re-iterates the entire DataLoader, which spawns a second set of
        # workers and deadlocks on Windows.  Instead we do a quick manual
        # BN update with a small sample from the loader.
        try:
            ema_model.train()
            _bn_batches = 0
            for _bn_imgs, _, _ in train_loader:
                _bn_imgs = _bn_imgs.to(device, non_blocking=True)
                with torch.no_grad(), torch.amp.autocast(device_type=device.type,
                                                         enabled=(device.type == "cuda")):
                    ema_model(_bn_imgs)
                _bn_batches += 1
                if _bn_batches >= 50:   # ~50 batches is enough for BN stats
                    break
            ema_model.eval()
        except Exception:
            pass  # BN update is non-critical

        # Validate using EMA model (better generalization)
        val_loss, steer_mae, speed_mae = validate(
            ema_model, val_loader, device)

        scheduler.step()

        dt = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, best_path)
            # Also save raw EMA weights for easy inference loading
            # The EMA model wraps the base model — extract its module state_dict
            torch.save(ema_model.module.state_dict(), save_dir / "best_val_model.pth")
            improved = " *"

        tqdm.write(f"  {epoch+1:5d}  {train_loss:12.6f}  {val_loss:12.6f}  "
                   f"{steer_mae:12.4f}  {speed_mae:18.2f}  "
                   f"{lr:10.2e}  {dt:5.1f}s{improved}")

    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Best val loss : {best_val_loss:.6f}")
    print(f"  Best model    : {best_path}")
    print(f"  (EMA weights  : {save_dir / 'best_val_model.pth'})")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
