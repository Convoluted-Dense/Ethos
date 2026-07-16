"""
train_steering_v3.py
====================
V3 training pipeline with:
  - Dual-Stream Two-Tower EfficientNet B1 architecture
    * Stream A: previous frame  (3ch) -> 1280-d feature vector
    * Stream B: current  frame  (3ch) -> 1280-d feature vector
    * Fusion  : cat([feat_curr, feat_curr - feat_prev]) -> 2560-d -> Head
  - 3 regression outputs: [steering, scaled_speed, lateral_offset]
  - STRICT domain pairing:
      real    prev  <-> real    curr   (true temporal motion signal)
      diffused prev <-> diffused curr  (same-environment augmentation)
      NEVER mix real and diffused in a pair
  - If no same-domain previous frame exists -> self-duplicate (cold start)
  - Coherent augmentations applied identically to both frames in a pair
"""

import os
import csv
import copy
import json
import math
import random
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
import wandb

# ---------------------------------------------------------------------------
# Repro: seed everything
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def worker_init_fn(worker_id):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RandomFog(object):
    """Simulate fog by blending a gray overlay over a tensor image."""
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img_tensor):
        if random.random() < self.p:
            fog_color = torch.tensor([0.7, 0.7, 0.7]).view(3, 1, 1)
            alpha = random.uniform(0.2, 0.6)
            img_tensor = img_tensor * (1 - alpha) + fog_color * alpha
        return img_tensor


# ---------------------------------------------------------------------------
# V3 Dataset -- strict same-domain frame pairing
# ---------------------------------------------------------------------------
class SteeringDatasetV3(Dataset):
    """
    Returns (prev_tensor, curr_tensor, labels) where:
      - prev_tensor, curr_tensor are each (3, H, W) normalized tensors
      - Domain pairing is strict: real<->real, diffused<->diffused ONLY
      - If no same-domain previous frame exists, curr is duplicated as prev
    """

    def __init__(self, img_dir, telemetry_path, split='train', block_size=200,
                 val_frac=0.1, images=None, frame_to_idx=None, telemetry=None,
                 max_speed=None):
        self.img_dir = img_dir
        self.split   = split

        # Dataset root detection (same logic as V2)
        if os.path.basename(self.img_dir) in ('img', 'img_diffused'):
            self.dataset_root = os.path.dirname(self.img_dir)
        else:
            self.dataset_root = self.img_dir

        # ── Load telemetry ───────────────────────────────────────────────────
        if telemetry is None or frame_to_idx is None:
            print(f"Loading telemetry from {telemetry_path}...")
            telemetry    = []
            frame_to_idx = {}
            with open(telemetry_path, 'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    telemetry.append(row)
                    raw = row['frame']
                    frame_to_idx[raw]                   = i
                    frame_to_idx[os.path.splitext(raw)[0]] = i

        self.telemetry    = telemetry
        self.frame_to_idx = frame_to_idx

        # ── Max speed ────────────────────────────────────────────────────────
        if max_speed is None:
            speeds = []
            for row in self.telemetry:
                try:
                    vx, vy = float(row['velX']), float(row['velY'])
                    speeds.append(math.sqrt(vx**2 + vy**2))
                except Exception:
                    pass
            self.max_speed = max(speeds) if speeds else 30.0
            if self.max_speed < 1.0:
                self.max_speed = 1.0
            print(f"Computed max speed for scaling: {self.max_speed:.2f} m/s")
        else:
            self.max_speed = max_speed

        # ── Build image lists ────────────────────────────────────────────────
        if images is None:
            raw_img_dir = os.path.join(self.dataset_root, 'img')
            print(f"Scanning raw images in {raw_img_dir}...")
            all_imgs = [img for img in os.listdir(raw_img_dir) if img.endswith('.jpg')]

            # Filter to telemetry-matched images only
            valid_imgs = [img for img in all_imgs
                          if img in self.frame_to_idx
                          or os.path.splitext(img)[0] in self.frame_to_idx]
            print(f"Filtered {len(all_imgs) - len(valid_imgs)} images without telemetry.")
            all_imgs = valid_imgs

            # Sort by telemetry index (chronological order)
            all_imgs.sort(key=lambda x: self.frame_to_idx.get(
                x, self.frame_to_idx.get(os.path.splitext(x)[0], 0)))

            # Block-based train/val split (keeps temporal structure)
            blocks      = [all_imgs[i:i + block_size] for i in range(0, len(all_imgs), block_size)]
            block_order = list(range(len(blocks)))
            random.Random(SEED).shuffle(block_order)
            n_val_blocks  = max(1, int(len(blocks) * val_frac))
            val_block_ids = set(block_order[:n_val_blocks])

            train_basenames, val_basenames = [], []
            for bi, block in enumerate(blocks):
                (val_basenames if bi in val_block_ids else train_basenames).extend(block)

            # Diffused images only go into train
            train_imgs = []
            for img in train_basenames:
                train_imgs.append(('img', img))
                diffused_path = os.path.join(self.dataset_root, 'img_diffused', img)
                if os.path.exists(diffused_path):
                    train_imgs.append(('img_diffused', img))

            val_imgs = [('img', img) for img in val_basenames]

            self._train_imgs = train_imgs
            self._val_imgs   = val_imgs
        else:
            self._train_imgs, self._val_imgs = images

        self.images = self._train_imgs if split == 'train' else self._val_imgs

        # ── Build chronological real-frame order lookup ───────────────────────
        # Only real frames form the temporal timeline
        seen = set()
        real_sorted_unique = []
        for (folder, img) in (self._train_imgs + self._val_imgs):
            if folder == 'img' and img not in seen:
                seen.add(img)
                real_sorted_unique.append(img)
        self._real_sorted = real_sorted_unique
        self._real_order  = {name: i for i, name in enumerate(real_sorted_unique)}

        # ── Build diffused-frame lookup (fast existence check) ────────────────
        diffused_dir = os.path.join(self.dataset_root, 'img_diffused')
        self._diffused_set = set()
        if os.path.isdir(diffused_dir):
            for f in os.listdir(diffused_dir):
                if f.endswith('.jpg'):
                    self._diffused_set.add(f)

        print(f"[{split}] {len(self.images)} samples | "
              f"{sum(1 for f,_ in self.images if f=='img_diffused')} diffused")

        # ── Augmentation handles ──────────────────────────────────────────────
        if split == 'train':
            self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4,
                                              saturation=0.4, hue=0.1)
            self.blur   = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            self.eraser = T.RandomErasing(p=0.3, scale=(0.02, 0.2),
                                          ratio=(0.3, 3.3), value=0)
            self.fog    = RandomFog(p=0.2)
        else:
            self.color_jitter = self.blur = self.eraser = self.fog = None

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # ── Public helpers ────────────────────────────────────────────────────────
    def get_split_lists(self):
        return self._train_imgs, self._val_imgs

    def __len__(self):
        return len(self.images)

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _resolve_telemetry_idx(self, basename):
        if basename in self.frame_to_idx:
            return self.frame_to_idx[basename]
        return self.frame_to_idx.get(os.path.splitext(basename)[0], -1)

    def _load_frame(self, folder, basename):
        """Load, crop, resize one frame -> PIL Image or None."""
        path = os.path.join(self.dataset_root, folder, basename)
        img_np = cv2.imread(path)
        if img_np is None:
            return None
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img    = Image.fromarray(img_np)
        img    = TF.crop(img, top=231, left=0, height=264, width=1280)
        img    = TF.resize(img, (240, 240))
        return img

    def _get_prev_frame(self, folder, basename):
        """
        Strict domain pairing:
          folder == 'img'          -> look for real previous frame only
          folder == 'img_diffused' -> look for diffused previous frame only
        Returns None on cold start (caller will self-duplicate curr).
        NEVER crosses domain boundaries.
        """
        pos = self._real_order.get(basename, -1)
        if pos <= 0:
            return None  # First frame in dataset -> cold start

        prev_basename = self._real_sorted[pos - 1]

        if folder == 'img':
            # Real domain: use real previous frame
            return self._load_frame('img', prev_basename)

        else:  # 'img_diffused'
            # Diffused domain: ONLY use diffused previous frame
            if prev_basename in self._diffused_set:
                img = self._load_frame('img_diffused', prev_basename)
                if img is not None:
                    return img
            # Diffused previous doesn't exist -> self-duplicate (cold start)
            return None

    def _to_tensor_aug(self, img, rng, is_train):
        """Convert PIL image to tensor, applying augmentations from rng."""
        if is_train:
            if rng.random() > 0.5:
                img = self.color_jitter(img)
            if rng.random() > 0.5:
                img = self.blur(img)
        t = TF.to_tensor(img)
        if is_train:
            if rng.random() > 0.5:
                t = self.eraser(t)
            t = self.fog(t)
            if rng.random() > 0.5:
                t = torch.clamp(t + torch.randn_like(t) * 0.05, 0.0, 1.0)
        return self.normalize(t)

    def __getitem__(self, idx):
        folder, basename = self.images[idx]

        # Load current frame
        curr_img = self._load_frame(folder, basename)
        if curr_img is None:
            return self.__getitem__((idx + 1) % len(self.images))

        # Load previous frame (strict same-domain or self-duplicate)
        prev_img = self._get_prev_frame(folder, basename)
        if prev_img is None:
            prev_img = curr_img.copy()  # Cold start / no same-domain prev

        # Coherent augmentation: same RNG seed for both frames
        is_train = (self.split == 'train')
        aug_seed = random.randint(0, 2**31)

        rng = random.Random(aug_seed)
        curr_t = self._to_tensor_aug(curr_img, rng, is_train)
        rng = random.Random(aug_seed)   # Reset -> identical augmentation
        prev_t = self._to_tensor_aug(prev_img, rng, is_train)

        # Labels
        tel_idx = self._resolve_telemetry_idx(basename)
        if tel_idx == -1:
            raise RuntimeError(f"No telemetry for {basename}")

        row      = self.telemetry[tel_idx]
        steering = float(row.get('steering_combined', row['steering']))
        vx, vy   = float(row['velX']), float(row['velY'])
        try:
            offset = float(row.get('steering_offset', 0.0) or 0.0)
        except (ValueError, TypeError):
            offset = 0.0

        speed        = math.sqrt(vx**2 + vy**2)
        scaled_speed = speed / self.max_speed

        labels = torch.tensor([steering, scaled_speed, offset], dtype=torch.float32)
        return prev_t, curr_t, labels


# ---------------------------------------------------------------------------
# V3 Model -- Dual-Stream EfficientNet B1, shared backbone
#
# Forward: feat_curr = Backbone(curr)
#          feat_prev = Backbone(prev)
#          fused     = cat([feat_curr, feat_curr - feat_prev])   # 2560-d
#          out       = Head(fused)                               # 3 values
#
# Using feat_curr - feat_prev as the explicit temporal delta means the
# head sees (WHERE we are + HOW MUCH we moved), not two raw feature vectors
# the head must implicitly diff itself.
# ---------------------------------------------------------------------------
class SteeringModelV3(nn.Module):
    """
    Dual-stream EfficientNet B1 with shared backbone weights.
    Fuses current-frame features with their temporal delta.
    """

    FEAT_DIM = 1280  # EfficientNet B1 global-pooled feature dim

    def __init__(self):
        super().__init__()

        # ── Backbone (shared) ─────────────────────────────────────────────────
        base    = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        # Keep everything up to (and including) the adaptive avg pool,
        # but discard the dropout + linear classifier.
        self.backbone = nn.Sequential(
            base.features,       # all conv stages
            base.avgpool,        # (B, 1280, 1, 1)
            nn.Flatten()         # (B, 1280)
        )

        # ── Freeze early backbone stages (1-4) ───────────────────────────────
        # Stage 0 (first conv) stays trainable; stages 1-4 are frozen.
        for i in range(1, 5):
            for param in base.features[i].parameters():
                param.requires_grad = False

        # ── Fusion head: 2560 -> 3 ────────────────────────────────────────────
        # Input: cat([feat_curr (1280), feat_curr - feat_prev (1280)])
        self.head = nn.Sequential(
            nn.Linear(self.FEAT_DIM * 2, 512),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 3)   # [steering, scaled_speed, lateral_offset]
        )

    def forward(self, prev, curr):
        """
        prev : (B, 3, H, W) -- previous frame
        curr : (B, 3, H, W) -- current  frame
        """
        feat_curr = self.backbone(curr)              # (B, 1280)
        feat_prev = self.backbone(prev)              # (B, 1280) same weights
        delta     = feat_curr - feat_prev            # explicit temporal motion
        fused     = torch.cat([feat_curr, delta], dim=1)  # (B, 2560)
        return self.head(fused)                      # (B, 3)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    wandb.init(project="beamng-ai", name="steering-v3-dual-stream")
    workspace_dir = os.path.dirname(os.path.abspath(__file__))

    img_dir = os.path.join(workspace_dir, 'dataset')
    if not os.path.exists(img_dir):
        img_dir = 'D:/dataset'

    telemetry_path = os.path.join(workspace_dir, 'dataset', 'telemetry.csv')
    if not os.path.exists(telemetry_path):
        telemetry_path = 'D:/dataset/telemetry.csv'
        print(f"Local telemetry not found. Falling back to {telemetry_path}")
    else:
        print(f"Using telemetry: {telemetry_path}")

    batch_size   = 48
    epochs       = 25
    lr           = 1e-3
    weight_decay = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build datasets
    train_dataset = SteeringDatasetV3(img_dir, telemetry_path, split='train')
    split_lists   = train_dataset.get_split_lists()
    val_dataset   = SteeringDatasetV3(
        img_dir, telemetry_path, split='val',
        images=split_lists,
        frame_to_idx=train_dataset.frame_to_idx,
        telemetry=train_dataset.telemetry,
        max_speed=train_dataset.max_speed,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )

    model = SteeringModelV3().to(device)
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} total | {trainable_p:,} trainable")

    # ── Loss: weighted Huber across 3 outputs ─────────────────────────────────
    W_STEER  = 0.70
    W_SPEED  = 0.15
    W_OFFSET = 0.15
    _huber   = nn.SmoothL1Loss(reduction='mean')

    def criterion(out, lbl):
        s = _huber(out[:, 0], lbl[:, 0])
        v = _huber(out[:, 1], lbl[:, 1])
        o = _huber(out[:, 2], lbl[:, 2])
        return W_STEER * s + W_SPEED * v + W_OFFSET * o, s, v, o

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    best_val = float('inf')

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tl = ts = tv = to = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for prev, curr, labels in pbar:
            prev   = prev.to(device)
            curr   = curr.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out              = model(prev, curr)
            loss, s, v, o    = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            n   = prev.size(0)
            tl += loss.item() * n
            ts += s.item()    * n
            tv += v.item()    * n
            to += o.item()    * n
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'st':   f'{s.item():.4f}',
                              'of':   f'{o.item():.4f}'})

        N  = len(train_dataset)
        tl /= N; ts /= N; tv /= N; to /= N

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        vl = vs = vv = vo = 0.0
        with torch.no_grad():
            for prev, curr, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                prev   = prev.to(device)
                curr   = curr.to(device)
                labels = labels.to(device)
                out              = model(prev, curr)
                loss, s, v, o    = criterion(out, labels)
                n   = prev.size(0)
                vl += loss.item() * n
                vs += s.item()    * n
                vv += v.item()    * n
                vo += o.item()    * n

        Nv = len(val_dataset)
        vl /= Nv; vs /= Nv; vv /= Nv; vo /= Nv

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train={tl:.4f} (st={ts:.4f} sp={tv:.4f} of={to:.4f}) | "
              f"Val={vl:.4f}   (st={vs:.4f} sp={vv:.4f} of={vo:.4f}) | "
              f"LR={cur_lr:.2e}")

        wandb.log({
            "Train Loss": tl, "Train Steer": ts, "Train Speed": tv, "Train Offset": to,
            "Val Loss":   vl, "Val Steer":   vs, "Val Speed":   vv, "Val Offset":   vo,
            "LR": cur_lr, "epoch": epoch + 1,
        })

        scheduler.step(vl)

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), 'best_steering_v3_model.pth')
            meta = {
                "max_speed": train_dataset.max_speed,
                "version":   3,
                "outputs":   ["steering", "scaled_speed", "lateral_offset"],
                "arch":      "dual_stream_efficientnet_b1",
            }
            with open('best_steering_v3_model_meta.json', 'w') as mf:
                json.dump(meta, mf, indent=2)
            print(f"--> Saved best V3 model | Val={vl:.4f} | max_speed={train_dataset.max_speed:.4f}")

    wandb.finish()


if __name__ == '__main__':
    main()
