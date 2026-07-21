"""
train_steering_v2.py
====================
V2 training pipeline with:
  - 10-frame temporal sequence input to EfficientNet B1 + LSTM
  - Condition input (-1, 0, 1) concatenated to LSTM hidden state
  - 3 outputs: [steering, scaled_speed, lateral_offset]
  - Diffusion-safe frame pairing (diffused prev -> diffused curr, never mixed)
  - Coherent augmentations applied identically to all frames in a sequence
"""

import os
import csv
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
torch.backends.cudnn.benchmark = True  # Optimize convolutions for fixed input size

def worker_init_fn(worker_id):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RandomFog(object):
    """Custom transform to simulate fog by blending a gray overlay."""
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img_tensor):
        if random.random() < self.p:
            fog_color = torch.tensor([0.7, 0.7, 0.7]).view(3, 1, 1)
            alpha = random.uniform(0.2, 0.6)
            img_tensor = img_tensor * (1 - alpha) + fog_color * alpha
        return img_tensor


# ---------------------------------------------------------------------------
# V2 Dataset -- 2-frame stacking with diffusion-safe pairing
# ---------------------------------------------------------------------------
class SteeringDatasetV2(Dataset):
    """
    Loads consecutive frame pairs and outputs 6-channel tensors.

    Frame pairing rules:
      - For image at position i in the sorted list, the previous frame is i-1.
      - If i == 0 or prev frame is from a different collection sequence, the
        current frame is duplicated as "previous" (cold start).
      - For diffused images: looks for img_diffused/<prev_name> first,
        then falls back to img/<prev_name> (same telemetry, different style).
    """
    def __init__(self, img_dir, telemetry_path, split='train', block_size=200,
                 val_frac=0.1, images=None, frame_to_idx=None, telemetry=None,
                 max_speed=None, domain='all'):
        self.img_dir = img_dir
        self.split = split

        if os.path.basename(self.img_dir) in ('img', 'img_diffused'):
            self.dataset_root = os.path.dirname(self.img_dir)
        else:
            self.dataset_root = self.img_dir

        # -- Load telemetry
        if telemetry is None or frame_to_idx is None:
            print(f"Loading telemetry from {telemetry_path}...")
            telemetry = []
            frame_to_idx = {}
            with open(telemetry_path, 'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    telemetry.append(row)
                    raw_frame = row['frame']
                    frame_to_idx[raw_frame] = i
                    frame_to_idx[os.path.splitext(raw_frame)[0]] = i

        self.telemetry = telemetry
        self.frame_to_idx = frame_to_idx

        # -- Compute max_speed
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

        # -- Build image list
        if images is None:
            raw_img_dir = os.path.join(self.dataset_root, 'img')
            print(f"Scanning raw images in {raw_img_dir}...")
            all_imgs = [img for img in os.listdir(raw_img_dir) if img.endswith('.jpg')]

            # Filter out images without matching telemetry
            valid_imgs = []
            for img in all_imgs:
                stem = os.path.splitext(img)[0]
                if img in self.frame_to_idx or stem in self.frame_to_idx:
                    valid_imgs.append(img)
            print(f"Filtered out {len(all_imgs) - len(valid_imgs)} images without matching telemetry.")
            all_imgs = valid_imgs

            # Sort chronologically via telemetry index
            all_imgs.sort(key=lambda x: self.frame_to_idx.get(
                x, self.frame_to_idx.get(os.path.splitext(x)[0], 0)))

            # Block-based split
            blocks = [all_imgs[i:i + block_size] for i in range(0, len(all_imgs), block_size)]
            block_order = list(range(len(blocks)))
            random.Random(SEED).shuffle(block_order)

            n_val_blocks = max(1, int(len(blocks) * val_frac))
            val_block_ids = set(block_order[:n_val_blocks])

            train_basenames, val_basenames = [], []
            for bi, block in enumerate(blocks):
                (val_basenames if bi in val_block_ids else train_basenames).extend(block)

            # Add diffused images to both train and val splits
            train_imgs = []
            for img in train_basenames:
                train_imgs.append(('img', img))
                diffused_path = os.path.join(self.dataset_root, 'img_diffused', img)
                if os.path.exists(diffused_path):
                    train_imgs.append(('img_diffused', img))

            val_imgs = []
            for img in val_basenames:
                val_imgs.append(('img', img))
                diffused_path = os.path.join(self.dataset_root, 'img_diffused', img)
                if os.path.exists(diffused_path):
                    val_imgs.append(('img_diffused', img))

            self._train_imgs = train_imgs
            self._val_imgs   = val_imgs
        else:
            self._train_imgs, self._val_imgs = images

        self.images = self._train_imgs if split == 'train' else self._val_imgs
        if domain == 'real':
            self.images = [img for img in self.images if img[0] == 'img']
        elif domain == 'diffused':
            self.images = [img for img in self.images if img[0] == 'img_diffused']

        # Build a fast basename -> sorted-real-index lookup for prev-frame resolution
        # Only real (non-diffused) frames form the temporal sequence
        real_sorted = [img for (folder, img) in
                       (self._train_imgs + self._val_imgs) if folder == 'img']
        # Deduplicate while preserving order
        seen = set()
        real_sorted_unique = []
        for n in real_sorted:
            if n not in seen:
                seen.add(n)
                real_sorted_unique.append(n)
        self._real_order = {name: i for i, name in enumerate(real_sorted_unique)}
        self._real_sorted = real_sorted_unique

        print(f"Found {len(self.images)} images for {split} split (includes diffused).")

        # -- Augmentation handles
        if split == 'train':
            self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            self.blur         = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            self.eraser       = T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
            self.fog          = RandomFog(p=0.2)
        else:
            self.color_jitter = None
            self.blur         = None
            self.eraser       = None
            self.fog          = None

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_split_lists(self):
        return self._train_imgs, self._val_imgs

    def __len__(self):
        return len(self.images)

    def _resolve_telemetry_idx(self, basename):
        if basename in self.frame_to_idx:
            return self.frame_to_idx[basename]
        return self.frame_to_idx.get(os.path.splitext(basename)[0], -1)

    def _load_frame(self, folder, basename):
        """Load, crop and resize one frame. Returns PIL Image or None."""
        path = os.path.join(self.dataset_root, folder, basename)
        img_np = cv2.imread(path)
        if img_np is None:
            return None
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_np)
        # Same crop as V1 -- no stretching, road region only
        img = TF.crop(img, top=231, left=0, height=264, width=1280)
        img = TF.resize(img, (240, 240))
        return img

    def _get_sequence_frames(self, folder, basename, seq_len=10):
        """
        Returns a list of `seq_len` PIL Images ending with the current frame.
        For diffused images: prefers diffused previous, falls back to real previous.
        If a previous frame doesn't exist (start of dataset), it pads by duplicating the oldest found frame.
        """
        pos = self._real_order.get(basename, -1)
        frames = []
        
        for p in range(pos - seq_len + 1, pos + 1):
            if p < 0:
                frames.append(None)
                continue
                
            curr_basename = self._real_sorted[p]
            
            img = None
            if folder == 'img':
                img = self._load_frame('img', curr_basename)
            elif folder == 'img_diffused':
                diffused_path = os.path.join(self.dataset_root, 'img_diffused', curr_basename)
                if os.path.exists(diffused_path):
                    img = self._load_frame('img_diffused', curr_basename)
                
            frames.append(img)
            
        first_valid = next((f for f in frames if f is not None), None)
        if first_valid is None:
            return None
            
        for i in range(seq_len):
            if frames[i] is None:
                frames[i] = first_valid.copy()
                
        return frames

    def _to_tensor_aug(self, img, rng, is_train):
        """Convert PIL image to tensor, applying augmentations driven by rng."""
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
                noise = torch.randn_like(t) * 0.05
                t = torch.clamp(t + noise, 0.0, 1.0)
        t = self.normalize(t)
        return t

    def __getitem__(self, idx):
        folder, basename = self.images[idx]

        frames = self._get_sequence_frames(folder, basename, seq_len=10)
        if frames is None or frames[-1] is None:
            return self.__getitem__((idx + 1) % len(self.images))

        is_train = (self.split == 'train')
        aug_seed = random.randint(0, 2**31)

        tensors = []
        for img in frames:
            rng = random.Random(aug_seed)
            tensors.append(self._to_tensor_aug(img, rng, is_train))

        stacked = torch.stack(tensors, dim=0)

        # -- Labels
        telemetry_idx = self._resolve_telemetry_idx(basename)
        if telemetry_idx == -1:
            raise RuntimeError(f"Image {basename} has no telemetry entry.")

        row = self.telemetry[telemetry_idx]
        try:
            steering = float(row.get('steering_combined', row.get('steering', 0.0)) or 0.0)
        except (ValueError, TypeError):
            steering = 0.0

        try:
            velX = float(row.get('velX', 0.0) or 0.0)
            velY = float(row.get('velY', 0.0) or 0.0)
        except (ValueError, TypeError):
            velX, velY = 0.0, 0.0
        try:
            offset = float(row.get('steering_offset', 0.0) or 0.0)
        except (ValueError, TypeError):
            offset = 0.0

        try:
            condition = float(row.get('condition', 0.0) or 0.0)
        except (ValueError, TypeError):
            condition = 0.0

        speed        = math.sqrt(velX**2 + velY**2)
        scaled_speed = speed / self.max_speed

        condition_tensor = torch.tensor([condition], dtype=torch.float32)
        labels_tensor = torch.tensor([steering, scaled_speed, offset], dtype=torch.float32)
        return stacked, condition_tensor, labels_tensor


# ---------------------------------------------------------------------------
# V2 Model -- EfficientNet B1 + LSTM with condition input, 4 outputs
# ---------------------------------------------------------------------------
class SteeringModelV2(nn.Module):
    """
    EfficientNet B1 modified for 10-frame sequence input using an LSTM.
    Takes condition (-1, 0, 1) as additional input.
    3 outputs: [steering, scaled_speed, lateral_offset].
    """
    def __init__(self):
        super(SteeringModelV2, self).__init__()

        weights = EfficientNet_B1_Weights.DEFAULT
        base = efficientnet_b1(weights=weights)
        
        # Backbone: everything up to the adaptive avg pool
        self.backbone = nn.Sequential(
            base.features,
            base.avgpool,
            nn.Flatten()  # (B, 1280)
        )

        # -- Freeze early stages 1-4
        for i in range(1, 5):
            for param in self.backbone[0][i].parameters():
                param.requires_grad = False

        # LSTM takes the 1280-d feature sequence
        self.lstm = nn.LSTM(input_size=1280, hidden_size=256, num_layers=1, batch_first=True)

        # -- Output head: 3 regression values
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 + 1, 128),  # +1 for condition
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 3)   # [steering, scaled_speed, lateral_offset]
        )

    def forward(self, x, condition):
        # x shape: (B, 10, 3, H, W), condition shape: (B, 1)
        B, seq_len, C, H, W = x.size()
        
        # Merge batch and sequence dims for backbone
        x_flat = x.view(B * seq_len, C, H, W)
        
        features = self.backbone(x_flat)  # (B*seq_len, 1280)
        
        # Unflatten
        features = features.view(B, seq_len, -1)  # (B, 5, 1280)
        
        # LSTM
        lstm_out, (hn, cn) = self.lstm(features)
        
        # Use the last hidden state representing the end of the sequence
        # hn shape is (num_layers, B, hidden_size) -> (1, B, 256)
        last_hidden = hn[-1]  # (B, 256)
        
        # Concatenate condition
        last_hidden = torch.cat((last_hidden, condition), dim=1) # (B, 257)
        
        return self.head(last_hidden)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    wandb.init(project="beamng-ai", name="steering-v2-2frame-3output")
    workspace_dir = os.path.dirname(os.path.abspath(__file__))

    img_dir = os.path.join(workspace_dir, 'dataset', 'data')
    if not os.path.exists(img_dir):
        img_dir = r'D:\_Ethos\dataset\data'

    telemetry_path = os.path.join(img_dir, 'telemetry.csv')
    print(f"Using telemetry file: {telemetry_path}")

    batch_size   = 24   
    epochs       = 25
    lr           = 3e-4
    weight_decay = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_train_dataset = SteeringDatasetV2(img_dir, telemetry_path, split='train', domain='all')
    split_lists = base_train_dataset.get_split_lists()
    
    real_train_dataset = SteeringDatasetV2(
        img_dir, telemetry_path, split='train', domain='real',
        images=split_lists, frame_to_idx=base_train_dataset.frame_to_idx,
        telemetry=base_train_dataset.telemetry, max_speed=base_train_dataset.max_speed
    )
    diffused_train_dataset = SteeringDatasetV2(
        img_dir, telemetry_path, split='train', domain='diffused',
        images=split_lists, frame_to_idx=base_train_dataset.frame_to_idx,
        telemetry=base_train_dataset.telemetry, max_speed=base_train_dataset.max_speed
    )
    val_dataset = SteeringDatasetV2(
        img_dir, telemetry_path, split='val', domain='all',
        images=split_lists, frame_to_idx=base_train_dataset.frame_to_idx,
        telemetry=base_train_dataset.telemetry, max_speed=base_train_dataset.max_speed
    )

    real_loader = DataLoader(
        real_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn
    )
    diffused_loader = DataLoader(
        diffused_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn
    )

    model = SteeringModelV2().to(device)
    total_params   = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total | {trainable_params:,} trainable")

    # -- Loss: 4-way weighted Huber
    steer_weight  = 0.55
    speed_weight  = 0.15
    offset_weight = 0.15
    _huber = nn.SmoothL1Loss(reduction='mean')

    def criterion(outputs, labels):
        s_l = _huber(outputs[:, 0], labels[:, 0])
        v_l = _huber(outputs[:, 1], labels[:, 1])
        o_l = _huber(outputs[:, 2], labels[:, 2])
        
        total_loss = steer_weight * s_l + speed_weight * v_l + offset_weight * o_l
        return total_loss, s_l, v_l, o_l

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')  # AMP gradient scaler

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # -- Train
        model.train()
        t_loss = t_sl = t_vl = t_ol = 0.0

        real_iter = iter(real_loader)
        diffused_iter = iter(diffused_loader)
        num_batches = len(real_loader) + len(diffused_loader)
        pbar = tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        active_iters = []
        if len(real_loader) > 0: active_iters.append((real_iter, "real"))
        if len(diffused_loader) > 0: active_iters.append((diffused_iter, "diffused"))
        
        while active_iters:
            import random
            idx = random.randint(0, len(active_iters) - 1)
            iterator, domain_name = active_iters[idx]
            
            try:
                imgs, conditions, labels = next(iterator)
            except StopIteration:
                active_iters.pop(idx)
                continue
                
            imgs       = imgs.to(device, non_blocking=True)
            conditions = conditions.to(device, non_blocking=True)
            labels     = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(imgs, conditions)
                loss, s_l, v_l, o_l = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            n      = imgs.size(0)
            t_loss += loss.item() * n
            t_sl   += s_l.item()  * n
            t_vl   += v_l.item()  * n
            t_ol   += o_l.item()  * n
            pbar.set_postfix({'loss': f"{loss.item():.3f}", 'st': f"{s_l.item():.3f}", 'dom': domain_name})
            pbar.update(1)
            
        pbar.close()

        N = len(real_train_dataset) + len(diffused_train_dataset)
        t_loss /= N; t_sl /= N; t_vl /= N; t_ol /= N

        # -- Validation
        model.eval()
        v_loss = v_sl = v_vl = v_ol = 0.0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for imgs, conditions, labels in pbar_val:
                imgs       = imgs.to(device, non_blocking=True)
                conditions = conditions.to(device, non_blocking=True)
                labels     = labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs    = model(imgs, conditions)
                    loss, s_l, v_l, o_l = criterion(outputs, labels)
                n      = imgs.size(0)
                v_loss += loss.item() * n
                v_sl   += s_l.item()  * n
                v_vl   += v_l.item()  * n
                v_ol   += o_l.item()  * n

        Nv     = len(val_dataset)
        v_loss /= Nv; v_sl /= Nv; v_vl /= Nv; v_ol /= Nv

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train={t_loss:.4f} (st={t_sl:.4f} sp={t_vl:.4f} of={t_ol:.4f}) | "
              f"Val={v_loss:.4f} (st={v_sl:.4f} sp={v_vl:.4f} of={v_ol:.4f}) | "
              f"LR={current_lr:.2e}")

        wandb.log({
            "Train Loss": t_loss, "Train Steer Loss": t_sl,
            "Train Speed Loss": t_vl, "Train Offset Loss": t_ol,
            "Val Loss": v_loss, "Val Steer Loss": v_sl,
            "Val Speed Loss": v_vl, "Val Offset Loss": v_ol,
            "Learning Rate": current_lr, "epoch": epoch + 1
        })

        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), 'best_steering_v2_model.pth')
            meta = {
                "max_speed": base_train_dataset.max_speed,
                "version": "2_lstm",
                "seq_len": 10,
                "outputs": ["steering", "scaled_speed", "lateral_offset"],
                "inputs": ["images", "condition"]
            }
            with open('best_steering_v2_model_meta.json', 'w') as mf:
                json.dump(meta, mf, indent=2)
            print(f"--> Saved best V2 model  Val Loss={v_loss:.4f}  max_speed={base_train_dataset.max_speed:.4f}")

    wandb.finish()


if __name__ == '__main__':
    main()
