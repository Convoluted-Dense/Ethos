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
            fog_color = torch.tensor([0.7, 0.7, 0.7]).view(3, 1, 1).to(img_tensor.device)
            alpha = random.uniform(0.2, 0.6)
            img_tensor = img_tensor * (1 - alpha) + fog_color * alpha
        return img_tensor

class SteeringVelocityDataset(Dataset):
    def __init__(self, img_dir, telemetry_path, split='train', block_size=200, val_frac=0.1, 
                 images=None, frame_to_idx=None, telemetry=None, max_speed=None):
        self.img_dir = img_dir
        self.split = split
        
        # Ensure we have the parent dataset directory
        if os.path.basename(self.img_dir) in ('img', 'img_diffused'):
            self.dataset_root = os.path.dirname(self.img_dir)
        else:
            self.dataset_root = self.img_dir
        
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
        
        if max_speed is None:
            # Find max speed dynamically from telemetry
            speeds = []
            for row in self.telemetry:
                try:
                    vx, vy = float(row['velX']), float(row['velY'])
                    speeds.append(math.sqrt(vx**2 + vy**2))
                except:
                    pass
            self.max_speed = max(speeds) if speeds else 30.0
            if self.max_speed < 1.0: self.max_speed = 1.0
            print(f"Computed max speed for scaling: {self.max_speed:.2f} m/s")
        else:
            self.max_speed = max_speed
        
        if images is None:
            raw_img_dir = os.path.join(self.dataset_root, 'img')
            print(f"Scanning raw images in {raw_img_dir}...")
            all_imgs = os.listdir(raw_img_dir)
            all_imgs = [img for img in all_imgs if img.endswith('.jpg')]
            
            # Filter out images without telemetry BEFORE sorting/splitting to avoid silent zero-labels
            valid_imgs = []
            for img in all_imgs:
                stem = os.path.splitext(img)[0]
                if img in self.frame_to_idx or stem in self.frame_to_idx:
                    valid_imgs.append(img)
            
            print(f"Filtered out {len(all_imgs) - len(valid_imgs)} images without matching telemetry.")
            all_imgs = valid_imgs
            
            # Sort chronologically using telemetry index instead of lexicographic string sort
            all_imgs.sort(key=lambda x: self.frame_to_idx.get(x, self.frame_to_idx.get(os.path.splitext(x)[0])))
            
            # Block-based split
            blocks = [all_imgs[i:i + block_size] for i in range(0, len(all_imgs), block_size)]
            block_order = list(range(len(blocks)))
            random.Random(SEED).shuffle(block_order)
            
            n_val_blocks = max(1, int(len(blocks) * val_frac))
            val_block_ids = set(block_order[:n_val_blocks])
            
            train_basenames, val_basenames = [], []
            for bi, block in enumerate(blocks):
                (val_basenames if bi in val_block_ids else train_basenames).extend(block)
                
            # Build list of image paths relative to dataset_root.
            # NOTE: diffused images are ONLY added to train, never to val.
            # Adding them to val would leak near-duplicates of training images into
            # the validation set and make val_loss an optimistic lie.
            train_imgs = []
            for img in train_basenames:
                train_imgs.append(os.path.join('img', img))
                diffused_img_path = os.path.join(self.dataset_root, 'img_diffused', img)
                if os.path.exists(diffused_img_path):
                    train_imgs.append(os.path.join('img_diffused', img))
                    
            val_imgs = []
            for img in val_basenames:
                # Real images only for validation — no diffused copies.
                val_imgs.append(os.path.join('img', img))
                    
            self._train_imgs = train_imgs
            self._val_imgs = val_imgs
        else:
            self._train_imgs, self._val_imgs = images
            
        self.images = self._train_imgs if split == 'train' else self._val_imgs
        print(f"Found {len(self.images)} images for {split} split.")
                
        # Augmentations
        if split == 'train':
            self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            self.blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            self.eraser = T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
            self.fog = RandomFog(p=0.2)
        else:
            self.color_jitter = None
            self.blur = None
            self.eraser = None
            self.fog = None
            
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_split_lists(self):
        return self._train_imgs, self._val_imgs

    def __len__(self):
        return len(self.images)

    def _resolve_telemetry_idx(self, img_name):
        base_name = os.path.basename(img_name)
        if base_name in self.frame_to_idx:
            return self.frame_to_idx[base_name]
        return self.frame_to_idx.get(os.path.splitext(base_name)[0], -1)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.dataset_root, img_name)
        
        try:
            # FIX (Bug 15): cv2.imwrite saves BGR JPEGs. PIL.Image.open would read the
            # raw bytes and return them as RGB, silently swapping R and B channels.
            # At inference, test_cnn.py explicitly converts BGR→RGB, so training must
            # match that colour ordering.  Use cv2 to load and convert properly.
            img_np = cv2.imread(img_path)          # reads as BGR
            if img_np is None:
                raise IOError(f"cv2 could not read {img_path}")
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # → correct RGB
            img = Image.fromarray(img_np)
        except Exception:
            # Fallback for corrupt JPEGs: recursively load an adjacent frame
            return self.__getitem__((idx + 1) % len(self.images))
        
        # Crop using user-selected region.
        # FIX (Bug 5): images are saved at 1280 px wide; use 1280 not 1279.
        img = TF.crop(img, top=231, left=0, height=264, width=1280)
        
        # Resize for EfficientNet B1
        img = TF.resize(img, (240, 240))
        
        is_flipped = False
        if self.split == 'train':
            if random.random() > 0.5:
                img = self.color_jitter(img)
            if random.random() > 0.5:
                img = self.blur(img)
            if random.random() > 0.5:
                img = TF.hflip(img)
                is_flipped = True
                
        img_tensor = TF.to_tensor(img)
        
        if self.split == 'train' and self.eraser is not None:
            img_tensor = self.eraser(img_tensor)
        
        if self.split == 'train' and self.fog is not None:
            img_tensor = self.fog(img_tensor)
            
        # Add random noise
        if self.split == 'train' and random.random() > 0.5:
            noise = torch.randn_like(img_tensor) * 0.05
            img_tensor = img_tensor + noise
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            
        img_tensor = self.normalize(img_tensor)
        
        telemetry_idx = self._resolve_telemetry_idx(img_name)
        if telemetry_idx == -1:
            raise RuntimeError(f"Filtered image {img_name} still has no telemetry. This should be impossible.")
            
        row = self.telemetry[telemetry_idx]
        # Use combined steering if present (new format), else raw steering (old format)
        steering = float(row.get('steering_combined', row['steering']))
        velX = float(row['velX'])
        velY = float(row['velY'])
        
        # Flip steering if image was horizontally flipped
        if is_flipped:
            steering = -steering
            
        # Speed scaling based on dynamically calculated max speed
        speed = math.sqrt(velX**2 + velY**2)
        scaled_speed = speed / self.max_speed
        
        labels_tensor = torch.tensor([steering, scaled_speed], dtype=torch.float32)
        return img_tensor, labels_tensor

# ---------------------------------------------------------------------------
# Model Architecture (EfficientNet B1)
# ---------------------------------------------------------------------------
class SteeringModel(nn.Module):
    def __init__(self):
        super(SteeringModel, self).__init__()
        
        weights = EfficientNet_B1_Weights.DEFAULT
        self.model = efficientnet_b1(weights=weights)
        
        # Freeze early feature blocks (stem and stages 1-4)
        for i in range(5):
            for param in self.model.features[i].parameters():
                param.requires_grad = False
                
        # Replace the final classification head
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 2) # Output: [steering, scaled_speed]
        )
        
    def forward(self, x):
        return self.model(x)

def main():
    wandb.init(project="beamng-ai", name="steering-velocity-model")
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve image directory: check img_diffused, then img, then fallback to D:/
    img_dir = os.path.join(workspace_dir, 'dataset')
    if not os.path.exists(img_dir):
        img_dir = 'D:/dataset'
            
    # Resolve telemetry path: check local dataset, then fallback to D:/
    telemetry_path = os.path.join(workspace_dir, 'dataset', 'telemetry.csv')
    if os.path.exists(telemetry_path):
        print(f"Using local telemetry file: {telemetry_path}")
    else:
        telemetry_path = 'D:/dataset/telemetry.csv'
        print(f"Local telemetry not found. Falling back to {telemetry_path}")
    
    batch_size = 64 # Reduced for EfficientNet B1 on 11GB VRAM
    epochs = 20
    lr = 0.001
    weight_decay = 1e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = SteeringVelocityDataset(img_dir, telemetry_path, split='train')
    split_lists = train_dataset.get_split_lists()
    val_dataset = SteeringVelocityDataset(
        img_dir, telemetry_path, split='val',
        images=split_lists, frame_to_idx=train_dataset.frame_to_idx, 
        telemetry=train_dataset.telemetry, max_speed=train_dataset.max_speed
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn
    )
    
    model = SteeringModel().to(device)
    
    # FIX (Bug 12): steering matters far more than speed prediction;
    # weight losses accordingly instead of treating them equally.
    steer_weight = 0.8
    speed_weight  = 0.2
    _huber = nn.SmoothL1Loss(reduction='mean')
    def criterion(outputs, labels):
        steer_loss = _huber(outputs[:, 0], labels[:, 0])
        speed_loss  = _huber(outputs[:, 1], labels[:, 1])
        return steer_weight * steer_loss + speed_weight * speed_loss

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for imgs, labels in pbar_val:
                imgs = imgs.to(device)
                labels = labels.to(device)
                
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                
        val_loss /= len(val_dataset)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
        
        wandb.log({
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Learning Rate": current_lr,
            "epoch": epoch + 1
        })
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_steering_velocity_model.pth')
            # FIX (Bug 14): persist max_speed so test_cnn.py can load it automatically
            # instead of relying on the hardcoded --max-speed 28.61 default.
            meta = {"max_speed": train_dataset.max_speed}
            with open('best_steering_velocity_model_meta.json', 'w') as mf:
                json.dump(meta, mf, indent=2)
            print(f"--> Saved new best model with Val Loss: {val_loss:.4f}  (max_speed={train_dataset.max_speed:.4f})")
            
    wandb.finish()

if __name__ == '__main__':
    main()
