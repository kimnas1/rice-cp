"""
Rice Quality Assessment - V5 Detection-Inspired Counting
=========================================================
This approach uses detection-inspired architecture for better counting:
1. FPN (Feature Pyramid Network) for multi-scale grain detection
2. Density map regression head (like CSRNet)  
3. Global Sum Pooling for count estimation
4. Much higher resolution (640px) to see individual grains
5. Focal-style loss for handling count variance
6. Strong augmentations optimized for counting

Key insight: Winner achieved 12-32 MAE on counts - they likely used
detection or density-based approaches, not simple regression.
"""

import os
import sys
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import timm
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================
class Config:
    """V5 Configuration - Detection-Inspired Counting"""
    DATA_DIR = Path("RiceData/Unido_AfricaRice_Challenge")
    IMAGE_DIR = None
    OUTPUT_DIR = Path("outputs/v5_detection")
    LOG_DIR = None
    
    # Model - Use ConvNeXt with FPN for multi-scale
    BACKBONE = "convnext_base.fb_in22k_ft_in1k"
    PRETRAINED = True
    
    # Training - HIGHER resolution for grain visibility
    IMG_SIZE = 512
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 4
    EPOCHS = 40
    N_FOLDS = 5
    FOLD_TO_RUN = [0]
    
    # Optimizer
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01
    
    # EMA
    USE_EMA = True
    EMA_DECAY = 0.999
    
    # Target groups
    COUNT_TARGETS = ['Count', 'Broken_Count', 'Long_Count', 'Medium_Count',
                     'Black_Count', 'Chalky_Count', 'Red_Count', 
                     'Yellow_Count', 'Green_Count']
    CONTINUOUS_TARGETS = ['WK_Length_Average', 'WK_Width_Average', 
                          'WK_LW_Ratio_Average', 'Average_L', 
                          'Average_a', 'Average_b']
    
    TARGET_COLS = COUNT_TARGETS + CONTINUOUS_TARGETS
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = True
    SEED = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ============================================================================
# EMA
# ============================================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data
    
    def apply(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]
    
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]


# ============================================================================
# Target Normalizer - Simple Z-score
# ============================================================================
class TargetNormalizer:
    def __init__(self):
        self.means = None
        self.stds = None
    
    def fit(self, targets):
        self.means = targets.mean(axis=0)
        self.stds = targets.std(axis=0)
        self.stds[self.stds < 1e-6] = 1.0
        return self
    
    def transform(self, targets):
        return (targets - self.means) / self.stds
    
    def inverse_transform(self, targets):
        return targets * self.stds + self.means


# ============================================================================
# Dataset
# ============================================================================
class RiceDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, normalizer=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.normalizer = normalizer
        self.is_train = is_train
        self.type_map = {'Paddy': 0, 'White': 1, 'Brown': 2}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(self.img_dir / f"{row['ID']}.png").convert('RGB'))
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        rice_type = self.type_map.get(row['Comment'], 0)
        
        if self.is_train:
            targets = row[Config.TARGET_COLS].values.astype(np.float32)
            if self.normalizer:
                targets = self.normalizer.transform(targets.reshape(1, -1))[0]
            return img, torch.tensor(targets, dtype=torch.float32), rice_type
        return img, row['ID']


# ============================================================================
# Augmentations - Optimized for counting
# ============================================================================
def get_train_transforms(img_size):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), ratio=(0.8, 1.2)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.25, rotate_limit=45, p=0.8),
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30),
        ], p=0.6),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05)),
            A.GaussianBlur(blur_limit=(3, 5)),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_valid_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ============================================================================
# Feature Pyramid Network (FPN) Neck
# ============================================================================
class FPN(nn.Module):
    """Simple FPN for multi-scale feature fusion"""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])
    
    def forward(self, features):
        # features: list of [C2, C3, C4, C5] from backbone
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode='nearest'
            )
        
        # FPN outputs
        outputs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return outputs


# ============================================================================
# Counting Head with Global Sum Pooling
# ============================================================================
class CountingHead(nn.Module):
    """
    Detection-inspired counting head:
    - Uses dilated convolutions like CSRNet
    - Global Sum Pooling (proven better than avg for counting)
    - Separate outputs for each count target
    """
    def __init__(self, in_channels, num_counts=9):
        super().__init__()
        
        # Dilated conv layers (CSRNet-inspired)
        self.dilated = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=4, dilation=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Density map predictor - one channel per count target
        self.density_pred = nn.Conv2d(128, num_counts, 1)
        
        # Also predict raw counts via regression (ensemble)
        self.count_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_counts)
        )
        
        self.num_counts = num_counts
    
    def forward(self, x):
        # x: feature map from FPN
        feat = self.dilated(x)
        
        # Density map prediction
        density = self.density_pred(feat)  # (B, num_counts, H, W)
        # Global sum pooling for count from density
        density_counts = density.sum(dim=(2, 3))  # (B, num_counts)
        
        # Direct regression counts
        reg_counts = self.count_regressor(feat)
        
        # Combine both (learn optimal weighting)
        return density_counts, reg_counts


# ============================================================================
# V5 Detection-Inspired Model
# ============================================================================
class RiceModelV5(nn.Module):
    """
    Detection-inspired architecture:
    - ConvNeXt backbone with multi-scale features
    - FPN neck for feature fusion
    - Counting head with density + regression
    - Separate continuous target head
    """
    def __init__(self, backbone_name, num_count_targets=9, 
                 num_continuous_targets=6, num_types=3, pretrained=True):
        super().__init__()
        
        # Backbone with multi-scale features
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True  # Returns all 4 stages by default
        )
        
        # Get feature channels
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.backbone(dummy)
            self.feat_channels = [f.shape[1] for f in feats]
        
        # FPN neck
        self.fpn = FPN(self.feat_channels, out_channels=256)
        
        # Counting head (uses FPN P3 - good balance of resolution and semantics)
        self.counting_head = CountingHead(256, num_count_targets)
        
        # Type embedding
        self.type_emb = nn.Embedding(num_types, 64)
        
        # Continuous target head (takes pooled features + type embedding)
        self.continuous_head = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_continuous_targets)
        )
        
        # Type classifier
        self.type_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feat_channels[-1], num_types)
        )
        
        # Learnable weight for density vs regression
        self.count_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x, rice_type=None):
        # Get multi-scale features
        backbone_feats = self.backbone(x)
        
        # FPN
        fpn_feats = self.fpn(backbone_feats)
        
        # Use P3 (index 1 in FPN outputs) for counting
        p3 = fpn_feats[1]
        
        # Counting
        density_counts, reg_counts = self.counting_head(p3)
        
        # Combine with learnable weight
        w = torch.sigmoid(self.count_weight)
        count_preds = w * density_counts + (1 - w) * reg_counts
        
        # Type classification
        type_logits = self.type_classifier(backbone_feats[-1])
        
        # Type embedding
        if rice_type is not None:
            type_e = self.type_emb(rice_type)
        else:
            type_e = self.type_emb(type_logits.argmax(1))
        
        # Continuous targets
        p4 = fpn_feats[2]
        pooled = F.adaptive_avg_pool2d(p4, 1).flatten(1)
        cont_in = torch.cat([pooled, type_e], dim=1)
        cont_preds = self.continuous_head(cont_in)
        
        # Combine all predictions
        predictions = torch.cat([count_preds, cont_preds], dim=1)
        
        return predictions, type_logits


# ============================================================================
# Loss
# ============================================================================
class CountFocusedLoss(nn.Module):
    """Higher weight on count targets + Huber loss"""
    def __init__(self, num_counts=9, count_weight=3.0):
        super().__init__()
        self.num_counts = num_counts
        self.count_weight = count_weight
    
    def forward(self, pred, target):
        loss = F.smooth_l1_loss(pred, target, reduction='none', beta=1.0)
        
        weights = torch.ones_like(loss)
        weights[:, :self.num_counts] = self.count_weight
        
        return (loss * weights).mean()


# ============================================================================
# Training
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, scaler, ema, config, accum):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training")
    for step, (imgs, targets, types) in enumerate(pbar):
        imgs = imgs.to(config.DEVICE)
        targets = targets.to(config.DEVICE)
        types = types.to(config.DEVICE)
        
        with autocast(enabled=config.USE_AMP):
            preds, type_logits = model(imgs, types)
            loss = criterion(preds, targets)
            type_loss = F.cross_entropy(type_logits, types)
            total = (loss + 0.1 * type_loss) / accum
        
        scaler.scale(total).backward()
        
        if (step + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)
        
        total_loss += loss.item() * accum
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, normalizer, config):
    model.eval()
    all_preds, all_targets = [], []
    
    for imgs, targets, types in tqdm(loader, desc="Validating"):
        imgs = imgs.to(config.DEVICE)
        types = types.to(config.DEVICE)
        
        with autocast(enabled=config.USE_AMP):
            preds, _ = model(imgs, types)
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())
    
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # Denormalize
    preds = normalizer.inverse_transform(preds)
    targets = normalizer.inverse_transform(targets)
    
    # Clip negative counts
    n_counts = len(Config.COUNT_TARGETS)
    preds[:, :n_counts] = np.maximum(preds[:, :n_counts], 0)
    
    mae = np.abs(preds - targets).mean(axis=0)
    count_mae = mae[:n_counts].mean()
    cont_mae = mae[n_counts:].mean()
    
    return mae.mean(), mae, count_mae, cont_mae


def print_metrics(mae, targets, count_mae, cont_mae):
    print("\nTarget                           MAE")
    print("─" * 40)
    for t, m in zip(targets, mae):
        flag = "✓" if m < 15 else "⚠️" if m > 30 else " "
        print(f"{flag} {t:25s} {m:8.2f}")
    print("─" * 40)
    print(f"Count Mean: {count_mae:.2f} | Cont Mean: {cont_mae:.2f}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("🌾 RICE V5 - DETECTION-INSPIRED COUNTING")
    print("=" * 70)
    print(f"Device: {Config.DEVICE}")
    print(f"Backbone: {Config.BACKBONE}")
    print(f"Image size: {Config.IMG_SIZE}")
    print(f"Batch: {Config.BATCH_SIZE} x {Config.ACCUMULATION_STEPS}")
    
    set_seed(Config.SEED)
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(Config.DATA_DIR / "Train.csv")
    print(f"Samples: {len(df)}")
    
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df['Comment'])):
        df.loc[val_idx, 'fold'] = fold
    
    results = []
    
    for fold in Config.FOLD_TO_RUN:
        print(f"\n{'=' * 70}\nFOLD {fold}\n{'=' * 70}")
        
        train_df = df[df['fold'] != fold]
        val_df = df[df['fold'] == fold]
        
        normalizer = TargetNormalizer()
        normalizer.fit(train_df[Config.TARGET_COLS].values.astype(np.float32))
        
        train_ds = RiceDataset(train_df, Config.IMAGE_DIR, 
                               get_train_transforms(Config.IMG_SIZE), normalizer)
        val_ds = RiceDataset(val_df, Config.IMAGE_DIR,
                             get_valid_transforms(Config.IMG_SIZE), normalizer)
        
        train_loader = DataLoader(train_ds, Config.BATCH_SIZE, shuffle=True, 
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, Config.BATCH_SIZE * 2, shuffle=False,
                                num_workers=4, pin_memory=True)
        
        model = RiceModelV5(
            Config.BACKBONE,
            num_count_targets=len(Config.COUNT_TARGETS),
            num_continuous_targets=len(Config.CONTINUOUS_TARGETS)
        ).to(Config.DEVICE)
        
        print(f"Model: {Config.BACKBONE}, FPN channels: {model.feat_channels}")
        
        ema = EMA(model, Config.EMA_DECAY) if Config.USE_EMA else None
        criterion = CountFocusedLoss(len(Config.COUNT_TARGETS), count_weight=3.0)
        optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, 
                         weight_decay=Config.WEIGHT_DECAY)
        scheduler = OneCycleLR(optimizer, max_lr=Config.LEARNING_RATE * 10,
                               epochs=Config.EPOCHS, 
                               steps_per_epoch=len(train_loader) // Config.ACCUMULATION_STEPS,
                               pct_start=0.1)
        scaler = GradScaler(enabled=Config.USE_AMP)
        
        best_count_mae = float('inf')
        patience = 0
        
        for epoch in range(1, Config.EPOCHS + 1):
            print(f"\n--- Epoch {epoch}/{Config.EPOCHS} ---")
            
            train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                     scaler, ema, Config, Config.ACCUMULATION_STEPS)
            
            if ema:
                ema.apply(model)
            
            overall, mae, count_mae, cont_mae = validate(model, val_loader, 
                                                          normalizer, Config)
            if ema:
                ema.restore(model)
            
            if count_mae < best_count_mae:
                best_count_mae = count_mae
                patience = 0
                torch.save({
                    'model': model.state_dict(),
                    'normalizer': {'means': normalizer.means, 'stds': normalizer.stds},
                    'best_count_mae': best_count_mae,
                    'epoch': epoch
                }, Config.OUTPUT_DIR / f"best_fold{fold}.pt")
                print(f"🏆 NEW BEST Count MAE: {count_mae:.2f}")
            else:
                patience += 1
            
            print(f"Loss: {train_loss:.4f} | MAE: {overall:.2f} | Count: {count_mae:.2f} | Cont: {cont_mae:.2f}")
            print_metrics(mae, Config.TARGET_COLS, count_mae, cont_mae)
            
            if patience >= 12:
                print("Early stopping")
                break
        
        results.append({'fold': fold, 'best_count_mae': best_count_mae})
        print(f"\n✅ Fold {fold} Best Count MAE: {best_count_mae:.2f}")
    
    print("\n" + "=" * 70)
    print("🏆 COMPLETE")
    for r in results:
        print(f"Fold {r['fold']}: Count MAE = {r['best_count_mae']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='RiceData/Unido_AfricaRice_Challenge')
    parser.add_argument('--output-dir', default='outputs/v5_detection')
    parser.add_argument('--model', default='convnext_base.fb_in22k_ft_in1k')
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--folds', default='0')
    args = parser.parse_args()
    
    Config.DATA_DIR = Path(args.data_dir)
    Config.IMAGE_DIR = Config.DATA_DIR / "unido_rice_images"
    Config.OUTPUT_DIR = Path(args.output_dir)
    Config.BACKBONE = args.model
    Config.IMG_SIZE = args.img_size
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.lr
    Config.FOLD_TO_RUN = [int(f) for f in args.folds.split(',')]
    
    main()
