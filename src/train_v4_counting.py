"""
Rice Quality Assessment - V4 Counting-Optimized Training Script
================================================================
Key Architecture Changes for Better Counting:
1. Swin Transformer backbone (better for counting/dense prediction)
2. Multi-scale feature aggregation (FPN-like)
3. Separate specialized heads: COUNT head vs CONTINUOUS head
4. Attention-based pooling for count features
5. Higher resolution (640px) for grain detail
6. Log-space prediction for count targets (handles large variance)
7. Huber loss for robustness to outliers
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
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
    """V4 Configuration - Counting Optimized"""
    # Paths - set via command line args
    DATA_DIR = Path("RiceData/Unido_AfricaRice_Challenge")
    IMAGE_DIR = None
    OUTPUT_DIR = Path("outputs/v4_counting")
    LOG_DIR = None
    
    # Model - Swin Transformer for better counting
    BACKBONE = "swin_base_patch4_window7_224"
    PRETRAINED = True
    
    # Training - Higher resolution for counting
    IMG_SIZE = 384  # Will be overridden by args
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 4  # Effective batch size = 16
    EPOCHS = 30
    N_FOLDS = 5
    FOLD_TO_RUN = [0]
    
    # Optimizer
    LEARNING_RATE = 5e-5  # Lower LR for Swin
    WEIGHT_DECAY = 0.05
    
    # EMA
    USE_EMA = True
    EMA_DECAY = 0.999
    
    # Mixup - stronger for counting
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2  # Lower alpha for regression
    MIXUP_PROB = 0.3
    
    # Target groups
    COUNT_TARGETS = ['Count', 'Broken_Count', 'Long_Count', 'Medium_Count',
                     'Black_Count', 'Chalky_Count', 'Red_Count', 
                     'Yellow_Count', 'Green_Count']
    CONTINUOUS_TARGETS = ['WK_Length_Average', 'WK_Width_Average', 
                          'WK_LW_Ratio_Average', 'Average_L', 
                          'Average_a', 'Average_b']
    
    TARGET_COLS = COUNT_TARGETS + CONTINUOUS_TARGETS
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = True
    
    # Log transform for counts (can be disabled via --no-log)
    USE_LOG_TRANSFORM = True
    
    SEED = 42

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# EMA
# ============================================================================
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()
    
    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================================
# Mixup for Regression
# ============================================================================
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # Ensure one sample dominates
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# Target Normalizer with Log Transform for Counts
# ============================================================================
class TargetNormalizer:
    """Normalizer with log-transform option for count targets"""
    def __init__(self, count_indices: List[int], use_log_for_counts: bool = True):
        self.count_indices = count_indices
        self.use_log = use_log_for_counts
        self.means = None
        self.stds = None
        self.is_fitted = False
    
    def fit(self, targets: np.ndarray):
        targets = targets.copy()
        # Apply log1p to count columns before fitting
        if self.use_log:
            targets[:, self.count_indices] = np.log1p(targets[:, self.count_indices])
        
        self.means = targets.mean(axis=0)
        self.stds = targets.std(axis=0)
        self.stds[self.stds < 1e-6] = 1.0
        self.is_fitted = True
        return self
    
    def transform(self, targets: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted")
        targets = targets.copy()
        if self.use_log:
            targets[:, self.count_indices] = np.log1p(targets[:, self.count_indices])
        return (targets - self.means) / self.stds
    
    def inverse_transform(self, targets: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted")
        result = targets * self.stds + self.means
        if self.use_log:
            result[:, self.count_indices] = np.expm1(result[:, self.count_indices])
        return result


# ============================================================================
# Dataset
# ============================================================================
class RiceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, 
                 transform=None, normalizer=None, target_cols=None,
                 is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.normalizer = normalizer
        self.target_cols = target_cols or Config.TARGET_COLS
        self.is_train = is_train
        
        self.type_mapping = {'Paddy': 0, 'White': 1, 'Brown': 2}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = self.img_dir / f"{row['ID']}.png"
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        rice_type = self.type_mapping.get(row['Comment'], 0)
        
        if self.is_train:
            targets = row[self.target_cols].values.astype(np.float32)
            if self.normalizer is not None:
                targets = self.normalizer.transform(targets.reshape(1, -1))[0]
            return image, torch.tensor(targets, dtype=torch.float32), rice_type
        else:
            return image, row['ID']


# ============================================================================
# Augmentations - Stronger for Counting
# ============================================================================
def get_train_transforms(img_size: int = 384):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7),
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.08)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),
        A.CoarseDropout(max_holes=12, max_height=img_size//16, max_width=img_size//16, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_valid_transforms(img_size: int = 384):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ============================================================================
# Attention Pooling Module
# ============================================================================
class AttentionPool(nn.Module):
    """Attention-based global pooling for better feature aggregation"""
    def __init__(self, in_features: int, hidden_features: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, 1)
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x: (B, N, C) where N is spatial dimension flattened
        attn_weights = self.attention(x)  # (B, N, 1)
        attn_weights = self.softmax(attn_weights)  # (B, N, 1)
        pooled = (x * attn_weights).sum(dim=1)  # (B, C)
        return pooled, attn_weights


# ============================================================================
# Multi-Scale Feature Aggregator
# ============================================================================
class MultiScaleAggregator(nn.Module):
    """Aggregates features from multiple backbone stages"""
    def __init__(self, in_channels_list: List[int], out_channels: int = 512):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels_list
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features_list):
        # Upsample all to largest size
        target_size = features_list[0].shape[-2:]
        projected = []
        for feat, proj in zip(features_list, self.projections):
            p = proj(feat)
            if p.shape[-2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
            projected.append(p)
        
        fused = torch.cat(projected, dim=1)
        return self.fusion(fused)


# ============================================================================
# V4 Counting Model
# ============================================================================
class RiceModelV4(nn.Module):
    """
    Counting-optimized model with:
    - Swin/ConvNeXt backbone
    - Separate heads for COUNT vs CONTINUOUS targets
    - Attention pooling for counts
    - Multi-scale feature aggregation
    """
    def __init__(self, backbone_name: str, num_count_targets: int = 9,
                 num_continuous_targets: int = 6, num_types: int = 3,
                 pretrained: bool = True):
        super().__init__()
        
        # Create backbone with feature extraction
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=False,
            num_classes=0,
            global_pool=''  # No pooling - we'll do custom
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.backbone(dummy)
            if len(feat.shape) == 3:  # Transformer output (B, N, C)
                self.feat_dim = feat.shape[-1]
                self.is_transformer = True
            else:  # CNN output (B, C, H, W)
                self.feat_dim = feat.shape[1]
                self.is_transformer = False
        
        # Type embedding
        self.type_embedding = nn.Embedding(num_types, 64)
        
        # Attention pooling for count features
        self.count_attention_pool = AttentionPool(self.feat_dim, hidden_features=256)
        
        # Standard pooling for continuous targets
        self.avg_pool = nn.AdaptiveAvgPool1d(1) if self.is_transformer else nn.AdaptiveAvgPool2d(1)
        
        # Count-specialized head (larger, deeper)
        self.count_head = nn.Sequential(
            nn.Linear(self.feat_dim + 64, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_count_targets)
        )
        
        # Continuous target head (simpler)
        self.continuous_head = nn.Sequential(
            nn.Linear(self.feat_dim + 64, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_continuous_targets)
        )
        
        # Type classifier (auxiliary)
        self.type_classifier = nn.Linear(self.feat_dim, num_types)
        
        self.num_count_targets = num_count_targets
        self.num_continuous_targets = num_continuous_targets
    
    def forward(self, x, rice_type=None):
        # Extract features
        features = self.backbone(x)
        
        if self.is_transformer:
            # Transformer: (B, N, C)
            feat_seq = features
        else:
            # CNN: (B, C, H, W) -> (B, N, C)
            B, C, H, W = features.shape
            feat_seq = features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Attention pooling for count targets
        count_features, attn_weights = self.count_attention_pool(feat_seq)  # (B, C)
        
        # Average pooling for continuous targets
        if self.is_transformer:
            cont_features = feat_seq.mean(dim=1)  # (B, C)
        else:
            cont_features = F.adaptive_avg_pool2d(features, 1).flatten(1)  # (B, C)
        
        # Type classification
        type_logits = self.type_classifier(cont_features)
        
        # Get type embedding
        if rice_type is not None:
            type_emb = self.type_embedding(rice_type)
        else:
            type_pred = type_logits.argmax(dim=1)
            type_emb = self.type_embedding(type_pred)
        
        # Combine features with type embedding
        count_combined = torch.cat([count_features, type_emb], dim=1)
        cont_combined = torch.cat([cont_features, type_emb], dim=1)
        
        # Predict
        count_preds = self.count_head(count_combined)
        cont_preds = self.continuous_head(cont_combined)
        
        # Combine predictions in correct order
        predictions = torch.cat([count_preds, cont_preds], dim=1)
        
        return predictions, type_logits, attn_weights


# ============================================================================
# Loss Functions
# ============================================================================
class CountingLoss(nn.Module):
    """
    Combined loss for counting:
    - Huber loss for robustness
    - Higher weight for count targets
    """
    def __init__(self, num_count_targets: int = 9, count_weight: float = 3.0):
        super().__init__()
        self.num_count = num_count_targets
        self.count_weight = count_weight
        self.huber = nn.SmoothL1Loss(reduction='none', beta=0.5)
    
    def forward(self, pred, target):
        loss = self.huber(pred, target)
        
        # Weight count targets more heavily
        weights = torch.ones_like(loss)
        weights[:, :self.num_count] = self.count_weight
        
        weighted_loss = (loss * weights).mean()
        
        # Also return individual losses for monitoring
        count_loss = loss[:, :self.num_count].mean()
        cont_loss = loss[:, self.num_count:].mean()
        
        return weighted_loss, {'count_loss': count_loss.item(), 'cont_loss': cont_loss.item()}


# ============================================================================
# Training Functions
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, ema, 
                    config, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    running_count_loss = 0.0
    running_cont_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training")
    for step, (images, targets, rice_types) in enumerate(pbar):
        images = images.to(config.DEVICE)
        targets = targets.to(config.DEVICE)
        rice_types = rice_types.to(config.DEVICE)
        
        # Apply Mixup
        if config.USE_MIXUP and random.random() < config.MIXUP_PROB:
            images, targets_a, targets_b, lam = mixup_data(images, targets, config.MIXUP_ALPHA)
            
            with autocast(enabled=config.USE_AMP):
                predictions, type_logits, _ = model(images, rice_types)
                loss_a, _ = criterion(predictions, targets_a)
                loss_b, _ = criterion(predictions, targets_b)
                loss = lam * loss_a + (1 - lam) * loss_b
                loss_dict = {'count_loss': 0, 'cont_loss': 0}
        else:
            with autocast(enabled=config.USE_AMP):
                predictions, type_logits, _ = model(images, rice_types)
                loss, loss_dict = criterion(predictions, targets)
        
        # Add auxiliary type loss
        with autocast(enabled=config.USE_AMP):
            type_loss = F.cross_entropy(type_logits, rice_types)
            total_loss = loss + 0.1 * type_loss
        
        total_loss = total_loss / accumulation_steps
        
        scaler.scale(total_loss).backward()
        
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if ema is not None:
                ema.update()
        
        running_loss += loss.item() * accumulation_steps
        running_count_loss += loss_dict.get('count_loss', 0)
        running_cont_loss += loss_dict.get('cont_loss', 0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cnt': f'{loss_dict.get("count_loss", 0):.4f}'
        })
    
    n = len(loader)
    return running_loss / n, running_count_loss / n, running_cont_loss / n


@torch.no_grad()
def validate(model, loader, normalizer, config):
    model.eval()
    all_preds = []
    all_targets = []
    
    for images, targets, rice_types in tqdm(loader, desc="Validating"):
        images = images.to(config.DEVICE)
        rice_types = rice_types.to(config.DEVICE)
        
        with autocast(enabled=config.USE_AMP):
            predictions, _, _ = model(images, rice_types)
        
        all_preds.append(predictions.cpu().numpy())
        all_targets.append(targets.numpy())
    
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # Denormalize
    preds_denorm = normalizer.inverse_transform(preds)
    targets_denorm = normalizer.inverse_transform(targets)
    
    # Clip negative counts
    num_counts = len(Config.COUNT_TARGETS)
    preds_denorm[:, :num_counts] = np.maximum(preds_denorm[:, :num_counts], 0)
    
    # Calculate per-target MAE
    mae_per_target = np.abs(preds_denorm - targets_denorm).mean(axis=0)
    overall_mae = mae_per_target.mean()
    
    # Separate count and continuous MAE
    count_mae = mae_per_target[:num_counts].mean()
    cont_mae = mae_per_target[num_counts:].mean()
    
    return overall_mae, mae_per_target, count_mae, cont_mae


def print_metrics(mae_per_target, target_cols, count_mae, cont_mae):
    print("\nTarget                           MAE")
    print("───────────────────────────────────")
    
    for col, mae in zip(target_cols, mae_per_target):
        flag = "⚠️" if mae > 30 else "✓ " if mae < 15 else "  "
        print(f"{flag} {col:25s} {mae:8.2f}")
    
    print("───────────────────────────────────")
    print(f"Count Targets Mean         {count_mae:8.2f}")
    print(f"Continuous Targets Mean    {cont_mae:8.2f}")
    print(f"Overall Mean               {mae_per_target.mean():8.2f}")


# ============================================================================
# Main Training Loop
# ============================================================================
def main():
    print("="*72)
    print("🌾 RICE V4 - COUNTING-OPTIMIZED MODEL")
    print("="*72)
    print(f"Device: {Config.DEVICE}")
    print(f"Backbone: {Config.BACKBONE}")
    print(f"Image size: {Config.IMG_SIZE}")
    print(f"Batch size: {Config.BATCH_SIZE} x {Config.ACCUMULATION_STEPS} = {Config.BATCH_SIZE * Config.ACCUMULATION_STEPS}")
    print(f"Count targets: {len(Config.COUNT_TARGETS)}")
    print(f"Continuous targets: {len(Config.CONTINUOUS_TARGETS)}")
    
    set_seed(Config.SEED)
    
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(Config.DATA_DIR / "Train.csv")
    print(f"\n📊 Train samples: {len(train_df)}")
    
    # Create folds
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    train_df['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df['Comment'])):
        train_df.loc[val_idx, 'fold'] = fold
    
    # Get count indices
    count_indices = list(range(len(Config.COUNT_TARGETS)))
    
    fold_results = []
    
    for fold in Config.FOLD_TO_RUN:
        print(f"\n{'='*72}")
        print(f"FOLD {fold}")
        print(f"{'='*72}")
        
        train_fold = train_df[train_df['fold'] != fold].reset_index(drop=True)
        valid_fold = train_df[train_df['fold'] == fold].reset_index(drop=True)
        print(f"Train: {len(train_fold)}, Valid: {len(valid_fold)}")
        
        # Fit normalizer - optionally use log transform for counts
        normalizer = TargetNormalizer(count_indices, use_log_for_counts=Config.USE_LOG_TRANSFORM)
        train_targets = train_fold[Config.TARGET_COLS].values.astype(np.float32)
        normalizer.fit(train_targets)
        
        if Config.USE_LOG_TRANSFORM:
            print("\n📊 Using log-transform for count targets")
        else:
            print("\n📊 Using standard z-score normalization (no log)")
        
        # Create datasets
        train_dataset = RiceDataset(
            train_fold, Config.IMAGE_DIR,
            transform=get_train_transforms(Config.IMG_SIZE),
            normalizer=normalizer, target_cols=Config.TARGET_COLS, is_train=True
        )
        valid_dataset = RiceDataset(
            valid_fold, Config.IMAGE_DIR,
            transform=get_valid_transforms(Config.IMG_SIZE),
            normalizer=normalizer, target_cols=Config.TARGET_COLS, is_train=True
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE,
            shuffle=True, num_workers=4, pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=Config.BATCH_SIZE * 2,
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        # Create model
        model = RiceModelV4(
            Config.BACKBONE,
            num_count_targets=len(Config.COUNT_TARGETS),
            num_continuous_targets=len(Config.CONTINUOUS_TARGETS),
            pretrained=Config.PRETRAINED
        ).to(Config.DEVICE)
        
        print(f"\n🔧 Model: {Config.BACKBONE}")
        print(f"   Feature dim: {model.feat_dim}")
        print(f"   Is Transformer: {model.is_transformer}")
        
        # EMA
        ema = EMA(model, decay=Config.EMA_DECAY) if Config.USE_EMA else None
        
        # Loss
        criterion = CountingLoss(
            num_count_targets=len(Config.COUNT_TARGETS),
            count_weight=3.0
        )
        
        # Optimizer with layer-wise LR decay for transformers
        optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE,
                         weight_decay=Config.WEIGHT_DECAY)
        
        # OneCycleLR for better convergence
        scheduler = OneCycleLR(
            optimizer,
            max_lr=Config.LEARNING_RATE * 10,
            epochs=Config.EPOCHS,
            steps_per_epoch=len(train_loader) // Config.ACCUMULATION_STEPS,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        scaler = GradScaler(enabled=Config.USE_AMP)
        
        best_mae = float('inf')
        best_count_mae = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(1, Config.EPOCHS + 1):
            print(f"\n{'─'*72}")
            print(f"Fold {fold} | Epoch {epoch}/{Config.EPOCHS}")
            print(f"{'─'*72}")
            
            train_loss, train_count_loss, train_cont_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, ema,
                Config, Config.ACCUMULATION_STEPS
            )
            
            if ema is not None:
                ema.apply_shadow()
            
            val_mae, mae_per_target, count_mae, cont_mae = validate(
                model, valid_loader, normalizer, Config
            )
            
            if ema is not None:
                ema.restore()
            
            # Step scheduler per batch, so don't step here
            
            # Track best by count MAE (our main goal)
            is_best = count_mae < best_count_mae
            if is_best:
                best_mae = val_mae
                best_count_mae = count_mae
                best_epoch = epoch
                patience_counter = 0
                
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'ema_shadow': ema.shadow if ema else None,
                    'normalizer_means': normalizer.means,
                    'normalizer_stds': normalizer.stds,
                    'count_indices': count_indices,
                    'best_mae': best_mae,
                    'best_count_mae': best_count_mae,
                    'epoch': epoch,
                    'fold': fold,
                    'config': {
                        'backbone': Config.BACKBONE,
                        'img_size': Config.IMG_SIZE,
                        'count_targets': Config.COUNT_TARGETS,
                        'continuous_targets': Config.CONTINUOUS_TARGETS,
                    }
                }
                torch.save(save_dict, Config.OUTPUT_DIR / f"best_model_fold{fold}.pt")
                print(f"🏆 NEW BEST COUNT MAE: {count_mae:.2f}")
            else:
                patience_counter += 1
            
            print(f"Train Loss: {train_loss:.4f} (count: {train_count_loss:.4f})")
            print(f"Val MAE: {val_mae:.2f} | Count MAE: {count_mae:.2f} | Cont MAE: {cont_mae:.2f}")
            print_metrics(mae_per_target, Config.TARGET_COLS, count_mae, cont_mae)
            
            if patience_counter >= 10:
                print(f"\n⚠️ Early stopping at epoch {epoch}")
                break
        
        print(f"\n✅ Fold {fold} complete!")
        print(f"   Best MAE: {best_mae:.2f}")
        print(f"   Best Count MAE: {best_count_mae:.2f}")
        print(f"   Best Epoch: {best_epoch}")
        
        fold_results.append({
            'fold': fold, 'best_mae': best_mae, 
            'best_count_mae': best_count_mae, 'best_epoch': best_epoch
        })
    
    print("\n" + "="*72)
    print("🏆 TRAINING COMPLETE")
    print("="*72)
    for r in fold_results:
        print(f"Fold {r['fold']}: MAE={r['best_mae']:.2f}, Count MAE={r['best_count_mae']:.2f}")
    
    avg_count_mae = np.mean([r['best_count_mae'] for r in fold_results])
    print(f"\n📊 Average Count MAE: {avg_count_mae:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='V4 Rice Counting Training')
    parser.add_argument('--data-dir', type=str, default='RiceData/Unido_AfricaRice_Challenge')
    parser.add_argument('--output-dir', type=str, default='outputs/v4_counting')
    parser.add_argument('--model', type=str, default='swin_base_patch4_window7_224')
    parser.add_argument('--img-size', type=int, default=384)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--folds', type=str, default='0')
    parser.add_argument('--no-log', action='store_true', help='Disable log transform for counts')
    args = parser.parse_args()
    
    Config.DATA_DIR = Path(args.data_dir)
    Config.IMAGE_DIR = Config.DATA_DIR / "unido_rice_images"
    Config.OUTPUT_DIR = Path(args.output_dir)
    Config.LOG_DIR = Config.OUTPUT_DIR.parent / "logs"
    Config.BACKBONE = args.model
    Config.IMG_SIZE = args.img_size
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.lr
    Config.FOLD_TO_RUN = [int(f) for f in args.folds.split(',')]
    Config.USE_LOG_TRANSFORM = not args.no_log
    
    main()
