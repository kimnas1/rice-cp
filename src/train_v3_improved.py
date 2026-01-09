"""
Rice Quality Assessment - V3 Improved Training Script
=====================================================
Key Improvements over Baseline V2:
1. ConvNeXt-Base backbone (better than EfficientNet for counting)
2. Higher resolution (512px)
3. Mixup augmentation for better generalization
4. EMA (Exponential Moving Average) for training stability
5. Count-weighted loss (2x weight for count targets)
6. Gradient accumulation for effective larger batch
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
    """V3 Configuration with SOTA improvements"""
    # Paths - set via command line args
    DATA_DIR = Path("RiceData/Unido_AfricaRice_Challenge")
    IMAGE_DIR = None  # Set dynamically
    OUTPUT_DIR = Path("outputs/v3_improved")
    LOG_DIR = None  # Set dynamically
    
    # Model - UPGRADED to ConvNeXt-Base
    BACKBONE = "convnext_base.fb_in22k_ft_in1k"  # ConvNeXt-Base pretrained
    PRETRAINED = True
    
    # Training - UPGRADED resolution
    IMG_SIZE = 512  # Higher resolution for grain detail
    BATCH_SIZE = 8  # Kaggle GPU can handle larger batch
    ACCUMULATION_STEPS = 2  # Effective batch size = 16
    EPOCHS = 25
    N_FOLDS = 5
    FOLD_TO_RUN = [0]  # Run just fold 0 for validation first
    
    # Optimizer
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    WARMUP_EPOCHS = 2
    
    # EMA
    USE_EMA = True
    EMA_DECAY = 0.9995
    
    # Mixup
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4
    MIXUP_PROB = 0.5
    
    # Count target weighting
    COUNT_TARGETS = ['Count', 'Broken_Count', 'Long_Count', 'Medium_Count', 
                     'Black_Count', 'Chalky_Count', 'Red_Count', 
                     'Yellow_Count', 'Green_Count']
    COUNT_WEIGHT = 2.0  # 2x weight for count targets
    
    # Target columns
    TARGET_COLS = [
        'Count', 'Broken_Count', 'Long_Count', 'Medium_Count',
        'Black_Count', 'Chalky_Count', 'Red_Count', 'Yellow_Count', 'Green_Count',
        'WK_Length_Average', 'WK_Width_Average', 'WK_LW_Ratio_Average',
        'Average_L', 'Average_a', 'Average_b'
    ]
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = True
    
    SEED = 42

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================
class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model: nn.Module, decay: float = 0.9995):
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
# Mixup
# ============================================================================
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4) -> Tuple:
    """Apply mixup augmentation to batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculate mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# Target Normalizer
# ============================================================================
class TargetNormalizer:
    """Z-score normalization for targets"""
    def __init__(self):
        self.means = None
        self.stds = None
        self.is_fitted = False
    
    def fit(self, targets: np.ndarray):
        self.means = targets.mean(axis=0)
        self.stds = targets.std(axis=0)
        self.stds[self.stds < 1e-6] = 1.0
        self.is_fitted = True
        return self
    
    def transform(self, targets: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted")
        return (targets - self.means) / self.stds
    
    def inverse_transform(self, targets: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted")
        return targets * self.stds + self.means


# ============================================================================
# Dataset
# ============================================================================
class RiceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, 
                 transform=None, normalizer: Optional[TargetNormalizer] = None,
                 is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.normalizer = normalizer
        self.is_train = is_train
        
        self.type_mapping = {'Paddy': 0, 'White': 1, 'Brown': 2}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.img_dir / f"{row['ID']}.png"
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Get rice type
        rice_type = self.type_mapping.get(row['Comment'], 0)
        
        if self.is_train:
            targets = row[Config.TARGET_COLS].values.astype(np.float32)
            if self.normalizer is not None:
                targets = self.normalizer.transform(targets.reshape(1, -1))[0]
            return image, torch.tensor(targets, dtype=torch.float32), rice_type
        else:
            return image, row['ID']


# ============================================================================
# Augmentations - Enhanced for V3
# ============================================================================
def get_train_transforms(img_size: int = 512):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05)),
            A.GaussianBlur(blur_limit=(3, 5)),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_valid_transforms(img_size: int = 512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ============================================================================
# Model - ConvNeXt with Multi-Target Regression Head
# ============================================================================
class RiceModelV3(nn.Module):
    """ConvNeXt-based model with type conditioning"""
    def __init__(self, backbone_name: str, num_targets: int = 15, 
                 num_types: int = 3, pretrained: bool = True):
        super().__init__()
        
        # ConvNeXt backbone
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )
        
        # Get feature dimension
        self.feat_dim = self.backbone.num_features
        
        # Type embedding
        self.type_embedding = nn.Embedding(num_types, 64)
        
        # Multi-target regression head with more capacity
        self.regressor = nn.Sequential(
            nn.Linear(self.feat_dim + 64, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_targets)
        )
        
        # Auxiliary type classifier
        self.type_classifier = nn.Linear(self.feat_dim, num_types)
    
    def forward(self, x, rice_type=None):
        features = self.backbone(x)
        
        type_logits = self.type_classifier(features)
        
        if rice_type is not None:
            type_emb = self.type_embedding(rice_type)
        else:
            type_pred = type_logits.argmax(dim=1)
            type_emb = self.type_embedding(type_pred)
        
        combined = torch.cat([features, type_emb], dim=1)
        predictions = self.regressor(combined)
        
        return predictions, type_logits


# ============================================================================
# Weighted Loss Function
# ============================================================================
class WeightedMultiTargetLoss(nn.Module):
    """Loss with higher weight for count targets"""
    def __init__(self, target_cols: List[str], count_targets: List[str], count_weight: float = 2.0):
        super().__init__()
        self.weights = torch.ones(len(target_cols))
        for i, col in enumerate(target_cols):
            if col in count_targets:
                self.weights[i] = count_weight
        self.weights = self.weights / self.weights.sum() * len(target_cols)  # Normalize
    
    def forward(self, pred, target):
        weights = self.weights.to(pred.device)
        loss = F.smooth_l1_loss(pred, target, reduction='none')
        weighted_loss = (loss * weights).mean()
        return weighted_loss


# ============================================================================
# Training Functions
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, ema, 
                    config, accumulation_steps=1):
    model.train()
    running_loss = 0.0
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
                predictions, type_logits = model(images, rice_types)
                loss = mixup_criterion(criterion, predictions, targets_a, targets_b, lam)
        else:
            with autocast(enabled=config.USE_AMP):
                predictions, type_logits = model(images, rice_types)
                loss = criterion(predictions, targets)
        
        # Add auxiliary type loss
        with autocast(enabled=config.USE_AMP):
            type_loss = F.cross_entropy(type_logits, rice_types)
            total_loss = loss + 0.1 * type_loss
        
        # Scale for gradient accumulation
        total_loss = total_loss / accumulation_steps
        
        scaler.scale(total_loss).backward()
        
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update EMA
            if ema is not None:
                ema.update()
        
        running_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, normalizer, config):
    model.eval()
    all_preds = []
    all_targets = []
    
    for images, targets, rice_types in tqdm(loader, desc="Validating"):
        images = images.to(config.DEVICE)
        rice_types = rice_types.to(config.DEVICE)
        
        with autocast(enabled=config.USE_AMP):
            predictions, _ = model(images, rice_types)
        
        all_preds.append(predictions.cpu().numpy())
        all_targets.append(targets.numpy())
    
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # Denormalize
    preds_denorm = normalizer.inverse_transform(preds)
    targets_denorm = normalizer.inverse_transform(targets)
    
    # Calculate per-target MAE
    mae_per_target = np.abs(preds_denorm - targets_denorm).mean(axis=0)
    overall_mae = mae_per_target.mean()
    
    return overall_mae, mae_per_target


def print_metrics(mae_per_target, target_cols, title=""):
    print(f"\n{title}")
    print("Target                           MAE")
    print("───────────────────────────────────")
    
    for col, mae in zip(target_cols, mae_per_target):
        flag = "⚠️" if mae > 50 else "  "
        print(f"{flag} {col:25s} {mae:8.2f}")
    
    print("───────────────────────────────────")
    print(f"Overall Mean                   {mae_per_target.mean():.2f}")


# ============================================================================
# Main Training Loop
# ============================================================================
def main():
    print("="*72)
    print("🌾 RICE QUALITY ASSESSMENT - V3 IMPROVED (CONVNEXT + MIXUP + EMA)")
    print("="*72)
    print(f"Device: {Config.DEVICE}")
    print(f"Backbone: {Config.BACKBONE}")
    print(f"Image size: {Config.IMG_SIZE}")
    print(f"Effective batch size: {Config.BATCH_SIZE * Config.ACCUMULATION_STEPS}")
    print(f"Mixup: {Config.USE_MIXUP} (alpha={Config.MIXUP_ALPHA})")
    print(f"EMA: {Config.USE_EMA} (decay={Config.EMA_DECAY})")
    
    set_seed(Config.SEED)
    
    # Create output directories
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
    
    # Training loop
    fold_results = []
    
    for fold in Config.FOLD_TO_RUN:
        print(f"\n{'='*72}")
        print(f"FOLD {fold}")
        print(f"{'='*72}")
        
        # Split data
        train_fold = train_df[train_df['fold'] != fold].reset_index(drop=True)
        valid_fold = train_df[train_df['fold'] == fold].reset_index(drop=True)
        print(f"Train: {len(train_fold)}, Valid: {len(valid_fold)}")
        
        # Fit normalizer on training data
        normalizer = TargetNormalizer()
        train_targets = train_fold[Config.TARGET_COLS].values.astype(np.float32)
        normalizer.fit(train_targets)
        
        print("\n📊 Target Normalization Statistics:")
        print(f"{'Target':35s} {'Mean':>12s} {'Std':>12s}")
        print("─" * 60)
        for i, col in enumerate(Config.TARGET_COLS):
            print(f"{col:35s} {normalizer.means[i]:12.2f} {normalizer.stds[i]:12.2f}")
        
        # Create datasets
        train_dataset = RiceDataset(
            train_fold, Config.IMAGE_DIR, 
            transform=get_train_transforms(Config.IMG_SIZE),
            normalizer=normalizer, is_train=True
        )
        valid_dataset = RiceDataset(
            valid_fold, Config.IMAGE_DIR,
            transform=get_valid_transforms(Config.IMG_SIZE),
            normalizer=normalizer, is_train=True
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
        model = RiceModelV3(
            Config.BACKBONE, 
            num_targets=len(Config.TARGET_COLS),
            pretrained=Config.PRETRAINED
        ).to(Config.DEVICE)
        
        print(f"\n🔧 Model: {Config.BACKBONE}")
        print(f"   Feature dim: {model.feat_dim}")
        
        # EMA
        ema = EMA(model, decay=Config.EMA_DECAY) if Config.USE_EMA else None
        
        # Loss with count weighting
        criterion = WeightedMultiTargetLoss(
            Config.TARGET_COLS, Config.COUNT_TARGETS, Config.COUNT_WEIGHT
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, 
                         weight_decay=Config.WEIGHT_DECAY)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        scaler = GradScaler(enabled=Config.USE_AMP)
        
        # Training
        best_mae = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(1, Config.EPOCHS + 1):
            print(f"\n{'─'*72}")
            print(f"Fold {fold} | Epoch {epoch}/{Config.EPOCHS}")
            print(f"{'─'*72}")
            
            # Train
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, ema,
                Config, Config.ACCUMULATION_STEPS
            )
            
            # Apply EMA weights for validation
            if ema is not None:
                ema.apply_shadow()
            
            # Validate
            val_mae, mae_per_target = validate(model, valid_loader, normalizer, Config)
            
            # Restore original weights
            if ema is not None:
                ema.restore()
            
            scheduler.step()
            
            # Print results
            is_best = val_mae < best_mae
            if is_best:
                best_mae = val_mae
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'ema_shadow': ema.shadow if ema else None,
                    'normalizer_means': normalizer.means,
                    'normalizer_stds': normalizer.stds,
                    'best_mae': best_mae,
                    'epoch': epoch,
                    'fold': fold
                }
                torch.save(save_dict, Config.OUTPUT_DIR / f"best_model_fold{fold}.pt")
                print(f"🏆 NEW BEST! MAE: {val_mae:.2f}")
            else:
                patience_counter += 1
            
            print(f"Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.2f}")
            print_metrics(mae_per_target, Config.TARGET_COLS)
            
            # Early stopping
            if patience_counter >= 7:
                print(f"\n⚠️ Early stopping at epoch {epoch}")
                break
        
        print(f"\n✅ Fold {fold} complete! Best MAE: {best_mae:.2f} at epoch {best_epoch}")
        fold_results.append({'fold': fold, 'best_mae': best_mae, 'best_epoch': best_epoch})
    
    # Summary
    print("\n" + "="*72)
    print("🏆 TRAINING COMPLETE")
    print("="*72)
    for r in fold_results:
        print(f"Fold {r['fold']}: Best MAE = {r['best_mae']:.2f} (epoch {r['best_epoch']})")
    
    avg_mae = np.mean([r['best_mae'] for r in fold_results])
    print(f"\n📊 Average MAE across folds: {avg_mae:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='V3 Rice Quality Training')
    parser.add_argument('--data-dir', type=str, default='RiceData/Unido_AfricaRice_Challenge',
                        help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='outputs/v3_improved',
                        help='Path to output directory')
    parser.add_argument('--model', type=str, default='convnext_base.fb_in22k_ft_in1k',
                        help='Model backbone name from timm')
    parser.add_argument('--img-size', type=int, default=512,
                        help='Image size for training')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--folds', type=str, default='0',
                        help='Comma-separated fold numbers to run (e.g., "0,1,2")')
    args = parser.parse_args()
    
    # Set config from args
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
    
    main()
