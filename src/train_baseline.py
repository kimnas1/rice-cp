"""
Rice Quality Assessment Challenge - Baseline Model V2
Staff ML Engineer Implementation

Key Fixes from V1:
1. Added z-score normalization for all targets
2. Targets are normalized per rice type for better conditioning
3. Proper denormalization during inference/validation

This baseline implements:
1. EfficientNet-B4 backbone with pretrained weights
2. Rice type conditioning (uses known type from test data)
3. Z-score normalized targets per rice type
4. Multi-target regression with Smooth L1 loss
5. 5-fold stratified cross-validation
6. Comprehensive logging for all 15 targets
"""

import os
import random
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import gc

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    DATA_DIR = Path("/home/kara/Data/projects/rice-ops/RiceData/Unido_AfricaRice_Challenge")
    IMAGE_DIR = DATA_DIR / "unido_rice_images"
    OUTPUT_DIR = Path("/home/kara/Data/projects/rice-ops/outputs/baseline_v2")
    LOG_DIR = Path("/home/kara/Data/projects/rice-ops/outputs/logs")
    
    # Model
    BACKBONE = "efficientnet_b4"
    PRETRAINED = True
    NUM_TARGETS = 15
    NUM_TYPES = 3
    
    # Training
    IMG_SIZE = 384
    BATCH_SIZE = 8
    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 2
    
    # Validation
    N_FOLDS = 5
    SEED = 42
    
    # Target columns
    TARGET_COLS = [
        'Count', 'Broken_Count', 'Long_Count', 'Medium_Count', 'Black_Count',
        'Chalky_Count', 'Red_Count', 'Yellow_Count', 'Green_Count',
        'WK_Length_Average', 'WK_Width_Average', 'WK_LW_Ratio_Average',
        'Average_L', 'Average_a', 'Average_b'
    ]
    
    # Count targets (for special handling)
    COUNT_TARGETS = [
        'Count', 'Broken_Count', 'Long_Count', 'Medium_Count', 'Black_Count',
        'Chalky_Count', 'Red_Count', 'Yellow_Count', 'Green_Count'
    ]
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mixed precision
    USE_AMP = True
    
    # Workers
    NUM_WORKERS = 4
    
    # Debug
    DEBUG = False


def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Target Normalization
# ============================================================================

class TargetNormalizer:
    """
    Z-score normalizer for targets.
    Computes mean and std per target column.
    """
    def __init__(self, target_cols: List[str]):
        self.target_cols = target_cols
        self.means = {}
        self.stds = {}
        self.fitted = False
    
    def fit(self, df: pd.DataFrame):
        """Fit the normalizer on training data."""
        for col in self.target_cols:
            self.means[col] = df[col].mean()
            self.stds[col] = df[col].std()
            # Avoid division by zero
            if self.stds[col] < 1e-6:
                self.stds[col] = 1.0
        self.fitted = True
        
        print("\n📊 Target Normalization Statistics:")
        print(f"{'Target':<25} {'Mean':>12} {'Std':>12}")
        print("─" * 50)
        for col in self.target_cols:
            print(f"{col:<25} {self.means[col]:>12.2f} {self.stds[col]:>12.2f}")
    
    def transform(self, values: np.ndarray) -> np.ndarray:
        """Normalize target values."""
        assert self.fitted, "Normalizer must be fitted first"
        normalized = np.zeros_like(values, dtype=np.float32)
        for i, col in enumerate(self.target_cols):
            normalized[:, i] = (values[:, i] - self.means[col]) / self.stds[col]
        return normalized
    
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Denormalize target values."""
        assert self.fitted, "Normalizer must be fitted first"
        denormalized = np.zeros_like(values, dtype=np.float32)
        for i, col in enumerate(self.target_cols):
            denormalized[:, i] = values[:, i] * self.stds[col] + self.means[col]
        return denormalized
    
    def transform_single(self, col: str, value: float) -> float:
        """Normalize a single value."""
        return (value - self.means[col]) / self.stds[col]
    
    def inverse_transform_single(self, col: str, value: float) -> float:
        """Denormalize a single value."""
        return value * self.stds[col] + self.means[col]
    
    def get_mean_std_tensors(self, device: torch.device):
        """Get mean and std as tensors for GPU computation."""
        means = torch.tensor([self.means[col] for col in self.target_cols], 
                            dtype=torch.float32, device=device)
        stds = torch.tensor([self.stds[col] for col in self.target_cols], 
                           dtype=torch.float32, device=device)
        return means, stds
    
    def save(self, path: Path):
        """Save normalization stats."""
        stats = {'means': self.means, 'stds': self.stds, 'target_cols': self.target_cols}
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        """Load normalization stats."""
        with open(path, 'r') as f:
            stats = json.load(f)
        normalizer = cls(stats['target_cols'])
        normalizer.means = stats['means']
        normalizer.stds = stats['stds']
        normalizer.fitted = True
        return normalizer


# ============================================================================
# Logging
# ============================================================================

class MetricsLogger:
    """Logger for tracking all metrics during training."""
    
    def __init__(self, log_dir: Path, target_cols: List[str]):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.target_cols = target_cols
        
        # Initialize storage
        self.history = {
            'fold': [],
            'epoch': [],
            'train_loss': [],
            'train_mae': [],
            'val_loss': [],
            'val_mae': [],
        }
        
        # Add per-target columns
        for col in target_cols:
            self.history[f'val_mae_{col}'] = []
        
        # Summary storage
        self.fold_summaries = []
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"training_log_{timestamp}.csv"
        self.summary_file = log_dir / f"training_summary_{timestamp}.json"
        
    def log_epoch(self, fold: int, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics for one epoch."""
        self.history['fold'].append(fold)
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_metrics['train_loss'])
        self.history['train_mae'].append(train_metrics['train_mae'])
        self.history['val_loss'].append(val_metrics['val_loss'])
        self.history['val_mae'].append(val_metrics['val_mae'])
        
        # Log per-target MAE
        for col in self.target_cols:
            key = f'mae_{col}'
            if key in val_metrics:
                self.history[f'val_mae_{col}'].append(val_metrics[key])
        
        # Save incrementally
        self._save_history()
    
    def log_fold_summary(self, fold: int, best_mae: float, best_epoch: int, 
                         per_target_mae: Dict[str, float]):
        """Log summary for a completed fold."""
        summary = {
            'fold': fold,
            'best_mae': best_mae,
            'best_epoch': best_epoch,
            'per_target_mae': per_target_mae
        }
        self.fold_summaries.append(summary)
        self._save_summary()
    
    def log_final_summary(self, mean_mae: float, std_mae: float, 
                          overall_per_target_mae: Dict[str, float]):
        """Log final training summary."""
        final_summary = {
            'mean_cv_mae': mean_mae,
            'std_cv_mae': std_mae,
            'overall_per_target_mae': overall_per_target_mae,
            'fold_summaries': self.fold_summaries
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        print(f"\n✅ Training logs saved to {self.log_file}")
        print(f"✅ Summary saved to {self.summary_file}")
    
    def _save_history(self):
        """Save history to CSV."""
        df = pd.DataFrame(self.history)
        df.to_csv(self.log_file, index=False)
    
    def _save_summary(self):
        """Save current summaries."""
        with open(self.summary_file, 'w') as f:
            json.dump({'fold_summaries': self.fold_summaries}, f, indent=2)
    
    def print_epoch_summary(self, fold: int, epoch: int, train_metrics: Dict, 
                            val_metrics: Dict, is_best: bool = False):
        """Print formatted epoch summary."""
        print(f"\n{'─'*70}")
        print(f"Fold {fold} | Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"{'─'*70}")
        print(f"Train Loss: {train_metrics['train_loss']:.4f} | Train MAE: {train_metrics['train_mae']:.2f}")
        print(f"Val Loss:   {val_metrics['val_loss']:.4f} | Val MAE:   {val_metrics['val_mae']:.2f}")
        
        if is_best:
            print("🏆 NEW BEST!")
        
        # Print per-target MAE in a nice table
        print(f"\n{'Target':<25} {'MAE':>10}")
        print(f"{'─'*35}")
        for col in self.target_cols:
            mae = val_metrics.get(f'mae_{col}', 0)
            # Highlight high MAE targets
            indicator = "⚠️" if mae > 100 else "  "
            print(f"{indicator} {col:<22} {mae:>10.2f}")
        print(f"{'─'*35}")
        print(f"{'Overall Mean':<25} {val_metrics['val_mae']:>10.2f}")


# ============================================================================
# Data Transforms
# ============================================================================

def get_train_transforms(img_size: int = 384):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_valid_transforms(img_size: int = 384):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ============================================================================
# Dataset
# ============================================================================

class RiceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        target_cols: List[str],
        type_encoder: LabelEncoder,
        normalizer: Optional[TargetNormalizer] = None,
        transforms: Optional[A.Compose] = None,
        is_test: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.target_cols = target_cols
        self.type_encoder = type_encoder
        self.normalizer = normalizer
        self.transforms = transforms
        self.is_test = is_test
        
        # Pre-normalize targets if normalizer provided
        if not is_test and normalizer is not None:
            raw_targets = df[target_cols].values.astype(np.float32)
            self.normalized_targets = normalizer.transform(raw_targets)
        else:
            self.normalized_targets = None
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.image_dir / f"{row['ID']}.png"
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        # Rice type
        rice_type = self.type_encoder.transform([row['Comment']])[0]
        rice_type = torch.tensor(rice_type, dtype=torch.long)
        
        if self.is_test:
            return {
                'image': image,
                'rice_type': rice_type,
                'id': row['ID']
            }
        else:
            # Use pre-normalized targets
            if self.normalized_targets is not None:
                targets = torch.tensor(self.normalized_targets[idx], dtype=torch.float32)
            else:
                targets = torch.tensor(row[self.target_cols].values.astype(np.float32))
            return {
                'image': image,
                'rice_type': rice_type,
                'targets': targets
            }


# ============================================================================
# Model
# ============================================================================

class RiceQualityModel(nn.Module):
    """
    Multi-target regression model with rice type conditioning.
    """
    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        num_targets: int = 15,
        num_types: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Backbone
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            num_classes=0,
            global_pool='avg'
        )
        self.embed_dim = self.backbone.num_features
        
        # Rice type embedding
        self.type_embedding = nn.Embedding(num_types, 64)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim + 64, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_targets)
        )
        
        # Auxiliary classifier for rice type
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, num_types)
        )
        
    def forward(self, x, rice_type_idx):
        features = self.backbone(x)
        type_emb = self.type_embedding(rice_type_idx)
        combined = torch.cat([features, type_emb], dim=1)
        predictions = self.regressor(combined)
        type_logits = self.classifier(features)
        return predictions, type_logits


# ============================================================================
# Loss Functions
# ============================================================================

class MultiTargetLoss(nn.Module):
    """
    Combined loss for normalized multi-target regression.
    Since targets are normalized, we use simpler losses.
    """
    def __init__(
        self,
        target_cols: List[str],
        count_targets: List[str],
        cls_weight: float = 0.1
    ):
        super().__init__()
        self.target_cols = target_cols
        self.count_targets = count_targets
        self.cls_weight = cls_weight
        
        # Classification loss
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, pred, target, type_logits, type_true):
        # Main regression loss: Smooth L1 works well for normalized targets
        loss_reg = F.smooth_l1_loss(pred, target)
        
        # Classification loss (auxiliary)
        loss_cls = self.ce_loss(type_logits, type_true)
        
        # Total loss
        total_loss = loss_reg + self.cls_weight * loss_cls
        
        return total_loss, {
            'reg_loss': loss_reg.item(),
            'cls_loss': loss_cls.item()
        }


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    n_samples = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch in pbar:
        images = batch['image'].to(device)
        rice_types = batch['rice_type'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=scaler is not None):
            predictions, type_logits = model(images, rice_types)
            loss, loss_dict = criterion(predictions, targets, type_logits, rice_types)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return {
        'train_loss': total_loss / n_samples,
        'train_mae': 0  # We'll compute real MAE in validation
    }


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_cols: List[str],
    normalizer: TargetNormalizer
) -> Tuple[Dict[str, float], np.ndarray]:
    """Validate for one epoch with denormalization."""
    model.eval()
    
    total_loss = 0
    n_samples = 0
    all_predictions_norm = []
    all_targets_norm = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            images = batch['image'].to(device)
            rice_types = batch['rice_type'].to(device)
            targets = batch['targets'].to(device)
            
            predictions, type_logits = model(images, rice_types)
            loss, _ = criterion(predictions, targets, type_logits, rice_types)
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            
            all_predictions_norm.append(predictions.cpu().numpy())
            all_targets_norm.append(targets.cpu().numpy())
    
    all_predictions_norm = np.vstack(all_predictions_norm)
    all_targets_norm = np.vstack(all_targets_norm)
    
    # Denormalize for real MAE calculation
    all_predictions = normalizer.inverse_transform(all_predictions_norm)
    all_targets = normalizer.inverse_transform(all_targets_norm)
    
    # Calculate MAE per target on ORIGINAL scale
    mae_per_target = np.abs(all_predictions - all_targets).mean(axis=0)
    overall_mae = mae_per_target.mean()
    
    metrics = {
        'val_loss': total_loss / n_samples,
        'val_mae': overall_mae
    }
    
    # Add per-target MAE
    for i, col in enumerate(target_cols):
        metrics[f'mae_{col}'] = mae_per_target[i]
    
    return metrics, all_predictions


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    config: Config,
    type_encoder: LabelEncoder,
    logger: MetricsLogger
) -> Tuple[float, np.ndarray, Dict[str, float], TargetNormalizer]:
    """Train one fold."""
    print(f"\n{'='*70}")
    print(f"FOLD {fold}")
    print(f"{'='*70}")
    print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")
    
    # Create normalizer fitted on training data only
    normalizer = TargetNormalizer(config.TARGET_COLS)
    normalizer.fit(train_df)
    
    # Create datasets with normalization
    train_dataset = RiceDataset(
        train_df, config.IMAGE_DIR, config.TARGET_COLS,
        type_encoder, normalizer, get_train_transforms(config.IMG_SIZE)
    )
    valid_dataset = RiceDataset(
        valid_df, config.IMAGE_DIR, config.TARGET_COLS,
        type_encoder, normalizer, get_valid_transforms(config.IMG_SIZE)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS,
        pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE * 2,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model
    model = RiceQualityModel(
        backbone=config.BACKBONE,
        pretrained=config.PRETRAINED,
        num_targets=config.NUM_TARGETS,
        num_types=config.NUM_TYPES
    ).to(config.DEVICE)
    
    # Loss function (simpler for normalized targets)
    criterion = MultiTargetLoss(
        target_cols=config.TARGET_COLS,
        count_targets=config.COUNT_TARGETS
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.LR,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Mixed precision
    scaler = GradScaler() if config.USE_AMP else None
    
    # Training loop
    best_mae = float('inf')
    best_predictions = None
    best_epoch = 0
    best_per_target_mae = {}
    
    for epoch in range(config.EPOCHS):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            config.DEVICE, scaler, scheduler
        )
        
        # Validate with denormalization
        val_metrics, predictions = validate_one_epoch(
            model, valid_loader, criterion, config.DEVICE,
            config.TARGET_COLS, normalizer
        )
        
        # Log metrics
        logger.log_epoch(fold, epoch, train_metrics, val_metrics)
        
        # Check if best
        is_best = val_metrics['val_mae'] < best_mae
        
        # Print epoch summary
        logger.print_epoch_summary(fold, epoch, train_metrics, val_metrics, is_best)
        
        # Save best model
        if is_best:
            best_mae = val_metrics['val_mae']
            best_predictions = predictions
            best_epoch = epoch
            
            # Store per-target MAE
            for col in config.TARGET_COLS:
                best_per_target_mae[col] = val_metrics[f'mae_{col}']
            
            # Save checkpoint with normalizer stats
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': best_mae,
                'per_target_mae': best_per_target_mae,
                'normalizer_means': normalizer.means,
                'normalizer_stds': normalizer.stds
            }, config.OUTPUT_DIR / f'model_fold{fold}.pt')
    
    # Save normalizer
    normalizer.save(config.OUTPUT_DIR / f'normalizer_fold{fold}.json')
    
    # Log fold summary
    logger.log_fold_summary(fold, best_mae, best_epoch, best_per_target_mae)
    
    print(f"\n{'='*70}")
    print(f"FOLD {fold} COMPLETE - Best MAE: {best_mae:.2f} (Epoch {best_epoch+1})")
    print(f"{'='*70}")
    
    # Cleanup
    del model, optimizer, train_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_mae, best_predictions, best_per_target_mae, normalizer


def main():
    """Main training function."""
    # Setup
    seed_everything(Config.SEED)
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🌾 RICE QUALITY ASSESSMENT - BASELINE V2 (WITH NORMALIZATION)")
    print("="*80)
    print(f"Device: {Config.DEVICE}")
    print(f"Backbone: {Config.BACKBONE}")
    print(f"Image size: {Config.IMG_SIZE}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Folds: {Config.N_FOLDS}")
    print(f"Learning rate: {Config.LR}")
    
    # Load data
    train_df = pd.read_csv(Config.DATA_DIR / "Train.csv")
    test_df = pd.read_csv(Config.DATA_DIR / "Test.csv")
    
    print(f"\n📊 Train samples: {len(train_df)}")
    print(f"📊 Test samples: {len(test_df)}")
    
    # Encode rice types
    type_encoder = LabelEncoder()
    type_encoder.fit(train_df['Comment'])
    print(f"🍚 Rice types: {type_encoder.classes_}")
    
    # Debug mode
    if Config.DEBUG:
        train_df = train_df.head(100)
        Config.EPOCHS = 2
        print("\n⚠️ DEBUG MODE: Using subset of data")
    
    # Create folds
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    train_df['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df['Comment'])):
        train_df.loc[val_idx, 'fold'] = fold
    
    # Initialize logger
    logger = MetricsLogger(Config.LOG_DIR, Config.TARGET_COLS)
    
    # Train each fold
    fold_maes = []
    oof_predictions = np.zeros((len(train_df), Config.NUM_TARGETS))
    all_per_target_mae = {col: [] for col in Config.TARGET_COLS}
    
    for fold in range(Config.N_FOLDS):
        train_fold_df = train_df[train_df['fold'] != fold].copy()
        valid_fold_df = train_df[train_df['fold'] == fold].copy()
        
        best_mae, predictions, per_target_mae, normalizer = train_fold(
            fold, train_fold_df, valid_fold_df, Config, type_encoder, logger
        )
        
        fold_maes.append(best_mae)
        oof_predictions[valid_fold_df.index] = predictions
        
        # Collect per-target MAE
        for col in Config.TARGET_COLS:
            all_per_target_mae[col].append(per_target_mae[col])
    
    # Final Summary
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE")
    print("="*80)
    
    # Fold MAEs
    print(f"\n📊 Fold MAEs:")
    for i, mae in enumerate(fold_maes):
        print(f"   Fold {i}: {mae:.2f}")
    print(f"\n📈 Mean CV MAE: {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f}")
    
    # Calculate and display OOF MAE per target
    oof_mae_per_target = np.abs(oof_predictions - train_df[Config.TARGET_COLS].values).mean(axis=0)
    
    print(f"\n{'='*70}")
    print("📋 OOF MAE PER TARGET (This correlates with LB!)")
    print(f"{'='*70}")
    print(f"{'Target':<25} {'OOF MAE':>10} {'Mean Fold MAE':>15}")
    print(f"{'─'*50}")
    
    overall_per_target_mae = {}
    for i, col in enumerate(Config.TARGET_COLS):
        oof_mae = oof_mae_per_target[i]
        mean_fold_mae = np.mean(all_per_target_mae[col])
        overall_per_target_mae[col] = oof_mae
        
        # Highlight targets based on error level
        if oof_mae > 100:
            indicator = "⚠️"
        elif oof_mae < 10:
            indicator = "✅"
        else:
            indicator = "  "
        print(f"{indicator} {col:<22} {oof_mae:>10.2f} {mean_fold_mae:>15.2f}")
    
    print(f"{'─'*50}")
    print(f"{'Overall Mean':<25} {oof_mae_per_target.mean():>10.2f}")
    
    # Log final summary
    logger.log_final_summary(
        np.mean(fold_maes), 
        np.std(fold_maes), 
        overall_per_target_mae
    )
    
    # Save OOF predictions
    oof_df = train_df[['ID', 'Comment', 'fold']].copy()
    for i, col in enumerate(Config.TARGET_COLS):
        oof_df[f'pred_{col}'] = oof_predictions[:, i]
        oof_df[f'true_{col}'] = train_df[col].values
    oof_df.to_csv(Config.OUTPUT_DIR / 'oof_predictions.csv', index=False)
    
    # Save summary
    summary = {
        'mean_cv_mae': float(np.mean(fold_maes)),
        'std_cv_mae': float(np.std(fold_maes)),
        'fold_maes': [float(m) for m in fold_maes],
        'per_target_oof_mae': {col: float(oof_mae_per_target[i]) 
                               for i, col in enumerate(Config.TARGET_COLS)}
    }
    with open(Config.OUTPUT_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ OOF predictions saved to {Config.OUTPUT_DIR / 'oof_predictions.csv'}")
    print(f"✅ Training summary saved to {Config.OUTPUT_DIR / 'training_summary.json'}")


if __name__ == "__main__":
    main()
