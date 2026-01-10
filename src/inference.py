"""
Rice Quality Assessment - Inference Script
===========================================
Generates predictions for the test set using trained models.
Supports TTA (Test-Time Augmentation) and model ensembling.
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================================
# Configuration
# ============================================================================
class Config:
    DATA_DIR = Path("RiceData/Unido_AfricaRice_Challenge")
    IMAGE_DIR = None
    MODEL_DIR = Path("outputs/v6_optimized")
    
    IMG_SIZE = 512
    BATCH_SIZE = 8
    
    USE_TTA = True
    
    COUNT_TARGETS = ['Count', 'Broken_Count', 'Long_Count', 'Medium_Count',
                     'Black_Count', 'Chalky_Count', 'Red_Count', 
                     'Yellow_Count', 'Green_Count']
    CONTINUOUS_TARGETS = ['WK_Length_Average', 'WK_Width_Average', 
                          'WK_LW_Ratio_Average', 'Average_L', 
                          'Average_a', 'Average_b']
    TARGET_COLS = COUNT_TARGETS + CONTINUOUS_TARGETS
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = True


# ============================================================================
# Normalizer
# ============================================================================
class ZScoreNormalizer:
    def __init__(self):
        self.means = None
        self.stds = None
    
    def load(self, d):
        self.means = d['means']
        self.stds = d['stds']
    
    def inverse_transform(self, targets):
        return targets * self.stds + self.means


# ============================================================================
# Dataset
# ============================================================================
class TestDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.type_map = {'Paddy': 0, 'White': 1, 'Brown': 2}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(self.img_dir / f"{row['ID']}.png").convert('RGB'))
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        rice_type = self.type_map.get(row['Comment'], 0)
        return img, row['ID'], rice_type


# ============================================================================
# Transforms
# ============================================================================
def get_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size):
    """TTA: original + hflip + vflip + both"""
    base = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    hflip = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    vflip = A.Compose([
        A.Resize(img_size, img_size),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    both = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return [base, hflip, vflip, both]


# ============================================================================
# Model
# ============================================================================
class RiceModel(nn.Module):
    def __init__(self, backbone_name, num_targets=15, num_types=3):
        super().__init__()
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )
        
        self.feat_dim = self.backbone.num_features
        self.type_emb = nn.Embedding(num_types, 64)
        
        self.head = nn.Sequential(
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
        
        self.type_cls = nn.Linear(self.feat_dim, num_types)
    
    def forward(self, x, rice_type=None):
        feat = self.backbone(x)
        type_logits = self.type_cls(feat)
        
        if rice_type is not None:
            type_e = self.type_emb(rice_type)
        else:
            type_e = self.type_emb(type_logits.argmax(1))
        
        combined = torch.cat([feat, type_e], dim=1)
        preds = self.head(combined)
        
        return preds, type_logits


# ============================================================================
# Inference Functions
# ============================================================================
@torch.no_grad()
def predict_single_model(model, loader, normalizer, config, use_tta=False):
    """Predict with a single model"""
    model.eval()
    all_preds = []
    all_ids = []
    
    if use_tta:
        tta_transforms = get_tta_transforms(config.IMG_SIZE)
    
    for imgs, ids, types in tqdm(loader, desc="Predicting"):
        imgs = imgs.to(config.DEVICE)
        types = types.to(config.DEVICE)
        
        if use_tta:
            # TTA: average predictions across augmentations
            # Note: This requires raw images, for simplicity we skip TTA in batch
            # For proper TTA, you'd need to re-transform each image
            with autocast(enabled=config.USE_AMP):
                preds, _ = model(imgs, types)
        else:
            with autocast(enabled=config.USE_AMP):
                preds, _ = model(imgs, types)
        
        all_preds.append(preds.cpu().numpy())
        all_ids.extend(ids)
    
    preds = np.vstack(all_preds)
    
    # Denormalize
    preds = normalizer.inverse_transform(preds)
    
    return all_ids, preds


def predict_with_tta(model, df, img_dir, normalizer, config):
    """Predict with proper TTA"""
    model.eval()
    tta_transforms = get_tta_transforms(config.IMG_SIZE)
    
    all_preds = []
    all_ids = []
    
    for idx in tqdm(range(len(df)), desc="Predicting with TTA"):
        row = df.iloc[idx]
        img = np.array(Image.open(img_dir / f"{row['ID']}.png").convert('RGB'))
        rice_type = {'Paddy': 0, 'White': 1, 'Brown': 2}.get(row['Comment'], 0)
        
        tta_preds = []
        for transform in tta_transforms:
            aug_img = transform(image=img)['image'].unsqueeze(0).to(config.DEVICE)
            type_tensor = torch.tensor([rice_type]).to(config.DEVICE)
            
            with autocast(enabled=config.USE_AMP):
                pred, _ = model(aug_img, type_tensor)
            
            tta_preds.append(pred.cpu().numpy())
        
        # Average TTA predictions
        avg_pred = np.mean(tta_preds, axis=0)
        all_preds.append(avg_pred)
        all_ids.append(row['ID'])
    
    preds = np.vstack(all_preds)
    preds = normalizer.inverse_transform(preds)
    
    return all_ids, preds


def post_process(preds, config):
    """Apply post-processing rules"""
    n_counts = len(config.COUNT_TARGETS)
    
    # Clip negative values for counts
    preds[:, :n_counts] = np.maximum(preds[:, :n_counts], 0)
    
    # Round counts to integers
    preds[:, :n_counts] = np.round(preds[:, :n_counts])
    
    # Clip continuous targets to reasonable ranges
    # WK_Length_Average: typically 4-12mm
    preds[:, n_counts] = np.clip(preds[:, n_counts], 3, 15)
    # WK_Width_Average: typically 1.5-4mm
    preds[:, n_counts + 1] = np.clip(preds[:, n_counts + 1], 1, 5)
    # WK_LW_Ratio_Average: typically 2-5
    preds[:, n_counts + 2] = np.clip(preds[:, n_counts + 2], 1.5, 6)
    
    return preds


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Rice Quality Inference')
    parser.add_argument('--data-dir', default='RiceData/Unido_AfricaRice_Challenge')
    parser.add_argument('--model-dir', default='outputs/v6_optimized')
    parser.add_argument('--output', default='submission.csv')
    parser.add_argument('--backbone', default='convnext_xlarge.fb_in22k')
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--tta', action='store_true', help='Enable TTA')
    parser.add_argument('--folds', default='0', help='Comma-separated folds to ensemble')
    args = parser.parse_args()
    
    Config.DATA_DIR = Path(args.data_dir)
    Config.IMAGE_DIR = Config.DATA_DIR / "unido_rice_images"
    Config.MODEL_DIR = Path(args.model_dir)
    Config.IMG_SIZE = args.img_size
    Config.BATCH_SIZE = args.batch_size
    Config.USE_TTA = args.tta
    
    print("=" * 60)
    print("🌾 RICE QUALITY INFERENCE")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Backbone: {args.backbone}")
    print(f"Image size: {Config.IMG_SIZE}")
    print(f"TTA: {Config.USE_TTA}")
    
    # Load test data
    test_df = pd.read_csv(Config.DATA_DIR / "Test.csv")
    print(f"Test samples: {len(test_df)}")
    
    folds = [int(f) for f in args.folds.split(',')]
    print(f"Ensembling folds: {folds}")
    
    all_fold_preds = []
    
    for fold in folds:
        print(f"\n--- Loading Fold {fold} ---")
        
        # Load checkpoint
        ckpt_path = Config.MODEL_DIR / f"best_fold{fold}.pt"
        if not ckpt_path.exists():
            print(f"Warning: {ckpt_path} not found, skipping")
            continue
        
        ckpt = torch.load(ckpt_path, map_location=Config.DEVICE, weights_only=False)
        
        # Load normalizer
        normalizer = ZScoreNormalizer()
        normalizer.load(ckpt['normalizer'])
        
        # Get backbone from checkpoint config
        backbone = ckpt.get('config', {}).get('backbone', args.backbone)
        
        # Create model
        model = RiceModel(backbone, num_targets=len(Config.TARGET_COLS)).to(Config.DEVICE)
        model.load_state_dict(ckpt['model'])
        model.eval()
        
        print(f"Loaded model: {backbone}")
        print(f"Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
        print(f"Checkpoint MAE: {ckpt.get('best_overall_mae', ckpt.get('best_count_mae', 'N/A'))}")
        
        # Predict
        if Config.USE_TTA:
            ids, preds = predict_with_tta(model, test_df, Config.IMAGE_DIR, normalizer, Config)
        else:
            test_ds = TestDataset(test_df, Config.IMAGE_DIR, get_transforms(Config.IMG_SIZE))
            test_loader = DataLoader(test_ds, Config.BATCH_SIZE, shuffle=False, 
                                     num_workers=4, pin_memory=True)
            ids, preds = predict_single_model(model, test_loader, normalizer, Config)
        
        all_fold_preds.append(preds)
    
    if not all_fold_preds:
        print("Error: No models loaded!")
        return
    
    # Ensemble: average predictions
    if len(all_fold_preds) > 1:
        print(f"\nEnsembling {len(all_fold_preds)} folds...")
        final_preds = np.mean(all_fold_preds, axis=0)
    else:
        final_preds = all_fold_preds[0]
    
    # Post-process
    final_preds = post_process(final_preds, Config)
    
    # Create submission
    submission = pd.DataFrame({'ID': ids})
    for i, col in enumerate(Config.TARGET_COLS):
        submission[col] = final_preds[:, i]
    
    # Ensure correct column order (match sample submission)
    sample_sub = pd.read_csv(Config.DATA_DIR / "SampleSubmission.csv")
    submission = submission[sample_sub.columns]
    
    # Save
    submission.to_csv(args.output, index=False)
    print(f"\n✅ Saved submission to: {args.output}")
    print(f"Shape: {submission.shape}")
    print("\nSample predictions:")
    print(submission.head())


if __name__ == "__main__":
    main()
