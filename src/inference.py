"""
Rice Quality Assessment Challenge - Inference Script
Generates predictions for test set with TTA

Usage:
    python src/inference.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    DATA_DIR = Path("/home/kara/Data/projects/rice-ops/RiceData/Unido_AfricaRice_Challenge")
    IMAGE_DIR = DATA_DIR / "unido_rice_images"
    MODEL_DIR = Path("/home/kara/Data/projects/rice-ops/outputs/baseline")
    OUTPUT_DIR = Path("/home/kara/Data/projects/rice-ops/submissions")
    
    # Model
    BACKBONE = "efficientnet_b4"
    NUM_TARGETS = 15
    NUM_TYPES = 3
    
    # Inference
    IMG_SIZE = 384
    BATCH_SIZE = 8
    N_FOLDS = 5
    USE_TTA = True
    N_TTA = 4
    
    # Target columns
    TARGET_COLS = [
        'Count', 'Broken_Count', 'Long_Count', 'Medium_Count', 'Black_Count',
        'Chalky_Count', 'Red_Count', 'Yellow_Count', 'Green_Count',
        'WK_Length_Average', 'WK_Width_Average', 'WK_LW_Ratio_Average',
        'Average_L', 'Average_a', 'Average_b'
    ]
    
    # Count targets (for post-processing)
    COUNT_TARGETS = [
        'Count', 'Broken_Count', 'Long_Count', 'Medium_Count', 'Black_Count',
        'Chalky_Count', 'Red_Count', 'Yellow_Count', 'Green_Count'
    ]
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Workers
    NUM_WORKERS = 4


# ============================================================================
# Model
# ============================================================================

class RiceQualityModel(nn.Module):
    """Same model architecture as training."""
    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = False,
        num_targets: int = 15,
        num_types: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            num_classes=0,
            global_pool='avg'
        )
        self.embed_dim = self.backbone.num_features
        
        self.type_embedding = nn.Embedding(num_types, 64)
        
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
        return predictions


# ============================================================================
# Dataset
# ============================================================================

class RiceTestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        type_encoder: LabelEncoder,
        transforms: Optional[A.Compose] = None
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.type_encoder = type_encoder
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = self.image_dir / f"{row['ID']}.png"
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        rice_type = self.type_encoder.transform([row['Comment']])[0]
        rice_type = torch.tensor(rice_type, dtype=torch.long)
        
        return {
            'image': image,
            'rice_type': rice_type,
            'id': row['ID']
        }


# ============================================================================
# Transforms
# ============================================================================

def get_test_transforms(img_size: int = 384):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_tta_transforms(img_size: int = 384):
    """Test-time augmentation transforms."""
    return [
        # Original
        A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Horizontal flip
        A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Vertical flip
        A.Compose([
            A.Resize(img_size, img_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Both flips
        A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]


# ============================================================================
# Inference Functions
# ============================================================================

def load_model(checkpoint_path: Path, config: Config) -> nn.Module:
    """Load trained model from checkpoint."""
    model = RiceQualityModel(
        backbone=config.BACKBONE,
        pretrained=False,
        num_targets=config.NUM_TARGETS,
        num_types=config.NUM_TYPES
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()
    
    return model


def predict_single_model(
    model: nn.Module,
    test_df: pd.DataFrame,
    config: Config,
    type_encoder: LabelEncoder,
    use_tta: bool = True
) -> np.ndarray:
    """Generate predictions for a single model."""
    
    if use_tta:
        transforms_list = get_tta_transforms(config.IMG_SIZE)
    else:
        transforms_list = [get_test_transforms(config.IMG_SIZE)]
    
    all_predictions = []
    
    for transforms in transforms_list:
        dataset = RiceTestDataset(
            test_df, config.IMAGE_DIR, type_encoder, transforms
        )
        loader = DataLoader(
            dataset, batch_size=config.BATCH_SIZE,
            shuffle=False, num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
        
        predictions = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting", leave=False):
                images = batch['image'].to(config.DEVICE)
                rice_types = batch['rice_type'].to(config.DEVICE)
                
                preds = model(images, rice_types)
                predictions.append(preds.cpu().numpy())
        
        predictions = np.vstack(predictions)
        all_predictions.append(predictions)
    
    # Average TTA predictions
    return np.mean(all_predictions, axis=0)


def post_process_predictions(
    predictions: np.ndarray,
    test_df: pd.DataFrame,
    config: Config
) -> np.ndarray:
    """Apply post-processing rules to predictions."""
    
    processed = predictions.copy()
    
    # 1. Clamp count targets to non-negative
    count_indices = [config.TARGET_COLS.index(col) for col in config.COUNT_TARGETS]
    processed[:, count_indices] = np.maximum(processed[:, count_indices], 0)
    
    # 2. Round count targets to integers
    processed[:, count_indices] = np.round(processed[:, count_indices])
    
    # 3. Type-specific constraints
    for idx, row in test_df.iterrows():
        rice_type = row['Comment']
        
        if rice_type == 'Paddy':
            # Chalky_Count should be 0 for Paddy
            chalky_idx = config.TARGET_COLS.index('Chalky_Count')
            processed[idx, chalky_idx] = 0
    
    return processed


def main():
    """Main inference function."""
    print("="*80)
    print("RICE QUALITY ASSESSMENT - INFERENCE")
    print("="*80)
    
    config = Config()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {config.DEVICE}")
    print(f"Using TTA: {config.USE_TTA}")
    print(f"Number of folds: {config.N_FOLDS}")
    
    # Load data
    train_df = pd.read_csv(config.DATA_DIR / "Train.csv")
    test_df = pd.read_csv(config.DATA_DIR / "Test.csv")
    
    print(f"\nTest samples: {len(test_df)}")
    
    # Encode rice types
    type_encoder = LabelEncoder()
    type_encoder.fit(train_df['Comment'])
    print(f"Rice types: {type_encoder.classes_}")
    
    # Load models and generate predictions
    all_fold_predictions = []
    
    for fold in range(config.N_FOLDS):
        checkpoint_path = config.MODEL_DIR / f'model_fold{fold}.pt'
        
        if not checkpoint_path.exists():
            print(f"⚠️ Model for fold {fold} not found, skipping...")
            continue
        
        print(f"\nLoading model for fold {fold}...")
        model = load_model(checkpoint_path, config)
        
        predictions = predict_single_model(
            model, test_df, config, type_encoder, config.USE_TTA
        )
        all_fold_predictions.append(predictions)
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    if len(all_fold_predictions) == 0:
        print("❌ No models found! Please train models first.")
        return
    
    # Average fold predictions
    print(f"\nAveraging predictions from {len(all_fold_predictions)} folds...")
    final_predictions = np.mean(all_fold_predictions, axis=0)
    
    # Post-processing
    print("Applying post-processing...")
    final_predictions = post_process_predictions(final_predictions, test_df, config)
    
    # Create submission
    submission = test_df[['ID']].copy()
    for i, col in enumerate(config.TARGET_COLS):
        submission[col] = final_predictions[:, i]
    
    # Save submission
    submission_path = config.OUTPUT_DIR / 'submission_baseline.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\n✅ Submission saved to {submission_path}")
    
    # Print sample predictions
    print("\nSample predictions:")
    print(submission.head(10).to_string())
    
    # Validation check
    print("\n📊 Prediction Statistics:")
    for col in config.TARGET_COLS:
        print(f"  {col}: mean={submission[col].mean():.2f}, std={submission[col].std():.2f}")


if __name__ == "__main__":
    main()
