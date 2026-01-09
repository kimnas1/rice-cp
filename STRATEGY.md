# 🌾 Rice Quality Assessment Challenge - Winning Strategy

## Target: 0.94+ on Public Leaderboard

---

## 📊 Problem Understanding

### Task
- **Input:** RGB images of rice grains (3840x2748 pixels)
- **Output:** 15 quality indicators per image
- **Metric:** Mean Absolute Error (MAE), weighted equally across all targets

### Three Rice Types (Critical Insight!)
| Type | Visual | Key Characteristics |
|------|--------|---------------------|
| **Paddy** | Golden/brown with husk | Largest grains, L*≈59, a*≈9, b*≈32 |
| **Brown** | Brown, no husk | Medium size, L*≈65, a*≈0, b*≈13 |
| **White** | White/grey, milled | Smallest appearance, L*≈71, a*≈-2, b*≈-4 |

### Target Variables
1. **Count-based (9):** Count, Broken_Count, Long_Count, Medium_Count, Black_Count, Chalky_Count, Red_Count, Yellow_Count, Green_Count
2. **Continuous (6):** WK_Length_Average, WK_Width_Average, WK_LW_Ratio_Average, Average_L, Average_a, Average_b

---

## 🔑 Key Insights from EDA

### 1. Rice Type is THE Most Important Feature
- **Type-conditional baseline MAE: 61.05** vs Mean baseline: 120.88
- **49.5% improvement** just from knowing rice type!
- This is available in both train AND test data

### 2. Target Correlations (Exploit These!)
- Strong groups: `(L*, a*, b*)` highly correlated (r > 0.8)
- Physical dimensions: `(WK_Length, WK_Width)` correlated (r = 0.76)
- Black_Count ↔ Chalky_Count: negative correlation (r = -0.80)
- Count ↔ Broken_Count: positive correlation (r = 0.86)

### 3. Zero-Inflation Issues
- **Medium_Count:** 64.8% zeros
- **Green_Count:** 60.2% zeros
- **Chalky_Count:** 37.2% zeros (only for non-Paddy rice)

### 4. Type-Specific Targets
- **Chalky_Count = 0** for all Paddy rice
- **Green_Count ≈ 0** for Brown rice
- **Black_Count** nearly equals **Count** for Paddy

---

## 🏗️ Model Architecture Strategy

### Phase 1: Strong Baseline (Expected CV: ~45-55 MAE)
```
Image → Backbone (EfficientNet-B4, 384px) → Global Pool → 
Rice Type Embedding (from known label) → Concat → 
FC Layers → 15 Outputs
```

### Phase 2: Multi-Task Architecture (Expected CV: ~35-45 MAE)
```
Image → Backbone (ConvNeXt-Base, 512px) → Global Pool →
├── Rice Type Classifier (auxiliary, 3-way) 
├── Type-Conditioned Regression Head (15 outputs)
└── Optional: Per-type specific heads
```

### Phase 3: Advanced (Target CV: ~25-35 MAE)
- Ensemble of different backbones (EfficientNet, ConvNeXt, SwinV2)
- Stacking with LightGBM on extracted features
- Hybrid: Vision + traditional CV features

---

## 📈 Loss Function Strategy

### For Count Targets (high MAE contribution)
```python
# Option 1: Smooth L1 (Huber) - handles outliers better
loss_count = F.smooth_l1_loss(pred, target, beta=50.0)

# Option 2: Weighted MAE - directly optimize metric
loss_count = (torch.abs(pred - target) * sample_weight).mean()

# Option 3: Log-transformation for high counts
pred_log = torch.log1p(pred_raw)
target_log = torch.log1p(target)
loss = F.mse_loss(pred_log, target_log)
```

### For Continuous Targets (L*a*b*, dimensions)
```python
# Standard MSE works well, values are normalized
loss_continuous = F.mse_loss(pred, target)
```

### Combined Loss
```python
total_loss = loss_count + lambda_cont * loss_continuous + lambda_cls * loss_classification
```

---

## 🔧 Implementation Plan

### Step 1: Data Pipeline
```python
# Transforms
train_transforms = A.Compose([
    A.RandomResizedCrop(384, 384, scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

### Step 2: Model Architecture
```python
class RiceQualityModel(nn.Module):
    def __init__(self, backbone='efficientnet_b4', num_targets=15, num_types=3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.embed_dim = self.backbone.num_features
        
        # Rice type embedding
        self.type_embedding = nn.Embedding(num_types, 64)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_targets)
        )
        
        # Type classifier head (auxiliary)
        self.classifier = nn.Linear(self.embed_dim, num_types)
    
    def forward(self, x, rice_type_idx):
        features = self.backbone(x)
        type_emb = self.type_embedding(rice_type_idx)
        combined = torch.cat([features, type_emb], dim=1)
        predictions = self.regressor(combined)
        type_logits = self.classifier(features)
        return predictions, type_logits
```

### Step 3: Training Configuration
```python
config = {
    'backbone': 'efficientnet_b4',
    'img_size': 384,
    'batch_size': 16,
    'epochs': 30,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'scheduler': 'cosine',
    'warmup_epochs': 2,
    'n_folds': 5,
    'seed': 42
}
```

### Step 4: Validation Strategy
- **5-Fold Stratified CV** by rice type
- Track MAE per target + overall
- Focus improvements on highest MAE contributors:
  1. Broken_Count (MAE ~180)
  2. Chalky_Count (MAE ~170)
  3. Count (MAE ~164)
  4. Long_Count (MAE ~140)
  5. Black_Count (MAE ~135)

---

## 🚀 Execution Roadmap

| Phase | Model | Expected CV MAE | Expected LB |
|-------|-------|-----------------|-------------|
| 1 | EfficientNet-B4 baseline | ~50-60 | ~55-65 |
| 2 | + Type conditioning | ~40-50 | ~45-55 |
| 3 | + Better augmentations + ConvNeXt | ~30-40 | ~35-45 |
| 4 | + Multi-backbone ensemble | ~25-35 | ~30-40 |
| 5 | + TTA + Post-processing | ~22-30 | ~25-35 |
| 6 | + Feature stacking | ~20-28 | ~22-30 |

**Note:** The 0.94 target likely refers to a normalized/scaled metric, not raw MAE.

---

## 📝 Post-Processing Ideas

1. **Clip predictions to valid ranges** based on rice type
2. **Round count predictions** to integers
3. **Enforce constraints:**
   - Chalky_Count = 0 for Paddy
   - All counts >= 0
   - Broken_Count + Long_Count + Medium_Count ≤ Count

---

## ⚡ Quick Wins

1. ✅ Use rice type as input feature (given in test data!)
2. ✅ Normalize targets per rice type before training
3. ✅ Heavy augmentation for the limited dataset (938 train)
4. ✅ Ensemble multiple seeds
5. ✅ TTA with 8x augmentation (flips + rotations)

---

## 🎯 First Submission Plan

1. Train baseline EfficientNet-B4 with type conditioning
2. 5-fold CV to get robust predictions
3. Simple TTA (4 flips)
4. Submit and establish baseline LB score
5. Iterate based on LB feedback
