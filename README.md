# Rice Quality Assessment Challenge

Multi-target regression model for rice quality prediction from images.

## Key Features
- ConvNeXt backbone with ImageNet-22k pretraining
- Z-score target normalization
- Mixup augmentation
- EMA (Exponential Moving Average) training
- Count-weighted loss function

## Training
```bash
python src/train_v3_improved.py
```

## Files
- `src/train_v3_improved.py` - V3 training with SOTA improvements
- `src/train_baseline.py` - Baseline V2 training
- `src/inference.py` - Inference script
- `src/eda_analysis.py` - EDA analysis

## Kaggle Usage
Upload dataset and run training notebook with GPU enabled.
