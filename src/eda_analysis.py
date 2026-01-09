"""
Rice Quality Assessment Challenge - Comprehensive EDA
Staff ML Engineer Analysis

This script performs deep exploratory data analysis to understand:
1. Target variable distributions and correlations
2. Rice type (Comment) based analysis
3. Image statistics
4. Challenge-winning strategy development
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import os
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
DATA_DIR = Path("/home/kara/Data/projects/rice-ops/RiceData/Unido_AfricaRice_Challenge")
IMAGE_DIR = DATA_DIR / "unido_rice_images"
OUTPUT_DIR = Path("/home/kara/Data/projects/rice-ops/outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("=" * 80)
print("RICE QUALITY ASSESSMENT CHALLENGE - COMPREHENSIVE EDA")
print("=" * 80)

train_df = pd.read_csv(DATA_DIR / "Train.csv")
test_df = pd.read_csv(DATA_DIR / "Test.csv")
sample_sub = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
data_dict = pd.read_csv(DATA_DIR / "DataDictionary.csv")

# Target columns
target_cols = ['Count', 'Broken_Count', 'Long_Count', 'Medium_Count', 'Black_Count', 
               'Chalky_Count', 'Red_Count', 'Yellow_Count', 'Green_Count',
               'WK_Length_Average', 'WK_Width_Average', 'WK_LW_Ratio_Average',
               'Average_L', 'Average_a', 'Average_b']

count_targets = ['Count', 'Broken_Count', 'Long_Count', 'Medium_Count', 'Black_Count', 
                 'Chalky_Count', 'Red_Count', 'Yellow_Count', 'Green_Count']
continuous_targets = ['WK_Length_Average', 'WK_Width_Average', 'WK_LW_Ratio_Average',
                      'Average_L', 'Average_a', 'Average_b']

print("\n" + "=" * 80)
print("1. DATASET OVERVIEW")
print("=" * 80)

print(f"\n📊 Train samples: {len(train_df)}")
print(f"📊 Test samples: {len(test_df)}")
print(f"📊 Total images available: {len(list(IMAGE_DIR.glob('*.png')))}")
print(f"📊 Number of targets: {len(target_cols)}")

print("\n📋 Data Dictionary:")
for _, row in data_dict.iterrows():
    print(f"  • {row['VARIABLE']}: {row['DESCRIPTION'][:80]}...")

print("\n📋 Rice Types (Comment column):")
print(train_df['Comment'].value_counts())
print("\n📋 Test Rice Types:")
print(test_df['Comment'].value_counts())

print("\n" + "=" * 80)
print("2. TARGET VARIABLE STATISTICS")
print("=" * 80)

# Compute statistics for each target
stats_df = train_df[target_cols].describe().T
stats_df['range'] = stats_df['max'] - stats_df['min']
stats_df['cv'] = stats_df['std'] / stats_df['mean']  # Coefficient of variation
stats_df['skewness'] = train_df[target_cols].skew()
stats_df['kurtosis'] = train_df[target_cols].kurtosis()

print("\n📊 Target Statistics:")
print(stats_df[['mean', 'std', 'min', 'max', 'range', 'cv', 'skewness']].round(2).to_string())

# Identify challenging targets (high variance, skewed)
print("\n⚠️ CHALLENGING TARGETS (high CV or skewness):")
challenging = stats_df[(stats_df['cv'] > 1.0) | (abs(stats_df['skewness']) > 2)]
if len(challenging) > 0:
    print(challenging[['mean', 'std', 'cv', 'skewness']].round(2).to_string())
else:
    print("  All targets have reasonable distributions")

print("\n" + "=" * 80)
print("3. ANALYSIS BY RICE TYPE")
print("=" * 80)

# Group statistics by rice type
rice_types = train_df['Comment'].unique()

for rice_type in rice_types:
    subset = train_df[train_df['Comment'] == rice_type]
    print(f"\n🍚 {rice_type.upper()} RICE (n={len(subset)}):")
    print(subset[target_cols].mean().round(2).to_string())

# Identify which targets vary most by rice type
print("\n📊 Target Variance by Rice Type (helps identify type-conditional predictions):")
for col in target_cols:
    means_by_type = train_df.groupby('Comment')[col].mean()
    overall_std = means_by_type.std()
    overall_mean = means_by_type.mean()
    rel_var = overall_std / overall_mean if overall_mean != 0 else 0
    print(f"  {col}: relative variance = {rel_var:.3f}")

print("\n" + "=" * 80)
print("4. CORRELATION ANALYSIS")
print("=" * 80)

# Compute correlation matrix
corr_matrix = train_df[target_cols].corr()

# Find highly correlated pairs
print("\n📊 Highly Correlated Target Pairs (|r| > 0.7):")
for i, col1 in enumerate(target_cols):
    for j, col2 in enumerate(target_cols):
        if i < j:
            corr = corr_matrix.loc[col1, col2]
            if abs(corr) > 0.7:
                print(f"  • {col1} <-> {col2}: r = {corr:.3f}")

# Save correlation heatmap
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, annot_kws={"size": 8})
plt.title('Target Variable Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_matrix.png', dpi=150)
plt.close()
print(f"\n✅ Saved correlation matrix to {OUTPUT_DIR / 'correlation_matrix.png'}")

print("\n" + "=" * 80)
print("5. COUNT-BASED TARGET ANALYSIS")
print("=" * 80)

# Analyze count-based targets
print("\n📊 Count Target Analysis:")
# Count = Broken_Count + Long_Count + Medium_Count (+others)
# Let's verify relationships

train_df['computed_total'] = train_df['Broken_Count'] + train_df['Long_Count'] + train_df['Medium_Count']
print(f"\nRelationship: Broken + Long + Medium vs Count")
print(f"  Correlation: {train_df['Count'].corr(train_df['computed_total']):.3f}")
print(f"  Mean difference: {(train_df['Count'] - train_df['computed_total']).mean():.2f}")

# Zero-inflation analysis
print("\n📊 Zero-Inflation Analysis (important for count targets):")
for col in count_targets:
    zero_pct = (train_df[col] == 0).mean() * 100
    print(f"  {col}: {zero_pct:.1f}% zeros")

print("\n" + "=" * 80)
print("6. L*a*b* COLOR SPACE ANALYSIS")
print("=" * 80)

print("\n📊 L*a*b* Color Values by Rice Type:")
lab_cols = ['Average_L', 'Average_a', 'Average_b']
for rice_type in rice_types:
    subset = train_df[train_df['Comment'] == rice_type]
    print(f"\n{rice_type}:")
    print(f"  L* (lightness): {subset['Average_L'].mean():.2f} ± {subset['Average_L'].std():.2f}")
    print(f"  a* (red-green): {subset['Average_a'].mean():.2f} ± {subset['Average_a'].std():.2f}")
    print(f"  b* (yellow-blue): {subset['Average_b'].mean():.2f} ± {subset['Average_b'].std():.2f}")

print("\n" + "=" * 80)
print("7. BASELINE MAE ESTIMATION")
print("=" * 80)

# Estimate MAE using simple strategies
print("\n📊 Baseline MAE (using mean prediction):")
total_mae = 0
for col in target_cols:
    mae = np.abs(train_df[col] - train_df[col].mean()).mean()
    print(f"  {col}: MAE = {mae:.2f}")
    total_mae += mae
    
mean_mae = total_mae / len(target_cols)
print(f"\n📊 Overall Mean MAE (mean prediction baseline): {mean_mae:.2f}")

# Type-conditional baseline
print("\n📊 Baseline MAE (type-conditional mean prediction):")
total_mae_cond = 0
for col in target_cols:
    type_means = train_df.groupby('Comment')[col].transform('mean')
    mae_cond = np.abs(train_df[col] - type_means).mean()
    print(f"  {col}: MAE = {mae_cond:.2f}")
    total_mae_cond += mae_cond

mean_mae_cond = total_mae_cond / len(target_cols)
print(f"\n📊 Overall Mean MAE (type-conditional baseline): {mean_mae_cond:.2f}")
print(f"📈 Improvement from type conditioning: {((mean_mae - mean_mae_cond) / mean_mae * 100):.1f}%")

print("\n" + "=" * 80)
print("8. IMAGE STATISTICS")
print("=" * 80)

# Sample a few images to get size info
sample_images = list(IMAGE_DIR.glob('*.png'))[:5]
print(f"\n📊 Sample Image Analysis:")
for img_path in sample_images:
    img = Image.open(img_path)
    print(f"  {img_path.name}: size={img.size}, mode={img.mode}")

print("\n" + "=" * 80)
print("9. TRAIN-TEST DISTRIBUTION COMPARISON")  
print("=" * 80)

print("\n📊 Rice Type Distribution (Train vs Test):")
train_type_dist = train_df['Comment'].value_counts(normalize=True) * 100
test_type_dist = test_df['Comment'].value_counts(normalize=True) * 100

comparison_df = pd.DataFrame({
    'Train %': train_type_dist,
    'Test %': test_type_dist
})
print(comparison_df.round(1).to_string())

print("\n" + "=" * 80)
print("10. KEY INSIGHTS & STRATEGY RECOMMENDATIONS")
print("=" * 80)

print("""
🎯 KEY INSIGHTS:

1. THREE RICE TYPES: Paddy, Brown, White - each with distinct characteristics
   - Paddy: brown/golden color, with husk, larger grains
   - Brown: brownish color, no husk, intermediate size
   - White: white/grey color, milled, smaller appearance

2. TARGET CHARACTERISTICS:
   - 9 count-based targets (integers, many zero-inflated)
   - 6 continuous targets (physical measurements and L*a*b* color)
   - Count targets show high variance and potential for segment-specific prediction

3. CRITICAL OBSERVATIONS:
   - Chalky_Count is ONLY relevant for White rice (0 for Paddy)
   - Black_Count is very high for Paddy (almost all grains)
   - Green_Count is mostly for White rice (immature grains visible)
   - L*a*b* values clearly separate rice types

4. CORRELATION PATTERNS:
   - Physical dimensions (WK_Length, WK_Width, WK_LW_Ratio) are related
   - Color values (L*, a*, b*) form another correlated group
   - Some count targets are mutually exclusive by rice type

🚀 STRATEGY FOR 0.94+ MAE:

PHASE 1: Strong Baseline
- Use rice type (Comment) as strong conditioning signal
- Multi-head regression: one head per rice type
- Pre-trained vision backbone (EfficientNet-B4/B5, ConvNeXt)

PHASE 2: Advanced Architecture
- Multi-task learning with shared backbone
- Auxiliary task: Rice type classification
- Count targets: Consider negative binomial or Poisson regression heads
- Continuous targets: Standard MSE/Huber loss

PHASE 3: Data & Training
- Heavy augmentation (rotation, flips, color jitter)
- Mixup/CutMix for regularization
- Test-time augmentation (TTA)
- K-fold cross-validation with stratification by rice type

PHASE 4: Ensemble
- Ensemble multiple architectures
- Blend vision models with gradient boosting on extracted features
- Use out-of-fold predictions for stacking
""")

# Save summary statistics
stats_df.to_csv(OUTPUT_DIR / 'target_statistics.csv')
print(f"\n✅ Analysis complete! Results saved to {OUTPUT_DIR}")

# Create distribution plots
fig, axes = plt.subplots(5, 3, figsize=(16, 20))
axes = axes.flatten()

for idx, col in enumerate(target_cols):
    ax = axes[idx]
    for rice_type in rice_types:
        subset = train_df[train_df['Comment'] == rice_type][col]
        ax.hist(subset, bins=30, alpha=0.5, label=rice_type, density=True)
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'target_distributions.png', dpi=150)
plt.close()
print(f"✅ Saved distribution plots to {OUTPUT_DIR / 'target_distributions.png'}")

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)
