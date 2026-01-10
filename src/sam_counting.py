"""
Rice Grain Counting - SAM + GroundingDINO Approach
===================================================
Tests segmentation-based counting using:
1. SAM (Segment Anything) for grain segmentation
2. GroundingDINO for zero-shot detection (optional)
3. Classification head for grain types

This is a test script to evaluate if SAM can accurately segment rice grains.
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import cv2

# Check for SAM availability
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("SAM not installed. Run: pip install segment-anything")

# Check for GroundingDINO availability
try:
    from groundingdino.util.inference import load_model as load_gdino, predict as gdino_predict
    GDINO_AVAILABLE = True
except ImportError:
    GDINO_AVAILABLE = False
    print("GroundingDINO not installed. Run: pip install groundingdino-py")


# ============================================================================
# Configuration
# ============================================================================
class Config:
    DATA_DIR = Path("RiceData/Unido_AfricaRice_Challenge")
    IMAGE_DIR = None
    OUTPUT_DIR = Path("outputs/sam_counting")
    
    # SAM settings
    SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
    SAM_MODEL_TYPE = "vit_h"
    
    # Grain filtering
    MIN_GRAIN_AREA = 50      # Minimum area for a grain mask (pixels)
    MAX_GRAIN_AREA = 15000   # Maximum area for a grain mask
    MIN_ASPECT_RATIO = 1.5   # Minimum length/width ratio
    MAX_ASPECT_RATIO = 8.0   # Maximum length/width ratio
    
    # GroundingDINO settings
    GDINO_CHECKPOINT = "groundingdino_swint_ogc.pth"
    GDINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    COUNT_TARGETS = ['Count', 'Broken_Count', 'Long_Count', 'Medium_Count',
                     'Black_Count', 'Chalky_Count', 'Red_Count', 
                     'Yellow_Count', 'Green_Count']


# ============================================================================
# SAM-based Grain Counter
# ============================================================================
class SAMGrainCounter:
    """Uses SAM to segment and count rice grains"""
    
    def __init__(self, checkpoint_path: str, model_type: str = "vit_h", device="cuda"):
        print(f"Loading SAM {model_type}...")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device)
        self.device = device
        
        # Create mask generator with fine-grained settings
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=32,          # More points for dense scenes
            pred_iou_thresh=0.86,         # Higher threshold for better masks
            stability_score_thresh=0.92,
            crop_n_layers=1,
            min_mask_region_area=50,      # Minimum grain area
        )
        print("SAM loaded!")
    
    def count_grains(self, image: np.ndarray, config: Config) -> Tuple[int, List[Dict]]:
        """
        Count rice grains in an image
        
        Returns:
            total_count: Number of detected grains
            grain_masks: List of grain mask info
        """
        # Generate all masks
        masks = self.mask_generator.generate(image)
        
        # Filter masks by grain-like properties
        grain_masks = []
        for mask in masks:
            area = mask['area']
            
            # Filter by area
            if area < config.MIN_GRAIN_AREA or area > config.MAX_GRAIN_AREA:
                continue
            
            # Get mask bbox to compute aspect ratio
            bbox = mask['bbox']  # x, y, w, h
            if bbox[2] == 0 or bbox[3] == 0:
                continue
            
            aspect_ratio = max(bbox[2], bbox[3]) / min(bbox[2], bbox[3])
            
            # Filter by aspect ratio (grains are elongated)
            if aspect_ratio < config.MIN_ASPECT_RATIO or aspect_ratio > config.MAX_ASPECT_RATIO:
                continue
            
            grain_masks.append({
                'area': area,
                'bbox': bbox,
                'aspect_ratio': aspect_ratio,
                'segmentation': mask['segmentation'],
                'predicted_iou': mask['predicted_iou'],
                'stability_score': mask['stability_score']
            })
        
        return len(grain_masks), grain_masks
    
    def visualize(self, image: np.ndarray, grain_masks: List[Dict], 
                  output_path: str = None) -> np.ndarray:
        """Visualize detected grains"""
        vis_img = image.copy()
        
        for mask_info in grain_masks:
            mask = mask_info['segmentation']
            color = np.random.randint(0, 255, 3).tolist()
            
            # Create colored overlay
            vis_img[mask] = vis_img[mask] * 0.5 + np.array(color) * 0.5
            
            # Draw bbox
            x, y, w, h = mask_info['bbox']
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
        
        # Add count text
        cv2.putText(vis_img, f"Count: {len(grain_masks)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        return vis_img


# ============================================================================
# Simple Grain Color Classifier
# ============================================================================
class GrainClassifier:
    """Classifies grain type based on color features"""
    
    def __init__(self):
        # Thresholds based on typical rice grain colors
        # These may need tuning based on your images
        pass
    
    def classify(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Classify grain type based on color
        
        Returns dict with probabilities for each grain type:
        - broken, long, medium, black, chalky, red, yellow, green
        """
        # Extract masked region colors
        masked_pixels = image[mask]
        if len(masked_pixels) == 0:
            return {}
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_masked = hsv[mask]
        
        mean_h = hsv_masked[:, 0].mean()
        mean_s = hsv_masked[:, 1].mean()
        mean_v = hsv_masked[:, 2].mean()
        
        # RGB means
        mean_r = masked_pixels[:, 0].mean()
        mean_g = masked_pixels[:, 1].mean()
        mean_b = masked_pixels[:, 2].mean()
        
        # Simple rule-based classification
        probs = {
            'broken': 0.0,
            'long': 0.0,
            'medium': 0.0,
            'black': 0.0,
            'chalky': 0.0,
            'red': 0.0,
            'yellow': 0.0,
            'green': 0.0
        }
        
        # Black: low value (dark)
        if mean_v < 80:
            probs['black'] = 0.8
        
        # Chalky: low saturation, high value (white-ish)
        if mean_s < 30 and mean_v > 200:
            probs['chalky'] = 0.8
        
        # Red: high red channel, medium hue
        if mean_r > mean_g + 30 and mean_r > mean_b + 30:
            probs['red'] = 0.7
        
        # Yellow: high red and green
        if mean_r > 180 and mean_g > 160 and mean_b < 140:
            probs['yellow'] = 0.7
        
        # Green: high green
        if mean_g > mean_r + 20 and mean_g > mean_b + 20:
            probs['green'] = 0.7
        
        return probs


# ============================================================================
# Main Counting Pipeline
# ============================================================================
def count_image(image_path: Path, sam_counter: SAMGrainCounter, 
                classifier: GrainClassifier, config: Config) -> Dict[str, float]:
    """Process a single image and return all counts"""
    
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # Get total grain count
    total_count, grain_masks = sam_counter.count_grains(image, config)
    
    # Classify each grain
    type_counts = {t: 0 for t in config.COUNT_TARGETS}
    type_counts['Count'] = total_count
    
    for mask_info in grain_masks:
        probs = classifier.classify(image, mask_info['segmentation'])
        
        # Add to type counts
        for grain_type, prob in probs.items():
            if prob > 0.5:  # Threshold
                if f'{grain_type.capitalize()}_Count' in type_counts:
                    type_counts[f'{grain_type.capitalize()}_Count'] += 1
    
    return type_counts


def evaluate_sam(config: Config, num_samples: int = 10):
    """Evaluate SAM counting accuracy on training samples"""
    
    if not SAM_AVAILABLE:
        print("SAM not available. Please install: pip install segment-anything")
        return
    
    # Check for checkpoint
    if not Path(config.SAM_CHECKPOINT).exists():
        print(f"SAM checkpoint not found: {config.SAM_CHECKPOINT}")
        print("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return
    
    # Load models
    sam_counter = SAMGrainCounter(config.SAM_CHECKPOINT, config.SAM_MODEL_TYPE, config.DEVICE)
    classifier = GrainClassifier()
    
    # Load training data for validation
    train_df = pd.read_csv(config.DATA_DIR / "Train.csv")
    
    # Sample images
    sample_df = train_df.sample(n=min(num_samples, len(train_df)), random_state=42)
    
    results = []
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nEvaluating SAM on {len(sample_df)} samples...")
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        image_path = config.IMAGE_DIR / f"{row['ID']}.png"
        if not image_path.exists():
            continue
        
        # Get SAM predictions
        image = np.array(Image.open(image_path).convert('RGB'))
        total_count, grain_masks = sam_counter.count_grains(image, config)
        
        # Ground truth
        gt_count = row['Count']
        
        # Calculate error
        error = abs(total_count - gt_count)
        
        results.append({
            'ID': row['ID'],
            'gt_count': gt_count,
            'sam_count': total_count,
            'error': error,
            'num_masks': len(grain_masks)
        })
        
        # Visualize first few
        if len(results) <= 5:
            vis_path = config.OUTPUT_DIR / f"vis_{row['ID']}.png"
            sam_counter.visualize(image, grain_masks, str(vis_path))
            print(f"  {row['ID']}: GT={gt_count}, SAM={total_count}, Error={error}")
    
    # Summary
    results_df = pd.DataFrame(results)
    mae = results_df['error'].mean()
    
    print("\n" + "=" * 50)
    print("SAM COUNTING EVALUATION RESULTS")
    print("=" * 50)
    print(f"Samples: {len(results_df)}")
    print(f"Mean Absolute Error (Count): {mae:.2f}")
    print(f"Min Error: {results_df['error'].min():.0f}")
    print(f"Max Error: {results_df['error'].max():.0f}")
    print(f"\nGT Range: {results_df['gt_count'].min():.0f} - {results_df['gt_count'].max():.0f}")
    print(f"SAM Range: {results_df['sam_count'].min():.0f} - {results_df['sam_count'].max():.0f}")
    
    # Save results
    results_df.to_csv(config.OUTPUT_DIR / "sam_evaluation.csv", index=False)
    print(f"\nResults saved to: {config.OUTPUT_DIR / 'sam_evaluation.csv'}")
    
    return results_df


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='SAM-based Rice Grain Counting')
    parser.add_argument('--data-dir', default='RiceData/Unido_AfricaRice_Challenge')
    parser.add_argument('--output-dir', default='outputs/sam_counting')
    parser.add_argument('--sam-checkpoint', default='sam_vit_h_4b8939.pth')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--min-area', type=int, default=50)
    parser.add_argument('--max-area', type=int, default=15000)
    args = parser.parse_args()
    
    Config.DATA_DIR = Path(args.data_dir)
    Config.IMAGE_DIR = Config.DATA_DIR / "unido_rice_images"
    Config.OUTPUT_DIR = Path(args.output_dir)
    Config.SAM_CHECKPOINT = args.sam_checkpoint
    Config.MIN_GRAIN_AREA = args.min_area
    Config.MAX_GRAIN_AREA = args.max_area
    
    print("=" * 60)
    print("🌾 SAM-BASED RICE GRAIN COUNTING TEST")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"SAM Checkpoint: {Config.SAM_CHECKPOINT}")
    print(f"Grain area range: {Config.MIN_GRAIN_AREA} - {Config.MAX_GRAIN_AREA}")
    
    # Run evaluation
    evaluate_sam(Config, args.num_samples)


if __name__ == "__main__":
    main()
