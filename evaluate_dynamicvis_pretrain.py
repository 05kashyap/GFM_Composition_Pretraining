#!/usr/bin/env python3
"""
Evaluate DynamicVis Pretrain model on fMoW dataset.

This script evaluates the pretrain model (with FPN + RoI extraction) which expects
bounding box annotations. It streams data directly from S3.

The pretrain model evaluates on:
- Per-image classification accuracy (using bounding box features)
- Top-1 and Top-5 accuracy
- Precision, Recall, F1-Score (macro)

Usage:
    # Evaluate pretrained weights 
    python evaluate_dynamicvis_pretrain.py --checkpoint /path/to/checkpoint.pth
    
    # Evaluate on a subset (faster)
    python evaluate_dynamicvis_pretrain.py --checkpoint /path/to/checkpoint.pth --num-samples 5000

    # Test with smaller images
    python evaluate_dynamicvis_pretrain.py --checkpoint /path/to/checkpoint.pth --use-msrgb
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root and DynamicVis to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "architectures" / "DynamicVis"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Import DynamicVis modules FIRST to register transforms and models
import dynamicvis  # noqa: F401 - registers modules

# Import mmdet components from DynamicVis local copy
from mmdet.models.roi_heads import GenericRoIExtractor  # noqa: F401
from mmdet.models.necks import FPN  # noqa: F401
from mmcv.cnn import GeneralizedAttention  # noqa: F401

# Now import dataset module
from datasets.fmow_s3_pretrain import (
    FMoWS3PretrainDataset, 
    LoadImageFromS3WithBbox,
    FMOW_CATEGORIES,
)

from mmengine.config import Config
from mmengine.registry import MODELS, init_default_scope
from mmengine.dataset import Compose, pseudo_collate


def compute_metrics(all_preds: np.ndarray, all_labels: np.ndarray, all_scores: np.ndarray, num_classes: int):
    """Compute evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score, top_k_accuracy_score,
        precision_score, recall_score, f1_score,
    )
    
    metrics = {}
    
    # Accuracy metrics
    metrics['top1_accuracy'] = accuracy_score(all_labels, all_preds) * 100
    
    if all_scores is not None and num_classes > 5:
        labels = list(range(num_classes))
        metrics['top5_accuracy'] = top_k_accuracy_score(all_labels, all_scores, k=5, labels=labels) * 100
    
    # Precision, Recall, F1 (macro averaged)
    metrics['precision'] = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    metrics['recall'] = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    metrics['f1_score'] = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    
    return metrics


def build_pretrain_model(checkpoint_path: str, num_classes: int = 63, img_size: int = 512, device: str = 'cuda'):
    """Build the full DynamicVis pretrain model (with FPN neck)."""
    
    # Use the original DynamicVis pretrain config
    config_path = project_root / "architectures" / "DynamicVis" / "configs_DynamicVis" / "fMoW" / "pretrain_dynamicvis_b_bf16_mamba.py"
    
    if config_path.exists():
        print(f"Loading model from config: {config_path}")
        cfg = Config.fromfile(str(config_path))
        
        # Override image size if needed
        if img_size != 512:
            cfg.model.backbone.img_size = img_size
        
        # Add data_preprocessor to model
        if 'data_preprocessor' in cfg and cfg.model.get('data_preprocessor') is None:
            cfg.model['data_preprocessor'] = cfg.data_preprocessor
        
        # Initialize mmdet scope
        init_default_scope('mmdet')
        
        # Build model
        model = MODELS.build(cfg.model)
    else:
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    
    model.eval()
    return model


def create_val_dataset_pretrain(img_size: int = 512, use_msrgb: bool = True, max_samples: int = None):
    """Create validation dataset for pretrain model with bounding boxes."""
    bgr_mean = [103.53, 116.28, 123.675]
    
    val_pipeline = [
        dict(type='LoadImageFromS3WithBbox', to_float32=True, max_edge=1024),
        dict(type='Resize', scale=(img_size, img_size), keep_ratio=True),
        dict(type='Pad', size=(img_size, img_size), pad_val=dict(img=tuple(bgr_mean))),
        dict(type='FilterAnnotations', min_gt_bbox_wh=(8, 8), keep_empty=True),
        dict(type='PackDetInputs'),
    ]
    
    dataset = FMoWS3PretrainDataset(
        bucket='spacenet-dataset',
        s3_prefix='Hosted-Datasets/fmow/fmow-rgb',
        split='val',
        pipeline=val_pipeline,
        use_msrgb=use_msrgb,
        max_samples=max_samples,
        enable_prefetch=True,
        prefetch_size=64,
        num_prefetch_workers=4,
    )
    
    return dataset


def evaluate_pretrain_model(
    model: nn.Module,
    dataset,
    num_samples: int = None,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str = 'cuda',
):
    """Evaluate pretrain model on dataset with bounding boxes.
    
    The pretrain model uses MIL (multiple instance learning) where:
    - Each image has multiple bounding boxes
    - Features are extracted for each box via RoI pooling
    - Final prediction aggregates across all boxes in an image
    
    For evaluation, we compute per-image accuracy by taking the
    most frequent predicted class among all bounding boxes.
    """
    from torch.utils.data import DataLoader, Subset
    import random
    
    # Subset if requested
    total_samples = len(dataset)
    if num_samples is not None and num_samples < total_samples:
        print(f"Using {num_samples} samples out of {total_samples}")
        indices = random.sample(range(total_samples), num_samples)
        dataset = Subset(dataset, indices)
    else:
        num_samples = total_samples
        print(f"Evaluating on all {num_samples} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=pseudo_collate,
    )
    
    # Normalization for pretrain model
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)
    
    all_preds = []
    all_labels = []
    all_scores = []
    
    model.eval()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
        for batch_idx, data in enumerate(pbar):
            try:
                if data is None:
                    continue
                
                # Handle batch data format
                if isinstance(data, dict):
                    if isinstance(data.get('inputs'), list):
                        inputs = torch.stack(data['inputs']).to(device).float()
                    else:
                        inputs = data['inputs'].to(device).float()
                    data_samples = data.get('data_samples', [])
                elif isinstance(data, list):
                    inputs_list = [d.get('inputs') for d in data if d is not None]
                    if not inputs_list:
                        continue
                    inputs = torch.stack(inputs_list).to(device).float()
                    data_samples = [d.get('data_samples') for d in data if d is not None]
                else:
                    continue
                
                # Normalize (BGR -> RGB already done in preprocessor)
                inputs = (inputs - mean) / std
                
                # Move data_samples to device and ensure they have gt_instances
                for ds in data_samples:
                    if ds is not None and hasattr(ds, 'gt_instances'):
                        ds.gt_instances.bboxes = ds.gt_instances.bboxes.to(device)
                        ds.gt_instances.labels = ds.gt_instances.labels.to(device)
                
                # Forward pass
                try:
                    # The pretrain model expects data_samples with gt_instances.bboxes
                    outputs = model.predict(inputs, data_samples)
                except Exception as e:
                    print(f"Forward pass error: {e}")
                    continue
                
                # Process outputs
                # outputs is a list of DataSample objects with pred_label
                for i, output in enumerate(outputs):
                    if data_samples[i] is None:
                        continue
                    
                    ds = data_samples[i]
                    
                    # Get ground truth - use majority label from bboxes
                    if hasattr(ds, 'gt_instances') and hasattr(ds.gt_instances, 'labels'):
                        gt_labels = ds.gt_instances.labels.cpu().numpy()
                        if len(gt_labels) > 0:
                            # Use most common label as GT (for multi-box images)
                            from collections import Counter
                            gt_label = Counter(gt_labels).most_common(1)[0][0]
                        else:
                            continue
                    else:
                        continue
                    
                    # Get prediction
                    if hasattr(output, 'pred_label'):
                        pred_label = output.pred_label.item()
                    elif hasattr(output, 'pred_score'):
                        pred_label = output.pred_score.argmax().item()
                    else:
                        continue
                    
                    # Get prediction scores if available
                    if hasattr(output, 'pred_score'):
                        scores = output.pred_score.cpu().numpy()
                    else:
                        scores = None
                    
                    all_preds.append(pred_label)
                    all_labels.append(gt_label)
                    if scores is not None:
                        all_scores.append(scores)
                
                # Update progress
                if len(all_preds) > 0:
                    running_acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
                    pbar.set_postfix({'acc': f'{running_acc:.2f}%', 'samples': len(all_preds)})
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if len(all_preds) == 0:
        print("Error: No samples were successfully processed!")
        return {'top1_accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores) if len(all_scores) > 0 else None
    
    num_classes = len(FMOW_CATEGORIES)
    metrics = compute_metrics(all_preds, all_labels, all_scores, num_classes)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate DynamicVis Pretrain model on fMoW')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--img-size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--use-msrgb', action='store_true', default=True,
                        help='Use msrgb images (smaller, faster)')
    parser.add_argument('--use-rgb', dest='use_msrgb', action='store_false',
                        help='Use full rgb images (larger)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for evaluation')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    print("=" * 60)
    print("DynamicVis Pretrain Evaluation on fMoW")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image size: {args.img_size}")
    print(f"Image type: {'msrgb' if args.use_msrgb else 'rgb'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num samples: {args.num_samples or 'all'}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Build model
    print("\nLoading model...")
    model = build_pretrain_model(
        args.checkpoint, 
        num_classes=63, 
        img_size=args.img_size, 
        device=args.device
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {num_params:.2f}M")
    
    # Create dataset
    print("\nLoading validation dataset...")
    dataset = create_val_dataset_pretrain(
        img_size=args.img_size,
        use_msrgb=args.use_msrgb,
        max_samples=args.num_samples,
    )
    
    # Evaluate
    print("\nStarting evaluation...")
    start_time = time.time()
    
    metrics = evaluate_pretrain_model(
        model=model,
        dataset=dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'Value':>15}")
    print("-" * 40)
    print(f"{'Top-1 Accuracy':<25} {metrics['top1_accuracy']:>14.2f}%")
    if 'top5_accuracy' in metrics:
        print(f"{'Top-5 Accuracy':<25} {metrics['top5_accuracy']:>14.2f}%")
    print(f"{'Precision (macro)':<25} {metrics['precision']:>14.2f}%")
    print(f"{'Recall (macro)':<25} {metrics['recall']:>14.2f}%")
    print(f"{'F1-Score (macro)':<25} {metrics['f1_score']:>14.2f}%")
    print("-" * 40)
    print(f"{'Evaluation time':<25} {elapsed_time:>13.1f}s")
    print("=" * 60)
    
    return metrics


if __name__ == '__main__':
    main()
