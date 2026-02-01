#!/usr/bin/env python3
"""
Evaluate DynamicVis model on fMoW dataset.

This script loads a pretrained DynamicVis model and evaluates it on a subset
of the fMoW validation set, reporting the same metrics used during training:
- Top-1 Accuracy
- Top-5 Accuracy  
- Precision (macro)
- Recall (macro)
- F1-Score (macro)

Usage:
    # Evaluate pretrained weights
    python evaluate_dynamicvis.py --checkpoint /path/to/checkpoint.pth
    
    # Evaluate on a subset (faster)
    python evaluate_dynamicvis.py --checkpoint /path/to/checkpoint.pth --num-samples 5000
    
    # Use specific batch size
    python evaluate_dynamicvis.py --checkpoint /path/to/checkpoint.pth --batch-size 64
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

# Import after path setup
import dynamicvis  # noqa: F401 - registers modules
from datasets.fmow_s3_mmpretrain import FMoWS3Dataset, LoadImageFromS3, FMOW_CATEGORIES

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


def compute_metrics(all_preds: np.ndarray, all_labels: np.ndarray, all_scores: np.ndarray, num_classes: int):
    """Compute evaluation metrics.
    
    Args:
        all_preds: Predicted class indices (N,)
        all_labels: Ground truth labels (N,)
        all_scores: Prediction scores (N, num_classes)
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, top_k_accuracy_score,
        precision_score, recall_score, f1_score,
        classification_report
    )
    
    metrics = {}
    
    # Accuracy metrics
    metrics['top1_accuracy'] = accuracy_score(all_labels, all_preds) * 100
    
    if all_scores is not None and num_classes > 5:
        metrics['top5_accuracy'] = top_k_accuracy_score(all_labels, all_scores, k=5) * 100
    
    # Precision, Recall, F1 (macro averaged)
    metrics['precision'] = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    metrics['recall'] = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    metrics['f1_score'] = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    
    return metrics


def build_model_from_config(config_path: str, checkpoint_path: str, device: str = 'cuda'):
    """Build model from config and load checkpoint.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    cfg = Config.fromfile(config_path)
    
    # Build model
    model = MODELS.build(cfg.model)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DDP training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # Load with strict=False to handle minor mismatches
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Warning: Missing keys: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Warning: Unexpected keys: {unexpected}")
    
    model.eval()
    return model


def build_simple_model(checkpoint_path: str, num_classes: int = 63, img_size: int = 224, device: str = 'cuda'):
    """Build a simple DynamicVis model without full config.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_classes: Number of classes
        img_size: Input image size
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    from mmpretrain.models import ImageClassifier
    from dynamicvis.models import DynamicVisBackbone, DynamicVisClsHead
    
    # Build model matching the pretrain config
    model = ImageClassifier(
        backbone=dict(
            type='DynamicVisBackbone',
            arch='b',
            path_type='forward_reverse_mean',
            sampling_scale=dict(type='fixed', val=0.1),
            global_token_cfg=dict(pos='head', num=-1),
            is_softmax_on_x=True,
            img_size=img_size,
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            spatial_token_keep_ratios=[8, 4, 2, 1],
            out_indices=(3,),
            out_type='avg_featmap',
        ),
        neck=None,
        head=dict(
            type='DynamicVisClsHead',
            num_classes=num_classes,
            in_channels=768,
            loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        ),
    )
    
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
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
    
    # Try to load, handling potential architecture mismatches
    try:
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:3]}...")
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    except Exception as e:
        print(f"Warning: Could not load all weights: {e}")
        print("Attempting partial load...")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() 
                         if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
    
    model.eval()
    return model


def create_val_dataset(num_samples: int = None, img_size: int = 224):
    """Create validation dataset.
    
    Args:
        num_samples: Number of samples to use (None for all)
        img_size: Image size for transforms
        
    Returns:
        Dataset and data info
    """
    from mmpretrain.datasets.transforms import (
        ResizeEdge, CenterCrop, PackInputs
    )
    
    val_pipeline = [
        dict(type='LoadImageFromS3', to_float32=True),
        dict(type='ResizeEdge', scale=int(img_size * 1.14), edge='short'),
        dict(type='CenterCrop', crop_size=img_size),
        dict(type='PackInputs'),
    ]
    
    # Build pipeline
    pipeline = Compose(val_pipeline)
    
    dataset = FMoWS3Dataset(
        bucket='spacenet-dataset',
        s3_prefix='Hosted-Datasets/fmow/fmow-rgb',
        manifest_key='Hosted-Datasets/fmow/fmow-rgb/manifest.json.bz2',
        local_manifest='data/manifest.json.bz2',
        split='val',
        pipeline=val_pipeline,
        enable_prefetch=True,
        prefetch_size=256,
        num_prefetch_workers=8,
    )
    
    return dataset


def evaluate_model(
    model: nn.Module,
    dataset,
    num_samples: int = None,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = 'cuda',
):
    """Evaluate model on dataset.
    
    Args:
        model: Model to evaluate
        dataset: Evaluation dataset
        num_samples: Number of samples to evaluate (None for all)
        batch_size: Batch size
        num_workers: Number of data loader workers
        device: Device for evaluation
        
    Returns:
        Dictionary of metrics
    """
    from torch.utils.data import DataLoader, Subset
    import random
    
    # Subset if requested
    if num_samples is not None and num_samples < len(dataset):
        print(f"Using {num_samples} samples out of {len(dataset)}")
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = Subset(dataset, indices)
    else:
        num_samples = len(dataset)
        print(f"Evaluating on all {num_samples} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    # Data preprocessor (normalization)
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)
    
    all_preds = []
    all_labels = []
    all_scores = []
    
    model.eval()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
        for batch_idx, data in enumerate(pbar):
            # Handle MMPretrain data format
            if isinstance(data, dict):
                inputs = data['inputs'].to(device).float()
                # Get labels from data_samples
                if 'data_samples' in data:
                    labels = torch.tensor([ds.gt_label.item() for ds in data['data_samples']]).to(device)
                else:
                    labels = data.get('gt_label', data.get('labels')).to(device)
            else:
                inputs, labels = data
                inputs = inputs.to(device).float()
                labels = labels.to(device)
            
            # Normalize
            inputs = (inputs - mean) / std
            
            # Forward pass
            try:
                outputs = model(inputs)
                
                # Handle different output formats
                if isinstance(outputs, (list, tuple)):
                    logits = outputs[0] if isinstance(outputs[0], torch.Tensor) else outputs[0].cpu()
                elif hasattr(outputs, 'head_outputs'):
                    logits = outputs.head_outputs
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    # Try to extract from DataSample
                    logits = outputs
                
                if isinstance(logits, torch.Tensor):
                    scores = torch.softmax(logits, dim=1)
                    preds = scores.argmax(dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_scores.extend(scores.cpu().numpy())
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
            
            # Update progress bar
            if len(all_preds) > 0:
                current_acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
                pbar.set_postfix({'acc': f'{current_acc:.2f}%'})
    
    # Compute final metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    metrics = compute_metrics(all_preds, all_labels, all_scores, num_classes=63)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate DynamicVis on fMoW')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional, will use default if not provided)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for evaluation')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    print("=" * 60)
    print("DynamicVis Evaluation on fMoW")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image size: {args.img_size}")
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
    if args.config:
        model = build_model_from_config(args.config, args.checkpoint, args.device)
    else:
        model = build_simple_model(args.checkpoint, num_classes=63, img_size=args.img_size, device=args.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {num_params:.2f}M")
    
    # Create dataset
    print("\nLoading validation dataset...")
    dataset = create_val_dataset(num_samples=args.num_samples, img_size=args.img_size)
    
    # Evaluate
    print("\nStarting evaluation...")
    start_time = time.time()
    
    metrics = evaluate_model(
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
    
    # Return metrics for programmatic use
    return metrics


if __name__ == '__main__':
    main()
