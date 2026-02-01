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
    # Evaluate pretrained weights (auto-detects architecture)
    python evaluate_dynamicvis.py --checkpoint /path/to/checkpoint.pth
    
    # Evaluate on a subset (faster)
    python evaluate_dynamicvis.py --checkpoint /path/to/checkpoint.pth --num-samples 5000
    
    # Force simple architecture (for models trained with our config)
    python evaluate_dynamicvis.py --checkpoint /path/to/checkpoint.pth --model-type simple
    
    # Force pretrain architecture (for official DynamicVis weights)
    python evaluate_dynamicvis.py --checkpoint /path/to/checkpoint.pth --model-type pretrain
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
# This includes the local mmdet with GenericRoIExtractor, FPN, etc.
import dynamicvis  # noqa: F401 - registers modules

# Import mmdet components from DynamicVis local copy
from mmdet.models.roi_heads import GenericRoIExtractor  # noqa: F401
from mmdet.models.necks import FPN  # noqa: F401
from mmcv.cnn import GeneralizedAttention  # noqa: F401 - from mmcv

# Now import dataset module to register LoadImageFromS3
from datasets import fmow_s3_mmpretrain  # noqa: F401 - registers LoadImageFromS3
from datasets.fmow_s3_mmpretrain import FMoWS3Dataset, FMOW_CATEGORIES

from mmengine.config import Config
from mmengine.dataset import Compose
from mmpretrain.registry import MODELS, TRANSFORMS


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
        # Need to specify labels since not all classes may be present in the sample
        labels = list(range(num_classes))
        metrics['top5_accuracy'] = top_k_accuracy_score(all_labels, all_scores, k=5, labels=labels) * 100
    
    # Precision, Recall, F1 (macro averaged)
    metrics['precision'] = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    metrics['recall'] = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    metrics['f1_score'] = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    
    return metrics


def detect_model_type(checkpoint_path: str) -> str:
    """Detect model architecture type from checkpoint.
    
    Returns:
        'pretrain' for official DynamicVis pretrained weights (with FPN)
        'simple' for our simplified classification model
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Check for FPN neck keys (pretrain model has pre_neck)
    has_fpn = any('pre_neck' in k or 'neck.roi_layers' in k for k in state_dict.keys())
    
    # Check head input channels
    head_weight_key = None
    for k in state_dict.keys():
        if 'head.fc.weight' in k or 'head.classifier.weight' in k:
            head_weight_key = k
            break
    
    if head_weight_key:
        head_weight = state_dict[head_weight_key]
        in_channels = head_weight.shape[1]
        if in_channels == 256:
            return 'pretrain'  # FPN output is 256
        elif in_channels == 768:
            return 'simple'  # Direct backbone output is 768
    
    if has_fpn:
        return 'pretrain'
    
    return 'simple'


def build_pretrain_model(checkpoint_path: str, num_classes: int = 63, img_size: int = 512, device: str = 'cuda'):
    """Build the full DynamicVis pretrain model (with FPN neck).
    
    This matches the official pretrained weights architecture.
    Uses mmengine with proper scope handling.
    """
    from mmengine.config import Config
    from mmengine.registry import init_default_scope, MODELS as ENGINE_MODELS
    
    # Use the original DynamicVis pretrain config
    config_path = Path(__file__).parent / "architectures" / "DynamicVis" / "configs_DynamicVis" / "fMoW" / "pretrain_dynamicvis_b_bf16_mamba.py"
    
    if config_path.exists():
        print(f"Loading model from config: {config_path}")
        cfg = Config.fromfile(str(config_path))
        
        # Override image size if needed
        if img_size != 512:
            cfg.model.backbone.img_size = img_size
        
        # IMPORTANT: The data_preprocessor is defined at top-level in config,
        # but ImageClassifier expects it inside the model config.
        # MMEngine Runner normally handles this, but we need to do it manually.
        if 'data_preprocessor' in cfg and cfg.model.get('data_preprocessor') is None:
            cfg.model['data_preprocessor'] = cfg.data_preprocessor
        
        # Initialize mmdet scope as specified in the config
        init_default_scope('mmdet')
        
        # Build model using mmengine's generic MODELS registry (respects default scope)
        model = ENGINE_MODELS.build(cfg.model)
    else:
        raise FileNotFoundError(
            f"Config not found at {config_path}. "
            "The pretrain model requires the original DynamicVis config file."
        )
    
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
        print(f"Missing keys ({len(missing)}): {missing[:3]}..." if len(missing) > 3 else f"Missing: {missing}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:3]}..." if len(unexpected) > 3 else f"Unexpected: {unexpected}")
    
    model.eval()
    return model


def build_simple_model(checkpoint_path: str, num_classes: int = 63, img_size: int = 224, device: str = 'cuda'):
    """Build a simple DynamicVis classifier (without FPN).
    
    This matches our training config for fine-tuning.
    """
    from mmpretrain.models import ImageClassifier
    
    model_cfg = dict(
        type='ImageClassifier',
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
    
    model = MODELS.build(model_cfg)
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
    
    # Remove 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
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


def create_val_dataset(img_size: int = 224):
    """Create validation dataset."""
    val_pipeline = [
        dict(type='LoadImageFromS3', to_float32=True),
        dict(type='ResizeEdge', scale=int(img_size * 1.14), edge='short'),
        dict(type='CenterCrop', crop_size=img_size),
        dict(type='PackInputs'),
    ]
    
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


def create_val_dataset_for_pretrain(img_size: int = 512):
    """Create validation dataset for pretrain model (larger images)."""
    bgr_mean = [103.53, 116.28, 123.675]  # BGR order
    
    val_pipeline = [
        dict(type='LoadImageFromS3', to_float32=True),
        dict(type='Resize', scale=(img_size, img_size), keep_ratio=True),
        dict(type='Pad', size=(img_size, img_size), pad_val=dict(img=tuple(bgr_mean))),
        dict(type='PackInputs'),
    ]
    
    dataset = FMoWS3Dataset(
        bucket='spacenet-dataset',
        s3_prefix='Hosted-Datasets/fmow/fmow-rgb',
        manifest_key='Hosted-Datasets/fmow/fmow-rgb/manifest.json.bz2',
        local_manifest='data/manifest.json.bz2',
        split='val',
        pipeline=val_pipeline,
        enable_prefetch=True,
        prefetch_size=64,  # Larger images need less prefetch
        num_prefetch_workers=4,
    )
    
    return dataset


def evaluate_model(
    model: nn.Module,
    dataset,
    model_type: str,
    num_samples: int = None,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = 'cuda',
):
    """Evaluate model on dataset."""
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
    
    # Use mmpretrain's collate function for DataSample objects
    from mmengine.dataset import pseudo_collate
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=pseudo_collate,  # Handles DataSample objects
    )
    
    # Normalization for pretrain model (DetDataPreprocessor handles this differently)
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
                # pseudo_collate returns a dict with lists:
                # {'inputs': [tensor, tensor, ...], 'data_samples': [DataSample, ...]}
                if isinstance(data, dict):
                    # Stack inputs (list of tensors -> batched tensor)
                    if isinstance(data['inputs'], list):
                        inputs = torch.stack(data['inputs']).to(device).float()
                    else:
                        inputs = data['inputs'].to(device).float()
                    
                    # Collect labels from data_samples
                    if 'data_samples' in data:
                        data_samples_list = data['data_samples']
                        labels = []
                        for ds in data_samples_list:
                            if hasattr(ds, 'gt_label'):
                                labels.append(ds.gt_label.item())
                        labels = torch.tensor(labels).to(device)
                    else:
                        labels = data.get('gt_label', data.get('labels')).to(device)
                        data_samples_list = None
                elif isinstance(data, list):
                    # Old format: list of dicts
                    inputs_list = [d['inputs'] for d in data]
                    inputs = torch.stack(inputs_list).to(device).float()
                    
                    labels = []
                    data_samples_list = []
                    for d in data:
                        ds = d['data_samples']
                        if hasattr(ds, 'gt_label'):
                            labels.append(ds.gt_label.item())
                        data_samples_list.append(ds)
                    labels = torch.tensor(labels).to(device)
                else:
                    inputs, labels = data
                    inputs = inputs.to(device).float()
                    labels = labels.to(device)
                    data_samples_list = None
                
                # Forward pass
                if model_type == 'pretrain':
                    # For pretrain model, we need to add full-image bboxes to data_samples
                    # because the model uses RoI extraction
                    from mmdet.structures import DetDataSample
                    from mmengine.structures import InstanceData
                    
                    h, w = inputs.shape[2], inputs.shape[3]
                    batch_size_curr = inputs.shape[0]
                    
                    # Create DetDataSample with full-image bbox for each sample
                    det_samples = []
                    for i in range(batch_size_curr):
                        det_sample = DetDataSample()
                        # Full image bbox: [x1, y1, x2, y2]
                        bbox = torch.tensor([[0, 0, w, h]], dtype=torch.float32, device=device)
                        gt_instances = InstanceData()
                        gt_instances.bboxes = bbox
                        gt_instances.labels = torch.tensor([labels[i].item()], dtype=torch.long, device=device)
                        det_sample.gt_instances = gt_instances
                        det_sample.set_metainfo({
                            'img_shape': (h, w),
                            'ori_shape': (h, w),
                            'scale_factor': (1.0, 1.0),
                            'batch_input_shape': (h, w),
                        })
                        det_samples.append(det_sample)
                    
                    # Use the model's data_preprocessor
                    batch_inputs = model.data_preprocessor({'inputs': inputs, 'data_samples': det_samples})
                    
                    # Get processed inputs and samples
                    processed_inputs = batch_inputs['inputs']
                    processed_samples = batch_inputs.get('data_samples', det_samples)
                    
                    outputs = model(processed_inputs, data_samples=processed_samples, mode='predict')
                    
                    # Extract predictions from data samples
                    for i, ds in enumerate(outputs):
                        if hasattr(ds, 'pred_score'):
                            score = ds.pred_score.cpu().numpy()
                            pred = score.argmax()
                            all_scores.append(score)
                            all_preds.append(pred)
                            all_labels.append(labels[i].cpu().item())
                else:
                    # Simple model - normalize and forward
                    inputs = (inputs - mean) / std
                    outputs = model(inputs)
                    
                    # Handle different output formats
                    if isinstance(outputs, (list, tuple)):
                        logits = outputs[0]
                    elif hasattr(outputs, 'head_outputs'):
                        logits = outputs.head_outputs
                    elif isinstance(outputs, torch.Tensor):
                        logits = outputs
                    else:
                        continue
                    
                    scores = torch.softmax(logits, dim=1)
                    preds = scores.argmax(dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_scores.extend(scores.cpu().numpy())
                
                # Update progress
                if len(all_preds) > 0:
                    current_acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
                    pbar.set_postfix({'acc': f'{current_acc:.2f}%'})
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores) if len(all_scores) > 0 else None
    
    return compute_metrics(all_preds, all_labels, all_scores, num_classes=63)


def main():
    parser = argparse.ArgumentParser(description='Evaluate DynamicVis on fMoW')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, choices=['auto', 'simple', 'pretrain'], default='auto',
                        help='Model architecture type (auto-detected by default)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for evaluation (auto-set based on model type)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--img-size', type=int, default=None,
                        help='Input image size (auto-set based on model type)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for evaluation')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Detect model type if auto
    if args.model_type == 'auto':
        print("Detecting model architecture...")
        args.model_type = detect_model_type(args.checkpoint)
        print(f"Detected model type: {args.model_type}")
    
    # Set defaults based on model type
    if args.model_type == 'pretrain':
        img_size = args.img_size or 512
        batch_size = args.batch_size or 8  # Smaller for 512x512 images
    else:
        img_size = args.img_size or 224
        batch_size = args.batch_size or 32
    
    print("=" * 60)
    print("DynamicVis Evaluation on fMoW")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Num samples: {args.num_samples or 'all'}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Build model
    print("\nLoading model...")
    if args.model_type == 'pretrain':
        model = build_pretrain_model(args.checkpoint, num_classes=63, img_size=img_size, device=args.device)
    else:
        model = build_simple_model(args.checkpoint, num_classes=63, img_size=img_size, device=args.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {num_params:.2f}M")
    
    # Create dataset - need to switch back to mmpretrain scope for transforms
    print("\nLoading validation dataset...")
    from mmengine.registry import init_default_scope
    init_default_scope('mmpretrain')  # Reset scope for dataset transforms
    
    if args.model_type == 'pretrain':
        dataset = create_val_dataset_for_pretrain(img_size=img_size)
    else:
        dataset = create_val_dataset(img_size=img_size)
    
    # Evaluate
    print("\nStarting evaluation...")
    start_time = time.time()
    
    metrics = evaluate_model(
        model=model,
        dataset=dataset,
        model_type=args.model_type,
        num_samples=args.num_samples,
        batch_size=batch_size,
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
