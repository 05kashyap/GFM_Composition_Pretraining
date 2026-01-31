#!/usr/bin/env python
"""
Training script for DynamicVis on fMoW dataset using MMEngine.
Streams data directly from AWS S3 without downloading the full dataset.

Usage:
    # Single GPU training
    python train_dynamicvis.py configs_dynamicvis/fmow_classification/dynamicvis_b_fmow_s3.py
    
    # Multi-GPU training (e.g., 4 GPUs)
    torchrun --nproc_per_node=4 train_dynamicvis.py configs_dynamicvis/fmow_classification/dynamicvis_b_fmow_s3.py
    
    # Resume training
    python train_dynamicvis.py configs_dynamicvis/fmow_classification/dynamicvis_b_fmow_s3.py --resume auto
"""

import sys
import os
import argparse
from pathlib import Path

# Add DynamicVis to path BEFORE any other imports
DYNAMICVIS_PATH = Path(__file__).parent / "architectures" / "DynamicVis"
sys.path.insert(0, str(DYNAMICVIS_PATH))
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.registry import RUNNERS


def parse_args():
    parser = argparse.ArgumentParser(description='Train DynamicVis on fMoW with S3 streaming')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='Resume from checkpoint. Use "auto" to resume from latest.'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Enable automatic-mixed-precision training'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Whether not to evaluate during training'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override settings in config. Format: key=value'
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher'
    )
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    
    # Custom options for fMoW S3
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    
    args = parser.parse_args()
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    return args


def setup_wandb_env():
    """Setup wandb environment variables from .env if available."""
    wandb_key = os.getenv('WANDB_API_KEY')
    if wandb_key:
        os.environ['WANDB_API_KEY'] = wandb_key
        print("WandB API key loaded from environment.")
    else:
        print("WARNING: WANDB_API_KEY not found. WandB may prompt for login.")


def merge_args(cfg, args):
    """Merge CLI arguments into config."""
    # Disable validation if requested
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None
    
    cfg.launcher = args.launcher
    
    # Work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./outputs', os.path.splitext(os.path.basename(args.config))[0])
    
    # AMP training
    if args.amp:
        if 'optim_wrapper' in cfg:
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')
    
    # Resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume
    
    # Override batch size
    if args.batch_size is not None:
        cfg.train_dataloader.batch_size = args.batch_size
        if cfg.get('val_dataloader'):
            cfg.val_dataloader.batch_size = args.batch_size
    
    # Override epochs
    if args.epochs is not None:
        max_epochs = args.epochs
        cfg.train_cfg.max_epochs = max_epochs
        
        # Update all schedulers to handle epoch range properly
        warmup_epochs = min(5, max_epochs // 2) if max_epochs > 1 else 0
        
        new_schedulers = []
        for scheduler in cfg.param_scheduler:
            if scheduler.get('type') == 'LinearLR':
                if max_epochs > warmup_epochs:
                    scheduler['end'] = warmup_epochs
                    new_schedulers.append(scheduler)
                # Skip warmup for very short runs
            elif scheduler.get('type') == 'CosineAnnealingLR':
                scheduler['begin'] = warmup_epochs
                scheduler['end'] = max_epochs
                new_schedulers.append(scheduler)
            else:
                new_schedulers.append(scheduler)
        
        # If only 1-2 epochs, use simple constant LR
        if max_epochs <= 2:
            cfg.param_scheduler = [
                dict(
                    type='ConstantLR',
                    factor=1.0,
                    by_epoch=True,
                    begin=0,
                    end=max_epochs,
                )
            ]
        else:
            cfg.param_scheduler = new_schedulers
    
    # Override learning rate
    if args.lr is not None:
        cfg.optim_wrapper.optimizer.lr = args.lr
        cfg.base_lr = args.lr
    
    # Disable wandb if requested
    if args.no_wandb:
        # Remove wandb from vis_backends
        if 'vis_backends' in cfg:
            cfg.vis_backends = [
                b for b in cfg.vis_backends 
                if b.get('type') != 'WandbVisBackend'
            ]
        if 'visualizer' in cfg and 'vis_backends' in cfg.visualizer:
            cfg.visualizer.vis_backends = [
                b for b in cfg.visualizer.vis_backends 
                if b.get('type') != 'WandbVisBackend'
            ]
    
    # Merge cfg-options
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    return cfg


def main():
    args = parse_args()
    
    # Verify AWS credentials
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("WARNING: AWS credentials not found in environment.")
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file.")
        print("Attempting to use unsigned requests (may not work for all buckets)...")
    else:
        print("AWS credentials loaded from environment.")
    
    # Setup wandb
    if not args.no_wandb:
        setup_wandb_env()
    
    # Load config
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    
    # Print config summary
    print("\n" + "=" * 60)
    print("Training Configuration Summary")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Work directory: {cfg.work_dir}")
    print(f"Batch size: {cfg.train_dataloader.batch_size}")
    print(f"Max epochs: {cfg.train_cfg.max_epochs}")
    print(f"Learning rate: {cfg.optim_wrapper.optimizer.lr}")
    print(f"Image size: {cfg.get('img_size', 224)}")
    print(f"Model: {cfg.model.backbone.type} ({cfg.model.backbone.arch})")
    print(f"Launcher: {cfg.launcher}")
    print(f"WandB: {'Enabled' if not args.no_wandb else 'Disabled'}")
    print("=" * 60 + "\n")
    
    # Build the runner
    runner = Runner.from_cfg(cfg)
    
    # Start training
    runner.train()


if __name__ == '__main__':
    main()
