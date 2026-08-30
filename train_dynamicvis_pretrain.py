#!/usr/bin/env python
"""
Training script for DynamicVis Pretrain on fMoW dataset using MMEngine.
Streams data directly from AWS S3 without downloading the full 350GB dataset.

This script trains the full DynamicVis pretrain model with:
- FPN neck for multi-scale features
- RoI extraction with bounding box annotations
- Multi-instance learning (MIL) classification

Usage:
    # Single GPU training
    python train_dynamicvis_pretrain.py configs_dynamicvis/fmow_pretrain/dynamicvis_b_fmow_s3_pretrain.py
    
    # Multi-GPU training (e.g., 4 GPUs)
    torchrun --nproc_per_node=4 train_dynamicvis_pretrain.py configs_dynamicvis/fmow_pretrain/dynamicvis_b_fmow_s3_pretrain.py
    
    # Resume training
    python train_dynamicvis_pretrain.py configs_dynamicvis/fmow_pretrain/dynamicvis_b_fmow_s3_pretrain.py --resume auto
    
    # Fine-tune from pretrained weights
    python train_dynamicvis_pretrain.py configs_dynamicvis/fmow_pretrain/dynamicvis_b_fmow_s3_pretrain.py --load-from /path/to/pretrained.pth
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
    parser = argparse.ArgumentParser(description='Train DynamicVis Pretrain on fMoW with S3 streaming')
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
        '--load-from',
        type=str,
        help='Load pretrained weights (for fine-tuning)'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Enable automatic-mixed-precision training (default: on via config)'
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
    
    # Custom options
    parser.add_argument('--batch-size', type=int, help='Override batch size per GPU')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--max-samples', type=int, help='Limit training samples (for debugging)')
    parser.add_argument('--use-rgb', action='store_true', help='Use full RGB images (larger) instead of msrgb')
    parser.add_argument('--data-root', type=str, help='Local data directory (downloaded via scripts/download_fmow.py)')
    parser.add_argument(
        '--dist-backend',
        type=str,
        choices=['nccl', 'gloo'],
        default=None,
        help='Override distributed backend (nccl for <=2 MIG slices, gloo for >2)'
    )
    
    args = parser.parse_args()
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    return args


def setup_wandb_env():
    """Setup wandb environment variables."""
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
    
    # Override distributed backend (nccl vs gloo)
    if args.dist_backend is not None:
        cfg.env_cfg.dist_cfg = dict(backend=args.dist_backend)
        if args.dist_backend == 'gloo':
            print(f"Using gloo backend (CPU-based collectives for multi-MIG DDP)")
    
    # Work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./outputs', os.path.splitext(os.path.basename(args.config))[0])
    
    # Load pretrained weights
    if args.load_from is not None:
        cfg.load_from = args.load_from
    
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
        
        # Update schedulers - handle very short runs properly
        warmup_epochs = min(5, max(1, max_epochs // 4))  # At least 1 epoch warmup
        
        if max_epochs <= 2:
            # For very short runs, use constant LR
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
            for scheduler in cfg.param_scheduler:
                if scheduler.get('type') == 'LinearLR':
                    scheduler['end'] = warmup_epochs
                elif scheduler.get('type') == 'CosineAnnealingLR':
                    scheduler['begin'] = warmup_epochs
                    scheduler['end'] = max_epochs
    
    # Override learning rate
    if args.lr is not None:
        cfg.optim_wrapper.optimizer.lr = args.lr
        cfg.base_lr = args.lr
    
    # Disable wandb if requested
    if args.no_wandb:
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
    else:
        # Update WandB config with all hyperparameters
        wandb_config = {
            'batch_size': cfg.train_dataloader.batch_size,
            'epochs': cfg.train_cfg.max_epochs,
            'learning_rate': cfg.optim_wrapper.optimizer.lr,
            'optimizer': cfg.optim_wrapper.optimizer.type,
            'weight_decay': cfg.optim_wrapper.optimizer.get('weight_decay', 0),
            'img_size': cfg.img_size,
            'num_classes': cfg.num_classes,
            'model_arch': cfg.model.backbone.get('arch', 'unknown'),
            'use_msrgb': cfg.train_dataloader.dataset.get('use_msrgb', True),
            'data_root': cfg.train_dataloader.dataset.get('data_root', 'S3'),
            'val_interval': cfg.train_cfg.get('val_interval', 1),
            'num_workers': cfg.train_dataloader.get('num_workers', 0),
            'amp_enabled': 'Amp' in cfg.optim_wrapper.get('type', ''),
        }
        
        # Update wandb init_kwargs in vis_backends
        for backend in cfg.get('vis_backends', []):
            if backend.get('type') == 'WandbVisBackend':
                backend['init_kwargs']['config'] = wandb_config
                backend['init_kwargs']['allow_val_change'] = True
        
        # Also update in visualizer.vis_backends
        if 'visualizer' in cfg and 'vis_backends' in cfg.visualizer:
            for backend in cfg.visualizer.vis_backends:
                if backend.get('type') == 'WandbVisBackend':
                    backend['init_kwargs']['config'] = wandb_config
                    backend['init_kwargs']['allow_val_change'] = True
    
    # Limit samples for debugging
    if args.max_samples is not None:
        cfg.train_dataloader.dataset.max_samples = args.max_samples
        if cfg.get('val_dataloader'):
            cfg.val_dataloader.dataset.max_samples = min(args.max_samples // 5, 5000)
    
    # Use RGB instead of msrgb
    if args.use_rgb:
        cfg.train_dataloader.dataset.use_msrgb = False
        if cfg.get('val_dataloader'):
            cfg.val_dataloader.dataset.use_msrgb = False
    
    # Use local data directory (for pre-downloaded data)
    if args.data_root:
        # Convert to absolute path to avoid working directory issues
        data_root_abs = os.path.abspath(args.data_root)
        cfg.train_dataloader.dataset.data_root = data_root_abs
        if cfg.get('val_dataloader'):
            cfg.val_dataloader.dataset.data_root = data_root_abs
        print(f"Using local data from: {data_root_abs}")
    
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
        print("Attempting to use unsigned requests for public S3 bucket...")
    
    # Setup wandb
    setup_wandb_env()
    
    # Load config
    print(f"Loading config from {args.config}")
    cfg = Config.fromfile(args.config)
    
    # Merge command line arguments
    cfg = merge_args(cfg, args)
    
    print("=" * 60)
    print("DynamicVis Pretrain Training on fMoW (S3 Streaming)")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Work dir: {cfg.work_dir}")
    print(f"Batch size: {cfg.train_dataloader.batch_size}")
    print(f"Max epochs: {cfg.train_cfg.max_epochs}")
    print(f"Learning rate: {cfg.optim_wrapper.optimizer.lr}")
    print(f"Image size: {cfg.img_size}")
    print(f"Using {'msrgb' if cfg.train_dataloader.dataset.get('use_msrgb', True) else 'rgb'} images")
    print("=" * 60)
    
    # Build runner
    runner = Runner.from_cfg(cfg)
    
    # Start training
    runner.train()


if __name__ == '__main__':
    main()
