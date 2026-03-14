#!/usr/bin/env python
"""
Training script for composition-aware DynamicVis on fMoW.

Identical to ``train_dynamicvis_pretrain.py`` but defaults to the
composition config/work-dir and skips S3-specific logic.

Prerequisites:
    1. ``embed_patches.py`` — cache DINOv3 patch embeddings.
    2. ``cluster_viz.py --save-cluster-data`` — produce cluster data.

Usage:
    # Single GPU
    python train_dynamicvis_composition.py \\
        configs_dynamicvis/fmow_composition/dynamicvis_b_fmow_composition.py

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=2 train_dynamicvis_composition.py \\
        configs_dynamicvis/fmow_composition/dynamicvis_b_fmow_composition.py

    # Resume
    python train_dynamicvis_composition.py \\
        configs_dynamicvis/fmow_composition/dynamicvis_b_fmow_composition.py \\
        --resume auto
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
DYNAMICVIS_PATH = Path(__file__).parent / "architectures" / "DynamicVis"
sys.path.insert(0, str(DYNAMICVIS_PATH))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DynamicVis with composition-aware loss on fMoW"
    )
    p.add_argument("config", help="Config file path")
    p.add_argument("--work-dir", help="Override work directory")
    p.add_argument(
        "--resume", nargs="?", type=str, const="auto",
        help='Resume from checkpoint ("auto" = latest)',
    )
    p.add_argument("--load-from", type=str, help="Load pretrained weights")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument(
        "--cfg-options", nargs="+", action=DictAction,
        help="Override config values: key=value",
    )
    p.add_argument(
        "--launcher", choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
    )
    p.add_argument("--local_rank", "--local-rank", type=int, default=0)

    # Convenience overrides
    p.add_argument("--batch-size", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--cluster-data-dir", type=str,
                   help="Override cluster_data_dir in config")
    p.add_argument("--l-comp", type=float, default=None,
                   help="Weight for cosine alignment loss (lambda_comp)")
    p.add_argument("--l-contrast", type=float, default=None,
                   help="Weight for InfoNCE contrastive loss (lambda_contrast)")
    p.add_argument("--l-smooth", type=float, default=None,
                   help="Weight for spatial smoothness loss (lambda_smooth)")
    p.add_argument(
        "--dist-backend", type=str, choices=["nccl", "gloo"],
        default=None,
        help="Override distributed backend",
    )

    args = p.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


# ── Merge CLI → config ───────────────────────────────────────────────

def merge_args(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply CLI overrides to the loaded MMEngine config."""

    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    if args.dist_backend is not None:
        cfg.env_cfg.dist_cfg = dict(backend=args.dist_backend)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = os.path.join(
            "./outputs",
            os.path.splitext(os.path.basename(args.config))[0],
        )

    if args.load_from is not None:
        cfg.load_from = args.load_from

    if args.resume == "auto":
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # ── Batch size ──
    if args.batch_size is not None:
        cfg.train_dataloader.batch_size = args.batch_size
        if cfg.get("val_dataloader"):
            cfg.val_dataloader.batch_size = args.batch_size

    # ── Epochs ──
    if args.epochs is not None:
        max_epochs = args.epochs
        cfg.train_cfg.max_epochs = max_epochs
        warmup_end = min(2, max(1, max_epochs // 10))
        if max_epochs <= 2:
            cfg.param_scheduler = [
                dict(type="ConstantLR", factor=1.0,
                     by_epoch=True, begin=0, end=max_epochs)
            ]
        else:
            for sched in cfg.param_scheduler:
                if sched.get("type") == "LinearLR":
                    sched["end"] = warmup_end
                elif sched.get("type") == "CosineAnnealingLR":
                    sched["begin"] = warmup_end
                    sched["end"] = max_epochs

    # ── LR ──
    if args.lr is not None:
        cfg.optim_wrapper.optimizer.lr = args.lr

    # ── WandB ──
    if args.no_wandb:
        for key in ("vis_backends",):
            if key in cfg:
                cfg[key] = [
                    b for b in cfg[key]
                    if b.get("type") != "WandbVisBackend"
                ]
        if "visualizer" in cfg and "vis_backends" in cfg.visualizer:
            cfg.visualizer.vis_backends = [
                b for b in cfg.visualizer.vis_backends
                if b.get("type") != "WandbVisBackend"
            ]

    # ── Max samples ──
    if args.max_samples is not None:
        cfg.train_dataloader.dataset.max_samples = args.max_samples
        if cfg.get("val_dataloader"):
            cfg.val_dataloader.dataset.max_samples = min(
                args.max_samples // 5, 5000
            )

    # ── Cluster data dir ──
    if args.cluster_data_dir is not None:
        cdd = os.path.abspath(args.cluster_data_dir)
        cfg.train_dataloader.dataset.cluster_data_dir = cdd
        if cfg.get("val_dataloader"):
            cfg.val_dataloader.dataset.cluster_data_dir = cdd

    # ── Loss component weights ──
    if args.l_comp is not None:
        cfg.model.head.lambda_comp = args.l_comp
    if args.l_contrast is not None:
        cfg.model.head.lambda_contrast = args.l_contrast
    if args.l_smooth is not None:
        cfg.model.head.lambda_smooth = args.l_smooth

    # ── Arbitrary overrides ──
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Wandb key
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key

    # Load + merge config
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)

    print("=" * 60)
    print("DynamicVis Composition-Aware Training on fMoW")
    print("=" * 60)
    print(f"Config:      {args.config}")
    print(f"Work dir:    {cfg.work_dir}")
    print(f"Batch size:  {cfg.train_dataloader.batch_size}")
    print(f"Max epochs:  {cfg.train_cfg.max_epochs}")
    print(f"LR:          {cfg.optim_wrapper.optimizer.lr}")
    print(f"Cluster dir: {cfg.train_dataloader.dataset.cluster_data_dir}")
    print(f"Loss type:   {cfg.model.head.get('loss_type', 'cosine')}")
    print(f"λ_comp:      {cfg.model.head.lambda_comp}")
    print(f"λ_contrast:  {cfg.model.head.lambda_contrast}")
    print(f"λ_smooth:    {cfg.model.head.lambda_smooth}")
    print("=" * 60)

    runner = Runner.from_cfg(cfg)
    runner.train()

    # ── Save final model weights ──────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        import torch
        work_dir = Path(cfg.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Extract the raw model (unwrap DDP if needed)
        model = runner.model
        if hasattr(model, "module"):
            model = model.module

        # Save full model state dict
        final_path = work_dir / "final_model.pth"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "meta": {
                    "epoch": cfg.train_cfg.max_epochs,
                    "config": args.config,
                },
            },
            final_path,
        )
        print(f"\nFinal model weights saved to: {final_path}")

        # Also save just the backbone weights (for downstream tasks)
        backbone_path = work_dir / "final_backbone.pth"
        backbone_sd = {
            k.replace("backbone.", ""): v
            for k, v in model.state_dict().items()
            if k.startswith("backbone.")
        }
        torch.save({"state_dict": backbone_sd}, backbone_path)
        print(f"Backbone weights saved to:    {backbone_path}")


if __name__ == "__main__":
    main()
