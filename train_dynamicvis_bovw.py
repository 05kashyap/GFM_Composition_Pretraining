#!/usr/bin/env python
"""
Training script for BoVW (Bag of Visual Words) DynamicVis on fMoW.

This is Phase 4 of the BoVW composition training pipeline. It trains the
DynamicVis backbone to predict soft histogram distributions over a visual
vocabulary using Sinkhorn EMD loss.

Prerequisites:
    1. ``extract_patch_tokens.py`` — extract DINOv3 patch tokens.
    2. ``build_vocabulary.py`` — build K-means visual vocabulary.
    3. ``generate_histograms.py`` — generate histogram targets.

Usage:
    # Single GPU
    python train_dynamicvis_bovw.py

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=8 train_dynamicvis_bovw.py

    # With options
    python train_dynamicvis_bovw.py \\
        --manifest data/fmow_manifest_train.json \\
        --histogram-dir outputs/bovw_histograms \\
        --vocab-dir outputs/bovw_vocabulary \\
        --pretrained-backbone weights/pretrain_dynamicvis_b_bf16_mamba_best_single-label_f1-score_epoch_170.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist

# Ensure imports work
_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "architectures" / "DynamicVis"))

from dotenv import load_dotenv
load_dotenv()

# Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Import DynamicVis package so MODELS.register_module decorators execute and
# DynamicVisBackbone becomes available to mmpretrain's registry.
import dynamicvis  # noqa: F401

# Import our modules
from datasets.fmow_bovw_dataset import FMoWBoVWDataset, bovw_collate_fn
from models.bovw_head import BoVWDynamicVis
from losses.bovw_loss import BoVWLoss

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DynamicVis with BoVW histogram prediction on fMoW"
    )

    # Data paths
    p.add_argument("--manifest", type=str, default="data/fmow_manifest_train.json",
                   help="Path to manifest.json")
    p.add_argument("--histogram-dir", type=str, default="outputs/bovw_histograms",
                   help="Path to Phase 3 output (histograms.npy)")
    p.add_argument("--vocab-dir", type=str, default="outputs/bovw_vocabulary",
                   help="Path to Phase 2 output (ground_cost.npy)")
    p.add_argument("--cell-labels", type=str, default="outputs/bovw_histograms/cell_labels.npy",
                   help="Path to cell_labels.npy")
    p.add_argument("--data-root", type=str, default="data/fmow",
                   help="Root directory for fMoW images")

    # Model
    p.add_argument("--pretrained-backbone", type=str,
                   default="weights/pretrain_dynamicvis_b_bf16_mamba_best_single-label_f1-score_epoch_170.pth",
                   help="Path to pretrained backbone weights")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train from scratch without loading pretrained backbone")
    p.add_argument("--vocab-size", type=int, default=512,
                   help="Visual vocabulary size (K)")
    p.add_argument("--hidden-dim", type=int, default=512,
                   help="Hidden dimension in prediction head")
    p.add_argument("--num-classes", type=int, default=63,
                   help="Number of fMoW classes for aux head")

    # Loss
    p.add_argument("--lambda-emd", type=float, default=1.0,
                   help="Weight for Sinkhorn EMD loss")
    p.add_argument("--lambda-cls", type=float, default=0.5,
                   help="Weight for auxiliary classification loss")
    p.add_argument("--lambda-mil", type=float, default=0.25,
                   help="Weight for MIL contrastive loss (CLIP-style)")
    p.add_argument("--sinkhorn-eps", type=float, default=0.05,
                   help="Sinkhorn regularization epsilon")
    p.add_argument("--sinkhorn-iters", type=int, default=50,
                   help="Sinkhorn iterations")

    # Training
    p.add_argument("--batch-size", type=int, default=32,
                   help="Per-GPU batch size")
    p.add_argument("--num-epochs", type=int, default=100,
                   help="Number of training epochs")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Base learning rate")
    p.add_argument("--weight-decay", type=float, default=0.05,
                   help="Weight decay for AdamW")
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="Number of warmup epochs")
    p.add_argument("--min-lr", type=float, default=5e-6,
                   help="Minimum learning rate after cosine decay")
    p.add_argument("--grad-clip", type=float, default=5.0,
                   help="Gradient clipping max norm")
    p.add_argument("--num-views", type=int, default=2,
                   help="Number of augmented views per sample")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers per GPU")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Max training samples (for storage-constrained runs)")

    # Output
    p.add_argument("--output-dir", type=str, default="outputs/bovw_training",
                   help="Output directory for checkpoints")
    p.add_argument("--log-interval", type=int, default=20,
                   help="Log every N iterations")
    p.add_argument("--save-interval", type=int, default=10,
                   help="Save checkpoint every N epochs")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to checkpoint to resume training from")
    p.add_argument("--reset-optim-on-resume", action="store_true",
                   help="Resume model weights but reinitialize optimizer/scaler state")

    # W&B
    p.add_argument("--wandb-project", type=str, default="satbae-bovw",
                   help="W&B project name")
    p.add_argument("--wandb-run-name", type=str, default=None,
                   help="W&B run name (auto-generated if not specified)")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable W&B logging")

    # Distributed
    p.add_argument("--local_rank", "--local-rank", type=int, default=0)
    p.add_argument("--dist-backend", type=str, default="nccl",
                   choices=["nccl", "gloo"])

    args = p.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return rank, world_size, local_rank
    return 0, 1, 0


def get_lr_scheduler(optimizer, warmup_epochs, num_epochs, min_lr, base_lr):
    """Create warmup + cosine decay scheduler."""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay
            import math
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return min_lr / base_lr + 0.5 * (1 - min_lr / base_lr) * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_pretrained_backbone(model: nn.Module, checkpoint_path: str, rank: int = 0):
    """Load pretrained backbone weights."""
    if not Path(checkpoint_path).exists():
        if rank == 0:
            print(f"Warning: Pretrained weights not found: {checkpoint_path}")
        return

    if rank == 0:
        print(f"Loading pretrained backbone from: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle common checkpoint formats
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")

    # Transform keys: strip 'backbone.' prefix if present
    transformed_dict = {}
    n_stripped = 0
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            new_key = k[len("backbone."):]
            n_stripped += 1
        else:
            new_key = k
        transformed_dict[new_key] = v

    # Get model backbone (handle DDP wrapper)
    backbone = model.module.backbone if hasattr(model, 'module') else model.backbone

    # Load into backbone
    missing, unexpected = backbone.load_state_dict(transformed_dict, strict=False)

    if rank == 0:
        n_total = len(list(backbone.parameters()))
        n_loaded = n_total - len(missing)
        print(f"  Loaded: {n_loaded}/{n_total} parameter groups")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")


def resume_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    resume_path: Optional[str],
    current_world_size: int = 1,
    current_batch_size: int = 1,
    reset_optim_on_resume: bool = False,
    rank: int = 0,
) -> tuple[int, float]:
    """Resume model and optimizer/scheduler/scaler states from a checkpoint."""
    if not resume_path:
        return 0, float('inf')

    ckpt_path = Path(resume_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    if rank == 0:
        print(f"Resuming training from checkpoint: {resume_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Resume checkpoint must be a dict containing 'state_dict'.")

    # Checkpoints are saved from the raw model, not the DDP wrapper.
    # Load into the unwrapped module so keys match exactly.
    raw_model = model.module if hasattr(model, "module") else model

    missing, unexpected = raw_model.load_state_dict(ckpt["state_dict"], strict=False)
    if rank == 0:
        print(f"  Model state loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    ckpt_world_size = ckpt.get("world_size")
    ckpt_batch_size = ckpt.get("batch_size")
    restore_optimizer_state = True
    restore_scaler_state = True
    if reset_optim_on_resume:
        restore_optimizer_state = False
        restore_scaler_state = False
        if rank == 0:
            print("  --reset-optim-on-resume set: skipping optimizer and scaler restore")
    elif ckpt_world_size is not None and ckpt_batch_size is not None:
        if ckpt_world_size != current_world_size or ckpt_batch_size != current_batch_size:
            restore_optimizer_state = False
            restore_scaler_state = False
            if rank == 0:
                print(
                    "  Warning: checkpoint was created with "
                    f"world_size={ckpt_world_size}, batch_size={ckpt_batch_size}; "
                    f"current run uses world_size={current_world_size}, batch_size={current_batch_size}."
                )
                print("  Skipping optimizer and scaler restore to avoid stale state.")

    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
        if rank == 0:
            print("  Scheduler state restored")
    elif rank == 0:
        print("  Warning: Scheduler state missing in checkpoint")

    if restore_optimizer_state:
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            if rank == 0:
                print("  Optimizer state restored")
        elif rank == 0:
            print("  Warning: Optimizer state missing in checkpoint")

    if restore_scaler_state and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
        if rank == 0:
            print("  GradScaler state restored")
    elif rank == 0 and not restore_scaler_state:
        print("  GradScaler state will be reinitialized for this run")

    start_epoch = int(ckpt.get("epoch", 0))
    best_loss = float(ckpt.get("loss", float('inf')))
    if rank == 0:
        print(f"  Resume epoch: {start_epoch}")
        print(f"  Best/current loss from checkpoint: {best_loss:.4f}")

    return start_epoch, best_loss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    args: argparse.Namespace,
    rank: int = 0,
    world_size: int = 1,
    use_wandb: bool = False,
):
    """Train for one epoch."""
    model.train()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_loss_emd = 0.0
    total_loss_cls = 0.0
    total_loss_mil = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(rank != 0))

    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        if isinstance(batch["inputs"], list):
            inputs = [x.to(device, non_blocking=True) for x in batch["inputs"]]
        else:
            inputs = batch["inputs"].to(device, non_blocking=True)
        data_samples = batch["data_samples"]

        optimizer.zero_grad()

        # Forward with AMP
        with autocast(dtype=torch.bfloat16):
            losses = model(inputs, data_samples, mode="loss")

        loss = losses["loss"]

        # Backward
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        total_loss += loss.item()
        total_loss_emd += losses["loss_emd"].item()
        total_loss_cls += losses["loss_cls"].item()
        total_loss_mil += losses.get("loss_mil", torch.tensor(0.0)).item()
        num_batches += 1

        # Log
        if rank == 0 and (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / num_batches
            avg_emd = total_loss_emd / num_batches
            avg_cls = total_loss_cls / num_batches
            avg_mil = total_loss_mil / num_batches
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "emd": f"{avg_emd:.4f}",
                "cls": f"{avg_cls:.4f}",
                "mil": f"{avg_mil:.4f}",
                "lr": f"{lr:.2e}",
                "grad": f"{grad_norm:.2f}",
            })

            # W&B per-iteration logging
            if use_wandb:
                global_step = epoch * len(dataloader) + batch_idx
                wandb.log({
                    "iter/loss": loss.item(),
                    "iter/loss_emd": losses["loss_emd"].item(),
                    "iter/loss_cls": losses["loss_cls"].item(),
                    "iter/loss_mil": losses.get("loss_mil", torch.tensor(0.0)).item(),
                    "iter/lr": lr,
                    "iter/grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                    "iter/step": global_step,
                }, step=global_step)

    avg_loss = total_loss / max(num_batches, 1)
    avg_emd = total_loss_emd / max(num_batches, 1)
    avg_cls = total_loss_cls / max(num_batches, 1)
    avg_mil = total_loss_mil / max(num_batches, 1)

    return avg_loss, avg_emd, avg_cls, avg_mil


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()

    # Initialize distributed
    if world_size > 1:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method="env://",
        )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("=" * 60)
        print("BoVW DynamicVis Training on fMoW")
        print("=" * 60)
        print(f"Manifest:       {args.manifest}")
        print(f"Histogram dir:  {args.histogram_dir}")
        print(f"Vocab dir:      {args.vocab_dir}")
        print(f"Output dir:     {args.output_dir}")
        print(f"Pretrained:     {'(disabled)' if args.no_pretrained else args.pretrained_backbone}")
        print(f"Vocab size:     {args.vocab_size}")
        print(f"Batch size:     {args.batch_size} x {world_size} GPUs")
        print(f"Epochs:         {args.num_epochs}")
        print(f"LR:             {args.lr}")
        print(f"λ_emd:          {args.lambda_emd}")
        print(f"λ_cls:          {args.lambda_cls}")
        print(f"λ_mil:          {args.lambda_mil}")
        print(f"Sinkhorn eps:   {args.sinkhorn_eps}")
        print(f"Sinkhorn iters: {args.sinkhorn_iters}")
        print(f"Max samples:    {args.max_samples if args.max_samples else 'all'}")
        print("=" * 60)

    # Initialize W&B
    use_wandb = WANDB_AVAILABLE and not args.no_wandb and rank == 0
    if use_wandb:
        wandb_run_name = args.wandb_run_name or f"bovw_K{args.vocab_size}_lr{args.lr}_e{args.sinkhorn_eps}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config={
                "manifest": args.manifest,
                "histogram_dir": args.histogram_dir,
                "vocab_dir": args.vocab_dir,
                "vocab_size": args.vocab_size,
                "hidden_dim": args.hidden_dim,
                "num_classes": args.num_classes,
                "lambda_emd": args.lambda_emd,
                "lambda_cls": args.lambda_cls,
                "lambda_mil": args.lambda_mil,
                "sinkhorn_eps": args.sinkhorn_eps,
                "sinkhorn_iters": args.sinkhorn_iters,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_epochs": args.warmup_epochs,
                "min_lr": args.min_lr,
                "grad_clip": args.grad_clip,
                "num_views": args.num_views,
                "world_size": world_size,
                "max_samples": args.max_samples,
            },
        )
        print(f"W&B initialized: {wandb.run.url}")
    elif rank == 0 and not WANDB_AVAILABLE:
        print("W&B not available (wandb not installed)")
    elif rank == 0 and args.no_wandb:
        print("W&B disabled via --no-wandb")

    # Create output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    train_dataset = FMoWBoVWDataset(
        manifest_path=args.manifest,
        histogram_dir=args.histogram_dir,
        cell_labels_path=args.cell_labels,
        data_root=args.data_root,
        img_size=512,
        split="train",
        num_views=args.num_views,
        max_samples=args.max_samples,
    )

    # Create sampler for distributed training
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        sampler = None

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=bovw_collate_fn,
    )

    if rank == 0:
        print(f"Dataset: {len(train_dataset)} samples")
        print(f"Batches per epoch: {len(train_loader)}")

    # Create model
    ground_cost_path = Path(args.vocab_dir) / "ground_cost.npy"
    if not ground_cost_path.exists():
        if rank == 0:
            print(f"Warning: Ground cost not found: {ground_cost_path}")
        ground_cost_path = None
    else:
        ground_cost_path = str(ground_cost_path)

    model = BoVWDynamicVis(
        backbone=dict(
            type='DynamicVisBackbone',
            arch='b',
            out_type='avg_featmap',
            out_indices=(3,),
        ),
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        ground_cost_path=ground_cost_path,
        lambda_emd=args.lambda_emd,
        lambda_cls=args.lambda_cls,
        lambda_mil=args.lambda_mil,
        sinkhorn_eps=args.sinkhorn_eps,
        sinkhorn_iters=args.sinkhorn_iters,
    )

    model = model.to(device)

    # Load pretrained backbone (unless --no-pretrained is set)
    if args.no_pretrained:
        if rank == 0:
            print("Training from scratch (--no-pretrained)")
    else:
        load_pretrained_backbone(model, args.pretrained_backbone, rank)

    # Wrap with DDP
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # Create scheduler
    scheduler = get_lr_scheduler(
        optimizer, args.warmup_epochs, args.num_epochs,
        args.min_lr, args.lr
    )

    # Create grad scaler for AMP
    scaler = GradScaler()

    # Optionally resume full training state
    start_epoch = 0
    best_loss = float('inf')
    if args.resume_from:
        start_epoch, best_loss = resume_training_state(
            model, optimizer, scheduler, scaler, args.resume_from,
            current_world_size=world_size,
            current_batch_size=args.batch_size,
            reset_optim_on_resume=args.reset_optim_on_resume,
            rank=rank,
        )

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        avg_loss, avg_emd, avg_cls, avg_mil = train_one_epoch(
            model, train_loader, optimizer, scaler,
            epoch, args, rank, world_size, use_wandb
        )

        scheduler.step()

        if rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch+1}/{args.num_epochs}: "
                  f"loss={avg_loss:.4f}, emd={avg_emd:.4f}, cls={avg_cls:.4f}, mil={avg_mil:.4f}, lr={lr:.2e}")

            # W&B logging
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "train/loss_emd": avg_emd,
                    "train/loss_cls": avg_cls,
                    "train/loss_mil": avg_mil,
                    "train/lr": lr,
                    "train/best_loss": best_loss if avg_loss >= best_loss else avg_loss,
                })

            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0 or avg_loss < best_loss:
                raw_model = model.module if hasattr(model, 'module') else model
                ckpt = {
                    "state_dict": raw_model.state_dict(),
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "world_size": world_size,
                    "batch_size": args.batch_size,
                    "effective_batch_size": args.batch_size * world_size,
                }

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(ckpt, output_dir / "best_model.pth")
                    print(f"  Saved best model (loss={avg_loss:.4f})")

                if (epoch + 1) % args.save_interval == 0:
                    torch.save(ckpt, output_dir / f"epoch_{epoch+1}.pth")

    # Save final model
    if rank == 0:
        raw_model = model.module if hasattr(model, 'module') else model

        # Full model (excluding aux_cls_head for downstream use)
        full_sd = {
            k: v for k, v in raw_model.state_dict().items()
            if not k.startswith("aux_cls_head.")
        }
        final_path = output_dir / "final_model.pth"
        torch.save({"state_dict": full_sd, "epoch": args.num_epochs}, final_path)
        print(f"\nFinal model saved to: {final_path}")

        # Backbone only for downstream tasks
        backbone_sd = {
            k.replace("backbone.", ""): v
            for k, v in raw_model.state_dict().items()
            if k.startswith("backbone.")
        }
        backbone_path = output_dir / "final_backbone.pth"
        torch.save({"state_dict": backbone_sd}, backbone_path)
        print(f"Backbone saved to: {backbone_path}")

        # Finish W&B
        if use_wandb:
            wandb.finish()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
