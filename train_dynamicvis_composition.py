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

# Register EMAUpdateHook with MMEngine's HOOKS registry
import utils.ema  # noqa: F401


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
    p.add_argument("--l-var", type=float, default=None,
                   help="Weight for variance regularization (lambda_var)")
    p.add_argument("--l-cov", type=float, default=None,
                   help="Weight for covariance regularization (lambda_cov)")
    p.add_argument("--l-cosine", type=float, default=None,
                   help="Weight for cosine direction alignment (lambda_cosine)")
    p.add_argument("--l-cls", type=float, default=None,
                   help="Weight for auxiliary classification loss (lambda_cls)")
    p.add_argument("--l-slot-contrast", type=float, default=None,
                   help="Weight for per-slot InfoNCE contrastive loss (lambda_slot_contrast)")
    p.add_argument("--l-slot-var", type=float, default=None,
                   help="Weight for per-slot variance hinge loss (lambda_slot_var)")
    p.add_argument("--l-slot-diversity", type=float, default=None,
                   help="Weight for slot diversity/orthogonality loss (lambda_slot_diversity)")
    p.add_argument("--num-registers", type=int, default=None,
                   help="Number of register tokens in slot decoder (default 4)")
    p.add_argument("--max-iters", type=int, default=None,
                   help="Switch to iter-based training and stop after this many iters")
    p.add_argument(
        "--dist-backend", type=str, choices=["nccl", "gloo"],
        default=None,
        help="Override distributed backend",
    )
    p.add_argument(
        "--use-dynamicvis-keys", action="store_true",
        help="Use DynamicVis stage-3 tokens as slot decoder keys/values "
             "instead of cached DINOv3 patch embeddings (ablation)",
    )
    p.add_argument(
        "--pretrained-backbone", type=str, default=None,
        help="Path to pretrained backbone weights (final_backbone.pth). "
             "Loads backbone weights before training. Use strict=False since "
             "featmap vs avg_featmap mode have identical weights.",
    )
    p.add_argument(
        "--ema-tau", type=float, default=None,
        help="EMA decay rate for target slot decoder (default 0.996). "
             "Set to 0 to disable EMA target network.",
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

    # ── Max iters (iter-based training) ──
    if args.max_iters is not None:
        cfg.train_cfg = dict(
            type='IterBasedTrainLoop',
            max_iters=args.max_iters,
        )
        # Use a simple linear warmup for 10% of iters, then constant
        warmup_iters = max(1, args.max_iters // 10)
        cfg.param_scheduler = [
            dict(type="LinearLR", start_factor=0.001, end_factor=1.0,
                 by_epoch=False, begin=0, end=warmup_iters),
            dict(type="ConstantLR", factor=1.0,
                 by_epoch=False, begin=warmup_iters, end=args.max_iters),
        ]
        # Disable epoch-based checkpoint saving
        cfg.default_hooks.checkpoint = dict(
            type='CheckpointHook',
            by_epoch=False,
            interval=args.max_iters + 1,  # effectively disable
            save_last=True,
        )

    # ── Epochs ──
    elif args.epochs is not None:
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
    if args.l_var is not None:
        cfg.model.head.lambda_var = args.l_var
    if args.l_cov is not None:
        cfg.model.head.lambda_cov = args.l_cov
    if args.l_cosine is not None:
        cfg.model.head.lambda_cosine = args.l_cosine
    if args.l_cls is not None:
        cfg.model.head.lambda_cls = args.l_cls
    if args.l_slot_contrast is not None:
        cfg.model.head.lambda_slot_contrast = args.l_slot_contrast
    if args.l_slot_var is not None:
        cfg.model.head.lambda_slot_var = args.l_slot_var
    if args.l_slot_diversity is not None:
        cfg.model.head.lambda_slot_diversity = args.l_slot_diversity

    # ── Num registers ──
    if args.num_registers is not None:
        cfg.model.num_registers = args.num_registers

    # ── DynamicVis keys ablation ──
    if args.use_dynamicvis_keys:
        cfg.model.use_dynamicvis_keys = True
        cfg.model.patch_dim = 768  # DynamicVis stage-3 dim

    # ── EMA tau ──
    if args.ema_tau is not None:
        cfg.model.ema_tau = args.ema_tau

    # ── Add EMAUpdateHook if EMA is enabled ──
    # The hook calls model.update_ema() after each training iteration
    ema_tau = cfg.model.get('ema_tau', 0.996)
    if ema_tau > 0:
        if not hasattr(cfg, 'custom_hooks') or cfg.custom_hooks is None:
            cfg.custom_hooks = []
        # Check if EMAUpdateHook is already in custom_hooks
        has_ema_hook = any(
            h.get('type') == 'EMAUpdateHook'
            for h in cfg.custom_hooks
            if isinstance(h, dict)
        )
        if not has_ema_hook:
            cfg.custom_hooks.append(dict(type='EMAUpdateHook', debug_step=10))

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
    if cfg.train_cfg.get('type') == 'IterBasedTrainLoop':
        print(f"Max iters:   {cfg.train_cfg.max_iters}")
    else:
        print(f"Max epochs:  {cfg.train_cfg.max_epochs}")
    print(f"LR:          {cfg.optim_wrapper.optimizer.lr}")
    print(f"Cluster dir: {cfg.train_dataloader.dataset.cluster_data_dir}")
    print(f"Loss type:   {cfg.model.head.get('loss_type', 'cosine')}")
    print(f"λ_comp:      {cfg.model.head.lambda_comp}")
    print(f"λ_cosine:    {cfg.model.head.get('lambda_cosine', 0.0)}")
    print(f"λ_var:       {cfg.model.head.get('lambda_var', 0.0)}")
    print(f"λ_cov:       {cfg.model.head.get('lambda_cov', 0.0)}")
    print(f"λ_contrast:  {cfg.model.head.lambda_contrast}")
    print(f"λ_smooth:    {cfg.model.head.lambda_smooth}")
    print(f"λ_cls:       {cfg.model.head.get('lambda_cls', 0.0)}")
    print(f"λ_slot_con:  {cfg.model.head.get('lambda_slot_contrast', 0.0)}")
    print(f"λ_slot_var:  {cfg.model.head.get('lambda_slot_var', 0.0)}")
    print(f"λ_slot_div:  {cfg.model.head.get('lambda_slot_diversity', 0.0)}")
    print(f"num_classes: {cfg.model.get('num_classes', 0)}")
    print(f"num_queries: {cfg.model.get('num_queries', 16)}")
    print(f"conditioned: {cfg.model.get('conditioned', False)}")
    print(f"dv_keys:     {cfg.model.get('use_dynamicvis_keys', False)}")
    print(f"ema_tau:     {cfg.model.get('ema_tau', 0.996)}")
    print(f"pretrained:  {args.pretrained_backbone or 'None'}")
    print("=" * 60)

    runner = Runner.from_cfg(cfg)

    # ── Load pretrained backbone weights (optional) ──
    if args.pretrained_backbone is not None:
        import torch
        pretrained_path = Path(args.pretrained_backbone)
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained backbone not found: {pretrained_path}"
            )
        print(f"\nLoading pretrained backbone from: {pretrained_path}")

        # Load checkpoint (weights_only=False for MMEngine checkpoints)
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)

        # Handle common checkpoint formats
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
                ckpt_format = "wrapped in 'state_dict'"
            elif "model" in ckpt:
                state_dict = ckpt["model"]
                ckpt_format = "wrapped in 'model'"
            elif "backbone" in ckpt:
                state_dict = ckpt["backbone"]
                ckpt_format = "wrapped in 'backbone'"
            else:
                state_dict = ckpt
                ckpt_format = "raw state_dict"
        else:
            raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")

        # Transform keys: strip 'backbone.' prefix if present
        # This handles checkpoints from full model saves where backbone keys have 'backbone.' prefix
        transformed_dict = {}
        n_stripped = 0
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                new_key = k[len("backbone."):]  # Strip 'backbone.' prefix
                n_stripped += 1
            else:
                new_key = k
            transformed_dict[new_key] = v

        if n_stripped > 0:
            key_transform = f"stripped 'backbone.' from {n_stripped} keys"
        else:
            key_transform = "no transformation needed"

        print(f"  Checkpoint format: {ckpt_format}")
        print(f"  Key transformation: {key_transform}")

        # Get model backbone
        model = runner.model
        if hasattr(model, "module"):
            model = model.module

        # Load into backbone
        backbone_params = set(model.backbone.state_dict().keys())
        missing, unexpected = model.backbone.load_state_dict(transformed_dict, strict=False)

        # Calculate successfully loaded
        loaded_keys = backbone_params - set(missing)
        n_loaded = len(loaded_keys)
        n_total = len(backbone_params)

        print(f"  Successfully loaded: {n_loaded} / {n_total} parameters")
        if missing:
            print(f"  Missing keys (new layers): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys (not in backbone): {len(unexpected)}")

        # Validate: >50% of backbone params should load
        if n_loaded < n_total * 0.5:
            raise ValueError(
                f"Weight loading failed — only {n_loaded}/{n_total} parameters loaded. "
                f"This suggests a key mismatch was not resolved. "
                f"Run: python scripts/debug_weight_loading.py {pretrained_path}"
            )

        print()

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

        # Save full model state dict (excluding training-only scaffolding:
        # aux_cls_head, prototypes, _proto_init, slot_decoder_ema — these are
        # not needed for downstream inference. Note: slot_decoder_ema is stored
        # as a plain Python object (not nn.Module), so it won't appear in
        # state_dict anyway, but we explicitly list it for documentation.)
        _training_only_prefixes = (
            "aux_cls_head.", "prototypes", "_proto_init", "slot_decoder_ema"
        )
        full_sd = {
            k: v for k, v in model.state_dict().items()
            if not k.startswith(_training_only_prefixes)
        }
        final_path = work_dir / "final_model.pth"
        torch.save(
            {
                "state_dict": full_sd,
                "meta": {
                    "epoch": cfg.train_cfg.get('max_epochs', None),
                    "max_iters": cfg.train_cfg.get('max_iters', None),
                    "config": args.config,
                },
            },
            final_path,
        )
        print(f"\nFinal model weights saved to: {final_path}")
        print(f"  (excluded training-only keys: {_training_only_prefixes})")

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
