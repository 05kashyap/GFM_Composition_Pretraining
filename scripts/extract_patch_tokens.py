#!/usr/bin/env python3
"""Extract raw patch tokens from FMoW cells using DINOv3 ViT-L/16.

This is Phase 1 of the BoVW composition training pipeline. It extracts raw
patch tokens (not pooled embeddings) from each 512×512 cell.

For each cell:
- Feed the 512×512 cell through DINOv3 ViT-L/16
- Extract spatial patch tokens (excluding CLS and register tokens)
- ViT-L/16 with 16×16 patches produces 32×32=1024 tokens per 512×512 cell
- Average pool every 2×2 tokens to get 16×16=256 tokens representing 32×32 regions
- Output shape per cell: (256, 1024)

Usage:
    python scripts/extract_patch_tokens.py \
        --data-root data/fmow \
        --manifest data/fmow_manifest_train.json \
        --output-dir outputs/patch_tokens_bovw

Supports multi-GPU sharding:
    python scripts/extract_patch_tokens.py --shard-index 0 --num-shards 4
    python scripts/extract_patch_tokens.py --shard-index 1 --num-shards 4
    ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# FMoW contains some extremely large satellite images.
Image.MAX_IMAGE_PIXELS = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import (
    DinoV3SatViTL16,
    _extract_checkpoint_state_dict,
    _strip_known_prefixes,
    seed_all,
)


# ---------------------------------------------------------------------------
# Cache key generation (same scheme as pipeline_utils.py)
# ---------------------------------------------------------------------------

def _weights_fingerprint(weights_path: str) -> Tuple[Optional[float], Optional[int]]:
    """Get modification time and size of weights file."""
    if not weights_path:
        return None, None
    try:
        st = os.stat(weights_path)
        return float(st.st_mtime), int(st.st_size)
    except Exception:
        return None, None


def compute_cache_key(image_path: str, weights_path: str, cell_size: int = 512) -> str:
    """Compute SHA1 cache key for a cell's patch tokens.

    Args:
        image_path: Path to the cell image
        weights_path: Path to DINOv3 weights
        cell_size: Expected cell size (512 for fMoW)

    Returns:
        SHA1 hash string
    """
    mtime, fsize = _weights_fingerprint(weights_path)
    payload = {
        "image": str(image_path),
        "weights_path": str(weights_path),
        "weights_mtime": mtime,
        "weights_size": fsize,
        "cell_size": cell_size,
        "output_type": "patch_tokens_bovw",
        "output_shape": [256, 1024],
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


# ---------------------------------------------------------------------------
# Patch Token Extractor
# ---------------------------------------------------------------------------

class PatchTokenExtractor:
    """Extract patch tokens from images using DINOv3 ViT-L/16."""

    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(
        self,
        device: str,
        weights_path: str,
        cell_size: int = 512,
    ):
        self.device = torch.device(device)
        self.weights_path = str(weights_path)
        self.cell_size = cell_size

        # Output patch grid: 16×16 = 256 patches of 32×32 pixels each
        self.output_patches_per_dim = 16  # 512 / 32 = 16
        self.output_total_patches = 256   # 16 × 16
        self.token_dim = 1024

        # ViT internal: 32×32 = 1024 tokens from 16×16 patches
        self.vit_patches_per_dim = 32     # 512 / 16 = 32

        # Build model
        self.model = DinoV3SatViTL16(
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            num_register_tokens=4,
        )
        self.model.eval()
        self.model.to(self.device)

        # Load weights
        wpath = Path(self.weights_path)
        if not wpath.exists():
            raise FileNotFoundError(f"Weights not found: {wpath}")

        ckpt = torch.load(str(wpath), map_location="cpu", weights_only=False)
        sd_raw = _extract_checkpoint_state_dict(ckpt)
        sd = {_strip_known_prefixes(k): v for k, v in sd_raw.items()
              if isinstance(v, torch.Tensor)}

        incompatible = self.model.load_state_dict(sd, strict=False)
        n_missing = len(incompatible.missing_keys)
        real_unexpected = [k for k in incompatible.unexpected_keys if "bias_mask" not in k]
        n_unexpected = len(real_unexpected)

        print(f"Loaded DINOv3 weights: {wpath.name} | "
              f"missing={n_missing} unexpected={n_unexpected}")
        if n_missing > 0:
            print(f"  Missing keys: {incompatible.missing_keys[:5]}")
        if n_unexpected > 0:
            print(f"  Unexpected keys: {real_unexpected[:5]}")

    @torch.inference_mode()
    def extract_tokens(self, images: torch.Tensor) -> np.ndarray:
        """Extract patch tokens from a batch of images.

        Args:
            images: Tensor of shape (B, 3, 512, 512), values in [0, 1]

        Returns:
            Numpy array of shape (B, 256, 1024) - patch tokens
        """
        B = images.shape[0]

        # Normalize
        mean = self._IMAGENET_MEAN.to(self.device)
        std = self._IMAGENET_STD.to(self.device)

        if self.device.type == "cuda" and not images.is_pinned():
            images = images.pin_memory()

        images = images.to(self.device, non_blocking=True)
        images = (images - mean) / std

        # Forward pass
        use_amp = self.device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            cls_out, patch_out = self.model(images)

        # patch_out shape: (B, 1024, 1024) = (B, 32×32 spatial tokens, 1024 dim)
        # Reshape to spatial grid
        patch_out = patch_out.float()
        patch_out = patch_out.view(B, self.vit_patches_per_dim, self.vit_patches_per_dim, self.token_dim)

        # Average pool 2×2 regions to get 16×16 tokens
        # (B, 32, 32, 1024) -> (B, 1024, 32, 32) -> pool -> (B, 1024, 16, 16) -> (B, 16, 16, 1024)
        patch_out = patch_out.permute(0, 3, 1, 2)  # (B, 1024, 32, 32)
        patch_out = F.avg_pool2d(patch_out, kernel_size=2, stride=2)  # (B, 1024, 16, 16)
        patch_out = patch_out.permute(0, 2, 3, 1)  # (B, 16, 16, 1024)

        # Flatten spatial dimensions
        patch_out = patch_out.reshape(B, self.output_total_patches, self.token_dim)  # (B, 256, 1024)

        return patch_out.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Image Loading
# ---------------------------------------------------------------------------

def load_cell_image(path: Path, target_size: int = 512) -> Optional[torch.Tensor]:
    """Load a cell image and resize to target size.

    Args:
        path: Path to image file
        target_size: Target size (512 for fMoW cells)

    Returns:
        Tensor of shape (3, H, W) with values in [0, 1], or None if failed
    """
    try:
        img = Image.open(path)
        img = img.convert("RGB")

        # Resize to target size if needed
        w, h = img.size
        if w != target_size or h != target_size:
            img = img.resize((target_size, target_size), Image.BICUBIC)

        # Convert to tensor
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)
        return img_tensor
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract raw patch tokens from FMoW cells using DINOv3 ViT-L/16.",
    )

    # Data
    parser.add_argument("--data-root", type=str, default="data/fmow",
                        help="Path to fMoW dataset root")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to manifest.json with cell list")
    parser.add_argument("--output-dir", type=str, default="outputs/patch_tokens_bovw",
                        help="Output directory for .npz files")

    # Model
    parser.add_argument("--weights-path", type=str,
                        default="weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
                        help="Path to DINOv3 weights")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # Processing
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of cells to process per batch")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of dataloader workers (for future use)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells where output .npz already exists")
    parser.add_argument("--seed", type=int, default=42)

    # Multi-GPU sharding
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Worker index (0-based) for parallel processing")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel workers")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max number of cells to process (for subset runs)")

    args = parser.parse_args()
    seed_all(args.seed)

    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        return 1

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    print(f"Loaded manifest with {len(manifest)} cells")

    # Apply max_samples limit before sharding (to ensure consistent subset across shards)
    if args.max_samples is not None and args.max_samples < len(manifest):
        manifest = manifest[:args.max_samples]
        print(f"Limited to --max-samples={args.max_samples} cells")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shard selection
    shard_manifest = manifest[args.shard_index::args.num_shards]
    print(f"[Shard {args.shard_index}/{args.num_shards}] Processing {len(shard_manifest)}/{len(manifest)} cells")

    # Build extractor
    print(f"Loading DINOv3 ViT-L/16 from {args.weights_path}...")
    extractor = PatchTokenExtractor(
        device=args.device,
        weights_path=args.weights_path,
    )
    print(f"Device: {extractor.device}")

    # Filter out already processed cells if resuming
    data_root = Path(args.data_root)
    cells_to_process = []
    skipped = 0
    not_found = 0

    # Known manifest path prefixes that need to be stripped
    # e.g., "Hosted-Datasets/fmow/fmow-rgb/train/..." -> "train/..."
    manifest_prefixes = [
        "Hosted-Datasets/fmow/fmow-rgb/",
        "Hosted-Datasets/fmow/fmow-rgb-prepped/",
    ]

    for entry in shard_manifest:
        raw_path = entry["img_path"]

        # Strip known manifest prefixes
        rel_path = raw_path
        for prefix in manifest_prefixes:
            if raw_path.startswith(prefix):
                rel_path = raw_path[len(prefix):]
                break

        img_path = data_root / rel_path
        cache_key = compute_cache_key(str(img_path), args.weights_path)
        output_path = output_dir / f"{cache_key}.npz"

        if args.resume and output_path.exists():
            skipped += 1
            continue

        # Check if image exists
        if not img_path.exists():
            not_found += 1
            if not_found <= 5:
                print(f"Warning: Image not found: {img_path}")
            elif not_found == 6:
                print("  (suppressing further 'not found' warnings...)")
            continue

        cells_to_process.append({
            "img_path": img_path,
            "cache_key": cache_key,
            "output_path": output_path,
        })

    if not_found > 0:
        print(f"Warning: {not_found} images not found (skipped)")

    if args.resume:
        print(f"Resuming: skipped {skipped} already processed cells")

    print(f"Cells to process: {len(cells_to_process)}")

    if not cells_to_process:
        print("All cells already processed. Exiting.")
        return 0

    # Process in batches
    t0 = time.time()
    cells_processed = 0
    total_tokens = 0
    batch_imgs = []
    batch_meta = []

    log_interval = 100

    for idx, cell_info in enumerate(tqdm(cells_to_process, desc="Processing cells")):
        # Load image
        img_tensor = load_cell_image(cell_info["img_path"])
        if img_tensor is None:
            continue

        batch_imgs.append(img_tensor)
        batch_meta.append(cell_info)

        # Process batch when full
        if len(batch_imgs) >= args.batch_size:
            batch_tensor = torch.stack(batch_imgs, dim=0)  # (B, 3, 512, 512)
            tokens = extractor.extract_tokens(batch_tensor)  # (B, 256, 1024)

            # Save each cell's tokens
            for i, meta in enumerate(batch_meta):
                np.savez(
                    meta["output_path"],
                    patch_tokens=tokens[i],  # (256, 1024)
                    img_path=str(meta["img_path"]),
                )
                cells_processed += 1
                total_tokens += 256

            batch_imgs = []
            batch_meta = []

            # Log progress
            if cells_processed % log_interval == 0 and cells_processed > 0:
                elapsed = time.time() - t0
                rate = cells_processed / elapsed
                print(f"  Processed {cells_processed} cells, {total_tokens} tokens "
                      f"({rate:.1f} cells/s)")

    # Process remaining cells in last batch
    if batch_imgs:
        batch_tensor = torch.stack(batch_imgs, dim=0)
        tokens = extractor.extract_tokens(batch_tensor)

        for i, meta in enumerate(batch_meta):
            np.savez(
                meta["output_path"],
                patch_tokens=tokens[i],
                img_path=str(meta["img_path"]),
            )
            cells_processed += 1
            total_tokens += 256

    elapsed = time.time() - t0

    # Compute output directory size
    total_size_bytes = sum(f.stat().st_size for f in output_dir.glob("*.npz"))
    total_size_gb = total_size_bytes / (1024 ** 3)

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"  Total cells processed: {cells_processed}")
    print(f"  Total patch tokens extracted: {total_tokens}")
    print(f"  Output directory: {output_dir}")
    print(f"  Output size: {total_size_gb:.2f} GB")
    print(f"  Time elapsed: {elapsed:.1f}s ({cells_processed/elapsed:.1f} cells/s)")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
