#!/usr/bin/env python3
"""Embed FMoW satellite image patches using DINOv3 ViT-L/16 and cache to disk.

This script handles:
  1. Loading and standardizing FMoW images (resize to grid multiples)
  2. Extracting small patches within non-overlapping grid cells
  3. Embedding patches with the DINOv3 ViT-L/16 model (fp32 attention)
  4. Caching per-cell embeddings as .npz files

Supports multi-GPU parallel sharding:
   python embed_patches.py --shard-index 0 --num-shards 8
   python embed_patches.py --shard-index 1 --num-shards 8
   ...

All shards write to the same --cache-dir; the clustering script reads from it.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

from pipeline_utils import (
    DinoV3SatViTL16Embedder,
    GridCell,
    MultiImageBatchedEmbedder,
    PrefetchLoader,
    extract_patches_in_grid,
    find_jpgs,
    load_and_standardize_image,
    sample_stratified,
    seed_all,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Embed FMoW patches with DINOv3 and cache to disk.",
    )

    # Data
    parser.add_argument("--data-root", type=str, default="data/fmow")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster-num-images", type=int, default=1000,
                        help="Number of images to embed (stratified sample from split). "
                             "Must match the value used in cluster_viz.py for consistency.")

    # Patching
    parser.add_argument("--large-size", type=int, default=512,
                        help="Grid cell size in pixels (images resized to multiples of this).")
    parser.add_argument("--small-size", type=int, default=128)
    parser.add_argument("--small-stride", type=int, default=64,
                        help="Uniform small-patch stride (overridden by --small-stride-x/y). "
                             "Default 64 gives 50%% overlap for 128px patches.")
    parser.add_argument("--small-stride-x", type=int, default=None,
                        help="Small-patch horizontal stride (defaults to --small-stride)")
    parser.add_argument("--small-stride-y", type=int, default=None,
                        help="Small-patch vertical stride (defaults to --small-stride)")

    # Model
    parser.add_argument("--weights-path", type=str,
                        default="weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    parser.add_argument("--pool-mode", type=str, default="cls_avg",
                        choices=["cls", "avg", "cls_avg"],
                        help="Embedding pooling mode: 'cls' (1024-d), 'avg' (1024-d), "
                             "'cls_avg' (concat, 2048-d, default).")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # Batching
    parser.add_argument("--embed-batch", type=int, default=2048,
                        help="Patches to accumulate before flushing to GPU.")
    parser.add_argument("--gpu-batch-size", type=int, default=512,
                        help="Patches per ViT forward pass (VRAM control).")

    # Cache
    parser.add_argument("--cache-dir", type=str, default="outputs/preprocess_cache_dinov3")

    # Multi-GPU sharding
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Worker index (0-based) for parallel embedding.")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel embedding workers.")

    args = parser.parse_args()

    # Resolve per-axis strides
    args.small_stride_x = args.small_stride_x if args.small_stride_x is not None else args.small_stride
    args.small_stride_y = args.small_stride_y if args.small_stride_y is not None else args.small_stride

    seed_all(args.seed)

    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    all_paths = find_jpgs(data_root, args.split)
    print(f"Total images found: {len(all_paths)}")
    print(f"Sliding window: patch={args.small_size}x{args.small_size}  "
          f"stride_x={args.small_stride_x}  stride_y={args.small_stride_y}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic stratified sampling (same RNG seed across all shards)
    rng_cluster = random.Random(args.seed + 1)
    cluster_n = min(int(args.cluster_num_images), len(all_paths))
    cluster_paths = sample_stratified(all_paths, split_dir, cluster_n, rng_cluster)

    # Shard selection
    shard_paths = cluster_paths[args.shard_index :: args.num_shards]
    print(f"[Shard {args.shard_index}/{args.num_shards}] "
          f"Embedding {len(shard_paths)}/{len(cluster_paths)} images", flush=True)
    print(f"[Shard {args.shard_index}] Batch size: {args.embed_batch} patches accumulated, "
          f"GPU batch size: {args.gpu_batch_size} patches per forward pass", flush=True)

    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available():
        print(f"Device: cuda (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device.type}")

    # Build embedder
    embedder = DinoV3SatViTL16Embedder(
        device=args.device,
        weights_path=args.weights_path,
        pool_mode=args.pool_mode,
    )

    # Build batched embedder
    batched_embedder = MultiImageBatchedEmbedder(
        embedder=embedder,
        cache_dir=cache_dir,
        grid_size=args.large_size,
        weights_path=args.weights_path,
        batch_size=args.embed_batch,
        gpu_batch_size=args.gpu_batch_size,
    )

    # Threaded prefetch: load+preprocess images while GPU embeds
    def _load_and_prepare(p: Path):
        img = load_and_standardize_image(p, grid_size=args.large_size)
        w, h = img.size
        cells = extract_patches_in_grid(
            w, h,
            large_size=args.large_size,
            small_size=args.small_size,
            small_stride_x=args.small_stride_x,
            small_stride_y=args.small_stride_y,
        )
        img_t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div_(255.0)
        return p, cells, img_t

    t0 = time.time()
    prefetcher = PrefetchLoader(
        shard_paths,
        load_fn=_load_and_prepare,
        num_workers=4,
        prefetch_count=16,
    )
    for p, cells, img_t in tqdm(
        prefetcher,
        desc=f"[Shard {args.shard_index}] Embedding",
        unit="img",
        dynamic_ncols=True,
        total=len(shard_paths),
    ):
        batched_embedder.add_image(image_path=p, img_tensor=img_t, cells=cells)

    batched_embedder.flush()
    elapsed = time.time() - t0

    print(f"[Shard {args.shard_index}] Embedding complete in {elapsed:.1f}s. "
          f"Cells: {batched_embedder.cells_processed}, "
          f"Patches: {batched_embedder.patches_embedded}, "
          f"Forward passes: {batched_embedder.forward_passes}", flush=True)
    print(f"[Shard {args.shard_index}] Cache dir: {cache_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
