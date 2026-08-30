#!/usr/bin/env python3
"""Generate soft-assignment histograms from patch tokens using visual vocabulary.

This is Phase 3 of the BoVW composition training pipeline. It converts each cell's
patch tokens into a soft histogram distribution over the visual vocabulary.

The script:
1. Loads centroids from Phase 2 and L2-normalizes them
2. For each cell, loads patch tokens from Phase 1 and L2-normalizes them
3. Computes soft assignment using RBF kernel with temperature beta
4. Saves all histograms as a memory-mapped numpy array for fast training access

Usage:
    python scripts/generate_histograms.py \
        --patch-token-dir outputs/patch_tokens_bovw \
        --vocab-dir outputs/bovw_vocabulary \
        --manifest data/fmow_manifest_train.json \
        --output-dir outputs/bovw_histograms \
        --beta 10.0

Requires: numpy, tqdm (optional)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize array along specified axis."""
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def soft_assign(tokens: np.ndarray, centroids: np.ndarray, beta: float = 10.0) -> np.ndarray:
    """Compute soft histogram assignment using RBF kernel.

    Args:
        tokens: (N, D) L2-normalised patch tokens
        centroids: (K, D) L2-normalised cluster centroids
        beta: RBF kernel temperature (higher = sharper assignments)

    Returns:
        histogram: (K,) L1-normalised probability distribution
    """
    # Cosine similarity = dot product after L2 norm
    sim = tokens @ centroids.T  # (N, K) cosine similarities

    # Cosine distance approximation (1 - sim gives [0, 2] range)
    dist_sq = 1.0 - sim  # (N, K) cosine distances

    # RBF kernel: exp(-beta * dist)
    weights = np.exp(-beta * dist_sq)  # (N, K)

    # Normalise rows (each token sums to 1 across clusters)
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)  # avoid division by zero
    weights = weights / row_sums

    # Sum over patches to get histogram
    histogram = weights.sum(axis=0)  # (K,)

    # L1 normalise → probability distribution
    hist_sum = histogram.sum()
    if hist_sum > 0:
        histogram = histogram / hist_sum

    return histogram.astype(np.float32)


def _weights_fingerprint(weights_path: str):
    """Get modification time and size of weights file."""
    if not weights_path:
        return None, None
    try:
        st = os.stat(weights_path)
        return float(st.st_mtime), int(st.st_size)
    except Exception:
        return None, None


def compute_cache_key(img_path: str, weights_path: str, cell_size: int = 512) -> str:
    """Compute SHA1 cache key matching Phase 1 extraction.

    Must match the exact format used in extract_patch_tokens.py
    """
    mtime, fsize = _weights_fingerprint(weights_path)
    payload = {
        "image": str(img_path),
        "weights_path": str(weights_path),
        "weights_mtime": mtime,
        "weights_size": fsize,
        "cell_size": cell_size,
        "output_type": "patch_tokens_bovw",
        "output_shape": [256, 1024],
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def process_single_cell(args_tuple: Tuple) -> Optional[Tuple[int, np.ndarray]]:
    """Process a single cell and return its histogram.

    Args:
        args_tuple: (cell_idx, npz_path, centroids, beta)

    Returns:
        (cell_idx, histogram) or None if failed
    """
    cell_idx, npz_path, centroids, beta = args_tuple

    try:
        # Load patch tokens
        data = np.load(npz_path)
        tokens = data['patch_tokens']  # (256, 1024)

        # L2-normalise tokens
        tokens = l2_normalize(tokens, axis=1)

        # Compute soft histogram
        histogram = soft_assign(tokens, centroids, beta)

        return (cell_idx, histogram)

    except Exception as e:
        return None


def compute_entropy(histogram: np.ndarray) -> float:
    """Compute entropy of a probability distribution."""
    # Avoid log(0)
    h = histogram[histogram > 0]
    return -np.sum(h * np.log(h + 1e-10))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate soft-assignment histograms from patch tokens.",
    )

    # Input/Output paths
    parser.add_argument("--patch-token-dir", type=str, required=True,
                        help="Path to Phase 1 output directory with .npz files")
    parser.add_argument("--vocab-dir", type=str, required=True,
                        help="Path to Phase 2 output directory with centroids.npy")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to manifest.json for cell ordering")
    parser.add_argument("--output-dir", type=str, default="outputs/bovw_histograms",
                        help="Output directory for histograms")

    # Histogram parameters
    parser.add_argument("--beta", type=float, default=10.0,
                        help="RBF kernel temperature (default: 10.0)")

    # Processing options
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial output")

    # For Phase 1 cache key compatibility
    parser.add_argument("--weights-path", type=str,
                        default="weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
                        help="DINOv3 weights path (for cache key matching)")
    parser.add_argument("--data-root", type=str, default="data/fmow",
                        help="Data root path (for cache key matching)")

    # Subset selection
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit to first N cells from manifest (for storage constraints)")

    args = parser.parse_args()

    # Setup paths
    patch_token_dir = Path(args.patch_token_dir)
    vocab_dir = Path(args.vocab_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BoVW Phase 3 - Histogram Target Generation")
    print("=" * 60)
    print(f"Patch token dir: {patch_token_dir}")
    print(f"Vocab dir: {vocab_dir}")
    print(f"Manifest: {args.manifest}")
    print(f"Output dir: {output_dir}")
    print(f"Beta: {args.beta}")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    # Load centroids
    centroids_path = vocab_dir / "centroids.npy"
    if not centroids_path.exists():
        print(f"Error: Centroids not found at {centroids_path}")
        print("Run Phase 2 (build_vocabulary.py) first.")
        return 1

    print("\nLoading centroids...")
    centroids = np.load(centroids_path)
    K, D = centroids.shape
    print(f"  Centroids shape: {centroids.shape}")

    # L2-normalise centroids
    centroids = l2_normalize(centroids, axis=1)
    norms = np.linalg.norm(centroids[:10], axis=1)
    print(f"  After L2-norm (first 10): mean={norms.mean():.6f}, std={norms.std():.6f}")

    # Load manifest
    print("\nLoading manifest...")
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    print(f"  Total cells in manifest: {len(manifest)}")

    # Apply max_samples limit
    if args.max_samples is not None and args.max_samples < len(manifest):
        manifest = manifest[:args.max_samples]
        print(f"  Limited to --max-samples={args.max_samples} cells")

    # Build mapping from manifest entries to npz files
    # Known manifest path prefixes that need to be stripped
    manifest_prefixes = [
        "Hosted-Datasets/fmow/fmow-rgb/",
        "Hosted-Datasets/fmow/fmow-rgb-prepped/",
    ]
    data_root = Path(args.data_root)

    print("\nMapping cells to patch token files...")
    cell_mappings = []
    missing_count = 0

    for idx, entry in enumerate(tqdm(manifest, desc="Checking files")):
        raw_path = entry.get("img_path", entry.get("image_path", ""))

        # Strip known manifest prefixes
        rel_path = raw_path
        for prefix in manifest_prefixes:
            if raw_path.startswith(prefix):
                rel_path = raw_path[len(prefix):]
                break

        img_path = data_root / rel_path

        # Compute cache key matching Phase 1
        cache_key = compute_cache_key(str(img_path), args.weights_path)
        npz_path = patch_token_dir / f"{cache_key}.npz"

        if npz_path.exists():
            cell_mappings.append({
                "idx": idx,
                "npz_path": npz_path,
                "img_path": raw_path,
            })
        else:
            missing_count += 1
            if missing_count <= 5:
                print(f"  Warning: Missing npz for {img_path}")
            elif missing_count == 6:
                print("  (suppressing further warnings...)")

    print(f"\nFound {len(cell_mappings)} cells with patch tokens")
    if missing_count > 0:
        print(f"Missing {missing_count} cells (run Phase 1 extraction first)")

    if len(cell_mappings) == 0:
        print("Error: No cells found. Check paths and run Phase 1 first.")
        return 1

    # Check for existing output (resume support)
    histograms_path = output_dir / "histograms.npy"
    cell_ids_path = output_dir / "cell_ids.npy"

    if args.resume and histograms_path.exists() and cell_ids_path.exists():
        print("\nResuming from existing output...")
        existing_histograms = np.load(histograms_path, mmap_mode='r')
        existing_cell_ids = np.load(cell_ids_path)
        existing_set = set(existing_cell_ids.tolist())

        # Filter to cells not yet processed
        cell_mappings = [c for c in cell_mappings if c["idx"] not in existing_set]
        print(f"  Already processed: {len(existing_set)}")
        print(f"  Remaining: {len(cell_mappings)}")

        if len(cell_mappings) == 0:
            print("All cells already processed.")
            # Still compute statistics
            histograms = np.load(histograms_path)
            cell_ids = existing_cell_ids
        else:
            # Will append to existing
            pass
    else:
        existing_histograms = None
        existing_cell_ids = None

    # Process cells
    if len(cell_mappings) > 0:
        print(f"\nProcessing {len(cell_mappings)} cells...")

        # Prepare arguments for parallel processing
        process_args = [
            (mapping["idx"], mapping["npz_path"], centroids, args.beta)
            for mapping in cell_mappings
        ]

        # Process in parallel
        results = []
        failed = 0

        if args.workers > 1:
            # Multi-process
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_single_cell, arg): arg[0]
                          for arg in process_args}

                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="Computing histograms"):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    else:
                        failed += 1
        else:
            # Single process (for debugging)
            for arg in tqdm(process_args, desc="Computing histograms"):
                result = process_single_cell(arg)
                if result is not None:
                    results.append(result)
                else:
                    failed += 1

        if failed > 0:
            print(f"Warning: {failed} cells failed to process")

        # Sort by cell index
        results.sort(key=lambda x: x[0])

        # Stack results
        new_cell_ids = np.array([r[0] for r in results], dtype=np.int64)
        new_histograms = np.stack([r[1] for r in results], axis=0)

        # Combine with existing if resuming
        if existing_histograms is not None and existing_cell_ids is not None:
            cell_ids = np.concatenate([existing_cell_ids, new_cell_ids])
            histograms = np.concatenate([np.array(existing_histograms), new_histograms], axis=0)
        else:
            cell_ids = new_cell_ids
            histograms = new_histograms

        # Save outputs
        print(f"\nSaving outputs...")
        print(f"  Histograms shape: {histograms.shape}")

        np.save(histograms_path, histograms.astype(np.float32))
        np.save(cell_ids_path, cell_ids)

        print(f"  Saved: {histograms_path}")
        print(f"  Saved: {cell_ids_path}")

    # Compute and print statistics
    print("\n" + "=" * 60)
    print("Histogram Statistics")
    print("=" * 60)

    histograms = np.load(histograms_path)
    cell_ids = np.load(cell_ids_path)

    print(f"Total histograms: {histograms.shape[0]}")
    print(f"Vocabulary size: {histograms.shape[1]}")

    # Value statistics
    print(f"\nValue statistics:")
    print(f"  Min: {histograms.min():.6f}")
    print(f"  Max: {histograms.max():.6f}")
    print(f"  Mean: {histograms.mean():.6f}")
    print(f"  Std: {histograms.std():.6f}")

    # Entropy statistics
    print(f"\nEntropy statistics:")
    entropies = np.array([compute_entropy(h) for h in histograms])
    max_entropy = np.log(K)
    print(f"  Mean entropy: {entropies.mean():.4f} (max possible: {max_entropy:.4f})")
    print(f"  Min entropy: {entropies.min():.4f}")
    print(f"  Max entropy: {entropies.max():.4f}")
    print(f"  Normalised entropy: {entropies.mean() / max_entropy:.4f}")

    # Pairwise L1 distance (sample 1000 pairs)
    print(f"\nPairwise L1 distance (1000 random pairs):")
    n_samples = min(1000, len(histograms) * (len(histograms) - 1) // 2)
    if len(histograms) >= 2:
        np.random.seed(42)
        indices = np.random.choice(len(histograms), size=(n_samples, 2), replace=True)
        # Ensure pairs are different
        indices = indices[indices[:, 0] != indices[:, 1]]
        if len(indices) > 0:
            l1_distances = np.abs(histograms[indices[:, 0]] - histograms[indices[:, 1]]).sum(axis=1)
            print(f"  Mean L1 distance: {l1_distances.mean():.4f}")
            print(f"  Std L1 distance: {l1_distances.std():.4f}")
            print(f"  Min L1 distance: {l1_distances.min():.4f}")
            print(f"  Max L1 distance: {l1_distances.max():.4f}")

            # Check if vocabulary is working correctly
            if l1_distances.mean() < 0.1:
                print("  WARNING: L1 distances are very small - histograms may be too similar!")
                print("           Consider increasing beta or checking vocabulary quality.")
            elif l1_distances.mean() > 0.3:
                print("  OK: L1 distances look healthy (vocabulary is discriminative)")
            else:
                print("  NOTE: L1 distances are moderate")
        else:
            print("  (not enough distinct pairs)")
    else:
        print("  (need at least 2 histograms)")

    # Sparsity statistics
    print(f"\nSparsity statistics:")
    zero_bins = (histograms < 1e-6).sum(axis=1)
    print(f"  Mean zero bins per histogram: {zero_bins.mean():.1f} / {K}")
    print(f"  Mean non-zero bins: {K - zero_bins.mean():.1f}")

    # Top bins analysis
    print(f"\nTop bins analysis:")
    top5_mass = np.sort(histograms, axis=1)[:, -5:].sum(axis=1)
    print(f"  Mean mass in top 5 bins: {top5_mass.mean():.4f}")
    top10_mass = np.sort(histograms, axis=1)[:, -10:].sum(axis=1)
    print(f"  Mean mass in top 10 bins: {top10_mass.mean():.4f}")

    print("\n" + "=" * 60)
    print("Phase 3 Complete!")
    print("=" * 60)
    print(f"  Histograms: {histograms_path}")
    print(f"  Cell IDs: {cell_ids_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
