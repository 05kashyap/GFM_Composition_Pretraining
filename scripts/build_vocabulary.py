#!/usr/bin/env python3
"""Build visual vocabulary from extracted patch tokens using FAISS K-means.

This is Phase 2 of the BoVW composition training pipeline. It clusters
patch tokens from Phase 1 to create a visual vocabulary.

The script:
1. Loads a random subsample of patch tokens from Phase 1 .npz files
2. L2-normalizes all tokens
3. Runs FAISS K-means clustering to find K centroids
4. Saves centroids and ground cost matrix for EMD loss

Usage:
    python scripts/build_vocabulary.py \
        --patch-token-dir outputs/patch_tokens_bovw \
        --output-dir outputs/bovw_vocabulary \
        --K 512 \
        --subsample 5000000

Requires: faiss-gpu (or faiss-cpu), numpy, matplotlib
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: FAISS not installed. Install with: pip install faiss-gpu (or faiss-cpu)")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Cluster histogram will not be saved.")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def seed_all(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    # FAISS uses its own seed in Kmeans constructor


def load_npz_files(patch_token_dir: Path) -> List[Path]:
    """Find all .npz files in the patch token directory."""
    npz_files = sorted(patch_token_dir.glob("*.npz"))
    return npz_files


def subsample_tokens(
    npz_files: List[Path],
    target_tokens: int,
    seed: int,
) -> np.ndarray:
    """Load and subsample tokens uniformly across cells.

    Args:
        npz_files: List of .npz file paths from Phase 1
        target_tokens: Total number of tokens to sample
        seed: Random seed for reproducibility

    Returns:
        tokens: (N, 1024) array of subsampled tokens
    """
    np.random.seed(seed)

    n_cells = len(npz_files)
    if n_cells == 0:
        raise ValueError("No .npz files found in patch token directory")

    # Calculate tokens per cell (with some buffer for rounding)
    tokens_per_cell = max(1, target_tokens // n_cells)

    print(f"Subsampling strategy:")
    print(f"  Total cells: {n_cells}")
    print(f"  Target tokens: {target_tokens:,}")
    print(f"  Tokens per cell: {tokens_per_cell}")

    # Shuffle cell order for randomness
    shuffled_indices = np.random.permutation(n_cells)

    all_tokens = []
    tokens_collected = 0

    for idx in tqdm(shuffled_indices, desc="Loading tokens"):
        if tokens_collected >= target_tokens:
            break

        npz_path = npz_files[idx]
        try:
            data = np.load(npz_path)
            cell_tokens = data['patch_tokens']  # (256, 1024)

            # How many tokens to sample from this cell
            n_available = cell_tokens.shape[0]
            n_to_sample = min(tokens_per_cell, n_available, target_tokens - tokens_collected)

            if n_to_sample < n_available:
                # Random sample without replacement
                sample_indices = np.random.choice(n_available, n_to_sample, replace=False)
                sampled = cell_tokens[sample_indices]
            else:
                sampled = cell_tokens

            all_tokens.append(sampled)
            tokens_collected += sampled.shape[0]

        except Exception as e:
            print(f"Warning: Failed to load {npz_path}: {e}")
            continue

    if not all_tokens:
        raise ValueError("No tokens could be loaded from any .npz file")

    tokens = np.concatenate(all_tokens, axis=0)

    # Final trim to exact target (if we overshot)
    if tokens.shape[0] > target_tokens:
        tokens = tokens[:target_tokens]

    print(f"  Collected {tokens.shape[0]:,} tokens from {len(all_tokens)} cells")

    return tokens.astype(np.float32)


def l2_normalize(tokens: np.ndarray) -> np.ndarray:
    """L2-normalize tokens along the feature dimension."""
    norms = np.linalg.norm(tokens, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-8)
    return tokens / norms


def run_kmeans(
    tokens: np.ndarray,
    K: int,
    seed: int,
    use_gpu: bool = True,
    niter: int = 100,
    nredo: int = 3,
) -> Tuple[np.ndarray, float]:
    """Run FAISS K-means clustering.

    Args:
        tokens: (N, D) L2-normalized tokens
        K: Number of clusters (vocabulary size)
        seed: Random seed
        use_gpu: Whether to use GPU acceleration
        niter: Number of iterations per run
        nredo: Number of runs with different initializations

    Returns:
        centroids: (K, D) cluster centroids
        quantization_error: Final quantization error
    """
    if not HAS_FAISS:
        raise ImportError("FAISS is required for K-means clustering")

    N, D = tokens.shape
    print(f"\nRunning FAISS K-means:")
    print(f"  Tokens: {N:,} x {D}")
    print(f"  K: {K}")
    print(f"  Iterations: {niter}")
    print(f"  Restarts: {nredo}")
    print(f"  GPU: {use_gpu}")

    # Check GPU availability
    if use_gpu:
        n_gpus = faiss.get_num_gpus()
        if n_gpus == 0:
            print("  Warning: No GPUs detected, falling back to CPU")
            use_gpu = False
        else:
            print(f"  Available GPUs: {n_gpus}")

    t0 = time.time()

    kmeans = faiss.Kmeans(
        D, K,
        niter=niter,
        nredo=nredo,
        verbose=True,
        gpu=use_gpu,
        seed=seed,
    )

    kmeans.train(tokens)

    elapsed = time.time() - t0
    print(f"\nK-means completed in {elapsed:.1f}s")

    # Get centroids
    centroids = kmeans.centroids  # (K, D)

    # Compute final quantization error
    # (average squared distance to nearest centroid)
    _, distances = kmeans.index.search(tokens, 1)
    quantization_error = float(np.mean(distances))

    print(f"Final quantization error: {quantization_error:.6f}")

    return centroids, quantization_error


def compute_ground_cost(centroids: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix between centroids.

    Args:
        centroids: (K, D) cluster centroids

    Returns:
        ground_cost: (K, K) pairwise cosine distances, clipped to [0, 1]
    """
    # L2-normalize centroids
    c_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # Cosine similarity
    sim = c_norm @ c_norm.T  # (K, K)

    # Cosine distance = 1 - similarity
    ground_cost = (1.0 - sim).astype(np.float32)

    # Clip to [0, 1] for numerical stability
    ground_cost = np.clip(ground_cost, 0.0, 1.0)

    # Ensure diagonal is exactly zero
    np.fill_diagonal(ground_cost, 0.0)

    return ground_cost


def compute_cluster_sizes(
    tokens: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Compute the number of tokens assigned to each cluster.

    Args:
        tokens: (N, D) tokens
        centroids: (K, D) centroids

    Returns:
        sizes: (K,) number of tokens per cluster
    """
    if not HAS_FAISS:
        # Fallback to numpy (slower)
        distances = np.linalg.norm(tokens[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)
    else:
        # Use FAISS for fast nearest neighbor search
        K, D = centroids.shape
        index = faiss.IndexFlatL2(D)
        index.add(centroids.astype(np.float32))
        _, assignments = index.search(tokens.astype(np.float32), 1)
        assignments = assignments.flatten()

    sizes = np.bincount(assignments, minlength=centroids.shape[0])
    return sizes


def save_cluster_histogram(
    sizes: np.ndarray,
    output_path: Path,
) -> None:
    """Save histogram of cluster sizes."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping histogram")
        return

    K = len(sizes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of cluster sizes
    ax1 = axes[0]
    ax1.hist(sizes, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Cluster Size (# tokens)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of Cluster Sizes (K={K})')
    ax1.axvline(np.mean(sizes), color='red', linestyle='--', label=f'Mean: {np.mean(sizes):.0f}')
    ax1.axvline(np.median(sizes), color='orange', linestyle='--', label=f'Median: {np.median(sizes):.0f}')
    ax1.legend()

    # Sorted cluster sizes (to see uniformity)
    ax2 = axes[1]
    sorted_sizes = np.sort(sizes)[::-1]
    ax2.bar(range(K), sorted_sizes, alpha=0.7)
    ax2.set_xlabel('Cluster (sorted by size)')
    ax2.set_ylabel('Cluster Size')
    ax2.set_title('Cluster Sizes (Sorted)')
    ax2.axhline(np.mean(sizes), color='red', linestyle='--', label=f'Mean: {np.mean(sizes):.0f}')

    # Mark if any cluster has >5% of tokens
    total_tokens = sizes.sum()
    threshold = 0.05 * total_tokens
    large_clusters = np.sum(sizes > threshold)
    if large_clusters > 0:
        ax2.axhline(threshold, color='purple', linestyle=':', label=f'5% threshold ({large_clusters} clusters exceed)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved cluster histogram to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build visual vocabulary from patch tokens using FAISS K-means.",
    )

    # Input/Output
    parser.add_argument("--patch-token-dir", type=str, required=True,
                        help="Path to Phase 1 output directory with .npz files")
    parser.add_argument("--output-dir", type=str, default="outputs/bovw_vocabulary",
                        help="Output directory for vocabulary files")

    # Clustering parameters
    parser.add_argument("--K", type=int, default=512,
                        help="Vocabulary size (number of clusters)")
    parser.add_argument("--subsample", type=int, default=5_000_000,
                        help="Number of tokens to subsample for clustering")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # K-means parameters
    parser.add_argument("--niter", type=int, default=100,
                        help="Number of K-means iterations")
    parser.add_argument("--nredo", type=int, default=3,
                        help="Number of K-means restarts")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration for K-means")

    args = parser.parse_args()

    if not HAS_FAISS:
        print("Error: FAISS is required. Install with: pip install faiss-gpu")
        return 1

    # Setup
    seed_all(args.seed)
    patch_token_dir = Path(args.patch_token_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BoVW Phase 2 - Visual Vocabulary Construction")
    print("=" * 60)
    print(f"Patch token dir: {patch_token_dir}")
    print(f"Output dir: {output_dir}")
    print(f"K (vocabulary size): {args.K}")
    print(f"Subsample size: {args.subsample:,}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Find .npz files
    print("\nFinding patch token files...")
    npz_files = load_npz_files(patch_token_dir)
    print(f"Found {len(npz_files):,} .npz files")

    if len(npz_files) == 0:
        print("Error: No .npz files found in patch token directory")
        return 1

    # Subsample tokens
    print("\nSubsampling tokens...")
    tokens = subsample_tokens(npz_files, args.subsample, args.seed)
    print(f"Token shape: {tokens.shape}")

    # L2-normalize
    print("\nL2-normalizing tokens...")
    tokens = l2_normalize(tokens)

    # Verify normalization
    norms = np.linalg.norm(tokens[:100], axis=1)
    print(f"  Norm check (first 100): mean={norms.mean():.6f}, std={norms.std():.6f}")

    # Run K-means
    centroids, quant_error = run_kmeans(
        tokens,
        K=args.K,
        seed=args.seed,
        use_gpu=not args.no_gpu,
        niter=args.niter,
        nredo=args.nredo,
    )

    print(f"\nCentroids shape: {centroids.shape}")

    # Save centroids
    centroids_path = output_dir / "centroids.npy"
    np.save(centroids_path, centroids.astype(np.float32))
    print(f"Saved centroids to {centroids_path}")

    # Compute and save ground cost matrix
    print("\nComputing ground cost matrix...")
    ground_cost = compute_ground_cost(centroids)

    # Verify ground cost properties
    print(f"  Ground cost shape: {ground_cost.shape}")
    print(f"  Ground cost range: [{ground_cost.min():.4f}, {ground_cost.max():.4f}]")
    print(f"  Ground cost diagonal (should be 0): {ground_cost.diagonal().sum():.6f}")
    print(f"  Ground cost symmetric: {np.allclose(ground_cost, ground_cost.T)}")

    ground_cost_path = output_dir / "ground_cost.npy"
    np.save(ground_cost_path, ground_cost)
    print(f"Saved ground cost matrix to {ground_cost_path}")

    # Compute cluster sizes and save histogram
    print("\nComputing cluster sizes...")
    cluster_sizes = compute_cluster_sizes(tokens, centroids)

    print(f"  Cluster size stats:")
    print(f"    Min: {cluster_sizes.min():,}")
    print(f"    Max: {cluster_sizes.max():,}")
    print(f"    Mean: {cluster_sizes.mean():.1f}")
    print(f"    Std: {cluster_sizes.std():.1f}")

    # Check for dominant clusters (>5% of tokens)
    total_tokens = cluster_sizes.sum()
    threshold = 0.05 * total_tokens
    large_clusters = np.sum(cluster_sizes > threshold)
    if large_clusters > 0:
        print(f"  WARNING: {large_clusters} cluster(s) contain >5% of tokens")
        large_indices = np.where(cluster_sizes > threshold)[0]
        for idx in large_indices[:5]:  # Show first 5
            pct = 100 * cluster_sizes[idx] / total_tokens
            print(f"    Cluster {idx}: {cluster_sizes[idx]:,} tokens ({pct:.2f}%)")
    else:
        print(f"  OK: No cluster contains >5% of tokens (vocabulary is well-distributed)")

    # Save histogram
    histogram_path = output_dir / "cluster_sizes.png"
    save_cluster_histogram(cluster_sizes, histogram_path)

    # Save cluster sizes as numpy array too
    sizes_path = output_dir / "cluster_sizes.npy"
    np.save(sizes_path, cluster_sizes)
    print(f"Saved cluster sizes to {sizes_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Vocabulary construction complete!")
    print("=" * 60)
    print(f"  Centroids: {centroids_path}")
    print(f"  Ground cost: {ground_cost_path}")
    print(f"  Cluster sizes: {sizes_path}")
    print(f"  Histogram: {histogram_path}")
    print(f"  Final quantization error: {quant_error:.6f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
