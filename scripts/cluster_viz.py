#!/usr/bin/env python3
"""Load cached FMoW patch embeddings, cluster with PCA + KMeans, and visualize.

Reads the per-cell .npz cache files produced by ``embed_patches.py``,
assembles them into per-image embedding matrices, then runs:
  1. PCA dimensionality reduction
  2. Clustering (MiniBatchKMeans by default; also supports HDBSCAN, BisectingKMeans, GMM)
  3. Visualization of cluster assignments overlaid on original images

Outputs are written to ``outputs/preprocess_viz_dinov3/<run_id>/``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

from pipeline_utils import (
    GridCell,
    Patch,
    colorize_small_patch_grid,
    compute_cache_path,
    extract_patches_in_grid,
    find_jpgs,
    fit_bisecting_kmeans,
    fit_gmm,
    fit_hdbscan,
    fit_pca,
    fit_sklearn_kmeans,
    flatten_cells,
    l2_normalize_np,
    load_and_standardize_image,
    load_cached_embeddings,
    overlay_grid,
    sample_stratified,
    save_viz_embeddings,
    seed_all,
    silhouette_optional,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_all_embeddings_from_cache(
    *,
    image_paths: List[Path],
    embedder_name: str,
    grid_size: int,
    large_size: int,
    small_size: int,
    small_stride_x: int,
    small_stride_y: int,
    weights_path: str,
    cache_dir: Path,
) -> Tuple[Dict[Path, np.ndarray], Dict[Path, List[GridCell]], Dict[Path, List[np.ndarray]], Dict[Path, List[Patch]]]:
    """Load cached embeddings for all images.

    Returns:
        embeddings_by_path: path → (N_patches, D) concatenated embeddings
        cells_by_path: path → list of GridCell
        cell_embeddings_by_path: path → list of per-cell embedding arrays
        patches_by_path: path → flat list of Patch
    """
    embeddings_by_path: Dict[Path, np.ndarray] = {}
    cells_by_path: Dict[Path, List[GridCell]] = {}
    cell_embeddings_by_path: Dict[Path, List[np.ndarray]] = {}
    patches_by_path: Dict[Path, List[Patch]] = {}

    missing_count = 0
    for p in tqdm(image_paths, desc="Loading cached embeddings", unit="img"):
        img = load_and_standardize_image(p, grid_size=grid_size)
        w, h = img.size
        cells = extract_patches_in_grid(
            w, h,
            large_size=large_size,
            small_size=small_size,
            small_stride_x=small_stride_x,
            small_stride_y=small_stride_y,
        )
        del img  # free memory

        cell_embs: List[np.ndarray] = []
        all_found = True
        for cell in cells:
            cp = compute_cache_path(
                cache_dir=cache_dir,
                image_path=p,
                embedder_name=embedder_name,
                grid_size=grid_size,
                cell=cell,
                weights_path=weights_path,
            )
            cached = load_cached_embeddings(cp)
            if cached is not None and cached.shape[0] == len(cell.patches):
                cell_embs.append(cached)
            else:
                all_found = False
                missing_count += 1
                break

        if all_found and cell_embs:
            embeddings_by_path[p] = np.concatenate(cell_embs, axis=0)
            cells_by_path[p] = cells
            cell_embeddings_by_path[p] = cell_embs
            patches_by_path[p] = flatten_cells(cells)

    if missing_count > 0:
        print(f"WARNING: {missing_count} images had missing cache entries — "
              "run embed_patches.py first to populate the cache.")

    return embeddings_by_path, cells_by_path, cell_embeddings_by_path, patches_by_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cluster and visualize cached FMoW patch embeddings.",
    )

    # Data
    parser.add_argument("--data-root", type=str, default="data/fmow")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster-num-images", type=int, default=1000,
                        help="Number of images (must match embed_patches.py).")
    parser.add_argument("--viz-num-images", type=int, default=25)

    # Patching (must match embed_patches.py)
    parser.add_argument("--large-size", type=int, default=512)
    parser.add_argument("--large-stride", type=int, default=512)
    parser.add_argument("--large-stride-x", type=int, default=None)
    parser.add_argument("--large-stride-y", type=int, default=None)
    parser.add_argument("--small-size", type=int, default=128)
    parser.add_argument("--small-stride", type=int, default=64)
    parser.add_argument("--small-stride-x", type=int, default=None)
    parser.add_argument("--small-stride-y", type=int, default=None)

    # Model info (for cache key computation — we don't load the model)
    parser.add_argument("--weights-path", type=str,
                        default="weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    parser.add_argument("--pool-mode", type=str, default="cls_avg",
                        choices=["cls", "avg", "cls_avg"])

    # Cache
    parser.add_argument("--cache-dir", type=str, default="outputs/preprocess_cache_dinov3")

    # Clustering
    parser.add_argument("--clusterer", type=str, default="sklearn_kmeans",
                        choices=["sklearn_kmeans", "bisecting_kmeans", "gmm", "hdbscan"])
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--fit-small-patches-per-image", type=int, default=64)

    # HDBSCAN
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=15)
    parser.add_argument("--hdbscan-min-samples", type=int, default=0)
    parser.add_argument("--hdbscan-jobs", type=int, default=8)
    parser.add_argument("--assign-noise-to-nearest-centroid", action="store_true")

    # Output
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--size-stats-n", type=int, default=1000)

    # Composition-aware training data export
    parser.add_argument("--save-cluster-data", action="store_true",
                        help="Export cluster centroids + per-cell compositional targets "
                             "for composition-aware training.")
    parser.add_argument("--cluster-data-dir", type=str, default="outputs/cluster_data",
                        help="Directory to write cluster data (centroids, manifest, targets).")
    parser.add_argument("--use-pca-targets", action="store_true",
                        help="Use PCA embeddings (256-d) as targets instead of centroid-averaged "
                             "embeddings (2048-d). Recommended for better target diversity.")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Only generate manifest.json without running clustering or computing "
                             "targets. Use this when training with loss_comp=0 (QSACL mode).")

    args = parser.parse_args()

    # Resolve per-axis strides
    args.small_stride_x = args.small_stride_x if args.small_stride_x is not None else args.small_stride
    args.small_stride_y = args.small_stride_y if args.small_stride_y is not None else args.small_stride
    args.large_stride_x = args.large_stride_x if args.large_stride_x is not None else args.large_stride
    args.large_stride_y = args.large_stride_y if args.large_stride_y is not None else args.large_stride

    seed_all(args.seed)

    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    all_paths = find_jpgs(data_root, args.split)
    print(f"Total images found: {len(all_paths)}")

    rng = random.Random(args.seed)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / "preprocess_viz_dinov3" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    if not cache_dir.exists():
        print(f"ERROR: Cache dir does not exist: {cache_dir}")
        print("Run embed_patches.py first to populate the cache.")
        return 1

    # ---------------------------------------------------------------
    # Size distribution (optional diagnostic)
    # ---------------------------------------------------------------
    size_n = min(args.size_stats_n, len(all_paths))
    size_paths = sample_stratified(all_paths, split_dir, size_n, rng)
    widths: List[int] = []
    heights: List[int] = []
    for p in size_paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
            widths.append(int(w))
            heights.append(int(h))
        except Exception:
            continue

    if widths and heights:
        w_arr = np.asarray(widths)
        h_arr = np.asarray(heights)
        print(
            "Image size stats (sample): "
            f"n={len(w_arr)} | mean_w={w_arr.mean():.1f}, mean_h={h_arr.mean():.1f} | "
            f"median_w={np.median(w_arr):.0f}, median_h={np.median(h_arr):.0f}"
        )
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(w_arr, bins=40)
        plt.title("Width distribution")
        plt.xlabel("width")
        plt.ylabel("count")
        plt.subplot(1, 2, 2)
        plt.hist(h_arr, bins=40)
        plt.title("Height distribution")
        plt.xlabel("height")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "image_size_distribution.png", dpi=160)
        plt.close()

    # ---------------------------------------------------------------
    # Deterministic stratified sample (must match embed_patches.py)
    # ---------------------------------------------------------------
    rng_cluster = random.Random(args.seed + 1)
    cluster_n = min(int(args.cluster_num_images), len(all_paths))
    cluster_paths = sample_stratified(all_paths, split_dir, cluster_n, rng_cluster)
    viz_n = min(int(args.viz_num_images), len(cluster_paths))
    viz_set: Set[Path] = set(cluster_paths[:viz_n])

    # Build embedder name to match cache keys (without loading the model)
    embedder_name = f"dinov3_sat:{Path(args.weights_path).name}:{args.pool_mode}"
    embedding_dim = 2048 if args.pool_mode == "cls_avg" else 1024

    # ---------------------------------------------------------------
    # Load all embeddings from cache
    # ---------------------------------------------------------------
    print(f"\n=== Loading cached embeddings for {len(cluster_paths)} images ===")
    embeddings_by_path, cells_by_path, cell_embeddings_by_path, patches_by_path = \
        _load_all_embeddings_from_cache(
            image_paths=cluster_paths,
            embedder_name=embedder_name,
            grid_size=args.large_size,
            large_size=args.large_size,
            small_size=args.small_size,
            small_stride_x=args.small_stride_x,
            small_stride_y=args.small_stride_y,
            weights_path=args.weights_path,
            cache_dir=cache_dir,
        )

    loaded_count = len(embeddings_by_path)
    total_patches = sum(e.shape[0] for e in embeddings_by_path.values())
    total_cells = sum(len(c) for c in cells_by_path.values())
    print(f"=== Loaded {loaded_count}/{len(cluster_paths)} images, "
          f"{total_cells} grid cells, {total_patches} total patches ===\n")

    if loaded_count == 0:
        print("ERROR: No embeddings found in cache. Run embed_patches.py first.")
        return 1

    # ---------------------------------------------------------------
    # Manifest-only mode: skip clustering, just export cell metadata
    # ---------------------------------------------------------------
    if getattr(args, 'manifest_only', False):
        print("\n=== Manifest-only mode: skipping clustering ===")
        _export_manifest_only(
            args=args,
            cluster_paths=cluster_paths,
            cells_by_path=cells_by_path,
            cell_embeddings_by_path=cell_embeddings_by_path,
            embedder_name=embedder_name,
        )
        print(f"\nManifest-only export complete.")
        print(f"Cache dir: {cache_dir}")
        return 0

    # ---------------------------------------------------------------
    # Clustering
    # ---------------------------------------------------------------
    clusterer_name = str(args.clusterer)
    shown = 0
    eval_report: Dict[str, object] = {}

    # Subsample embeddings for fit
    fit_per_image = max(1, int(args.fit_small_patches_per_image))
    fit_embs: List[np.ndarray] = []
    t0 = time.time()

    for p in tqdm(cluster_paths, desc="Subsampling embeddings (fit set)", unit="img"):
        embs_full = embeddings_by_path.get(p)
        if embs_full is None:
            continue
        n_patches = embs_full.shape[0]
        if fit_per_image < n_patches:
            pick = rng.sample(range(n_patches), k=fit_per_image)
            embs = embs_full[pick]
        else:
            embs = embs_full
        fit_embs.append(embs)

    if not fit_embs:
        print("ERROR: No embeddings available for fitting.")
        return 1

    X = np.concatenate(fit_embs, axis=0)
    X = l2_normalize_np(X)
    print(f"Fit embedding matrix: {X.shape} (N,D)", flush=True)

    pca, X_pca = fit_pca(X, pca_dim=int(args.pca_dim), seed=args.seed)
    X_pca = l2_normalize_np(X_pca.astype(np.float32))

    # Save full embedding matrix for CPU-only clustering sweeps
    sweep_path = out_dir / "embedding_data.npz"
    image_paths_str = np.array([str(p) for p in cluster_paths], dtype=object)
    patches_per_image = X.shape[0] // len(cluster_paths) if len(cluster_paths) > 0 else 0
    np.savez_compressed(
        sweep_path,
        X_raw=X.astype(np.float16, copy=False),
        X_pca=X_pca.astype(np.float16, copy=False),
        image_paths=image_paths_str,
        patches_per_image=patches_per_image,
        pca_dim=int(args.pca_dim),
        grid_size=int(args.large_size),
        small_size=int(args.small_size),
        small_stride_x=int(args.small_stride_x),
        small_stride_y=int(args.small_stride_y),
    )
    print(f"Saved embedding data for CPU sweep: {sweep_path} ({X_pca.shape})")

    # Save PCA model
    with open(out_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)
    print(f"Saved PCA model: {out_dir / 'pca_model.pkl'}")

    # ---------------------------------------------------------------
    # Fit clusterer
    # ---------------------------------------------------------------
    if clusterer_name == "hdbscan":
        min_samples = int(args.hdbscan_min_samples)
        hdb, fit_labels = fit_hdbscan(
            X_pca,
            min_cluster_size=int(args.hdbscan_min_cluster_size),
            min_samples=min_samples,
            n_jobs=int(args.hdbscan_jobs),
        )

        n_noise = int((fit_labels < 0).sum())
        n_total = int(fit_labels.shape[0])
        n_clusters = int(len(set(fit_labels.tolist())) - (1 if -1 in fit_labels else 0))
        noise_frac = float(n_noise / max(n_total, 1))
        sil = silhouette_optional(X_pca, fit_labels, seed=args.seed)

        print(f"HDBSCAN: clusters={n_clusters} | noise={n_noise}/{n_total} ({noise_frac*100:.2f}%)")
        if sil is not None:
            print(f"Silhouette (non-noise): {sil:.4f}")

        eval_report = {
            "method": "hdbscan+pca",
            "fit_points": int(n_total),
            "pca_dim": int(args.pca_dim),
            "hdbscan_min_cluster_size": int(args.hdbscan_min_cluster_size),
            "hdbscan_min_samples": int(min_samples) if min_samples > 0 else None,
            "n_clusters": int(n_clusters),
            "noise_points": int(n_noise),
            "noise_fraction": float(noise_frac),
            "silhouette_non_noise": float(sil) if sil is not None else None,
        }

        valid = fit_labels[fit_labels >= 0]
        if valid.size == 0:
            counts = np.asarray([n_total], dtype=np.int64)
            order = np.asarray([0], dtype=np.int64)
            centroids = np.zeros((1, X_pca.shape[1]), dtype=np.float32)
        else:
            max_label = int(valid.max())
            counts = np.bincount(valid.astype(np.int64), minlength=max_label + 1)
            order = np.argsort(-counts)
            centroids = np.zeros((counts.shape[0], X_pca.shape[1]), dtype=np.float32)
            for cid in range(counts.shape[0]):
                mask = fit_labels == cid
                if not np.any(mask):
                    continue
                centroids[cid] = X_pca[mask].mean(axis=0)
            centroids = l2_normalize_np(centroids)

        try:
            import hdbscan as hdbscan_lib
        except Exception:
            hdbscan_lib = None

        for p in cluster_paths:
            if p not in viz_set or shown >= viz_n:
                continue

            img = load_and_standardize_image(p, grid_size=args.large_size)
            small_patches_full = patches_by_path.get(p, [])

            embs_all = embeddings_by_path.get(p)
            if embs_all is None:
                continue
            embs_all = l2_normalize_np(embs_all)
            embs_pca = pca.transform(embs_all).astype(np.float32)
            embs_pca = l2_normalize_np(embs_pca)

            if hdbscan_lib is not None and valid.size != 0:
                pred_labels, _strength = hdbscan_lib.approximate_predict(hdb, embs_pca)
            else:
                pred_labels = np.full((embs_pca.shape[0],), -1, dtype=np.int64)

            if args.assign_noise_to_nearest_centroid and valid.size != 0:
                noise_mask = pred_labels < 0
                if np.any(noise_mask):
                    sims = embs_pca[noise_mask] @ centroids.T
                    nn = np.argmax(sims, axis=1)
                    pred_labels[noise_mask] = nn.astype(np.int64)

            if args.save_embeddings:
                save_viz_embeddings(out_dir, shown, p, small_patches_full, embs_all, pred_labels,
                                     cells=cells_by_path.get(p), emb_pca=embs_pca)

            _render_viz(
                img, p, small_patches_full, pred_labels,
                args=args, embedder_name=embedder_name,
                title_extra=f"clusters={counts.shape[0]}",
                out_dir=out_dir, shown_idx=shown,
            )
            shown += 1

    else:
        # Fixed-k clustering
        k = int(args.k)
        if clusterer_name == "sklearn_kmeans":
            model, fit_labels = fit_sklearn_kmeans(X_pca, k=k, seed=args.seed)
        elif clusterer_name == "bisecting_kmeans":
            model, fit_labels = fit_bisecting_kmeans(X_pca, k=k, seed=args.seed)
        elif clusterer_name == "gmm":
            model, fit_labels = fit_gmm(X_pca, k=k, seed=args.seed)
        else:
            raise RuntimeError(f"Unknown clusterer: {clusterer_name}")

        counts = np.bincount(fit_labels.astype(np.int64), minlength=k)
        order = np.argsort(-counts)
        sil = silhouette_optional(X_pca, fit_labels, seed=args.seed)
        if sil is not None:
            print(f"Silhouette: {sil:.4f}")

        eval_report = {
            "method": clusterer_name,
            "fit_points": int(X_pca.shape[0]),
            "pca_dim": int(args.pca_dim),
            "k": int(k),
            "silhouette": float(sil) if sil is not None else None,
        }

        for p in cluster_paths:
            if p not in viz_set or shown >= viz_n:
                continue

            img = load_and_standardize_image(p, grid_size=args.large_size)
            small_patches_full = patches_by_path.get(p, [])

            embs_all = embeddings_by_path.get(p)
            if embs_all is None:
                continue
            embs_all = l2_normalize_np(embs_all)
            embs_pca = pca.transform(embs_all).astype(np.float32)
            embs_pca = l2_normalize_np(embs_pca)
            pred_labels = model.predict(embs_pca)

            if args.save_embeddings:
                save_viz_embeddings(out_dir, shown, p, small_patches_full, embs_all, pred_labels,
                                     cells=cells_by_path.get(p), emb_pca=embs_pca)

            _render_viz(
                img, p, small_patches_full, pred_labels,
                args=args, embedder_name=embedder_name,
                title_extra=f"{clusterer_name} k={k}",
                out_dir=out_dir, shown_idx=shown,
            )
            shown += 1

    # ---------------------------------------------------------------
    # Cluster size plot
    # ---------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(int(counts.shape[0])), counts[order])
    plt.title(f"Cluster sizes ({clusterer_name}, k={int(args.k)})")
    plt.xlabel("clusters sorted by size")
    plt.ylabel("# small patches")
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_sizes.png", dpi=160)
    plt.close()

    print("\nCluster histogram (sorted by size):")
    top = min(50, int(counts.shape[0]))
    for rank in range(top):
        cid = int(order[rank])
        csz = int(counts[cid])
        print(f"  cluster {cid:03d}: {csz}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    summary = {
        "run_id": run_id,
        "data_root": str(data_root),
        "split": args.split,
        "cluster_num_images": int(cluster_n),
        "viz_num_images": int(viz_n),
        "grid_size": int(args.large_size),
        "small_size": int(args.small_size),
        "small_stride_x": int(args.small_stride_x),
        "small_stride_y": int(args.small_stride_y),
        "weights_path": str(args.weights_path),
        "embedder": embedder_name,
        "embedding_dim": int(embedding_dim),
        "clusterer": clusterer_name,
        "k": int(args.k),
        "pca_dim": int(args.pca_dim),
        "cluster_sizes": counts.tolist(),
        "evaluation": eval_report,
        "cache": {
            "cache_dir": str(cache_dir),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------------------------------------------------------
    # Export cluster data for composition-aware training
    # ---------------------------------------------------------------
    if args.save_cluster_data and clusterer_name != "hdbscan":
        _export_cluster_data(
            args=args,
            cluster_paths=cluster_paths,
            embeddings_by_path=embeddings_by_path,
            cells_by_path=cells_by_path,
            cell_embeddings_by_path=cell_embeddings_by_path,
            pca=pca,
            model=model,
            k=int(args.k),
            embedding_dim=embedding_dim,
            embedder_name=embedder_name,
            use_pca_targets=args.use_pca_targets,
        )
    elif args.save_cluster_data and clusterer_name == "hdbscan":
        print("WARNING: --save-cluster-data not supported with HDBSCAN (variable cluster count). "
              "Use a fixed-k clusterer like sklearn_kmeans.")

    print(f"\nSaved outputs to: {out_dir}")
    print(f"Cache dir: {cache_dir}")
    return 0


# ---------------------------------------------------------------------------
# Manifest-only export (no clustering, for QSACL mode)
# ---------------------------------------------------------------------------

def _export_manifest_only(
    *,
    args,
    cluster_paths: List[Path],
    cells_by_path: Dict[Path, List[GridCell]],
    cell_embeddings_by_path: Dict[Path, List[np.ndarray]],
    embedder_name: str,
) -> None:
    """Export only manifest.json without clustering or targets.

    Use this when training with loss_comp=0 (QSACL mode) where targets.npy
    is not needed. The manifest provides cell metadata for the dataset.

    Produces:
        cluster_data_dir/
            manifest.json  — per-cell metadata (image path, row, col, n_patches, etc.)
    """
    cluster_data_dir = Path(args.cluster_data_dir)
    cluster_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Exporting manifest to {cluster_data_dir} ===")

    manifest_entries: List[dict] = []

    for p in tqdm(cluster_paths, desc="Building manifest", unit="img"):
        cells = cells_by_path.get(p)
        cell_embs_list = cell_embeddings_by_path.get(p)
        if cells is None or cell_embs_list is None:
            continue

        for ci, (cell, cell_emb) in enumerate(zip(cells, cell_embs_list)):
            if cell_emb is None:
                continue

            manifest_entries.append({
                "image_path": str(p),
                "cell_row": int(cell.row),
                "cell_col": int(cell.col),
                "cell_x0": int(cell.x0),
                "cell_y0": int(cell.y0),
                "cell_x1": int(cell.x1),
                "cell_y1": int(cell.y1),
                "n_patches": len(cell.patches),
                "target_index": -1,  # No target in manifest-only mode
                # Metadata for cache path reconstruction
                "embedder_name": embedder_name,
                "weights_path": str(args.weights_path),
                "small_size": int(args.small_size),
                "small_stride_x": int(args.small_stride_x),
                "small_stride_y": int(args.small_stride_y),
            })

    # Save manifest
    manifest = {
        "embedder_name": embedder_name,
        "weights_path": str(args.weights_path),
        "grid_size": int(args.large_size),
        "small_size": int(args.small_size),
        "small_stride_x": int(args.small_stride_x),
        "small_stride_y": int(args.small_stride_y),
        "embedding_dim": 0,  # No targets
        "n_images": len(set(e["image_path"] for e in manifest_entries)),
        "n_cells": len(manifest_entries),
        "manifest_only": True,  # Flag to indicate no targets.npy
        "cells": manifest_entries,
    }
    manifest_path = cluster_data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Saved manifest: {len(manifest_entries)} cells to {manifest_path}")


# ---------------------------------------------------------------------------
# Cluster data export for composition-aware training
# ---------------------------------------------------------------------------

def _export_cluster_data(
    *,
    args,
    cluster_paths: List[Path],
    embeddings_by_path: Dict[Path, np.ndarray],
    cells_by_path: Dict[Path, List[GridCell]],
    cell_embeddings_by_path: Dict[Path, List[np.ndarray]],
    pca,
    model,
    k: int,
    embedding_dim: int,
    embedder_name: str,
    use_pca_targets: bool = False,
) -> None:
    """Export cluster centroids and per-cell compositional targets.

    Produces:
        cluster_data_dir/
            centroids.npy       — (k, embedding_dim) cluster centers in original DINOv3 space
            targets.npy         — (N_total_cells, target_dim) compositional target per cell
            manifest.json       — per-cell metadata (image path, row, col, target index, etc.)
            pca_model.pkl       — fitted PCA model
            kmeans_model.pkl    — fitted KMeans model

    When ``use_pca_targets=True``, targets are 256-d PCA embeddings (mean of PCA patches
    per cell), which have MUCH more diversity than the 2048-d centroid-averaged targets.
    This is recommended to avoid target collapse (pairwise cos ~0.0 vs ~0.75).
    """
    cluster_data_dir = Path(args.cluster_data_dir)
    cluster_data_dir.mkdir(parents=True, exist_ok=True)

    pca_dim = int(args.pca_dim)
    target_dim = pca_dim if use_pca_targets else embedding_dim

    print(f"\n=== Exporting cluster data to {cluster_data_dir} ===")
    if use_pca_targets:
        print(f"    Using PCA targets: {pca_dim}-d (recommended for diversity)")
    else:
        print(f"    Using centroid-averaged targets: {embedding_dim}-d")

    # Step 1: Compute cluster centroids in ORIGINAL embedding space.
    # For each cluster, collect all original-space embeddings assigned to it,
    # then average them.
    print("Computing cluster centroids in original embedding space...")
    centroid_accum = np.zeros((k, embedding_dim), dtype=np.float64)
    centroid_count = np.zeros(k, dtype=np.int64)

    for p in tqdm(cluster_paths, desc="Assigning clusters (centroids)", unit="img"):
        embs_raw = embeddings_by_path.get(p)
        if embs_raw is None:
            continue
        embs_norm = l2_normalize_np(embs_raw)
        embs_pca = pca.transform(embs_norm).astype(np.float32)
        embs_pca = l2_normalize_np(embs_pca)
        labels = model.predict(embs_pca)
        for cid in range(k):
            mask = labels == cid
            if np.any(mask):
                centroid_accum[cid] += embs_raw[mask].astype(np.float64).sum(axis=0)
                centroid_count[cid] += int(mask.sum())

    # Compute mean centroid, L2-normalize
    for cid in range(k):
        if centroid_count[cid] > 0:
            centroid_accum[cid] /= centroid_count[cid]
    centroids_raw = l2_normalize_np(centroid_accum.astype(np.float32))
    np.save(cluster_data_dir / "centroids.npy", centroids_raw)
    print(f"  Saved centroids: {centroids_raw.shape} to {cluster_data_dir / 'centroids.npy'}")

    # Step 2: For each image, for each grid cell, compute the compositional target.
    # If use_pca_targets: target = mean of PCA patches per cell (256-d)
    # Otherwise: target = mean of centroids_raw[cluster_id] for each small patch (2048-d)
    print("Computing per-cell compositional targets...")
    manifest_entries: List[dict] = []
    all_targets: List[np.ndarray] = []
    target_idx = 0

    for p in tqdm(cluster_paths, desc="Computing targets", unit="img"):
        cells = cells_by_path.get(p)
        cell_embs_list = cell_embeddings_by_path.get(p)
        if cells is None or cell_embs_list is None:
            continue

        for ci, (cell, cell_emb) in enumerate(zip(cells, cell_embs_list)):
            if cell_emb is None:
                continue

            # Transform to PCA space (needed for both modes)
            embs_norm = l2_normalize_np(cell_emb)
            embs_pca = pca.transform(embs_norm).astype(np.float32)
            embs_pca = l2_normalize_np(embs_pca)

            if use_pca_targets:
                # PCA target: mean of PCA patches (256-d), much more diverse!
                target = embs_pca.mean(axis=0)
            else:
                # Original approach: mean of cluster centroids (2048-d)
                labels = model.predict(embs_pca)
                target = centroids_raw[labels].mean(axis=0)

            target = target / (np.linalg.norm(target) + 1e-12)  # L2 normalize
            all_targets.append(target)

            manifest_entries.append({
                "image_path": str(p),
                "cell_row": int(cell.row),
                "cell_col": int(cell.col),
                "cell_x0": int(cell.x0),
                "cell_y0": int(cell.y0),
                "cell_x1": int(cell.x1),
                "cell_y1": int(cell.y1),
                "n_patches": len(cell.patches),
                "target_index": target_idx,
            })
            target_idx += 1

    targets_array = np.stack(all_targets, axis=0).astype(np.float32)
    np.save(cluster_data_dir / "targets.npy", targets_array)
    print(f"  Saved targets: {targets_array.shape} to {cluster_data_dir / 'targets.npy'}")

    # Verify target diversity
    if len(all_targets) > 100:
        sample_idx = np.random.choice(len(all_targets), min(1000, len(all_targets)), replace=False)
        sample = targets_array[sample_idx]
        cos_sim = sample @ sample.T
        np.fill_diagonal(cos_sim, 0)
        mean_cos = cos_sim.sum() / (len(sample) * (len(sample) - 1))
        print(f"  Target diversity check: pairwise cosine = {mean_cos:.4f} "
              f"({'good' if mean_cos < 0.5 else 'WARNING: may be too collapsed'})")

    # Step 3: Save manifest
    manifest = {
        "grid_size": int(args.large_size),
        "small_size": int(args.small_size),
        "small_stride_x": int(args.small_stride_x),
        "small_stride_y": int(args.small_stride_y),
        "embedding_dim": int(target_dim),  # Use target_dim, not embedding_dim
        "original_embedding_dim": int(embedding_dim),
        "use_pca_targets": use_pca_targets,
        "k": k,
        "pca_dim": int(args.pca_dim),
        "clusterer": str(args.clusterer),
        "embedder_name": embedder_name,
        "weights_path": str(args.weights_path),
        "pool_mode": str(args.pool_mode),
        "n_images": len(set(e["image_path"] for e in manifest_entries)),
        "n_cells": len(manifest_entries),
        "cells": manifest_entries,
    }
    manifest_path = cluster_data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Saved manifest: {len(manifest_entries)} cells from "
          f"{manifest['n_images']} images to {manifest_path}")

    # Step 4: Save PCA and KMeans models for inference on new images
    with open(cluster_data_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open(cluster_data_dir / "kmeans_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved PCA + KMeans models to {cluster_data_dir}")

    print(f"=== Cluster data export complete: {cluster_data_dir} ===")


# ---------------------------------------------------------------------------
# Visualization rendering
# ---------------------------------------------------------------------------

def _render_viz(
    img: Image.Image,
    image_path: Path,
    small_patches: List[Patch],
    pred_labels: np.ndarray,
    *,
    args,
    embedder_name: str,
    title_extra: str,
    out_dir: Path,
    shown_idx: int,
) -> None:
    """Render a 3-panel visualization: original | large grid | cluster overlay."""
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    axs[0].imshow(img)
    axs[0].set_title("Standardized image (resized to grid)")
    axs[0].axis("off")

    axs[1].imshow(img)
    pw, ph = img.size
    overlay_grid(
        axs[1], w=pw, h=ph, patch=args.large_size, stride=args.large_stride,
        color="lime", lw=1.0,
        stride_x=args.large_stride_x, stride_y=args.large_stride_y,
    )
    axs[1].set_title(f"Large grid {args.large_size}  sx={args.large_stride_x} sy={args.large_stride_y}")
    axs[1].axis("off")

    axs[2].imshow(img)
    overlay = colorize_small_patch_grid(small_patches, pred_labels.astype(np.int64), grid_w=pw, grid_h=ph)
    axs[2].imshow(overlay)
    axs[2].set_title(f"Sliding window {args.small_size}  sx={args.small_stride_x} sy={args.small_stride_y}")
    axs[2].axis("off")

    fig.suptitle(f"{image_path} | embedder={embedder_name} | {title_extra}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"image_{shown_idx:02d}_clusters.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
