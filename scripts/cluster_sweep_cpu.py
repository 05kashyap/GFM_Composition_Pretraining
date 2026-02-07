#!/usr/bin/env python3
"""CPU-only clustering sweep over pre-computed embeddings.

Loads the embedding_data.npz + pca_model.pkl produced by
viz_fmow_patch_embed_cluster_dinov3.py and runs clustering for
multiple k values without any GPU or re-embedding.

For each k value:
  1. Fit MiniBatchKMeans on the full PCA embedding matrix.
  2. Compute silhouette score.
  3. Plot cluster size distribution.
  4. Visualize 25 images with per-patch cluster overlays.

Usage:
  python scripts/cluster_sweep_cpu.py \
      --embedding-dir outputs/preprocess_viz_dinov3/<run_id> \
      --k-values 30,60,100,150

Outputs saved to: outputs/cluster_sweep/<sweep_id>/k_<value>/
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

# Handle large satellite images
Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------
# Dataclass / helpers reused from the main script
# ---------------------------------------------------------------
@dataclass(frozen=True)
class Patch:
    x0: int
    y0: int
    x1: int
    y1: int


def _extract_grid_patches(w: int, h: int, patch: int, stride: int) -> List[Patch]:
    out: List[Patch] = []
    for y0 in range(0, h - patch + 1, stride):
        for x0 in range(0, w - patch + 1, stride):
            out.append(Patch(x0=x0, y0=y0, x1=x0 + patch, y1=y0 + patch))
    return out


def _resize_max_edge(img: Image.Image, max_edge: int) -> Image.Image:
    if max_edge <= 0:
        return img
    w, h = img.size
    scale = max_edge / max(w, h)
    if scale >= 1.0:
        return img
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)


def _pad_to_square(img: Image.Image, size: int, fill=(0, 0, 0)) -> Image.Image:
    w, h = img.size
    if w == size and h == size:
        return img
    canvas = Image.new("RGB", (size, size), fill)
    x0 = (size - w) // 2
    y0 = (size - h) // 2
    canvas.paste(img, (x0, y0))
    return canvas


def _load_and_standardize_image(path: Path, max_edge: int, pad_size: int) -> Image.Image:
    img = Image.open(path)
    try:
        img.draft("RGB", (max_edge, max_edge))
    except Exception:
        pass
    img = img.convert("RGB")
    img = _resize_max_edge(img, max_edge=max_edge)
    img = _pad_to_square(img, size=pad_size)
    return img


def _l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _overlay_grid(ax, w: int, h: int, patch: int, stride: int, color: str, lw: float = 1.0):
    for y in range(0, h + 1, stride):
        ax.plot([0, w], [y, y], color=color, linewidth=lw)
    for x in range(0, w + 1, stride):
        ax.plot([x, x], [0, h], color=color, linewidth=lw)


def _colorize_small_patch_grid(
    small_patches: List[Patch], labels: np.ndarray, grid_w: int, grid_h: int
) -> np.ndarray:
    n_labels = max(int(labels.max()) + 1, 1) if labels.size > 0 else 1
    if n_labels <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("nipy_spectral", n_labels)

    overlay = np.zeros((grid_h, grid_w, 4), dtype=np.float32)
    for p, lab in zip(small_patches, labels):
        if n_labels <= 20:
            r, g, b, _ = cmap(int(lab) % 20)
        else:
            r, g, b, _ = cmap(int(lab) / max(n_labels - 1, 1))
        overlay[p.y0 : p.y1, p.x0 : p.x1, 0] = r
        overlay[p.y0 : p.y1, p.x0 : p.x1, 1] = g
        overlay[p.y0 : p.y1, p.x0 : p.x1, 2] = b
        overlay[p.y0 : p.y1, p.x0 : p.x1, 3] = 0.45
    return overlay


def _silhouette_optional(
    x: np.ndarray, labels: np.ndarray, seed: int, sample_size: int = 20000
) -> Optional[float]:
    mask = labels >= 0
    if mask.sum() < 3:
        return None
    labs = labels[mask]
    if len(np.unique(labs)) < 2:
        return None

    x2 = x[mask]
    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        return None

    if x2.shape[0] > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x2.shape[0], size=sample_size, replace=False)
        x2 = x2[idx]
        labs = labs[idx]

    try:
        return float(silhouette_score(x2, labs, metric="euclidean"))
    except Exception:
        return None


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only clustering sweep over pre-computed embeddings."
    )
    parser.add_argument(
        "--embedding-dir",
        type=str,
        required=True,
        help="Path to the output directory from the GPU embedding run "
        "(contains embedding_data.npz and pca_model.pkl).",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="60",
        help="Comma-separated list of k values to try, e.g. '30,60,100,150'.",
    )
    parser.add_argument("--viz-num-images", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clusterer",
        type=str,
        default="sklearn_kmeans",
        choices=["sklearn_kmeans", "bisecting_kmeans", "gmm"],
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    if not k_values:
        print("ERROR: No k values provided.")
        return 1

    emb_dir = Path(args.embedding_dir)
    npz_path = emb_dir / "embedding_data.npz"
    pca_path = emb_dir / "pca_model.pkl"

    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run the GPU embedding script first.")
        return 1

    # ---- Load embeddings ----
    print(f"Loading embeddings from {npz_path} ...")
    data = np.load(npz_path, allow_pickle=True)
    X_pca = data["X_pca"].astype(np.float32)
    X_pca = _l2_normalize_np(X_pca)
    image_paths = [str(p) for p in data["image_paths"]]
    patches_per_image = int(data["patches_per_image"])
    pad_size = int(data["pad_size"])
    small_size = int(data["small_size"])
    small_stride = int(data["small_stride"])
    max_edge = int(data["max_edge"])

    # Optional: also load PCA dim info
    pca_dim = int(data["pca_dim"]) if "pca_dim" in data else X_pca.shape[1]

    num_images = len(image_paths)
    total_patches = X_pca.shape[0]
    expected_patches = num_images * patches_per_image

    print(f"  Images: {num_images}")
    print(f"  Patches per image: {patches_per_image}")
    print(f"  Total patch embeddings: {total_patches} (expected {expected_patches})")
    print(f"  PCA dim: {pca_dim}")
    print(f"  Pad size: {pad_size}, Small patch: {small_size}, Stride: {small_stride}")
    print(f"  Max edge: {max_edge}")
    print(f"  k values to sweep: {k_values}")
    print()

    if total_patches != expected_patches:
        print(
            f"WARNING: total patches ({total_patches}) != images*patches_per_image "
            f"({expected_patches}). Proceeding anyway."
        )

    # Pick viz images (first N from the list)
    viz_n = min(args.viz_num_images, num_images)
    rng = random.Random(args.seed)
    viz_indices = rng.sample(range(num_images), k=viz_n)
    viz_indices.sort()

    # Build the small-patch grid template (same for all images since they're all padded to pad_size)
    small_patches_template = _extract_grid_patches(
        pad_size, pad_size, patch=small_size, stride=small_stride
    )
    assert len(small_patches_template) == patches_per_image, (
        f"Grid gives {len(small_patches_template)} patches but data has {patches_per_image}"
    )

    # Sweep output dir
    sweep_id = time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path("outputs") / "cluster_sweep" / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[int, dict] = {}

    for k in k_values:
        print(f"{'='*60}")
        print(f"Clustering with k={k} ...")
        print(f"{'='*60}")

        k_dir = sweep_dir / f"k_{k:04d}"
        k_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()

        # ---- Fit clusterer ----
        if args.clusterer == "sklearn_kmeans":
            from sklearn.cluster import MiniBatchKMeans

            model = MiniBatchKMeans(
                n_clusters=k, random_state=args.seed, batch_size=1024, n_init=3, max_iter=300
            )
            fit_labels = model.fit_predict(X_pca)
        elif args.clusterer == "bisecting_kmeans":
            from sklearn.cluster import BisectingKMeans

            model = BisectingKMeans(n_clusters=k, random_state=args.seed, n_init=3, max_iter=300)
            fit_labels = model.fit_predict(X_pca)
        elif args.clusterer == "gmm":
            from sklearn.mixture import GaussianMixture

            model = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                random_state=args.seed,
                n_init=3,
                max_iter=300,
            )
            fit_labels = model.fit_predict(X_pca)
        else:
            raise ValueError(f"Unknown clusterer: {args.clusterer}")

        dt = time.time() - t0
        print(f"  Clustering took {dt:.1f}s")

        # ---- Evaluation ----
        counts = np.bincount(fit_labels.astype(np.int64), minlength=k)
        order = np.argsort(-counts)
        sil = _silhouette_optional(X_pca, fit_labels, seed=args.seed)

        print(f"  Silhouette: {sil:.4f}" if sil is not None else "  Silhouette: N/A")
        print(f"  Largest cluster: {int(counts[order[0]])} patches")
        print(f"  Smallest cluster: {int(counts[order[-1]])} patches")

        result = {
            "k": k,
            "clusterer": args.clusterer,
            "total_patches": int(total_patches),
            "num_images": num_images,
            "patches_per_image": patches_per_image,
            "pca_dim": pca_dim,
            "silhouette": float(sil) if sil is not None else None,
            "cluster_time_s": round(dt, 2),
            "cluster_sizes": counts.tolist(),
        }
        all_results[k] = result

        # ---- Cluster size distribution plot ----
        plt.figure(figsize=(12, 4))
        plt.bar(np.arange(k), counts[order])
        plt.title(f"Cluster sizes ({args.clusterer}, k={k}, sil={sil:.4f})" if sil else f"Cluster sizes ({args.clusterer}, k={k})")
        plt.xlabel("clusters sorted by size")
        plt.ylabel("# small patches")
        plt.tight_layout()
        plt.savefig(k_dir / "cluster_sizes.png", dpi=160)
        plt.close()

        # ---- Print top clusters ----
        print(f"\n  Top-20 clusters (of {k}):")
        for rank in range(min(20, k)):
            cid = int(order[rank])
            csz = int(counts[cid])
            print(f"    cluster {cid:03d}: {csz}")

        # ---- Visualize images ----
        print(f"\n  Visualizing {viz_n} images ...")
        for shown, img_idx in enumerate(viz_indices):
            img_path = Path(image_paths[img_idx])
            if not img_path.exists():
                print(f"    WARNING: image not found: {img_path}, skipping")
                continue

            # Get this image's embeddings (slice from the big matrix)
            start = img_idx * patches_per_image
            end = start + patches_per_image
            img_pca = X_pca[start:end]

            # Predict cluster labels
            pred_labels = model.predict(img_pca)

            # Load and standardize image for visualization
            img = _load_and_standardize_image(img_path, max_edge=max_edge, pad_size=pad_size)
            pw, ph = img.size

            fig, axs = plt.subplots(1, 3, figsize=(16, 6))

            # Panel 1: Standardized image
            axs[0].imshow(img)
            axs[0].set_title("Standardized image")
            axs[0].axis("off")

            # Panel 2: Large patch grid overlay
            large_size = 512  # Matches the config
            _overlay_grid(axs[1], w=pw, h=ph, patch=large_size, stride=large_size, color="lime", lw=1.5)
            axs[1].imshow(img)
            _overlay_grid(axs[1], w=pw, h=ph, patch=large_size, stride=large_size, color="lime", lw=1.5)
            axs[1].set_title(f"Large grid {large_size}x{large_size}")
            axs[1].axis("off")

            # Panel 3: Cluster overlay
            axs[2].imshow(img)
            overlay = _colorize_small_patch_grid(
                small_patches_template, pred_labels.astype(np.int64), grid_w=pw, grid_h=ph
            )
            axs[2].imshow(overlay)
            axs[2].set_title(f"Small patches colored by cluster (k={k})")
            axs[2].axis("off")

            # Extract class name from path for suptitle
            try:
                class_name = img_path.parent.parent.name
            except Exception:
                class_name = "?"
            fig.suptitle(f"{class_name} | k={k} | {args.clusterer}", fontsize=11)
            fig.tight_layout()
            fig.savefig(k_dir / f"image_{shown:02d}_clusters.png", dpi=160)
            plt.close(fig)

        # Save per-k summary
        (k_dir / "result.json").write_text(json.dumps(result, indent=2))
        print(f"  Saved to {k_dir}")
        print()

    # ---- Summary across k values ----
    summary = {
        "sweep_id": sweep_id,
        "embedding_dir": str(emb_dir),
        "clusterer": args.clusterer,
        "num_images": num_images,
        "patches_per_image": patches_per_image,
        "total_patches": int(total_patches),
        "pca_dim": pca_dim,
        "k_values": k_values,
        "results": {str(k): all_results[k] for k in k_values},
    }
    (sweep_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- Silhouette vs k plot ----
    sils = [all_results[k].get("silhouette") for k in k_values]
    valid_k = [k for k, s in zip(k_values, sils) if s is not None]
    valid_s = [s for s in sils if s is not None]
    if valid_k:
        plt.figure(figsize=(8, 4))
        plt.plot(valid_k, valid_s, "o-", markersize=8)
        for kv, sv in zip(valid_k, valid_s):
            plt.annotate(f"{sv:.3f}", (kv, sv), textcoords="offset points", xytext=(0, 10), ha="center")
        plt.title(f"Silhouette score vs k ({args.clusterer})")
        plt.xlabel("k")
        plt.ylabel("Silhouette")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(sweep_dir / "silhouette_vs_k.png", dpi=160)
        plt.close()

    print(f"Sweep complete. All outputs saved to: {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
