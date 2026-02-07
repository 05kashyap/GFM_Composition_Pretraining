#!/usr/bin/env python3
"""Visualize an FMoW patch->embed->cluster preprocessing workflow.

This script is intentionally standalone and does NOT touch any of the existing
DynamicVis training pipeline.

Workflow:
0) (Optional) Sample N images and plot width/height distribution.
1) Load local FMoW images from data/fmow/<split>/...
2) Resize so max(width,height) <= --max-edge, then pad to --pad-size x --pad-size.
3) Divide each padded image into LARGE patches (default 256x256)
4) Divide each LARGE patch into SMALL patches (default 64x64)
5) Embed each SMALL patch with a pretrained encoder (DINOv2 or MAE via HF)
6) Cluster SMALL-patch embeddings with a streaming MiniBatch-KMeans-like loop.
7) Visualize intermediate steps and plot cluster sizes.

Outputs are saved under: outputs/preprocess_viz/<run_id>/

Example:
  python scripts/viz_fmow_patch_embed_cluster.py \
        --data-root data/fmow --split val \
        --size-stats-n 1000 --cluster-num-images 500 --viz-num-images 25 \
        --pad-size 1024 --large-size 256 --small-size 64 --k 50 --embedder dinov2

Notes:
- DINOv2 and MAE weights will download via HuggingFace if not already cached.
- If downloading is not possible, use --embedder resnet18 as a fully local fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# FMoW contains some extremely large satellite images.
# Pillow raises DecompressionBombError for very large pixel counts as a safety
# guard; for this offline research pipeline we explicitly disable that limit.
Image.MAX_IMAGE_PIXELS = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Patch:
    """A patch of an image with its coordinates in the parent image."""

    x0: int
    y0: int
    x1: int
    y1: int


def _find_jpgs(data_root: Path, split: str) -> List[Path]:
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    # Expected structure: split/class/scene/*.jpg
    paths = list(split_dir.rglob("*.jpg")) + list(split_dir.rglob("*.jpeg"))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise RuntimeError(f"No jpg images found under {split_dir}")

    # Keep deterministic ordering by default (we still sample via seed)
    return sorted(paths)


def _sample_stratified(
    all_paths: List[Path], split_dir: Path, n: int, rng: random.Random
) -> List[Path]:
    """Sample n images with equal representation from every class.

    Class is determined by the first sub-directory level under *split_dir*.
    E.g. for ``split_dir/airport/scene_0/img.jpg`` the class is ``airport``.
    """
    class_to_paths: Dict[str, List[Path]] = defaultdict(list)
    for p in all_paths:
        try:
            rel = p.relative_to(split_dir)
            class_name = rel.parts[0]
            class_to_paths[class_name].append(p)
        except (ValueError, IndexError):
            continue

    if not class_to_paths:
        print("WARNING: Could not determine classes for stratified sampling, falling back to random.")
        return rng.sample(all_paths, k=min(n, len(all_paths)))

    num_classes = len(class_to_paths)
    per_class = max(1, n // num_classes)
    remainder = max(0, n - per_class * num_classes)

    sampled: List[Path] = []
    class_names = sorted(class_to_paths.keys())

    for i, cls in enumerate(class_names):
        pool = class_to_paths[cls]
        take = per_class + (1 if i < remainder else 0)
        take = min(take, len(pool))
        sampled.extend(rng.sample(pool, k=take))

    rng.shuffle(sampled)
    actual_n = min(n, len(sampled))
    sampled = sampled[:actual_n]

    print(
        f"Stratified sampling: {actual_n} images from {num_classes} classes "
        f"(~{per_class} per class)"
    )
    return sampled


def _resize_max_edge(img: Image.Image, max_edge: int) -> Image.Image:
    if max_edge <= 0:
        return img
    w, h = img.size
    scale = max_edge / max(w, h)
    if scale >= 1.0:
        return img
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)


def _pad_to_square(img: Image.Image, size: int, fill: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Pad an image to a size x size canvas, centering the original.

    Assumes the image is already <= size in both dimensions.
    """

    w, h = img.size
    if w > size or h > size:
        raise ValueError(f"Image {w}x{h} is larger than pad size {size}. Reduce via --max-edge first.")
    if w == size and h == size:
        return img
    canvas = Image.new("RGB", (size, size), fill)
    x0 = (size - w) // 2
    y0 = (size - h) // 2
    canvas.paste(img, (x0, y0))
    return canvas


def _load_and_standardize_image(path: Path, max_edge: int, pad_size: int) -> Image.Image:
    img = Image.open(path)
    # Hint to decoder to downscale on load when possible (JPEG), to avoid
    # materializing huge full-resolution arrays before our own resize.
    try:
        img.draft("RGB", (max_edge, max_edge))
    except Exception:
        pass
    img = img.convert("RGB")
    img = _resize_max_edge(img, max_edge=max_edge)
    img = _pad_to_square(img, size=pad_size, fill=(0, 0, 0))
    return img


def _extract_grid_patches(w: int, h: int, patch: int, stride: int) -> List[Patch]:
    if patch <= 0 or stride <= 0:
        raise ValueError("patch and stride must be positive")

    patches: List[Patch] = []
    # Only full patches (simplest + most reproducible)
    for y0 in range(0, h - patch + 1, stride):
        for x0 in range(0, w - patch + 1, stride):
            patches.append(Patch(x0=x0, y0=y0, x1=x0 + patch, y1=y0 + patch))
    return patches


def _crop(img: Image.Image, patch: Patch) -> Image.Image:
    return img.crop((patch.x0, patch.y0, patch.x1, patch.y1))


def _overlay_grid(ax, w: int, h: int, patch: int, stride: int, color: str, lw: float = 1.0):
    for y in range(0, h + 1, stride):
        ax.plot([0, w], [y, y], color=color, linewidth=lw)
    for x in range(0, w + 1, stride):
        ax.plot([x, x], [0, h], color=color, linewidth=lw)


class Embedder:
    def __init__(self, device: str):
        self.device = torch.device(device)

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError

    def embed_pil(self, images: Sequence[Image.Image], batch_size: int) -> torch.Tensor:
        """Return embeddings (N, D) on CPU."""
        raise NotImplementedError


class HFVisionEmbedder(Embedder):
    def __init__(
        self,
        device: str,
        model_id: str,
        family: str,
    ):
        super().__init__(device=device)
        self.model_id = model_id
        self.family = family

        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval().to(self.device)

        # Infer embedding dim with a tiny forward pass
        with torch.no_grad():
            dummy = Image.new("RGB", (224, 224), (127, 127, 127))
            inputs = self.processor(images=[dummy], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            # Common HF vision models expose last_hidden_state
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                d = int(out.last_hidden_state.shape[-1])
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                d = int(out.pooler_output.shape[-1])
            else:
                raise RuntimeError(
                    f"Cannot infer embedding dim for model_id={model_id}; output keys={out.keys()}"
                )
        self._embedding_dim = d

    @property
    def name(self) -> str:
        return f"hf:{self.model_id}"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @torch.inference_mode()
    def embed_pil(self, images: Sequence[Image.Image], batch_size: int) -> torch.Tensor:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()

        all_embs: List[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=list(batch), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)

            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                # Use CLS token
                embs = out.last_hidden_state[:, 0, :]
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                embs = out.pooler_output
            else:
                raise RuntimeError(
                    f"Model output did not contain last_hidden_state/pooler_output; keys={out.keys()}"
                )

            all_embs.append(embs.detach().cpu())

        return torch.cat(all_embs, dim=0)


class ResNet18Embedder(Embedder):
    def __init__(self, device: str):
        super().__init__(device=device)
        import torchvision

        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
        # Remove classifier head
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(self.device)
        self.transforms = weights.transforms()
        self._embedding_dim = 512

    @property
    def name(self) -> str:
        return "torchvision:resnet18"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @torch.inference_mode()
    def embed_pil(self, images: Sequence[Image.Image], batch_size: int) -> torch.Tensor:
        all_embs: List[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            x = torch.stack([self.transforms(img) for img in batch], dim=0).to(self.device)
            embs = self.model(x).detach().cpu()
            all_embs.append(embs)
        return torch.cat(all_embs, dim=0)
def build_embedder(embedder: str, device: str, hf_model_id: str = "") -> Embedder:
    """Create embedder with explicit defaults.

    Notes:
    - You can override any HF-backed embedder via --hf-model-id.
    - DINOv3 model IDs may evolve; if the default below fails, pass the
      correct HuggingFace model id explicitly.
    """

    hf_model_id = (hf_model_id or "").strip()

    if embedder == "dinov2":
        model_id = hf_model_id or "facebook/dinov2-base"
        return HFVisionEmbedder(device=device, model_id=model_id, family="dinov2")
    if embedder == "dinov3l":
        # Best-effort default; override with --hf-model-id if needed.
        model_id = hf_model_id or "facebook/dinov3-vitl16"
        return HFVisionEmbedder(device=device, model_id=model_id, family="dinov3")
    if embedder == "mae":
        model_id = hf_model_id or "facebook/vit-mae-base"
        return HFVisionEmbedder(device=device, model_id=model_id, family="mae")
    if embedder == "hf":
        if not hf_model_id:
            raise ValueError("--embedder hf requires --hf-model-id")
        return HFVisionEmbedder(device=device, model_id=hf_model_id, family="hf")
    if embedder == "resnet18":
        return ResNet18Embedder(device=device)

    raise ValueError(f"Unknown embedder: {embedder}")


def _save_image_embeddings(
    out_dir: Path,
    shown_idx: int,
    image_path: Path,
    patches: List[Patch],
    emb_raw: np.ndarray,
    labels: np.ndarray,
    emb_pca: Optional[np.ndarray] = None,
) -> None:
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    patch_xyxy = np.asarray([[p.x0, p.y0, p.x1, p.y1] for p in patches], dtype=np.int32)

    save_path = emb_dir / f"image_{shown_idx:02d}.npz"
    if emb_pca is None:
        np.savez_compressed(
            save_path,
            image_path=str(image_path),
            patch_xyxy=patch_xyxy,
            emb_raw=emb_raw.astype(np.float32, copy=False),
            labels=labels.astype(np.int64, copy=False),
        )
    else:
        np.savez_compressed(
            save_path,
            image_path=str(image_path),
            patch_xyxy=patch_xyxy,
            emb_raw=emb_raw.astype(np.float32, copy=False),
            emb_pca=emb_pca.astype(np.float32, copy=False),
            labels=labels.astype(np.int64, copy=False),
        )


def _cache_key(
    *,
    image_path: Path,
    embedder_name: str,
    max_edge: int,
    pad_size: int,
    small_size: int,
    small_stride: int,
    local_weights: str,
) -> str:
    st = None
    try:
        if local_weights:
            st = os.stat(local_weights)
    except Exception:
        st = None
    payload = {
        "image": str(image_path),
        "embedder": str(embedder_name),
        "max_edge": int(max_edge),
        "pad_size": int(pad_size),
        "small_size": int(small_size),
        "small_stride": int(small_stride),
        "weights_path": str(local_weights),
        "weights_mtime": float(st.st_mtime) if st is not None else None,
        "weights_size": int(st.st_size) if st is not None else None,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _get_or_compute_small_patch_embeddings(
    *,
    cache_dir: Path,
    image_path: Path,
    embedder: Embedder,
    img: Image.Image,
    small_patches_full: List[Patch],
    small_imgs: List[Image.Image],
    embed_batch: int,
    max_edge: int,
    pad_size: int,
    small_size: int,
    small_stride: int,
    local_weights: str,
    enable_cache: bool,
) -> np.ndarray:
    """Return embeddings for all small patches of an image, using disk cache if enabled."""

    if not enable_cache:
        embs = embedder.embed_pil(small_imgs, batch_size=embed_batch).numpy().astype(np.float32)
        return embs

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(
        image_path=image_path,
        embedder_name=embedder.name,
        max_edge=max_edge,
        pad_size=pad_size,
        small_size=small_size,
        small_stride=small_stride,
        local_weights=local_weights,
    )
    fpath = cache_dir / f"{key}.npz"
    if fpath.exists():
        try:
            data = np.load(fpath, allow_pickle=False)
            embs = data["emb"].astype(np.float32, copy=False)
            if embs.shape[0] != len(small_patches_full):
                raise RuntimeError("cached embedding count mismatch")
            return embs
        except Exception:
            # Corrupt/old cache; recompute.
            pass

    embs = embedder.embed_pil(small_imgs, batch_size=embed_batch).numpy().astype(np.float32)
    # Store float16 to reduce disk.
    np.savez_compressed(
        fpath,
        emb=embs.astype(np.float16, copy=False),
        image_path=str(image_path),
    )
    return embs


class StreamingKMeans:
    """A tiny streaming MiniBatch-KMeans-like implementation.

    Uses cosine-like distance by L2-normalizing vectors and comparing dot products.
    """

    def __init__(self, k: int, seed: int):
        if k <= 1:
            raise ValueError("k must be >= 2")
        self.k = k
        self.seed = seed
        self._g = torch.Generator(device="cpu")
        self._g.manual_seed(seed)
        self.centroids: Optional[torch.Tensor] = None  # (k, d) CPU, float32
        self.counts = torch.zeros(k, dtype=torch.long)

    def _maybe_init(self, X: torch.Tensor) -> None:
        if self.centroids is not None:
            return
        if X.ndim != 2:
            raise ValueError("X must be (N, D)")
        n = X.shape[0]
        if n < self.k:
            raise ValueError(f"Need at least k samples to init: n={n}, k={self.k}")
        idx = torch.randperm(n, generator=self._g)[: self.k]
        self.centroids = X[idx].clone()
        # Give each centroid a non-zero count so future etas are stable
        self.counts += 1

    @torch.no_grad()
    def partial_fit(self, X: torch.Tensor) -> None:
        X = F.normalize(X.float().cpu(), dim=1)
        self._maybe_init(X)
        assert self.centroids is not None

        dots = X @ self.centroids.t()  # (n, k)
        dist = 2.0 - 2.0 * dots
        assign = torch.argmin(dist, dim=1)

        for j in range(self.k):
            mask = assign == j
            if not torch.any(mask):
                continue
            xj = X[mask]
            self.counts[j] += xj.shape[0]
            eta = 1.0 / float(self.counts[j].item())
            self.centroids[j] = (1.0 - eta) * self.centroids[j] + eta * xj.mean(dim=0)
            self.centroids[j] = F.normalize(self.centroids[j], dim=0)

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            raise RuntimeError("KMeans not initialized")
        X = F.normalize(X.float().cpu(), dim=1)
        dots = X @ self.centroids.t()
        dist = 2.0 - 2.0 * dots
        return torch.argmin(dist, dim=1)


def _l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _fit_pca(x: np.ndarray, pca_dim: int, seed: int):
    try:
        from sklearn.decomposition import PCA
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for PCA. Install with: pip install scikit-learn\n"
            f"Root error: {e}"
        )

    pca_dim = int(pca_dim)
    if pca_dim <= 0 or pca_dim > x.shape[1]:
        raise ValueError(f"Invalid pca_dim={pca_dim} for embedding_dim={x.shape[1]}")
    pca = PCA(n_components=pca_dim, random_state=seed)
    x_pca = pca.fit_transform(x)
    return pca, x_pca


def _fit_hdbscan(x: np.ndarray, min_cluster_size: int, min_samples: int, n_jobs: int):
    try:
        import hdbscan
    except Exception as e:
        raise RuntimeError(
            "hdbscan is required for HDBSCAN clustering. Install with: pip install hdbscan\n"
            f"Root error: {e}"
        )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples) if int(min_samples) > 0 else None,
        metric="euclidean",
        core_dist_n_jobs=int(n_jobs),
        prediction_data=True,
    )
    labels = clusterer.fit_predict(x)
    return clusterer, labels


def _silhouette_optional(x: np.ndarray, labels: np.ndarray, seed: int, sample_size: int = 20000) -> Optional[float]:
    """Compute silhouette on non-noise points if feasible."""

    # Remove noise
    mask = labels >= 0
    if mask.sum() < 3:
        return None
    labs = labels[mask]
    # Need at least 2 clusters
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


# ---------------------------------------------------------------------------
# Additional clustering back-ends (parametric, fixed-k)
# ---------------------------------------------------------------------------

def _fit_sklearn_kmeans(x: np.ndarray, k: int, seed: int, batch_size: int = 1024):
    """MiniBatchKMeans from scikit-learn (fast, scalable to large N)."""
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for sklearn_kmeans. "
            "Install with: pip install scikit-learn\n"
            f"Root error: {e}"
        )
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=batch_size,
        n_init=3,
        max_iter=300,
    )
    labels = km.fit_predict(x)
    return km, labels


def _fit_bisecting_kmeans(x: np.ndarray, k: int, seed: int):
    """Bisecting K-Means (hierarchical divisive) from scikit-learn >= 1.1."""
    try:
        from sklearn.cluster import BisectingKMeans
    except Exception as e:
        raise RuntimeError(
            "scikit-learn >= 1.1 is required for bisecting_kmeans.\n"
            f"Root error: {e}"
        )
    bkm = BisectingKMeans(
        n_clusters=k,
        random_state=seed,
        n_init=3,
        max_iter=300,
    )
    labels = bkm.fit_predict(x)
    return bkm, labels


def _fit_gmm(x: np.ndarray, k: int, seed: int):
    """Gaussian Mixture Model (soft clustering, diagonal covariance)."""
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for gmm. "
            "Install with: pip install scikit-learn\n"
            f"Root error: {e}"
        )
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=seed,
        n_init=3,
        max_iter=300,
    )
    labels = gmm.fit_predict(x)
    return gmm, labels


def _colorize_small_patch_grid(
    small_patches: List[Patch],
    labels: np.ndarray,
    grid_w: int,
    grid_h: int,
) -> np.ndarray:
    """Create an RGB overlay for a small-patch grid."""

    # Pick a colormap that can handle many clusters
    n_labels = max(int(labels.max()) + 1, 1) if labels.size > 0 else 1
    if n_labels <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("nipy_spectral", n_labels)
    overlay = np.zeros((grid_h, grid_w, 4), dtype=np.float32)

    # Each patch is filled with its cluster color at alpha=0.45
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/fmow")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)

    # Stage 0: image size statistics
    parser.add_argument("--size-stats-n", type=int, default=1000)

    # Stage 1/2: clustering + visualization
    parser.add_argument("--cluster-num-images", type=int, default=500)
    parser.add_argument("--viz-num-images", type=int, default=10)
    parser.add_argument("--max-edge", type=int, default=1024, help="Resize image so max(w,h)<=max-edge")
    parser.add_argument("--pad-size", type=int, default=1024, help="Pad each image to pad-size x pad-size")
    parser.add_argument("--large-size", type=int, default=256)
    parser.add_argument("--large-stride", type=int, default=256)
    parser.add_argument("--small-size", type=int, default=64)
    parser.add_argument("--small-stride", type=int, default=64)

    parser.add_argument(
        "--embedder",
        type=str,
        default="dinov2",
        choices=["dinov2", "dinov3l", "mae", "hf", "resnet18"],
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="",
        help="HuggingFace model id override (used by dinov2/dinov3l/mae/hf).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed-batch", type=int, default=64)

    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="If set, save per-viz-image small-patch embeddings+labels as .npz under outputs/.../embeddings/",
    )

    parser.add_argument(
        "--cache-embeddings",
        action="store_true",
        help="If set, cache per-image small-patch embeddings on disk to avoid recomputation across runs.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="outputs/preprocess_cache",
        help="Directory to store cached per-image embeddings when --cache-embeddings is set.",
    )

    parser.add_argument(
        "--clusterer", type=str, default="sklearn_kmeans",
        choices=["sklearn_kmeans", "bisecting_kmeans", "gmm", "hdbscan", "stream_kmeans"],
    )

    # Shared embed/PCA params (used by hdbscan, sklearn_kmeans, bisecting_kmeans, gmm)
    parser.add_argument("--fit-small-patches-per-image", type=int, default=32)
    parser.add_argument("--pca-dim", type=int, default=64)

    # HDBSCAN pipeline params
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=15)
    parser.add_argument("--hdbscan-min-samples", type=int, default=0, help="0 => use default/None")
    parser.add_argument("--hdbscan-jobs", type=int, default=8)
    parser.add_argument(
        "--assign-noise-to-nearest-centroid",
        action="store_true",
        help="If set, assign HDBSCAN noise points (-1) to nearest cluster centroid for viz",
    )

    # Fixed-k params (sklearn_kmeans, bisecting_kmeans, gmm, stream_kmeans)
    parser.add_argument("--k", type=int, default=50, help="Number of clusters")
    parser.add_argument("--kmeans-iters", type=int, default=200)
    parser.add_argument(
        "--small-patches-per-image",
        type=int,
        default=0,
        help="If >0, randomly sample this many small patches per image (else use all)",
    )

    args = parser.parse_args()

    _seed_all(args.seed)

    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    all_paths = _find_jpgs(data_root, args.split)
    print(f"Total images found: {len(all_paths)}")

    rng = random.Random(args.seed)

    # Create output directory
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / "preprocess_viz" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)

    # Print device banner
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Device: cuda (GPU: {gpu_name})")
    else:
        print(f"Device: {device.type}")

    # Stage 0: image size distribution for a sample of images
    size_n = min(args.size_stats_n, len(all_paths))
    size_paths = _sample_stratified(all_paths, split_dir, size_n, rng)
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
    else:
        print("WARNING: Could not compute size distribution (no readable images in sample).")

    # Stage 1/2: choose images for clustering+viz
    cluster_n = min(args.cluster_num_images, len(all_paths))
    cluster_paths = _sample_stratified(all_paths, split_dir, cluster_n, rng)
    viz_n = min(args.viz_num_images, cluster_n)
    viz_set = set(cluster_paths[:viz_n])

    # Build embedder
    try:
        embedder = build_embedder(args.embedder, device=args.device, hf_model_id=args.hf_model_id)
    except Exception as e:
        raise RuntimeError(
            f"Failed to init embedder '{args.embedder}'. "
            f"If you are offline, try --embedder resnet18.\nRoot error: {e}"
        )

    def iter_small_patches_for_image(img: Image.Image) -> Tuple[List[Patch], List[Image.Image]]:
        """Return (small_patches_in_full_image, small_patch_images)."""
        pw, ph = img.size
        if pw != args.pad_size or ph != args.pad_size:
            raise RuntimeError("Expected padded image")

        # Small patches directly on full padded image
        small_patches_full = _extract_grid_patches(
            pw, ph, patch=args.small_size, stride=args.small_stride
        )
        if args.small_patches_per_image and args.small_patches_per_image > 0:
            # Sample a subset for faster runs
            if args.small_patches_per_image < len(small_patches_full):
                small_patches_full = rng.sample(small_patches_full, k=args.small_patches_per_image)

        small_imgs = [_crop(img, sp) for sp in small_patches_full]
        return small_patches_full, small_imgs
    # --- Clustering ---
    clusterer_name = str(args.clusterer)
    counts: np.ndarray
    order: np.ndarray
    eval_report: Dict[str, object] = {}
    shown = 0

    if clusterer_name == "hdbscan":
        # Pass A: collect a fit subset of embeddings (e.g., 32 small patches per image)
        fit_per_image = max(1, int(args.fit_small_patches_per_image))
        fit_embs: List[np.ndarray] = []
        t0 = time.time()

        for i, p in enumerate(cluster_paths):
            img = _load_and_standardize_image(p, max_edge=args.max_edge, pad_size=args.pad_size)
            small_patches_full, small_imgs = iter_small_patches_for_image(img)

            # sample fit_per_image patches
            if fit_per_image < len(small_imgs):
                pick = rng.sample(range(len(small_imgs)), k=fit_per_image)
                small_imgs_fit = [small_imgs[j] for j in pick]
            else:
                small_imgs_fit = small_imgs

            embs = embedder.embed_pil(small_imgs_fit, batch_size=args.embed_batch).numpy().astype(np.float32)
            fit_embs.append(embs)

            if (i + 1) % 50 == 0 or (i + 1) == cluster_n:
                dt = time.time() - t0
                ips = (i + 1) / max(dt, 1e-6)
                print(f"Embedding (fit set): {i+1}/{cluster_n} images | {ips:.2f} img/s")

        X = np.concatenate(fit_embs, axis=0)
        X = _l2_normalize_np(X)
        print(f"Fit embedding matrix: {X.shape} (N,D)")

        # PCA
        pca, X_pca = _fit_pca(X, pca_dim=int(args.pca_dim), seed=args.seed)
        X_pca = _l2_normalize_np(X_pca.astype(np.float32))
        print(f"After PCA: {X_pca.shape}")

        # HDBSCAN
        min_samples = int(args.hdbscan_min_samples)
        hdb, fit_labels = _fit_hdbscan(
            X_pca,
            min_cluster_size=int(args.hdbscan_min_cluster_size),
            min_samples=min_samples,
            n_jobs=int(args.hdbscan_jobs),
        )

        n_noise = int((fit_labels < 0).sum())
        n_total = int(fit_labels.shape[0])
        n_clusters = int(len(set(fit_labels.tolist())) - (1 if -1 in fit_labels else 0))
        noise_frac = float(n_noise / max(n_total, 1))
        sil = _silhouette_optional(X_pca, fit_labels, seed=args.seed)

        print(f"HDBSCAN: clusters={n_clusters} | noise={n_noise}/{n_total} ({noise_frac*100:.2f}%)")
        if sil is not None:
            print(f"Silhouette (non-noise): {sil:.4f}")
        else:
            print("Silhouette (non-noise): N/A")

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

        # Cluster size histogram based on fit set labels (excluding noise).
        # If HDBSCAN yields only noise on small runs, still complete the script:
        # treat everything as a single "unknown" cluster for plotting/viz.
        valid = fit_labels[fit_labels >= 0]
        if valid.size == 0:
            print(
                "WARNING: HDBSCAN produced only noise on the fit set. "
                "Proceeding with a single 'unknown' cluster for viz/histogram. "
                "For real runs, lower --hdbscan-min-cluster-size and/or increase fit data."
            )
            counts = np.asarray([n_total], dtype=np.int64)
            order = np.asarray([0], dtype=np.int64)
            centroids = np.zeros((1, X_pca.shape[1]), dtype=np.float32)
        else:
            max_label = int(valid.max())
            counts = np.bincount(valid.astype(np.int64), minlength=max_label + 1)
            order = np.argsort(-counts)

        # Precompute centroids in PCA space for optional noise reassignment
        if valid.size != 0:
            centroids = np.zeros((counts.shape[0], X_pca.shape[1]), dtype=np.float32)
            for cid in range(counts.shape[0]):
                mask = fit_labels == cid
                if not np.any(mask):
                    continue
                centroids[cid] = X_pca[mask].mean(axis=0)
            centroids = _l2_normalize_np(centroids)

        # Pass B: per-image assignment for viz images (10 images requested)
        try:
            import hdbscan
        except Exception:
            hdbscan = None

        for img_i, p in enumerate(cluster_paths):
            if p not in viz_set or shown >= viz_n:
                continue

            img = _load_and_standardize_image(p, max_edge=args.max_edge, pad_size=args.pad_size)
            small_patches_full, small_imgs = iter_small_patches_for_image(img)
            embs_all = _get_or_compute_small_patch_embeddings(
                cache_dir=cache_dir,
                image_path=p,
                embedder=embedder,
                img=img,
                small_patches_full=small_patches_full,
                small_imgs=small_imgs,
                embed_batch=args.embed_batch,
                max_edge=args.max_edge,
                pad_size=args.pad_size,
                small_size=args.small_size,
                small_stride=args.small_stride,
                local_weights=args.local_weights,
                enable_cache=bool(args.cache_embeddings),
            )
            embs_all = _l2_normalize_np(embs_all)
            embs_pca = pca.transform(embs_all).astype(np.float32)
            embs_pca = _l2_normalize_np(embs_pca)

            if hdbscan is not None and valid.size != 0:
                pred_labels, _strength = hdbscan.approximate_predict(hdb, embs_pca)
            else:
                pred_labels = np.full((embs_pca.shape[0],), -1, dtype=np.int64)

            # If no clusters exist, map everything to 0 (unknown) so the overlay still works.
            if valid.size == 0:
                pred_labels[:] = 0

            if args.assign_noise_to_nearest_centroid and valid.size != 0:
                noise_mask = pred_labels < 0
                if np.any(noise_mask):
                    # cosine distance for normalized vectors
                    sims = embs_pca[noise_mask] @ centroids.T
                    nn = np.argmax(sims, axis=1)
                    pred_labels[noise_mask] = nn.astype(np.int64)

            if args.save_embeddings:
                _save_image_embeddings(
                    out_dir=out_dir,
                    shown_idx=shown,
                    image_path=p,
                    patches=small_patches_full,
                    emb_raw=embs_all,
                    emb_pca=embs_pca,
                    labels=pred_labels,
                )

            fig, axs = plt.subplots(1, 3, figsize=(16, 6))

            axs[0].imshow(img)
            axs[0].set_title("Standardized image (resized + padded)")
            axs[0].axis("off")

            axs[1].imshow(img)
            pw, ph = img.size
            _overlay_grid(
                axs[1],
                w=pw,
                h=ph,
                patch=args.large_size,
                stride=args.large_stride,
                color="lime",
                lw=1.0,
            )
            axs[1].set_title(f"Large grid {args.large_size}x{args.large_size}")
            axs[1].axis("off")

            axs[2].imshow(img)
            overlay = _colorize_small_patch_grid(
                small_patches_full,
                pred_labels.astype(np.int64),
                grid_w=pw,
                grid_h=ph,
            )
            axs[2].imshow(overlay)
            axs[2].set_title("Small patches colored by cluster")
            axs[2].axis("off")

            fig.suptitle(
                f"{p.relative_to(Path.cwd()) if p.is_absolute() else p} | embedder={embedder.name} | clusters={counts.shape[0]}",
                fontsize=11,
            )
            fig.tight_layout()
            fig.savefig(out_dir / f"image_{shown:02d}_clusters.png", dpi=160)
            plt.close(fig)
            shown += 1

    elif clusterer_name in ("sklearn_kmeans", "bisecting_kmeans", "gmm"):
        # ---------------------------------------------------------------
        # Fixed-k clustering: MiniBatchKMeans / BisectingKMeans / GMM
        # ---------------------------------------------------------------
        k = int(args.k)
        fit_per_image = max(1, int(args.fit_small_patches_per_image))
        fit_embs: List[np.ndarray] = []
        t0 = time.time()

        for i, p in enumerate(cluster_paths):
            img = _load_and_standardize_image(p, max_edge=args.max_edge, pad_size=args.pad_size)
            small_patches_full, small_imgs = iter_small_patches_for_image(img)

            if fit_per_image < len(small_imgs):
                pick = rng.sample(range(len(small_imgs)), k=fit_per_image)
                small_imgs_fit = [small_imgs[j] for j in pick]
            else:
                small_imgs_fit = small_imgs

            embs = embedder.embed_pil(small_imgs_fit, batch_size=args.embed_batch).numpy().astype(np.float32)
            fit_embs.append(embs)

            if (i + 1) % 50 == 0 or (i + 1) == cluster_n:
                dt = time.time() - t0
                ips = (i + 1) / max(dt, 1e-6)
                print(f"Embedding (fit set): {i+1}/{cluster_n} images | {ips:.2f} img/s")

        X = np.concatenate(fit_embs, axis=0)
        X = _l2_normalize_np(X)
        print(f"Fit embedding matrix: {X.shape} (N,D)")

        # PCA
        pca, X_pca = _fit_pca(X, pca_dim=int(args.pca_dim), seed=args.seed)
        X_pca = _l2_normalize_np(X_pca.astype(np.float32))
        print(f"After PCA: {X_pca.shape}")

        # Cluster
        if clusterer_name == "sklearn_kmeans":
            model, fit_labels = _fit_sklearn_kmeans(X_pca, k=k, seed=args.seed)
        elif clusterer_name == "bisecting_kmeans":
            model, fit_labels = _fit_bisecting_kmeans(X_pca, k=k, seed=args.seed)
        else:  # gmm
            model, fit_labels = _fit_gmm(X_pca, k=k, seed=args.seed)

        n_clusters_found = int(len(set(fit_labels.tolist())))
        sil = _silhouette_optional(X_pca, fit_labels, seed=args.seed)
        print(f"{clusterer_name} (k={k}): found {n_clusters_found} non-empty clusters")
        if sil is not None:
            print(f"Silhouette: {sil:.4f}")

        eval_report = {
            "method": clusterer_name,
            "fit_points": int(X_pca.shape[0]),
            "pca_dim": int(args.pca_dim),
            "k": k,
            "n_clusters": n_clusters_found,
            "silhouette": float(sil) if sil is not None else None,
        }

        counts = np.bincount(fit_labels.astype(np.int64), minlength=k)
        order = np.argsort(-counts)

        # Pass B: per-image assignment for viz images
        for img_i, p in enumerate(cluster_paths):
            if p not in viz_set or shown >= viz_n:
                continue

            img = _load_and_standardize_image(p, max_edge=args.max_edge, pad_size=args.pad_size)
            small_patches_full, small_imgs = iter_small_patches_for_image(img)
            embs_all = _get_or_compute_small_patch_embeddings(
                cache_dir=cache_dir,
                image_path=p,
                embedder=embedder,
                img=img,
                small_patches_full=small_patches_full,
                small_imgs=small_imgs,
                embed_batch=args.embed_batch,
                max_edge=args.max_edge,
                pad_size=args.pad_size,
                small_size=args.small_size,
                small_stride=args.small_stride,
                local_weights=args.local_weights,
                enable_cache=bool(args.cache_embeddings),
            )
            embs_all = _l2_normalize_np(embs_all)
            embs_pca = pca.transform(embs_all).astype(np.float32)
            embs_pca = _l2_normalize_np(embs_pca)

            pred_labels = model.predict(embs_pca)

            if args.save_embeddings:
                _save_image_embeddings(
                    out_dir=out_dir,
                    shown_idx=shown,
                    image_path=p,
                    patches=small_patches_full,
                    emb_raw=embs_all,
                    emb_pca=embs_pca,
                    labels=pred_labels,
                )

            fig, axs = plt.subplots(1, 3, figsize=(16, 6))

            axs[0].imshow(img)
            axs[0].set_title("Standardized image (resized + padded)")
            axs[0].axis("off")

            axs[1].imshow(img)
            pw, ph = img.size
            _overlay_grid(
                axs[1], w=pw, h=ph,
                patch=args.large_size, stride=args.large_stride,
                color="lime", lw=1.0,
            )
            axs[1].set_title(f"Large grid {args.large_size}x{args.large_size}")
            axs[1].axis("off")

            axs[2].imshow(img)
            overlay = _colorize_small_patch_grid(
                small_patches_full,
                pred_labels.astype(np.int64),
                grid_w=pw,
                grid_h=ph,
            )
            axs[2].imshow(overlay)
            axs[2].set_title("Small patches colored by cluster")
            axs[2].axis("off")

            fig.suptitle(
                f"{p.relative_to(Path.cwd()) if p.is_absolute() else p} | "
                f"embedder={embedder.name} | {clusterer_name} k={k}",
                fontsize=11,
            )
            fig.tight_layout()
            fig.savefig(out_dir / f"image_{shown:02d}_clusters.png", dpi=160)
            plt.close(fig)
            shown += 1

    else:
        # Fallback: streaming kmeans over all small patches of all images
        k = int(args.k)
        kmeans = StreamingKMeans(k=k, seed=args.seed)
        t0 = time.time()

        for i, p in enumerate(cluster_paths):
            img = _load_and_standardize_image(p, max_edge=args.max_edge, pad_size=args.pad_size)
            _small_patches, small_imgs = iter_small_patches_for_image(img)
            # For stream_kmeans we embed all small patches; caching applies.
            embs_np = _get_or_compute_small_patch_embeddings(
                cache_dir=cache_dir,
                image_path=p,
                embedder=embedder,
                img=img,
                small_patches_full=_small_patches,
                small_imgs=small_imgs,
                embed_batch=args.embed_batch,
                max_edge=args.max_edge,
                pad_size=args.pad_size,
                small_size=args.small_size,
                small_stride=args.small_stride,
                local_weights=args.local_weights,
                enable_cache=bool(args.cache_embeddings),
            )
            embs = torch.from_numpy(embs_np)
            kmeans.partial_fit(embs)
            if (i + 1) % 50 == 0 or (i + 1) == cluster_n:
                dt = time.time() - t0
                ips = (i + 1) / max(dt, 1e-6)
                print(f"Centroid learning: {i+1}/{cluster_n} images | {ips:.2f} img/s")

        counts = np.zeros(k, dtype=np.int64)
        for img_i, p in enumerate(cluster_paths):
            img = _load_and_standardize_image(p, max_edge=args.max_edge, pad_size=args.pad_size)
            small_patches_full, small_imgs = iter_small_patches_for_image(img)
            embs_np = _get_or_compute_small_patch_embeddings(
                cache_dir=cache_dir,
                image_path=p,
                embedder=embedder,
                img=img,
                small_patches_full=small_patches_full,
                small_imgs=small_imgs,
                embed_batch=args.embed_batch,
                max_edge=args.max_edge,
                pad_size=args.pad_size,
                small_size=args.small_size,
                small_stride=args.small_stride,
                local_weights=args.local_weights,
                enable_cache=bool(args.cache_embeddings),
            )
            embs = torch.from_numpy(embs_np)
            lab_t = kmeans.predict(embs)
            labs = lab_t.numpy().astype(np.int64)
            counts += np.bincount(labs, minlength=k)

            if p in viz_set and shown < viz_n:
                if args.save_embeddings:
                    _save_image_embeddings(
                        out_dir=out_dir,
                        shown_idx=shown,
                        image_path=p,
                        patches=small_patches_full,
                        emb_raw=embs.detach().cpu().numpy().astype(np.float32, copy=False),
                        labels=labs,
                    )
                fig, axs = plt.subplots(1, 3, figsize=(16, 6))
                axs[0].imshow(img)
                axs[0].set_title("Standardized image (resized + padded)")
                axs[0].axis("off")
                axs[1].imshow(img)
                pw, ph = img.size
                _overlay_grid(
                    axs[1],
                    w=pw,
                    h=ph,
                    patch=args.large_size,
                    stride=args.large_stride,
                    color="lime",
                    lw=1.0,
                )
                axs[1].set_title(f"Large grid {args.large_size}x{args.large_size}")
                axs[1].axis("off")
                axs[2].imshow(img)
                overlay = _colorize_small_patch_grid(
                    small_patches_full,
                    labs,
                    grid_w=pw,
                    grid_h=ph,
                )
                axs[2].imshow(overlay)
                axs[2].set_title("Small patches colored by cluster")
                axs[2].axis("off")
                fig.tight_layout()
                fig.savefig(out_dir / f"image_{shown:02d}_clusters.png", dpi=160)
                plt.close(fig)
                shown += 1

            if (img_i + 1) % 50 == 0 or (img_i + 1) == cluster_n:
                print(f"Assignment pass: processed {img_i+1}/{cluster_n} images")

        order = np.argsort(-counts)

    # Cluster size plot
    num_clusters_plotted = int(counts.shape[0])
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(num_clusters_plotted), counts[order])
    title = "Cluster sizes"
    if clusterer_name == "hdbscan":
        title = f"Cluster sizes (HDBSCAN, clusters={counts.shape[0]})"
    elif clusterer_name in ("sklearn_kmeans", "bisecting_kmeans", "gmm"):
        title = f"Cluster sizes ({clusterer_name}, k={int(args.k)})"
    else:
        title = f"Cluster sizes (k={int(args.k)})"
    plt.title(title)
    plt.xlabel("clusters sorted by size")
    plt.ylabel("# small patches")
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_sizes.png", dpi=160)
    plt.close()

    # Print a simple histogram to stdout (sorted)
    print("\nCluster histogram (sorted by size):")
    top = min(50, int(counts.shape[0]))
    for rank in range(top):
        cid = int(order[rank])
        csz = int(counts[cid])
        print(f"  cluster {cid:03d}: {csz}")

    # Save a short text summary
    summary = {
        "run_id": run_id,
        "data_root": str(data_root),
        "split": args.split,
        "size_stats_n": int(size_n),
        "cluster_num_images": int(cluster_n),
        "viz_num_images": int(viz_n),
        "max_edge": args.max_edge,
        "pad_size": args.pad_size,
        "large_size": args.large_size,
        "large_stride": args.large_stride,
        "small_size": args.small_size,
        "small_stride": args.small_stride,
        "embedder": embedder.name,
        "device": args.device,
        "embedding_dim": embedder.embedding_dim,
        "clusterer": clusterer_name,
        "k": int(args.k),
        "kmeans_iters": int(args.kmeans_iters),
        "pca_dim": int(args.pca_dim),
        "hdbscan_min_cluster_size": int(args.hdbscan_min_cluster_size),
        "hdbscan_min_samples": int(args.hdbscan_min_samples),
        "assign_noise_to_nearest_centroid": bool(args.assign_noise_to_nearest_centroid),
        "small_patches_per_image": int(args.small_patches_per_image),
        "cluster_sizes": counts.tolist(),
        "evaluation": eval_report,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved outputs to: {out_dir}")
    print(f"Size distribution plot: {out_dir / 'image_size_distribution.png'}")
    print(f"Cluster size plot: {out_dir / 'cluster_sizes.png'}")
    print(f"Cluster overlay images (viz): {out_dir}/image_*_clusters.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
