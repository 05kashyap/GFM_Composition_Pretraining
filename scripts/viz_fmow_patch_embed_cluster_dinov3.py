#!/usr/bin/env python3
"""Visualize an FMoW patch->embed->cluster preprocessing workflow (DINOv3-SAT local weights).

This is a standalone script and does NOT touch the existing DynamicVis pipeline.

Key differences vs the non-DINOv3 script:
- Uses local DINOv3 ViT-L/16 .pth weights from disk (no HuggingFace required).
- Adds an on-disk embedding cache so patch embeddings don't need recomputation.

Workflow:
0) (Optional) Sample N images and plot width/height distribution.
1) Load local FMoW images from data/fmow/<split>/...
2) Resize so max(width,height) <= --max-edge, then pad to --pad-size x --pad-size.
3) Divide each padded image into a SMALL-patch grid (default 64x64).
4) Embed selected small patches with DINOv3 weights.
5) Cluster embeddings (HDBSCAN or fixed-k clusterers).
6) Visualize intermediate steps and plot cluster sizes.

Outputs are saved under: outputs/preprocess_viz_dinov3/<run_id>/
Cache is stored under: outputs/preprocess_cache_dinov3/ (configurable)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# FMoW contains some extremely large satellite images.
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
    x0: int
    y0: int
    x1: int
    y1: int


def _find_jpgs(data_root: Path, split: str) -> List[Path]:
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    paths = list(split_dir.rglob("*.jpg")) + list(split_dir.rglob("*.jpeg"))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise RuntimeError(f"No jpg images found under {split_dir}")

    return sorted(paths)


def _sample_stratified(all_paths: List[Path], split_dir: Path, n: int, rng: random.Random) -> List[Path]:
    class_to_paths: Dict[str, List[Path]] = defaultdict(list)
    for p in all_paths:
        try:
            rel = p.relative_to(split_dir)
            class_to_paths[rel.parts[0]].append(p)
        except Exception:
            continue

    if not class_to_paths:
        print("WARNING: Could not determine classes; falling back to random sampling")
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
    sampled = sampled[: min(n, len(sampled))]
    print(
        f"Stratified sampling: {len(sampled)} images from {num_classes} classes (~{per_class} per class)"
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
    w, h = img.size
    if w > size or h > size:
        raise ValueError(f"Image {w}x{h} larger than pad {size}; reduce via --max-edge")
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
    img = _pad_to_square(img, size=pad_size, fill=(0, 0, 0))
    return img


def _extract_grid_patches(w: int, h: int, patch: int, stride: int) -> List[Patch]:
    if patch <= 0 or stride <= 0:
        raise ValueError("patch and stride must be positive")
    out: List[Patch] = []
    for y0 in range(0, h - patch + 1, stride):
        for x0 in range(0, w - patch + 1, stride):
            out.append(Patch(x0=x0, y0=y0, x1=x0 + patch, y1=y0 + patch))
    return out


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
        raise NotImplementedError


def _extract_checkpoint_state_dict(ckpt: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "teacher", "student", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if ckpt and all(isinstance(k, str) for k in ckpt.keys()):
            if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
                return ckpt  # type: ignore[return-value]
    raise RuntimeError("Could not locate state_dict in checkpoint")


def _strip_known_prefixes(k: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "backbone.",
        "teacher.",
        "student.",
        "encoder.",
        "visual.",
        "vit.",
    )
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p) :]
                changed = True
    return k


class DinoV3SatViTL16Embedder(Embedder):
    """Local DINOv3 ViT-L/16 embedder backed by torchvision ViT.

    This loads the provided .pth into a torchvision ViT-L/16 skeleton.
    Embeddings are the pre-head outputs (heads replaced with Identity).

    Inputs are resized to 224x224 and normalized with ImageNet stats.
    """

    def __init__(self, device: str, weights_path: str):
        super().__init__(device=device)
        self.weights_path = str(weights_path)

        try:
            from torchvision.models import vit_l_16
            import torchvision.transforms as T
            from torchvision.transforms import InterpolationMode
        except Exception as e:
            raise RuntimeError(f"torchvision is required. Root error: {e}")

        model = vit_l_16(weights=None)
        model.heads = torch.nn.Identity()
        self.model = model.eval().to(self.device)

        self.transforms = T.Compose(
            [
                T.Resize(224, interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        wpath = Path(self.weights_path)
        if not wpath.exists():
            raise FileNotFoundError(f"Weights not found: {wpath}")

        ckpt = torch.load(str(wpath), map_location="cpu")
        sd_raw = _extract_checkpoint_state_dict(ckpt)
        sd = {_strip_known_prefixes(k): v for k, v in sd_raw.items() if isinstance(v, torch.Tensor)}

        incompatible = self.model.load_state_dict(sd, strict=False)
        try:
            print(
                f"Loaded DINOv3 SAT weights: {wpath.name} | "
                f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
            )
        except Exception:
            pass

        self._embedding_dim = int(getattr(self.model, "hidden_dim", 1024))

    @property
    def name(self) -> str:
        return f"dinov3_sat:{Path(self.weights_path).name}"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @torch.inference_mode()
    def embed_pil(self, images: Sequence[Image.Image], batch_size: int) -> torch.Tensor:
        all_embs: List[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            x = torch.stack([self.transforms(im) for im in batch], dim=0).to(self.device)
            embs = self.model(x).detach().cpu()
            all_embs.append(embs)
        return torch.cat(all_embs, dim=0)


def _l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _fit_pca(x: np.ndarray, pca_dim: int, seed: int):
    try:
        from sklearn.decomposition import PCA
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for PCA. Install with: pip install scikit-learn\n" + str(e)
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
            "hdbscan is required for HDBSCAN. Install with: pip install hdbscan\n" + str(e)
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


def _fit_sklearn_kmeans(x: np.ndarray, k: int, seed: int, batch_size: int = 1024):
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for sklearn_kmeans. Install with: pip install scikit-learn\n" + str(e)
        )

    km = MiniBatchKMeans(
        n_clusters=int(k),
        random_state=int(seed),
        batch_size=int(batch_size),
        n_init=3,
        max_iter=300,
    )
    labels = km.fit_predict(x)
    return km, labels


def _fit_bisecting_kmeans(x: np.ndarray, k: int, seed: int):
    try:
        from sklearn.cluster import BisectingKMeans
    except Exception as e:
        raise RuntimeError(
            "scikit-learn >= 1.1 required for bisecting_kmeans.\n" + str(e)
        )

    bkm = BisectingKMeans(
        n_clusters=int(k),
        random_state=int(seed),
        n_init=3,
        max_iter=300,
    )
    labels = bkm.fit_predict(x)
    return bkm, labels


def _fit_gmm(x: np.ndarray, k: int, seed: int):
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for gmm. Install with: pip install scikit-learn\n" + str(e)
        )

    gmm = GaussianMixture(
        n_components=int(k),
        covariance_type="diag",
        random_state=int(seed),
        n_init=3,
        max_iter=300,
    )
    labels = gmm.fit_predict(x)
    return gmm, labels


def _silhouette_optional(x: np.ndarray, labels: np.ndarray, seed: int, sample_size: int = 20000) -> Optional[float]:
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


def _colorize_small_patch_grid(small_patches: List[Patch], labels: np.ndarray, grid_w: int, grid_h: int) -> np.ndarray:
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


def _save_viz_embeddings(out_dir: Path, shown_idx: int, image_path: Path, patches: List[Patch], emb_raw: np.ndarray, labels: np.ndarray, emb_pca: Optional[np.ndarray] = None) -> None:
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


def _weights_fingerprint(weights_path: str) -> Tuple[Optional[float], Optional[int]]:
    if not weights_path:
        return None, None
    try:
        st = os.stat(weights_path)
        return float(st.st_mtime), int(st.st_size)
    except Exception:
        return None, None


def _patches_fingerprint(patches: List[Patch]) -> str:
    arr = np.asarray([[p.x0, p.y0, p.x1, p.y1] for p in patches], dtype=np.int32)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _cache_key(*, image_path: Path, embedder_name: str, max_edge: int, pad_size: int, weights_path: str, patches_fp: str) -> str:
    mtime, fsize = _weights_fingerprint(weights_path)
    payload = {
        "image": str(image_path),
        "embedder": str(embedder_name),
        "max_edge": int(max_edge),
        "pad_size": int(pad_size),
        "weights_path": str(weights_path),
        "weights_mtime": mtime,
        "weights_size": fsize,
        "patches_fp": patches_fp,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _get_or_compute_embeddings_cached(
    *,
    cache_dir: Path,
    image_path: Path,
    embedder: Embedder,
    max_edge: int,
    pad_size: int,
    patches: List[Patch],
    patch_images: List[Image.Image],
    embed_batch: int,
    weights_path: str,
    cache: bool,
) -> np.ndarray:
    if not cache:
        return embedder.embed_pil(patch_images, batch_size=embed_batch).numpy().astype(np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    patches_fp = _patches_fingerprint(patches)
    key = _cache_key(
        image_path=image_path,
        embedder_name=embedder.name,
        max_edge=max_edge,
        pad_size=pad_size,
        weights_path=weights_path,
        patches_fp=patches_fp,
    )
    fpath = cache_dir / f"{key}.npz"

    if fpath.exists():
        try:
            data = np.load(fpath, allow_pickle=False)
            emb = data["emb"].astype(np.float32, copy=False)
            return emb
        except Exception:
            pass

    emb = embedder.embed_pil(patch_images, batch_size=embed_batch).numpy().astype(np.float32)
    np.savez_compressed(
        fpath,
        emb=emb.astype(np.float16, copy=False),
        image_path=str(image_path),
    )
    return emb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/fmow")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--size-stats-n", type=int, default=1000)

    parser.add_argument("--cluster-num-images", type=int, default=1000)
    parser.add_argument("--viz-num-images", type=int, default=25)
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument("--pad-size", type=int, default=1024)

    parser.add_argument("--large-size", type=int, default=512)
    parser.add_argument("--large-stride", type=int, default=512)
    parser.add_argument("--small-size", type=int, default=128)
    parser.add_argument("--small-stride", type=int, default=128)

    parser.add_argument("--weights-path", type=str, default="weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed-batch", type=int, default=64)

    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--cache-embeddings", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="outputs/preprocess_cache_dinov3")

    parser.add_argument(
        "--clusterer",
        type=str,
        default="sklearn_kmeans",
        choices=["sklearn_kmeans", "bisecting_kmeans", "gmm", "hdbscan", "stream_kmeans"],
    )

    parser.add_argument("--fit-small-patches-per-image", type=int, default=64)
    parser.add_argument("--pca-dim", type=int, default=128)

    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=15)
    parser.add_argument("--hdbscan-min-samples", type=int, default=0)
    parser.add_argument("--hdbscan-jobs", type=int, default=8)
    parser.add_argument("--assign-noise-to-nearest-centroid", action="store_true")

    parser.add_argument("--k", type=int, default=60)

    args = parser.parse_args()

    _seed_all(args.seed)

    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    all_paths = _find_jpgs(data_root, args.split)
    print(f"Total images found: {len(all_paths)}")

    rng = random.Random(args.seed)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / "preprocess_viz_dinov3" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available():
        print(f"Device: cuda (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device.type}")

    # Stage 0: size distribution
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
        print("WARNING: Could not compute size distribution")

    # Sample images for clustering/viz
    cluster_n = min(int(args.cluster_num_images), len(all_paths))
    cluster_paths = _sample_stratified(all_paths, split_dir, cluster_n, rng)
    viz_n = min(int(args.viz_num_images), len(cluster_paths))
    viz_set = set(cluster_paths[:viz_n])

    # Embedder
    embedder = DinoV3SatViTL16Embedder(device=args.device, weights_path=args.weights_path)

    def iter_small_patches_for_image(img: Image.Image) -> Tuple[List[Patch], List[Image.Image]]:
        pw, ph = img.size
        if pw != args.pad_size or ph != args.pad_size:
            raise RuntimeError("Expected padded image")
        small_patches_full = _extract_grid_patches(pw, ph, patch=args.small_size, stride=args.small_stride)
        small_imgs = [_crop(img, sp) for sp in small_patches_full]
        return small_patches_full, small_imgs

    clusterer_name = str(args.clusterer)
    shown = 0
    eval_report: Dict[str, object] = {}

    # ---------------------------------------------------------------
    # Clustering
    # ---------------------------------------------------------------
    if clusterer_name == "hdbscan":
        fit_per_image = max(1, int(args.fit_small_patches_per_image))
        fit_embs: List[np.ndarray] = []
        t0 = time.time()

        for i, p in enumerate(cluster_paths):
            img = _load_and_standardize_image(p, max_edge=args.max_edge, pad_size=args.pad_size)
            small_patches_full, small_imgs = iter_small_patches_for_image(img)

            if fit_per_image < len(small_imgs):
                pick = rng.sample(range(len(small_imgs)), k=fit_per_image)
                patches_fit = [small_patches_full[j] for j in pick]
                imgs_fit = [small_imgs[j] for j in pick]
            else:
                patches_fit = small_patches_full
                imgs_fit = small_imgs

            embs = _get_or_compute_embeddings_cached(
                cache_dir=cache_dir,
                image_path=p,
                embedder=embedder,
                max_edge=args.max_edge,
                pad_size=args.pad_size,
                patches=patches_fit,
                patch_images=imgs_fit,
                embed_batch=args.embed_batch,
                weights_path=args.weights_path,
                cache=bool(args.cache_embeddings),
            )
            fit_embs.append(embs)

            if (i + 1) % 50 == 0 or (i + 1) == cluster_n:
                dt = time.time() - t0
                ips = (i + 1) / max(dt, 1e-6)
                print(f"Embedding (fit set): {i+1}/{cluster_n} images | {ips:.2f} img/s")

        X = np.concatenate(fit_embs, axis=0)
        X = _l2_normalize_np(X)
        print(f"Fit embedding matrix: {X.shape} (N,D)")

        pca, X_pca = _fit_pca(X, pca_dim=int(args.pca_dim), seed=args.seed)
        X_pca = _l2_normalize_np(X_pca.astype(np.float32))

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
            centroids = _l2_normalize_np(centroids)

        try:
            import hdbscan
        except Exception:
            hdbscan = None

        for p in cluster_paths:
            if p not in viz_set or shown >= viz_n:
                continue

            img = _load_and_standardize_image(p, max_edge=args.max_edge, pad_size=args.pad_size)
            small_patches_full, small_imgs = iter_small_patches_for_image(img)

            embs_all = _get_or_compute_embeddings_cached(
                cache_dir=cache_dir,
                image_path=p,
                embedder=embedder,
                max_edge=args.max_edge,
                pad_size=args.pad_size,
                patches=small_patches_full,
                patch_images=small_imgs,
                embed_batch=args.embed_batch,
                weights_path=args.weights_path,
                cache=bool(args.cache_embeddings),
            )
            embs_all = _l2_normalize_np(embs_all)
            embs_pca = pca.transform(embs_all).astype(np.float32)
            embs_pca = _l2_normalize_np(embs_pca)

            if hdbscan is not None and valid.size != 0:
                pred_labels, _strength = hdbscan.approximate_predict(hdb, embs_pca)
            else:
                pred_labels = np.full((embs_pca.shape[0],), -1, dtype=np.int64)

            if args.assign_noise_to_nearest_centroid and valid.size != 0:
                noise_mask = pred_labels < 0
                if np.any(noise_mask):
                    sims = embs_pca[noise_mask] @ centroids.T
                    nn = np.argmax(sims, axis=1)
                    pred_labels[noise_mask] = nn.astype(np.int64)

            if args.save_embeddings:
                _save_viz_embeddings(out_dir, shown, p, small_patches_full, embs_all, pred_labels, emb_pca=embs_pca)

            fig, axs = plt.subplots(1, 3, figsize=(16, 6))
            axs[0].imshow(img)
            axs[0].set_title("Standardized image (resized + padded)")
            axs[0].axis("off")

            axs[1].imshow(img)
            pw, ph = img.size
            _overlay_grid(axs[1], w=pw, h=ph, patch=args.large_size, stride=args.large_stride, color="lime", lw=1.0)
            axs[1].set_title(f"Large grid {args.large_size}x{args.large_size}")
            axs[1].axis("off")

            axs[2].imshow(img)
            overlay = _colorize_small_patch_grid(small_patches_full, pred_labels.astype(np.int64), grid_w=pw, grid_h=ph)
            axs[2].imshow(overlay)
            axs[2].set_title("Small patches colored by cluster")
            axs[2].axis("off")

            fig.suptitle(f"{p} | embedder={embedder.name} | clusters={counts.shape[0]}", fontsize=11)
            fig.tight_layout()
            fig.savefig(out_dir / f"image_{shown:02d}_clusters.png", dpi=160)
            plt.close(fig)
            shown += 1

    else:
        # Fixed-k clustering (and stream_kmeans fallback)
        fit_per_image = max(1, int(args.fit_small_patches_per_image))
        fit_embs: List[np.ndarray] = []
        t0 = time.time()

        for i, p in enumerate(cluster_paths):
            img = _load_and_standardize_image(p, max_edge=args.max_edge, pad_size=args.pad_size)
            small_patches_full, small_imgs = iter_small_patches_for_image(img)

            if fit_per_image < len(small_imgs):
                pick = rng.sample(range(len(small_imgs)), k=fit_per_image)
                patches_fit = [small_patches_full[j] for j in pick]
                imgs_fit = [small_imgs[j] for j in pick]
            else:
                patches_fit = small_patches_full
                imgs_fit = small_imgs

            embs = _get_or_compute_embeddings_cached(
                cache_dir=cache_dir,
                image_path=p,
                embedder=embedder,
                max_edge=args.max_edge,
                pad_size=args.pad_size,
                patches=patches_fit,
                patch_images=imgs_fit,
                embed_batch=args.embed_batch,
                weights_path=args.weights_path,
                cache=bool(args.cache_embeddings),
            )
            fit_embs.append(embs)

            if (i + 1) % 50 == 0 or (i + 1) == cluster_n:
                dt = time.time() - t0
                ips = (i + 1) / max(dt, 1e-6)
                print(f"Embedding (fit set): {i+1}/{cluster_n} images | {ips:.2f} img/s")

        X = np.concatenate(fit_embs, axis=0)
        X = _l2_normalize_np(X)
        print(f"Fit embedding matrix: {X.shape} (N,D)")

        pca, X_pca = _fit_pca(X, pca_dim=int(args.pca_dim), seed=args.seed)
        X_pca = _l2_normalize_np(X_pca.astype(np.float32))

        # ---- save full embedding matrix for CPU-only clustering sweeps ----
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
            pad_size=int(args.pad_size),
            small_size=int(args.small_size),
            small_stride=int(args.small_stride),
            max_edge=int(args.max_edge),
        )
        print(f"Saved embedding data for CPU sweep: {sweep_path} ({X_pca.shape})")
        # ---- also save PCA model ----
        import pickle
        with open(out_dir / "pca_model.pkl", "wb") as f:
            pickle.dump(pca, f)
        print(f"Saved PCA model: {out_dir / 'pca_model.pkl'}")
        # ----------------------------------------------------------------

        k = int(args.k)
        if clusterer_name == "sklearn_kmeans" or clusterer_name == "stream_kmeans":
            model, fit_labels = _fit_sklearn_kmeans(X_pca, k=k, seed=args.seed)
        elif clusterer_name == "bisecting_kmeans":
            model, fit_labels = _fit_bisecting_kmeans(X_pca, k=k, seed=args.seed)
        elif clusterer_name == "gmm":
            model, fit_labels = _fit_gmm(X_pca, k=k, seed=args.seed)
        else:
            raise RuntimeError(f"Unknown clusterer: {clusterer_name}")

        counts = np.bincount(fit_labels.astype(np.int64), minlength=k)
        order = np.argsort(-counts)
        sil = _silhouette_optional(X_pca, fit_labels, seed=args.seed)
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

            img = _load_and_standardize_image(p, max_edge=args.max_edge, pad_size=args.pad_size)
            small_patches_full, small_imgs = iter_small_patches_for_image(img)

            embs_all = _get_or_compute_embeddings_cached(
                cache_dir=cache_dir,
                image_path=p,
                embedder=embedder,
                max_edge=args.max_edge,
                pad_size=args.pad_size,
                patches=small_patches_full,
                patch_images=small_imgs,
                embed_batch=args.embed_batch,
                weights_path=args.weights_path,
                cache=bool(args.cache_embeddings),
            )
            embs_all = _l2_normalize_np(embs_all)
            embs_pca = pca.transform(embs_all).astype(np.float32)
            embs_pca = _l2_normalize_np(embs_pca)
            pred_labels = model.predict(embs_pca)

            if args.save_embeddings:
                _save_viz_embeddings(out_dir, shown, p, small_patches_full, embs_all, pred_labels, emb_pca=embs_pca)

            fig, axs = plt.subplots(1, 3, figsize=(16, 6))
            axs[0].imshow(img)
            axs[0].set_title("Standardized image (resized + padded)")
            axs[0].axis("off")

            axs[1].imshow(img)
            pw, ph = img.size
            _overlay_grid(axs[1], w=pw, h=ph, patch=args.large_size, stride=args.large_stride, color="lime", lw=1.0)
            axs[1].set_title(f"Large grid {args.large_size}x{args.large_size}")
            axs[1].axis("off")

            axs[2].imshow(img)
            overlay = _colorize_small_patch_grid(small_patches_full, pred_labels.astype(np.int64), grid_w=pw, grid_h=ph)
            axs[2].imshow(overlay)
            axs[2].set_title("Small patches colored by cluster")
            axs[2].axis("off")

            fig.suptitle(f"{p} | embedder={embedder.name} | {clusterer_name} k={k}", fontsize=11)
            fig.tight_layout()
            fig.savefig(out_dir / f"image_{shown:02d}_clusters.png", dpi=160)
            plt.close(fig)
            shown += 1

    # Cluster size plot
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

    summary = {
        "run_id": run_id,
        "data_root": str(data_root),
        "split": args.split,
        "cluster_num_images": int(cluster_n),
        "viz_num_images": int(viz_n),
        "max_edge": int(args.max_edge),
        "pad_size": int(args.pad_size),
        "small_size": int(args.small_size),
        "small_stride": int(args.small_stride),
        "weights_path": str(args.weights_path),
        "embedder": embedder.name,
        "device": args.device,
        "embedding_dim": int(embedder.embedding_dim),
        "clusterer": clusterer_name,
        "k": int(args.k),
        "pca_dim": int(args.pca_dim),
        "cluster_sizes": counts.tolist(),
        "evaluation": eval_report,
        "cache": {
            "enabled": bool(args.cache_embeddings),
            "cache_dir": str(cache_dir),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved outputs to: {out_dir}")
    print(f"Cache dir: {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
