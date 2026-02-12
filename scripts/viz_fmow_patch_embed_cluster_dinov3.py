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
3) Divide each padded image into overlapping patches via a sliding window
   (default 128x128 with 50% overlap, stride 64). Independent horizontal
   and vertical strides are supported (--small-stride-x / --small-stride-y).
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
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
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

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


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
    """Legacy grid extraction (uniform stride). Prefer _extract_sliding_window_patches."""
    return _extract_sliding_window_patches(w, h, patch_w=patch, patch_h=patch,
                                           stride_x=stride, stride_y=stride)


def _extract_sliding_window_patches(
    w: int, h: int,
    patch_w: int, patch_h: int,
    stride_x: int, stride_y: int,
) -> List[Patch]:
    """Sliding-window patch extraction with independent horizontal/vertical strides.

    Args:
        w, h: Image dimensions.
        patch_w, patch_h: Window (patch) width and height.
        stride_x: Horizontal stride (step between columns).
        stride_y: Vertical stride (step between rows).

    Returns:
        List of Patch objects covering the image. Only full patches are generated
        (no partial patches at the right/bottom edges).
    """
    if patch_w <= 0 or patch_h <= 0:
        raise ValueError("patch_w and patch_h must be positive")
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError("stride_x and stride_y must be positive")
    out: List[Patch] = []
    for y0 in range(0, h - patch_h + 1, stride_y):
        for x0 in range(0, w - patch_w + 1, stride_x):
            out.append(Patch(x0=x0, y0=y0, x1=x0 + patch_w, y1=y0 + patch_h))
    return out


def _crop(img: Image.Image, patch: Patch) -> Image.Image:
    return img.crop((patch.x0, patch.y0, patch.x1, patch.y1))


def _overlay_grid(ax, w: int, h: int, patch: int, stride: int, color: str, lw: float = 1.0,
                   stride_x: Optional[int] = None, stride_y: Optional[int] = None):
    """Draw grid lines. If stride_x/stride_y are given they override *stride*."""
    sx = stride_x if stride_x is not None else stride
    sy = stride_y if stride_y is not None else stride
    for y in range(0, h + 1, sy):
        ax.plot([0, w], [y, y], color=color, linewidth=lw)
    for x in range(0, w + 1, sx):
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


# ---------------------------------------------------------------
# Native DINOv3 ViT-L/16 model definition (matches checkpoint keys)
# ---------------------------------------------------------------

class _RoPE2D(torch.nn.Module):
    """2-D Rotary Position Embedding for vision transformers."""

    def __init__(self, dim: int, n_periods: int = 16, max_res: int = 256):
        super().__init__()
        # The checkpoint stores learned periods of shape (n_periods,)
        self.register_buffer("periods", torch.arange(n_periods).float(), persistent=True)
        self.max_res = max_res
        self._dim = dim

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """x: (B, N, D) where N = h*w (no cls/register tokens)."""
        device = x.device
        dtype = x.dtype
        half = self._dim // 4

        freqs = 1.0 / (10000.0 ** (self.periods / half))  # (half,)
        gy = torch.arange(h, device=device, dtype=dtype)
        gx = torch.arange(w, device=device, dtype=dtype)
        fy = torch.outer(gy, freqs)  # (h, half)
        fx = torch.outer(gx, freqs)  # (w, half)

        fy = fy[:, None, :].expand(h, w, half).reshape(h * w, half)
        fx = fx[None, :, :].expand(h, w, half).reshape(h * w, half)
        angles = torch.cat([fy, fy, fx, fx], dim=-1)  # (h*w, dim)

        cos_ = torch.cos(angles).unsqueeze(0)
        sin_ = torch.sin(angles).unsqueeze(0)

        x1 = x[..., : self._dim]
        x2 = x[..., self._dim :]

        x1_rot = torch.stack([-x1[..., half : 2 * half],
                               x1[..., : half],
                               -x1[..., 3 * half :],
                               x1[..., 2 * half : 3 * half]], dim=-1).reshape_as(x1)
        x1 = x1 * cos_ + x1_rot * sin_
        return torch.cat([x1, x2], dim=-1)


class _Attention(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int = 16, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        # bias_mask is stored in the checkpoint; register as buffer
        self.register_buffer("qkv_bias_mask", torch.ones(dim * 3), persistent=False)
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class _Mlp(torch.nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = torch.nn.Linear(dim, hidden)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _LayerScale(torch.nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = torch.nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class _Block(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int = 16, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads=num_heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, mlp_ratio=mlp_ratio)
        self.ls2 = _LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class _PatchEmbed(torch.nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)


class DinoV3SatViTL16(torch.nn.Module):
    """Native DINOv3/DINOv2-style ViT-L/16 that matches the checkpoint exactly.

    Architecture: ViT-L/16, dim=1024, depth=24, heads=16, mlp_ratio=4
    Features: RoPE, layer scale, 4 register (storage) tokens, cls token.
    """

    def __init__(self, patch_size: int = 16, embed_dim: int = 1024,
                 depth: int = 24, num_heads: int = 16, mlp_ratio: float = 4.0,
                 num_register_tokens: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.storage_tokens = torch.nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        self.mask_token = torch.nn.Parameter(torch.zeros(1, embed_dim))
        self.patch_embed = _PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        self.rope_embed = _RoPE2D(dim=embed_dim)
        self.blocks = torch.nn.ModuleList([
            _Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.local_cls_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) -> (B, embed_dim) cls token output."""
        B = x.shape[0]
        # Patch embed
        tokens = self.patch_embed(x)  # (B, N, D)
        h = x.shape[2] // self.patch_size
        w = x.shape[3] // self.patch_size

        # Prepend cls + register tokens
        cls = self.cls_token.expand(B, -1, -1)
        reg = self.storage_tokens.expand(B, -1, -1)
        tokens = torch.cat([cls, reg, tokens], dim=1)  # (B, 1+R+N, D)

        # Forward through blocks
        for blk in self.blocks:
            tokens = blk(tokens)

        # Norm and return cls token
        tokens = self.norm(tokens)
        return tokens[:, 0]  # cls token


class DinoV3SatViTL16Embedder(Embedder):
    """Local DINOv3 ViT-L/16 embedder using native architecture matching checkpoint.

    Loads the .pth into a custom ViT-L/16 that matches the DINOv3 checkpoint
    structure exactly (fused QKV, layer scale, RoPE, register tokens).

    Inputs are resized to 224x224 and normalized with ImageNet stats.
    """

    def __init__(self, device: str, weights_path: str):
        super().__init__(device=device)
        self.weights_path = str(weights_path)

        import torchvision.transforms as T
        from torchvision.transforms import InterpolationMode

        model = DinoV3SatViTL16(
            patch_size=16, embed_dim=1024, depth=24,
            num_heads=16, mlp_ratio=4.0, num_register_tokens=4,
        )
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

        ckpt = torch.load(str(wpath), map_location="cpu", weights_only=False)
        sd_raw = _extract_checkpoint_state_dict(ckpt)
        sd = {_strip_known_prefixes(k): v for k, v in sd_raw.items() if isinstance(v, torch.Tensor)}

        # Map checkpoint key "attn.qkv.bias_mask" -> registered buffer name
        mapped_sd: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            mapped_sd[k] = v

        # Load with strict=False since qkv.bias_mask is stored under a dotted
        # name in the checkpoint ("attn.qkv.bias_mask") which doesn't map 1:1
        # to the buffer we register ("attn.qkv_bias_mask").  These are just
        # binary masks, not trainable weights, so skipping them is safe.
        incompatible = self.model.load_state_dict(mapped_sd, strict=False)
        n_missing = len(incompatible.missing_keys)
        # Filter out qkv_bias_mask from unexpected (benign)
        real_unexpected = [k for k in incompatible.unexpected_keys if "bias_mask" not in k]
        n_unexpected = len(real_unexpected)
        print(
            f"Loaded DINOv3 SAT weights: {wpath.name} | "
            f"missing={n_missing} unexpected={n_unexpected}"
            f"{' (+ 24 qkv_bias_mask buffers skipped)' if len(incompatible.unexpected_keys) > n_unexpected else ''}"
        )
        if n_missing > 0:
            print(f"  Missing keys: {incompatible.missing_keys[:10]}")
        if n_unexpected > 0:
            print(f"  Unexpected keys: {real_unexpected[:10]}")
        if n_missing > 0 or n_unexpected > 0:
            print("  WARNING: significant key mismatch — embeddings may be incorrect!")

        self._embedding_dim = 1024

    @property
    def name(self) -> str:
        return f"dinov3_sat:{Path(self.weights_path).name}"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    # ImageNet normalization constants (as tensors for fast GPU normalize)
    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.inference_mode()
    def embed_pil(self, images: Sequence[Image.Image], batch_size: int) -> torch.Tensor:
        all_embs: List[torch.Tensor] = []
        use_amp = self.device.type == "cuda"
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            x = torch.stack([self.transforms(im) for im in batch], dim=0)
            if self.device.type == "cuda":
                x = x.pin_memory()
            x = x.to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                embs = self.model(x)
            all_embs.append(embs.float().cpu())
        return torch.cat(all_embs, dim=0)

    @torch.inference_mode()
    def embed_tensor_patches(
        self,
        img_tensor: torch.Tensor,
        patches: List["Patch"],
        batch_size: int,
    ) -> torch.Tensor:
        """Embed patches from a pre-converted image tensor (fast path).

        Instead of 225 individual PIL.crop + PIL→tensor + resize transforms,
        this converts the full image to a tensor ONCE, uses tensor slicing for
        crops, and runs gpu-batched resize + normalize.

        Optimizations: pre-allocated crop buffer, pinned memory, fp16 autocast.

        Args:
            img_tensor: (3, H, W) float32 tensor in [0,1], already on CPU.
            patches: list of Patch objects (pixel coords within img_tensor).
            batch_size: patches per forward pass through the ViT.

        Returns:
            (N, D) embedding tensor on CPU.
        """
        mean = self._IMAGENET_MEAN.to(self.device)
        std = self._IMAGENET_STD.to(self.device)
        use_amp = self.device.type == "cuda"

        all_embs: List[torch.Tensor] = []
        for i in range(0, len(patches), batch_size):
            chunk = patches[i : i + batch_size]
            # Pre-allocate crop buffer instead of torch.stack
            patch_h = chunk[0].y1 - chunk[0].y0
            patch_w = chunk[0].x1 - chunk[0].x0
            crops = torch.empty(len(chunk), 3, patch_h, patch_w, dtype=torch.float32)
            for j, p in enumerate(chunk):
                crops[j] = img_tensor[:, p.y0:p.y1, p.x0:p.x1]
            # Pin memory for async CPU→GPU transfer
            if self.device.type == "cuda":
                crops = crops.pin_memory()
            crops = crops.to(self.device, non_blocking=True)
            # Batched GPU resize to 224x224
            crops = F.interpolate(crops, size=(224, 224), mode="bicubic", align_corners=False)
            # Normalize
            crops = (crops - mean) / std
            with torch.cuda.amp.autocast(enabled=use_amp):
                embs = self.model(crops)
            all_embs.append(embs.float().cpu())
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
    """Colorize patches by cluster label with alpha-blending for overlapping regions."""
    n_labels = max(int(labels.max()) + 1, 1) if labels.size > 0 else 1
    if n_labels <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("nipy_spectral", n_labels)

    # Accumulate colours with a count buffer so overlapping windows blend.
    color_acc = np.zeros((grid_h, grid_w, 3), dtype=np.float64)
    count = np.zeros((grid_h, grid_w), dtype=np.float64)

    for p, lab in zip(small_patches, labels):
        if n_labels <= 20:
            r, g, b, _ = cmap(int(lab) % 20)
        else:
            r, g, b, _ = cmap(int(lab) / max(n_labels - 1, 1))
        color_acc[p.y0 : p.y1, p.x0 : p.x1, 0] += r
        color_acc[p.y0 : p.y1, p.x0 : p.x1, 1] += g
        color_acc[p.y0 : p.y1, p.x0 : p.x1, 2] += b
        count[p.y0 : p.y1, p.x0 : p.x1] += 1.0

    # Average where there is coverage
    mask = count > 0
    overlay = np.zeros((grid_h, grid_w, 4), dtype=np.float32)
    for c in range(3):
        overlay[:, :, c][mask] = (color_acc[:, :, c][mask] / count[mask]).astype(np.float32)
    overlay[:, :, 3][mask] = 0.45
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


def _compute_cache_path(
    cache_dir: Path,
    image_path: Path,
    embedder_name: str,
    max_edge: int,
    pad_size: int,
    patches: List[Patch],
    weights_path: str,
) -> Path:
    """Compute the cache file path for an image's embeddings."""
    patches_fp = _patches_fingerprint(patches)
    key = _cache_key(
        image_path=image_path,
        embedder_name=embedder_name,
        max_edge=max_edge,
        pad_size=pad_size,
        weights_path=weights_path,
        patches_fp=patches_fp,
    )
    return cache_dir / f"{key}.npz"


def _load_cached_embeddings(fpath: Path) -> Optional[np.ndarray]:
    """Load cached embeddings from disk, returns None if not found or corrupted."""
    if not fpath.exists():
        return None
    try:
        data = np.load(fpath, allow_pickle=False)
        return data["emb"].astype(np.float32, copy=False)
    except Exception:
        return None


def _save_embeddings_to_cache(fpath: Path, emb: np.ndarray, image_path: Path) -> None:
    """Save embeddings to cache file (uncompressed for speed)."""
    fpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        fpath,
        emb=emb.astype(np.float16, copy=False),
        image_path=str(image_path),
    )


@dataclass
class _PendingImage:
    """Tracks an image waiting for its patches to be embedded."""
    image_path: Path
    img_tensor: torch.Tensor
    patches: List[Patch]
    cache_path: Path
    start_idx: int  # index into the global batch where this image's patches start
    num_patches: int


class MultiImageBatchedEmbedder:
    """Batches patches across multiple images for efficient GPU utilization.

    Instead of embedding one image (e.g., 225 patches) at a time, this class
    accumulates patches from multiple images until batch_size is reached,
    then runs a single forward pass and caches results per-image.

    This allows proper utilization of large batch sizes (e.g., 2048) by
    processing patches from ~9 images in one GPU forward pass.
    """

    def __init__(
        self,
        embedder: "DinoV3SatViTL16Embedder",
        cache_dir: Path,
        max_edge: int,
        pad_size: int,
        weights_path: str,
        batch_size: int,
        gpu_batch_size: int = 512,
    ):
        self.embedder = embedder
        self.cache_dir = cache_dir
        self.max_edge = max_edge
        self.pad_size = pad_size
        self.weights_path = weights_path
        self.batch_size = batch_size
        self.gpu_batch_size = gpu_batch_size

        # Accumulated patches waiting to be embedded
        self._pending_images: List[_PendingImage] = []
        self._pending_tensors: List[torch.Tensor] = []  # individual patch tensors
        self._total_pending: int = 0

        # Stats
        self.images_processed = 0
        self.patches_embedded = 0
        self.forward_passes = 0

    def add_image(
        self,
        image_path: Path,
        img_tensor: torch.Tensor,
        patches: List[Patch],
    ) -> None:
        """Add an image's patches to the buffer. May trigger embedding if buffer is full."""
        cache_path = _compute_cache_path(
            self.cache_dir,
            image_path,
            self.embedder.name,
            self.max_edge,
            self.pad_size,
            patches,
            self.weights_path,
        )

        # Skip if already cached
        if cache_path.exists():
            self.images_processed += 1
            return

        # Add to pending buffer
        pending = _PendingImage(
            image_path=image_path,
            img_tensor=img_tensor,
            patches=patches,
            cache_path=cache_path,
            start_idx=self._total_pending,
            num_patches=len(patches),
        )
        self._pending_images.append(pending)
        self._total_pending += len(patches)

        # Flush if we've accumulated enough patches
        while self._total_pending >= self.batch_size:
            self._flush_batch()

    def flush(self) -> None:
        """Process any remaining patches in the buffer."""
        while self._pending_images:
            self._flush_batch()

    def _flush_batch(self) -> None:
        """Embed one batch worth of patches and cache results."""
        if not self._pending_images:
            return

        # Collect patches up to batch_size
        patches_to_embed: List[Tuple[torch.Tensor, int, int, int, int]] = []  # (tensor slice args)
        images_in_batch: List[_PendingImage] = []
        batch_count = 0

        # Determine which images fit in this batch
        remaining_pending: List[_PendingImage] = []
        for pending in self._pending_images:
            if batch_count + pending.num_patches <= self.batch_size or batch_count == 0:
                # Include this image (always include at least one image even if > batch_size)
                images_in_batch.append(pending)
                batch_count += pending.num_patches
            else:
                remaining_pending.append(pending)
        self._pending_images = remaining_pending

        if not images_in_batch:
            return

        # Pre-allocate crop buffer to avoid torch.stack allocation overhead
        p0 = images_in_batch[0].patches[0]
        patch_h = p0.y1 - p0.y0
        patch_w = p0.x1 - p0.x0
        crops_tensor = torch.empty(batch_count, 3, patch_h, patch_w, dtype=torch.float32)
        offset = 0
        for img_info in images_in_batch:
            for p in img_info.patches:
                crops_tensor[offset] = img_info.img_tensor[:, p.y0:p.y1, p.x0:p.x1]
                offset += 1

        embeddings = self._embed_batch(crops_tensor)  # (N, D)

        # Distribute embeddings back to their source images and cache
        offset = 0
        for img_info in images_in_batch:
            n = img_info.num_patches
            img_emb = embeddings[offset : offset + n]
            _save_embeddings_to_cache(img_info.cache_path, img_emb, img_info.image_path)
            offset += n
            self.images_processed += 1

        self.patches_embedded += batch_count
        self.forward_passes += 1
        self._total_pending = sum(p.num_patches for p in self._pending_images)

    @torch.inference_mode()
    def _embed_batch(self, crops: torch.Tensor) -> np.ndarray:
        """Run embedding on a batch of crops. Returns (N, D) numpy array.

        Optimizations over the baseline:
        - fp16 autocast (~2x throughput on modern GPUs)
        - Pinned memory for async CPU→GPU transfer
        - Double-buffered CUDA streams: the next sub-batch is transferred
          while the current sub-batch is being computed
        """
        device = self.embedder.device
        mean = self.embedder._IMAGENET_MEAN.to(device)
        std = self.embedder._IMAGENET_STD.to(device)
        sub_batch_size = self.gpu_batch_size

        # Pin source tensor for async CPU→GPU transfer
        if device.type == "cuda" and not crops.is_pinned():
            crops = crops.pin_memory()

        all_embs: List[torch.Tensor] = []

        if device.type == "cuda":
            # Double-buffered CUDA streams: overlap next transfer with current compute
            compute_stream = torch.cuda.current_stream(device)
            transfer_stream = torch.cuda.Stream(device)
            n = crops.shape[0]
            starts = list(range(0, n, sub_batch_size))

            # Pre-transfer first sub-batch on the transfer stream
            with torch.cuda.stream(transfer_stream):
                first_end = min(sub_batch_size, n)
                next_batch = crops[0:first_end].to(device, non_blocking=True)

            for si, start in enumerate(starts):
                # Wait for the transfer of the current sub-batch to complete
                compute_stream.wait_stream(transfer_stream)
                batch = next_batch

                # Kick off transfer for the next sub-batch (overlaps with compute below)
                next_start = start + sub_batch_size
                if next_start < n:
                    with torch.cuda.stream(transfer_stream):
                        next_end = min(next_start + sub_batch_size, n)
                        next_batch = crops[next_start:next_end].to(device, non_blocking=True)

                # Compute: resize → normalize → model forward (fp16)
                batch = F.interpolate(batch, size=(224, 224), mode="bicubic", align_corners=False)
                batch = (batch - mean) / std
                with torch.cuda.amp.autocast(enabled=True):
                    embs = self.embedder.model(batch)
                all_embs.append(embs.float().cpu())
        else:
            # CPU fallback (no streams, no autocast)
            for i in range(0, crops.shape[0], sub_batch_size):
                batch = crops[i : i + sub_batch_size].to(device)
                batch = F.interpolate(batch, size=(224, 224), mode="bicubic", align_corners=False)
                batch = (batch - mean) / std
                embs = self.embedder.model(batch)
                all_embs.append(embs.cpu())

        return torch.cat(all_embs, dim=0).numpy().astype(np.float32)

# ---------------------------------------------------------------------------
# Threaded prefetch loader: keeps GPU fed by loading images ahead of time
# ---------------------------------------------------------------------------

class _PrefetchLoader:
    """Threaded prefetcher: pre-applies *load_fn* to upcoming items so results
    are ready when the consumer (GPU) asks for them.

    Uses a bounded sliding window of futures so at most *prefetch_count*
    loaded items are held in memory at once, preventing OOM while ensuring
    the GPU never stalls waiting for CPU-bound image loading.
    """

    def __init__(self, items, load_fn, num_workers: int = 4, prefetch_count: int = 16):
        self._items = list(items)
        self._load_fn = load_fn
        self._num_workers = num_workers
        self._prefetch_count = prefetch_count

    def __iter__(self):
        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            futures: deque = deque()
            item_iter = iter(self._items)

            # Fill initial prefetch window
            for _ in range(min(self._prefetch_count, len(self._items))):
                try:
                    item = next(item_iter)
                    futures.append(executor.submit(self._load_fn, item))
                except StopIteration:
                    break

            while futures:
                result = futures.popleft().result()
                # Refill the window
                try:
                    item = next(item_iter)
                    futures.append(executor.submit(self._load_fn, item))
                except StopIteration:
                    pass
                yield result

    def __len__(self):
        return len(self._items)


def _embed_images_batched(
    *,
    image_paths: List[Path],
    embedder: Embedder,
    max_edge: int,
    pad_size: int,
    small_size: int,
    small_stride_x: int,
    small_stride_y: int,
    embed_batch: int,
    gpu_batch_size: int,
    weights_path: str,
    cache_dir: Path,
    use_cache: bool,
    device: str,
) -> Tuple[List[np.ndarray], List[List[Patch]]]:
    """Embed patches from multiple images with cross-image batching + threaded loading.

    Optimizations over the baseline implementation:
    - Threaded image loading (4 workers, 16-item prefetch) keeps GPU fed
    - Uses tensor-based cropping & MultiImageBatchedEmbedder for cross-image batching
    - All downstream optimizations (fp16, pinned memory, CUDA streams) apply automatically

    Returns:
        Tuple of (list of embeddings per image, list of patches per image)
    """
    all_embeddings: List[Optional[np.ndarray]] = []
    all_patches: List[List[Patch]] = []
    use_tensor_fast = isinstance(embedder, DinoV3SatViTL16Embedder)

    # First pass: threaded loading + cache check
    pending_items: List[Tuple[int, Path, List[Patch], torch.Tensor]] = []

    def _load_one_image(idx_and_path):
        idx, img_path = idx_and_path
        img = _load_and_standardize_image(img_path, max_edge=max_edge, pad_size=pad_size)
        w, h = img.size
        patches = _extract_sliding_window_patches(
            w, h,
            patch_w=small_size, patch_h=small_size,
            stride_x=small_stride_x, stride_y=small_stride_y,
        )
        # Pre-convert to tensor once (avoids repeated PIL→numpy→tensor later)
        img_t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div_(255.0)
        return idx, img_path, patches, img_t

    print(f"[Batched] Loading images with threaded prefetch ({len(image_paths)} images, 4 workers)...")
    prefetcher = _PrefetchLoader(
        list(enumerate(image_paths)),
        load_fn=_load_one_image,
        num_workers=4,
        prefetch_count=16,
    )
    for idx, img_path, patches, img_t in tqdm(prefetcher, desc="Scanning images", total=len(image_paths)):
        all_patches.append(patches)

        # Check cache
        if use_cache:
            cache_path = _compute_cache_path(
                cache_dir=cache_dir,
                image_path=img_path,
                embedder_name=embedder.name,
                max_edge=max_edge,
                pad_size=pad_size,
                patches=patches,
                weights_path=weights_path,
            )
            cached = _load_cached_embeddings(cache_path)
            if cached is not None and cached.shape[0] == len(patches):
                all_embeddings.append(cached)
                continue

        # Not cached - need to compute
        pending_items.append((idx, img_path, patches, img_t))
        all_embeddings.append(None)  # Placeholder

    if not pending_items:
        print(f"[Batched] All {len(image_paths)} images found in cache!")
        return all_embeddings, all_patches

    total_pending_patches = sum(len(p[2]) for p in pending_items)
    print(f"[Batched] Need to embed {len(pending_items)} images ({total_pending_patches} total patches)")

    # Second pass: embed using cross-image batching with MultiImageBatchedEmbedder
    if use_tensor_fast:
        cache_dir_path = Path(cache_dir) if not isinstance(cache_dir, Path) else cache_dir
        batched = MultiImageBatchedEmbedder(
            embedder=embedder,
            cache_dir=cache_dir_path,
            max_edge=max_edge,
            pad_size=pad_size,
            weights_path=weights_path,
            batch_size=embed_batch,
            gpu_batch_size=gpu_batch_size,
        )
        for orig_idx, img_path, patches, img_t in tqdm(pending_items, desc="Embedding images"):
            batched.add_image(image_path=img_path, img_tensor=img_t, patches=patches)
        batched.flush()

        # Read back results from cache
        for orig_idx, img_path, patches, img_t in pending_items:
            cache_path = _compute_cache_path(
                cache_dir=cache_dir_path,
                image_path=img_path,
                embedder_name=embedder.name,
                max_edge=max_edge,
                pad_size=pad_size,
                patches=patches,
                weights_path=weights_path,
            )
            cached = _load_cached_embeddings(cache_path)
            if cached is not None:
                all_embeddings[orig_idx] = cached
            else:
                # Fallback: embed single image directly
                emb = embedder.embed_tensor_patches(img_t, patches, batch_size=gpu_batch_size)
                all_embeddings[orig_idx] = emb.numpy().astype(np.float32)
    else:
        # PIL-based fallback for non-DINOv3 embedders
        for orig_idx, img_path, patches, img_t in tqdm(pending_items, desc="Embedding images (PIL)"):
            img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            patch_imgs = [_crop(pil_img, p) for p in patches]
            emb = embedder.embed_pil(patch_imgs, batch_size=gpu_batch_size)
            emb_array = emb.numpy().astype(np.float32)

            if use_cache:
                cache_path = _compute_cache_path(
                    cache_dir=cache_dir,
                    image_path=img_path,
                    embedder_name=embedder.name,
                    max_edge=max_edge,
                    pad_size=pad_size,
                    patches=patches,
                    weights_path=weights_path,
                )
                _save_embeddings_to_cache(cache_path, emb_array, img_path)

            all_embeddings[orig_idx] = emb_array

    return all_embeddings, all_patches

def _get_or_compute_embeddings_cached(
    *,
    cache_dir: Path,
    image_path: Path,
    embedder: Embedder,
    max_edge: int,
    pad_size: int,
    patches: List[Patch],
    patch_images: Optional[List[Image.Image]] = None,
    img_tensor: Optional[torch.Tensor] = None,
    embed_batch: int,
    weights_path: str,
    cache: bool,
) -> np.ndarray:
    """Compute (or load cached) patch embeddings.

    Fast path: if *img_tensor* is provided and the embedder is
    DinoV3SatViTL16Embedder, we use tensor-based cropping and GPU-batched
    resize/normalize — avoiding 225 individual PIL crops per image.
    """
    use_tensor_fast = (
        img_tensor is not None
        and isinstance(embedder, DinoV3SatViTL16Embedder)
    )

    def _embed() -> np.ndarray:
        if use_tensor_fast:
            return embedder.embed_tensor_patches(
                img_tensor, patches, batch_size=embed_batch
            ).numpy().astype(np.float32)
        else:
            if patch_images is None:
                raise ValueError("patch_images required when tensor fast path is unavailable")
            return embedder.embed_pil(
                patch_images, batch_size=embed_batch
            ).numpy().astype(np.float32)

    if not cache:
        return _embed()

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

    emb = _embed()
    np.savez(
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
    parser.add_argument("--large-stride", type=int, default=512,
                        help="Uniform large-patch stride (overridden by --large-stride-x/y)")
    parser.add_argument("--large-stride-x", type=int, default=None,
                        help="Large-patch horizontal stride (defaults to --large-stride)")
    parser.add_argument("--large-stride-y", type=int, default=None,
                        help="Large-patch vertical stride (defaults to --large-stride)")
    parser.add_argument("--small-size", type=int, default=128)
    parser.add_argument("--small-stride", type=int, default=64,
                        help="Uniform small-patch stride (overridden by --small-stride-x/y). "
                             "Default 64 gives 50%% overlap for 128px patches.")
    parser.add_argument("--small-stride-x", type=int, default=None,
                        help="Small-patch horizontal stride (defaults to --small-stride)")
    parser.add_argument("--small-stride-y", type=int, default=None,
                        help="Small-patch vertical stride (defaults to --small-stride)")

    parser.add_argument("--weights-path", type=str, default="weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed-batch", type=int, default=2048,
                        help="Number of patches to accumulate before embedding (across multiple images). "
                             "With 128px patches + 64px stride on 1024px images (~225 patches/image), "
                             "2048 batches ~9 images per flush for good GPU utilization.")
    parser.add_argument("--gpu-batch-size", type=int, default=512,
                        help="Actual batch size for GPU forward passes (patches per ViT inference). "
                             "Increase this to use more VRAM. Default 512.")

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

    # Multi-GPU parallel embedding support
    parser.add_argument("--embed-only", action="store_true",
                        help="Embed and cache all patches, then exit (for multi-GPU parallel preprocessing)")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Worker index for parallel embedding (0-based)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel embedding workers")

    args = parser.parse_args()

    # Resolve per-axis strides (fall back to uniform --small-stride / --large-stride)
    args.small_stride_x = args.small_stride_x if args.small_stride_x is not None else args.small_stride
    args.small_stride_y = args.small_stride_y if args.small_stride_y is not None else args.small_stride
    args.large_stride_x = args.large_stride_x if args.large_stride_x is not None else args.large_stride
    args.large_stride_y = args.large_stride_y if args.large_stride_y is not None else args.large_stride

    _seed_all(args.seed)

    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    all_paths = _find_jpgs(data_root, args.split)
    print(f"Total images found: {len(all_paths)}")
    print(f"Sliding window: patch={args.small_size}x{args.small_size}  "
          f"stride_x={args.small_stride_x}  stride_y={args.small_stride_y}")

    rng = random.Random(args.seed)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / "preprocess_viz_dinov3" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available():
        print(f"Device: cuda (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device.type}")

    # Stage 0: size distribution (skip in embed-only mode)
    if not args.embed_only:
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

    # Restore RNG state for deterministic cluster_paths regardless of embed-only
    rng = random.Random(args.seed)
    # _sample_stratified for size_paths consumed some RNG; re-seed for consistency
    # Actually we need the same cluster_paths across all shards, so re-seed here.
    rng_cluster = random.Random(args.seed + 1)

    # Sample images for clustering/viz
    cluster_n = min(int(args.cluster_num_images), len(all_paths))
    cluster_paths = _sample_stratified(all_paths, split_dir, cluster_n, rng_cluster)
    viz_n = min(int(args.viz_num_images), len(cluster_paths))
    viz_set = set(cluster_paths[:viz_n])

    # Embedder
    embedder = DinoV3SatViTL16Embedder(device=args.device, weights_path=args.weights_path)

    def iter_small_patches_for_image(
        img: Image.Image,
    ) -> Tuple[List[Patch], Optional[List[Image.Image]], torch.Tensor]:
        """Return (patches, optional PIL crops, image tensor).

        The image tensor is always returned for the fast embedding path.
        PIL crops are only created when the embedder does NOT support the
        tensor fast path (non-DINOv3), to avoid CPU overhead.
        """
        pw, ph = img.size
        if pw != args.pad_size or ph != args.pad_size:
            raise RuntimeError("Expected padded image")
        small_patches_full = _extract_sliding_window_patches(
            pw, ph,
            patch_w=args.small_size, patch_h=args.small_size,
            stride_x=args.small_stride_x, stride_y=args.small_stride_y,
        )
        # Convert image to tensor once — used by the fast embedding path
        img_t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div_(255.0)
        # Only create PIL crops if the embedder doesn't support tensor patches
        if isinstance(embedder, DinoV3SatViTL16Embedder):
            small_imgs = None  # not needed — tensor slicing is used instead
        else:
            small_imgs = [_crop(img, sp) for sp in small_patches_full]
        return small_patches_full, small_imgs, img_t

    # ---------------------------------------------------------------
    # Multi-GPU embed-only mode: cache embeddings for a shard, then exit.
    # Usage: run N workers in parallel, each with --embed-only --shard-index i --num-shards N
    # Then run the full pipeline once (reads all embeddings from cache).
    # ---------------------------------------------------------------
    if args.embed_only:
        shard_paths = cluster_paths[args.shard_index::args.num_shards]
        print(f"[Shard {args.shard_index}/{args.num_shards}] "
              f"Embed-only mode: processing {len(shard_paths)}/{len(cluster_paths)} images",
              flush=True)
        print(f"[Shard {args.shard_index}] Batch size: {args.embed_batch} patches accumulated, "
              f"GPU batch size: {args.gpu_batch_size} patches per forward pass", flush=True)

        # Use multi-image batched embedder for efficient GPU utilization
        batched_embedder = MultiImageBatchedEmbedder(
            embedder=embedder,
            cache_dir=cache_dir,
            max_edge=args.max_edge,
            pad_size=args.pad_size,
            weights_path=args.weights_path,
            batch_size=args.embed_batch,
            gpu_batch_size=args.gpu_batch_size,
        )

        # Threaded prefetch: load+preprocess images in a thread pool while
        # the main thread feeds patches to the GPU.  This eliminates the
        # "GPU idle while CPU loads next image" bottleneck.
        def _load_and_prepare_image(p: Path):
            """Load + standardize + extract patches (runs in thread pool)."""
            img = _load_and_standardize_image(p, max_edge=args.max_edge, pad_size=args.pad_size)
            small_patches_full, _small_imgs, img_t = iter_small_patches_for_image(img)
            return p, small_patches_full, img_t

        prefetcher = _PrefetchLoader(
            shard_paths,
            load_fn=_load_and_prepare_image,
            num_workers=4,
            prefetch_count=16,
        )
        for p, small_patches_full, img_t in tqdm(
            prefetcher,
            desc=f"[Shard {args.shard_index}] Embedding images",
            unit="img",
            dynamic_ncols=True,
            total=len(shard_paths),
        ):
            batched_embedder.add_image(
                image_path=p,
                img_tensor=img_t,
                patches=small_patches_full,
            )

        # Flush any remaining patches
        batched_embedder.flush()

        print(f"[Shard {args.shard_index}] Embedding complete. "
              f"Images: {batched_embedder.images_processed}, "
              f"Patches: {batched_embedder.patches_embedded}, "
              f"Forward passes: {batched_embedder.forward_passes}", flush=True)
        print(f"[Shard {args.shard_index}] Cache dir: {cache_dir}", flush=True)
        return 0

    clusterer_name = str(args.clusterer)
    shown = 0
    eval_report: Dict[str, object] = {}

    # ---------------------------------------------------------------
    # Pre-compute all embeddings using cross-image batching
    # This populates the cache so subsequent per-image lookups are fast
    # ---------------------------------------------------------------
    print(f"\n=== Pre-embedding {len(cluster_paths)} images with cross-image batching ===")
    print(f"    embed_batch={args.embed_batch} (patches accumulated), "
          f"gpu_batch_size={args.gpu_batch_size} (patches per forward pass)")
    
    all_embeddings, all_patches = _embed_images_batched(
        image_paths=cluster_paths,
        embedder=embedder,
        max_edge=args.max_edge,
        pad_size=args.pad_size,
        small_size=args.small_size,
        small_stride_x=args.small_stride_x,
        small_stride_y=args.small_stride_y,
        embed_batch=args.embed_batch,
        gpu_batch_size=args.gpu_batch_size,
        weights_path=args.weights_path,
        cache_dir=cache_dir,
        use_cache=args.cache_embeddings,
        device=args.device,
    )
    
    # Create a lookup dict for fast access during clustering/viz
    embeddings_by_path: Dict[Path, np.ndarray] = {
        p: emb for p, emb in zip(cluster_paths, all_embeddings) if emb is not None
    }
    patches_by_path: Dict[Path, List[Patch]] = {
        p: patches for p, patches in zip(cluster_paths, all_patches)
    }
    
    total_patches = sum(len(p) for p in all_patches)
    print(f"=== Pre-embedding complete: {len(cluster_paths)} images, {total_patches} total patches ===\n")

    # ---------------------------------------------------------------
    # Clustering
    # ---------------------------------------------------------------
    if clusterer_name == "hdbscan":
        fit_per_image = max(1, int(args.fit_small_patches_per_image))
        fit_embs: List[np.ndarray] = []
        t0 = time.time()

        # Use pre-computed embeddings with subsampling for fit
        for i, p in enumerate(tqdm(cluster_paths,
                                    desc="Subsampling embeddings (HDBSCAN fit)",
                                    unit="img",
                                    dynamic_ncols=True)):
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

        X = np.concatenate(fit_embs, axis=0)
        X = _l2_normalize_np(X)
        print(f"Fit embedding matrix: {X.shape} (N,D)", flush=True)

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
            small_patches_full = patches_by_path.get(p, [])

            # Use pre-computed embeddings
            embs_all = embeddings_by_path.get(p)
            if embs_all is None:
                continue
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
            _overlay_grid(axs[1], w=pw, h=ph, patch=args.large_size, stride=args.large_stride,
                          color="lime", lw=1.0,
                          stride_x=args.large_stride_x, stride_y=args.large_stride_y)
            axs[1].set_title(f"Large grid {args.large_size}  sx={args.large_stride_x} sy={args.large_stride_y}")
            axs[1].axis("off")

            axs[2].imshow(img)
            overlay = _colorize_small_patch_grid(small_patches_full, pred_labels.astype(np.int64), grid_w=pw, grid_h=ph)
            axs[2].imshow(overlay)
            axs[2].set_title(f"Sliding window {args.small_size}  sx={args.small_stride_x} sy={args.small_stride_y}")
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

        # Use pre-computed embeddings with subsampling for fit
        for i, p in enumerate(tqdm(cluster_paths,
                                    desc="Subsampling embeddings (fit set)",
                                    unit="img",
                                    dynamic_ncols=True)):
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

        X = np.concatenate(fit_embs, axis=0)
        X = _l2_normalize_np(X)
        print(f"Fit embedding matrix: {X.shape} (N,D)", flush=True)

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
            small_stride_x=int(args.small_stride_x),
            small_stride_y=int(args.small_stride_y),
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
            small_patches_full = patches_by_path.get(p, [])

            # Use pre-computed embeddings
            embs_all = embeddings_by_path.get(p)
            if embs_all is None:
                continue
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
            _overlay_grid(axs[1], w=pw, h=ph, patch=args.large_size, stride=args.large_stride,
                          color="lime", lw=1.0,
                          stride_x=args.large_stride_x, stride_y=args.large_stride_y)
            axs[1].set_title(f"Large grid {args.large_size}  sx={args.large_stride_x} sy={args.large_stride_y}")
            axs[1].axis("off")

            axs[2].imshow(img)
            overlay = _colorize_small_patch_grid(small_patches_full, pred_labels.astype(np.int64), grid_w=pw, grid_h=ph)
            axs[2].imshow(overlay)
            axs[2].set_title(f"Sliding window {args.small_size}  sx={args.small_stride_x} sy={args.small_stride_y}")
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
        "small_stride_x": int(args.small_stride_x),
        "small_stride_y": int(args.small_stride_y),
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
