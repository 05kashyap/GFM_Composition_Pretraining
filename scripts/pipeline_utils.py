#!/usr/bin/env python3
"""Shared utilities for the FMoW patch→embed→cluster pipeline.

Contains:
- Data classes (Patch, GridCell)
- Image loading and standardization
- Patch extraction (grid cells + sliding window)
- DINOv3 ViT-L/16 model definition and embedder
- Embedding cache I/O
- Multi-image batched embedder with GPU optimizations
- Threaded prefetch loader
- Clustering and PCA helpers
- Visualization helpers
"""

from __future__ import annotations

import hashlib
import json
import os
import random
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
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Patch:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class GridCell:
    """A single large-patch (grid cell) and the small patches it contains."""
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int
    patches: List[Patch]


# ---------------------------------------------------------------------------
# Image discovery and loading
# ---------------------------------------------------------------------------

def find_jpgs(data_root: Path, split: str) -> List[Path]:
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")
    paths = list(split_dir.rglob("*.jpg")) + list(split_dir.rglob("*.jpeg"))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise RuntimeError(f"No jpg images found under {split_dir}")
    return sorted(paths)


def sample_stratified(all_paths: List[Path], split_dir: Path, n: int, rng: random.Random) -> List[Path]:
    if n >= len(all_paths):
        sampled = list(all_paths)
        rng.shuffle(sampled)
        print(f"Stratified sampling: requested {n} >= available {len(all_paths)}; using all images")
        return sampled

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
    sampled_set = set()
    class_names = sorted(class_to_paths.keys())
    for i, cls in enumerate(class_names):
        pool = class_to_paths[cls]
        take = per_class + (1 if i < remainder else 0)
        take = min(take, len(pool))
        picked = rng.sample(pool, k=take)
        sampled.extend(picked)
        sampled_set.update(picked)

    # If some classes had fewer samples than their quota, redistribute the
    # unused quota across the remaining images so we still reach n whenever
    # enough images are available.
    deficit = n - len(sampled)
    if deficit > 0:
        remaining = [p for p in all_paths if p not in sampled_set]
        if remaining:
            extra = rng.sample(remaining, k=min(deficit, len(remaining)))
            sampled.extend(extra)

    rng.shuffle(sampled)
    sampled = sampled[: min(n, len(sampled))]
    print(
        f"Stratified sampling: {len(sampled)} images from {num_classes} classes (~{per_class} per class)"
    )
    return sampled


def resize_to_grid(img: Image.Image, grid_size: int) -> Image.Image:
    """Resize image so both dims are the nearest multiple of *grid_size*."""
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    w, h = img.size
    new_w = max(grid_size, round(w / grid_size) * grid_size)
    new_h = max(grid_size, round(h / grid_size) * grid_size)
    if new_w == w and new_h == h:
        return img
    return img.resize((new_w, new_h), Image.BICUBIC)


def load_and_standardize_image(path: Path, grid_size: int) -> Image.Image:
    """Load image, convert to RGB, resize so dims are multiples of *grid_size*."""
    img = Image.open(path)
    try:
        img.draft("RGB", (grid_size * 8, grid_size * 8))
    except Exception:
        pass
    img = img.convert("RGB")
    img = resize_to_grid(img, grid_size=grid_size)
    return img


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patches_in_grid(
    img_w: int,
    img_h: int,
    large_size: int,
    small_size: int,
    small_stride_x: int,
    small_stride_y: int,
) -> List[GridCell]:
    """Extract small patches confined within non-overlapping large grid cells."""
    if img_w % large_size != 0 or img_h % large_size != 0:
        raise ValueError(
            f"Image {img_w}x{img_h} not evenly divisible by large_size={large_size}. "
            "Resize to a multiple of large_size first."
        )
    cells: List[GridCell] = []
    grid_row = 0
    for ly in range(0, img_h, large_size):
        grid_col = 0
        for lx in range(0, img_w, large_size):
            cell_patches: List[Patch] = []
            for sy in range(0, large_size - small_size + 1, small_stride_y):
                for sx in range(0, large_size - small_size + 1, small_stride_x):
                    cell_patches.append(Patch(
                        x0=lx + sx, y0=ly + sy,
                        x1=lx + sx + small_size, y1=ly + sy + small_size,
                    ))
            cells.append(GridCell(
                row=grid_row, col=grid_col,
                x0=lx, y0=ly,
                x1=lx + large_size, y1=ly + large_size,
                patches=cell_patches,
            ))
            grid_col += 1
        grid_row += 1
    return cells


def flatten_cells(cells: List[GridCell]) -> List[Patch]:
    """Flatten grid cells into a single ordered list of small patches."""
    return [p for c in cells for p in c.patches]


def crop(img: Image.Image, patch: Patch) -> Image.Image:
    return img.crop((patch.x0, patch.y0, patch.x1, patch.y1))


def overlay_grid(ax, w: int, h: int, patch: int, stride: int, color: str, lw: float = 1.0,
                  stride_x: Optional[int] = None, stride_y: Optional[int] = None):
    """Draw grid lines on a matplotlib axis."""
    sx = stride_x if stride_x is not None else stride
    sy = stride_y if stride_y is not None else stride
    for y in range(0, h + 1, sy):
        ax.plot([0, w], [y, y], color=color, linewidth=lw)
    for x in range(0, w + 1, sx):
        ax.plot([x, x], [0, h], color=color, linewidth=lw)


# ---------------------------------------------------------------------------
# DINOv3 ViT-L/16 model definition (matches checkpoint keys exactly)
# ---------------------------------------------------------------------------

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
        "module.", "model.", "backbone.", "teacher.",
        "student.", "encoder.", "visual.", "vit.",
    )
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                changed = True
    return k


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    quarter = x.shape[-1] // 4
    x1, x2, x3, x4 = (x[..., :quarter], x[..., quarter:2*quarter],
                        x[..., 2*quarter:3*quarter], x[..., 3*quarter:])
    return torch.cat([-x2, x1, -x4, x3], dim=-1)


class _RoPE2D(torch.nn.Module):
    def __init__(self, dim: int, n_periods: int = 16, max_res: int = 256):
        super().__init__()
        self.register_buffer("periods", torch.arange(n_periods).float(), persistent=True)
        self.n_periods = n_periods
        self.max_res = max_res
        self._dim = dim

    def get_cos_sin(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        half = self.n_periods
        freqs = 1.0 / (10000.0 ** (self.periods / half))
        gy = torch.arange(h, device=device, dtype=dtype)
        gx = torch.arange(w, device=device, dtype=dtype)
        fy = torch.outer(gy, freqs)
        fx = torch.outer(gx, freqs)
        fy = fy[:, None, :].expand(h, w, half).reshape(h * w, half)
        fx = fx[None, :, :].expand(h, w, half).reshape(h * w, half)
        angles = torch.cat([fy, fy, fx, fx], dim=-1)
        return torch.cos(angles), torch.sin(angles)


class _Attention(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int = 16, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.register_buffer("qkv_bias_mask", torch.ones(dim * 3), persistent=False)
        self.proj = torch.nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        num_prefix: int = 0,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if rope_cos is not None:
            cos = rope_cos.unsqueeze(0).unsqueeze(0)
            sin = rope_sin.unsqueeze(0).unsqueeze(0)
            q_patch = q[:, :, num_prefix:] * cos + _rope_rotate_half(q[:, :, num_prefix:]) * sin
            k_patch = k[:, :, num_prefix:] * cos + _rope_rotate_half(k[:, :, num_prefix:]) * sin
            q = torch.cat([q[:, :, :num_prefix], q_patch], dim=2)
            k = torch.cat([k[:, :, :num_prefix], k_patch], dim=2)

        # Compute attention in fp32 to prevent fp16 overflow → NaN.
        # Q·K^T dot products over dim=64 can exceed fp16 max (65504),
        # causing inf → softmax NaN.  We must disable autocast here because
        # autocast re-casts explicit .float() inputs back to fp16 for matmuls.
        input_dtype = q.dtype
        with torch.cuda.amp.autocast(enabled=False):
            q = q.float()
            k = k.float()
            v = v.float()
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x.to(input_dtype)
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

    def forward(
        self, x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        num_prefix: int = 0,
    ) -> torch.Tensor:
        x = x + self.ls1(self.attn(
            self.norm1(x), rope_cos=rope_cos, rope_sin=rope_sin, num_prefix=num_prefix,
        ))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class _PatchEmbed(torch.nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        tokens = self.patch_embed(x)
        h = x.shape[2] // self.patch_size
        w = x.shape[3] // self.patch_size

        num_prefix = 1 + self.num_register_tokens
        cls = self.cls_token.expand(B, -1, -1)
        reg = self.storage_tokens.expand(B, -1, -1)
        tokens = torch.cat([cls, reg, tokens], dim=1)

        rope_cos, rope_sin = self.rope_embed.get_cos_sin(
            h, w, device=tokens.device, dtype=tokens.dtype,
        )

        for blk in self.blocks:
            tokens = blk(tokens, rope_cos=rope_cos, rope_sin=rope_sin, num_prefix=num_prefix)

        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        patch_out = tokens[:, num_prefix:]
        return cls_out, patch_out


class DinoV3SatViTL16Embedder(Embedder):
    """Local DINOv3 ViT-L/16 embedder using native architecture matching checkpoint."""

    _POOL_MODES = ("cls", "avg", "cls_avg")

    def __init__(self, device: str, weights_path: str, pool_mode: str = "cls_avg"):
        super().__init__(device=device)
        self.weights_path = str(weights_path)
        if pool_mode not in self._POOL_MODES:
            raise ValueError(f"pool_mode must be one of {self._POOL_MODES}, got '{pool_mode}'")
        self.pool_mode = pool_mode

        import torchvision.transforms as T

        model = DinoV3SatViTL16(
            patch_size=16, embed_dim=1024, depth=24,
            num_heads=16, mlp_ratio=4.0, num_register_tokens=4,
        )
        self.model = model.eval().to(self.device)

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        wpath = Path(self.weights_path)
        if not wpath.exists():
            raise FileNotFoundError(f"Weights not found: {wpath}")

        ckpt = torch.load(str(wpath), map_location="cpu", weights_only=False)
        sd_raw = _extract_checkpoint_state_dict(ckpt)
        sd = {_strip_known_prefixes(k): v for k, v in sd_raw.items() if isinstance(v, torch.Tensor)}

        mapped_sd: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            mapped_sd[k] = v

        incompatible = self.model.load_state_dict(mapped_sd, strict=False)
        n_missing = len(incompatible.missing_keys)
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

        self._embedding_dim = 2048 if pool_mode == "cls_avg" else 1024
        print(f"  Pool mode: {pool_mode} -> embedding_dim={self._embedding_dim}")

    @property
    def name(self) -> str:
        return f"dinov3_sat:{Path(self.weights_path).name}:{self.pool_mode}"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _pool(self, cls_emb: torch.Tensor, patch_embs: torch.Tensor) -> torch.Tensor:
        if self.pool_mode == "cls":
            return cls_emb
        avg = patch_embs.mean(dim=1)
        if self.pool_mode == "avg":
            return avg
        return torch.cat([cls_emb, avg], dim=-1)

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
                cls_emb, patch_embs = self.model(x)
            embs = self._pool(cls_emb, patch_embs)
            all_embs.append(embs.float().cpu())
        return torch.cat(all_embs, dim=0)

    @torch.inference_mode()
    def embed_tensor_patches(
        self, img_tensor: torch.Tensor, patches: List[Patch], batch_size: int,
    ) -> torch.Tensor:
        mean = self._IMAGENET_MEAN.to(self.device)
        std = self._IMAGENET_STD.to(self.device)
        use_amp = self.device.type == "cuda"
        all_embs: List[torch.Tensor] = []
        for i in range(0, len(patches), batch_size):
            chunk = patches[i : i + batch_size]
            patch_h = chunk[0].y1 - chunk[0].y0
            patch_w = chunk[0].x1 - chunk[0].x0
            crops = torch.empty(len(chunk), 3, patch_h, patch_w, dtype=torch.float32)
            for j, p in enumerate(chunk):
                crops[j] = img_tensor[:, p.y0:p.y1, p.x0:p.x1]
            if self.device.type == "cuda":
                crops = crops.pin_memory()
            crops = crops.to(self.device, non_blocking=True)
            crops = (crops - mean) / std
            with torch.cuda.amp.autocast(enabled=use_amp):
                cls_emb, patch_embs = self.model(crops)
            embs = self._pool(cls_emb, patch_embs)
            all_embs.append(embs.float().cpu())
        return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Embedding cache I/O
# ---------------------------------------------------------------------------

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


def cache_key(*, image_path: Path, embedder_name: str, grid_size: int,
              weights_path: str, patches_fp: str,
              cell_row: int, cell_col: int) -> str:
    mtime, fsize = _weights_fingerprint(weights_path)
    payload = {
        "image": str(image_path),
        "embedder": str(embedder_name),
        "grid_size": int(grid_size),
        "cell_row": int(cell_row),
        "cell_col": int(cell_col),
        "weights_path": str(weights_path),
        "weights_mtime": mtime,
        "weights_size": fsize,
        "patches_fp": patches_fp,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def compute_cache_path(
    cache_dir: Path, image_path: Path, embedder_name: str,
    grid_size: int, cell: GridCell, weights_path: str,
) -> Path:
    patches_fp = _patches_fingerprint(cell.patches)
    key = cache_key(
        image_path=image_path, embedder_name=embedder_name,
        grid_size=grid_size, weights_path=weights_path,
        patches_fp=patches_fp, cell_row=cell.row, cell_col=cell.col,
    )
    return cache_dir / f"{key}.npz"


def load_cached_embeddings(fpath: Path) -> Optional[np.ndarray]:
    if not fpath.exists():
        return None
    try:
        data = np.load(fpath, allow_pickle=False)
        return data["emb"].astype(np.float32, copy=False)
    except Exception:
        return None


def save_embeddings_to_cache(fpath: Path, emb: np.ndarray, image_path: Path,
                             cell_row: int = 0, cell_col: int = 0) -> None:
    if np.any(np.isnan(emb)):
        raise RuntimeError(
            f"Refusing to cache NaN embeddings for {image_path} "
            f"(cell row={cell_row} col={cell_col}, path={fpath}). "
            f"{int(np.isnan(emb).sum())}/{emb.size} values are NaN."
        )
    fpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        fpath,
        emb=emb.astype(np.float16, copy=False),
        image_path=str(image_path),
        cell_row=cell_row,
        cell_col=cell_col,
    )


# ---------------------------------------------------------------------------
# Multi-image batched embedder
# ---------------------------------------------------------------------------

@dataclass
class _PendingCell:
    image_path: Path
    img_tensor: torch.Tensor
    cell: GridCell
    cache_path: Path
    num_patches: int


class MultiImageBatchedEmbedder:
    """Batches patches across multiple grid cells for efficient GPU utilization."""

    def __init__(
        self,
        embedder: DinoV3SatViTL16Embedder,
        cache_dir: Path,
        grid_size: int,
        weights_path: str,
        batch_size: int,
        gpu_batch_size: int = 512,
    ):
        self.embedder = embedder
        self.cache_dir = cache_dir
        self.grid_size = grid_size
        self.weights_path = weights_path
        self.batch_size = batch_size
        self.gpu_batch_size = gpu_batch_size
        self._pending_cells: List[_PendingCell] = []
        self._total_pending: int = 0
        self.cells_processed = 0
        self.patches_embedded = 0
        self.forward_passes = 0

    def add_image(self, image_path: Path, img_tensor: torch.Tensor, cells: List[GridCell]) -> None:
        for cell in cells:
            cache_path = compute_cache_path(
                self.cache_dir, image_path, self.embedder.name,
                self.grid_size, cell, self.weights_path,
            )
            if cache_path.exists():
                self.cells_processed += 1
                continue
            pending = _PendingCell(
                image_path=image_path, img_tensor=img_tensor,
                cell=cell, cache_path=cache_path,
                num_patches=len(cell.patches),
            )
            self._pending_cells.append(pending)
            self._total_pending += len(cell.patches)
        while self._total_pending >= self.batch_size:
            self._flush_batch()

    def flush(self) -> None:
        while self._pending_cells:
            self._flush_batch()

    def _flush_batch(self) -> None:
        if not self._pending_cells:
            return
        cells_in_batch: List[_PendingCell] = []
        batch_count = 0
        remaining: List[_PendingCell] = []
        for pending in self._pending_cells:
            if batch_count + pending.num_patches <= self.batch_size or batch_count == 0:
                cells_in_batch.append(pending)
                batch_count += pending.num_patches
            else:
                remaining.append(pending)
        self._pending_cells = remaining
        if not cells_in_batch:
            return

        p0 = cells_in_batch[0].cell.patches[0]
        patch_h = p0.y1 - p0.y0
        patch_w = p0.x1 - p0.x0
        crops_tensor = torch.empty(batch_count, 3, patch_h, patch_w, dtype=torch.float32)
        offset = 0
        for cell_info in cells_in_batch:
            for p in cell_info.cell.patches:
                crops_tensor[offset] = cell_info.img_tensor[:, p.y0:p.y1, p.x0:p.x1]
                offset += 1

        embeddings = self._embed_batch(crops_tensor)

        offset = 0
        for cell_info in cells_in_batch:
            n = cell_info.num_patches
            cell_emb = embeddings[offset : offset + n]
            save_embeddings_to_cache(
                cell_info.cache_path, cell_emb, cell_info.image_path,
                cell_row=cell_info.cell.row, cell_col=cell_info.cell.col,
            )
            offset += n
            self.cells_processed += 1
        self.patches_embedded += batch_count
        self.forward_passes += 1
        self._total_pending = sum(c.num_patches for c in self._pending_cells)

    @torch.inference_mode()
    def _embed_batch(self, crops: torch.Tensor) -> np.ndarray:
        device = self.embedder.device
        mean = self.embedder._IMAGENET_MEAN.to(device)
        std = self.embedder._IMAGENET_STD.to(device)
        sub_batch_size = self.gpu_batch_size

        if device.type == "cuda" and not crops.is_pinned():
            crops = crops.pin_memory()

        all_embs: List[torch.Tensor] = []

        if device.type == "cuda":
            compute_stream = torch.cuda.current_stream(device)
            transfer_stream = torch.cuda.Stream(device)
            n = crops.shape[0]
            starts = list(range(0, n, sub_batch_size))

            with torch.cuda.stream(transfer_stream):
                first_end = min(sub_batch_size, n)
                next_batch = crops[0:first_end].to(device, non_blocking=True)

            for si, start in enumerate(starts):
                compute_stream.wait_stream(transfer_stream)
                batch = next_batch
                next_start = start + sub_batch_size
                if next_start < n:
                    with torch.cuda.stream(transfer_stream):
                        next_end = min(next_start + sub_batch_size, n)
                        next_batch = crops[next_start:next_end].to(device, non_blocking=True)
                batch = (batch - mean) / std
                with torch.cuda.amp.autocast(enabled=True):
                    cls_emb, patch_embs = self.embedder.model(batch)
                embs = self.embedder._pool(cls_emb, patch_embs)
                all_embs.append(embs.float().cpu())
        else:
            for i in range(0, crops.shape[0], sub_batch_size):
                batch = crops[i : i + sub_batch_size].to(device)
                batch = (batch - mean) / std
                cls_emb, patch_embs = self.embedder.model(batch)
                embs = self.embedder._pool(cls_emb, patch_embs)
                all_embs.append(embs.cpu())

        result = torch.cat(all_embs, dim=0).numpy().astype(np.float32)
        if np.any(np.isnan(result)):
            nan_count = int(np.isnan(result).sum())
            raise RuntimeError(
                f"NaN detected in embeddings immediately after model forward pass! "
                f"{nan_count}/{result.size} values are NaN. "
                f"This indicates the fp16 overflow fix in _Attention.forward() is not working. "
                f"Check that torch.cuda.amp.autocast(enabled=False) wraps the attention matmul."
            )
        return result


# ---------------------------------------------------------------------------
# Threaded prefetch loader
# ---------------------------------------------------------------------------

class PrefetchLoader:
    """Threaded prefetcher: pre-applies *load_fn* to upcoming items."""

    def __init__(self, items, load_fn, num_workers: int = 4, prefetch_count: int = 16):
        self._items = list(items)
        self._load_fn = load_fn
        self._num_workers = num_workers
        self._prefetch_count = prefetch_count

    def __iter__(self):
        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            futures: deque = deque()
            item_iter = iter(self._items)
            for _ in range(min(self._prefetch_count, len(self._items))):
                try:
                    item = next(item_iter)
                    futures.append(executor.submit(self._load_fn, item))
                except StopIteration:
                    break
            while futures:
                result = futures.popleft().result()
                try:
                    item = next(item_iter)
                    futures.append(executor.submit(self._load_fn, item))
                except StopIteration:
                    pass
                yield result

    def __len__(self):
        return len(self._items)


# ---------------------------------------------------------------------------
# Clustering and PCA helpers
# ---------------------------------------------------------------------------

def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def fit_pca(x: np.ndarray, pca_dim: int, seed: int):
    try:
        from sklearn.decomposition import PCA
    except Exception as e:
        raise RuntimeError("scikit-learn is required for PCA.\n" + str(e))
    pca_dim = int(pca_dim)
    if pca_dim <= 0 or pca_dim > x.shape[1]:
        raise ValueError(f"Invalid pca_dim={pca_dim} for embedding_dim={x.shape[1]}")
    pca = PCA(n_components=pca_dim, random_state=seed)
    x_pca = pca.fit_transform(x)
    return pca, x_pca


def fit_hdbscan(x: np.ndarray, min_cluster_size: int, min_samples: int, n_jobs: int):
    try:
        import hdbscan
    except Exception as e:
        raise RuntimeError("hdbscan is required.\n" + str(e))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples) if int(min_samples) > 0 else None,
        metric="euclidean",
        core_dist_n_jobs=int(n_jobs),
        prediction_data=True,
    )
    labels = clusterer.fit_predict(x)
    return clusterer, labels


def fit_sklearn_kmeans(x: np.ndarray, k: int, seed: int, batch_size: int = 1024):
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:
        raise RuntimeError("scikit-learn is required.\n" + str(e))
    km = MiniBatchKMeans(
        n_clusters=int(k), random_state=int(seed),
        batch_size=int(batch_size), n_init=3, max_iter=300,
    )
    labels = km.fit_predict(x)
    return km, labels


def fit_bisecting_kmeans(x: np.ndarray, k: int, seed: int):
    try:
        from sklearn.cluster import BisectingKMeans
    except Exception as e:
        raise RuntimeError("scikit-learn >= 1.1 required.\n" + str(e))
    bkm = BisectingKMeans(n_clusters=int(k), random_state=int(seed), n_init=3, max_iter=300)
    labels = bkm.fit_predict(x)
    return bkm, labels


def fit_gmm(x: np.ndarray, k: int, seed: int):
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as e:
        raise RuntimeError("scikit-learn is required.\n" + str(e))
    gmm = GaussianMixture(
        n_components=int(k), covariance_type="diag",
        random_state=int(seed), n_init=3, max_iter=300,
    )
    labels = gmm.fit_predict(x)
    return gmm, labels


def silhouette_optional(x: np.ndarray, labels: np.ndarray, seed: int, sample_size: int = 20000) -> Optional[float]:
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


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def colorize_small_patch_grid(small_patches: List[Patch], labels: np.ndarray,
                              grid_w: int, grid_h: int) -> np.ndarray:
    n_labels = max(int(labels.max()) + 1, 1) if labels.size > 0 else 1
    if n_labels <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("nipy_spectral", n_labels)
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
    mask = count > 0
    overlay = np.zeros((grid_h, grid_w, 4), dtype=np.float32)
    for c in range(3):
        overlay[:, :, c][mask] = (color_acc[:, :, c][mask] / count[mask]).astype(np.float32)
    overlay[:, :, 3][mask] = 0.45
    return overlay


def save_viz_embeddings(out_dir: Path, shown_idx: int, image_path: Path,
                        patches: List[Patch], emb_raw: np.ndarray,
                        labels: np.ndarray, cells: Optional[List[GridCell]] = None,
                        emb_pca: Optional[np.ndarray] = None) -> None:
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    patch_xyxy = np.asarray([[p.x0, p.y0, p.x1, p.y1] for p in patches], dtype=np.int32)
    cell_index = np.zeros(len(patches), dtype=np.int32)
    if cells is not None:
        idx = 0
        for ci, cell in enumerate(cells):
            for _ in cell.patches:
                if idx < len(patches):
                    cell_index[idx] = ci
                idx += 1
    save_path = emb_dir / f"image_{shown_idx:02d}.npz"
    save_kwargs = dict(
        image_path=str(image_path),
        patch_xyxy=patch_xyxy,
        emb_raw=emb_raw.astype(np.float32, copy=False),
        labels=labels.astype(np.int64, copy=False),
        cell_index=cell_index,
    )
    if emb_pca is not None:
        save_kwargs["emb_pca"] = emb_pca.astype(np.float32, copy=False)
    if cells is not None:
        cell_xyxy = np.asarray([[c.x0, c.y0, c.x1, c.y1] for c in cells], dtype=np.int32)
        save_kwargs["cell_xyxy"] = cell_xyxy
    np.savez_compressed(save_path, **save_kwargs)
