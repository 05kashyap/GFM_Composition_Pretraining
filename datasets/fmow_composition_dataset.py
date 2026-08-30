"""
FMoW Composition Dataset for composition-aware DynamicVis training.

Loads pre-computed cell metadata (produced by ``cluster_viz.py
--save-cluster-data``) together with FMoW images.  Each sample is a single
grid cell (large patch) cropped from a source image.

Supports multi-view augmentation (QSACL-style) where each cell produces
N augmented views for contrastive learning across view pairs.

Optionally loads DINOv3 patch embeddings from the offline pipeline for
per-slot contrastive learning via QuerySlotDecoder.

Registration:
    Registered as ``'FMoWCompositionDataset'`` with both mmpretrain and mmdet
    DATASETS registries so it can be referenced in MMEngine configs via
    ``custom_imports``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

Image.MAX_IMAGE_PIXELS = None

# MMEngine data-sample container
from mmengine.structures import BaseDataElement

# Register with mmpretrain DATASETS
from mmpretrain.registry import DATASETS as MMPRETRAIN_DATASETS

# Also import mmdet DATASETS if available (for cross-compatibility)
try:
    from mmdet.registry import DATASETS as MMDET_DATASETS
    _HAS_MMDET = True
except ImportError:
    _HAS_MMDET = False

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CompositionDataSample(BaseDataElement):
    """Minimal data sample carrying composition metadata.

    Attributes set per sample:
        composition_target  (torch.Tensor): (D,) compositional target vector.
        image_id            (int): index of the source image.
        cell_row            (int): grid row of this cell.
        cell_col            (int): grid column of this cell.
        dominant_label      (int): fMoW category id (0–62), or -1 if unlabeled.
        patch_embeddings    (torch.Tensor): (N_patches, 2048) DINOv3 patch embeddings,
                            or None if not loaded.
    """
    pass


# ──────────────────────────────────────────────────────────────────────
# Patch embedding cache path computation
# ──────────────────────────────────────────────────────────────────────

def _weights_fingerprint(weights_path: str) -> Tuple[float, int]:
    """Get mtime and size for weights file fingerprinting."""
    p = Path(weights_path)
    if p.exists():
        stat = p.stat()
        return (stat.st_mtime, stat.st_size)
    return (0.0, 0)


def _compute_patches_for_cell(
    cell_x0: int, cell_y0: int, cell_x1: int, cell_y1: int,
    small_size: int, small_stride_x: int, small_stride_y: int,
) -> List[Tuple[int, int, int, int]]:
    """Compute patch coordinates within a cell.

    Returns list of (x0, y0, x1, y1) tuples for each patch.
    """
    patches = []
    cell_w = cell_x1 - cell_x0
    cell_h = cell_y1 - cell_y0

    y = 0
    while y + small_size <= cell_h:
        x = 0
        while x + small_size <= cell_w:
            # Absolute coordinates
            patches.append((
                cell_x0 + x,
                cell_y0 + y,
                cell_x0 + x + small_size,
                cell_y0 + y + small_size,
            ))
            x += small_stride_x
        y += small_stride_y

    return patches


def _patches_fingerprint(patches: List[Tuple[int, int, int, int]]) -> str:
    """Compute SHA1 fingerprint of patch coordinates."""
    arr = np.asarray(patches, dtype=np.int32)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _compute_cache_key(
    image_path: str,
    embedder_name: str,
    grid_size: int,
    weights_path: str,
    patches_fp: str,
    cell_row: int,
    cell_col: int,
) -> str:
    """Compute cache key for a cell's patch embeddings.

    NOTE: This must match the cache_key() function in scripts/pipeline_utils.py
    exactly, otherwise the dataset will not find the cached embeddings.
    """
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


# ──────────────────────────────────────────────────────────────────────
# Image augmentation helpers (multi-view QSACL-style)
# ──────────────────────────────────────────────────────────────────────

def _base_augmentation() -> T.Compose:
    """Base augmentation pipeline used for all views."""
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _global_view_transform(img: Image.Image, size: int) -> torch.Tensor:
    """Global view: Resize to ``size`` then apply augmentation."""
    transform = T.Compose([
        T.Resize((size, size)),
        _base_augmentation(),
    ])
    return transform(img)


def _local_view_transform(img: Image.Image, crop_size: int, target_size: int) -> torch.Tensor:
    """Local view: RandomCrop to ``crop_size``, resize to ``target_size``, then augment.

    The crop captures a smaller region of the image, but the output is resized
    to match the backbone's expected input size (e.g., 512x512).
    """
    transform = T.Compose([
        T.RandomCrop(crop_size),
        T.Resize((target_size, target_size)),  # Resize to backbone input size
        _base_augmentation(),
    ])
    return transform(img)


def _train_transform(img: Image.Image, size: int) -> torch.Tensor:
    """Single-view train transform (for backward compatibility)."""
    transform = T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(img)


def _val_transform(img: Image.Image, size: int) -> torch.Tensor:
    """Val transform: resize only, no augmentation."""
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(img)


def _generate_multi_views(
    img: Image.Image,
    global_size: int,
    num_global: int = 2,
    num_local: int = 6,
    local_sizes: List[int] = [192, 256, 320],
) -> List[torch.Tensor]:
    """Generate N augmented views following SkySense multi-crop strategy.

    Args:
        img: PIL Image to augment (should be at least as large as global_size).
        global_size: Size for global views (512).
        num_global: Number of global views (default 2).
        num_local: Number of local views (default 6).
        local_sizes: Pool of sizes to sample local crop sizes from.

    Returns:
        List of N tensors (num_global + num_local views).
    """
    views = []

    # Global views: Resize to global_size, then augment
    for _ in range(num_global):
        views.append(_global_view_transform(img, global_size))

    # Local views: RandomCrop(size), resize to global_size, then augment
    # Need to resize img first so we can take meaningful crops
    # Local crops are taken from the full cell image
    img_resized = img.resize((global_size, global_size), Image.BILINEAR)
    for _ in range(num_local):
        crop_size = random.choice(local_sizes)
        views.append(_local_view_transform(img_resized, crop_size, global_size))

    return views


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

@MMPRETRAIN_DATASETS.register_module()
class FMoWCompositionDataset(Dataset):
    """Per-cell composition dataset driven by ``cluster_viz.py`` exports.

    Supports multi-view augmentation (QSACL-style) where each cell produces
    N augmented views for contrastive learning across view pairs.

    Args:
        cluster_data_dir: Directory containing ``manifest.json`` and
            optionally ``targets.npy`` and ``cell_labels.npy``.
        img_size: Images / cells are resized to ``(img_size, img_size)``.
        split: ``'train'`` or ``'val'`` — controls augmentation.
        val_ratio: Fraction of images held out for validation when
            ``split='val'``.  Images are split deterministically by hash
            so train/val sets are disjoint.
        max_samples: Cap the dataset size (for debugging).
        patch_embed_dir: Directory containing .npz patch embedding files
            from embed_patches.py.  If None, patch embeddings will not be
            loaded (QuerySlotDecoder will not receive inputs).
        num_views: Number of augmented views per cell (default 1 for backward
            compatibility).  When >1, output is a list of tensors.  Following
            SkySense: 2 global views + (num_views-2) local views.
        local_crop_sizes: Pool of crop sizes for local views (default [192, 256, 320]).
    """

    def __init__(
        self,
        cluster_data_dir: str,
        img_size: int = 512,
        split: str = "train",
        val_ratio: float = 0.1,
        max_samples: Optional[int] = None,
        patch_embed_dir: Optional[str] = None,
        num_views: int = 1,
        local_crop_sizes: Optional[List[int]] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.split = split
        self.num_views = num_views
        self.local_crop_sizes = local_crop_sizes or [192, 256, 320]

        cluster_data_dir = Path(cluster_data_dir)
        manifest_path = cluster_data_dir / "manifest.json"
        targets_path = cluster_data_dir / "targets.npy"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}.  Run "
                f"cluster_viz.py --save-cluster-data first."
            )

        # Load manifest
        with open(manifest_path) as f:
            manifest = json.load(f)
        all_cells: List[dict] = manifest["cells"]
        self.embedding_dim: int = manifest["embedding_dim"]
        self.grid_size: int = manifest["grid_size"]

        # Store patch embedding parameters from manifest
        self._small_size: int = manifest.get("small_size", 128)
        self._small_stride_x: int = manifest.get("small_stride_x", 64)
        self._small_stride_y: int = manifest.get("small_stride_y", 128)
        self._embedder_name: str = manifest.get("embedder_name", "")
        self._weights_path: str = manifest.get("weights_path", "")
        self._original_embedding_dim: int = manifest.get("original_embedding_dim", 2048)

        # Patch embedding directory
        self._patch_embed_dir: Optional[Path] = None
        if patch_embed_dir is not None:
            self._patch_embed_dir = Path(patch_embed_dir)
            if not self._patch_embed_dir.exists():
                logger.warning(
                    f"patch_embed_dir does not exist: {self._patch_embed_dir}. "
                    f"Patch embeddings will not be loaded."
                )
                self._patch_embed_dir = None

        # Load pre-computed targets (optional - only needed if loss_comp > 0)
        if targets_path.exists():
            self.targets = np.load(targets_path, mmap_mode="r")
            assert self.targets.shape[0] >= len(all_cells), (
                f"targets.npy has {self.targets.shape[0]} rows but manifest "
                f"has {len(all_cells)} cells"
            )
            self._has_targets = True
        else:
            self.targets = None
            self._has_targets = False
            logger.info(
                f"targets.npy not found in {cluster_data_dir}. "
                f"Composition targets will not be loaded (loss_comp should be 0)."
            )

        # Optionally load fMoW cell labels (produced by assign_cell_labels.py)
        labels_path = cluster_data_dir / "cell_labels.npy"
        if labels_path.exists():
            self._all_labels = np.load(labels_path, mmap_mode="r")
            assert self._all_labels.shape[0] >= len(all_cells), (
                f"cell_labels.npy has {self._all_labels.shape[0]} rows but "
                f"manifest has {len(all_cells)} cells"
            )
            self._has_labels = True
        else:
            self._all_labels = None
            self._has_labels = False

        # Assign a stable integer image_id per unique image path
        unique_images = sorted(set(c["image_path"] for c in all_cells))
        self._img_to_id: Dict[str, int] = {p: i for i, p in enumerate(unique_images)}

        # Deterministic train / val split based on image path hash
        # Use hashlib instead of hash() to be deterministic across
        # DDP ranks (Python randomizes hash() per process).
        import hashlib
        train_cells, val_cells = [], []
        for global_idx, cell in enumerate(all_cells):
            # Attach the global index so we can look up labels later
            cell["_global_idx"] = global_idx
            h = int(hashlib.md5(cell["image_path"].encode()).hexdigest(), 16) % 10000
            if h < int(val_ratio * 10000):
                val_cells.append(cell)
            else:
                train_cells.append(cell)

        self.cells = train_cells if split == "train" else val_cells

        if max_samples is not None:
            self.cells = self.cells[:max_samples]

        # Track missing embedding warnings (throttled to avoid log spam)
        self._missing_embed_warned = set()

        print(
            f"FMoWCompositionDataset({split}): {len(self.cells)} cells from "
            f"{len(set(c['image_path'] for c in self.cells))} images  "
            f"(target dim {self.embedding_dim}, grid {self.grid_size}px, "
            f"labels={'yes' if self._has_labels else 'no'}, "
            f"patch_embeds={'yes' if self._patch_embed_dir else 'no'}, "
            f"num_views={self.num_views})"
        )

    # ------------------------------------------------------------------ #
    # Patch embedding loading
    # ------------------------------------------------------------------ #

    def _load_patch_embeddings(self, entry: dict) -> Optional[torch.Tensor]:
        """Load patch embeddings for a cell from the cache.

        Args:
            entry: Cell entry dict from manifest with image_path, cell coords.

        Returns:
            Tensor of shape (N_patches, patch_dim) or None if not found.
        """
        if self._patch_embed_dir is None:
            return None

        # Compute patch coordinates for this cell
        patches = _compute_patches_for_cell(
            entry["cell_x0"], entry["cell_y0"],
            entry["cell_x1"], entry["cell_y1"],
            self._small_size, self._small_stride_x, self._small_stride_y,
        )

        # Compute cache key
        patches_fp = _patches_fingerprint(patches)
        cache_key = _compute_cache_key(
            image_path=entry["image_path"],
            embedder_name=self._embedder_name,
            grid_size=self.grid_size,
            weights_path=self._weights_path,
            patches_fp=patches_fp,
            cell_row=entry["cell_row"],
            cell_col=entry["cell_col"],
        )
        cache_path = self._patch_embed_dir / f"{cache_key}.npz"

        if not cache_path.exists():
            # Warn only once per unique cell to avoid log spam
            cell_id = (entry["image_path"], entry["cell_row"], entry["cell_col"])
            if cell_id not in self._missing_embed_warned:
                logger.warning(
                    f"Patch embedding cache not found for cell "
                    f"{entry['image_path']}:{entry['cell_row']},{entry['cell_col']}. "
                    f"Expected: {cache_path}"
                )
                self._missing_embed_warned.add(cell_id)
            return None

        try:
            data = np.load(cache_path, allow_pickle=False)
            emb = data["emb"].astype(np.float32, copy=False)
            return torch.from_numpy(emb)
        except Exception as e:
            logger.warning(f"Failed to load patch embeddings from {cache_path}: {e}")
            return None

    # ------------------------------------------------------------------ #
    # PyTorch Dataset interface
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, idx: int) -> dict:
        entry = self.cells[idx]

        # ---- Load + crop cell from image ----
        img = Image.open(entry["image_path"]).convert("RGB")
        cell_img = img.crop(
            (entry["cell_x0"], entry["cell_y0"],
             entry["cell_x1"], entry["cell_y1"])
        )

        # ---- Generate views based on num_views and split ----
        if self.split == "train" and self.num_views > 1:
            # Multi-view: 2 global + (num_views - 2) local crops
            num_global = min(2, self.num_views)
            num_local = max(0, self.num_views - 2)
            img_tensors = _generate_multi_views(
                cell_img,
                global_size=self.img_size,
                num_global=num_global,
                num_local=num_local,
                local_sizes=self.local_crop_sizes,
            )
        elif self.split == "train":
            # Single-view train
            img_tensors = _train_transform(cell_img, self.img_size)
        else:
            # Validation: single view, no augmentation
            img_tensors = _val_transform(cell_img, self.img_size)

        # ---- Compositional target (optional) ----
        if self._has_targets:
            target = torch.from_numpy(
                self.targets[entry["target_index"]].copy()
            ).float()
        else:
            # Return zeros if targets not loaded
            target = torch.zeros(self.embedding_dim, dtype=torch.float32)

        # ---- Load patch embeddings (for QuerySlotDecoder) ----
        # Only load if patch_embed_dir is configured; otherwise leave as None
        # (the model handles this when use_dynamicvis_keys=True)
        patch_embeddings = None
        if self._patch_embed_dir is not None:
            patch_embeddings = self._load_patch_embeddings(entry)
            # If missing from cache, return zeros of expected shape
            if patch_embeddings is None:
                expected_n_patches = entry.get("n_patches", 28)
                patch_embeddings = torch.zeros(
                    expected_n_patches, self._original_embedding_dim
                )

        # ---- Pack into data sample ----
        data_sample = CompositionDataSample()
        data_sample.composition_target = target
        data_sample.image_id = self._img_to_id[entry["image_path"]]
        data_sample.cell_row = entry["cell_row"]
        data_sample.cell_col = entry["cell_col"]
        data_sample.patch_embeddings = patch_embeddings

        # fMoW dominant label (-1 = unlabeled / background)
        if self._has_labels:
            data_sample.dominant_label = int(
                self._all_labels[entry["_global_idx"]]
            )
        else:
            data_sample.dominant_label = -1

        return {"inputs": img_tensors, "data_samples": data_sample}

    # ------------------------------------------------------------------ #
    # MMEngine compatibility
    # ------------------------------------------------------------------ #

    @property
    def metainfo(self) -> dict:
        return {
            "dataset_type": "FMoWCompositionDataset",
            "grid_size": self.grid_size,
            "embedding_dim": self.embedding_dim,
        }


# Register with mmdet registry too (if available)
if _HAS_MMDET:
    MMDET_DATASETS.register_module(module=FMoWCompositionDataset, force=True)


# ──────────────────────────────────────────────────────────────────────
# Multi-view collate function
# ──────────────────────────────────────────────────────────────────────

def multiview_collate_fn(batch: List[dict]) -> dict:
    """Collate function for multi-view datasets.

    Handles both single-view (backward compatible) and multi-view inputs.

    For single-view:
        inputs: (B, C, H, W) tensor

    For multi-view:
        inputs: list of N tensors, each (B, C, H_i, W_i)
        where H_i, W_i may vary for local crops

    Args:
        batch: List of dicts with 'inputs' and 'data_samples' keys.
            'inputs' can be a single tensor or a list of tensors.

    Returns:
        dict with:
            'inputs': stacked tensor (single-view) or list of stacked tensors (multi-view)
            'data_samples': list of CompositionDataSample objects
    """
    data_samples = [item["data_samples"] for item in batch]

    # Check if we have multi-view inputs
    first_inputs = batch[0]["inputs"]
    is_multiview = isinstance(first_inputs, list)

    if is_multiview:
        # Multi-view: inputs is a list of N tensors per sample
        # We need to group views by index and stack within each group
        num_views = len(first_inputs)
        inputs_list = []

        for view_idx in range(num_views):
            # Collect all samples' view_idx tensor
            view_tensors = [item["inputs"][view_idx] for item in batch]

            # Check if all tensors have the same shape (global views should)
            shapes = [t.shape for t in view_tensors]
            if len(set(shapes)) == 1:
                # All same shape - can stack directly
                inputs_list.append(torch.stack(view_tensors, dim=0))
            else:
                # Different shapes (local crops) - pad to max size
                max_h = max(s[1] for s in shapes)
                max_w = max(s[2] for s in shapes)
                padded = []
                for t in view_tensors:
                    if t.shape[1] < max_h or t.shape[2] < max_w:
                        # Pad to max size (right/bottom padding)
                        pad_h = max_h - t.shape[1]
                        pad_w = max_w - t.shape[2]
                        t = F.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=0)
                    padded.append(t)
                inputs_list.append(torch.stack(padded, dim=0))

        return {"inputs": inputs_list, "data_samples": data_samples}
    else:
        # Single-view: stack directly
        inputs = torch.stack([item["inputs"] for item in batch], dim=0)
        return {"inputs": inputs, "data_samples": data_samples}

