"""
FMoW Composition Dataset for composition-aware DynamicVis training.

Loads pre-computed compositional targets (produced by ``cluster_viz.py
--save-cluster-data``) together with FMoW images.  Each sample is a single
grid cell (large patch) cropped from a source image.

Registration:
    Registered as ``'FMoWCompositionDataset'`` with both mmpretrain and mmdet
    DATASETS registries so it can be referenced in MMEngine configs via
    ``custom_imports``.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

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


class CompositionDataSample(BaseDataElement):
    """Minimal data sample carrying composition metadata.

    Attributes set per sample:
        composition_target  (torch.Tensor): (D,) compositional target vector.
        image_id            (int): index of the source image.
        cell_row            (int): grid row of this cell.
        cell_col            (int): grid column of this cell.
    """
    pass


# ──────────────────────────────────────────────────────────────────────
# Image augmentation helpers (lightweight, no mmcv dependency)
# ──────────────────────────────────────────────────────────────────────

def _train_transform(img: Image.Image, size: int) -> torch.Tensor:
    """Resize to ``size``, random flip, colour jitter, to tensor."""
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),          # [0,1] float32, CHW
    ])
    return transform(img)


def _val_transform(img: Image.Image, size: int) -> torch.Tensor:
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])
    return transform(img)


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

@MMPRETRAIN_DATASETS.register_module()
class FMoWCompositionDataset(Dataset):
    """Per-cell composition dataset driven by ``cluster_viz.py`` exports.

    Args:
        cluster_data_dir: Directory containing ``manifest.json`` and
            ``targets.npy`` (produced by cluster_viz.py --save-cluster-data).
        img_size: Images / cells are resized to ``(img_size, img_size)``.
        split: ``'train'`` or ``'val'`` — controls augmentation.
        val_ratio: Fraction of images held out for validation when
            ``split='val'``.  Images are split deterministically by hash
            so train/val sets are disjoint.
        max_samples: Cap the dataset size (for debugging).
    """

    def __init__(
        self,
        cluster_data_dir: str,
        img_size: int = 512,
        split: str = "train",
        val_ratio: float = 0.1,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.split = split

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

        # Load pre-computed targets (memory-mapped for efficiency)
        self.targets = np.load(targets_path, mmap_mode="r")
        assert self.targets.shape[0] >= len(all_cells), (
            f"targets.npy has {self.targets.shape[0]} rows but manifest "
            f"has {len(all_cells)} cells"
        )

        # Assign a stable integer image_id per unique image path
        unique_images = sorted(set(c["image_path"] for c in all_cells))
        self._img_to_id: Dict[str, int] = {p: i for i, p in enumerate(unique_images)}

        # Deterministic train / val split based on image path hash
        # Use hashlib instead of hash() to be deterministic across
        # DDP ranks (Python randomizes hash() per process).
        import hashlib
        train_cells, val_cells = [], []
        for cell in all_cells:
            h = int(hashlib.md5(cell["image_path"].encode()).hexdigest(), 16) % 10000
            if h < int(val_ratio * 10000):
                val_cells.append(cell)
            else:
                train_cells.append(cell)

        self.cells = train_cells if split == "train" else val_cells

        if max_samples is not None:
            self.cells = self.cells[:max_samples]

        print(
            f"FMoWCompositionDataset({split}): {len(self.cells)} cells from "
            f"{len(set(c['image_path'] for c in self.cells))} images  "
            f"(target dim {self.embedding_dim}, grid {self.grid_size}px)"
        )

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

        # ---- Augment + to tensor ----
        if self.split == "train":
            img_tensor = _train_transform(cell_img, self.img_size)
        else:
            img_tensor = _val_transform(cell_img, self.img_size)

        # ---- Compositional target ----
        target = torch.from_numpy(
            self.targets[entry["target_index"]].copy()
        ).float()

        # ---- Pack into data sample ----
        data_sample = CompositionDataSample()
        data_sample.composition_target = target
        data_sample.image_id = self._img_to_id[entry["image_path"]]
        data_sample.cell_row = entry["cell_row"]
        data_sample.cell_col = entry["cell_col"]

        return {"inputs": img_tensor, "data_samples": data_sample}

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
