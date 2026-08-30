"""
FMoW BoVW Dataset for BoVW composition-aware DynamicVis training.

Loads pre-computed soft histogram targets (produced by ``generate_histograms.py``)
together with FMoW cell images. Each sample is a single 512x512 grid cell cropped
from a source image.

Supports multi-view augmentation where each cell produces N augmented views.

Registration:
    Registered as ``'FMoWBoVWDataset'`` with mmpretrain DATASETS registry.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

Image.MAX_IMAGE_PIXELS = None

# MMEngine data-sample container
from mmengine.structures import BaseDataElement

# Register with mmpretrain DATASETS
from mmpretrain.registry import DATASETS as MMPRETRAIN_DATASETS

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BoVWDataSample(BaseDataElement):
    """Minimal data sample carrying BoVW metadata.

    Attributes set per sample:
        histogram_target    (torch.Tensor): (K,) soft histogram target.
        dominant_label      (int): fMoW category id (0-62), or -1 if unlabeled.
        image_id            (int): index into manifest.
        cell_row            (int): grid row of this cell.
        cell_col            (int): grid column of this cell.
    """
    pass


# --------------------------------------------------------------------------- #
# Image augmentation helpers
# --------------------------------------------------------------------------- #

def _base_augmentation() -> T.Compose:
    """Base augmentation pipeline used for all views."""
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _train_transform(img: Image.Image, size: int) -> torch.Tensor:
    """Train transform: resize + augmentation."""
    transform = T.Compose([
        T.Resize((size, size)),
        _base_augmentation(),
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


def _generate_views(
    img: Image.Image,
    size: int,
    num_views: int,
    is_train: bool,
) -> List[torch.Tensor]:
    """Generate N augmented views.

    Args:
        img: PIL Image to augment.
        size: Output size (512).
        num_views: Number of views to generate.
        is_train: If True, apply augmentation; else just resize.

    Returns:
        List of N tensors.
    """
    views = []
    for _ in range(num_views):
        if is_train:
            views.append(_train_transform(img, size))
        else:
            views.append(_val_transform(img, size))
    return views


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

@MMPRETRAIN_DATASETS.register_module()
class FMoWBoVWDataset(Dataset):
    """Per-cell BoVW dataset driven by histogram targets from Phase 3.

    Args:
        manifest_path: Path to manifest.json (from cluster_viz.py or similar).
        histogram_dir: Directory containing histograms.npy and cell_ids.npy
            from generate_histograms.py.
        cell_labels_path: Path to cell_labels.npy (from assign_cell_labels.py).
            Can be None if labels are not needed.
        data_root: Root directory for fMoW images. Manifest paths are relative
            to this directory.
        img_size: Images / cells are resized to ``(img_size, img_size)``.
        split: ``'train'`` or ``'val'`` — controls augmentation.
        val_ratio: Fraction of images held out for validation.
        num_views: Number of augmented views per cell (default 2).
        max_samples: Cap the dataset size (for debugging).
    """

    def __init__(
        self,
        manifest_path: str,
        histogram_dir: str,
        cell_labels_path: Optional[str] = None,
        data_root: str = "data/fmow",
        img_size: int = 512,
        split: str = "train",
        val_ratio: float = 0.1,
        num_views: int = 2,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.split = split
        self.num_views = num_views
        self.data_root = Path(data_root)

        manifest_path = Path(manifest_path)
        histogram_dir = Path(histogram_dir)

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. "
                f"Run cluster_viz.py --save-cluster-data first."
            )

        # Load manifest
        with open(manifest_path) as f:
            manifest_data = json.load(f)

        # Handle both manifest formats
        if isinstance(manifest_data, list):
            # Direct list format (from fmow_manifest_train.json)
            all_cells = manifest_data
        elif isinstance(manifest_data, dict) and "cells" in manifest_data:
            # Wrapped format (from cluster_viz.py)
            all_cells = manifest_data["cells"]
        else:
            raise ValueError(f"Unknown manifest format: {type(manifest_data)}")

        # Load histogram targets
        histograms_path = histogram_dir / "histograms.npy"
        cell_ids_path = histogram_dir / "cell_ids.npy"

        if not histograms_path.exists():
            raise FileNotFoundError(
                f"Histograms not found: {histograms_path}. "
                f"Run generate_histograms.py first."
            )

        # Load as memory-mapped for efficiency
        self.histograms = np.load(histograms_path, mmap_mode="r")
        self.cell_ids = np.load(cell_ids_path)
        self.vocab_size = self.histograms.shape[1]

        # Build map from manifest index to histogram index
        self._cell_id_to_hist_idx: Dict[int, int] = {
            int(cell_id): idx for idx, cell_id in enumerate(self.cell_ids)
        }

        # Load cell labels (optional)
        self._all_labels = None
        if cell_labels_path is not None:
            cell_labels_path = Path(cell_labels_path)
            if cell_labels_path.exists():
                self._all_labels = np.load(cell_labels_path, mmap_mode="r")
            else:
                logger.warning(f"Cell labels not found: {cell_labels_path}")

        # Known manifest path prefixes that need to be stripped
        self._manifest_prefixes = [
            "Hosted-Datasets/fmow/fmow-rgb/",
            "Hosted-Datasets/fmow/fmow-rgb-prepped/",
        ]

        # Filter to cells with histogram targets and do train/val split
        train_cells, val_cells = [], []
        for global_idx, cell in enumerate(all_cells):
            # Check if this cell has a histogram
            if global_idx not in self._cell_id_to_hist_idx:
                continue

            # Get image path
            raw_path = cell.get("img_path", cell.get("image_path", ""))
            if not raw_path:
                continue

            # Store necessary metadata
            cell_data = {
                "_global_idx": global_idx,
                "raw_path": raw_path,
            }

            # Deterministic train/val split based on image path hash
            h = int(hashlib.md5(raw_path.encode()).hexdigest(), 16) % 10000
            if h < int(val_ratio * 10000):
                val_cells.append(cell_data)
            else:
                train_cells.append(cell_data)

        self.cells = train_cells if split == "train" else val_cells

        if max_samples is not None:
            self.cells = self.cells[:max_samples]

        print(
            f"FMoWBoVWDataset({split}): {len(self.cells)} cells "
            f"(vocab_size={self.vocab_size}, num_views={self.num_views}, "
            f"labels={'yes' if self._all_labels is not None else 'no'})"
        )

    def _get_image_path(self, raw_path: str) -> Path:
        """Convert manifest path to actual file path."""
        rel_path = raw_path
        for prefix in self._manifest_prefixes:
            if raw_path.startswith(prefix):
                rel_path = raw_path[len(prefix):]
                break
        return self.data_root / rel_path

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, idx: int) -> dict:
        cell = self.cells[idx]
        global_idx = cell["_global_idx"]

        # Get image path
        img_path = self._get_image_path(cell["raw_path"])

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            # Return a blank image
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

        # Generate views
        is_train = self.split == "train"
        if self.num_views > 1:
            img_tensors = _generate_views(img, self.img_size, self.num_views, is_train)
        else:
            if is_train:
                img_tensors = _train_transform(img, self.img_size)
            else:
                img_tensors = _val_transform(img, self.img_size)

        # Get histogram target
        hist_idx = self._cell_id_to_hist_idx[global_idx]
        histogram = torch.from_numpy(
            self.histograms[hist_idx].copy()
        ).float()

        # Get dominant label
        if self._all_labels is not None and global_idx < len(self._all_labels):
            dominant_label = int(self._all_labels[global_idx])
        else:
            dominant_label = -1

        # Pack into data sample
        data_sample = BoVWDataSample()
        data_sample.histogram_target = histogram
        data_sample.dominant_label = dominant_label
        data_sample.image_id = global_idx

        return {"inputs": img_tensors, "data_samples": data_sample}

    @property
    def metainfo(self) -> dict:
        return {
            "dataset_type": "FMoWBoVWDataset",
            "vocab_size": self.vocab_size,
            "num_views": self.num_views,
        }


# --------------------------------------------------------------------------- #
# Multi-view collate function
# --------------------------------------------------------------------------- #

def bovw_collate_fn(batch: List[dict]) -> dict:
    """Collate function for BoVW datasets.

    Handles both single-view and multi-view inputs.

    Args:
        batch: List of dicts with 'inputs' and 'data_samples' keys.

    Returns:
        dict with:
            'inputs': stacked tensor (single-view) or list of stacked tensors (multi-view)
            'data_samples': list of BoVWDataSample objects
    """
    data_samples = [item["data_samples"] for item in batch]

    first_inputs = batch[0]["inputs"]
    is_multiview = isinstance(first_inputs, list)

    if is_multiview:
        num_views = len(first_inputs)
        inputs_list = []

        for view_idx in range(num_views):
            view_tensors = [item["inputs"][view_idx] for item in batch]
            inputs_list.append(torch.stack(view_tensors, dim=0))

        return {"inputs": inputs_list, "data_samples": data_samples}
    else:
        inputs = torch.stack([item["inputs"] for item in batch], dim=0)
        return {"inputs": inputs, "data_samples": data_samples}
