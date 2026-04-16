#!/usr/bin/env python3
"""Assign fMoW category labels to composition-training grid cells.

Reads the composition-training manifest (``outputs/cluster_data/manifest.json``)
and the fMoW JSON sidecar annotations to compute, for every grid cell:

* **dominant_label** — the single most-overlapping fMoW category (``argmax`` of
  IoU-weighted counts), or ``-1`` when no annotation overlaps the cell above a
  minimum IoU threshold.
* **multi_hot** — a ``(63,)`` float32 vector with 1.0 for every category whose
  cumulative IoU with the cell exceeds a threshold.

Outputs are saved to the same ``cluster_data`` directory so the composition
dataset can pick them up:

    cell_labels.npy        (N_cells,)     int64   dominant label per cell
    cell_multilabels.npy   (N_cells, 63)  float32 multi-hot labels per cell

Usage::

    python scripts/assign_cell_labels.py \
        --cluster-data-dir outputs/cluster_data \
        --iou-threshold 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------------------
# fMoW category list — must match the order used by the rest of the codebase.
# (Reverse alphabetical, zoo=0 … false_detection=62.)
# ---------------------------------------------------------------------------
FMOW_CATEGORIES = [
    "zoo", "wind_farm", "water_treatment_facility", "waste_disposal",
    "tunnel_opening", "tower", "toll_booth", "swimming_pool", "surface_mine",
    "storage_tank", "stadium", "space_facility", "solar_farm", "smokestack",
    "single-unit_residential", "shopping_mall", "shipyard", "runway",
    "road_bridge", "recreational_facility", "railway_bridge", "race_track",
    "prison", "port", "police_station", "place_of_worship",
    "parking_lot_or_garage", "park", "oil_or_gas_facility", "office_building",
    "nuclear_powerplant", "multi-unit_residential", "military_facility",
    "lighthouse", "lake_or_pond", "interchange", "impoverished_settlement",
    "hospital", "helipad", "ground_transportation_station", "golf_course",
    "gas_station", "fountain", "flooded_road", "fire_station",
    "factory_or_powerplant", "electric_substation", "educational_institution",
    "debris_or_rubble", "dam", "crop_field", "construction_site",
    "car_dealership", "burial_site", "border_checkpoint", "barn",
    "archaeological_site", "aquaculture", "amusement_park", "airport_terminal",
    "airport_hangar", "airport", "false_detection",
]
NUM_CLASSES = len(FMOW_CATEGORIES)  # 63
CAT2LABEL: Dict[str, int] = {cat: i for i, cat in enumerate(FMOW_CATEGORIES)}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iou(box_a: Tuple[float, float, float, float],
         box_b: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two (x0, y0, x1, y1) boxes."""
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _resize_dims(orig_w: int, orig_h: int, grid_size: int) -> Tuple[int, int]:
    """Mirror the ``resize_to_grid`` logic from pipeline_utils."""
    new_w = max(grid_size, round(orig_w / grid_size) * grid_size)
    new_h = max(grid_size, round(orig_h / grid_size) * grid_size)
    return new_w, new_h


# ---------------------------------------------------------------------------
# Annotation loading + caching
# ---------------------------------------------------------------------------

def _load_annotations(json_path: str) -> Optional[dict]:
    """Load an fMoW JSON sidecar, or return None if not found."""
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


def _parse_bboxes_rescaled(
    ann: dict,
    orig_w: int,
    orig_h: int,
    grid_size: int,
) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    """Parse bboxes from annotation dict, rescale to resized-image coords.

    Returns list of (label_id, (x0, y0, x1, y1)) in resized pixel space.
    """
    new_w, new_h = _resize_dims(orig_w, orig_h, grid_size)
    sx = new_w / orig_w
    sy = new_h / orig_h

    result = []
    for bb in ann.get("bounding_boxes", []):
        cat = bb.get("category", "")
        if cat not in CAT2LABEL:
            continue
        label_id = CAT2LABEL[cat]
        # fMoW box format: [x, y, w, h]
        bx, by, bw, bh = bb["box"]
        x0 = bx * sx
        y0 = by * sy
        x1 = (bx + bw) * sx
        y1 = (by + bh) * sy
        result.append((label_id, (x0, y0, x1, y1)))
    return result


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def assign_labels(
    manifest: dict,
    grid_size: int,
    iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign dominant label and multi-hot labels to every cell.

    Returns:
        dominant_labels: (N,) int64 array. -1 = unlabeled / background.
        multi_hot:       (N, 63) float32 array.
    """
    cells = manifest["cells"]
    n_cells = len(cells)

    dominant_labels = np.full(n_cells, -1, dtype=np.int64)
    multi_hot = np.zeros((n_cells, NUM_CLASSES), dtype=np.float32)

    # Group cells by image path for efficient annotation loading
    image_to_cell_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, cell in enumerate(cells):
        image_to_cell_indices[cell["image_path"]].append(idx)

    n_images = len(image_to_cell_indices)
    n_labeled = 0
    n_no_json = 0

    for img_idx, (image_path, cell_indices) in enumerate(
        sorted(image_to_cell_indices.items())
    ):
        if (img_idx + 1) % 5000 == 0 or img_idx + 1 == n_images:
            print(
                f"  [{img_idx + 1}/{n_images}] images processed, "
                f"{n_labeled} labeled cells so far"
            )

        # Derive JSON sidecar path from image path
        json_path = image_path.replace("_rgb.jpg", "_rgb.json")
        ann = _load_annotations(json_path)
        if ann is None:
            n_no_json += 1
            continue

        # Get original image dimensions from annotation
        orig_w = ann.get("img_width")
        orig_h = ann.get("img_height")
        if orig_w is None or orig_h is None:
            # Fall back: open image to get size (slow but rare)
            try:
                with Image.open(image_path) as img:
                    orig_w, orig_h = img.size
            except Exception:
                n_no_json += 1
                continue

        # Parse and rescale bounding boxes
        bboxes = _parse_bboxes_rescaled(ann, orig_w, orig_h, grid_size)
        if not bboxes:
            continue

        # Assign labels for each cell of this image
        for cell_idx in cell_indices:
            cell = cells[cell_idx]
            cell_box = (
                float(cell["cell_x0"]),
                float(cell["cell_y0"]),
                float(cell["cell_x1"]),
                float(cell["cell_y1"]),
            )

            # Accumulate IoU-weighted label counts
            label_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
            max_iou = 0.0

            for label_id, bbox in bboxes:
                iou = _iou(bbox, cell_box)
                if iou > 0:
                    label_counts[label_id] += iou
                    max_iou = max(max_iou, iou)

            if max_iou >= iou_threshold:
                dominant_labels[cell_idx] = int(np.argmax(label_counts))
                n_labeled += 1

            # Multi-hot: any class with cumulative IoU above threshold
            multi_hot[cell_idx] = (label_counts > iou_threshold).astype(
                np.float32
            )

    print(f"\nSummary:")
    print(f"  Total cells:         {n_cells}")
    print(f"  Labeled cells:       {n_labeled} ({100*n_labeled/n_cells:.1f}%)")
    print(f"  Unlabeled cells:     {n_cells - n_labeled} ({100*(n_cells-n_labeled)/n_cells:.1f}%)")
    print(f"  Images without JSON: {n_no_json}")

    # Per-class breakdown
    print(f"\n  Per-class breakdown (dominant labels):")
    for c in range(NUM_CLASSES):
        count = int((dominant_labels == c).sum())
        if count > 0:
            print(f"    [{c:2d}] {FMOW_CATEGORIES[c]:40s} {count:6d}")

    return dominant_labels, multi_hot


def main():
    parser = argparse.ArgumentParser(
        description="Assign fMoW category labels to composition cells."
    )
    parser.add_argument(
        "--cluster-data-dir",
        type=str,
        default="outputs/cluster_data",
        help="Directory with manifest.json (and where outputs are saved).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.05,
        help="Minimum IoU for a bbox to contribute to a cell's label.",
    )
    args = parser.parse_args()

    cluster_dir = Path(args.cluster_data_dir)
    manifest_path = cluster_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        sys.exit(1)

    print(f"Loading manifest from {manifest_path} ...")
    with open(manifest_path) as f:
        manifest = json.load(f)
    grid_size = manifest["grid_size"]
    print(
        f"  {manifest['n_cells']} cells from {manifest['n_images']} images, "
        f"grid_size={grid_size}"
    )

    print(f"\nAssigning labels (IoU threshold={args.iou_threshold}) ...")
    dominant_labels, multi_hot = assign_labels(
        manifest, grid_size, args.iou_threshold
    )

    # Save outputs
    labels_path = cluster_dir / "cell_labels.npy"
    multilabels_path = cluster_dir / "cell_multilabels.npy"
    np.save(labels_path, dominant_labels)
    np.save(multilabels_path, multi_hot)
    print(f"\nSaved:")
    print(f"  {labels_path}  shape={dominant_labels.shape}  dtype={dominant_labels.dtype}")
    print(f"  {multilabels_path}  shape={multi_hot.shape}  dtype={multi_hot.dtype}")


if __name__ == "__main__":
    main()
