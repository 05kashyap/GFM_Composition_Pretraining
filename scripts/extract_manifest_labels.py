#!/usr/bin/env python3
"""Extract fMoW class labels from manifest image paths.

For the BoVW pipeline, this creates cell_labels.npy by parsing the class name
directly from the image path structure.

Path format: train/<category>/<category>_<id>/<category>_<id>_<seq>_rgb.jpg

Usage:
    python scripts/extract_manifest_labels.py \
        --manifest data/fmow_manifest_train.json \
        --output-dir outputs/bovw_histograms

The output cell_labels.npy will be indexed by manifest order (same as histograms).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# fMoW category list — same order as assign_cell_labels.py
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
CAT2LABEL = {cat: i for i, cat in enumerate(FMOW_CATEGORIES)}


def extract_category_from_path(img_path: str) -> str | None:
    """Extract fMoW category from image path.

    Expected formats:
        train/<category>/<category>_<id>/<category>_<id>_<seq>_rgb.jpg
        Hosted-Datasets/fmow/fmow-rgb/train/<category>/<category>_<id>/...
    """
    # Try to find category in path
    # Pattern: match the category folder name after train/ or val/
    match = re.search(r'(?:train|val)/([^/]+)/', img_path)
    if match:
        return match.group(1)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract fMoW labels from manifest paths"
    )
    parser.add_argument(
        "--manifest", type=str, default="data/fmow_manifest_train.json",
        help="Path to manifest.json"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/bovw_histograms",
        help="Output directory for cell_labels.npy"
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest from {manifest_path}...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Handle list or dict format
    if isinstance(manifest, list):
        cells = manifest
    elif isinstance(manifest, dict) and "cells" in manifest:
        cells = manifest["cells"]
    else:
        cells = manifest

    n_cells = len(cells)
    print(f"  Found {n_cells} cells")

    # Extract labels
    labels = np.full(n_cells, -1, dtype=np.int64)
    label_counts = {cat: 0 for cat in FMOW_CATEGORIES}
    n_labeled = 0
    n_unknown_category = 0

    for idx, cell in enumerate(cells):
        img_path = cell.get("img_path", cell.get("image_path", ""))
        if not img_path:
            continue

        category = extract_category_from_path(img_path)
        if category is None:
            continue

        if category in CAT2LABEL:
            label_id = CAT2LABEL[category]
            labels[idx] = label_id
            label_counts[category] += 1
            n_labeled += 1
        else:
            n_unknown_category += 1
            if n_unknown_category <= 5:
                print(f"  Warning: Unknown category '{category}' in {img_path}")

    # Summary
    print(f"\nSummary:")
    print(f"  Total cells:      {n_cells}")
    print(f"  Labeled cells:    {n_labeled} ({100*n_labeled/n_cells:.1f}%)")
    print(f"  Unlabeled cells:  {n_cells - n_labeled}")
    if n_unknown_category > 0:
        print(f"  Unknown category: {n_unknown_category}")

    # Per-class breakdown
    print(f"\nPer-class breakdown:")
    for cat in FMOW_CATEGORIES:
        count = label_counts[cat]
        if count > 0:
            label_id = CAT2LABEL[cat]
            print(f"  [{label_id:2d}] {cat:40s} {count:6d}")

    # Save
    output_path = output_dir / "cell_labels.npy"
    np.save(output_path, labels)
    print(f"\nSaved: {output_path} (shape={labels.shape}, dtype={labels.dtype})")

    # Verify compatibility with histograms if they exist
    histograms_path = output_dir / "histograms.npy"
    if histograms_path.exists():
        histograms = np.load(histograms_path, mmap_mode="r")
        if histograms.shape[0] == n_cells:
            print(f"  ✓ Compatible with histograms.npy ({histograms.shape[0]} entries)")
        else:
            print(f"  ⚠ Warning: histogram count ({histograms.shape[0]}) != manifest ({n_cells})")


if __name__ == "__main__":
    main()
