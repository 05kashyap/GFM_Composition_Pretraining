#!/usr/bin/env python3
"""
Filter fMoW manifest to only include files that exist on disk.

This script reads the training manifest and checks which image files
actually exist in the data directory, outputting a filtered manifest.

Usage:
    python scripts/filter_manifest.py
    python scripts/filter_manifest.py --data-root /path/to/fmow
    python scripts/filter_manifest.py --output data/fmow_manifest_train_filtered.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# Known manifest path prefixes that need to be stripped
MANIFEST_PREFIXES = [
    "Hosted-Datasets/fmow/fmow-rgb/",
    "Hosted-Datasets/fmow/fmow-rgb-prepped/",
]


def strip_prefix(path: str) -> str:
    """Strip known manifest prefixes from path."""
    for prefix in MANIFEST_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Filter fMoW manifest to only include existing files"
    )
    parser.add_argument(
        "--manifest", type=str, default="data/fmow_manifest_train.json",
        help="Path to input manifest (default: data/fmow_manifest_train.json)"
    )
    parser.add_argument(
        "--data-root", type=str, default="data/fmow",
        help="Root directory containing fMoW images (default: data/fmow)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for filtered manifest (default: overwrite input)"
    )
    parser.add_argument(
        "--backup", action="store_true",
        help="Create backup of original manifest before overwriting"
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    data_root = Path(args.data_root)
    output_path = Path(args.output) if args.output else manifest_path

    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        return 1

    if not data_root.exists():
        print(f"Error: Data root not found: {data_root}")
        return 1

    # Load manifest
    print(f"Loading manifest: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    print(f"Total entries in manifest: {len(manifest)}")

    # Filter to existing files
    filtered = []
    missing = 0
    class_counts = defaultdict(lambda: {"found": 0, "missing": 0})

    print(f"Checking file existence in: {data_root}")
    for entry in tqdm(manifest, desc="Filtering"):
        img_path = entry["img_path"]
        rel_path = strip_prefix(img_path)
        full_path = data_root / rel_path

        # Extract class from path (e.g., train/airport/... -> airport)
        parts = rel_path.split("/")
        class_name = parts[1] if len(parts) >= 2 else "unknown"

        if full_path.exists():
            filtered.append(entry)
            class_counts[class_name]["found"] += 1
        else:
            missing += 1
            class_counts[class_name]["missing"] += 1

    # Print summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Original entries:  {len(manifest)}")
    print(f"  Found on disk:     {len(filtered)}")
    print(f"  Missing:           {missing}")
    print(f"  Coverage:          {100 * len(filtered) / len(manifest):.1f}%")
    print()

    # Print per-class breakdown
    print("Per-class breakdown:")
    for class_name in sorted(class_counts.keys()):
        counts = class_counts[class_name]
        total = counts["found"] + counts["missing"]
        pct = 100 * counts["found"] / total if total > 0 else 0
        print(f"  {class_name:40s}: {counts['found']:6d} / {total:6d} ({pct:5.1f}%)")

    print()

    # Backup original if requested
    if args.backup and output_path == manifest_path:
        backup_path = manifest_path.with_suffix(".json.bak")
        print(f"Creating backup: {backup_path}")
        with open(backup_path, "w") as f:
            json.dump(manifest, f)

    # Write filtered manifest
    print(f"Writing filtered manifest: {output_path}")
    with open(output_path, "w") as f:
        json.dump(filtered, f)

    print()
    print("Done!")
    print(f"Filtered manifest has {len(filtered)} entries")
    print()
    print("You can now use --data-fraction on the filtered manifest:")
    print(f"  sbatch run_gpu_extract_patch_tokens.sh --data-fraction 0.5")
    print(f"  (This will process 50% of your {len(filtered)} downloaded images)")

    return 0


if __name__ == "__main__":
    exit(main())
