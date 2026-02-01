#!/usr/bin/env python3
"""
Download fMoW dataset from S3 to local storage.

This script downloads the fMoW-rgb dataset from the SpaceNet public S3 bucket.
By default, it downloads the smaller msrgb images (~35GB), but can optionally
download the full rgb images (~350GB).

Usage:
    # Download msrgb images (recommended, ~35GB)
    python scripts/download_fmow.py --output-dir /path/to/fmow --split train
    
    # Download full rgb images (~350GB) 
    python scripts/download_fmow.py --output-dir /path/to/fmow --split train --use-rgb
    
    # Download both train and val
    python scripts/download_fmow.py --output-dir /path/to/fmow --split all

msrgb vs rgb:
    - rgb: Full satellite images (~350GB total)
    - msrgb: Cropped regions around bounding boxes (~35GB total)
    
    For DynamicVis training, msrgb is sufficient and recommended since the model
    only uses the bounding box regions anyway.
"""

import argparse
import bz2
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm


def create_s3_client():
    """Create an S3 client for the public bucket (no credentials needed)."""
    config = Config(
        signature_version=UNSIGNED,
        max_pool_connections=100,
        connect_timeout=30,
        read_timeout=60,
        retries={
            'max_attempts': 10,
            'mode': 'adaptive'
        }
    )
    return boto3.client('s3', config=config, region_name='us-east-1')


def download_manifest(s3_client, bucket: str, s3_prefix: str, output_dir: Path) -> List[str]:
    """Download and parse the manifest file."""
    manifest_key = f"{s3_prefix}/manifest.json.bz2"
    local_manifest = output_dir / "manifest.json.bz2"
    
    if not local_manifest.exists():
        print(f"Downloading manifest from s3://{bucket}/{manifest_key}")
        local_manifest.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, manifest_key, str(local_manifest))
    else:
        print(f"Using cached manifest: {local_manifest}")
    
    with bz2.BZ2File(local_manifest, 'rb') as f:
        return json.load(f)


def filter_files(
    all_files: List[str], 
    splits: List[str], 
    use_msrgb: bool = True
) -> List[str]:
    """Filter files by split and image type."""
    suffix = '_msrgb.jpg' if use_msrgb else '_rgb.jpg'
    
    filtered = []
    for file_path in all_files:
        parts = file_path.split('/')
        if len(parts) < 2:
            continue
        
        split_dir = parts[0]
        if split_dir not in splits:
            continue
        
        # Include images and their JSON annotations
        if file_path.endswith(suffix) or file_path.endswith('.json'):
            # Only include JSON that matches our image type
            if file_path.endswith('.json'):
                # Check if corresponding image would be included
                base = file_path.replace('.json', '')
                img_would_exist = any(
                    base.endswith(s.replace('.jpg', '')) 
                    for s in [suffix]
                )
                if not (base + suffix.replace('.jpg', '')).endswith('_msrgb' if use_msrgb else '_rgb'):
                    continue
            filtered.append(file_path)
    
    return filtered


def download_file(s3_client, bucket: str, s3_prefix: str, file_path: str, output_dir: Path) -> bool:
    """Download a single file from S3."""
    s3_key = f"{s3_prefix}/{file_path}"
    local_path = output_dir / file_path
    
    if local_path.exists():
        return True  # Already downloaded
    
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, s3_key, str(local_path))
        return True
    except Exception as e:
        print(f"\nError downloading {file_path}: {e}")
        return False


def download_dataset(
    output_dir: str,
    splits: List[str],
    use_msrgb: bool = True,
    num_workers: int = 32,
    max_files: Optional[int] = None,
):
    """Download the fMoW dataset."""
    bucket = "spacenet-dataset"
    s3_prefix = "Hosted-Datasets/fmow/fmow-rgb"
    output_path = Path(output_dir)
    
    print(f"Output directory: {output_path}")
    print(f"Splits: {splits}")
    print(f"Image type: {'msrgb (~35GB)' if use_msrgb else 'rgb (~350GB)'}")
    print(f"Workers: {num_workers}")
    print()
    
    # Create S3 client
    s3_client = create_s3_client()
    
    # Download manifest
    all_files = download_manifest(s3_client, bucket, s3_prefix, output_path)
    print(f"Total files in manifest: {len(all_files)}")
    
    # Filter files
    files_to_download = filter_files(all_files, splits, use_msrgb)
    
    # We need both images and their JSON annotations
    # Add JSON files for each image
    image_files = [f for f in files_to_download if f.endswith('.jpg')]
    json_files = [f.replace('.jpg', '.json') for f in image_files]
    files_to_download = image_files + json_files
    
    # Remove duplicates and sort
    files_to_download = sorted(set(files_to_download))
    
    if max_files:
        # Take equal parts images and jsons
        image_files = [f for f in files_to_download if f.endswith('.jpg')][:max_files]
        json_files = [f.replace('.jpg', '.json') for f in image_files]
        files_to_download = image_files + json_files
    
    print(f"Files to download: {len(files_to_download)}")
    
    # Check what's already downloaded
    already_downloaded = 0
    to_download = []
    for f in files_to_download:
        local_path = output_path / f
        if local_path.exists():
            already_downloaded += 1
        else:
            to_download.append(f)
    
    print(f"Already downloaded: {already_downloaded}")
    print(f"Remaining to download: {len(to_download)}")
    
    if not to_download:
        print("All files already downloaded!")
        return
    
    # Estimate size
    avg_size_msrgb = 50 * 1024  # ~50KB per msrgb image
    avg_size_rgb = 500 * 1024   # ~500KB per rgb image
    avg_size = avg_size_msrgb if use_msrgb else avg_size_rgb
    estimated_size_gb = (len(to_download) * avg_size) / (1024**3)
    print(f"Estimated download size: ~{estimated_size_gb:.1f} GB")
    print()
    
    # Confirm
    response = input("Proceed with download? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Download with progress bar
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_file, s3_client, bucket, s3_prefix, f, output_path): f
            for f in to_download
        }
        
        with tqdm(total=len(to_download), desc="Downloading") as pbar:
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                else:
                    fail_count += 1
                pbar.update(1)
    
    print()
    print(f"Download complete!")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total files: {already_downloaded + success_count}")
    
    # Print usage instructions
    print()
    print("=" * 60)
    print("To use the downloaded dataset, update your config:")
    print()
    print(f"    data_root = '{output_path}'")
    print()
    print("Or use the --data-root argument when training:")
    print()
    print(f"    python train_dynamicvis_pretrain.py --data-root {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download fMoW dataset from S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./data/fmow',
        help='Local directory to save the dataset (default: ./data/fmow)'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test', 'all'],
        default='all',
        help='Which split(s) to download (default: all)'
    )
    parser.add_argument(
        '--use-rgb',
        action='store_true',
        help='Download full RGB images (~350GB) instead of msrgb (~35GB)'
    )
    parser.add_argument(
        '--workers', '-j',
        type=int,
        default=32,
        help='Number of parallel download workers (default: 32)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        help='Limit number of files to download (for testing)'
    )
    
    args = parser.parse_args()
    
    # Determine splits
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]
    
    download_dataset(
        output_dir=args.output_dir,
        splits=splits,
        use_msrgb=not args.use_rgb,
        num_workers=args.workers,
        max_files=args.max_files,
    )


if __name__ == '__main__':
    main()
