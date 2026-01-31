"""
fMoW S3 Streaming Dataset for MMPretrain/DynamicVis.
Streams images directly from AWS S3 without downloading the entire dataset.
Compatible with MMEngine's training framework.

Optimized for GPU utilization with:
- Connection pooling and persistent HTTP sessions
- Async prefetching with thread pool
- Smart retry logic with exponential backoff
- Configurable prefetch buffer
"""

import os
import io
import bz2
import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
import boto3
from botocore.config import Config
from botocore import UNSIGNED
from tqdm import tqdm

from mmengine.dataset import BaseDataset, Compose
from mmpretrain.registry import DATASETS, TRANSFORMS

# Allow large images
Image.MAX_IMAGE_PIXELS = None


def create_optimized_s3_client(use_credentials: bool = True):
    """Create an S3 client optimized for high-throughput streaming.
    
    Uses connection pooling and retry configuration for better performance.
    """
    # Optimized boto3 config for streaming
    config = Config(
        signature_version=UNSIGNED if not use_credentials else None,
        max_pool_connections=50,  # Increase connection pool for parallel fetches
        connect_timeout=10,
        read_timeout=30,
        retries={
            'max_attempts': 3,
            'mode': 'adaptive'  # Adaptive retry mode with backoff
        }
    )
    
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    
    if use_credentials and os.getenv("AWS_ACCESS_KEY_ID"):
        return boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region,
            config=config,
        )
    else:
        # Use unsigned requests for public buckets
        unsigned_config = Config(
            signature_version=UNSIGNED,
            max_pool_connections=50,
            connect_timeout=10,
            read_timeout=30,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        return boto3.client("s3", config=unsigned_config, region_name=region)


class S3ImagePrefetcher:
    """Asynchronous image prefetcher for S3 streaming.
    
    Maintains a buffer of prefetched images to minimize GPU idle time.
    Uses a thread pool to fetch images in parallel.
    """
    
    def __init__(
        self,
        s3_client,
        bucket: str,
        prefetch_size: int = 64,
        num_prefetch_workers: int = 8,
    ):
        self.s3_client = s3_client
        self.bucket = bucket
        self.prefetch_size = prefetch_size
        self.num_workers = num_prefetch_workers
        
        # Prefetch buffer: s3_key -> image_bytes
        self.buffer: Dict[str, bytes] = {}
        self.buffer_lock = threading.Lock()
        
        # Pending prefetch requests
        self.pending_keys: set = set()
        self.pending_lock = threading.Lock()
        
        # Thread pool for parallel prefetching
        self.executor = ThreadPoolExecutor(max_workers=num_prefetch_workers)
        
        # Stats for monitoring
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _fetch_single(self, s3_key: str) -> Optional[bytes]:
        """Fetch a single image from S3."""
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            return obj['Body'].read()
        except Exception as e:
            print(f"Prefetch error for {s3_key}: {e}")
            return None
    
    def prefetch(self, s3_keys: List[str]):
        """Submit keys for prefetching in background."""
        keys_to_fetch = []
        
        with self.buffer_lock:
            with self.pending_lock:
                for key in s3_keys[:self.prefetch_size]:
                    if key not in self.buffer and key not in self.pending_keys:
                        keys_to_fetch.append(key)
                        self.pending_keys.add(key)
        
        # Submit fetch tasks
        for key in keys_to_fetch:
            future = self.executor.submit(self._prefetch_and_store, key)
    
    def _prefetch_and_store(self, s3_key: str):
        """Fetch and store in buffer."""
        img_bytes = self._fetch_single(s3_key)
        
        with self.buffer_lock:
            if img_bytes is not None:
                # Evict old entries if buffer is full
                if len(self.buffer) >= self.prefetch_size:
                    # Remove oldest entry (FIFO-ish)
                    oldest_key = next(iter(self.buffer))
                    del self.buffer[oldest_key]
                self.buffer[s3_key] = img_bytes
        
        with self.pending_lock:
            self.pending_keys.discard(s3_key)
    
    def get(self, s3_key: str) -> Optional[bytes]:
        """Get image bytes, from buffer if available, otherwise fetch directly."""
        # Check buffer first
        with self.buffer_lock:
            if s3_key in self.buffer:
                self.cache_hits += 1
                return self.buffer.pop(s3_key)  # Remove from buffer after use
        
        # Cache miss - fetch directly
        self.cache_misses += 1
        return self._fetch_single(s3_key)
    
    def get_stats(self) -> Dict[str, int]:
        """Get prefetcher statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'buffer_size': len(self.buffer),
        }
    
    def shutdown(self):
        """Shutdown the prefetcher."""
        self.executor.shutdown(wait=False)


# fMoW categories (63 classes including false_detection)
FMOW_CATEGORIES = [
    "zoo", "wind_farm", "water_treatment_facility", "waste_disposal", "tunnel_opening",
    "tower", "toll_booth", "swimming_pool", "surface_mine", "storage_tank", "stadium",
    "space_facility", "solar_farm", "smokestack", "single-unit_residential", "shopping_mall",
    "shipyard", "runway", "road_bridge", "recreational_facility", "railway_bridge",
    "race_track", "prison", "port", "police_station", "place_of_worship",
    "parking_lot_or_garage", "park", "oil_or_gas_facility", "office_building",
    "nuclear_powerplant", "multi-unit_residential", "military_facility", "lighthouse",
    "lake_or_pond", "interchange", "impoverished_settlement", "hospital", "helipad",
    "ground_transportation_station", "golf_course", "gas_station", "fountain",
    "flooded_road", "fire_station", "factory_or_powerplant", "electric_substation",
    "educational_institution", "debris_or_rubble", "dam", "crop_field", "construction_site",
    "car_dealership", "burial_site", "border_checkpoint", "barn", "archaeological_site",
    "aquaculture", "amusement_park", "airport_terminal", "airport_hangar", "airport",
    "false_detection"
]


@TRANSFORMS.register_module()
class LoadImageFromS3:
    """Load an image from S3 bucket with prefetching support.
    
    Supports both direct fetch and prefetcher-based fetch for better
    GPU utilization.
    
    Required Keys:
        - s3_key
        - s3_client OR prefetcher
        - bucket
    
    Modified Keys:
        - img
        - img_shape
        - ori_shape
    """
    
    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = 'color',
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _load_with_retry(self, s3_client, bucket: str, s3_key: str) -> Optional[bytes]:
        """Load image bytes with retry logic."""
        for attempt in range(self.max_retries):
            try:
                obj = s3_client.get_object(Bucket=bucket, Key=s3_key)
                return obj['Body'].read()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise e
        return None
    
    def __call__(self, results: Dict) -> Optional[Dict]:
        bucket = results['bucket']
        s3_key = results['s3_key']
        
        try:
            # Try prefetcher first if available
            if 'prefetcher' in results and results['prefetcher'] is not None:
                img_bytes = results['prefetcher'].get(s3_key)
            else:
                # Fall back to direct fetch with retry
                s3_client = results['s3_client']
                img_bytes = self._load_with_retry(s3_client, bucket, s3_key)
            
            if img_bytes is None:
                return None
            
            img = Image.open(io.BytesIO(img_bytes))
            
            if self.color_type == 'color':
                img = img.convert('RGB')
            
            img = np.array(img)
            
            if self.to_float32:
                img = img.astype(np.float32)
            
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = img.shape[:2]
            
            return results
            
        except Exception as e:
            print(f"Error loading image from S3 {s3_key}: {e}")
            return None
    
    def __repr__(self):
        return f'{self.__class__.__name__}(to_float32={self.to_float32})'


@DATASETS.register_module()
class FMoWS3Dataset(BaseDataset):
    """fMoW dataset that streams from AWS S3.
    
    This dataset streams images directly from AWS S3 without downloading
    the entire dataset locally. It's compatible with MMEngine's training framework.
    
    Args:
        bucket: S3 bucket name
        s3_prefix: Prefix path within the bucket
        manifest_key: S3 key for the manifest file
        local_manifest: Local path to cache the manifest
        split: Dataset split ('train', 'val', 'test')
        pipeline: Data processing pipeline
        test_mode: Whether in test mode
        enable_prefetch: Enable async prefetching for better GPU utilization
        prefetch_size: Number of images to prefetch ahead
        num_prefetch_workers: Number of threads for prefetching
    """
    
    METAINFO = {
        'classes': FMOW_CATEGORIES,
    }
    
    def __init__(
        self,
        bucket: str = "spacenet-dataset",
        s3_prefix: str = "Hosted-Datasets/fmow/fmow-rgb",
        manifest_key: str = "Hosted-Datasets/fmow/fmow-rgb/manifest.json.bz2",
        local_manifest: str = "data/manifest.json.bz2",
        split: str = "train",
        pipeline: List[Dict] = None,
        test_mode: bool = False,
        enable_prefetch: bool = True,
        prefetch_size: int = 1024,  # Prefetch ~4 batches ahead (for batch_size=256)
        num_prefetch_workers: int = 16,  # More workers for parallel S3 fetches
        **kwargs
    ):
        self.bucket = bucket
        self.s3_prefix = s3_prefix
        self.manifest_key = manifest_key
        self.local_manifest = local_manifest
        self.split = split
        self.enable_prefetch = enable_prefetch
        
        # Setup optimized S3 client
        use_credentials = bool(os.getenv("AWS_ACCESS_KEY_ID"))
        self.s3_client = create_optimized_s3_client(use_credentials)
        
        # Setup prefetcher for async image loading
        self.prefetcher = None
        if enable_prefetch:
            self.prefetcher = S3ImagePrefetcher(
                s3_client=self.s3_client,
                bucket=bucket,
                prefetch_size=prefetch_size,
                num_prefetch_workers=num_prefetch_workers,
            )
        
        # Build class to idx mapping
        self.class_to_idx = {cat: i for i, cat in enumerate(FMOW_CATEGORIES)}
        
        # Track last accessed index for prefetch lookahead
        self._last_idx = 0
        self._prefetch_lookahead = prefetch_size // 2
        
        super().__init__(
            ann_file='',
            metainfo=self.METAINFO,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs
        )
    
    def load_data_list(self) -> List[Dict]:
        """Load dataset annotations from manifest."""
        # Ensure local manifest directory exists
        Path(self.local_manifest).parent.mkdir(parents=True, exist_ok=True)
        
        # Download manifest if not exists
        if not os.path.exists(self.local_manifest):
            print(f"Downloading manifest file {self.manifest_key}...")
            self.s3_client.download_file(self.bucket, self.manifest_key, self.local_manifest)
            print("Manifest download complete.")
        
        print(f"Loading manifest file for '{self.split}' split...")
        
        with bz2.BZ2File(self.local_manifest, "rb") as f:
            manifest_data = json.load(f)
        
        data_list = []
        print(f"Parsing manifest filenames for '{self.split}' split...")
        
        for img_path in tqdm(manifest_data, desc=f"Parsing {self.split}"):
            if not img_path.endswith((".jpg", ".jpeg", ".png")):
                continue
            
            parts = img_path.split("/")
            if len(parts) < 3:
                continue
            
            split_dir = parts[0]
            
            # Only parse files matching requested split
            if split_dir != self.split:
                continue
            
            category = parts[1]
            
            if category not in self.class_to_idx:
                continue
            
            label_idx = self.class_to_idx[category]
            s3_key = f"{self.s3_prefix}/{img_path}"
            
            data_list.append({
                'img_path': img_path,
                's3_key': s3_key,
                'gt_label': label_idx,
            })
        
        print(f"Found {len(data_list)} images for '{self.split}' split.")
        return data_list
    
    def _trigger_prefetch(self, current_idx: int):
        """Trigger prefetching of upcoming images."""
        if not self.enable_prefetch or self.prefetcher is None:
            return
        
        # Prefetch images ahead of current position
        data_list = self.data_list
        num_samples = len(data_list)
        
        # Guard against empty data list
        if num_samples == 0:
            return
        
        # Generate list of upcoming S3 keys to prefetch
        upcoming_keys = []
        for offset in range(1, min(self._prefetch_lookahead + 1, num_samples)):
            future_idx = (current_idx + offset) % num_samples
            upcoming_keys.append(data_list[future_idx]['s3_key'])
        
        # Submit for prefetching
        if upcoming_keys:
            self.prefetcher.prefetch(upcoming_keys)
    
    def prepare_data(self, idx: int) -> Dict:
        """Get data info by index and apply pipeline."""
        data_info = self.get_data_info(idx)
        
        # Add S3 client and bucket to data_info for the pipeline
        data_info['s3_client'] = self.s3_client
        data_info['bucket'] = self.bucket
        data_info['prefetcher'] = self.prefetcher
        
        return self.pipeline(data_info)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get item with error handling and prefetching."""
        # Trigger prefetch for upcoming items
        self._trigger_prefetch(idx)
        
        try:
            data = self.prepare_data(idx)
            if data is None:
                # Return next valid sample
                return self.__getitem__((idx + 1) % len(self))
            return data
        except Exception as e:
            print(f"Error getting item {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))
    
    def get_prefetch_stats(self) -> Dict[str, Any]:
        """Get prefetcher statistics for monitoring."""
        if self.prefetcher is not None:
            return self.prefetcher.get_stats()
        return {}
    
    def __del__(self):
        """Cleanup prefetcher on deletion."""
        if hasattr(self, 'prefetcher') and self.prefetcher is not None:
            self.prefetcher.shutdown()


# For backward compatibility with the existing training script
class FMoWS3DatasetSimple:
    """
    Simple PyTorch Dataset for S3 streaming (non-MMEngine version).
    For use with standard PyTorch DataLoader.
    """
    
    def __init__(
        self,
        bucket: str,
        s3_prefix: str,
        manifest_key: str,
        local_manifest: str,
        split: str = "train",
        transform=None,
    ):
        self.bucket = bucket
        self.s3_prefix = s3_prefix
        self.transform = transform
        self.split = split
        
        # Setup S3 client
        if not os.getenv("AWS_ACCESS_KEY_ID"):
            self.s3_client = boto3.client(
                "s3",
                config=Config(signature_version=UNSIGNED),
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )
        else:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )
        
        # Ensure local manifest directory exists
        Path(local_manifest).parent.mkdir(parents=True, exist_ok=True)
        
        # Download manifest if not exists
        if not os.path.exists(local_manifest):
            print(f"Downloading manifest file {manifest_key}...")
            self.s3_client.download_file(bucket, manifest_key, local_manifest)
            print("Manifest download complete.")
        
        # Parse manifest
        print(f"Loading manifest file for '{split}' split...")
        self.metadata = []
        self.class_to_idx = {cat: i for i, cat in enumerate(FMOW_CATEGORIES)}
        self.idx_to_class = {i: cat for i, cat in enumerate(FMOW_CATEGORIES)}
        
        with bz2.BZ2File(local_manifest, "rb") as f:
            manifest_data = json.load(f)
        
        print(f"Parsing manifest filenames for '{split}' split...")
        
        for img_path in tqdm(manifest_data, desc=f"Parsing {split}"):
            if not img_path.endswith((".jpg", ".jpeg", ".png")):
                continue
            
            parts = img_path.split("/")
            if len(parts) < 3:
                continue
            
            split_dir = parts[0]
            
            if split_dir != split:
                continue
            
            category = parts[1]
            
            if category not in self.class_to_idx:
                continue
            
            label_idx = self.class_to_idx[category]
            self.metadata.append((img_path, label_idx))
        
        print(f"Found {len(self.metadata)} images for '{split}' split.")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        img_path, label = self.metadata[idx]
        full_s3_key = f"{self.s3_prefix}/{img_path}"
        
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket, Key=full_s3_key)
            img_bytes = obj['Body'].read()
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {full_s3_key}: {e}")
            # Return placeholder
            if self.transform:
                return torch.zeros((3, 224, 224)), -1
            return Image.new("RGB", (224, 224)), -1
