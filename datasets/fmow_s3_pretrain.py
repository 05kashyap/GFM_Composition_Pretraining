"""
fMoW S3 Streaming Dataset for DynamicVis Pretraining.

This dataset streams the fMoW-rgb data directly from AWS S3 with bounding box
annotations, matching the format expected by DynamicVis pretrained models.

Key differences from classification-only dataset:
- Returns bounding boxes (gt_bboxes) and their labels (gt_bboxes_labels)
- Uses detection-style data format for DynamicVisPretrainClassifier
- Supports ROI-based feature extraction

The fMoW-rgb dataset is ~350GB. This implementation streams on-demand without
requiring full download.

Usage:
    # In config file:
    train_dataloader = dict(
        dataset=dict(
            type='FMoWS3PretrainDataset',
            bucket='spacenet-dataset',
            s3_prefix='Hosted-Datasets/fmow/fmow-rgb',
            split='train',
            pipeline=train_pipeline,
        ),
    )
"""

import os
import io
import bz2
import json
import queue
import threading
import time
import random
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image
import boto3
from botocore.config import Config
from botocore import UNSIGNED
from tqdm import tqdm
import mmcv

from mmengine.dataset import BaseDataset, Compose

# Register with both mmdet and mmpretrain registries
try:
    from mmdet.registry import DATASETS as DET_DATASETS
    from mmdet.registry import TRANSFORMS as DET_TRANSFORMS
except ImportError:
    DET_DATASETS = None
    DET_TRANSFORMS = None

try:
    from mmpretrain.registry import DATASETS as PRE_DATASETS
    from mmpretrain.registry import TRANSFORMS as PRE_TRANSFORMS
except ImportError:
    PRE_DATASETS = None
    PRE_TRANSFORMS = None

# Allow large images
Image.MAX_IMAGE_PIXELS = None

# fMoW categories (63 classes) - order matches DynamicVis pretrained checkpoint
# The checkpoint uses reverse alphabetical order (zoo=0, airport=60, false_detection=62)
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
    "airport_hangar", "airport", "false_detection"
]


def create_optimized_s3_client(use_credentials: bool = True):
    """Create an S3 client optimized for high-throughput streaming."""
    config = Config(
        signature_version=UNSIGNED if not use_credentials else None,
        max_pool_connections=50,
        connect_timeout=10,
        read_timeout=60,  # Larger images need more time
        retries={
            'max_attempts': 5,
            'mode': 'adaptive'
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
        return boto3.client(
            "s3",
            config=Config(
                signature_version=UNSIGNED,
                max_pool_connections=50,
                connect_timeout=10,
                read_timeout=60,
                retries={'max_attempts': 5, 'mode': 'adaptive'}
            ),
            region_name=region,
        )


# Helper decorator to register in both mmdet and mmpretrain
def register_transform(cls):
    """Register transform in both mmdet and mmpretrain registries."""
    if DET_TRANSFORMS is not None:
        DET_TRANSFORMS.register_module()(cls)
    if PRE_TRANSFORMS is not None:
        PRE_TRANSFORMS.register_module()(cls)
    return cls


def register_dataset(cls):
    """Register dataset in both mmdet and mmpretrain registries."""
    if DET_DATASETS is not None:
        DET_DATASETS.register_module()(cls)
    if PRE_DATASETS is not None:
        PRE_DATASETS.register_module()(cls)
    return cls


@register_transform
class LoadImageFromS3WithBbox:
    """Load image from S3 and process bounding boxes.
    
    This transform loads both the image and its JSON annotation,
    creating gt_bboxes and gt_bboxes_labels for detection-style training.
    """
    
    def __init__(
        self,
        to_float32: bool = True,
        color_type: str = 'color',
        imdecode_backend: str = 'cv2',
        max_edge: int = 1024,  # Resize if larger than this
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.max_edge = max_edge
    
    def __call__(self, results: dict) -> Optional[dict]:
        """Load image and bounding boxes from results dict."""
        try:
            img_bytes = results.get('img_bytes')
            if img_bytes is None:
                return None
            
            # Decode image
            img = mmcv.imfrombytes(
                img_bytes,
                flag=self.color_type,
                backend=self.imdecode_backend
            )
            
            if img is None:
                return None
            
            # Resize if too large (to save memory and bandwidth)
            h, w = img.shape[:2]
            scale = 1.0
            if max(h, w) > self.max_edge:
                scale = self.max_edge / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                img = mmcv.imresize(img, (new_w, new_h))
            
            if self.to_float32:
                img = img.astype(np.float32)
            
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = (h, w)
            results['scale_factor'] = (scale, scale)
            
            # Scale bounding boxes if image was resized
            if scale != 1.0 and 'gt_bboxes' in results:
                results['gt_bboxes'] = results['gt_bboxes'] * scale
            
            return results
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return None


@register_transform
class LoadImageFromImgbytesS3(LoadImageFromS3WithBbox):
    """Alias for LoadImageFromS3WithBbox to match DynamicVis naming."""
    pass


@register_dataset
class FMoWS3PretrainDataset(BaseDataset):
    """
    fMoW S3 Streaming Dataset for DynamicVis Pretraining.
    
    Streams images directly from S3 with bounding box annotations for
    detection-style pretraining used by DynamicVis pretrained models.
    
    Args:
        bucket: S3 bucket name (default: spacenet-dataset)
        s3_prefix: S3 prefix for fMoW data
        split: Dataset split ('train', 'val', 'test')
        pipeline: Data processing pipeline
        use_msrgb: Whether to use msrgb images (smaller) or rgb (larger)
        max_samples: Maximum samples to use (for debugging/testing)
        cache_manifest: Whether to cache the parsed manifest locally
        shuffle_samples: Whether to shuffle samples at init
        enable_prefetch: Enable background prefetching
        prefetch_size: Number of samples to prefetch
        num_prefetch_workers: Number of prefetch threads
    """
    
    METAINFO = {
        'classes': FMOW_CATEGORIES,
    }
    
    def __init__(
        self,
        bucket: str = 'spacenet-dataset',
        s3_prefix: str = 'Hosted-Datasets/fmow/fmow-rgb',
        split: str = 'train',
        pipeline: List[dict] = None,
        use_msrgb: bool = True,  # msrgb is ~10x smaller than rgb
        max_samples: Optional[int] = None,
        cache_manifest: bool = True,
        shuffle_samples: bool = True,
        enable_prefetch: bool = True,
        prefetch_size: int = 512,
        num_prefetch_workers: int = 8,
        **kwargs,
    ):
        self.bucket = bucket
        self.s3_prefix = s3_prefix
        self.split = split
        self.use_msrgb = use_msrgb
        self.max_samples = max_samples
        self.cache_manifest = cache_manifest
        self.shuffle_samples = shuffle_samples
        self.enable_prefetch = enable_prefetch
        self.prefetch_size = prefetch_size
        self.num_prefetch_workers = num_prefetch_workers
        
        # Category mapping
        self.cat2label = {cat: i for i, cat in enumerate(FMOW_CATEGORIES)}
        self.label2cat = {i: cat for i, cat in enumerate(FMOW_CATEGORIES)}
        
        # S3 client
        use_credentials = bool(os.getenv("AWS_ACCESS_KEY_ID"))
        self.s3_client = create_optimized_s3_client(use_credentials)
        
        # Initialize base dataset
        super().__init__(
            ann_file='',
            metainfo=self.METAINFO,
            pipeline=pipeline or [],
            lazy_init=False,  # Ensure data_list is loaded immediately
            serialize_data=False,  # Don't serialize to avoid issues with custom data
            **kwargs,
        )
        
        # Verify data_list was loaded
        print(f"Dataset initialized with {len(self.data_list)} samples")
        
        # Setup prefetch thread if enabled
        if self.enable_prefetch:
            self._setup_prefetch()
    
    def _setup_prefetch(self):
        """Setup background prefetch worker."""
        self.prefetch_queue = queue.Queue(maxsize=self.prefetch_size)
        self.prefetch_indices = queue.Queue()
        self.prefetch_stop = threading.Event()
        self.prefetch_executor = ThreadPoolExecutor(max_workers=self.num_prefetch_workers)
        
        # Start prefetch thread
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def _prefetch_worker(self):
        """Background worker that prefetches samples."""
        while not self.prefetch_stop.is_set():
            try:
                idx = self.prefetch_indices.get(timeout=0.1)
                data = self._fetch_sample(idx)
                self.prefetch_queue.put((idx, data), timeout=1.0)
            except queue.Empty:
                continue
            except queue.Full:
                continue
            except Exception as e:
                print(f"Prefetch error: {e}")
    
    def _fetch_sample(self, idx: int) -> Dict[str, Any]:
        """Fetch a single sample from S3."""
        sample_info = self.data_list[idx]
        img_key = sample_info['img_path']
        json_key = sample_info['json_path']
        
        try:
            # Fetch image
            img_obj = self.s3_client.get_object(Bucket=self.bucket, Key=img_key)
            img_bytes = img_obj['Body'].read()
            
            # Fetch JSON annotation
            json_obj = self.s3_client.get_object(Bucket=self.bucket, Key=json_key)
            json_data = json.loads(json_obj['Body'].read().decode('utf-8'))
            
            # Parse bounding boxes
            bboxes = []
            labels = []
            for bbox_info in json_data.get('bounding_boxes', []):
                if 'raw_location' in bbox_info:
                    continue  # Skip raw locations
                
                category = bbox_info['category']
                if category not in self.cat2label:
                    continue
                
                # Box is [x, y, w, h], convert to [x1, y1, x2, y2]
                x, y, w, h = bbox_info['box']
                bboxes.append([x, y, x + w, y + h])
                labels.append(self.cat2label[category])
            
            return {
                'img_bytes': img_bytes,
                'gt_bboxes': np.array(bboxes, dtype=np.float32).reshape((-1, 4)),
                'gt_bboxes_labels': np.array(labels, dtype=np.int64),
                'img_path': img_key,
            }
            
        except Exception as e:
            print(f"Error fetching sample {idx} ({img_key}): {e}")
            return None
    
    def load_data_list(self) -> List[dict]:
        """Load the list of samples from S3."""
        cache_file = Path(f'data/fmow_manifest_{self.split}.json')
        
        # Try to load from cache
        if self.cache_manifest and cache_file.exists():
            print(f"Loading cached manifest from {cache_file}")
            with open(cache_file) as f:
                data_list = json.load(f)
            if self.max_samples:
                data_list = data_list[:self.max_samples]
            print(f"Loaded {len(data_list)} samples from cache")
            return data_list
        
        # Download and parse manifest
        print(f"Building sample list from S3 for split '{self.split}'...")
        manifest_key = f'{self.s3_prefix}/manifest.json.bz2'
        
        # Download manifest
        local_manifest = Path('data/manifest.json.bz2')
        local_manifest.parent.mkdir(parents=True, exist_ok=True)
        
        if not local_manifest.exists():
            print(f"Downloading manifest from s3://{self.bucket}/{manifest_key}")
            self.s3_client.download_file(self.bucket, manifest_key, str(local_manifest))
        
        # Parse manifest
        with bz2.BZ2File(local_manifest, 'rb') as f:
            all_files = json.load(f)
        
        # Filter files for this split
        suffix = '_msrgb.jpg' if self.use_msrgb else '_rgb.jpg'
        data_list = []
        
        print(f"Parsing manifest for {self.split} split (using {'msrgb' if self.use_msrgb else 'rgb'})...")
        for file_path in tqdm(all_files, desc=f"Parsing {self.split}"):
            if not file_path.endswith(suffix):
                continue
            
            parts = file_path.split('/')
            if len(parts) < 3:
                continue
            
            split_dir = parts[0]
            if split_dir != self.split:
                continue
            
            # Build paths
            img_path = f'{self.s3_prefix}/{file_path}'
            json_path = img_path.replace('.jpg', '.json')
            
            data_list.append({
                'img_path': img_path,
                'json_path': json_path,
            })
        
        if self.shuffle_samples:
            random.shuffle(data_list)
        
        if self.max_samples:
            data_list = data_list[:self.max_samples]
        
        # Cache the manifest
        if self.cache_manifest:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(data_list, f)
            print(f"Cached manifest to {cache_file}")
        
        print(f"Found {len(data_list)} samples for '{self.split}' split")
        return data_list
    
    def get_data_info(self, idx: int) -> dict:
        """Get data info for a specific index."""
        return self.data_list[idx]
    
    def prepare_data(self, idx: int) -> dict:
        """Prepare data for training/evaluation."""
        # Try prefetch queue first
        if self.enable_prefetch:
            # Queue up more indices for prefetch
            for i in range(min(10, len(self.data_list))):
                next_idx = (idx + i + 1) % len(self.data_list)
                try:
                    self.prefetch_indices.put_nowait(next_idx)
                except queue.Full:
                    pass
            
            # Try to get from prefetch queue
            try:
                cached_idx, cached_data = self.prefetch_queue.get_nowait()
                if cached_idx == idx and cached_data is not None:
                    return self.pipeline(cached_data)
            except queue.Empty:
                pass
        
        # Fetch directly
        data = self._fetch_sample(idx)
        if data is None:
            # Return a placeholder on error
            data = {
                'img_bytes': None,
                'gt_bboxes': np.zeros((0, 4), dtype=np.float32),
                'gt_bboxes_labels': np.zeros((0,), dtype=np.int64),
                'img_path': 'error',
            }
        
        return self.pipeline(data)
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __del__(self):
        """Cleanup prefetch resources."""
        if hasattr(self, 'prefetch_stop'):
            self.prefetch_stop.set()
        if hasattr(self, 'prefetch_executor'):
            self.prefetch_executor.shutdown(wait=False)


@register_dataset
class FMoWS3PretrainWebDataset(FMoWS3PretrainDataset):
    """Streaming S3 version for DynamicVis configs.
    
    This provides a streaming alternative to DynamicVis's PretrainFmowWebDataset
    which requires local tar files. Use this when you want to stream from S3.
    
    Note: The original PretrainFmowWebDataset uses WebDataset tar files.
    This version streams directly from S3.
    """
    
    def __init__(
        self,
        shards_path_or_url: str = None,  # Ignored, for compatibility
        data_name: str = 'Fmow',  # Ignored, for compatibility
        per_gpu_batch_size: int = 1,  # Ignored
        num_workers: int = 0,  # Ignored
        shuffle_buffer_size: int = 1000,  # Maps to prefetch_size
        test_mode: bool = False,
        pipeline: List[dict] = None,
        **kwargs,
    ):
        # Map to parent arguments
        split = 'val' if test_mode else 'train'
        
        super().__init__(
            split=split,
            pipeline=pipeline,
            shuffle_samples=not test_mode,
            prefetch_size=shuffle_buffer_size,
            **kwargs,
        )
        
        self.test_mode = test_mode
    
    def real_len(self):
        """Compatibility method."""
        return len(self.data_list)


# Helper function to create dataset easily
def create_fmow_s3_pretrain_dataset(
    split: str = 'train',
    pipeline: List[dict] = None,
    use_msrgb: bool = True,
    max_samples: Optional[int] = None,
    **kwargs,
) -> FMoWS3PretrainDataset:
    """
    Create an fMoW S3 streaming dataset for pretraining.
    
    Args:
        split: 'train', 'val', or 'test'
        pipeline: Data processing pipeline
        use_msrgb: Use smaller msrgb images (recommended for training)
        max_samples: Limit number of samples (for debugging)
        **kwargs: Additional arguments passed to dataset
    
    Returns:
        FMoWS3PretrainDataset instance
    """
    return FMoWS3PretrainDataset(
        bucket='spacenet-dataset',
        s3_prefix='Hosted-Datasets/fmow/fmow-rgb',
        split=split,
        pipeline=pipeline,
        use_msrgb=use_msrgb,
        max_samples=max_samples,
        **kwargs,
    )
