"""
fMoW S3 Streaming Dataset for MMPretrain/DynamicVis.
Streams images directly from AWS S3 without downloading the entire dataset.
Compatible with MMEngine's training framework.
"""

import os
import io
import bz2
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

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
    """Load an image from S3 bucket.
    
    Required Keys:
        - s3_key
        - s3_client
        - bucket
    
    Modified Keys:
        - img
        - img_shape
        - ori_shape
    """
    
    def __init__(self, to_float32: bool = False, color_type: str = 'color'):
        self.to_float32 = to_float32
        self.color_type = color_type
    
    def __call__(self, results: Dict) -> Optional[Dict]:
        s3_client = results['s3_client']
        bucket = results['bucket']
        s3_key = results['s3_key']
        
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=s3_key)
            img_bytes = obj['Body'].read()
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
        **kwargs
    ):
        self.bucket = bucket
        self.s3_prefix = s3_prefix
        self.manifest_key = manifest_key
        self.local_manifest = local_manifest
        self.split = split
        
        # Setup S3 client
        if not os.getenv("AWS_ACCESS_KEY_ID"):
            # Use unsigned requests for public buckets
            self.s3_client = boto3.client(
                "s3",
                config=Config(signature_version=UNSIGNED),
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )
        else:
            # Use credentials from environment
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )
        
        # Build class to idx mapping
        self.class_to_idx = {cat: i for i, cat in enumerate(FMOW_CATEGORIES)}
        
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
    
    def prepare_data(self, idx: int) -> Dict:
        """Get data info by index and apply pipeline."""
        data_info = self.get_data_info(idx)
        
        # Add S3 client and bucket to data_info for the pipeline
        data_info['s3_client'] = self.s3_client
        data_info['bucket'] = self.bucket
        
        return self.pipeline(data_info)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get item with error handling."""
        try:
            data = self.prepare_data(idx)
            if data is None:
                # Return next valid sample
                return self.__getitem__((idx + 1) % len(self))
            return data
        except Exception as e:
            print(f"Error getting item {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))


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
