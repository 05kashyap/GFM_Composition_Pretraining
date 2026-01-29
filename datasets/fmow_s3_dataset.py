"""
fMoW S3 Streaming Dataset.
Streams images directly from AWS S3 without downloading the entire dataset.
"""

import os
import io
import bz2
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import boto3
from botocore.config import Config
from botocore import UNSIGNED
from tqdm import tqdm

# Allow large images
Image.MAX_IMAGE_PIXELS = None


class FMoWS3Dataset(Dataset):
    """
    PyTorch Dataset for streaming fMoW data directly from AWS S3.
    
    Args:
        bucket: S3 bucket name
        s3_prefix: Prefix path within the bucket
        manifest_key: S3 key for the manifest file
        local_manifest: Local path to cache the manifest
        split: Dataset split ('train', 'val', 'test')
        transform: Image transforms to apply
        use_unsigned: Use unsigned requests (no AWS credentials needed for public buckets)
    """
    
    def __init__(
        self,
        bucket: str,
        s3_prefix: str,
        manifest_key: str,
        local_manifest: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        use_unsigned: bool = False,
    ):
        self.bucket = bucket
        self.s3_prefix = s3_prefix
        self.transform = transform
        self.split = split
        
        # Setup S3 client
        if use_unsigned or not os.getenv("AWS_ACCESS_KEY_ID"):
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
        
        # Ensure local manifest directory exists
        Path(local_manifest).parent.mkdir(parents=True, exist_ok=True)
        
        # Download manifest if not exists
        if not os.path.exists(local_manifest):
            print(f"Downloading manifest file {manifest_key}...")
            self.s3_client.download_file(bucket, manifest_key, local_manifest)
            print("Manifest download complete.")
        
        # Parse manifest
        print(f"Loading manifest file for '{split}' split...")
        self.metadata: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        
        with bz2.BZ2File(local_manifest, "rb") as f:
            manifest_data = json.load(f)
        
        print(f"Parsing manifest filenames for '{split}' split...")
        current_class_idx = 0
        
        for img_path in tqdm(manifest_data, desc=f"Parsing {split}"):
            if not img_path.endswith((".jpg", ".jpeg", ".png")):
                continue
            
            parts = img_path.split("/")
            if len(parts) < 3:
                continue
            
            split_dir = parts[0]
            
            # Only parse files matching requested split
            if split_dir != split:
                continue
            
            category = parts[1]
            
            if category not in self.class_to_idx:
                self.class_to_idx[category] = current_class_idx
                self.idx_to_class[current_class_idx] = category
                current_class_idx += 1
            
            label_idx = self.class_to_idx[category]
            self.metadata.append((img_path, label_idx))
        
        if len(self.metadata) == 0:
            print(f"WARNING: Found 0 images for split '{split}'.")
        else:
            print(f"Found {len(self.metadata)} images in {len(self.class_to_idx)} classes for '{split}' split.")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.metadata[idx]
        full_s3_key = f"{self.s3_prefix}/{img_path}"
        
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket, Key=full_s3_key)
            img_bytes = obj["Body"].read()
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            # Return placeholder on error
            print(f"Error loading image {full_s3_key}: {e}")
            if self.transform:
                # Return a properly sized zero tensor
                return torch.zeros((3, 224, 224)), -1
            return Image.new("RGB", (224, 224)), -1
    
    def get_class_name(self, idx: int) -> str:
        """Get class name from index."""
        return self.idx_to_class.get(idx, "unknown")


def get_fmow_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and validation transforms for fMoW dataset.
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return train_transform, val_transform