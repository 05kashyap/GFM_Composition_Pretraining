"""This module is responsible for all interactions with the raw image data.
Reading 4-Band TIFFs: The core function will use rasterio.open() to load a TIFF file. The dataset.read() method will return a NumPy array with the shape (num_bands, height, width), which in this case will be (4, H, W).25 The bands can then be reordered if necessary (e.g., to
(H, W, 4)) for processing with libraries like OpenCV.
Tiling Strategy: A function tile_image(image_array, tile_size, stride) will be implemented. This function will iterate over the large image array, extracting patches of tile_size x tile_size with a given stride. An overlap (i.e., stride < tile_size) is recommended to ensure objects on tile edges are fully captured in at least one tile.
Data Augmentation: For training, a set of transformations using torchvision.transforms or albumentations should be defined. These will include RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, and ColorJitter. These augmentations are applied to the anchor and positive samples in a triplet to teach the model rotational and photometric invariance.
PyTorch Dataset/DataLoader: A custom torch.utils.data.Dataset class, TripletDataset, will be created. Its __getitem__ method will be responsible for sampling an anchor image, finding a corresponding positive sample (same class, different instance), and sampling a random negative sample (different class or background). This dataset will then be wrapped in a DataLoader to provide batches of triplets to the training script.
"""

import os
import random
from typing import List, Tuple, Optional, Dict
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2


# def load_tiff_image(file_path: str) -> np.ndarray:
#     with rasterio.open(file_path) as src:
#         arr = src.read()  # (C, H, W)
#     arr = np.transpose(arr, (1, 2, 0))  # (H, W, C)
#     if arr.shape[2] >= 3:
#         arr = arr[:, :, :3]  # keep RGB only
#     return arr.astype(np.float32)

def load_satellite_image(file_path: str) -> np.ndarray:
    """Load a satellite image from TIFF or common RGB formats.

    TIFF files are read with rasterio and can keep up to 4 channels.
    PNG/JPG/JPEG files are read as RGB.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".tif", ".tiff"):
        with rasterio.open(file_path) as src:
            arr = src.read()  # (C, H, W)
        arr = np.transpose(arr, (1, 2, 0))  # (H, W, C)
        if arr.shape[2] >= 4:
            arr = arr[:, :, :4]  # Keep RGB + NIR
        elif arr.shape[2] == 3:
            # Handle cases where only RGB is available, pad with zeros for NIR.
            nir = np.zeros((arr.shape[0], arr.shape[1], 1), dtype=arr.dtype)
            arr = np.concatenate([arr, nir], axis=2)
        return arr.astype(np.float32)

    if ext in (".png", ".jpg", ".jpeg"):
        arr = np.asarray(Image.open(file_path).convert("RGB"), dtype=np.float32)
        return arr

    raise ValueError(f"Unsupported image format: {file_path}")


def load_tiff_image(file_path: str) -> np.ndarray:
    # Backwards-compatible alias.
    return load_satellite_image(file_path)

def tile_image(image_array: np.ndarray, tile_size: int, stride: int) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """Extract tiles from a large image array.
    
    Args:
        image_array: Input image array of shape (H, W, C)
        tile_size: Size of each tile (tile_size x tile_size)
        stride: Step size between tiles
    
    Returns:
        List of (tile_array, (x_offset, y_offset)) tuples
    """
    tiles = []
    height, width = image_array.shape[:2]
    channels = image_array.shape[2] if image_array.ndim == 3 else 1
    
    # Calculate number of tiles needed to cover the entire image
    n_rows = (height + stride - 1) // stride
    n_cols = (width + stride - 1) // stride
    
    for row in range(n_rows):
        for col in range(n_cols):
            y = row * stride
            x = col * stride
            
            # Create empty tile filled with zeros (black)
            tile = np.zeros((tile_size, tile_size, channels), dtype=image_array.dtype)
            
            # Calculate the actual region to extract from the image
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            
            # Calculate how much of the tile to fill
            tile_h = y_end - y
            tile_w = x_end - x
            
            # Copy the image region into the tile (top-left portion)
            tile[:tile_h, :tile_w] = image_array[y:y_end, x:x_end]
            
            tiles.append((tile, (x, y)))
    
    return tiles


class FourChannelColorJitter(A.ImageOnlyTransform):
    """Custom ColorJitter that works with 4-channel images by applying jitter only to RGB channels."""
    
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
        # Create a ColorJitter for RGB channels only
        self.rgb_jitter = A.ColorJitter(
            brightness=brightness, 
            contrast=contrast, 
            saturation=saturation, 
            hue=hue, 
            always_apply=True, 
            p=1.0
        )
    
    def apply(self, img, **params):
        # Apply jitter to RGB channels and keep any extra channels unchanged.
        if img.ndim != 3 or img.shape[2] < 3:
            return img

        rgb_img = img[:, :, :3]
        tail = img[:, :, 3:] if img.shape[2] > 3 else None

        rgb_jittered = self.rgb_jitter(image=rgb_img)['image']
        if tail is None:
            return rgb_jittered
        return np.concatenate([rgb_jittered, tail], axis=2)


def get_augmentation_transforms(is_training: bool = True) -> A.Compose:
    def _normalize_any_channels(img: np.ndarray, **kwargs) -> np.ndarray:
        # Input is scaled to [0, 1] in __getitem__, normalize to [-1, 1].
        return (img - 0.5) / 0.5

    transforms = [
        A.Lambda(image=_normalize_any_channels),
        ToTensorV2(),
    ]
    if is_training:
        aug = [
            FourChannelColorJitter(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]
        transforms = aug + transforms
    return A.Compose(transforms)


class TripletDataset(Dataset):
    """Custom PyTorch Dataset for triplet learning."""
    
    def __init__(self, 
                 data_dir: str, 
                 tile_size: int = 224, 
                 stride: int = 84, 
                 transform: Optional[A.Compose] = None):
        """Initialize the TripletDataset.
        
        Args:
            data_dir: Directory containing the training images
            tile_size: Size of tiles to extract
            stride: Stride for tile extraction
            transform: Augmentation transforms to apply
        """
        self.data_dir = data_dir
        self.tile_size = tile_size
        self.stride = stride
        self.transform = transform or get_augmentation_transforms(is_training=True)
        
        # Load and preprocess all images
        self.image_files = self._load_image_files()
        self.tiles_data = self._extract_all_tiles()
        
        # Group tiles by image for positive/negative sampling
        self.image_to_tiles = self._group_tiles_by_image()
        
    def _load_image_files(self) -> List[str]:
        """Load all TIFF image files from the data directory."""
        image_files = []
        for file in os.listdir(self.data_dir):
            if file.lower().endswith(('.tif', '.tiff')):
                image_files.append(os.path.join(self.data_dir, file))
        return image_files
    
    def _extract_all_tiles(self) -> List[Dict]:
        """Extract tiles from all images and store metadata."""
        tiles_data = []
        
        for img_idx, img_path in enumerate(self.image_files):
            image_array = load_satellite_image(img_path)
            tiles = tile_image(image_array, self.tile_size, self.stride)
            
            for tile_idx, (tile, coords) in enumerate(tiles):
                tiles_data.append({
                    'tile': tile,
                    'image_idx': img_idx,
                    'image_path': img_path,
                    'coordinates': coords,
                    'tile_idx': tile_idx
                })
                
        return tiles_data
    
    def _group_tiles_by_image(self) -> Dict[int, List[int]]:
        """Group tile indices by their source image."""
        image_to_tiles = {}
        for tile_idx, tile_data in enumerate(self.tiles_data):
            img_idx = tile_data['image_idx']
            if img_idx not in image_to_tiles:
                image_to_tiles[img_idx] = []
            image_to_tiles[img_idx].append(tile_idx)
        return image_to_tiles
    
    def __len__(self) -> int:
        return len(self.tiles_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a triplet (anchor, positive, negative) sample."""
        # Get anchor
        anchor_data = self.tiles_data[idx]
        anchor_img = anchor_data['tile'].copy()
        anchor_img_idx = anchor_data['image_idx']
        anchor_coords = anchor_data['coordinates']
        
        # Get positive (spatially nearby tile from same image)
        same_image_tiles = self.image_to_tiles[anchor_img_idx]
        positive_candidates = []
        
        # Define proximity threshold (e.g., within 3 tile distances)
        proximity_threshold = self.tile_size * 3
        
        for tile_idx in same_image_tiles:
            if tile_idx != idx:
                candidate_coords = self.tiles_data[tile_idx]['coordinates']
                # Calculate distance between tile centers
                distance = np.sqrt((anchor_coords[0] - candidate_coords[0])**2 + 
                                (anchor_coords[1] - candidate_coords[1])**2)
                
                if distance <= proximity_threshold:
                    positive_candidates.append(tile_idx)
        
        # Choose positive sample
        if positive_candidates:
            pos_idx = random.choice(positive_candidates)
        elif len(same_image_tiles) > 1:
            # Fallback to any other tile from same image
            pos_idx = random.choice([t for t in same_image_tiles if t != idx])
        else:
            pos_idx = idx
            
        positive_img = self.tiles_data[pos_idx]['tile'].copy()
        
        # Get negative (tile from different image) - unchanged
        different_images = [img_idx for img_idx in self.image_to_tiles.keys() 
                        if img_idx != anchor_img_idx]
        
        if len(different_images) > 0:
            neg_img_idx = random.choice(different_images)
            neg_tile_idx = random.choice(self.image_to_tiles[neg_img_idx])
        else:
            neg_tile_idx = random.choice(range(len(self.tiles_data)))
            
        negative_img = self.tiles_data[neg_tile_idx]['tile'].copy()
        
        # Ensure images are in the correct data type and range
        anchor_img = anchor_img.astype(np.float32)
        positive_img = positive_img.astype(np.float32)
        negative_img = negative_img.astype(np.float32)
        
        # Normalize to 0-1 range if needed (assuming input is 0-255 or similar)
        if anchor_img.max() > 1.0:
            anchor_img = anchor_img / 255.0
            positive_img = positive_img / 255.0
            negative_img = negative_img / 255.0
        
        # Apply transforms
        if self.transform:
            anchor_img = self.transform(image=anchor_img)['image']
            positive_img = self.transform(image=positive_img)['image']
            negative_img = self.transform(image=negative_img)['image']
        
        return anchor_img, positive_img, negative_img


def get_triplet_loader(data_dir: str, 
                      batch_size: int = 32, 
                      tile_size: int = 224, 
                      stride: int = 84, 
                      num_workers: int = 4, 
                      shuffle: bool = True) -> DataLoader:
    """Create a DataLoader for triplet training.
    
    Args:
        data_dir: Directory containing training images
        batch_size: Batch size for training
        tile_size: Size of tiles to extract
        stride: Stride for tile extraction
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data
        
    Returns:
        PyTorch DataLoader
    """
    dataset = TripletDataset(
        data_dir=data_dir,
        tile_size=tile_size,
        stride=stride,
        transform=get_augmentation_transforms(is_training=True)
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


class InferenceDataset(Dataset):
    """Dataset for inference/indexing without triplet sampling."""
    
    def __init__(self, 
                 image_paths: List[str], 
                 tile_size: int = 224, 
                 stride: int = 84):
        """Initialize the InferenceDataset.
        
        Args:
            image_paths: List of paths to images
            tile_size: Size of tiles to extract
            stride: Stride for tile extraction
        """
        self.image_paths = image_paths
        self.tile_size = tile_size
        self.stride = stride
        self.transform = get_augmentation_transforms(is_training=False)
        
        # Extract all tiles with metadata
        self.tiles_data = self._extract_all_tiles()
        
    def _extract_all_tiles(self) -> List[Dict]:
        """Extract tiles from all images."""
        tiles_data = []
        
        for img_path in self.image_paths:
            image_array = load_satellite_image(img_path)
            tiles = tile_image(image_array, self.tile_size, self.stride)
            
            for tile_idx, (tile, coords) in enumerate(tiles):
                tiles_data.append({
                    'tile': tile,
                    'image_path': img_path,
                    'coordinates': coords,
                    'tile_idx': tile_idx
                })
                
        return tiles_data
    
    def __len__(self) -> int:
        return len(self.tiles_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get a tile and its metadata.
        
        Args:
            idx: Index of the tile
            
        Returns:
            Tuple of (tile_tensor, metadata_dict)
        """
        tile_data = self.tiles_data[idx]
        tile = tile_data['tile'].copy()
        
        # Ensure correct data type and range
        tile = tile.astype(np.float32)
        if tile.max() > 1.0:
            tile = tile / 255.0
        
        if self.transform:
            tile = self.transform(image=tile)['image']
            
        metadata = {
            'image_path': tile_data['image_path'],
            'coordinates': tile_data['coordinates'],
            'tile_idx': tile_data['tile_idx']
        }
        
        return tile, metadata


def custom_collate_fn(batch):
    """Custom collate function for inference dataset."""
    tiles = []
    metadata = []
    
    for tile, meta in batch:
        tiles.append(tile)
        metadata.append(meta)
    
    # Stack tiles into batch tensor
    tiles_batch = torch.stack(tiles)
    
    return tiles_batch, metadata


def get_inference_loader(image_paths: List[str], 
                        batch_size: int = 32, 
                        tile_size: int = 224, 
                        stride: int = 84, 
                        num_workers: int = 4) -> DataLoader:
    """Create a DataLoader for inference/indexing."""
    dataset = InferenceDataset(
        image_paths=image_paths,
        tile_size=tile_size,
        stride=stride
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # Add this line
    )