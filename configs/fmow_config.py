"""
Configuration for fMoW dataset training with DynamicVis.
"""

from dataclasses import dataclass


@dataclass
class FMoWConfig:
    """Configuration for fMoW training."""
    
    # AWS S3 Configuration
    s3_bucket: str = "spacenet-dataset"
    s3_prefix: str = "Hosted-Datasets/fmow/fmow-rgb"
    manifest_key: str = "Hosted-Datasets/fmow/fmow-rgb/manifest.json.bz2"
    local_manifest: str = "data/manifest.json.bz2"
    aws_region: str = "us-east-1"
    
    # Dataset Configuration
    num_classes: int = 63  # fMoW has 62 categories + 1 "false detection"
    image_size: int = 224
    
    # Model Configuration
    model_type: str = "dynamicvis_base"  # Options: dynamicvis_small, dynamicvis_base, dynamicvis_large
    pretrained: bool = True
    
    # Training Configuration
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    epochs: int = 100
    
    # Data Augmentation
    use_mixup: bool = True
    mixup_alpha: float = 0.8
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    
    # Normalization (ImageNet stats)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)