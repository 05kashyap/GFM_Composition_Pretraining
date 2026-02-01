"""Dataset modules."""

from .fmow_s3_dataset import FMoWS3Dataset, get_fmow_transforms

# MMPretrain compatible datasets
try:
    from .fmow_s3_mmpretrain import (
        FMoWS3Dataset as FMoWS3DatasetMM,
        FMoWS3DatasetSimple,
        LoadImageFromS3,
        FMOW_CATEGORIES,
    )
    __all__ = [
        "FMoWS3Dataset", 
        "get_fmow_transforms",
        "FMoWS3DatasetMM",
        "FMoWS3DatasetSimple", 
        "LoadImageFromS3",
        "FMOW_CATEGORIES",
    ]
except ImportError:
    # MMPretrain not installed, use basic dataset only
    __all__ = ["FMoWS3Dataset", "get_fmow_transforms"]

# Pretrain dataset for DynamicVis (with bounding boxes)
try:
    from .fmow_s3_pretrain import (
        FMoWS3PretrainDataset,
        FMoWS3PretrainWebDataset,
        LoadImageFromS3WithBbox,
        LoadImageFromImgbytesS3,
        create_fmow_s3_pretrain_dataset,
        FMOW_CATEGORIES as FMOW_PRETRAIN_CATEGORIES,
    )
    __all__.extend([
        "FMoWS3PretrainDataset",
        "FMoWS3PretrainWebDataset",
        "LoadImageFromS3WithBbox",
        "LoadImageFromImgbytesS3",
        "create_fmow_s3_pretrain_dataset",
    ])
except ImportError as e:
    print(f"Warning: Could not import fmow_s3_pretrain: {e}")