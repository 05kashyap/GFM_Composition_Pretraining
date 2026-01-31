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