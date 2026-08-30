"""Unified model adapter interface for geospatial foundation models."""

import torch
import torch.nn as nn
from typing import Optional, List
from pathlib import Path


def create_model(
    model_type: str,
    model_path: str,
    embedding_dim: int = 384,
    device: Optional[torch.device] = None,
    config_path: Optional[str] = None,
    use_multi_scale: bool = False,
    layer_indices: Optional[List[int]] = None,
    img_size: int = 224,
    in_chans: int = 4
) -> nn.Module:
    """Factory function to create a model encoder.

    Args:
        model_type: Type of model ('prithvi' or 'dynamicvis' or 'prithvi2').
        model_path: Path to the model checkpoint.
        embedding_dim: Dimension of output embeddings.
        device: PyTorch device for model.
        config_path: Path to config file (required for DynamicVis).
        use_multi_scale: Whether to use multi-scale features (Prithvi only).
        layer_indices: List of layer indices for multi-scale features (Prithvi only).
        img_size: Input image size (default: 224).
        in_chans: Number of input channels (default: 4 for R, G, B, NIR).

    Returns:
        Initialized model encoder.

    Examples:
        # Prithvi model
        model = create_model(
            model_type='prithvi',
            model_path='models/Prithvi_EO_V1_100M.pt',
            embedding_dim=384,
            use_multi_scale=True,
            layer_indices=[3, 6, 9, 11]
        )

        # DynamicVis model
        model = create_model(
            model_type='dynamicvis',
            model_path='models/pretrain_dynamicvis_b_bf16_mamba_best.pth',
            config_path='configs/pretrain_dynamicvis_b_bf16_mamba.py',
            embedding_dim=768
        )
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = model_type.lower()

    if model_type == 'dynamicvis':
        if config_path is None:
            raise ValueError("config_path must be provided for DynamicVis model")
        
        from eval.adapters.dynamicvis_adapter import DynamicVisEncoder

        print(f"✓ Creating DynamicVis encoder (embedding_dim={embedding_dim})")
        model = DynamicVisEncoder(
            model_path=model_path,
            config_path=config_path,
            embedding_dim=embedding_dim,
            device=device
        )
    elif model_type in {'prithvi', 'prithvi2', 'prithvi_v2'}:
        from eval.adapters.prithvi_v2_adapter import PrithviEncoderV2

        print(f"✓ Creating Prithvi v2 encoder (embedding_dim={embedding_dim})")
        model = PrithviEncoderV2(
            model_path=model_path,
            embedding_dim=embedding_dim,
            device=device,
            use_multi_scale=use_multi_scale,
            layer_indices=layer_indices,
            img_size=img_size,
            in_chans=in_chans,
        )
    else:
        raise ValueError(
            f"Unsupported model_type: {model_type}. Supported values are dynamicvis, prithvi, prithvi2, and prithvi_v2."
        )

    print(f"✓ Model loaded successfully on {device}")
    return model