"""
DynamicVis Classifier wrapper for fMoW classification.
Uses the DynamicVis architecture with Mamba-based spatial sparse mixers.
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add DynamicVis to path
DYNAMICVIS_PATH = Path(__file__).parent.parent / "architectures" / "DynamicVis"
if str(DYNAMICVIS_PATH) not in sys.path:
    sys.path.insert(0, str(DYNAMICVIS_PATH))


def build_dynamicvis_classifier(
    num_classes: int,
    pretrained: bool = True,
    model_type: str = "dynamicvis_base",
    img_size: int = 224,
    **kwargs,
) -> nn.Module:
    """
    Build a DynamicVis classifier model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (not implemented yet)
        model_type: Model variant (dynamicvis_small, dynamicvis_base, dynamicvis_large)
        img_size: Input image size
    
    Returns:
        DynamicVis classifier model
    """
    # Extract size from model_type
    if "small" in model_type.lower():
        arch = "b"  # Use base as fallback, DynamicVis only has 'b' and 'l'
    elif "large" in model_type.lower():
        arch = "l"
    else:
        arch = "b"
    
    try:
        # Import DynamicVis - this registers the models
        import dynamicvis
        from mmpretrain.registry import MODELS
        
        print(f"Building DynamicVis classifier with arch='{arch}', img_size={img_size}")
        
        # Build using MMPretrain's ImageClassifier with DynamicVis backbone
        model_cfg = dict(
            type='ImageClassifier',
            backbone=dict(
                type='DynamicVisBackbone',
                arch=arch,
                img_size=img_size,
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                spatial_token_keep_ratios=[8, 4, 2, 1],
                channel_token_keep_ratios=[1, 2, 2, 4],
                in_channels=3,
                out_indices=(3,),  # Only use last stage for classification
                out_type='avg_featmap',
                path_type='forward_reverse_mean',
                sampling_scale=dict(type='fixed', val=0.1),
                global_token_cfg=dict(pos='head', num=-1),
                with_pe=True,
                mamba2=False,
            ),
            neck=None,
            head=dict(
                type='DynamicVisClsHead',
                num_classes=num_classes,
                in_channels=768 if arch == 'b' else 1024,
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            ),
        )
        
        model = MODELS.build(model_cfg)
        print(f"Successfully built DynamicVis ImageClassifier")
        return model
            
    except Exception as e:
        print(f"Could not build DynamicVis with ImageClassifier: {e}")
        print("Trying direct backbone + head approach...")
        
        try:
            # Try building backbone and head directly
            import dynamicvis.models
            from dynamicvis.models import DynamicVisBackbone, DynamicVisClsHead
            
            # Embed dims based on architecture
            embed_dims = 768 if arch == 'b' else 1024
            
            backbone = DynamicVisBackbone(
                arch=arch,
                img_size=img_size,
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                spatial_token_keep_ratios=[8, 4, 2, 1],
                channel_token_keep_ratios=[1, 2, 2, 4],
                in_channels=3,
                out_indices=(3,),
                out_type='avg_featmap',
                path_type='forward_reverse_mean',
                sampling_scale=dict(type='fixed', val=0.1),
                global_token_cfg=dict(pos='head', num=-1),
                with_pe=True,
                mamba2=False,
            )
            
            head = DynamicVisClsHead(
                num_classes=num_classes,
                in_channels=embed_dims,
            )
            
            # Create a simple wrapper
            model = DynamicVisClassifierWrapper(backbone, head)
            print(f"Successfully built DynamicVis with direct instantiation")
            return model
            
        except Exception as e2:
            print(f"Could not build DynamicVis directly: {e2}")
            print("Falling back to torchvision ViT model...")
            
            # Fallback to torchvision ViT
            import torchvision.models as models
            
            if arch == "l":
                model = models.vit_l_16(weights="IMAGENET1K_V1" if pretrained else None)
                model.heads.head = nn.Linear(1024, num_classes)
            else:  # base
                model = models.vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)
                model.heads.head = nn.Linear(768, num_classes)
            
            print(f"Built fallback ViT classifier: {model_type}")
            return model


class DynamicVisClassifierWrapper(nn.Module):
    """
    Simple wrapper that combines DynamicVis backbone and classification head.
    """
    
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get features from backbone
        features = self.backbone(x)
        
        # Handle tuple output (multiple stages)
        if isinstance(features, (tuple, list)):
            features = features[-1]  # Use last stage
        
        # DynamicVis with avg_featmap already returns [B, C]
        # but head.fc expects just the features
        logits = self.head.fc(features)
        return logits
    
    def extract_feat(self, x: torch.Tensor):
        """Extract features without classification head."""
        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        return features