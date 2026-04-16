"""
BoVW (Bag of Visual Words) model and head for DynamicVis.

``BoVWDynamicVis`` wraps the DynamicVis backbone with a histogram prediction head
that outputs a soft distribution over a visual vocabulary. Trained with Sinkhorn
EMD loss against pre-computed histogram targets from DINOv3 patch tokens.

This is a simpler and more stable alternative to the QSACL-based composition
training which suffered from slot collapse issues.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure DynamicVis is importable
_DYNVIS = Path(__file__).resolve().parent.parent / "architectures" / "DynamicVis"
if str(_DYNVIS) not in sys.path:
    sys.path.insert(0, str(_DYNVIS))

from mmengine.model import BaseModel, BaseDataPreprocessor
from mmpretrain.registry import MODELS


# --------------------------------------------------------------------------- #
# BoVWDataPreprocessor — handles multi-view inputs
# --------------------------------------------------------------------------- #

@MODELS.register_module()
class BoVWDataPreprocessor(BaseDataPreprocessor):
    """Data preprocessor for BoVW training.

    Handles both single-view and multi-view inputs:
      - Single-view: inputs is a (3, H, W) tensor per sample
      - Multi-view: inputs is a list of N (3, H, W) tensors per sample

    Performs:
      1. Move data to device
      2. Stack tensors across batch dimension
      3. No normalization (already done in dataset transforms)
    """

    def __init__(self, non_blocking: bool = False):
        super().__init__(non_blocking=non_blocking)

    def forward(self, data: dict, training: bool = True) -> dict:
        """Process a batch from collate function."""
        data = self.cast_data(data)

        if isinstance(data, dict) and 'inputs' in data:
            inputs = data['inputs']
            data_samples = data.get('data_samples', None)

            # Check if multi-view
            is_multiview = (
                isinstance(inputs, (list, tuple)) and
                len(inputs) > 0 and
                isinstance(inputs[0], (list, tuple))
            )

            if is_multiview:
                num_views = len(inputs)
                inputs_list = []

                for view_idx in range(num_views):
                    view_tensors = inputs[view_idx]
                    if isinstance(view_tensors, (list, tuple)):
                        inputs_list.append(torch.stack(list(view_tensors), dim=0))
                    else:
                        inputs_list.append(view_tensors)

                return {"inputs": inputs_list, "data_samples": data_samples}
            else:
                if isinstance(inputs, (list, tuple)):
                    inputs = torch.stack(list(inputs), dim=0)
                return {"inputs": inputs, "data_samples": data_samples}

        return data


# --------------------------------------------------------------------------- #
# BoVWHead — histogram prediction MLP
# --------------------------------------------------------------------------- #

@MODELS.register_module()
class BoVWHead(nn.Module):
    """Histogram prediction head for BoVW training.

    Architecture::

        backbone features (B, in_channels)
          -> Linear(in_channels, hidden_dim)
          -> LayerNorm -> GELU
          -> Linear(hidden_dim, vocab_size)
          -> Softmax (at inference / loss time)

    No final LayerNorm before softmax — softmax handles normalisation.

    Args:
        in_channels: Backbone embedding dimension (768 for arch='b').
        vocab_size: Size of visual vocabulary (K=512 by default).
        hidden_dim: Hidden layer size in the MLP.
    """

    def __init__(
        self,
        in_channels: int = 768,
        vocab_size: int = 512,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        # 2-layer prediction MLP (as specified)
        self.prediction_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Compute histogram logits.

        Args:
            feats: (B, in_channels) backbone features.

        Returns:
            logits: (B, vocab_size) raw logits (before softmax).
        """
        return self.prediction_head(feats)


# --------------------------------------------------------------------------- #
# BoVWDynamicVis — top-level model
# --------------------------------------------------------------------------- #

@MODELS.register_module()
class BoVWDynamicVis(BaseModel):
    """DynamicVis backbone + BoVW histogram prediction head.

    Predicts soft histogram distributions over a visual vocabulary using
    Sinkhorn EMD loss against pre-computed targets from DINOv3 patch tokens.

    This model combines three loss components matching vanilla DynamicVis pretraining:
      - **EMD loss** — Sinkhorn optimal transport for histogram alignment (BoVW-specific)
      - **Classification loss** — Label-smoothed cross-entropy on aux head
      - **MIL contrastive loss** — CLIP-style bidirectional feature-class alignment

    Optional components:
      - **Auxiliary classification head** — ``nn.Linear(backbone_dim, num_classes)``
        for label-guided training. Receives stop-gradiented backbone features.

    Config usage::

        model = dict(
            type='BoVWDynamicVis',
            backbone=dict(type='mmpretrain.DynamicVisBackbone', ...),
            vocab_size=512,
            hidden_dim=512,
            num_classes=63,
            ground_cost_path='outputs/bovw_vocabulary/ground_cost.npy',
            lambda_emd=1.0,
            lambda_cls=0.5,
            lambda_mil=0.25,
        )
    """

    NUM_CLASSES = 63  # fMoW categories

    def __init__(
        self,
        backbone: dict,
        vocab_size: int = 512,
        hidden_dim: int = 512,
        num_classes: int = 63,
        ground_cost_path: Optional[str] = None,
        lambda_emd: float = 1.0,
        lambda_cls: float = 0.5,
        lambda_mil: float = 0.25,
        sinkhorn_eps: float = 0.05,
        sinkhorn_iters: int = 50,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )

        self.vocab_size = vocab_size

        # Ensure backbone is in avg_featmap mode
        backbone = backbone.copy()
        backbone['out_type'] = 'avg_featmap'
        backbone['out_indices'] = (3,)

        self.backbone = MODELS.build(backbone)

        # Infer backbone dimension from backbone config
        # Default to 768 for arch='b'
        backbone_dim = 768
        if 'arch' in backbone:
            arch = backbone['arch']
            if arch == 'b':
                backbone_dim = 768
            elif arch == 's':
                backbone_dim = 384
            elif arch == 'l':
                backbone_dim = 1024
        self.backbone_dim = backbone_dim

        # Histogram prediction head
        self.head = BoVWHead(
            in_channels=backbone_dim,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
        )

        # Auxiliary classification head (training scaffold)
        # Note: MIL loss uses backbone features directly with gradients
        self.num_classes = num_classes
        if num_classes > 0:
            self.aux_cls_head = nn.Linear(backbone_dim, num_classes)
        else:
            self.aux_cls_head = None

        # Loss function with MIL support
        from losses.bovw_loss import BoVWLoss
        self.loss_fn = BoVWLoss(
            ground_cost_path=ground_cost_path,
            lambda_emd=lambda_emd,
            lambda_cls=lambda_cls,
            lambda_mil=lambda_mil,
            num_classes=num_classes,
            feature_dim=backbone_dim,
            sinkhorn_eps=sinkhorn_eps,
            sinkhorn_iters=sinkhorn_iters,
        )

    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run backbone. Returns (B, backbone_dim) global feature."""
        feats = self.backbone(inputs)
        if isinstance(feats, (tuple, list)):
            return feats[-1]
        return feats

    def forward(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        data_samples: Optional[List] = None,
        mode: str = "loss",
    ):
        """Unified forward with multi-view support.

        Args:
            inputs: Either:
                - (B, 3, H, W) single view tensor
                - List of N tensors, each (B, 3, H, W) for multi-view
            data_samples: Per-sample metadata with:
                - histogram_target: (K,) target histogram
                - dominant_label: fMoW class (optional)
            mode: ``'loss'`` → dict of losses,
                  ``'predict'`` → data_samples with predictions,
                  ``'tensor'`` → raw histogram logits.
        """
        # Handle multi-view: use first view for loss/predict
        if isinstance(inputs, list):
            # For loss mode, we can optionally use consistency loss across views
            # For now, just use first view
            inputs = inputs[0]

        # Extract features
        feat = self.extract_feat(inputs)

        # Compute histogram logits
        logits = self.head(feat)  # (B, vocab_size)
        pred_hist = F.softmax(logits, dim=-1)  # (B, vocab_size)

        if mode == "loss":
            # Compute aux classification logits (stop-grad from backbone for cls loss)
            cls_logits = None
            if self.aux_cls_head is not None:
                cls_logits = self.aux_cls_head(feat.detach())

            # Get targets from data_samples
            histogram_targets = torch.stack(
                [ds.histogram_target for ds in data_samples], dim=0
            ).to(pred_hist.device)

            labels = torch.tensor(
                [getattr(ds, "dominant_label", -1) for ds in data_samples],
                dtype=torch.long, device=pred_hist.device,
            )

            # Compute loss (backbone features passed for MIL loss)
            losses = self.loss_fn(
                pred_hist=pred_hist,
                target_hist=histogram_targets,
                cls_logits=cls_logits,
                labels=labels,
                backbone_feats=feat,  # For MIL contrastive loss
            )

            return losses

        elif mode == "predict":
            # Attach predictions to data_samples
            if data_samples is not None:
                for ds, hist in zip(data_samples, pred_hist):
                    ds.pred_histogram = hist.detach().cpu()
                    # Compute entropy as a confidence measure
                    h = hist[hist > 0].detach()
                    entropy = -torch.sum(h * torch.log(h + 1e-10))
                    ds.pred_entropy = entropy.cpu()
            return data_samples

        elif mode == "tensor":
            return pred_hist

        else:
            raise ValueError(f"Unknown mode: {mode}")
