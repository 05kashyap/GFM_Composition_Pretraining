"""
Composition-aware model and head for DynamicVis.

``CompositionHead`` replaces the detection-style FPN + RoI + ClsHead used in
the original DynamicVis pretraining.  It adds a projection MLP that maps the
backbone's global average-pooled embedding to the DINOv3 embedding space and
computes the composition-aware loss against pre-computed compositional targets.

``CompositionAwareDynamicVis`` is the top-level model registered with
mmpretrain's MODELS registry so that it can be instantiated from a config dict
via ``custom_imports``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure DynamicVis + losses are importable
_DYNVIS = Path(__file__).resolve().parent.parent / "architectures" / "DynamicVis"
if str(_DYNVIS) not in sys.path:
    sys.path.insert(0, str(_DYNVIS))

from mmengine.model import BaseModel
from mmpretrain.registry import MODELS

# Import our composition loss — registered by side-effect
import losses.composition_loss  # noqa: F401
from losses.composition_loss import CompositionAwareLoss


# --------------------------------------------------------------------------- #
# CompositionHead
# --------------------------------------------------------------------------- #

@MODELS.register_module()
class CompositionHead(nn.Module):
    """Projection MLP + composition-aware loss.

    Architecture::

        backbone features (B, in_channels)
          -> Linear(in_channels, hidden_dim)
          -> LayerNorm -> GELU
          -> Linear(hidden_dim, proj_dim)
          -> [optional L2 normalise — only when loss_type='cosine']
          -> CompositionAwareLoss vs compositional targets

    The final LayerNorm is intentionally omitted so the MLP output has
    diverse norms and directions.  Earlier versions with a final
    LayerNorm (+ L2 normalise) caused all projections to collapse to
    near-identical points, killing both smoothness and alignment gradients.

    Args:
        in_channels:  Backbone embedding dimension (768 for arch='b').
        proj_dim:     Output projection dimension (2048 to match DINOv3).
        hidden_dim:   Hidden layer size in the MLP.
        loss_type:    ``'cosine'`` or ``'mse'``.  ``'mse'`` uses MSE on
            raw projections (optionally standardised targets) and avoids
            the vanishing-gradient problem near alignment.
        tau:          Temperature for contrastive loss.
        lambda_comp:  Weight for alignment loss.
        lambda_contrast: Weight for InfoNCE contrastive loss (0 to disable).
        lambda_smooth:   Weight for spatial smoothness loss.
        standardise_targets: Standardise targets per-dimension (MSE only).
    """

    def __init__(
        self,
        in_channels: int = 768,
        proj_dim: int = 2048,
        hidden_dim: int = 768,
        loss_type: str = "mse",
        tau: float = 0.5,
        lambda_comp: float = 1.0,
        lambda_contrast: float = 0.0,
        lambda_smooth: float = 0.1,
        standardise_targets: bool = True,
    ):
        super().__init__()
        self.loss_type = loss_type

        # Projection MLP.  No final LayerNorm — it was found to collapse
        # all outputs to the same point (cos≈0.997 pairwise).
        self.proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        self.loss_fn = CompositionAwareLoss(
            loss_type=loss_type,
            tau=tau,
            lambda_comp=lambda_comp,
            lambda_contrast=lambda_contrast,
            lambda_smooth=lambda_smooth,
            standardise_targets=standardise_targets,
        )

    def forward(self, feats: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Project backbone features.

        Returns L2-normalised embeddings for cosine mode, raw otherwise.
        """
        x = feats[-1] if isinstance(feats, (tuple, list)) else feats
        x = self.proj(x)
        if self.loss_type == "cosine":
            return F.normalize(x, dim=1)
        return x

    def loss(
        self,
        feats: Tuple[torch.Tensor, ...],
        data_samples: List,
    ) -> Dict[str, torch.Tensor]:
        """Compute composition-aware loss.

        ``data_samples`` must carry per-sample metadata set by the dataset:
          - ``composition_target``  (D,) tensor — the compositional target.
          - ``image_id``            int — source image index.
          - ``cell_row``            int — grid row.
          - ``cell_col``            int — grid column.
        """
        x = feats[-1] if isinstance(feats, (tuple, list)) else feats

        # Project (normalise only for cosine mode)
        f = self.proj(x)
        if self.loss_type == "cosine":
            f = F.normalize(f, dim=1)

        # Gather targets from data_samples
        targets = torch.stack(
            [ds.composition_target for ds in data_samples], dim=0
        ).to(f.device)
        # Only L2-normalise targets for cosine mode
        if self.loss_type == "cosine":
            targets = F.normalize(targets, dim=1)

        # Adjacency metadata (for smoothness)
        image_ids = torch.tensor(
            [ds.image_id for ds in data_samples],
            dtype=torch.long, device=f.device,
        )
        cell_rows = torch.tensor(
            [ds.cell_row for ds in data_samples],
            dtype=torch.long, device=f.device,
        )
        cell_cols = torch.tensor(
            [ds.cell_col for ds in data_samples],
            dtype=torch.long, device=f.device,
        )

        return self.loss_fn(
            f=f,
            t=targets,
            image_ids=image_ids,
            cell_rows=cell_rows,
            cell_cols=cell_cols,
        )

    def predict(
        self,
        feats: Tuple[torch.Tensor, ...],
        data_samples: Optional[List] = None,
    ) -> List:
        """Return projected embeddings (for evaluation).

        Always L2-normalises predictions so cosine similarity with
        targets is meaningful regardless of the training loss type.
        """
        x = feats[-1] if isinstance(feats, (tuple, list)) else feats
        f = F.normalize(self.proj(x), dim=1)  # always normalise for eval
        if data_samples is not None:
            for ds, emb in zip(data_samples, f):
                ds.pred_embedding = emb.detach().cpu()
                if hasattr(ds, 'composition_target'):
                    target = ds.composition_target.to(emb.device)
                    target = F.normalize(target.float(), dim=0)
                    cos_sim = F.cosine_similarity(
                        emb.float().unsqueeze(0),
                        target.unsqueeze(0),
                    )
                    ds.pred_score = cos_sim.detach().cpu()
                    ds.pred_label = 0
                    ds.gt_label = 0
        return data_samples


# --------------------------------------------------------------------------- #
# CompositionAwareDynamicVis — top-level model
# --------------------------------------------------------------------------- #

@MODELS.register_module()
class CompositionAwareDynamicVis(BaseModel):
    """DynamicVis backbone + CompositionHead.

    This replaces ``DynamicVisPretrainClassifier`` for composition-aware
    training.  No FPN / RoI neck — just a global average-pool backbone
    embedding projected to DINOv3 space.

    Config usage::

        model = dict(
            type='CompositionAwareDynamicVis',
            backbone=dict(type='mmpretrain.DynamicVisBackbone', ...),
            head=dict(type='CompositionHead', ...),
        )
    """

    def __init__(
        self,
        backbone: dict,
        head: dict,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        self.backbone = MODELS.build(backbone)
        self.head: CompositionHead = MODELS.build(head)

    def extract_feat(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Run backbone.  Returns tuple of feature tensors."""
        return self.backbone(inputs)

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List] = None,
        mode: str = "loss",
    ):
        """Unified forward.

        Args:
            inputs: (B, 3, H, W) images.
            data_samples: Per-sample metadata.
            mode: ``'loss'`` → dict of losses,
                  ``'predict'`` → data_samples with predictions,
                  ``'tensor'`` → raw projected embeddings.
        """
        feats = self.extract_feat(inputs)

        if mode == "loss":
            return self.head.loss(feats, data_samples)
        elif mode == "predict":
            return self.head.predict(feats, data_samples)
        elif mode == "tensor":
            return self.head(feats)
        else:
            raise ValueError(f"Unknown mode: {mode}")
