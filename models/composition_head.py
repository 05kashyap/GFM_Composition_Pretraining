"""
Composition-aware model and head for DynamicVis.

``CompositionHead`` replaces the detection-style FPN + RoI + ClsHead used in
the original DynamicVis pretraining.  It adds a projection MLP that maps the
backbone's global average-pooled embedding to the DINOv3 embedding space and
computes the composition-aware loss against pre-computed compositional targets.

``CompositionAwareDynamicVis`` is the top-level model registered with
mmpretrain's MODELS registry so that it can be instantiated from a config dict
via ``custom_imports``.

Supports two modes for QuerySlotDecoder keys/values:
  - DINOv3 patch embeddings (default): Loads cached .npz embeddings from disk
  - DynamicVis stage-3 tokens: Uses live spatial tokens from the backbone
    (set use_dynamicvis_keys=True)
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

from mmengine.model import BaseModel, BaseDataPreprocessor
from mmpretrain.registry import MODELS

# Import our composition loss — registered by side-effect
import losses.composition_loss  # noqa: F401
from losses.composition_loss import CompositionAwareLoss


# --------------------------------------------------------------------------- #
# MultiViewDataPreprocessor — handles multi-view inputs from QSACL dataset
# --------------------------------------------------------------------------- #

@MODELS.register_module()
class MultiViewDataPreprocessor(BaseDataPreprocessor):
    """Data preprocessor for multi-view QSACL training.

    Handles both single-view (backward compatible) and multi-view inputs:
      - Single-view: inputs is a (C, H, W) tensor per sample
      - Multi-view: inputs is a list of N (C, H_i, W_i) tensors per sample

    Performs:
      1. Move data to device
      2. Stack tensors across batch dimension
      3. No normalization (already done in dataset transforms)

    Args:
        non_blocking: Whether to use non-blocking data transfer.
    """

    def __init__(self, non_blocking: bool = False):
        super().__init__(non_blocking=non_blocking)

    def forward(self, data: dict, training: bool = True) -> dict:
        """Process a batch from pseudo_collate.

        pseudo_collate transposes dict-of-lists into dict where:
            - 'inputs' is a list of N lists (one per view), each inner list has B tensors
            - 'data_samples' is a list of B CompositionDataSample

        We need to stack each view's tensors into (B, C, H, W) tensors.

        Returns:
            dict with:
                - 'inputs': list of N stacked tensors, each (B, C, H, W)
                - 'data_samples': list of CompositionDataSample
        """
        # Move data to device
        data = self.cast_data(data)

        # Handle dict format from pseudo_collate (the default case now)
        if isinstance(data, dict) and 'inputs' in data:
            inputs = data['inputs']
            data_samples = data.get('data_samples', None)

            # Check if multi-view: inputs is a list of lists (views grouped)
            # Structure: inputs[view_idx] = [tensor_s0, tensor_s1, ...] (B unstacked tensors)
            is_multiview = (
                isinstance(inputs, (list, tuple)) and
                len(inputs) > 0 and
                isinstance(inputs[0], (list, tuple))
            )

            if is_multiview:
                # Multi-view: stack each view's tensors
                num_views = len(inputs)
                inputs_list = []

                for view_idx in range(num_views):
                    view_tensors = inputs[view_idx]  # List of B tensors for this view
                    # Ensure they're all tensors
                    if not isinstance(view_tensors, (list, tuple)):
                        # Already stacked somehow
                        inputs_list.append(view_tensors)
                        continue

                    # Pad different-sized local crops to max size
                    shapes = [t.shape for t in view_tensors]
                    if len(set(shapes)) == 1:
                        # All same shape, just stack
                        inputs_list.append(torch.stack(list(view_tensors), dim=0))
                    else:
                        # Different shapes, need to pad
                        max_h = max(s[1] for s in shapes)
                        max_w = max(s[2] for s in shapes)
                        padded = []
                        for t in view_tensors:
                            if t.shape[1] < max_h or t.shape[2] < max_w:
                                pad_h = max_h - t.shape[1]
                                pad_w = max_w - t.shape[2]
                                t = F.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=0)
                            padded.append(t)
                        inputs_list.append(torch.stack(padded, dim=0))

                return {"inputs": inputs_list, "data_samples": data_samples}
            else:
                # Single-view: inputs is a list of B tensors, stack them
                if isinstance(inputs, (list, tuple)):
                    inputs = torch.stack(list(inputs), dim=0)
                return {"inputs": inputs, "data_samples": data_samples}

        # Legacy: handle list of dicts format (backward compatibility)
        if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], dict):
            batch = data
            data_samples = [item["data_samples"] for item in batch]
            first_inputs = batch[0]["inputs"]
            is_multiview = isinstance(first_inputs, list)

            if is_multiview:
                num_views = len(first_inputs)
                inputs_list = []
                for view_idx in range(num_views):
                    view_tensors = [item["inputs"][view_idx] for item in batch]
                    shapes = [t.shape for t in view_tensors]
                    if len(set(shapes)) == 1:
                        inputs_list.append(torch.stack(view_tensors, dim=0))
                    else:
                        max_h = max(s[1] for s in shapes)
                        max_w = max(s[2] for s in shapes)
                        padded = []
                        for t in view_tensors:
                            if t.shape[1] < max_h or t.shape[2] < max_w:
                                pad_h = max_h - t.shape[1]
                                pad_w = max_w - t.shape[2]
                                t = F.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=0)
                            padded.append(t)
                        inputs_list.append(torch.stack(padded, dim=0))
                return {"inputs": inputs_list, "data_samples": data_samples}
            else:
                inputs = torch.stack([item["inputs"] for item in batch], dim=0)
                return {"inputs": inputs, "data_samples": data_samples}

        # Passthrough for already processed data
        return data


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
          -> Linear(hidden_dim, hidden_dim)
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
        proj_dim:     Output projection dimension (256 for PCA targets, 2048 for raw DINOv3).
        hidden_dim:   Hidden layer size in the MLP.
        loss_type:    ``'cosine'`` or ``'mse'``.  ``'mse'`` uses MSE on
            raw projections (optionally standardised targets) and avoids
            the vanishing-gradient problem near alignment.
        tau:          Temperature for contrastive loss.
        lambda_comp:  Weight for alignment loss.
        lambda_var:   Weight for variance regularization (anti-collapse).
        lambda_cov:   Weight for covariance regularization (decorrelation).
        var_gamma:    Target std for variance hinge loss.
        lambda_contrast: Weight for InfoNCE contrastive loss (0 to disable).
        lambda_smooth:   Weight for spatial smoothness loss.
        standardise_targets: Standardise targets per-dimension (MSE only).
        lambda_cls: Weight for auxiliary classification loss (0 to disable).
        lambda_slot_contrast: Weight for per-slot InfoNCE (0 to disable).
        lambda_slot_var: Weight for per-slot variance loss (0 to disable).
        lambda_slot_diversity: Weight for slot diversity (orthogonality) loss (0 to disable).
        slot_var_gamma: Target std for per-slot variance hinge.
        slot_contrast_tau: Temperature for per-slot InfoNCE.
    """

    def __init__(
        self,
        in_channels: int = 768,
        proj_dim: int = 2048,
        hidden_dim: int = 1536,
        loss_type: str = "mse",
        tau: float = 0.5,
        lambda_comp: float = 1.0,
        lambda_cosine: float = 0.0,
        lambda_var: float = 5.0,
        lambda_cov: float = 1.0,
        var_gamma: float = 1.0,
        lambda_contrast: float = 0.0,
        lambda_smooth: float = 0.1,
        standardise_targets: bool = True,
        lambda_cls: float = 0.0,
        lambda_slot_contrast: float = 0.0,
        lambda_slot_var: float = 0.0,
        lambda_slot_diversity: float = 0.0,
        slot_var_gamma: float = 1.0,
        slot_contrast_tau: float = 0.1,
    ):
        super().__init__()
        self.loss_type = loss_type

        # 3-layer projection MLP.  No final LayerNorm — it was found to
        # collapse all outputs to the same point (cos~0.997 pairwise).
        self.proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        self.loss_fn = CompositionAwareLoss(
            loss_type=loss_type,
            tau=tau,
            lambda_comp=lambda_comp,
            lambda_cosine=lambda_cosine,
            lambda_var=lambda_var,
            lambda_cov=lambda_cov,
            var_gamma=var_gamma,
            lambda_contrast=lambda_contrast,
            lambda_smooth=lambda_smooth,
            standardise_targets=standardise_targets,
            lambda_cls=lambda_cls,
            lambda_slot_contrast=lambda_slot_contrast,
            lambda_slot_var=lambda_slot_var,
            lambda_slot_diversity=lambda_slot_diversity,
            slot_var_gamma=slot_var_gamma,
            slot_contrast_tau=slot_contrast_tau,
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
        cls_logits: Optional[torch.Tensor] = None,
        slots: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        online_slots_list: Optional[List[torch.Tensor]] = None,
        target_slots: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute composition-aware loss.

        ``data_samples`` must carry per-sample metadata set by the dataset:
          - ``composition_target``  (D,) tensor — the compositional target.
          - ``image_id``            int — source image index.
          - ``cell_row``            int — grid row.
          - ``cell_col``            int — grid column.
          - ``dominant_label``      int — fMoW class (0–62) or -1 (optional).

        Args:
            feats: Backbone features (tuple or single tensor).
            data_samples: Per-sample metadata from the dataset.
            cls_logits: Pre-computed logits from aux classification head
                (stop-gradiented).  Used for L_cls.
            slots: (deprecated) Either:
                - (B, num_queries, slot_dim) single slot tensor (single view)
                - List of N tensors (B, num_queries, slot_dim) for multi-view
            online_slots_list: List of N-1 tensors (B, m, slot_dim) for online views.
                Gradients flow through these.
            target_slots: (B, m, slot_dim) target slots (stop-gradiented).
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

        # fMoW dominant labels (for label-guided losses)
        dominant_labels = torch.tensor(
            [getattr(ds, "dominant_label", -1) for ds in data_samples],
            dtype=torch.long, device=f.device,
        )

        return self.loss_fn(
            f=f,
            t=targets,
            image_ids=image_ids,
            cell_rows=cell_rows,
            cell_cols=cell_cols,
            dominant_labels=dominant_labels,
            cls_logits=cls_logits,
            slots=slots,
            online_slots_list=online_slots_list,
            target_slots=target_slots,
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
                    ds.pred_label = torch.tensor(0)
                    ds.gt_label = torch.tensor(0)
        return data_samples


# --------------------------------------------------------------------------- #
# CompositionAwareDynamicVis — top-level model
# --------------------------------------------------------------------------- #

@MODELS.register_module()
class CompositionAwareDynamicVis(BaseModel):
    """DynamicVis backbone + CompositionHead + optional QuerySlotDecoder.

    This replaces ``DynamicVisPretrainClassifier`` for composition-aware
    training.  No FPN / RoI neck — just a global average-pool backbone
    embedding projected to DINOv3 space.

    Optional components (activated by setting ``num_classes > 0``):
      - **Auxiliary classification head** — ``nn.Linear(backbone_dim, num_classes)``
        for label-guided training.  Receives stop-gradiented backbone features
        so the classification gradient shapes the backbone but not the
        composition projection.  Excluded from final backbone export.

    Optional QuerySlotDecoder (activated by setting ``num_queries > 0``):
      - **Slot decoder** — Cross-attention module that produces slot embeddings
        from patch embeddings for per-slot contrastive learning.

    Two modes for slot decoder keys/values:
      - **DINOv3 patch embeddings** (default, ``use_dynamicvis_keys=False``):
        Loads cached .npz embeddings from data_samples. Backbone uses
        ``out_type='avg_featmap'``.
      - **DynamicVis stage-3 tokens** (``use_dynamicvis_keys=True``):
        Uses live spatial tokens (B, 256, 768) from backbone stage-3.
        Backbone uses ``out_type='featmap'`` and we manually compute the
        global average for the head.

    EMA Target Network (BYOL-style):
      When ``ema_tau > 0``, creates an EMA copy of the slot decoder for the
      target branch. The EMA copy has no gradients and is updated after each
      optimizer step with: theta_ema = tau * theta_ema + (1 - tau) * theta_online.
      This prevents trivial collapse in BYOL-style contrastive learning.

    Config usage::

        model = dict(
            type='CompositionAwareDynamicVis',
            backbone=dict(type='mmpretrain.DynamicVisBackbone', ...),
            head=dict(type='CompositionHead', ...),
            num_classes=63,
            num_queries=16,
            slot_dim=256,
            use_dynamicvis_keys=False,  # or True for ablation
            ema_tau=0.996,  # EMA decay for target slot decoder
        )
    """

    NUM_CLASSES = 63  # fMoW categories

    def __init__(
        self,
        backbone: dict,
        head: dict,
        num_classes: int = 0,
        num_queries: int = 16,
        slot_dim: int = 256,
        patch_dim: int = 2048,
        num_heads: int = 8,
        num_registers: int = 4,
        use_dynamicvis_keys: bool = False,
        conditioned: bool = True,
        backbone_dim: int = 768,
        ema_tau: float = 0.996,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )

        self.use_dynamicvis_keys = use_dynamicvis_keys
        self.backbone_dim = backbone_dim

        # ---- Configure backbone output mode based on use_dynamicvis_keys ----
        if use_dynamicvis_keys:
            # Need spatial feature map for slot decoder keys/values
            backbone = backbone.copy()
            backbone['out_type'] = 'featmap'
            backbone['out_indices'] = (3,)
            # Override patch_dim for DynamicVis stage-3 tokens (768-d)
            patch_dim = 768

        self.backbone = MODELS.build(backbone)
        self.head: CompositionHead = MODELS.build(head)

        # ---- Auxiliary classification head (training scaffold) ----
        self.num_classes = num_classes
        if num_classes > 0:
            # Infer backbone dim from head's in_channels config
            inferred_backbone_dim = self.head.proj[0].in_features  # first Linear layer
            self.aux_cls_head = nn.Linear(inferred_backbone_dim, num_classes)
        else:
            self.aux_cls_head = None

        # ---- QuerySlotDecoder for per-slot contrastive learning ----
        self.num_queries = num_queries
        self.ema_tau = ema_tau
        self.slot_decoder_ema = None  # Will be set after slot_decoder creation

        if num_queries > 0:
            from models.query_slot_decoder import QuerySlotDecoder
            self.slot_decoder = QuerySlotDecoder(
                patch_dim=patch_dim,
                num_queries=num_queries,
                slot_dim=slot_dim,
                num_heads=num_heads,
                conditioned=conditioned,
                backbone_dim=backbone_dim,
                num_registers=num_registers,
            )

            # ---- EMA target network for BYOL-style slot learning ----
            # Creates a separate copy of slot_decoder with EMA-smoothed weights.
            # The EMA copy is NOT registered as a submodule, so its parameters
            # won't appear in model.parameters() or be included in the optimizer.
            if ema_tau > 0:
                from utils.ema import EMAModel
                self.slot_decoder_ema = EMAModel(self.slot_decoder, tau=ema_tau)
        else:
            self.slot_decoder = None

    def extract_feat(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Run backbone.  Returns tuple of feature tensors."""
        return self.backbone(inputs)

    def update_ema(self) -> None:
        """Update EMA target network weights from online slot decoder.

        Should be called AFTER each optimizer.step() during training.
        The EMA update rule is:
            theta_ema = tau * theta_ema + (1 - tau) * theta_online

        This is a no-op if EMA is disabled (ema_tau <= 0 or no slot_decoder).
        """
        if self.slot_decoder_ema is not None and self.slot_decoder is not None:
            self.slot_decoder_ema.update(self.slot_decoder)

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
                - List of N tensors, each (B, 3, H_i, W_i) for multi-view
            data_samples: Per-sample metadata.  Each sample should have:
                - composition_target: target embedding
                - image_id, cell_row, cell_col: adjacency metadata
                - dominant_label: fMoW class (optional)
                - patch_embeddings: (N_patches, patch_dim) DINOv3 embeddings
                  (only required when use_dynamicvis_keys=False)
            mode: ``'loss'`` → dict of losses,
                  ``'predict'`` → data_samples with predictions,
                  ``'tensor'`` → raw projected embeddings.
        """
        # Check if multi-view input
        is_multiview = isinstance(inputs, list)

        if is_multiview:
            # Multi-view: process each view through backbone
            return self._forward_multiview(inputs, data_samples, mode)
        else:
            # Single view: original path
            return self._forward_single(inputs, data_samples, mode)

    def _forward_single(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List],
        mode: str,
    ):
        """Single-view forward (backward compatible)."""
        raw_feats = self.extract_feat(inputs)

        # ---- Handle different backbone output modes ----
        if self.use_dynamicvis_keys:
            # Backbone in featmap mode: (B, 768, 16, 16)
            feat_map = raw_feats[0] if isinstance(raw_feats, (tuple, list)) else raw_feats
            # Global average pool for head and aux classifier
            global_feat = feat_map.mean(dim=[-2, -1])  # (B, 768)
            # Reshape spatial tokens for slot decoder: (B, 256, 768)
            slot_keys = feat_map.flatten(2).transpose(1, 2)
            # Wrap as tuple for compatibility with head
            feats = (global_feat,)
        else:
            # Backbone in avg_featmap mode: (B, 768)
            feats = raw_feats
            slot_keys = None  # Will load from data_samples

        if mode == "loss":
            # Pre-compute aux classification logits if the head exists
            cls_logits = None
            backbone_feat = feats[-1] if isinstance(feats, (tuple, list)) else feats
            if self.aux_cls_head is not None and data_samples is not None:
                # Stop-grad: cls gradient should not flow through composition head
                cls_logits = self.aux_cls_head(backbone_feat.detach())

            # ---- Get slot decoder keys/values ----
            if self.use_dynamicvis_keys:
                # Use DynamicVis stage-3 tokens as keys/values
                patch_embeddings = slot_keys
            else:
                # Extract and stack patch embeddings from data_samples
                patch_embeddings = None
                if (self.slot_decoder is not None and data_samples is not None
                        and hasattr(data_samples[0], 'patch_embeddings')
                        and data_samples[0].patch_embeddings is not None):
                    # Stack patch embeddings: (B, N_patches, patch_dim)
                    patch_embeddings = torch.stack(
                        [ds.patch_embeddings for ds in data_samples], dim=0
                    ).to(inputs.device)

            # Run slot decoder if available and keys/values provided
            slots = None
            if self.slot_decoder is not None and patch_embeddings is not None:
                # Pass backbone_feat for conditioned queries (gradient path to backbone)
                slots = self.slot_decoder(patch_embeddings, backbone_feat=backbone_feat)

            losses = self.head.loss(
                feats, data_samples,
                cls_logits=cls_logits,
                slots=slots,
            )

            return losses
        elif mode == "predict":
            return self.head.predict(feats, data_samples)
        elif mode == "tensor":
            return self.head(feats)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _forward_multiview(
        self,
        inputs_list: List[torch.Tensor],
        data_samples: Optional[List],
        mode: str,
    ):
        """Multi-view forward with asymmetric BYOL-style slot learning.

        Implements DINO/BYOL multi-crop strategy:
        - Target view: index 1 (second global view) — stop-gradiented backbone
        - Online views: indices 0, 2, 3, 4, 5, 6, 7 — gradients flow through backbone

        This asymmetric formulation prevents symmetric collapse: the online branch
        must produce slots that match a target it cannot directly influence.
        """
        if mode != "loss":
            # For predict/tensor mode, just use first view
            return self._forward_single(inputs_list[0], data_samples, mode)

        # Get device from first view
        device = inputs_list[0].device

        # ---- Get shared patch embeddings (view-agnostic) ----
        patch_embeddings = None
        if not self.use_dynamicvis_keys:
            if (self.slot_decoder is not None and data_samples is not None
                    and hasattr(data_samples[0], 'patch_embeddings')
                    and data_samples[0].patch_embeddings is not None):
                patch_embeddings = torch.stack(
                    [ds.patch_embeddings for ds in data_samples], dim=0
                ).to(device)

        # ---- Helper to extract features for a view ----
        def extract_view_feats(view_inputs):
            raw_feats = self.extract_feat(view_inputs)
            if self.use_dynamicvis_keys:
                feat_map = raw_feats[0] if isinstance(raw_feats, (tuple, list)) else raw_feats
                global_feat = feat_map.mean(dim=[-2, -1])
                slot_keys = feat_map.flatten(2).transpose(1, 2)
                return (global_feat,), slot_keys
            else:
                return raw_feats, None

        def get_patch_embeddings_for_view(slot_keys):
            if self.use_dynamicvis_keys:
                return slot_keys
            return patch_embeddings

        # ---- Process view 0 first (reuse for cls/projection + online slot) ----
        feats_0, slot_keys_0 = extract_view_feats(inputs_list[0])
        backbone_feat_0 = feats_0[-1] if isinstance(feats_0, (tuple, list)) else feats_0

        # Cls logits and projection from view 0
        first_view_feats = feats_0
        first_view_cls_logits = None
        if self.aux_cls_head is not None and data_samples is not None:
            first_view_cls_logits = self.aux_cls_head(backbone_feat_0.detach())
        first_view_projection = self.head(feats_0)

        # ---- Target view: index 1 (second global view) — STOP-GRADIENT ----
        # Uses EMA slot decoder (if available) for BYOL-style asymmetric learning.
        # The EMA network provides a slowly-evolving target that the online
        # network cannot directly influence, preventing trivial collapse.
        target_slots = None
        if self.slot_decoder is not None:
            with torch.no_grad():
                feats_1, slot_keys_1 = extract_view_feats(inputs_list[1])
                backbone_feat_1 = feats_1[-1] if isinstance(feats_1, (tuple, list)) else feats_1
                view_patch_emb_1 = get_patch_embeddings_for_view(slot_keys_1)
                if view_patch_emb_1 is not None:
                    # Use EMA slot decoder for target branch (proper BYOL)
                    # Fall back to regular slot_decoder if EMA not available
                    if self.slot_decoder_ema is not None:
                        # Sync device on first use (handles DDP moving model to GPU after init)
                        self.slot_decoder_ema.sync_device(self.slot_decoder)
                        slot_decoder_target = self.slot_decoder_ema.module
                    else:
                        slot_decoder_target = self.slot_decoder
                    target_slots = slot_decoder_target(
                        view_patch_emb_1, backbone_feat=backbone_feat_1.detach()
                    )

        # ---- Online views: indices 0, 2, 3, 4, 5, 6, 7 — GRADIENTS FLOW ----
        online_slots_list = []
        if self.slot_decoder is not None:
            # View 0: already have backbone_feat_0
            view_patch_emb_0 = get_patch_embeddings_for_view(slot_keys_0)
            if view_patch_emb_0 is not None:
                online_slots_0 = self.slot_decoder(view_patch_emb_0, backbone_feat=backbone_feat_0)
                online_slots_list.append(online_slots_0)

            # Views 2-7: local crops — detach backbone_feat to save memory
            for view_idx in range(2, len(inputs_list)):
                feats_v, slot_keys_v = extract_view_feats(inputs_list[view_idx])
                backbone_feat_v = feats_v[-1] if isinstance(feats_v, (tuple, list)) else feats_v
                view_patch_emb_v = get_patch_embeddings_for_view(slot_keys_v)
                if view_patch_emb_v is not None:
                    # Detach backbone feat for local crops to save memory
                    online_slots_v = self.slot_decoder(view_patch_emb_v, backbone_feat=backbone_feat_v.detach())
                    online_slots_list.append(online_slots_v)

        # ---- Compute losses using asymmetric slot structure ----
        losses = self.head.loss(
            first_view_feats, data_samples,
            cls_logits=first_view_cls_logits,
            online_slots_list=online_slots_list if len(online_slots_list) > 0 else None,
            target_slots=target_slots,
        )

        return losses

