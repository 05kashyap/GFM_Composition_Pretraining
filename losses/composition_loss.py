"""
Composition-aware loss for DynamicVis training.

Components:
  1. **Alignment (L_comp)** — distillation from DINOv3 teacher embeddings.
     Supports ``'cosine'`` (1 - cos) and ``'mse'`` (mean squared error)
     modes.  MSE on standardised targets is recommended — it provides
     richer gradient signal in all 2048 dimensions and avoids the
     vanishing-gradient problem of cosine loss near alignment.
  2. **Variance (L_var)** — VICReg-style hinge loss that forces each
     embedding dimension to maintain variance >= gamma across the batch.
     Prevents representation collapse (all embeddings -> constant).
  3. **Covariance (L_cov)** — penalises correlation between different
     embedding dimensions, forcing the network to use all dimensions.
  4. **Smoothness** — embeddings from spatially adjacent grid cells within
     the same image should be similar (cosine distance).  Now supports
     label-structured smoothness that reduces target similarity at class
     boundaries.
  5. **L_cls** — Auxiliary label-smoothed CE on backbone features (stop-grad).
  6. **L_slot_contrast** — Per-slot supervised contrastive loss using fMoW
     labels: same-class slots across the batch are positives.  This avoids
     the degenerate self-similarity formulation of naive InfoNCE.
  7. **L_slot_var** — Per-slot variance hinge loss on slot embeddings.

An optional InfoNCE contrastive term is retained for experimentation but
is disabled by default (``lambda_contrast=0``) because DINOv3 targets
are too correlated (mean pairwise cos ~ 0.70) for InfoNCE to work.

References:
  - VICReg: Bardes et al., "VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning", ICLR 2022.
  - notebooks/lossfn-test.ipynb — prototype implementation
"""
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS


@MODELS.register_module()
class CompositionAwareLoss(nn.Module):
    """Composition-aware loss combining alignment, variance/covariance
    regularization, (optional) contrastive, smoothness, and label-guided terms.

    Total loss::

        L = lambda_comp         * L_comp
          + lambda_cosine       * L_cosine
          + lambda_var          * L_var
          + lambda_cov          * L_cov
          + lambda_contrast     * L_contrast
          + lambda_smooth       * L_smooth
          + lambda_cls          * L_cls          (label-smoothed CE on backbone feats)
          + lambda_slot_contrast * L_slot_contrast (per-slot InfoNCE)
          + lambda_slot_var     * L_slot_var     (per-slot variance hinge)

    Args:
        loss_type: ``'cosine'`` uses ``1 - cos(f, t)`` on L2-normalised
            vectors.  ``'mse'`` uses MSE on raw (optionally standardised)
            embeddings — recommended for stronger gradients.
        tau: Temperature for the InfoNCE contrastive loss.
        lambda_comp: Weight for the alignment term.
        lambda_cosine: Weight for an additional cosine alignment term
            applied on **raw** (pre-standardisation) embeddings.  Provides
            directional gradient signal complementary to MSE.
        lambda_var: Weight for the variance regularization term.
            Prevents collapse by forcing per-dimension variance >= gamma.
        lambda_cov: Weight for the covariance regularization term.
            Decorrelates embedding dimensions.
        var_gamma: Target standard deviation for the variance hinge loss.
        lambda_contrast: Weight for the contrastive term (0 to disable).
        lambda_smooth: Weight for the spatial smoothness term.  Uses a
            target-relative hinge formulation that is collapse-resistant.
            When labels are available, similarity targets at class boundaries
            are reduced.
        standardise_targets: When True **and** ``loss_type='mse'``,
            standardise targets to zero-mean / unit-variance per
            dimension using a running EMA.  This amplifies gradient
            signal in the few dimensions that actually vary across the
            DINOv3 target space.
        lambda_cls: Weight for the auxiliary classification loss.
            Uses label-smoothed cross-entropy on stop-gradiented backbone
            features.  0 to disable.
        lambda_slot_contrast: Weight for per-slot InfoNCE contrastive loss
            on QuerySlotDecoder outputs.  0 to disable.
        lambda_slot_var: Weight for per-slot variance hinge loss.
            0 to disable.
        lambda_slot_diversity: Weight for slot diversity (orthogonality) loss.
            Penalises high cosine similarity between different slots within
            the same sample. Prevents backbone-conditioned offsets from
            pushing all queries to the same direction. 0 to disable.
        slot_var_gamma: Target std for per-slot variance hinge.
        slot_contrast_tau: Temperature for per-slot InfoNCE.
        label_smoothing: Epsilon for label-smoothed CE in L_cls.
    """

    def __init__(
        self,
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
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        assert loss_type in ("cosine", "mse"), f"Unknown loss_type: {loss_type}"
        self.loss_type = loss_type
        self.tau = tau
        self.lambda_comp = lambda_comp
        self.lambda_cosine = lambda_cosine
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.var_gamma = var_gamma
        self.lambda_contrast = lambda_contrast
        self.lambda_smooth = lambda_smooth
        self.standardise_targets = standardise_targets and (loss_type == "mse")
        self.lambda_cls = lambda_cls
        self.lambda_slot_contrast = lambda_slot_contrast
        self.lambda_slot_var = lambda_slot_var
        self.lambda_slot_diversity = lambda_slot_diversity
        self.slot_var_gamma = slot_var_gamma
        self.slot_contrast_tau = slot_contrast_tau
        self.label_smoothing = label_smoothing

        # Running target statistics (initialised lazily on first forward).
        # We register real (1-d) placeholder tensors instead of None so that
        # (a) DDP broadcasts them at construction, (b) .state_dict() always
        # includes them, and (c) .to(device) moves them correctly.
        if self.standardise_targets:
            self.register_buffer("_target_mean", torch.zeros(1))
            self.register_buffer("_target_std", torch.ones(1))
            self.register_buffer(
                "_target_init", torch.zeros(1, dtype=torch.bool)
            )
            self._ema_momentum = 0.1

    # ------------------------------------------------------------------
    # Distributed helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _all_gather_with_grad(t: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all DDP ranks, preserving local gradients.

        Returns the concatenation of ``t`` from every rank.  The local
        rank's shard keeps its autograd graph so that gradients flow back
        through the variance / covariance computation.
        """
        if not (torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1):
            return t
        world_size = torch.distributed.get_world_size()
        gathered = [torch.zeros_like(t) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, t.detach())
        # Replace local shard with the original (grad-attached) tensor
        gathered[torch.distributed.get_rank()] = t
        return torch.cat(gathered, dim=0)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def forward(
        self,
        f: torch.Tensor,
        t: torch.Tensor,
        image_ids: Optional[torch.Tensor] = None,
        cell_rows: Optional[torch.Tensor] = None,
        cell_cols: Optional[torch.Tensor] = None,
        dominant_labels: Optional[torch.Tensor] = None,
        backbone_feat: Optional[torch.Tensor] = None,
        cls_logits: Optional[torch.Tensor] = None,
        slots: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        online_slots_list: Optional[List[torch.Tensor]] = None,
        target_slots: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined composition-aware loss.

        Args:
            f: (B, D) projected embeddings.  For ``loss_type='cosine'``,
               must be L2-normalised; for ``'mse'`` they are raw.
            t: (B, D) compositional targets.
            image_ids, cell_rows, cell_cols: adjacency metadata for
               smoothness (``None`` -> smoothness = 0).
            dominant_labels: (B,) int64 fMoW category labels (0–62), or
               -1 for unlabeled cells.  ``None`` disables all label-guided
               losses.
            backbone_feat: (B, backbone_dim) stop-gradiented backbone
               features.  Currently unused — retained for forward compatibility.
            cls_logits: (B, num_classes) pre-computed logits from the aux
               classification head (stop-gradiented from backbone).  Used
               by ``L_cls``.
            slots: (deprecated) Either:
               - (B, num_queries, slot_dim) single slot tensor (single view)
               - List of N tensors (B, num_queries, slot_dim) for multi-view
               Used for per-slot contrastive and variance losses.
            online_slots_list: List of N-1 tensors (B, m, slot_dim) for online views.
               Gradients flow through these. Used for BYOL-style slot loss.
            target_slots: (B, m, slot_dim) target slots (stop-gradiented).
               Used for BYOL-style slot loss.

        Returns:
            dict with ``loss``, ``loss_comp``, ``loss_var``, ``loss_cov``,
            ``loss_contrast``, ``loss_smooth``, ``loss_cls``,
            ``loss_slot_contrast``, ``loss_slot_var``.
        """
        f_fp32 = f.float()
        t_fp32 = t.float()

        # Gather embeddings across GPUs for VICReg losses.
        # Variance and covariance need large batches to give meaningful
        # statistics; per-GPU batches can be as small as 4 samples.
        f_gathered = self._all_gather_with_grad(f_fp32)

        # ---- alignment ----
        # Skip alignment when targets are empty (manifest-only mode) or disabled
        if t_fp32.shape[1] == 0 or self.lambda_comp == 0:
            loss_comp = f.new_tensor(0.0)
        elif self.loss_type == "cosine":
            loss_comp = self._alignment_cosine(f_fp32, t_fp32)
        else:
            loss_comp = self._alignment_mse(f_fp32, t_fp32)

        # ---- cosine direction alignment (on raw / pre-standardisation) ----
        if self.lambda_cosine > 0 and t_fp32.shape[1] > 0:
            loss_cosine = self._alignment_cosine(f_fp32, t_fp32)
        else:
            loss_cosine = f.new_tensor(0.0)

        # ---- variance regularization (anti-collapse) ----
        if self.lambda_var > 0:
            loss_var = self._variance(f_gathered)
        else:
            loss_var = f.new_tensor(0.0)

        # ---- covariance regularization (decorrelation) ----
        if self.lambda_cov > 0:
            loss_cov = self._covariance(f_gathered)
        else:
            loss_cov = f.new_tensor(0.0)

        # ---- contrastive (optional) ----
        if self.lambda_contrast > 0 and t_fp32.shape[1] > 0:
            loss_contrast = self._contrastive(f_fp32, t_fp32)
        else:
            loss_contrast = f.new_tensor(0.0)

        # ---- smoothness (target-relative, now label-structured) ----
        # Skip smoothness when targets are empty (manifest-only mode)
        if t_fp32.shape[1] == 0 or self.lambda_smooth == 0:
            loss_smooth = f.new_tensor(0.0)
        else:
            loss_smooth = self._smoothness(
                f_fp32, t_fp32, image_ids, cell_rows, cell_cols, dominant_labels
            )

        # ---- label-guided losses (gated on label availability) ----
        # Use graph-connected zeros when cls_logits is provided so that
        # aux_cls_head parameters always participate in the backward pass
        # (avoids needing find_unused_parameters=True in DDP).
        loss_cls = (cls_logits * 0).sum() if cls_logits is not None else f.new_tensor(0.0)

        labeled_mask = dominant_labels >= 0 if dominant_labels is not None else None

        # L_cls: label-smoothed CE on pre-computed logits from aux head
        # (no collectives — safe to gate on local data)
        if (self.lambda_cls > 0 and cls_logits is not None
                and labeled_mask is not None and labeled_mask.any()):
            loss_cls = self._label_smooth_ce_from_logits(
                cls_logits, dominant_labels, labeled_mask
            )

        # ---- Per-slot losses (from QuerySlotDecoder) ----
        loss_slot_contrast = f.new_tensor(0.0)
        loss_slot_var = f.new_tensor(0.0)
        loss_slot_diversity = f.new_tensor(0.0)

        # BYOL-style asymmetric slot loss (preferred)
        if online_slots_list is not None and target_slots is not None:
            # Per-slot BYOL contrastive loss
            if self.lambda_slot_contrast > 0:
                loss_slot_contrast = self._slot_byol(online_slots_list, target_slots)

            # Per-slot variance hinge loss (use first online view only)
            if self.lambda_slot_var > 0 and len(online_slots_list) > 0:
                loss_slot_var = self._slot_variance(online_slots_list[0])

            # Slot diversity loss (use first online view only)
            if self.lambda_slot_diversity > 0 and len(online_slots_list) > 0:
                loss_slot_diversity = self._slot_query_diversity(online_slots_list[0])

        # Legacy symmetric slot loss (deprecated, for backward compatibility)
        elif slots is not None:
            # Handle both single-view (tensor) and multi-view (list of tensors)
            if isinstance(slots, list):
                slots_list = slots
            else:
                slots_list = [slots]

            # Per-slot contrastive loss (multi-view QSACL) - deprecated
            if self.lambda_slot_contrast > 0:
                loss_slot_contrast = self._slot_infonce(slots_list)

            # Per-slot variance hinge loss (use first view only)
            if self.lambda_slot_var > 0:
                loss_slot_var = self._slot_variance(slots_list[0])

            # Slot diversity loss (use first view only)
            if self.lambda_slot_diversity > 0:
                loss_slot_diversity = self._slot_query_diversity(slots_list[0])

        total = (
            self.lambda_comp * loss_comp
            + self.lambda_cosine * loss_cosine
            + self.lambda_var * loss_var
            + self.lambda_cov * loss_cov
            + self.lambda_contrast * loss_contrast
            + self.lambda_smooth * loss_smooth
            + self.lambda_cls * loss_cls
            + self.lambda_slot_contrast * loss_slot_contrast
            + self.lambda_slot_var * loss_slot_var
            + self.lambda_slot_diversity * loss_slot_diversity
        )

        return dict(
            loss=total,
            loss_comp=loss_comp,
            loss_cosine=loss_cosine,
            loss_var=loss_var,
            loss_cov=loss_cov,
            loss_contrast=loss_contrast,
            loss_smooth=loss_smooth,
            loss_cls=loss_cls,
            loss_slot_contrast=loss_slot_contrast,
            loss_slot_var=loss_slot_var,
            loss_slot_diversity=loss_slot_diversity,
        )

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def _alignment_cosine(
        self, f: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Cosine alignment: ``mean(1 - cos(f, t))``.  Range [0, 2]."""
        return 1.0 - F.cosine_similarity(f, t, dim=1).mean()

    def _alignment_mse(
        self, f: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """MSE on raw projections vs (optionally standardised) targets.

        When ``standardise_targets=True``, targets are z-scored using a
        running EMA of per-dimension mean / std so that every dimension
        contributes equally to the gradient.

        **IMPORTANT**: Both f and t are mean-centered across the batch
        before computing MSE. This removes the "common direction" component
        that causes collapse when targets are highly correlated (cos ~0.75).
        Without centering, the MSE gradient pulls all embeddings toward the
        same target centroid, collapsing representations. With centering,
        MSE only penalizes differences in relative structure.

        Expects fp32 inputs (caller is responsible for upcasting).
        """
        t_std = self._standardise(t) if self.standardise_targets else t

        # Mean-center to remove collapse-inducing common direction
        f_centered = f - f.mean(dim=0, keepdim=True)
        t_centered = t_std - t_std.mean(dim=0, keepdim=True)

        return F.mse_loss(f_centered, t_centered)

    # ---- VICReg-style regularization ----

    def _variance(self, f: torch.Tensor) -> torch.Tensor:
        """Variance regularization (VICReg).

        Hinge loss that activates when any embedding dimension's standard
        deviation across the batch falls below ``var_gamma``::

            L_var = (1/D) * sum_j max(0, gamma - std(f_j))

        This creates a strong repulsive gradient that breaks representation
        collapse.  At collapse (std ~ 0), the gradient is large and pushes
        embeddings apart; once std >= gamma, the term vanishes.
        """
        # std across batch dimension for each feature dim
        std_f = f.std(dim=0)  # (D,)
        return F.relu(self.var_gamma - std_f).mean()

    def _covariance(self, f: torch.Tensor) -> torch.Tensor:
        """Covariance regularization (VICReg).

        Penalises squared off-diagonal elements of the embedding
        covariance matrix, encouraging different dimensions to encode
        independent information::

            L_cov = (1/D) * sum_{i != j} C_{i,j}^2

        where C is the (D, D) covariance matrix of the batch embeddings.
        """
        B, D = f.shape
        f_centered = f - f.mean(dim=0)
        cov = (f_centered.T @ f_centered) / max(B - 1, 1)  # (D, D)
        # Zero out the diagonal — we only penalise off-diagonal
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag / D

    # ---- target standardisation (EMA) ----

    def _standardise(self, t: torch.Tensor) -> torch.Tensor:
        """Z-score targets using an EMA of batch statistics.

        Bug-fix notes (B1 / B3):
        - B1: Buffers are now real tensors (not None), resized once on first
          batch via ``torch.zeros_like`` + ``.copy_()``, then updated in-place
          with ``.lerp_()``.  This keeps them in ``_buffers`` so that
          ``state_dict()``, ``.to(device)``, and DDP all track them.
        - B3: In distributed training, ``batch_mean`` / ``batch_std`` are
          all-reduced across ranks before the EMA update, so every GPU
          maintains identical running statistics.
        """
        with torch.no_grad():
            batch_mean = t.mean(dim=0)
            batch_std = t.std(dim=0).clamp(min=1e-6)

            # B3: synchronise batch stats across GPUs so EMA stays consistent.
            # Use SUM + manual divide instead of AVG because the gloo
            # backend (used for >2 MIG slices) does not support ReduceOp.AVG.
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                torch.distributed.all_reduce(
                    batch_mean, op=torch.distributed.ReduceOp.SUM
                )
                batch_mean.div_(world_size)
                torch.distributed.all_reduce(
                    batch_std, op=torch.distributed.ReduceOp.SUM
                )
                batch_std.div_(world_size)

            if not self._target_init.item():
                # First batch -> resize placeholders and seed with batch stats
                self._target_mean = torch.zeros_like(batch_mean)
                self._target_std = torch.ones_like(batch_std)
                self._target_mean.copy_(batch_mean)
                self._target_std.copy_(batch_std)
                self._target_init.fill_(True)
            elif self.training:
                m = self._ema_momentum
                self._target_mean.lerp_(batch_mean, m)
                self._target_std.lerp_(batch_std, m)

        return (t - self._target_mean) / self._target_std

    # ---- contrastive ----

    def _contrastive(self, f: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """InfoNCE contrastive loss (L2-normalises internally).

        Expects fp32 inputs (caller is responsible for upcasting).
        """
        f_n = F.normalize(f, dim=1)
        t_n = F.normalize(t, dim=1)
        sim = torch.matmul(f_n, t_n.T) / self.tau
        labels = torch.arange(f.size(0), device=f.device)
        return F.cross_entropy(sim, labels)

    # ---- smoothness ----

    def _smoothness(
        self,
        f: torch.Tensor,
        t: torch.Tensor,
        image_ids: Optional[torch.Tensor],
        cell_rows: Optional[torch.Tensor],
        cell_cols: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Target-relative smoothness between spatially adjacent patches.

        Hinge loss: ``mean(max(0, cos(t_i, t_j) - cos(f_i, f_j)))``

        This pushes the model's embeddings for adjacent cells to be at
        least as similar as their targets are.  Unlike absolute cosine
        smoothness (``1 - cos(f_i, f_j)``), this formulation is
        collapse-resistant — it does not push similarity beyond what the
        targets require, so it provides zero gradient once the model
        matches the target similarity structure.

        When ``labels`` are provided (fMoW dominant labels), target
        similarity at class boundaries (adjacent cells with different
        labels) is reduced by 50%.  This injects semantic structure into
        the smoothness signal and should activate the loss even when the
        model already matches raw DINOv3 target similarities.
        """
        if image_ids is None or cell_rows is None or cell_cols is None:
            return f.new_tensor(0.0)

        B = f.size(0)
        if B < 2:
            return f.new_tensor(0.0)

        same_img = image_ids.unsqueeze(0) == image_ids.unsqueeze(1)
        row_diff = (cell_rows.unsqueeze(0) - cell_rows.unsqueeze(1)).abs()
        col_diff = (cell_cols.unsqueeze(0) - cell_cols.unsqueeze(1)).abs()
        adjacent = same_img & ((row_diff + col_diff) == 1)
        adjacent = adjacent.triu(diagonal=1)
        idx_i, idx_j = adjacent.nonzero(as_tuple=True)

        if idx_i.numel() == 0:
            if self.lambda_smooth > 0 and self.training:
                import warnings
                warnings.warn(
                    "Smoothness loss: no adjacent pairs found in this batch. "
                    "Check that image_id/cell_row/cell_col are set correctly "
                    "in data_samples and that ImageGroupSampler groups enough "
                    "cells from the same image.",
                    stacklevel=2,
                )
            return f.new_tensor(0.0)

        cos_f = F.cosine_similarity(f[idx_i], f[idx_j], dim=1)
        cos_t = F.cosine_similarity(t[idx_i], t[idx_j], dim=1)

        # Label-structured smoothness: reduce target similarity at class
        # boundaries so the model learns to separate different-class cells
        if labels is not None:
            labels_i = labels[idx_i]
            labels_j = labels[idx_j]
            # Both cells must be labeled and have different labels
            cross_boundary = (
                (labels_i >= 0) & (labels_j >= 0) & (labels_i != labels_j)
            )
            cos_t = torch.where(cross_boundary, cos_t * 0.5, cos_t)

        return F.relu(cos_t - cos_f).mean()

    # ---- label-guided losses ----

    def _label_smooth_ce_from_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        labeled_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Label-smoothed CE from pre-computed classification logits.

        Args:
            logits: (B, num_classes) raw logits from aux_cls_head.
            labels: (B,) int64 ground-truth labels (-1 for unlabeled).
            labeled_mask: (B,) bool mask where labels >= 0.
        """
        logits_labeled = logits[labeled_mask]
        labels_labeled = labels[labeled_mask]

        num_classes = logits.size(1)
        eps = self.label_smoothing

        # One-hot with label smoothing
        one_hot = torch.zeros_like(logits_labeled).scatter_(
            1, labels_labeled.unsqueeze(1), 1.0
        )
        smoothed = one_hot * (1.0 - eps) + eps / num_classes

        log_probs = F.log_softmax(logits_labeled, dim=1)
        loss = -(smoothed * log_probs).sum(dim=1).mean()
        return loss

    # ---- Per-slot losses (QuerySlotDecoder) ----

    def _hungarian_match_slots(
        self,
        online_slots: torch.Tensor,
        target_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Find optimal slot correspondence using Hungarian matching.

        For each sample in the batch, finds the permutation of target slots
        that maximizes total cosine similarity with online slots.

        Args:
            online_slots: (B, m, d) online slot embeddings
            target_slots: (B, m, d) target slot embeddings

        Returns:
            matched_targets: (B, m, d) target slots reordered to match online slots
        """
        from scipy.optimize import linear_sum_assignment

        B, m, d = online_slots.shape
        device = online_slots.device

        # Normalise for cosine similarity
        online_n = F.normalize(online_slots, dim=-1)  # (B, m, d)
        target_n = F.normalize(target_slots, dim=-1)  # (B, m, d)

        # Compute pairwise cosine similarity: (B, m_online, m_target)
        # sim[b, i, j] = cosine_sim(online[b, i], target[b, j])
        sim_matrix = torch.bmm(online_n, target_n.transpose(1, 2))  # (B, m, m)

        # Cost matrix = negative similarity (Hungarian minimizes cost)
        # Convert to float32 for numpy (bfloat16 not supported)
        cost_matrix = -sim_matrix.detach().cpu().float().numpy()

        # Find optimal assignment for each sample in batch
        matched_indices = []
        for b in range(B):
            row_ind, col_ind = linear_sum_assignment(cost_matrix[b])
            matched_indices.append(col_ind)

        # Reorder target slots according to matching
        # Convert list of numpy arrays to single numpy array first (faster)
        import numpy as np
        matched_indices = torch.from_numpy(np.array(matched_indices)).to(device=device, dtype=torch.long)  # (B, m)

        # Gather matched targets: for each batch, select target slots in matched order
        # matched_indices[b, i] = which target slot matches online slot i
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, m)  # (B, m)
        matched_targets = target_slots[batch_indices, matched_indices]  # (B, m, d)

        return matched_targets

    def _slot_byol(
        self,
        online_slots_list: List[torch.Tensor],
        target_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Asymmetric BYOL-style slot loss with Hungarian matching.

        For each online view, finds optimal slot correspondence via Hungarian
        matching, then computes negative cosine similarity against matched
        target slots.

        Hungarian matching solves the slot assignment problem: different
        augmentations cause slots to attend to different regions, so slot i
        in online may not correspond to slot i in target. We find the optimal
        1-to-1 matching that maximizes total similarity.

        Args:
            online_slots_list: List of tensors, each (B, m, slot_dim).
                Gradients flow through these.
            target_slots: (B, m, slot_dim) stop-gradiented target slots.

        Returns:
            Scalar loss (more negative = online slots better match target).
        """
        if not online_slots_list:
            return target_slots.new_tensor(0.0)

        loss = 0.0
        for online_slots in online_slots_list:
            # Find optimal slot correspondence via Hungarian matching
            matched_targets = self._hungarian_match_slots(online_slots, target_slots)

            # Normalise both branches per slot
            online_n = F.normalize(online_slots, dim=-1)   # (B, m, slot_dim)
            target_n = F.normalize(matched_targets, dim=-1)   # (B, m, slot_dim)

            # Negative cosine similarity averaged over batch and slots
            cos_sim = (online_n * target_n).sum(dim=-1)    # (B, m)
            loss += -cos_sim.mean()

        return loss / len(online_slots_list)

    def _slot_infonce(
        self, slots_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Multi-view QSACL (Query-Slot Attention Contrastive Learning).

        Computes InfoNCE per slot position across all view pairs. For each
        slot position i:
          - Positives: slot i from view A of cell X paired with slot i from
            view B of the same cell X (for all view pairs A, B)
          - Negatives: slot i from any view of cell Y (different cells)

        This is proper two-view contrastive learning where positives are the
        same cell seen under different augmentations. No labels are used —
        the contrastive signal comes purely from view correspondence.

        With N=8 views per cell (2 global + 6 local), each cell contributes
        N×(N-1)/2 positive pairs per slot position.

        Args:
            slots_list: List of N tensors, each (B, m, slot_dim) where:
                - N = number of views
                - B = batch size
                - m = number of query slots
                - slot_dim = slot embedding dimension

        Returns:
            Scalar loss averaged over all slots and view pairs.
        """
        N = len(slots_list)
        if N < 2:
            # Need at least 2 views for contrastive learning
            return slots_list[0].new_tensor(0.0)

        B, m, d = slots_list[0].shape
        if B < 2:
            return slots_list[0].new_tensor(0.0)

        temperature = self.slot_contrast_tau
        loss = 0.0
        n_pairs = 0

        for i in range(m):
            # Collect slot i from all views: (N, B, d)
            slot_i_views = torch.stack([s[:, i, :] for s in slots_list], dim=0)
            slot_i_views = F.normalize(slot_i_views, dim=-1)  # L2 normalize

            # For each pair of views (v1, v2), compute InfoNCE
            for v1 in range(N):
                for v2 in range(v1 + 1, N):
                    z1 = slot_i_views[v1]  # (B, d)
                    z2 = slot_i_views[v2]  # (B, d)

                    # Similarity matrix: (B, B) where [i, j] = sim(z1[i], z2[j])
                    # Diagonal [i, i] = sim of same cell across views (positive)
                    logits = torch.matmul(z1, z2.T) / temperature  # (B, B)

                    # Labels: each sample's positive is at the diagonal position
                    labels = torch.arange(B, device=z1.device)

                    # Symmetric InfoNCE: both directions
                    loss_v1_to_v2 = F.cross_entropy(logits, labels)
                    loss_v2_to_v1 = F.cross_entropy(logits.T, labels)
                    loss += (loss_v1_to_v2 + loss_v2_to_v1) / 2
                    n_pairs += 1

        # Average over all slots and view pairs
        if n_pairs == 0:
            return slots_list[0].new_tensor(0.0)

        return loss / (m * n_pairs)

    def _slot_variance(self, slots: torch.Tensor) -> torch.Tensor:
        """Per-slot variance hinge loss on QuerySlotDecoder outputs.

        VICReg-style variance regularization applied per-slot to prevent
        collapse.  Each slot embedding should have variance >= gamma across
        the batch in each dimension.

        This is easier to optimize than whole-embedding variance because
        slot_dim (256) is smaller than proj_dim (2048) and slots are
        semantically focused.

        Args:
            slots: (B, num_queries, slot_dim) slot embeddings.

        Returns:
            Scalar loss averaged over all slots.
        """
        B, m, d = slots.shape
        if B < 2:
            return slots.new_tensor(0.0)

        gamma = self.slot_var_gamma
        loss = 0.0

        for i in range(m):
            # Standard deviation across batch for slot i
            std = slots[:, i, :].std(dim=0)  # (slot_dim,)
            # Hinge loss: penalize dimensions with std < gamma
            loss += F.relu(gamma - std).mean()

        return loss / m

    def _slot_query_diversity(self, slots: torch.Tensor) -> torch.Tensor:
        """Slot diversity (orthogonality) loss.

        Penalises high cosine similarity between different query slots within
        the same sample. This prevents the backbone-conditioned offsets from
        pushing all queries to the same direction.

        The loss is the mean squared off-diagonal cosine similarity:
            L = (1 / m(m-1)) * sum_{i != j} cos(slot_i, slot_j)^2

        Args:
            slots: (B, num_queries, slot_dim) slot embeddings.

        Returns:
            Scalar loss averaged over batch and slot pairs.
        """
        B, m, d = slots.shape
        if m < 2:
            return slots.new_tensor(0.0)

        # L2-normalize slots for cosine similarity
        slots_n = F.normalize(slots, dim=-1)  # (B, m, slot_dim)

        # Compute pairwise cosine similarity between slots within each sample
        sim = torch.bmm(slots_n, slots_n.transpose(1, 2))  # (B, m, m)

        # Only penalize off-diagonal elements (different slots)
        mask = ~torch.eye(m, dtype=torch.bool, device=slots.device)  # (m, m)
        off_diag = sim[:, mask].pow(2).mean()  # mean over (B, m*(m-1))

        return off_diag

