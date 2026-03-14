"""
Composition-aware loss for DynamicVis training.

Two main components:
  1. **Alignment (L_comp)** — distillation from DINOv3 teacher embeddings.
     Supports ``'cosine'`` (1 − cos) and ``'mse'`` (mean squared error)
     modes.  MSE on standardised targets is recommended — it provides
     richer gradient signal in all 2048 dimensions and avoids the
     vanishing-gradient problem of cosine loss near alignment.
  2. **Smoothness** — embeddings from spatially adjacent grid cells within
     the same image should be similar (cosine distance).

An optional InfoNCE contrastive term is retained for experimentation but
is disabled by default (``lambda_contrast=0``) because DINOv3 targets
are too correlated (mean pairwise cos ≈ 0.70) for InfoNCE to work.

References:
  • notebooks/lossfn-test.ipynb — prototype implementation
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS


@MODELS.register_module()
class CompositionAwareLoss(nn.Module):
    """Composition-aware loss combining alignment, (optional) contrastive,
    and smoothness terms.

    Total loss::

        L = λ_comp     * L_comp
          + λ_contrast * L_contrast
          + λ_smooth   * L_smooth

    Args:
        loss_type: ``'cosine'`` uses ``1 − cos(f, t)`` on L2-normalised
            vectors.  ``'mse'`` uses MSE on raw (optionally standardised)
            embeddings — recommended for stronger gradients.
        tau: Temperature for the InfoNCE contrastive loss.
        lambda_comp: Weight for the alignment term.
        lambda_contrast: Weight for the contrastive term (0 to disable).
        lambda_smooth: Weight for the spatial smoothness term.
        standardise_targets: When True **and** ``loss_type='mse'``,
            standardise targets to zero-mean / unit-variance per
            dimension using a running EMA.  This amplifies gradient
            signal in the few dimensions that actually vary across the
            DINOv3 target space.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        tau: float = 0.5,
        lambda_comp: float = 1.0,
        lambda_contrast: float = 0.0,
        lambda_smooth: float = 0.1,
        standardise_targets: bool = True,
    ):
        super().__init__()
        assert loss_type in ("cosine", "mse"), f"Unknown loss_type: {loss_type}"
        self.loss_type = loss_type
        self.tau = tau
        self.lambda_comp = lambda_comp
        self.lambda_contrast = lambda_contrast
        self.lambda_smooth = lambda_smooth
        self.standardise_targets = standardise_targets and (loss_type == "mse")

        # Running target statistics (initialised lazily on first forward)
        if self.standardise_targets:
            self.register_buffer("_target_mean", None)
            self.register_buffer("_target_std", None)
            self._ema_momentum = 0.01

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
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined composition-aware loss.

        Args:
            f: (B, D) projected embeddings.  For ``loss_type='cosine'``,
               must be L2-normalised; for ``'mse'`` they are raw.
            t: (B, D) compositional targets.
            image_ids, cell_rows, cell_cols: adjacency metadata for
               smoothness (``None`` → smoothness = 0).

        Returns:
            dict with ``loss``, ``loss_comp``, ``loss_contrast``,
            ``loss_smooth``.
        """
        # ---- alignment ----
        if self.loss_type == "cosine":
            loss_comp = self._alignment_cosine(f, t)
        else:
            loss_comp = self._alignment_mse(f, t)

        # ---- contrastive (optional) ----
        if self.lambda_contrast > 0:
            loss_contrast = self._contrastive(f, t)
        else:
            loss_contrast = f.new_tensor(0.0)

        # ---- smoothness ----
        loss_smooth = self._smoothness(f, image_ids, cell_rows, cell_cols)

        total = (
            self.lambda_comp * loss_comp
            + self.lambda_contrast * loss_contrast
            + self.lambda_smooth * loss_smooth
        )

        return dict(
            loss=total,
            loss_comp=loss_comp,
            loss_contrast=loss_contrast,
            loss_smooth=loss_smooth,
        )

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def _alignment_cosine(
        self, f: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Cosine alignment: ``mean(1 − cos(f, t))``.  Range [0, 2]."""
        return 1.0 - F.cosine_similarity(f.float(), t.float(), dim=1).mean()

    def _alignment_mse(
        self, f: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """MSE on raw projections vs (optionally standardised) targets.

        When ``standardise_targets=True``, targets are z-scored using a
        running EMA of per-dimension mean / std so that every dimension
        contributes equally to the gradient.
        """
        f_fp32 = f.float()
        t_fp32 = t.float()

        if self.standardise_targets:
            t_fp32 = self._standardise(t_fp32)

        return F.mse_loss(f_fp32, t_fp32)

    # ---- target standardisation (EMA) ----

    def _standardise(self, t: torch.Tensor) -> torch.Tensor:
        """Z-score targets using an EMA of batch statistics."""
        with torch.no_grad():
            batch_mean = t.mean(dim=0)
            batch_std = t.std(dim=0).clamp(min=1e-6)

            if self._target_mean is None:
                # First batch → use batch stats directly
                self._target_mean = batch_mean.clone()
                self._target_std = batch_std.clone()
            elif self.training:
                m = self._ema_momentum
                self._target_mean.lerp_(batch_mean, m)
                self._target_std.lerp_(batch_std, m)

        return (t - self._target_mean) / self._target_std

    # ---- contrastive ----

    def _contrastive(self, f: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """InfoNCE contrastive loss (L2-normalises internally)."""
        f_n = F.normalize(f.float(), dim=1)
        t_n = F.normalize(t.float(), dim=1)
        sim = torch.matmul(f_n, t_n.T) / self.tau
        labels = torch.arange(f.size(0), device=f.device)
        return F.cross_entropy(sim, labels)

    # ---- smoothness ----

    def _smoothness(
        self,
        f: torch.Tensor,
        image_ids: Optional[torch.Tensor],
        cell_rows: Optional[torch.Tensor],
        cell_cols: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Cosine smoothness between spatially adjacent patches.

        ``1 − cos(f_i, f_j)`` for adjacent cells (same image, Manhattan
        distance == 1).  Scale-invariant, so it works regardless of
        whether projections are L2-normalised.
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
            return f.new_tensor(0.0)

        f_fp32 = f.float()
        cos_sim = F.cosine_similarity(f_fp32[idx_i], f_fp32[idx_j], dim=1)
        return (1.0 - cos_sim).mean()


