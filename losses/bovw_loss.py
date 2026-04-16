"""
BoVW Loss with Sinkhorn EMD approximation and MIL contrastive loss.

Computes the Earth Mover's Distance (EMD) between predicted and target
histogram distributions using the Sinkhorn-Knopp algorithm for differentiable
optimal transport.

Also includes Multi-Instance Learning (MIL) contrastive loss from the vanilla
DynamicVis pretraining, which provides a CLIP-style bidirectional alignment
between object features and learned category embeddings.

The ground cost matrix (pairwise cosine distances between visual vocabulary
centroids) is precomputed in Phase 2 and loaded at model construction.

References:
  - Cuturi, M. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
    NIPS 2013.
  - DynamicVis: Dynamic Vision with Satellite Imagery
"""
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.dist import get_dist_info

from mmpretrain.registry import MODELS


# --------------------------------------------------------------------------- #
# MIL Cross Entropy Loss (from DynamicVis)
# --------------------------------------------------------------------------- #

class MILCrossEntropy(nn.Module):
    """Multi-Instance Learning Cross-Entropy loss.

    CLIP-style contrastive loss that sums softmax probabilities over positive
    targets and takes negative log.

    Used for bidirectional feature-to-class and class-to-feature alignment.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        dim: int = -1,
        avg_positives: bool = False,
    ) -> torch.Tensor:
        """Compute MIL cross-entropy loss.

        Args:
            pred_logits: (N, C) logits from cosine similarity.
            target: (N, C) multi-hot target labels (1 for positive classes).
            dim: Dimension to sum over.
            avg_positives: Whether to average over positive targets (True) or sum (False).

        Returns:
            Scalar loss value.
        """
        probs = F.softmax(pred_logits, dim=-1)

        # Only consider samples with valid (positive) targets
        valid_mask = torch.any(target > 0, dim=-1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_logits.device)

        probs = probs[valid_mask]
        target = target[valid_mask]

        if avg_positives:
            # Average the logits over positive targets
            loss = -torch.log(
                torch.sum(target * probs, dim=dim) / (torch.sum(target, dim=dim) + 1e-6)
            )
        else:
            # Sum the logits over positive targets
            loss = -torch.log(torch.sum(target * probs, dim=dim) + 1e-8)

        return loss.mean()


@MODELS.register_module()
class BoVWLoss(nn.Module):
    """BoVW loss combining Sinkhorn EMD, classification, and MIL contrastive loss.

    Total loss::

        L = lambda_emd * L_emd + lambda_cls * L_cls + lambda_mil * L_mil

    This matches the vanilla DynamicVis pretraining loss structure with:
      - EMD loss for histogram alignment (BoVW-specific)
      - Classification loss with label smoothing (same as vanilla)
      - MIL contrastive loss (same as vanilla DynamicVis)

    Args:
        ground_cost_path: Path to precomputed ground cost matrix (K, K).
            This is the pairwise cosine distance between vocabulary centroids.
        lambda_emd: Weight for the EMD loss term.
        lambda_cls: Weight for the auxiliary classification loss.
        lambda_mil: Weight for the MIL contrastive loss (default 0.25).
        num_classes: Number of fMoW categories (63).
        feature_dim: Backbone feature dimension (768 for arch='b').
        sinkhorn_eps: Entropy regularization for Sinkhorn algorithm.
            Lower values give sharper transport plans but may be numerically unstable.
            Typical range: 0.01 - 0.1.
        sinkhorn_iters: Number of Sinkhorn iterations.
            More iterations give more accurate EMD but slower computation.
        label_smoothing: Label smoothing epsilon for classification loss.
    """

    def __init__(
        self,
        ground_cost_path: Optional[str] = None,
        lambda_emd: float = 1.0,
        lambda_cls: float = 0.5,
        lambda_mil: float = 0.25,
        num_classes: int = 63,
        feature_dim: int = 768,
        sinkhorn_eps: float = 0.05,
        sinkhorn_iters: int = 50,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.lambda_emd = lambda_emd
        self.lambda_cls = lambda_cls
        self.lambda_mil = lambda_mil
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Load precomputed ground cost matrix
        if ground_cost_path is not None:
            cost = torch.from_numpy(np.load(ground_cost_path)).float()
            self.register_buffer('ground_cost', cost)  # (K, K)
            self.vocab_size = cost.shape[0]
        else:
            # Placeholder - will use L1 distance between histogram bins
            self.ground_cost = None
            self.vocab_size = None

        # MIL loss components (from DynamicVis pretraining)
        if lambda_mil > 0:
            self.category_embedding = nn.Embedding(num_classes, feature_dim)
            # Learnable temperature: log(1/0.07) ≈ 2.66 → exp() ≈ 14.3
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.mil_loss_fn = MILCrossEntropy()
        else:
            self.category_embedding = None
            self.logit_scale = None
            self.mil_loss_fn = None

    def sinkhorn(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Sinkhorn-Knopp differentiable EMD approximation.

        Computes the regularized optimal transport distance between predicted
        and target histograms using the iterative Sinkhorn algorithm.

        Uses the primal formulation EMD = <T, C> which is always non-negative.

        Args:
            pred: (B, K) predicted histogram (softmax output, sums to 1).
            target: (B, K) target histogram (L1 normalised, sums to 1).

        Returns:
            Scalar loss (mean over batch).
        """
        B, K = pred.shape
        eps = self.sinkhorn_eps

        # Get ground cost matrix
        if self.ground_cost is not None:
            C = self.ground_cost.unsqueeze(0).expand(B, -1, -1)  # (B, K, K)
        else:
            # Fallback: use bin index distance (L1 on histogram indices)
            idx = torch.arange(K, device=pred.device, dtype=pred.dtype)
            C = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().float()  # (K, K)
            C = C / C.max()  # Normalize to [0, 1]
            C = C.unsqueeze(0).expand(B, -1, -1)

        # Clamp probabilities to avoid log(0)
        pred_clamped = pred.clamp(min=1e-8)
        target_clamped = target.clamp(min=1e-8)

        # Renormalize to ensure sum = 1 after clamping
        pred_clamped = pred_clamped / pred_clamped.sum(dim=1, keepdim=True)
        target_clamped = target_clamped / target_clamped.sum(dim=1, keepdim=True)

        # Log-domain Sinkhorn for numerical stability
        log_pred = torch.log(pred_clamped)    # (B, K)
        log_target = torch.log(target_clamped)  # (B, K)

        # Initialize scaling vectors (in log domain)
        # u, v such that T_ij = u_i * K_ij * v_j where K_ij = exp(-C_ij / eps)
        log_u = torch.zeros_like(pred)  # (B, K)
        log_v = torch.zeros_like(pred)  # (B, K)

        # Precompute log kernel: log K = -C / eps
        log_K = -C / eps  # (B, K, K)

        # Sinkhorn iterations
        for _ in range(self.sinkhorn_iters):
            # Update log_u: u = target / (K @ v)
            # log_u = log_target - logsumexp(log_K + log_v, dim=2)
            log_u = log_target - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)

            # Update log_v: v = pred / (K.T @ u)
            # log_v = log_pred - logsumexp(log_K.T + log_u, dim=1)
            # Note: log_K.T swaps dims 1 and 2
            log_v = log_pred - torch.logsumexp(log_K.transpose(1, 2) + log_u.unsqueeze(1), dim=2)

        # Compute transport plan T = diag(u) @ K @ diag(v)
        # log_T = log_u[:, :, None] + log_K + log_v[:, None, :]
        log_T = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)  # (B, K, K)
        T = torch.exp(log_T)

        # EMD = <T, C> = sum of elementwise product of transport plan and cost
        # This is always non-negative since T >= 0 and C >= 0
        emd = (T * C).sum(dim=[1, 2])

        return emd.mean()

    def forward(
        self,
        pred_hist: torch.Tensor,
        target_hist: torch.Tensor,
        cls_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        backbone_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute BoVW loss.

        Args:
            pred_hist: (B, K) predicted histogram (softmax output).
            target_hist: (B, K) target histogram from Phase 3.
            cls_logits: (B, num_classes) classification logits (optional).
            labels: (B,) ground truth labels, -1 for unlabeled (optional).
            backbone_feats: (B, D) backbone features for MIL loss (optional).

        Returns:
            Dict with 'loss', 'loss_emd', 'loss_cls', 'loss_mil'.
        """
        # EMD loss
        if self.lambda_emd > 0:
            loss_emd = self.sinkhorn(pred_hist, target_hist)
        else:
            loss_emd = pred_hist.new_tensor(0.0)

        # Classification loss
        loss_cls = pred_hist.new_tensor(0.0)
        if (self.lambda_cls > 0 and cls_logits is not None and labels is not None):
            labeled_mask = labels >= 0
            if labeled_mask.any():
                loss_cls = self._label_smooth_ce(
                    cls_logits[labeled_mask],
                    labels[labeled_mask],
                )

        # MIL contrastive loss (CLIP-style bidirectional alignment)
        loss_mil = pred_hist.new_tensor(0.0)
        if (self.lambda_mil > 0 and backbone_feats is not None and labels is not None
                and self.category_embedding is not None):
            labeled_mask = labels >= 0
            if labeled_mask.any():
                loss_mil = self._compute_mil_loss(
                    backbone_feats[labeled_mask],
                    labels[labeled_mask],
                )

        # Total loss
        total = (
            self.lambda_emd * loss_emd
            + self.lambda_cls * loss_cls
            + self.lambda_mil * loss_mil
        )

        return dict(
            loss=total,
            loss_emd=loss_emd,
            loss_cls=loss_cls,
            loss_mil=loss_mil,
        )

    def _compute_mil_loss(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CLIP-style MIL contrastive loss.

        This is bidirectional: feature-to-class and class-to-feature alignment.

        Args:
            feats: (N, D) L2-normalized backbone features for labeled samples.
            labels: (N,) ground truth class labels.

        Returns:
            Scalar loss value.
        """
        # L2 normalize features
        feats = F.normalize(feats, p=2, dim=-1)

        # L2 normalize category embeddings
        class_embeddings = F.normalize(self.category_embedding.weight, p=2, dim=-1)

        # Learnable temperature scaling
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # Compute similarity: (N, num_classes)
        logits_feat_to_class = logit_scale * feats @ class_embeddings.t()
        logits_class_to_feat = logits_feat_to_class.t()  # (num_classes, N)

        # Create one-hot target labels: (N, num_classes)
        target_labels = torch.zeros(
            feats.size(0), self.num_classes,
            device=feats.device, dtype=feats.dtype
        )
        target_labels[torch.arange(feats.size(0), device=feats.device), labels] = 1.0

        # Bidirectional MIL loss
        mil_loss_f2c = self.mil_loss_fn(logits_feat_to_class, target_labels)
        mil_loss_c2f = self.mil_loss_fn(logits_class_to_feat, target_labels.t())

        return (mil_loss_f2c + mil_loss_c2f) / 2.0

    def _label_smooth_ce(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Label-smoothed cross-entropy loss.

        Args:
            logits: (N, num_classes) raw logits.
            labels: (N,) ground truth labels.

        Returns:
            Scalar loss.
        """
        num_classes = logits.size(1)
        eps = self.label_smoothing

        # One-hot with label smoothing
        one_hot = torch.zeros_like(logits).scatter_(
            1, labels.unsqueeze(1), 1.0
        )
        smoothed = one_hot * (1.0 - eps) + eps / num_classes

        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smoothed * log_probs).sum(dim=1).mean()

        return loss
