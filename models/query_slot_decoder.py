"""
QuerySlotDecoder — Cross-attention module for per-slot contrastive learning.

Takes patch embeddings and produces slot-level embeddings via learnable query
cross-attention.  Each slot learns to attend to specific semantic patterns in
the patch space.

Supports two sources for keys/values:
  - **DINOv3 patch embeddings** (default): 28 patches × 2048-d from offline cache
  - **DynamicVis stage-3 tokens**: 256 tokens × 768-d from live backbone features

**Backbone-Conditioned Residual Queries** (when conditioned=True):
  The decoder uses fixed base queries for slot diversity, plus image-specific
  offsets derived from the backbone's global feature vector. This creates a
  gradient path from slot losses back to the backbone while preserving slot
  diversity.

Architecture::

    patch_embeddings (B, N_patches, patch_dim)
        -> Linear(patch_dim, slot_dim) -> LayerNorm            [patch projection]
        -> queries (m, slot_dim) [learnable base queries]
        -> (optional) + query_proj(backbone_feat)              [image-specific offsets]
        -> MultiheadAttention(q=queries, k=v=projected_patches)
        -> LayerNorm
        -> slots (B, m, slot_dim)

The slot embeddings are used for per-slot supervised contrastive learning, where
samples with the same fMoW class label are treated as positives for each slot.

Usage:
    # DINOv3 patch embeddings with backbone conditioning (default)
    decoder = QuerySlotDecoder(patch_dim=2048, num_queries=16, slot_dim=256,
                               conditioned=True, backbone_dim=768)
    slots = decoder(patch_embeddings, backbone_feat=global_feat)

    # Without conditioning (backward compatible)
    decoder = QuerySlotDecoder(patch_dim=2048, num_queries=16, slot_dim=256,
                               conditioned=False)
    slots = decoder(patch_embeddings)
"""

from typing import Optional

import torch
import torch.nn as nn


class QuerySlotDecoder(nn.Module):
    """Query-based slot decoder for per-slot contrastive learning.

    Args:
        patch_dim: Dimension of input patch embeddings.
            - 2048 for DINOv3 cls_avg pooling (default)
            - 768 for DynamicVis stage-3 tokens
        num_queries: Number of learnable query slots (default 16).
        slot_dim: Output dimension per slot (default 256).
        num_heads: Number of attention heads (default 8).
        conditioned: If True, add image-specific residual offsets to queries
            derived from backbone features. Creates gradient path to backbone.
        backbone_dim: Dimension of backbone global features (default 768).
            Only used when conditioned=True.
        num_registers: Number of learnable register tokens (default 4).
            Registers are appended to keys/values and give queries something
            to "dump" artifact attention onto, freeing semantic slots.

    Raises:
        ValueError: If patch_dim <= 0 or slot_dim <= 0.
    """

    def __init__(
        self,
        patch_dim: int = 2048,
        num_queries: int = 16,
        slot_dim: int = 256,
        num_heads: int = 8,
        conditioned: bool = True,
        backbone_dim: int = 768,
        num_registers: int = 4,
    ):
        super().__init__()

        # Validate parameters
        if patch_dim <= 0:
            raise ValueError(
                f"patch_dim must be positive, got {patch_dim}. "
                f"Use 2048 for DINOv3 embeddings or 768 for DynamicVis stage-3 tokens."
            )
        if slot_dim <= 0:
            raise ValueError(
                f"slot_dim must be positive, got {slot_dim}. "
                f"Typical values: 256 (default) or 128."
            )
        if num_queries <= 0:
            raise ValueError(f"num_queries must be positive, got {num_queries}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if slot_dim % num_heads != 0:
            raise ValueError(
                f"slot_dim ({slot_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.patch_dim = patch_dim
        self.num_queries = num_queries
        self.slot_dim = slot_dim
        self.conditioned = conditioned
        self.num_registers = num_registers

        # Learnable base query vectors — each query learns to attend to specific
        # semantic patterns in the patch embedding space
        self.queries = nn.Parameter(torch.randn(num_queries, slot_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Learnable register tokens — absorb artifact attention patterns,
        # freeing semantic slots to attend to meaningful content
        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(num_registers, slot_dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None

        # Backbone-conditioned query projection (creates gradient path to backbone)
        if conditioned:
            self.query_proj = nn.Linear(backbone_dim, num_queries * slot_dim)
            # Initialize with small weights so offsets start near zero
            # Training begins with approximately fixed queries, growing more
            # image-specific as training proceeds
            nn.init.normal_(self.query_proj.weight, std=0.01)
            nn.init.zeros_(self.query_proj.bias)
        else:
            self.query_proj = None

        # Project patch embeddings to slot dimension
        self.patch_proj = nn.Linear(patch_dim, slot_dim)

        # Cross-attention: queries attend to patch embeddings
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=slot_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Layer norms for stability
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_patches = nn.LayerNorm(slot_dim)

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        backbone_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute slot embeddings from patch embeddings.

        Args:
            patch_embeddings: (B, N_patches, patch_dim) input embeddings.
                - For DINOv3: N_patches=28, patch_dim=2048
                - For DynamicVis stage-3: N_patches=256, patch_dim=768
            backbone_feat: (B, backbone_dim) global backbone features.
                Required when conditioned=True for gradient path to backbone.

        Returns:
            slots: (B, num_queries, slot_dim) slot embeddings.
        """
        B = patch_embeddings.shape[0]

        # Project and normalise keys/values
        kv = self.norm_patches(self.patch_proj(patch_embeddings))  # (B, N, slot_dim)

        # Append register tokens to kv — these absorb artifact attention
        if self.num_registers > 0 and self.register_tokens is not None:
            registers = self.register_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, R, slot_dim)
            kv = torch.cat([kv, registers], dim=1)  # (B, N+R, slot_dim)

        # Base queries — fixed learnable, ensure slot diversity
        base_q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, m, slot_dim)

        if self.conditioned and backbone_feat is not None:
            # Image-specific residual offsets — gradient path to backbone
            offsets = self.query_proj(backbone_feat)  # (B, m * slot_dim)
            offsets = offsets.view(B, self.num_queries, self.slot_dim)  # (B, m, slot_dim)
            queries = base_q + offsets  # (B, m, slot_dim)
        else:
            queries = base_q

        # Cross-attention: queries attend over patches + registers
        slots, _ = self.cross_attn(queries, kv, kv)

        # Output normalization
        slots = self.norm_slots(slots)  # (B, m, slot_dim)

        return slots
