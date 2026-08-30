"""
Image-grouped sampler for composition-aware training.

Groups cells from the same source image into the same mini-batch so that
the smoothness loss can find spatially adjacent pairs.

Registration:
    ``'ImageGroupSampler'`` — registered with mmengine's DATA_SAMPLERS.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterator, Optional, Sized

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class ImageGroupSampler(Sampler):
    """Sampler that groups cells from the same image together.

    Each "group" is all cells belonging to the same source image.  Groups
    are shuffled across epochs, and cells within a group are shuffled too.
    Groups are packed sequentially into the iteration order, so a batch
    of size B will typically contain several cells from the same image
    (enabling the smoothness loss to find adjacent pairs).

    Supports distributed training — groups are partitioned across ranks.

    Args:
        dataset: The dataset to sample from.  Must have a ``cells``
            attribute (list of dicts with ``'image_path'``).
        cells_per_group: Maximum number of cells to include per image
            group in each sampling round.  Set to 0 or -1 for all cells.
        shuffle: Whether to shuffle groups and intra-group order.
        seed: Random seed for reproducibility across DDP ranks.
        round_up: If True, pad the last batch to keep all batches equal.
    """

    def __init__(
        self,
        dataset: Sized,
        cells_per_group: int = 8,
        shuffle: bool = True,
        seed: int = 42,
        round_up: bool = True,
    ):
        self.dataset = dataset
        self.cells_per_group = cells_per_group if cells_per_group > 0 else 999999
        self.shuffle = shuffle
        self.seed = seed
        self.round_up = round_up
        self.epoch = 0

        # Distributed setup
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

        # Build image → cell indices mapping
        self._groups: dict[str, list[int]] = defaultdict(list)
        for idx, cell in enumerate(dataset.cells):
            self._groups[cell["image_path"]].append(idx)

        self._group_keys = sorted(self._groups.keys())

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            # Shuffle group order (same order on all ranks for partitioning)
            group_order = torch.randperm(len(self._group_keys), generator=g).tolist()
        else:
            group_order = list(range(len(self._group_keys)))

        # Partition groups across ranks
        # Each rank gets every num_replicas-th group
        rank_groups = group_order[self.rank :: self.num_replicas]

        indices = []
        for gi in rank_groups:
            key = self._group_keys[gi]
            cell_indices = self._groups[key].copy()

            if self.shuffle:
                random.Random(self.seed + self.epoch + gi).shuffle(cell_indices)

            # Take up to cells_per_group cells from this image
            indices.extend(cell_indices[: self.cells_per_group])

        return iter(indices)

    def __len__(self) -> int:
        total = 0
        for key in self._group_keys:
            total += min(len(self._groups[key]), self.cells_per_group)
        # Approximate per-rank size
        return math.ceil(total / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
