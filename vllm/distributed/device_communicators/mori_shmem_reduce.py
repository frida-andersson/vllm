# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Optimized all-reduce for dispatch-free Expert Parallelism.

This module provides an all-reduce wrapper for the dispatch-free EP
finalize step, where each GPU's partial MoE output must be summed across
all EP ranks.

It delegates to the GroupCoordinator's all_reduce which goes through
vLLM's standard dispatch chain (quick_reduce -> custom_allreduce ->
pynccl), ensuring CUDAGraph compatibility on all paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)


class MoriShmemReducer:
    """All-reduce wrapper for dispatch-free EP.

    Delegates to the GroupCoordinator's all_reduce, which internally
    selects the best strategy (quick_reduce, custom_allreduce, or
    pynccl) and is CUDAGraph-safe on all paths.

    Args:
        ep_group: The GroupCoordinator for the EP/TP group.
        world_size: Number of GPUs in the group.
        rank: This GPU's rank in the group.
    """

    def __init__(
        self,
        ep_group: GroupCoordinator,
        world_size: int,
        rank: int,
    ):
        self.ep_group = ep_group
        self.world_size = world_size
        self.rank = rank
        self.strategy = "group_coordinator"

        logger.info(
            "MoriShmemReducer initialized: strategy=%s, world_size=%d, "
            "rank=%d",
            self.strategy,
            self.world_size,
            self.rank,
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """Perform all-reduce across EP ranks.

        Args:
            input_: The tensor to all-reduce (typically MoE expert output,
                    shape [M, K], dtype bf16).

        Returns:
            A new tensor containing the sum across all EP ranks.
        """
        if self.world_size <= 1:
            return input_

        return self.ep_group.all_reduce(input_)
