# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Optimized all-reduce for dispatch-free Expert Parallelism using MORI P2P.

Uses MORI's symmetric shared memory infrastructure for direct GPU-to-GPU
P2P reads via XGMI, avoiding the overhead of RCCL / vLLM custom-allreduce
dispatch chains.

Falls back to GroupCoordinator.all_reduce() if MORI shmem cannot be
initialised.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)

# The MORI P2P all-reduce uses cross-device barriers that are not
# compatible with CUDA graph capture.  Default to GroupCoordinator's
# CUDAGraph-safe all-reduce (QuickReduce / custom_allreduce on MI300X).
# Set VLLM_MORI_AR_P2P=1 to use MORI P2P (only with --enforce-eager).
_USE_MORI_P2P = os.environ.get("VLLM_MORI_AR_P2P", "0") == "1"


class MoriShmemReducer:
    """All-reduce wrapper for dispatch-free EP.

    On MI300X (intra-node XGMI) this uses MORI's P2P all-reduce kernel
    which reads each peer's contribution directly from symmetric shared
    memory and sums them in a single GPU kernel launch.

    If MORI shmem is unavailable (e.g. not ROCm, shmem init failed),
    falls back to GroupCoordinator.all_reduce().

    Args:
        ep_group: The GroupCoordinator for the EP/TP group.
        world_size: Number of GPUs in the group.
        rank: This GPU's rank in the group.
        max_num_tokens: Max tokens per inference step (for buffer pre-alloc).
        hidden_dim: Hidden dimension K of MoE output tensor.
    """

    def __init__(
        self,
        ep_group: GroupCoordinator,
        world_size: int,
        rank: int,
        max_num_tokens: int = 256,
        hidden_dim: int = 3072,
    ):
        self.ep_group = ep_group
        self.world_size = world_size
        self.rank = rank
        self.max_num_tokens = max_num_tokens
        self.hidden_dim = hidden_dim
        self.strategy = "group_coordinator"  # default fallback
        self._mori_ar_op = None

        if not _USE_MORI_P2P:
            logger.info(
                "MoriShmemReducer: Using GroupCoordinator all-reduce "
                "(CUDAGraph-safe). Set VLLM_MORI_AR_P2P=1 + "
                "--enforce-eager to use MORI P2P kernel."
            )
            return

        # Try to set up MORI P2P all-reduce (requires --enforce-eager).
        try:
            self._init_mori_allreduce()
        except Exception as e:
            logger.warning(
                "MoriShmemReducer: Failed to initialize MORI P2P "
                "all-reduce (%s). Falling back to GroupCoordinator.",
                e,
            )

    def _init_mori_allreduce(self):
        """Create the MORI P2P all-reduce handle.

        If MORI shmem has already been initialized (e.g. by
        MoriAll2AllManager when DP > 1), we reuse it.  Otherwise we
        bootstrap shmem ourselves using the EP / TP group's CPU
        process-group -- this is the common case for TP+EP with DP=1.
        """
        import mori
        import torch
        from mori.ops.allreduce import MoriAllReduceOp

        # ── Ensure MORI shmem is initialized ─────────────────────────
        if not mori.shmem.shmem_is_initialized():
            logger.info(
                "MoriShmemReducer: MORI shmem not yet initialized. "
                "Bootstrapping from EP/TP group (rank=%d, ws=%d).",
                self.rank,
                self.world_size,
            )
            cpu_group = self.ep_group.cpu_group
            # Register the Gloo CPU group so MORI's bootstrap can use
            # it for coordination (allgather of IPC handles, etc.).
            torch._C._distributed_c10d._register_process_group(
                "mori", cpu_group
            )
            mori.shmem.shmem_torch_process_group_init("mori")
            logger.info("MoriShmemReducer: MORI shmem initialized OK.")
        else:
            logger.info(
                "MoriShmemReducer: MORI shmem already initialized "
                "(by MoriAll2AllManager or another component)."
            )

        npes = mori.shmem.shmem_npes()
        my_pe = mori.shmem.shmem_mype()
        logger.info(
            "MoriShmemReducer: shmem ready (npes=%d, my_pe=%d).",
            npes,
            my_pe,
        )

        # ── Create the lightweight all-reduce handle ──────────────────
        self._mori_ar_op = MoriAllReduceOp(
            rank=my_pe,
            world_size=npes,
            max_num_tokens=self.max_num_tokens,
            hidden_dim=self.hidden_dim,
        )
        self.strategy = "mori_p2p"
        logger.info(
            "MoriShmemReducer: Using MORI P2P all-reduce "
            "(world_size=%d, rank=%d, max_tokens=%d, hidden=%d).",
            npes,
            my_pe,
            self.max_num_tokens,
            self.hidden_dim,
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """Perform all-reduce across EP ranks.

        Args:
            input_: The tensor to all-reduce (typically MoE expert output,
                    shape [M, K], dtype bf16).

        Returns:
            Tensor containing the sum across all EP ranks.
        """
        if self.world_size <= 1:
            return input_

        if self._mori_ar_op is not None:
            M = input_.shape[0]
            if M <= self.max_num_tokens:
                out = self._mori_ar_op.allreduce(input_)
                return out[:M]
            # Input exceeds pre-allocated MORI buffer (e.g. during
            # profiling run).  Fall back to GroupCoordinator.
            # This is NOT on the CUDAGraph-captured path.

        # Fallback to GroupCoordinator dispatch chain.
        return self.ep_group.all_reduce(input_)
