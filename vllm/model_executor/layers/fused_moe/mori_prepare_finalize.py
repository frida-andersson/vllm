# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform

if TYPE_CHECKING:
    import mori

logger = init_logger(__name__)

# Feature flag to enable dispatch-free optimization for TP+EP
_DISPATCH_FREE_ENABLED = os.environ.get("VLLM_MORI_EP_DISPATCH_FREE", "1") == "1"


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using MoRI kernels.

    Two operating modes:

    1. **Standard mode** -- uses MORI dispatch/combine (All-to-All) for
       DP+EP topologies.  Requires a valid ``mori_op``.

    2. **Dispatch-free mode** (TP+EP) -- all GPUs already hold the full
       token batch after TP all-reduce, so dispatch is skipped entirely.
       Each GPU filters locally, computes on its experts, then all-reduces
       using a MORI-shmem-backed reducer that selects the optimal
       communication strategy.
       ``mori_op`` may be ``None`` in this mode.
    """

    def __init__(
        self,
        mori_op: mori.ops.EpDispatchCombineOp | None,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
        # Optional parameters for TP+EP dispatch-free optimization
        num_local_experts: int | None = None,
        rank_expert_offset: int | None = None,
        ep_group: object | None = None,  # GroupCoordinator (graph-safe)
        enable_dispatch_free: bool = True,
        hidden_dim: int = 3072,
    ):
        super().__init__()
        self.mori_op = mori_op
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch

        # TP+EP dispatch-free optimization
        self.use_dispatch_free = (
            _DISPATCH_FREE_ENABLED
            and enable_dispatch_free
            and num_local_experts is not None
            and rank_expert_offset is not None
            and ep_group is not None
        )
        self.num_local_experts = num_local_experts
        self.rank_expert_offset = rank_expert_offset
        self.ep_group = ep_group
        self._shmem_reducer = None

        if self.use_dispatch_free:
            # Initialize the MORI-shmem-backed reducer for the EP
            # all-reduce.  This sets up MORI's symmetric shared memory
            # infrastructure for P2P all-reduce via XGMI.
            from vllm.distributed.device_communicators.mori_shmem_reduce import (
                MoriShmemReducer,
            )

            self._shmem_reducer = MoriShmemReducer(
                ep_group=ep_group,
                world_size=ep_group.world_size,
                rank=ep_group.rank_in_group,
                max_num_tokens=max_tokens_per_rank,
                hidden_dim=hidden_dim,
            )
            logger.info(
                "MORI-EP dispatch-free optimization enabled. "
                "Dispatch All-to-All will be skipped. "
                "Using MoriShmemReducer (strategy=%s) for aggregation.",
                self._shmem_reducer.strategy,
            )

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def output_is_reduced(self) -> bool:
        return True

    def num_dispatchers(self):
        return self.num_dispatchers_

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        """
        Returns a tuple of:
        - quantized + dispatched a.
        - Optional quantized + dispatched a1_scales.
        - Optional ExpertTokensMetadata containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        if self.use_dispatch_free:
            # TP+EP mode: skip dispatch, filter locally
            return self._prepare_dispatch_free(
                a1, topk_weights, topk_ids, quant_config
            )
        
        # Standard mode: use MORI dispatch
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support "
                "defer_input_quant=True in standard (dispatch) mode."
            )
        assert not apply_router_weight_on_input, (
            "mori does not support apply_router_weight_on_input=True now."
        )
        assert self.mori_op is not None, (
            "mori_op is required for standard (non-dispatch-free) mode"
        )
        a1, scale = self._quantize_if_needed(a1, quant_config)

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self.mori_op.dispatch(a1, topk_weights, scale, topk_ids)

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_token_num, expert_num_tokens_cpu=None
        )

        return (
            dispatch_a1,
            dispatch_scale,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if self.use_dispatch_free:
            # TP+EP mode: use all-reduce instead of MORI combine
            self._finalize_dispatch_free(output, fused_expert_output)
        else:
            # Standard mode: use MORI combine
            assert self.mori_op is not None, (
                "mori_op is required for standard (non-dispatch-free) mode"
            )
            num_token = output.shape[0]
            result = self.mori_op.combine(
                fused_expert_output,
                None,
                topk_ids,
            )[0]
            output.copy_(result[:num_token])
    
    def _quantize_if_needed(
        self,
        a1: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply FP8 quantization if enabled."""
        scale = None
        if self.use_fp8_dispatch:
            from aiter import QuantType, get_hip_quant

            if quant_config.is_block_quantized:
                quant_func = get_hip_quant(QuantType.per_1x128)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())
            elif quant_config.is_per_act_token:
                quant_func = get_hip_quant(QuantType.per_Token)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())
        return a1, scale
    
    def _prepare_dispatch_free(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        """Dispatch-free prepare for TP+EP mode (CUDAGraph-compatible).

        This is a **pass-through**: activations, topk_ids and topk_weights
        are forwarded to the expert kernel unchanged.  The expert kernel
        (AITER) receives the *global* expert IDs together with the
        ``expert_map`` / ``expert_mask`` that the FusedMoE layer already
        provides.  AITER's ``moe_sorting_fwd`` uses that mask to skip
        GEMMs for non-local experts entirely, so each GPU only computes
        the ~1 local expert per token (out of topk=8), avoiding cache
        thrashing from touching all 256 experts' weights.

        No ID remapping, no zero-weighting, no expert counting is done
        here -- all of that is handled inside the AITER kernel via the
        expert mask.
        """
        # Pure pass-through.  expert_tokens_meta = None lets the expert
        # kernel derive token counts from topk_ids + expert_map itself.
        return (a1, None, None, topk_ids, topk_weights)
    
    def _finalize_dispatch_free(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
    ) -> None:
        """Dispatch-free finalize: all-reduce the expert output across EP ranks.

        Because prepare passed **all** tokens through (same shape as input)
        and non-local expert slots were zero-weighted, the expert output
        already has the correct (M, K) shape.  Each rank's contribution is
        non-zero only for its local experts, so a SUM all-reduce across the
        EP group produces the correct combined result.

        Uses the MoriShmemReducer which selects the optimal all-reduce
        strategy based on tensor size:
        - Small tensors: P2P 1-stage (custom_allreduce) for low latency
        - Larger tensors: RCCL ring (pynccl) for bandwidth efficiency
        """
        assert self._shmem_reducer is not None, (
            "MoriShmemReducer required for dispatch-free finalize"
        )

        # fused_expert_output is (M, K) -- same shape as output.
        # The MoriShmemReducer selects between custom_allreduce (P2P
        # 1-stage) and pynccl (RCCL ring) based on tensor size.
        # Both paths are CUDAGraph-safe.
        reduced = self._shmem_reducer.all_reduce(fused_expert_output)
        output.copy_(reduced)
