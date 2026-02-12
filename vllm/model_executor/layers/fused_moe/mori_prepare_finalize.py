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
       Each GPU filters locally, computes on its experts, then all-reduces.
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
        
        
        if self.use_dispatch_free:
            logger.info(
                "MORI-EP dispatch-free optimization enabled. "
                "Dispatch All-to-All will be skipped. "
                "Using CUDAGraph-compatible all-reduce for aggregation."
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

        All tensors keep their original shape -- no ``nonzero()`` or
        dynamic indexing.  Non-local expert slots are zero-weighted and
        their IDs clamped into [0, num_local_experts) so the AITER kernel
        runs on a fixed-size buffer.  The ``expert_num_tokens`` metadata
        tells AITER how many tokens each local expert actually has.
        """
        assert self.rank_expert_offset is not None
        assert self.num_local_experts is not None

        # 1. Identify which (token, slot) pairs target a local expert
        local_expert_ids = topk_ids - self.rank_expert_offset
        is_local = (local_expert_ids >= 0) & (
            local_expert_ids < self.num_local_experts
        )

        # 2. Zero-weight non-local slots, clamp IDs to valid range
        topk_weights = topk_weights * is_local.float()
        local_expert_ids = local_expert_ids.clamp(0, self.num_local_experts - 1)

        # 3. Count tokens per local expert (fully static shapes,
        #    CUDAGraph-compatible -- no boolean indexing or nonzero).
        #    Use the zero-weighted IDs directly; non-local slots have
        #    weight == 0 so they won't contribute to expert computation,
        #    but we still count only truly-local slots for metadata.
        local_ids_flat = local_expert_ids.reshape(-1).to(torch.int64)
        is_local_flat = is_local.reshape(-1).to(torch.int64)
        expert_counts = torch.zeros(
            self.num_local_experts, dtype=torch.int64, device=topk_ids.device
        )
        expert_counts.scatter_add_(0, local_ids_flat, is_local_flat)
        expert_counts = expert_counts.to(torch.int32)

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_counts, expert_num_tokens_cpu=None
        )

        # a1 is passed through unchanged (same shape as input).
        return (a1, None, expert_tokens_meta, local_expert_ids, topk_weights)
    
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
        """
        assert self.ep_group is not None, (
            "Expert parallel group required for dispatch-free finalize"
        )

        # fused_expert_output is (M, K) -- same shape as output.
        # Use GroupCoordinator.all_reduce which is CUDAGraph-safe
        # (it dispatches through custom_all_reduce / pynccl).
        reduced = self.ep_group.all_reduce(fused_expert_output)
        output.copy_(reduced)
