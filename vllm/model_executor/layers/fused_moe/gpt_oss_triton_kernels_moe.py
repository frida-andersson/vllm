# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_triton_kernels

logger = init_logger(__name__)

USE_CK_MOE = os.environ.get("VLLM_USE_CK_MOE", "0") == "1"
if USE_CK_MOE:
    try:
        from vllm.model_executor.layers.fused_moe.ck_moe_padded import (
            ck_mxfp4_w4a8_experts,
        )
        logger.info("CK MoE with padding loaded successfully")
    except ImportError as e:
        logger.warning("CK MoE import failed: %s. Falling back to Triton.", e)
        USE_CK_MOE = False

USE_ASM_MOE = os.environ.get("VLLM_USE_ASM_MOE", "0") == "1"
if USE_ASM_MOE:
    logger.info("ASM MoE (fused_moe_1stage) enabled")

use_legacy_triton_kernels = False

if has_triton_kernels():
    try:
        import triton_kernels.swiglu
        from triton_kernels.matmul_ogs import (
            FnSpecs,
            FusedActivation,
            GatherIndx,
            RoutingData,
            ScatterIndx,
            matmul_ogs,
        )
        from triton_kernels.tensor import (
            BIT,
            Bitmatrix,
        )
        from triton_kernels.topk import topk

        try:
            from triton_kernels.tensor import (
                SparseMatrix,
                make_ragged_tensor_metadata,
            )
        except ImportError:
            if current_platform.is_rocm():
                logger.warning_once("Using legacy triton_kernels on ROCm")
                use_legacy_triton_kernels = True
            else:
                raise
    except (AttributeError, ImportError) as e:
        logger.error(
            "Failed to import Triton kernels. Please make sure your triton "
            "version is compatible. Error: %s",
            e,
        )


@triton.jit
def pack_bitmatrix(
    bitmatrix,
    topk_ids,
    n_rows,  # n_rows in bitmatrix / topk_ids
    bm_cols: tl.constexpr,  # n int32_t bitpacks in bitmatrix
    n_expts_act,  # num_topk
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Packs topk_ids into a bitmatrix.
    code reference:
    https://github.com/triton-lang/triton/blob/dd1bbc52b34d202dfe5ffea1e04fb16166c5c04e/python/triton_kernels/bench/distributed.py#L264
    """
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
    mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
    indices = tl.load(topk_ids + offsets, mask=mask, other=-1)
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)

    # Iterate through all the relevant bitmatrix columns.
    for i in range(bm_cols):
        # When BLOCK_SIZE_K=32, offs is just the column index.
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        # All topks that need to go into this column has the correct bit set.
        # Other bits are 0. x is a 2D tensor.
        x = tl.where(
            div[:, :, None] == offs[None, None, :], (one << rem)[:, :, None], 0
        )
        # Reduce x to get a single int32_t bitpack.
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * bm_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)


def legacy_routing_from_bitmatrix(
    bitmatrix: "Bitmatrix",
    expt_scal: torch.Tensor,
    expt_indx: torch.Tensor,
    n_expts_tot: int,
    n_expts_act: int,
) -> tuple["RoutingData", "GatherIndx", "ScatterIndx"]:
    """
    Replacement for the removed triton_kernels.routing.routing_from_bitmatrix.
    Creates routing data from a bitmatrix representation.
    """
    if use_legacy_triton_kernels:
        from triton_kernels.routing import routing_from_bitmatrix

        return routing_from_bitmatrix(
            bitmatrix, expt_scal, expt_indx, n_expts_tot, n_expts_act
        )
    sparse_logits = SparseMatrix(indx=expt_indx, vals=expt_scal, mask=bitmatrix)
    dispatch_indx = sparse_logits.mask_metadata.row_sorted_indx
    combine_indx = sparse_logits.mask_metadata.col_sorted_indx
    ragged_batch_metadata = make_ragged_tensor_metadata(
        sparse_logits.mask_metadata.col_sum,
        dispatch_indx.shape[0],
    )
    gate_scal = sparse_logits.vals.flatten()[combine_indx]
    routing_data = RoutingData(
        gate_scal,
        ragged_batch_metadata.block_sizes,
        n_expts_tot,
        n_expts_act,
        ragged_batch_metadata,
    )
    gather_idx = GatherIndx(combine_indx, dispatch_indx)
    scatter_idx = ScatterIndx(dispatch_indx, combine_indx)
    return routing_data, gather_idx, scatter_idx


def legacy_routing(
    logits: torch.Tensor,
    n_expts_act: int,
    sm_first: bool = False,
) -> tuple["RoutingData", "GatherIndx", "ScatterIndx"]:
    """
    Replacement for the removed triton_kernels.routing.routing function.
    Computes routing data from gating logits.
    """
    if use_legacy_triton_kernels:
        from triton_kernels.routing import routing

        return routing(logits, n_expts_act, sm_first=sm_first)
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    sparse_logits = topk(logits, n_expts_act, apply_softmax=not sm_first)
    return legacy_routing_from_bitmatrix(
        sparse_logits.mask,
        sparse_logits.vals,
        sparse_logits.indx,
        logits.shape[-1],
        n_expts_act,
    )


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: MoEActivation = MoEActivation.SWIGLUOAI,
    quant_config: FusedMoEQuantConfig | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    unpadded_N_w1=None,
    unpadded_K_w1=None,
    unpadded_N_w2=None,
    unpadded_K_w2=None,
) -> torch.Tensor:
    if (
        quant_config is not None
        and quant_config.use_mxfp4_w4a8
        and rocm_aiter_ops.is_enabled()
    ):
        if USE_ASM_MOE:
            return _asm_moe_1stage_forward(
                hidden_states, w1, w2,
                gating_output, topk, renormalize,
                quant_config=quant_config,
            )

        if USE_CK_MOE:
            return ck_mxfp4_w4a8_experts(
                hidden_states, w1, w2,
                gating_output, topk, renormalize,
                quant_config=quant_config,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                unpadded_N_w1=unpadded_N_w1,
                unpadded_K_w1=unpadded_K_w1,
                unpadded_N_w2=unpadded_N_w2,
                unpadded_K_w2=unpadded_K_w2,
            )

        from aiter.ops.triton.moe_routing.routing import routing as aiter_routing

        routing_data, gather_idx, scatter_idx = aiter_routing(
            gating_output, topk, sm_first=not renormalize
        )
        return triton_kernel_fused_mxfp4_w4a8_experts(
            None,
            hidden_states,
            w1,
            w2,
            routing_data,
            gather_idx,
            scatter_idx,
            activation=activation.value,
            quant_config=quant_config,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            unpadded_N_w1=unpadded_N_w1,
            unpadded_K_w1=unpadded_K_w1,
            unpadded_N_w2=unpadded_N_w2,
            unpadded_K_w2=unpadded_K_w2,
        )

    if expert_map is not None:
        # With expert parallelism, legacy_routing produces routing data
        # using global expert IDs which don't correspond to local weight
        # indices.  Split the routing into topk selection + expert_map
        # remapping + local routing data construction (matching the
        # approach used by OAITritonExperts.apply).
        from triton_kernels.topk import topk as topk_fn

        sm_first = not renormalize
        logits = gating_output
        if sm_first:
            logits = torch.softmax(logits, dim=-1)
        sparse_logits = topk_fn(logits, topk, apply_softmax=not sm_first)
        # sparse_logits.indx contains global expert IDs – remap to local.
        topk_ids = expert_map[sparse_logits.indx.to(torch.long)]
        topk_weights = sparse_logits.vals
        local_num_experts = w1.size(0)
        routing_data, gather_idx, scatter_idx = make_routing_data(
            topk_ids, topk_weights, local_num_experts
        )
        # expert_map already applied; pass None downstream.
        effective_expert_map = None
        effective_global_num_experts = local_num_experts
    else:
        routing_data, gather_idx, scatter_idx = legacy_routing(
            gating_output, topk, sm_first=not renormalize
        )
        effective_expert_map = expert_map
        effective_global_num_experts = global_num_experts

    output = torch.empty_like(hidden_states)
    effective_quant_config = (
        quant_config if quant_config is not None else FUSED_MOE_UNQUANTIZED_CONFIG
    )

    return triton_kernel_fused_experts(
        output,
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        topk=topk,
        activation=activation,
        quant_config=effective_quant_config,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=effective_global_num_experts,
        expert_map=effective_expert_map,
    )


# This is a triton implementation of the fused_experts function
def triton_kernel_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    routing_data,  # RoutingData
    gather_indx,  # GatherIndx
    scatter_indx,  # ScatterIndx
    topk: int,
    activation: MoEActivation = MoEActivation.SWIGLUOAI,
    quant_config: FusedMoEQuantConfig | None = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    intermediate_cache: torch.Tensor | None = None,
    a1q_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton implementation of fused expert computation using OAI kernels."""
    assert activation == MoEActivation.SWIGLUOAI, (
        "Only SWIGLUOAI activation is supported"
    )
    assert quant_config is not None

    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert quant_config.w1_bias is None or quant_config.w1_bias.dtype == torch.float32
    assert quant_config.w2_bias is None or quant_config.w2_bias.dtype == torch.float32

    # Shape check, only check non-mxfp4
    assert hidden_states.ndim == 2
    assert hidden_states.shape[-1] == w1.shape[-2]
    assert w2.shape[-1] == w1.shape[1]

    batch_dim = 1
    M, K = hidden_states.shape[-2:]
    E, _, N = w1.shape

    if global_num_experts == -1:
        global_num_experts = E

    if intermediate_cache is None:
        intermediate_cache = torch.empty(
            (batch_dim, M * topk, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    # Add batch_dim to output buffer because matmul_ogs expects 3D output
    intermediate_cache = _resize_cache(
        intermediate_cache, (batch_dim, M * topk, N // 2)
    )
    output_tensor = _resize_cache(output_tensor, (batch_dim, M, K))

    act = (
        FusedActivation(
            FnSpecs(
                "swiglu",
                triton_kernels.swiglu.swiglu_fn,
                ("alpha", "limit"),
                reduction_n=2,
            ),
            (swiglu_alpha, swiglu_limit),
        )
        if not use_legacy_triton_kernels
        else FusedActivation(
            FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")),
            (swiglu_alpha, swiglu_limit),
            2,
        )
    )
    gammas = routing_data.gate_scal if routing_data else None

    matmul_ogs(
        hidden_states,
        w1,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_indx,
        precision_config=quant_config.w1_precision,
        gammas=gammas if apply_router_weight_on_input else None,
        fused_activation=act,
        y=intermediate_cache,
    )

    matmul_ogs(
        intermediate_cache.view(M * topk, N // 2),
        w2,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=quant_config.w2_precision,
        gammas=None if apply_router_weight_on_input else gammas,
        y=output_tensor,
    )
    output_tensor = output_tensor.view(M, K)
    return output_tensor


# This is a triton implementation of the fused_experts function
def triton_kernel_fused_mxfp4_w4a8_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    routing_data,  # RoutingData
    gather_indx,  # GatherIndx
    scatter_indx,  # ScatterIndx
    activation: str = "silu",
    quant_config: FusedMoEQuantConfig | None = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    a1q_scale: torch.Tensor | None = None,
    unpadded_N_w1=None,
    unpadded_K_w1=None,
    unpadded_N_w2=None,
    unpadded_K_w2=None,
) -> torch.Tensor:
    assert quant_config is not None
    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert quant_config.w1_bias is None or quant_config.w1_bias.dtype == torch.float32
    assert quant_config.w2_bias is None or quant_config.w2_bias.dtype == torch.float32

    # Shape check, only check non-mxfp4
    assert hidden_states.shape[-1] == w1.shape[-2]
    assert w2.shape[-1] == w1.shape[1]

    E, _, N = w1.shape

    if global_num_experts == -1:
        global_num_experts = E

    gammas = routing_data.gate_scal if routing_data else None

    from aiter.ops.triton.moe_op_gemm_a8w4 import moe_gemm_a8w4
    from aiter.ops.triton.quant_moe import downcast_to_static_fp8

    assert quant_config.w1_precision is not None, (
        "w1_precision in quant config can't be None"
    )
    assert quant_config.w2_precision is not None, (
        "w2_precision in quant config can't be None"
    )

    hidden_states = downcast_to_static_fp8(
        hidden_states, quant_config.w1_precision.flex_ctx.lhs_data.scale
    )

    intermediate_cache1 = moe_gemm_a8w4(
        hidden_states,
        w1.storage.data,
        None,
        quant_config.w1_precision.weight_scale.storage.data,
        quant_config.w1_precision.flex_ctx.lhs_data.scale,
        quant_config.w2_precision.flex_ctx.lhs_data.scale,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_indx,
        gammas=gammas if apply_router_weight_on_input else None,
        swizzle_mx_scale="CDNA4_SCALE",
        out_dtype=torch.float8_e4m3fn,
        apply_swiglu=True,
        alpha=swiglu_alpha,
        limit=swiglu_limit,
        unpadded_N=unpadded_N_w1,
        unpadded_K=unpadded_K_w1,
    )

    intermediate_cache3 = moe_gemm_a8w4(
        intermediate_cache1,
        w2.storage.data,
        None,
        quant_config.w2_precision.weight_scale.storage.data,
        quant_config.w2_precision.flex_ctx.lhs_data.scale,
        None,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale="CDNA4_SCALE",
        unpadded_N=unpadded_N_w2,
        unpadded_K=unpadded_K_w2,
    )

    return intermediate_cache3


def make_routing_data(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
) -> tuple["RoutingData", torch.Tensor, torch.Tensor]:
    topk_ids = topk_ids.to(torch.int16)
    topk_weights = topk_weights.to(torch.bfloat16)

    n_rows, num_topk = topk_ids.size()

    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32

    bm_cols = triton.cdiv(num_local_experts, BLOCK_SIZE_K)  # n_bitpacks
    bitmatrix = torch.zeros(
        (n_rows, bm_cols), dtype=torch.uint32, device=topk_ids.device
    )

    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    pack_bitmatrix[grid](
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols,
        num_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    bitmatrix_shape = [n_rows, bm_cols * 32]
    bitmatrix_shape_max = [n_rows, None]
    bitmatrix = (
        Bitmatrix(
            bitmatrix, dtype=BIT, shape=bitmatrix_shape, shape_max=bitmatrix_shape_max
        )
        if not use_legacy_triton_kernels
        else Bitmatrix(
            bitmatrix,
            shape=bitmatrix_shape,
            shape_max=bitmatrix_shape_max,
            scratchpad=None,
        )
    )

    # matmul_ogs expects invalid topk_weights to be -1s
    topk_weights = torch.where(topk_ids == -1, -1.0, topk_weights)
    routing_data, gather_indx, scatter_indx = legacy_routing_from_bitmatrix(
        bitmatrix, topk_weights, topk_ids, num_local_experts, num_topk
    )

    return routing_data, gather_indx, scatter_indx


class BaseOAITritonExperts(mk.FusedMoEExpertsModular):
    @staticmethod
    def _supports_current_device() -> bool:
        raise NotImplementedError(
            "OAITritonExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        raise NotImplementedError(
            "OAITritonExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        raise NotImplementedError(
            "OAITritonExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        raise NotImplementedError(
            "OAITritonExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        raise NotImplementedError(
            "OAITritonExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    def supports_expert_map(self) -> bool:
        return True

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        """
        Extract the MoE problem size from the given tensor arguments:
        - a: The hidden states, input to the MoE layer.
        - w1: The first set of expert weights.
        - w2: The second set of expert weights.
        - topk_ids: The topk ids.
        Note: extracting the problem shape from the weight and activation
        tensors is not obvious.  It needs to be done this way specifically
        due to subtle issues with particular kernels, e.g. the int4 kernels
        divide the trailing dimension by two, so it's not "correct" to
        extract N or K from the trailing dimension of w1 or w2.  Similarly,
        some kernels transpose the weights, so this needs to be kept in mind.
        Note: This implementation covers most cases. However, if experts
        require a specialized implementation, like MarlinExperts, they are free
        to override this function.
        """
        assert w1.dim() == 3 and w2.dim() == 3
        E, _, N = w1.size()
        K = a1.size(-1)

        assert a1.dim() == 2
        assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
        M = a1.size(0)

        assert topk_ids.dim() == 2
        topk = topk_ids.size(1)

        return E, M, N, K, topk

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Weight application and reduction happens in the fused_experts kernel.
        return TopKWeightAndReduceNoOP()

    def _make_routing_data(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_local_experts: int,
    ) -> tuple["RoutingData", torch.Tensor, torch.Tensor]:
        return make_routing_data(topk_ids, topk_weights, num_local_experts)


class OAITritonExperts(BaseOAITritonExperts):
    """OAI Triton-based fused MoE expert implementation."""

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_chunking(self) -> bool:
        return True

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # workspace are allocated inside the kernel
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (0, 0)
        workspace2 = (M * topk, activation_out_dim)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if self.quant_config is None:
            self.quant_config: FusedMoEQuantConfig = FUSED_MOE_UNQUANTIZED_CONFIG

        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        routing_data, gather_indx, scatter_indx = self._make_routing_data(
            topk_ids, topk_weights, local_num_experts
        )

        topk = topk_ids.size(1)
        triton_kernel_fused_experts(
            output,
            hidden_states,
            w1,
            w2,
            routing_data,
            gather_indx,
            scatter_indx,
            topk=topk,
            activation=activation,
            quant_config=self.quant_config,
            apply_router_weight_on_input=False,
            global_num_experts=local_num_experts,
            expert_map=None,  # applied already
            intermediate_cache=workspace2,
            a1q_scale=a1q_scale,
        )


class UnfusedOAITritonExperts(BaseOAITritonExperts):
    """
    A Triton based MoE expert class that operates on expert standard
    format and explicitly keeps the activation and reduction (moe_sum) steps
    unfused from the matmul_ogs kernel. This exposes injection points
    for activation and moe_sum.

    One use case for it is to inject LoRA modules on the activation and moe_sum.
    """

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_chunking(self) -> bool:
        return True

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # workspace are allocated inside the kernel
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M * topk, activation_out_dim)
        workspace2 = (M * topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor):
        ops.moe_sum(input, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # Use local variable to help mypy narrow the type after None check
        quant_config = self.quant_config
        if quant_config is None:
            quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        routing_data, gather_indx, scatter_indx = self._make_routing_data(
            topk_ids, topk_weights, local_num_experts
        )

        topk = topk_ids.size(1)

        # type check, uint8 means mxfp4
        assert hidden_states.dtype == torch.bfloat16
        assert (
            quant_config.w1_bias is None or quant_config.w1_bias.dtype == torch.float32
        )
        assert (
            quant_config.w2_bias is None or quant_config.w2_bias.dtype == torch.float32
        )

        # Shape check, only check non-mxfp4
        assert hidden_states.ndim == 2
        assert hidden_states.shape[-1] == w1.shape[-2]
        assert w2.shape[-1] == w1.shape[1]

        batch_dim = 1
        M, K = hidden_states.shape
        E, _, N = w1.shape

        if global_num_experts == -1:
            global_num_experts = E

        # Note that the output tensor might be in workspace13
        intermediate_cache1 = _resize_cache(workspace2, (batch_dim, M * topk, N))
        intermediate_cache3 = _resize_cache(workspace2, (batch_dim, M * topk, K))
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(workspace13, (M * topk, activation_out_dim))

        gammas = routing_data.gate_scal if routing_data else None

        matmul_ogs(
            hidden_states,
            w1,
            quant_config.w1_bias,
            routing_data,
            gather_indx=gather_indx,
            precision_config=quant_config.w1_precision,
            gammas=gammas if apply_router_weight_on_input else None,
            fused_activation=None,
            y=intermediate_cache1,
        )

        self.activation(
            activation,
            intermediate_cache2,
            intermediate_cache1.view(-1, N)[gather_indx.dst_indx],
        )

        # matmul_ogs grouped reduction fuse sum across multiple experts:
        # y[dst_indx // n_expts_act, :] += x
        # Need to set n_expts_act to 1 to unfuse moe_sum
        routing_data.n_expts_act = 1

        matmul_ogs(
            intermediate_cache2[gather_indx.src_indx],
            w2,
            quant_config.w2_bias,
            routing_data,
            scatter_indx=scatter_indx,
            precision_config=quant_config.w2_precision,
            gammas=None if apply_router_weight_on_input else gammas,
            y=intermediate_cache3,
        )

        self.moe_sum(intermediate_cache3.view(-1, topk, K), output)


_flydsl_cache: dict = {}


def _unswizzle_cdna4_scale(data):
    """Reverse CDNA4 scale swizzle: [E, SCALE_K*32, N//32] -> [E, N, SCALE_K]."""
    key = ("unswizzle", data.data_ptr())
    if key in _flydsl_cache:
        return _flydsl_cache[key]
    E, SK32, Nd32 = data.shape
    N = Nd32 * 32
    SK = SK32 // 32
    d = data.transpose(-1, -2)
    d = d.reshape(E, N // 32, SK // 8, 4, 16, 2, 2, 1)
    d = d.permute(0, 1, 6, 4, 2, 5, 3, 7).contiguous()
    result = d.reshape(E, N, SK)
    _flydsl_cache[key] = result
    return result


def _prepare_flydsl_weight(w, scale_cdna4, gate_up):
    """Prepare weight+scale for FlyDSL: transpose, unswizzle, shuffle. Cached."""
    key = ("flydsl_w", w.data_ptr() if hasattr(w, 'data_ptr') else id(w))
    if key in _flydsl_cache:
        return _flydsl_cache[key]

    from tests.kernels.utils.fp4_utils import shuffle_weight_w4, shuffle_scale_w4

    data = w.storage.data if hasattr(w, 'storage') else w
    E = data.shape[0]
    N = data.shape[2]

    w_t = data.transpose(1, 2).contiguous()
    w_shuf = shuffle_weight_w4(w_t, 16, gate_up, True)

    scale_orig = _unswizzle_cdna4_scale(scale_cdna4)
    scale_flat = scale_orig.reshape(E * N, -1)
    scale_shuf = shuffle_scale_w4(scale_flat, E, gate_up)

    result = (w_shuf, scale_shuf)
    _flydsl_cache[key] = result
    return result


_flydsl_exe_cache: dict = {}


def _asm_moe_1stage_forward(
    hidden_states: torch.Tensor,
    w1,
    w2,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    quant_config=None,
):
    """Dispatch MoE through FlyDSL compile_moe_gemm1/gemm2 kernels.

    Stage1 (gate+up+SiGLU): MXFP4 activations x MXFP4 weights
    Stage2 (down projection): MXFP4 activations x MXFP4 weights
    Uses Quark's native MXFP4 weights directly (no dequant/requant).
    Weights are shuffled once and cached.
    """
    from kernels.moe_gemm_2stage import compile_moe_gemm1, compile_moe_gemm2
    from tests.kernels.utils.fp4_utils import fp8_e8m0
    from aiter.fused_moe import moe_sorting, fp4_utils

    M = hidden_states.shape[0]
    model_dim = hidden_states.shape[-1]
    device = hidden_states.device

    w1_data = w1.storage.data if hasattr(w1, 'storage') else w1
    E = w1_data.shape[0]
    N_w1 = w1_data.shape[2]
    N_w2 = (w2.storage.data if hasattr(w2, 'storage') else w2).shape[2]
    inter_dim = N_w1 // 2

    w1_scale_cdna4 = quant_config.w1_precision.weight_scale.storage.data
    w2_scale_cdna4 = quant_config.w2_precision.weight_scale.storage.data

    w1_shuf, w1_s_shuf = _prepare_flydsl_weight(w1, w1_scale_cdna4, gate_up=True)
    w2_shuf, w2_s_shuf = _prepare_flydsl_weight(w2, w2_scale_cdna4, gate_up=False)

    # Routing
    sm_first = not renormalize
    logits = gating_output.float()
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    topk_vals, topk_ids = torch.topk(logits, k=topk, dim=-1)
    if not sm_first:
        topk_weights = torch.softmax(topk_vals, dim=-1)
    else:
        topk_weights = topk_vals
    topk_ids_i32 = topk_ids.to(torch.int32)

    tile_m = 32
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = \
        moe_sorting(topk_ids_i32, topk_weights.to(torch.float32), E, tile_m, None)
    blocks = sorted_expert_ids.shape[0]

    # Compile kernels (cached by FlyDSL internally)
    exe_key = ("exe", model_dim, inter_dim, E, topk)
    if exe_key not in _flydsl_exe_cache:
        exe1 = compile_moe_gemm1(
            model_dim=model_dim, inter_dim=inter_dim, experts=E, topk=topk,
            x_dtype="fp8", w_dtype="fp4", out_dtype="f16",
            tile_m=tile_m, tile_n=256, tile_k=256,
            doweight_stage1=False, use_cshuffle_epilog=False, enable_bias=False,
        )
        exe2 = compile_moe_gemm2(
            model_dim=model_dim, inter_dim=inter_dim, experts=E, topk=topk,
            x_dtype="fp8", w_dtype="fp4", out_dtype="f16",
            tile_m=tile_m, tile_n=256, tile_k=256, doweight_stage2=True,
        )
        _flydsl_exe_cache[exe_key] = (exe1, exe2)
    exe1, exe2 = _flydsl_exe_cache[exe_key]

    stream_ptr = torch.cuda.current_stream().cuda_stream
    bias_1d = torch.empty((0,), device=device, dtype=torch.float32)

    # Stage 1: fp8 activations, fp4 weights, ones activation scale
    x_fp8 = hidden_states.to(DTYPE_FP8)
    scale_x = torch.ones([M, model_dim // 32], dtype=fp8_e8m0, device=device)
    out_s1 = torch.empty((M, topk, inter_dim), device=device, dtype=torch.float16)

    exe1(out_s1, x_fp8, w1_shuf, scale_x.view(-1).contiguous(), w1_s_shuf,
         sorted_ids, sorted_expert_ids, sorted_weights.view(-1).contiguous(),
         num_valid_ids, bias_1d, M, inter_dim, model_dim, int(blocks), stream_ptr)

    # Stage 2: fp8 intermediate (direct cast), ones scale
    a2_fp8 = out_s1.to(DTYPE_FP8)
    a2_scale = torch.ones([M, topk, inter_dim // 32], dtype=fp8_e8m0, device=device)
    out_s2 = torch.zeros((M, model_dim), device=device, dtype=torch.float16)

    exe2(out_s2, a2_fp8.contiguous().view(M * topk, inter_dim), w2_shuf,
         a2_scale.view(-1).contiguous(), w2_s_shuf,
         sorted_ids, sorted_expert_ids, sorted_weights.view(-1).contiguous(),
         num_valid_ids, bias_1d, M, model_dim, inter_dim, int(blocks), stream_ptr)

    return out_s2.to(hidden_states.dtype)
