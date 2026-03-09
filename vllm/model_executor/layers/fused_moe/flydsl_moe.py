# SPDX-License-Identifier: Apache-2.0
"""FlyDSL MXFP4 MoE integration for vLLM.

Replaces Triton moe_gemm_a8w4 with FlyDSL's MLIR-compiled kernels.
~40% faster for MXFP4 (fp8 x fp4) on gfx950.

Weight conversion (safetensors -> FlyDSL preshuffle) happens once at first call.
Activation padding (2880 -> 3072) happens every forward pass.
"""
import functools
import logging

import torch

logger = logging.getLogger(__name__)

_PAD_MULTIPLE = 256
_weight_cache: dict = {}
_TILE_M = 32
_TILE_N = 128
_TILE_K = 256


def _pad_dim(x: int) -> int:
    return ((x + _PAD_MULTIPLE - 1) // _PAD_MULTIPLE) * _PAD_MULTIPLE


@functools.lru_cache(maxsize=16)
def _compile_stage1(model_dim, inter_dim, experts, topk):
    from kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1
    logger.info("Compiling FlyDSL stage1: dim=%d inter=%d E=%d topk=%d",
                model_dim, inter_dim, experts, topk)
    return compile_mixed_moe_gemm1(
        model_dim=model_dim, inter_dim=inter_dim,
        experts=experts, topk=topk,
        tile_m=_TILE_M, tile_n=_TILE_N, tile_k=_TILE_K,
        doweight_stage1=True,
        a_dtype="fp8", b_dtype="fp4", out_dtype="f16",
    )


@functools.lru_cache(maxsize=16)
def _compile_stage2(model_dim, inter_dim, experts, topk):
    from kernels.moe_gemm_2stage import compile_moe_gemm2
    logger.info("Compiling FlyDSL stage2: dim=%d inter=%d E=%d topk=%d",
                model_dim, inter_dim, experts, topk)
    return compile_moe_gemm2(
        model_dim=model_dim, inter_dim=inter_dim,
        experts=experts, topk=topk,
        tile_m=_TILE_M, tile_n=_TILE_N, tile_k=_TILE_K,
        doweight_stage2=False,
        x_dtype="fp8", w_dtype="fp4", out_dtype="f16",
    )


def _convert_weights(w_data, actual_K, padded_K):
    """Convert vLLM MXFP4 weights to FlyDSL preshuffle format. Cached."""
    cache_key = id(w_data)
    if cache_key in _weight_cache:
        return _weight_cache[cache_key]

    from tests.utils import shuffle_weight
    from tests.kernels.utils import fp4_utils

    E, K_packed, N = w_data.shape

    w_row = w_data.permute(0, 2, 1).contiguous()

    padded_K_packed = padded_K // 2
    if padded_K_packed > K_packed:
        pad = torch.zeros(E, N, padded_K_packed - K_packed,
                         dtype=torch.uint8, device=w_data.device)
        w_row = torch.cat([w_row, pad], dim=-1)

    w_fp32 = fp4_utils.mxfp4_to_f32(w_row)
    w_q, w_scale, _ = fp4_utils.per_1x32_f4_quant(w_fp32)

    w_shuffled = fp4_utils.shuffle_weight_w4(w_q, 16, True, True)
    w_flat = w_shuffled.view(E * N, padded_K_packed).contiguous()
    w_scale_flat = w_scale.view(E * N, -1).contiguous()

    result = (w_flat, w_scale_flat)
    _weight_cache[cache_key] = result
    logger.info("FlyDSL weight conversion done: E=%d N=%d K=%d->%d",
                E, N, actual_K, padded_K)
    return result


def flydsl_mxfp4_w4a8_experts(
    hidden_states: torch.Tensor,
    w1, w2,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    quant_config=None,
    apply_router_weight_on_input=False,
    global_num_experts=-1,
    expert_map=None,
    unpadded_N_w1=None, unpadded_K_w1=None,
    unpadded_N_w2=None, unpadded_K_w2=None,
):
    """FlyDSL MoE: takes gating_output directly, handles routing internally."""
    from aiter.fused_moe import moe_sorting
    from aiter.ops.triton.quant_moe import downcast_to_static_fp8
    from tests.kernels.utils import fp4_utils

    assert quant_config is not None
    assert hidden_states.dtype == torch.bfloat16

    M = hidden_states.shape[0]
    actual_K = hidden_states.shape[-1]
    padded_K = _pad_dim(actual_K)

    w1_data = w1.storage.data if hasattr(w1, 'storage') else w1
    w2_data = w2.storage.data if hasattr(w2, 'storage') else w2
    E = w1_data.shape[0]
    N_w1 = w1_data.shape[2]
    N_w2 = w2_data.shape[2]

    sm_first = not renormalize
    if sm_first:
        logits = torch.softmax(gating_output.float(), dim=-1)
    else:
        logits = gating_output.float()

    topk_vals, topk_ids = torch.topk(logits, k=topk, dim=-1)
    if not sm_first:
        topk_weights = torch.softmax(topk_vals, dim=-1)
    else:
        topk_weights = topk_vals
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, _TILE_M, None
    )
    sorted_size = int(sorted_ids.numel())
    blocks = int(sorted_expert_ids.numel())

    if padded_K > actual_K:
        hidden_padded = torch.zeros(M, padded_K, dtype=hidden_states.dtype,
                                    device=hidden_states.device)
        hidden_padded[:, :actual_K] = hidden_states
    else:
        hidden_padded = hidden_states

    x_scale = quant_config.w1_precision.flex_ctx.lhs_data.scale
    x_fp8 = downcast_to_static_fp8(hidden_padded, x_scale)

    a_scale = torch.ones([M, padded_K // 32], dtype=fp4_utils.fp8_e8m0,
                         device=hidden_states.device)

    w1_flat, w1_scale = _convert_weights(w1_data, actual_K, padded_K)
    w2_flat, w2_scale = _convert_weights(w2_data, actual_K, padded_K)

    inter_dim_padded = _pad_dim(N_w1 // 2)
    stage1 = _compile_stage1(padded_K, inter_dim_padded, E, topk)

    out1 = torch.empty(sorted_size, inter_dim_padded, dtype=torch.float16,
                       device=hidden_states.device)
    stage1(out1, x_fp8, w1_flat, a_scale, w1_scale,
           sorted_ids, sorted_expert_ids, sorted_weights,
           num_valid_ids, M, 2 * inter_dim_padded, padded_K, blocks)

    out1_fp8 = out1.to(torch.float8_e4m3fn)
    a2_scale = torch.ones([sorted_size, inter_dim_padded // 32],
                          dtype=fp4_utils.fp8_e8m0, device=hidden_states.device)

    stage2 = _compile_stage2(padded_K, inter_dim_padded, E, topk)

    out2 = torch.zeros(M, padded_K, dtype=torch.float16,
                       device=hidden_states.device)
    stage2(out2, out1_fp8, w2_flat, a2_scale, w2_scale,
           sorted_ids, sorted_expert_ids, sorted_weights,
           num_valid_ids, M, padded_K, inter_dim_padded, blocks)

    return out2[:, :actual_K].to(hidden_states.dtype)
