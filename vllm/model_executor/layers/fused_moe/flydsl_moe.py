# SPDX-License-Identifier: Apache-2.0
"""FlyDSL MXFP4 MoE integration for vLLM.

Uses compile_moe_gemm1/gemm2 with w_dtype="fp4" (NOT compile_mixed_moe_gemm1).
Weight tensors stay 3D [E, N, K_packed], scales use shuffle_scale_w4.
"""
import functools
import logging

import torch

logger = logging.getLogger(__name__)

_PAD_MULTIPLE = 256
_weight_cache: dict = {}
_TILE_M = 32
_TILE_N = 256
_TILE_K = 256


def _pad_dim(x: int) -> int:
    return ((x + _PAD_MULTIPLE - 1) // _PAD_MULTIPLE) * _PAD_MULTIPLE


@functools.lru_cache(maxsize=16)
def _compile_stage1(model_dim, inter_dim, experts, topk, tile_n):
    from kernels.moe_gemm_2stage import compile_moe_gemm1
    logger.info("Compiling FlyDSL stage1 (fp4): dim=%d inter=%d E=%d topk=%d tile_n=%d",
                model_dim, inter_dim, experts, topk, tile_n)
    return compile_moe_gemm1(
        model_dim=model_dim, inter_dim=inter_dim,
        experts=experts, topk=topk,
        tile_m=_TILE_M, tile_n=tile_n, tile_k=_TILE_K,
        doweight_stage1=True,
        x_dtype="fp8", w_dtype="fp4", out_dtype="f16",
        use_cshuffle_epilog=False,
    )


@functools.lru_cache(maxsize=16)
def _compile_stage2(model_dim, inter_dim, experts, topk, tile_n):
    from kernels.moe_gemm_2stage import compile_moe_gemm2
    logger.info("Compiling FlyDSL stage2 (fp4): dim=%d inter=%d E=%d topk=%d tile_n=%d",
                model_dim, inter_dim, experts, topk, tile_n)
    return compile_moe_gemm2(
        model_dim=model_dim, inter_dim=inter_dim,
        experts=experts, topk=topk,
        tile_m=_TILE_M, tile_n=tile_n, tile_k=_TILE_K,
        doweight_stage2=False,
        x_dtype="fp8", w_dtype="fp4", out_dtype="f16",
        use_cshuffle_epilog=True,
    )


def _convert_weights(w_data, actual_K, padded_K, actual_N, padded_N, E):
    """Convert vLLM MXFP4 weights to FlyDSL format. Cached.
    
    FP4 path: weights stay 3D [E, N, K_packed], scales use shuffle_scale_w4.
    """
    cache_key = id(w_data)
    if cache_key in _weight_cache:
        return _weight_cache[cache_key]

    from tests.kernels.utils import fp4_utils

    _, K_packed, N = w_data.shape
    logger.info("_convert_weights: E=%d K_packed=%d N=%d shape=%s stride=%s dtype=%s",
                E, K_packed, N, list(w_data.shape), w_data.stride(), w_data.dtype)

    # Convert col-major [E, K_packed, N] to row-major [E, N, K_packed]
    w_row = w_data.permute(0, 2, 1).contiguous()

    # Pad K
    padded_K_packed = padded_K // 2
    if padded_K_packed > K_packed:
        w_row = torch.cat([w_row, torch.zeros(E, N, padded_K_packed - K_packed,
                          dtype=torch.uint8, device=w_data.device)], dim=-1)

    # Pad N
    if padded_N > N:
        w_row = torch.cat([w_row, torch.zeros(E, padded_N - N, padded_K_packed,
                          dtype=torch.uint8, device=w_data.device)], dim=1)

    # Dequantize -> requantize to get FlyDSL-compatible fp4x2 + scales
    w_fp32 = fp4_utils.mxfp4_to_f32(w_row)
    del w_row
    w_q, w_scale, _ = fp4_utils.per_1x32_f4_quant(w_fp32)
    del w_fp32
    torch.cuda.empty_cache()

    # Preshuffle weights (stays 3D, NOT flattened)
    w_shuffled = fp4_utils.shuffle_weight_w4(w_q, 16, True, True)

    # Shuffle scales (FP4 path uses shuffle_scale_w4, not just view)
    w_scale_shuffled = fp4_utils.shuffle_scale_w4(w_scale, E, True)

    result = (w_shuffled, w_scale_shuffled)
    _weight_cache[cache_key] = result
    logger.info("FlyDSL weight conversion done: E=%d N=%d->%d K=%d->%d",
                E, actual_N, padded_N, actual_K, padded_K)
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
    N_w1 = w1_data.shape[2]  # 5760
    N_w2 = w2_data.shape[2]  # 2880

    inter_dim_padded = _pad_dim(N_w1 // 2)
    padded_N_w1 = 2 * inter_dim_padded
    padded_N_w2 = padded_K

    # --- Routing ---
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

    # --- Pad activations ---
    if padded_K > actual_K:
        hidden_padded = torch.zeros(M, padded_K, dtype=hidden_states.dtype,
                                    device=hidden_states.device)
        hidden_padded[:, :actual_K] = hidden_states
    else:
        hidden_padded = hidden_states

    # --- Quantize to FP8 ---
    x_scale = quant_config.w1_precision.flex_ctx.lhs_data.scale
    x_fp8 = downcast_to_static_fp8(hidden_padded, x_scale)

    # Activation scale for FP4 path
    a_scale = torch.ones([M, padded_K // 32], dtype=fp4_utils.fp8_e8m0, device=hidden_states.device)

    # --- Convert weights (cached) ---
    w1_kernel, w1_scale = _convert_weights(w1_data, actual_K, padded_K, N_w1, padded_N_w1, E)
    w2_kernel, w2_scale = _convert_weights(w2_data, actual_K, inter_dim_padded, N_w2, padded_N_w2, E)

    # --- Stage 1 ---
    logger.info("Stage1 args: M=%d E=%d topk=%d padded_K=%d inter=%d actual_K=%d N_w1=%d",
                M, E, topk, padded_K, inter_dim_padded, actual_K, N_w1)
    logger.info("  x_fp8: %s %s contiguous=%s", list(x_fp8.shape), x_fp8.dtype, x_fp8.is_contiguous())
    logger.info("  w1_kernel: %s %s contiguous=%s", list(w1_kernel.shape), w1_kernel.dtype, w1_kernel.is_contiguous())
    logger.info("  w1_scale: %s %s contiguous=%s", list(w1_scale.shape), w1_scale.dtype, w1_scale.is_contiguous())
    logger.info("  a_scale: %s %s", list(a_scale.shape), a_scale.dtype)
    logger.info("  sorted: ids=%s eids=%s blocks=%d", list(sorted_ids.shape), list(sorted_expert_ids.shape), blocks)
    stage1 = _compile_stage1(padded_K, inter_dim_padded, E, topk, _TILE_N)
    out1 = torch.empty(M, topk, inter_dim_padded, dtype=torch.float16, device=hidden_states.device)
    bias1 = torch.empty(0, device=hidden_states.device, dtype=torch.float32)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    stage1(out1,
           x_fp8.contiguous().view(M, padded_K),
           w1_kernel, a_scale.view(-1), w1_scale,
           sorted_ids, sorted_expert_ids, sorted_weights,
           num_valid_ids, bias1,
           M, 2 * inter_dim_padded, padded_K, blocks,
           stream_ptr)

    # --- Quantize intermediate ---
    out1_fp8 = out1.view(-1, inter_dim_padded).to(torch.float8_e4m3fn)
    a2_scale = torch.ones([M * topk, inter_dim_padded // 32],
                          dtype=fp4_utils.fp8_e8m0, device=hidden_states.device)

    # --- Stage 2 ---
    stage2 = _compile_stage2(padded_K, inter_dim_padded, E, topk, _TILE_N)
    out2 = torch.zeros(M, padded_K, dtype=torch.float16, device=hidden_states.device)
    bias2 = torch.empty(0, device=hidden_states.device, dtype=torch.float32)

    stage2(out2,
           out1_fp8.contiguous().view(-1),
           w2_kernel, a2_scale.view(-1), w2_scale,
           sorted_ids, sorted_expert_ids, sorted_weights,
           num_valid_ids, bias2,
           M, padded_K, inter_dim_padded, blocks,
           stream_ptr)

    # --- Trim and convert ---
    return out2[:, :actual_K].to(hidden_states.dtype)
