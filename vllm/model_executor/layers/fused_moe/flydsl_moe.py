# SPDX-License-Identifier: Apache-2.0
"""FlyDSL MXFP4 MoE integration for vLLM.

Properly dequantizes MXFP4 (applying MX block scales), then requantizes
with FlyDSL's expected format. This preserves numeric accuracy.
"""
import functools
import glob
import logging

import torch

logger = logging.getLogger(__name__)

_PAD_MULTIPLE = 256
_weight_cache: dict = {}
_original_scales: dict = {}
_TILE_M = 32
_TILE_N = 256
_TILE_K = 256


def _pad_dim(x: int) -> int:
    return ((x + _PAD_MULTIPLE - 1) // _PAD_MULTIPLE) * _PAD_MULTIPLE


@functools.lru_cache(maxsize=16)
def _compile_stage1(model_dim, inter_dim, experts, topk, tile_n):
    from kernels.moe_gemm_2stage import compile_moe_gemm1
    logger.info("Compiling FlyDSL stage1: dim=%d inter=%d E=%d topk=%d",
                model_dim, inter_dim, experts, topk)
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
    logger.info("Compiling FlyDSL stage2: dim=%d inter=%d E=%d topk=%d",
                model_dim, inter_dim, experts, topk)
    return compile_moe_gemm2(
        model_dim=model_dim, inter_dim=inter_dim,
        experts=experts, topk=topk,
        tile_m=_TILE_M, tile_n=tile_n, tile_k=_TILE_K,
        doweight_stage2=False,
        x_dtype="fp8", w_dtype="fp4", out_dtype="f16",
        use_cshuffle_epilog=True,
    )


def _mxfp4_dequant_with_scales(w_fp4_uint8, scales_uint8, block_size=32):
    """Properly dequantize MXFP4: apply per-block E8M0 scales.

    w_fp4_uint8: [*, K_packed] uint8 (2 fp4 values per byte)
    scales_uint8: [*, K//32] uint8 (E8M0 format)
    Returns: [*, K] float32
    """
    from tests.kernels.utils import fp4_utils

    # Unpack fp4 nibbles to float32 values in [-6, 6]
    raw_fp32 = fp4_utils.mxfp4_to_f32(w_fp4_uint8)

    # Convert E8M0 scales to float32: scale = 2^(e8m0 - 127)
    E8M0_BIAS = 127
    scale_int = scales_uint8.to(torch.int32)
    scale_f32 = torch.pow(2.0, (scale_int.float() - E8M0_BIAS))

    # Apply scales: each scale covers block_size consecutive elements
    # raw_fp32: [*, K], scale_f32: [*, K//32]
    # Expand scale to match: repeat each scale for block_size elements
    scale_expanded = scale_f32.repeat_interleave(block_size, dim=-1)

    # Trim to match K (in case K is not multiple of block_size)
    K = raw_fp32.shape[-1]
    scale_expanded = scale_expanded[..., :K]

    return raw_fp32 * scale_expanded


def _load_original_scales(layer_idx, proj_name, E, device):
    """Load original per-block MX scales from safetensors."""
    key = (layer_idx, proj_name)
    if key in _original_scales:
        return _original_scales[key]

    from safetensors import safe_open
    paths = sorted(glob.glob(
        "/root/.cache/huggingface/hub/models--amd--gpt-oss-120b*/snapshots/*/*.safetensors"))

    scales = []
    for path in paths:
        with safe_open(path, framework="pt") as f:
            for skey in f.keys():
                if f"layers.{layer_idx}.mlp.experts." in skey and proj_name in skey and "weight_scale" in skey:
                    scales.append((skey, f.get_tensor(skey).to(device)))
        if len(scales) >= E:
            break

    if len(scales) < E:
        logger.warning("Only found %d/%d scales for layer %d %s", len(scales), E, layer_idx, proj_name)
        return None

    # Sort by expert index
    scales.sort(key=lambda x: int(x[0].split("experts.")[1].split(".")[0]))
    result = torch.stack([s[1] for s in scales[:E]])
    _original_scales[key] = result
    return result


def _convert_weights(w_data, padded_K, padded_N, E, layer_idx, proj_name, device):
    """Dequant with proper MX scales, then requant for FlyDSL. Cached."""
    cache_key = id(w_data)
    if cache_key in _weight_cache:
        return _weight_cache[cache_key]

    from tests.kernels.utils import fp4_utils

    _, K_packed, N = w_data.shape

    # Row-major weights
    w_row = w_data.permute(0, 2, 1).contiguous()  # [E, N, K_packed]

    # Load original scales
    orig_scales = _load_original_scales(layer_idx, proj_name, E, device)
    if orig_scales is not None:
        # orig_scales: [E, N_orig, K//32] where N_orig may differ from N (vLLM pads N)
        N_orig = orig_scales.shape[1]
        K_scale_orig = orig_scales.shape[2]

        # Properly dequantize: fp4_value * mx_scale
        # Only dequant the original (unpadded) portion
        w_orig = w_row[:, :N_orig, :K_scale_orig * 16]  # K_packed for original K
        w_fp32 = _mxfp4_dequant_with_scales(w_orig, orig_scales)
        logger.info("Proper dequant: %s -> %s min=%.2f max=%.2f",
                    list(w_orig.shape), list(w_fp32.shape), w_fp32.min(), w_fp32.max())
    else:
        # Fallback: dequant without scales (lossy)
        w_fp32 = fp4_utils.mxfp4_to_f32(w_row)
        logger.warning("Using lossy dequant (no original scales found)")

    # Pad to target dimensions
    full_K = padded_K
    if w_fp32.shape[-1] < full_K:
        w_fp32 = torch.cat([w_fp32,
            torch.zeros(E, w_fp32.shape[1], full_K - w_fp32.shape[-1],
                       dtype=torch.float32, device=device)], dim=-1)
    if w_fp32.shape[1] < padded_N:
        w_fp32 = torch.cat([w_fp32,
            torch.zeros(E, padded_N - w_fp32.shape[1], full_K,
                       dtype=torch.float32, device=device)], dim=1)

    # Requantize with FlyDSL format
    w_q, w_scale, _ = fp4_utils.per_1x32_f4_quant(w_fp32)
    del w_fp32
    torch.cuda.empty_cache()

    w_shuffled = fp4_utils.shuffle_weight_w4(w_q, 16, True, True)
    w_scale_shuffled = fp4_utils.shuffle_scale_w4(w_scale, E, True)

    result = (w_shuffled, w_scale_shuffled)
    _weight_cache[cache_key] = result
    logger.info("FlyDSL weight done: E=%d [%d,%d]->[%d,%d]",
                E, N, K_packed*2, padded_N, padded_K)
    return result


_layer_counter = 0


def flydsl_mxfp4_w4a8_experts(
    hidden_states, w1, w2, gating_output, topk, renormalize,
    quant_config=None, apply_router_weight_on_input=False,
    global_num_experts=-1, expert_map=None,
    unpadded_N_w1=None, unpadded_K_w1=None,
    unpadded_N_w2=None, unpadded_K_w2=None,
):
    global _layer_counter
    from aiter.fused_moe import moe_sorting
    from aiter.ops.triton.quant_moe import downcast_to_static_fp8
    from tests.kernels.utils import fp4_utils

    assert quant_config is not None
    M = hidden_states.shape[0]
    actual_K = hidden_states.shape[-1]
    padded_K = _pad_dim(actual_K)
    device = hidden_states.device

    w1_data = w1.storage.data if hasattr(w1, 'storage') else w1
    w2_data = w2.storage.data if hasattr(w2, 'storage') else w2
    E = w1_data.shape[0]
    N_w1 = w1_data.shape[2]
    N_w2 = w2_data.shape[2]

    inter_dim_padded = _pad_dim(N_w1 // 2)
    padded_N_w1 = 2 * inter_dim_padded
    padded_N_w2 = padded_K

    layer_idx = _layer_counter % 36
    if id(w1_data) not in _weight_cache:
        _layer_counter += 1
        layer_idx = (_layer_counter - 1) % 36

    # Routing
    sm_first = not renormalize
    logits = torch.softmax(gating_output.float(), dim=-1) if sm_first else gating_output.float()
    topk_vals, topk_ids = torch.topk(logits, k=topk, dim=-1)
    topk_weights = topk_vals if sm_first else torch.softmax(topk_vals, dim=-1)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, _TILE_M, None)
    blocks = int(sorted_expert_ids.numel())

    # Pad activations
    if padded_K > actual_K:
        hidden_padded = torch.zeros(M, padded_K, dtype=hidden_states.dtype, device=device)
        hidden_padded[:, :actual_K] = hidden_states
    else:
        hidden_padded = hidden_states

    x_scale = quant_config.w1_precision.flex_ctx.lhs_data.scale
    x_fp8 = downcast_to_static_fp8(hidden_padded, x_scale)
    a_scale = torch.ones([M, padded_K // 32], dtype=fp4_utils.fp8_e8m0, device=device)

    # Convert weights (with proper MX scale application)
    w1_kernel, w1_scale = _convert_weights(
        w1_data, padded_K, padded_N_w1, E, layer_idx, "gate_up_proj", device)
    w2_kernel, w2_scale = _convert_weights(
        w2_data, inter_dim_padded, padded_N_w2, E, layer_idx, "down_proj", device)

    # Stage 1
    stage1 = _compile_stage1(padded_K, inter_dim_padded, E, topk, _TILE_N)
    out1 = torch.empty(M, topk, inter_dim_padded, dtype=torch.float16, device=device)
    bias1 = torch.empty(0, device=device, dtype=torch.float32)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    stage1(out1, x_fp8.contiguous().view(M, padded_K),
           w1_kernel, a_scale.view(-1), w1_scale,
           sorted_ids, sorted_expert_ids, sorted_weights,
           num_valid_ids, bias1,
           M, 2 * inter_dim_padded, padded_K, blocks, stream_ptr)

    # Quantize intermediate
    out1_fp8 = out1.view(-1, inter_dim_padded).to(torch.float8_e4m3fn)
    a2_scale = torch.ones([M * topk, inter_dim_padded // 32],
                          dtype=fp4_utils.fp8_e8m0, device=device)

    # Stage 2
    stage2 = _compile_stage2(padded_K, inter_dim_padded, E, topk, _TILE_N)
    out2 = torch.zeros(M, padded_K, dtype=torch.float16, device=device)
    bias2 = torch.empty(0, device=device, dtype=torch.float32)

    stage2(out2, out1_fp8.contiguous().view(-1),
           w2_kernel, a2_scale.view(-1), w2_scale,
           sorted_ids, sorted_expert_ids, sorted_weights,
           num_valid_ids, bias2,
           M, padded_K, inter_dim_padded, blocks, stream_ptr)

    return out2[:, :actual_K].to(hidden_states.dtype)
