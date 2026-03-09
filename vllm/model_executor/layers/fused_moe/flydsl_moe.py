# SPDX-License-Identifier: Apache-2.0
"""FlyDSL MXFP4 MoE integration for vLLM.

Directly preshuffles stored MXFP4 weight bytes and original E8M0 scales
for FlyDSL's kernel layout. No dequant/requant -- pure byte permutation.
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
                if (f"layers.{layer_idx}.mlp.experts." in skey
                    and proj_name in skey and "weight_scale" in skey):
                    scales.append((skey, f.get_tensor(skey).to(device)))
        if len(scales) >= E:
            break

    scales.sort(key=lambda x: int(x[0].split("experts.")[1].split(".")[0]))
    result = torch.stack([s[1] for s in scales[:E]])
    _original_scales[key] = result
    return result


def _convert_weights(w_data, padded_K, padded_N, E, layer_idx, proj_name, is_gate_up, device):
    """Preshuffle stored MXFP4 weights + original scales. No dequant/requant.

    Pure byte permutation matching FlyDSL's MFMA lane access pattern.
    """
    cache_key = id(w_data)
    if cache_key in _weight_cache:
        return _weight_cache[cache_key]

    from tests.kernels.utils import fp4_utils

    _, K_packed, N = w_data.shape
    actual_K_scale = K_packed * 2 // 32  # K//32

    # Weights: col-major [E, K_packed, N] -> row-major [E, N, K_packed], view as fp4x2
    w_fp4 = w_data.permute(0, 2, 1).contiguous().view(torch.float4_e2m1fn_x2)

    # Load original scales: [E, N_orig, K_scale_orig]
    orig_scales = _load_original_scales(layer_idx, proj_name, E, device)
    N_orig = orig_scales.shape[1]
    K_scale_orig = orig_scales.shape[2]
    s_e8m0 = orig_scales.view(fp4_utils.fp8_e8m0)

    target_K_packed = padded_K // 2
    target_K_scale = padded_K // 32

    # Pad K (weights)
    if target_K_packed > K_packed:
        w_fp4 = torch.cat([w_fp4,
            torch.zeros(E, N, target_K_packed - K_packed, dtype=torch.uint8,
                       device=device).view(torch.float4_e2m1fn_x2)], dim=-1)
    # Pad K (scales)
    if target_K_scale > K_scale_orig:
        s_e8m0 = torch.cat([s_e8m0,
            torch.zeros(E, N_orig, target_K_scale - K_scale_orig, dtype=torch.uint8,
                       device=device).view(fp4_utils.fp8_e8m0)], dim=-1)

    # Pad N (weights -- vLLM may have already padded, so check)
    if padded_N > N:
        w_fp4 = torch.cat([w_fp4,
            torch.zeros(E, padded_N - N, target_K_packed, dtype=torch.uint8,
                       device=device).view(torch.float4_e2m1fn_x2)], dim=1)
    # Pad N (scales -- original is always unpadded)
    if padded_N > N_orig:
        s_e8m0 = torch.cat([s_e8m0,
            torch.zeros(E, padded_N - N_orig, target_K_scale, dtype=torch.uint8,
                       device=device).view(fp4_utils.fp8_e8m0)], dim=1)

    # Preshuffle weights (pure byte permutation)
    w_shuf = fp4_utils.shuffle_weight_w4(w_fp4, 16, is_gate_up, True)

    # Preshuffle scales: flatten to [E*padded_N, target_K_scale] then shuffle
    s_flat = s_e8m0.reshape(E * padded_N, target_K_scale).contiguous()
    s_shuf = fp4_utils.shuffle_scale_w4(s_flat, E, is_gate_up)

    result = (w_shuf, s_shuf)
    _weight_cache[cache_key] = result
    logger.info("FlyDSL preshuffle done (no dequant): E=%d N=%d->%d K=%d->%d",
                E, N_orig, padded_N, K_packed * 2, padded_K)
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

    # Convert weights (direct preshuffle, no dequant)
    w1_kernel, w1_scale = _convert_weights(
        w1_data, padded_K, padded_N_w1, E, layer_idx, "gate_up_proj", True, device)
    w2_kernel, w2_scale = _convert_weights(
        w2_data, inter_dim_padded, padded_N_w2, E, layer_idx, "down_proj", False, device)

    # Stage 1
    stage1 = _compile_stage1(padded_K, inter_dim_padded, E, topk, _TILE_N)
    out1 = torch.empty(M, topk, inter_dim_padded, dtype=torch.float16, device=device)
    bias1 = torch.empty(0, device=device, dtype=torch.float32)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    stage1(out1, x_fp8.contiguous().view(M, padded_K),
           w1_kernel, a_scale.view(-1), w1_scale,
           sorted_ids, sorted_expert_ids, sorted_weights,
           num_valid_ids, bias1,
           M, inter_dim_padded, padded_K, blocks, stream_ptr)

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
