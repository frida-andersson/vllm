# SPDX-License-Identifier: Apache-2.0
"""FlyDSL MXFP4 MoE integration for vLLM.

Loads original MX scales from safetensors (vLLM's processed scales are in
CDNA4-swizzled format incompatible with FlyDSL). Directly preshuffles stored
fp4 weight bytes and original e8m0 scales.
"""
import functools
import glob
import logging
import os

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
    logger.info("Compiling FlyDSL stage1 (fp4): dim=%d inter=%d E=%d topk=%d",
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
    logger.info("Compiling FlyDSL stage2 (fp4): dim=%d inter=%d E=%d topk=%d",
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
            keys = list(f.keys())
            for eidx in range(1000):
                skey = f"model.layers.{layer_idx}.mlp.experts.{eidx}.{proj_name}.weight_scale"
                if skey in keys:
                    scales.append(f.get_tensor(skey).to(device))
        if len(scales) >= E:
            break

    if len(scales) < E:
        logger.warning("Only found %d/%d expert scales for layer %d %s",
                       len(scales), E, layer_idx, proj_name)
        return None

    result = torch.stack(scales[:E])
    _original_scales[key] = result
    logger.info("Loaded original scales: layer=%d proj=%s shape=%s",
                layer_idx, proj_name, list(result.shape))
    return result


def _convert_weights(w_data, padded_K, padded_N, E, layer_idx, proj_name, device):
    """Preshuffle MXFP4 weights + original MX scales for FlyDSL. Cached."""
    cache_key = id(w_data)
    if cache_key in _weight_cache:
        return _weight_cache[cache_key]

    from tests.kernels.utils import fp4_utils

    _, K_packed, N = w_data.shape

    # Weights: col-major [E, K_packed, N] -> row-major [E, N, K_packed]
    w_row = w_data.permute(0, 2, 1).contiguous()
    w_fp4 = w_row.view(torch.float4_e2m1fn_x2)

    # Load ORIGINAL scales from safetensors: [E, N, K//32]
    orig_scales = _load_original_scales(layer_idx, proj_name, E, device)
    if orig_scales is None:
        raise RuntimeError(f"Could not load original scales for layer {layer_idx} {proj_name}")

    s_e8m0 = orig_scales.view(fp4_utils.fp8_e8m0)
    K_scale = s_e8m0.shape[-1]

    # Pad K
    target_K_packed = padded_K // 2
    target_K_scale = padded_K // 32
    if target_K_packed > K_packed:
        w_fp4 = torch.cat([w_fp4,
            torch.zeros(E, N, target_K_packed - K_packed, dtype=torch.uint8,
                       device=device).view(torch.float4_e2m1fn_x2)], dim=-1)
    if target_K_scale > K_scale:
        s_e8m0 = torch.cat([s_e8m0,
            torch.zeros(E, N, target_K_scale - K_scale, dtype=torch.uint8,
                       device=device).view(fp4_utils.fp8_e8m0)], dim=-1)

    # Pad N
    if padded_N > N:
        w_fp4 = torch.cat([w_fp4,
            torch.zeros(E, padded_N - N, target_K_packed, dtype=torch.uint8,
                       device=device).view(torch.float4_e2m1fn_x2)], dim=1)
        s_e8m0 = torch.cat([s_e8m0,
            torch.zeros(E, padded_N - N, target_K_scale, dtype=torch.uint8,
                       device=device).view(fp4_utils.fp8_e8m0)], dim=1)

    # Preshuffle weights
    w_shuffled = fp4_utils.shuffle_weight_w4(w_fp4, 16, True, True)

    # Reshape scales to [E*padded_N, target_K_scale] then shuffle
    s_flat = s_e8m0.reshape(E * padded_N, target_K_scale).contiguous()
    w_scale_shuffled = fp4_utils.shuffle_scale_w4(s_flat, E, True)

    result = (w_shuffled, w_scale_shuffled)
    _weight_cache[cache_key] = result
    logger.info("FlyDSL weight conversion done: E=%d N=%d->%d K_packed=%d->%d scale_K=%d->%d",
                E, N, padded_N, K_packed, target_K_packed, K_scale, target_K_scale)
    return result


# Track layer index for scale loading
_layer_counter = 0


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
    """FlyDSL MoE: takes gating_output, handles routing and weight conversion."""
    global _layer_counter
    from aiter.fused_moe import moe_sorting
    from aiter.ops.triton.quant_moe import downcast_to_static_fp8
    from tests.kernels.utils import fp4_utils

    assert quant_config is not None
    assert hidden_states.dtype == torch.bfloat16

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

    # Determine layer index from call order (each layer calls this twice: w1+w2)
    # The weight cache key (tensor id) disambiguates layers
    layer_idx = _layer_counter % 36  # 36 MoE layers
    if id(w1_data) not in _weight_cache:
        _layer_counter += 1
        layer_idx = (_layer_counter - 1) % 36

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
    blocks = int(sorted_expert_ids.numel())

    # --- Pad activations ---
    if padded_K > actual_K:
        hidden_padded = torch.zeros(M, padded_K, dtype=hidden_states.dtype, device=device)
        hidden_padded[:, :actual_K] = hidden_states
    else:
        hidden_padded = hidden_states

    # --- Quantize to FP8 ---
    x_scale = quant_config.w1_precision.flex_ctx.lhs_data.scale
    x_fp8 = downcast_to_static_fp8(hidden_padded, x_scale)
    a_scale = torch.ones([M, padded_K // 32], dtype=fp4_utils.fp8_e8m0, device=device)

    # --- Convert weights (cached, loads original scales from disk) ---
    w1_kernel, w1_scale = _convert_weights(
        w1_data, padded_K, padded_N_w1, E, layer_idx, "gate_up_proj", device)
    w2_kernel, w2_scale = _convert_weights(
        w2_data, inter_dim_padded, padded_N_w2, E, layer_idx, "down_proj", device)

    # --- Stage 1 ---
    stage1 = _compile_stage1(padded_K, inter_dim_padded, E, topk, _TILE_N)
    out1 = torch.empty(M, topk, inter_dim_padded, dtype=torch.float16, device=device)
    bias1 = torch.empty(0, device=device, dtype=torch.float32)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    stage1(out1, x_fp8.contiguous().view(M, padded_K),
           w1_kernel, a_scale.view(-1), w1_scale,
           sorted_ids, sorted_expert_ids, sorted_weights,
           num_valid_ids, bias1,
           M, 2 * inter_dim_padded, padded_K, blocks, stream_ptr)

    # --- Quantize intermediate ---
    out1_fp8 = out1.view(-1, inter_dim_padded).to(torch.float8_e4m3fn)
    a2_scale = torch.ones([M * topk, inter_dim_padded // 32],
                          dtype=fp4_utils.fp8_e8m0, device=device)

    # --- Stage 2 ---
    stage2 = _compile_stage2(padded_K, inter_dim_padded, E, topk, _TILE_N)
    out2 = torch.zeros(M, padded_K, dtype=torch.float16, device=device)
    bias2 = torch.empty(0, device=device, dtype=torch.float32)

    stage2(out2, out1_fp8.contiguous().view(-1),
           w2_kernel, a2_scale.view(-1), w2_scale,
           sorted_ids, sorted_expert_ids, sorted_weights,
           num_valid_ids, bias2,
           M, padded_K, inter_dim_padded, blocks, stream_ptr)

    return out2[:, :actual_K].to(hidden_states.dtype)
