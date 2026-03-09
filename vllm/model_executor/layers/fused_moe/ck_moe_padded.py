# SPDX-License-Identifier: Apache-2.0
"""CK/AITER 2-stage MoE with dimension padding for gpt-oss-120b.

Pads K from 2880 to 3072 so K//32=96 passes AITER's assertion.
Uses original safetensors scales instead of CDNA4-swizzled vLLM scales.
"""
import glob
import logging
import torch

logger = logging.getLogger(__name__)

_PAD_MULTIPLE = 128
_cache: dict = {}  # cached (w_padded, s_padded) by tensor id
_scale_cache: dict = {}  # cached original scales by (layer_idx, proj_name)
_layer_counter = 0


def _pad_dim(x: int) -> int:
    target = ((x + _PAD_MULTIPLE - 1) // _PAD_MULTIPLE) * _PAD_MULTIPLE
    while (target // 32) % 4 != 0:
        target += _PAD_MULTIPLE
    return target


def _load_scales(layer_idx, proj_name, E, device):
    key = (layer_idx, proj_name)
    if key in _scale_cache:
        return _scale_cache[key]

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
    _scale_cache[key] = result
    logger.info("Loaded original scales: layer=%d proj=%s shape=%s", layer_idx, proj_name, list(result.shape))
    return result


def _get_padded(w_data, layer_idx, proj_name, padded_K, device):
    """Get padded weight + original scale. Cached."""
    ckey = id(w_data)
    if ckey in _cache:
        return _cache[ckey]

    E, K_packed, N = w_data.shape
    padded_K_packed = padded_K // 2
    padded_K_scale = padded_K // 32

    # Weight: col-major [E, K_packed, N] -> row-major [E, N, K_packed]
    w_fp4 = w_data.view(torch.float4_e2m1fn_x2) if w_data.dtype == torch.uint8 else w_data
    w_row = w_fp4.permute(0, 2, 1).contiguous()

    # Pad weight K
    if padded_K_packed > K_packed:
        pad = torch.zeros(E, N, padded_K_packed - K_packed,
                         dtype=torch.uint8, device=device).view(torch.float4_e2m1fn_x2)
        w_row = torch.cat([w_row, pad], dim=-1)

    # Load original scales: [E, N_orig, K//32]
    s_orig = _load_scales(layer_idx, proj_name, E, device)
    N_orig = s_orig.shape[1]
    K_scale_orig = s_orig.shape[2]

    # Pad scale K
    if padded_K_scale > K_scale_orig:
        pad = torch.zeros(E, N_orig, padded_K_scale - K_scale_orig,
                         dtype=s_orig.dtype, device=device)
        s_orig = torch.cat([s_orig, pad], dim=-1)

    # Pad scale N (original is unpadded, weight N may be padded by vLLM)
    if N > N_orig:
        pad = torch.zeros(E, N - N_orig, padded_K_scale,
                         dtype=s_orig.dtype, device=device)
        s_orig = torch.cat([s_orig, pad], dim=1)

    result = (w_row, s_orig)
    _cache[ckey] = result
    logger.info("CK MoE cached: E=%d N=%d K=%d->%d scale=%s",
                E, N, K_packed*2, padded_K, list(s_orig.shape))
    return result


def ck_mxfp4_w4a8_experts(
    hidden_states, w1, w2, gating_output, topk, renormalize,
    quant_config=None, apply_router_weight_on_input=False,
    global_num_experts=-1, expert_map=None,
    unpadded_N_w1=None, unpadded_K_w1=None,
    unpadded_N_w2=None, unpadded_K_w2=None,
):
    global _layer_counter
    import aiter
    from aiter import QuantType, ActivationType
    from aiter.fused_moe import fused_moe_2stages, moe_sorting

    assert quant_config is not None
    M = hidden_states.shape[0]
    actual_K = hidden_states.shape[-1]
    padded_K = _pad_dim(actual_K)
    device = hidden_states.device

    w1_data = w1.storage.data if hasattr(w1, 'storage') else w1
    w2_data = w2.storage.data if hasattr(w2, 'storage') else w2
    E = w1_data.shape[0]

    # Determine layer index for scale loading
    layer_idx = _layer_counter % 36
    if id(w1_data) not in _cache:
        _layer_counter += 1
        layer_idx = (_layer_counter - 1) % 36

    # Get padded weights + original scales (cached)
    w1_row, w1_scale = _get_padded(w1_data, layer_idx, "gate_up_proj", padded_K, device)
    w2_row, w2_scale = _get_padded(w2_data, layer_idx, "down_proj", padded_K, device)

    # Pad hidden_states
    if padded_K > actual_K:
        hidden_padded = torch.zeros(M, padded_K, dtype=hidden_states.dtype, device=device)
        hidden_padded[:, :actual_K] = hidden_states
    else:
        hidden_padded = hidden_states

    # Routing
    sm_first = not renormalize
    logits = torch.softmax(gating_output.float(), dim=-1) if sm_first else gating_output.float()
    topk_vals, topk_ids = torch.topk(logits, k=topk, dim=-1)
    topk_weights = topk_vals if sm_first else torch.softmax(topk_vals, dim=-1)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    block_m = 32
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, block_m, None)

    moe_out = torch.empty(M, padded_K, dtype=hidden_states.dtype, device=device)

    fused_moe_2stages(
        hidden_states=hidden_padded,
        w1=w1_row, w2=w2_row,
        topk=topk,
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        moe_out=moe_out,
        isG1U1=True,
        block_size_M=block_m,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        q_dtype_a=aiter.dtypes.fp4x2,
        q_dtype_w=aiter.dtypes.fp4x2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    return moe_out[:, :actual_K]
