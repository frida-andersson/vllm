# SPDX-License-Identifier: Apache-2.0
"""CK/AITER 2-stage MoE with dimension padding for gpt-oss-120b.

Pads K from 2880 to 3072 so K//32=96 passes the (N_i // 2) % 2 == 0
assertion in AITER's mxfp4_sort. Uses the existing fused_moe_2stages
CK kernel which is already optimized for MXFP4 on gfx950.
"""
import logging
import torch

logger = logging.getLogger(__name__)

_PAD_MULTIPLE = 128  # K//32 must be divisible by 4, so K must be multiple of 128
_padded_weight_cache: dict = {}


def _pad_dim(x: int) -> int:
    """Pad to next multiple where K//32 is divisible by 4."""
    target = ((x + _PAD_MULTIPLE - 1) // _PAD_MULTIPLE) * _PAD_MULTIPLE
    while (target // 32) % 4 != 0:
        target += _PAD_MULTIPLE
    return target


def _pad_weights_and_scales(w_data, w_scale_data, actual_K, padded_K):
    """Pad weight and scale K dimensions. Cached."""
    cache_key = id(w_data)
    if cache_key in _padded_weight_cache:
        return _padded_weight_cache[cache_key]

    E, N, K_packed = w_data.shape  # row-major: [E, N, K_packed]
    actual_K_packed = K_packed
    padded_K_packed = padded_K // 2

    _, _, K_scale = w_scale_data.shape  # row-major: [E, N, K_scale]
    padded_K_scale = padded_K // 32

    # Pad weight K dimension (row-major: [E, N, K_packed])
    if padded_K_packed > actual_K_packed:
        pad_w = torch.zeros(E, N, padded_K_packed - actual_K_packed,
                           dtype=torch.uint8, device=w_data.device).view(w_data.dtype)
        w_padded = torch.cat([w_data, pad_w], dim=-1)
    else:
        w_padded = w_data

    # Pad scale K dimension (row-major: [E, N, K_scale])
    if padded_K_scale > K_scale:
        pad_s = torch.zeros(E, N, padded_K_scale - K_scale,
                           dtype=torch.uint8, device=w_scale_data.device).view(w_scale_data.dtype)
        s_padded = torch.cat([w_scale_data, pad_s], dim=-1)
    else:
        s_padded = w_scale_data

    result = (w_padded, s_padded)
    _padded_weight_cache[cache_key] = result
    logger.info("CK MoE padding: E=%d N=%d K=%d->%d K_scale=%d->%d",
                E, N, actual_K, padded_K, K_scale, padded_K_scale)
    return result


def ck_mxfp4_w4a8_experts(
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
    """Route MXFP4 MoE through AITER's CK 2-stage kernel with K-padding."""
    import aiter
    from aiter import QuantType, ActivationType
    from aiter.fused_moe import fused_moe_2stages, moe_sorting

    assert quant_config is not None
    assert hidden_states.dtype == torch.bfloat16

    M = hidden_states.shape[0]
    actual_K = hidden_states.shape[-1]
    padded_K = _pad_dim(actual_K)
    device = hidden_states.device

    w1_data = w1.storage.data if hasattr(w1, 'storage') else w1
    w2_data = w2.storage.data if hasattr(w2, 'storage') else w2
    E = w1_data.shape[0]

    # Weights: col-major [E, K_packed, N] -> row-major [E, N, K_packed]
    if w1_data.dtype == torch.uint8:
        w1_data = w1_data.view(torch.float4_e2m1fn_x2)
    if w2_data.dtype == torch.uint8:
        w2_data = w2_data.view(torch.float4_e2m1fn_x2)

    w1_row = w1_data.permute(0, 2, 1).contiguous()
    w2_row = w2_data.permute(0, 2, 1).contiguous()

    # Scales: vLLM's processed scales are CDNA4-swizzled [E, K, N] -- NOT usable.
    # Load original per-block E8M0 scales from safetensors: [E, N_orig, K//32]
    import glob
    from safetensors import safe_open

    def load_scales(layer_idx, proj_name):
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
        return torch.stack([s[1] for s in scales[:E]])

    # Determine layer index (rough: use weight cache key count)
    global _layer_idx_counter
    if not hasattr(ck_mxfp4_w4a8_experts, '_layer_idx'):
        ck_mxfp4_w4a8_experts._layer_idx = 0
    layer_idx = ck_mxfp4_w4a8_experts._layer_idx % 36
    if id(w1_data) not in _padded_weight_cache:
        ck_mxfp4_w4a8_experts._layer_idx += 1
        layer_idx = (ck_mxfp4_w4a8_experts._layer_idx - 1) % 36

    w1_scale_orig = load_scales(layer_idx, "gate_up_proj")  # [E, N_orig, K//32]
    w2_scale_orig = load_scales(layer_idx, "down_proj")     # [E, N_orig, K//32]
    logger.info("Loaded scales: w1=%s w2=%s", list(w1_scale_orig.shape), list(w2_scale_orig.shape))

    # Pad scale N if weight N was padded by vLLM
    N_w1 = w1_row.shape[1]
    N_w2 = w2_row.shape[1]
    if w1_scale_orig.shape[1] < N_w1:
        pad = torch.zeros(E, N_w1 - w1_scale_orig.shape[1], w1_scale_orig.shape[2],
                         dtype=w1_scale_orig.dtype, device=device)
        w1_scale_orig = torch.cat([w1_scale_orig, pad], dim=1)
    if w2_scale_orig.shape[1] < N_w2:
        pad = torch.zeros(E, N_w2 - w2_scale_orig.shape[1], w2_scale_orig.shape[2],
                         dtype=w2_scale_orig.dtype, device=device)
        w2_scale_orig = torch.cat([w2_scale_orig, pad], dim=1)

    w1_scale_row = w1_scale_orig
    w2_scale_row = w2_scale_orig

    # Pad hidden_states: [M, actual_K] -> [M, padded_K]
    if padded_K > actual_K:
        hidden_padded = torch.zeros(M, padded_K, dtype=hidden_states.dtype, device=device)
        hidden_padded[:, :actual_K] = hidden_states
    else:
        hidden_padded = hidden_states

    # Pad weights and scales K dim (now row-major: [E, N, K_packed])
    w1_padded, w1s_padded = _pad_weights_and_scales(w1_row, w1_scale_row, actual_K, padded_K)
    w2_padded, w2s_padded = _pad_weights_and_scales(w2_row, w2_scale_row, actual_K, padded_K)

    # Routing: gating_output -> topk -> moe_sorting
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

    block_m = 32
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, block_m, None
    )

    # Allocate output
    moe_out = torch.empty(M, padded_K, dtype=hidden_states.dtype, device=device)

    # Call CK 2-stage kernel
    fused_moe_2stages(
        hidden_states=hidden_padded,
        w1=w1_padded, w2=w2_padded,
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
        w1_scale=w1s_padded,
        w2_scale=w2s_padded,
    )

    # Trim output back to actual dimensions
    return moe_out[:, :actual_K]
