# SPDX-License-Identifier: Apache-2.0
"""CK/AITER 1-stage MoE with K-padding for gpt-oss-120b MXFP4.

Passes weights in SAME col-major format as Triton (no transpose!).
Overrides get_inter_dim to handle col-major shapes correctly.
"""
import glob
import logging
import torch

logger = logging.getLogger(__name__)

_PAD_MULTIPLE = 128
_cache: dict = {}
_scale_cache: dict = {}
_patched = False


def _pad_dim(x: int) -> int:
    target = ((x + _PAD_MULTIPLE - 1) // _PAD_MULTIPLE) * _PAD_MULTIPLE
    while (target // 32) % 4 != 0:
        target += _PAD_MULTIPLE
    return target


def _patch_get_inter_dim():
    """Override get_inter_dim to handle col-major [E, K_packed, N] weights."""
    global _patched
    if _patched:
        return
    _patched = True

    import aiter.fused_moe as fm
    _orig = fm.get_inter_dim.__wrapped__ if hasattr(fm.get_inter_dim, '__wrapped__') else fm.get_inter_dim

    import functools

    @functools.lru_cache(maxsize=2048)
    def patched_get_inter_dim(w1_shape, w2_shape):
        # Col-major: w1=[E, K_packed, 2*inter_dim], w2=[E, K_packed, model_dim]
        # K_packed stores 2 fp4 values per element
        E = w1_shape[0]
        K_packed = w1_shape[1]
        N_w1 = w1_shape[2]  # 2 * inter_dim
        N_w2 = w2_shape[2]  # model_dim

        model_dim = N_w2
        inter_dim = N_w1 // 2

        logger.info("patched get_inter_dim: E=%d model_dim=%d inter_dim=%d (col-major)",
                    E, model_dim, inter_dim)
        return E, model_dim, inter_dim

    fm.get_inter_dim = patched_get_inter_dim
    logger.info("Patched get_inter_dim for col-major weights")


def ck_mxfp4_w4a8_experts(
    hidden_states, w1, w2, gating_output, topk, renormalize,
    quant_config=None, apply_router_weight_on_input=False,
    global_num_experts=-1, expert_map=None,
    unpadded_N_w1=None, unpadded_K_w1=None,
    unpadded_N_w2=None, unpadded_K_w2=None,
):
    import aiter
    from aiter import QuantType, ActivationType
    from aiter.fused_moe import fused_moe_1stage, moe_sorting

    _patch_get_inter_dim()

    assert quant_config is not None
    M = hidden_states.shape[0]
    actual_K = hidden_states.shape[-1]
    padded_K = _pad_dim(actual_K)
    device = hidden_states.device

    # Get weight and scale tensors AS-IS (col-major, same as Triton receives)
    w1_data = w1.storage.data if hasattr(w1, 'storage') else w1
    w2_data = w2.storage.data if hasattr(w2, 'storage') else w2
    w1_scale = quant_config.w1_precision.weight_scale.storage.data.contiguous()
    w2_scale = quant_config.w2_precision.weight_scale.storage.data.contiguous()
    E = w1_data.shape[0]

    # View as fp4x2 if uint8
    if w1_data.dtype == torch.uint8:
        w1_data = w1_data.view(torch.float4_e2m1fn_x2)
    if w2_data.dtype == torch.uint8:
        w2_data = w2_data.view(torch.float4_e2m1fn_x2)

    # NO transpose -- pass col-major [E, K_packed, N] directly
    # The CK kernel should handle this the same way Triton does

    # Pad hidden_states if needed
    if padded_K > actual_K:
        hidden_padded = torch.zeros(M, padded_K, dtype=hidden_states.dtype, device=device)
        hidden_padded[:, :actual_K] = hidden_states
    else:
        hidden_padded = hidden_states

    # Routing
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
        topk_ids, topk_weights, E, block_m, None)

    moe_out = torch.empty(M, actual_K, dtype=hidden_states.dtype, device=device)

    fused_moe_1stage(
        hidden_states=hidden_padded,
        w1=w1_data, w2=w2_data,
        topk=topk,
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        moe_buf=moe_out,
        isG1U1=True,
        block_size_M=block_m,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        q_dtype_a=aiter.dtypes.fp4x2,
        q_dtype_w=aiter.dtypes.fp4x2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    return moe_out
