# SPDX-License-Identifier: Apache-2.0
"""Optimized MoE for MXFP4: reuses existing stage1 kernel but replaces
stage2 + reduce_grouped with a single stage2 kernel that does atomic
reduction inline, saving one kernel launch per MoE layer.

Enable with VLLM_USE_FUSED_MOE=1.
"""
import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)


@triton.jit
def _unswizzle_mx_scale_cdna4(
    x, BLOCK_N: tl.constexpr, MX_SCALE_BLOCK_K: tl.constexpr,
    N_PRESHUFFLE_FACTOR: tl.constexpr = 32,
):
    x = x.reshape(BLOCK_N // N_PRESHUFFLE_FACTOR,
                  MX_SCALE_BLOCK_K // 8, 4, 16, 2, 2, 1)
    x = x.permute(0, 5, 3, 1, 4, 2, 6)
    x = x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)
    return x


@triton.jit
def _moe_stage2_reduce(
    Y,                # [M, N2] output, bf16, zero-initialized
    stride_y_m,
    stride_y_n,
    X,                # [total_sorted, K2] intermediate, FP8
    stride_x_m,
    stride_x_k,
    W,                # [E, K2_packed, N2] weight, MXFP4
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WMxScale,         # [E, SK*32, N2//32] scale, CDNA4
    stride_ws_e,
    stride_ws_k,
    stride_ws_n,
    X_static_scale,   # scalar fp32
    Gammas,           # [total_sorted] routing weights, bf16
    ScatterIndx,      # [total_sorted] -> output token index
    ExptHist,
    ExptOffs,
    ExptData,
    N, K,
    N_EXPTS_ACT: tl.constexpr,
    grid_m, grid_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    MASK_K_LIMIT: tl.constexpr,
    W_CACHE_MODIFIER: tl.constexpr,
):
    """Stage2 GEMM with inline atomic reduction.

    Each block computes intermediate @ W2 for one expert's M-tile,
    applies routing weight, and atomically adds to the output at
    the correct token position (via ScatterIndx).
    Eliminates the separate reduce_grouped kernel.
    """
    MX_PACK: tl.constexpr = 32
    MX_SBK: tl.constexpr = BLOCK_K // MX_PACK
    WKD: tl.constexpr = 2
    PBK: tl.constexpr = BLOCK_K // WKD
    NKP: tl.constexpr = 32
    PMB: tl.constexpr = MX_SBK * NKP
    SBN: tl.constexpr = BLOCK_N // NKP

    pid = tl.program_id(0)
    pid_mn = pid
    pid_m = pid_mn // grid_n
    pid_n = pid_mn % grid_n

    expt_data = tl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M_expert = tl.load(ExptHist + expt_id)
    start_m = tl.load(ExptOffs + expt_id)

    offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_m_wrap = offs_m % M_expert
    mask_m = offs_m < M_expert

    # Read from intermediate (expert-sorted order)
    offs_x_m = (start_m + offs_m_wrap).to(tl.int64)
    offs_x_k = tl.arange(0, BLOCK_K)
    XPtrs = X + offs_x_m[:, None] * stride_x_m + offs_x_k[None, :].to(tl.int64) * stride_x_k

    # W2 pointers for this expert
    W_e = W + expt_id.to(tl.int64) * stride_w_e
    offs_w_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_w_n_wrap = offs_w_n % N
    offs_w_k = tl.arange(0, PBK)
    WPtrs = (W_e + offs_w_k[:, None].to(tl.int64) * stride_w_k
             + offs_w_n_wrap[None, :].to(tl.int64) * stride_w_n)

    # Scale pointers
    WS_e = WMxScale + expt_id.to(tl.int64) * stride_ws_e
    offs_ws_n = (pid_n * SBN + tl.arange(0, SBN)) % N
    offs_ws_k = tl.arange(0, PMB)
    WSPtrs = (WS_e + offs_ws_k[None, :].to(tl.int64) * stride_ws_k
              + offs_ws_n[:, None].to(tl.int64) * stride_ws_n)

    x_scales = tl.full((BLOCK_M, MX_SBK), 127, dtype=tl.uint8)

    # GEMM loop
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_iter = tl.cdiv(K, BLOCK_K)
    if not EVEN_K:
        num_k_iter -= 1

    for ki in range(num_k_iter):
        x = tl.load(XPtrs, mask=mask_m[:, None])
        w = tl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)
        ws = _unswizzle_mx_scale_cdna4(
            tl.load(WSPtrs, cache_modifier=W_CACHE_MODIFIER),
            BLOCK_N, MX_SBK)
        acc = tl.dot_scaled(x, x_scales, "e4m3", w, ws, "e2m1",
                            acc=acc, fast_math=True)
        XPtrs += BLOCK_K * stride_x_k
        WPtrs += PBK * stride_w_k
        WSPtrs += PMB * stride_ws_k

    if not EVEN_K:
        mask_k = offs_x_k < MASK_K_LIMIT
        mask_wk = offs_w_k < (MASK_K_LIMIT // WKD)
        x = tl.load(XPtrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(WPtrs, mask=mask_wk[:, None], other=0,
                    cache_modifier=W_CACHE_MODIFIER)
        ws = _unswizzle_mx_scale_cdna4(
            tl.load(WSPtrs, cache_modifier=W_CACHE_MODIFIER),
            BLOCK_N, MX_SBK)
        acc = tl.dot_scaled(x, x_scales, "e4m3", w, ws, "e2m1",
                            acc=acc, fast_math=True)

    # Scale compensation
    if X_static_scale is not None:
        acc = acc * tl.load(X_static_scale)

    # Apply routing weights
    if Gammas is not None:
        gammas = tl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        acc *= gammas[:, None]

    # Scatter-reduce: atomic add to output at token positions
    # ScatterIndx maps from expert-sorted to output token position
    offs_out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_out_n < N

    scatter_offs = start_m + offs_m_wrap
    token_idxs = tl.load(ScatterIndx + scatter_offs, mask=mask_m, other=-1)
    valid = mask_m & (token_idxs >= 0)

    out_ptrs = (Y + token_idxs[:, None].to(tl.int64) * stride_y_m
                + offs_out_n[None, :].to(tl.int64) * stride_y_n)
    out_mask = valid[:, None] & mask_n[None, :]
    tl.atomic_add(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def fused_moe_persistent(
    hidden_states: torch.Tensor,
    w1,
    w2,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    quant_config=None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    activation=None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    **kwargs,
):
    """Optimized MoE: stage1 (existing) + stage2 with inline reduction.

    Saves one kernel launch per layer by folding reduce_grouped into stage2.
    """
    from aiter.ops.triton.moe_routing.routing import routing as aiter_routing
    from aiter.ops.triton.quant_moe import downcast_to_static_fp8
    from aiter.ops.triton.moe.moe_op_gemm_a8w4 import (
        moe_gemm_a8w4, get_kernel_config,
    )

    assert quant_config is not None

    w1_data = w1.storage.data if hasattr(w1, 'storage') else w1
    w2_data = w2.storage.data if hasattr(w2, 'storage') else w2
    w1_scale = quant_config.w1_precision.weight_scale.storage.data
    w2_scale = quant_config.w2_precision.weight_scale.storage.data
    a1_scale = quant_config.w1_precision.flex_ctx.lhs_data.scale
    a2_scale = quant_config.w2_precision.flex_ctx.lhs_data.scale

    unpadded_N_w1 = getattr(quant_config, 'unpadded_N_w1', None)
    unpadded_K_w1 = getattr(quant_config, 'unpadded_K_w1', None)
    unpadded_N_w2 = getattr(quant_config, 'unpadded_N_w2', None)
    unpadded_K_w2 = getattr(quant_config, 'unpadded_K_w2', None)

    routing_data, gather_idx, scatter_idx = aiter_routing(
        gating_output, topk, sm_first=not renormalize)
    gammas = routing_data.gate_scal

    hidden_fp8 = downcast_to_static_fp8(hidden_states, a1_scale)

    # Stage 1: existing AITER kernel (X @ W1 + SwiGLU + FP8 quant)
    intermediate = moe_gemm_a8w4(
        hidden_fp8, w1_data, None, w1_scale, a1_scale, a2_scale, None,
        routing_data, gather_indx=gather_idx, scatter_indx=None,
        gammas=gammas if apply_router_weight_on_input else None,
        swizzle_mx_scale="CDNA4_SCALE",
        out_dtype=torch.float8_e4m3fn,
        apply_swiglu=True, alpha=swiglu_alpha, limit=swiglu_limit,
        unpadded_N=unpadded_N_w1, unpadded_K=unpadded_K_w1,
    )

    # Stage 2: custom kernel with inline atomic reduction
    M = hidden_states.shape[0]
    N2 = w2_data.shape[-1]
    K2 = intermediate.shape[-1]

    if unpadded_N_w2 is not None:
        N2 = unpadded_N_w2
    if unpadded_K_w2 is not None:
        K2 = unpadded_K_w2

    config2 = get_kernel_config(
        scatter_idx.shape[0] // topk, N2, K2, routing_data)
    BLOCK_M = config2["block_m"]
    BLOCK_N = config2["block_n"]
    BLOCK_K = config2["block_k"]

    M_route = gather_idx.shape[0] if gather_idx is not None else intermediate.shape[0]
    grid_m = routing_data.n_blocks(M_route, BLOCK_M)
    grid_n = triton.cdiv(N2, BLOCK_N)

    output = torch.zeros(
        (M, N2), device=hidden_states.device, dtype=torch.bfloat16)

    expt_data = routing_data.expt_data
    gammas_stage2 = None if apply_router_weight_on_input else gammas

    # Build inverse scatter map: for each expert-sorted row, which output token?
    # scatter_idx is [M * topk], scatter_idx.view(M, topk)[t, s] = expert-sorted row for token t, slot s
    # We need inverse: expert_row -> token_idx
    inv_scatter = torch.full((M_route,), -1, device=hidden_states.device, dtype=torch.int32)
    scatter_view = scatter_idx.view(-1, topk)
    for s in range(topk):
        col = scatter_view[:, s]
        valid = col >= 0
        inv_scatter[col[valid].long()] = torch.arange(M, device=hidden_states.device, dtype=torch.int32)[valid]

    _moe_stage2_reduce[(grid_m * grid_n,)](
        output, output.stride(0), output.stride(1),
        intermediate, intermediate.stride(0), intermediate.stride(1),
        w2_data, w2_data.stride(0), w2_data.stride(1), w2_data.stride(2),
        w2_scale, w2_scale.stride(0), w2_scale.stride(1), w2_scale.stride(2),
        a2_scale,
        gammas_stage2,
        inv_scatter,
        expt_data.hist,
        expt_data.token_offs_raw,
        expt_data.block_pid_map,
        N2, K2,
        N_EXPTS_ACT=topk,
        grid_m=grid_m, grid_n=grid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=config2["group_m"],
        EVEN_K=(K2 % BLOCK_K == 0),
        MASK_K_LIMIT=(K2 % BLOCK_K),
        W_CACHE_MODIFIER=config2["w_cache_modifier"],
        num_warps=config2["num_warps"],
        num_stages=config2["num_stages"],
    )

    return output.to(hidden_states.dtype)
