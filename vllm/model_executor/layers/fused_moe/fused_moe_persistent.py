# SPDX-License-Identifier: Apache-2.0
"""Optimized MoE for MXFP4: reuses existing stage1 kernel, provides a
stage2 kernel that applies routing weights inline (no separate gamma
multiply), and uses the standard reduce_grouped for CUDA-graph-compatible
topk reduction.

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
def _moe_stage2_weighted(
    Y,                # [split_k, M_sorted, N2] output, expert-sorted
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,                # [M_sorted, K2] intermediate, FP8
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
    Gammas,           # [M_sorted] routing weights, bf16
    ExptHist,
    ExptOffs,
    ExptData,
    N, K,
    grid_m, grid_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    MASK_K_LIMIT: tl.constexpr,
    W_CACHE_MODIFIER: tl.constexpr,
):
    """Stage2 GEMM with routing weights applied inline.

    Writes to expert-sorted output buffer (no atomics).
    Compatible with CUDA graph capture.
    Uses reduce_grouped afterwards for topk reduction.
    """
    MX_PACK: tl.constexpr = 32
    MX_SBK: tl.constexpr = BLOCK_K // MX_PACK
    WKD: tl.constexpr = 2
    PBK: tl.constexpr = BLOCK_K // WKD
    NKP: tl.constexpr = 32
    PMB: tl.constexpr = MX_SBK * NKP
    SBN: tl.constexpr = BLOCK_N // NKP

    pid = tl.program_id(0)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

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

    # Apply routing weights inline (fused with GEMM output)
    if Gammas is not None:
        gammas = tl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        acc *= gammas[:, None]

    # Write to expert-sorted output (regular store, CUDA-graph compatible)
    offs_out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_out_n < N
    Y_ptrs = (Y + (start_m + offs_m).to(tl.int64)[:, None] * stride_y_m
              + offs_out_n[None, :].to(tl.int64) * stride_y_n)
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(Y_ptrs, acc, mask=mask)


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
    """Optimized MoE: stage1 (existing) + stage2 with inline routing weights.

    Stage2 applies gammas inside the GEMM kernel (saves one pass over data).
    Uses standard reduce_grouped for CUDA-graph-compatible topk reduction.
    """
    from aiter.ops.triton.moe_routing.routing import routing as aiter_routing
    from aiter.ops.triton.quant_moe import downcast_to_static_fp8
    from aiter.ops.triton.moe.moe_op_gemm_a8w4 import (
        moe_gemm_a8w4, get_kernel_config, reduce_grouped, allocate_output,
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

    # Stage 2: our kernel with inline routing weights
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

    # Expert-sorted output (same layout as baseline stage2)
    y_stage2 = torch.zeros(
        (1, M_route, N2), device=hidden_states.device, dtype=torch.bfloat16)
    y_final = torch.empty(
        (M, N2), device=hidden_states.device, dtype=hidden_states.dtype)

    expt_data = routing_data.expt_data
    gammas_stage2 = None if apply_router_weight_on_input else gammas

    _moe_stage2_weighted[(grid_m * grid_n,)](
        y_stage2,
        y_stage2.stride(0), y_stage2.stride(1), y_stage2.stride(2),
        intermediate, intermediate.stride(0), intermediate.stride(1),
        w2_data, w2_data.stride(0), w2_data.stride(1), w2_data.stride(2),
        w2_scale, w2_scale.stride(0), w2_scale.stride(1), w2_scale.stride(2),
        a2_scale,
        gammas_stage2,
        expt_data.hist,
        expt_data.token_offs_raw,
        expt_data.block_pid_map,
        N2, K2,
        grid_m=grid_m, grid_n=grid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=config2["group_m"],
        EVEN_K=(K2 % BLOCK_K == 0),
        MASK_K_LIMIT=(K2 % BLOCK_K),
        W_CACHE_MODIFIER=config2["w_cache_modifier"],
        num_warps=config2["num_warps"],
        num_stages=config2["num_stages"],
    )

    # Reduce: sum topk expert contributions per token (CUDA-graph compatible)
    group_indx = scatter_idx.view(-1, topk)
    y_final = reduce_grouped(
        y_stage2, group_indx, y_final,
        False, swiglu_alpha, swiglu_limit, 1,
        out_dtype=hidden_states.dtype,
    )

    return y_final
