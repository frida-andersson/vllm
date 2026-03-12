# SPDX-License-Identifier: Apache-2.0
"""Persistent fused Triton MoE kernel for MXFP4 (FP8 x FP4).

Single kernel launch that does:
  Phase 1: X @ W1 (FP8xFP4) + SwiGLU + FP8 quant -> intermediate
  Global barrier
  Phase 2: intermediate @ W2 (FP8xFP4) + routing weight + reduce -> output

Eliminates kernel launch overhead and Python-level coordination between stages.
"""
import os
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
def _swiglu(input, alpha, limit):
    gelu, linear = tl.split(tl.reshape(
        input, (input.shape[0], input.shape[1] // 2, 2)))
    gelu = gelu.to(tl.float32)
    gelu = tl.minimum(gelu, limit)
    linear = linear.to(tl.float32)
    linear = tl.minimum(linear, limit)
    linear = tl.maximum(linear, -limit)
    s = gelu / (1 + tl.exp2(-1.44269504089 * alpha * gelu))
    return tl.fma(s, linear, s)


@triton.jit
def _compute_static_fp8_quant(tensor, scale):
    tensor = tensor.to(tl.float32)
    tensor = tensor / scale
    tensor = tensor.to(tl.float8e4nv)
    return tensor


@triton.jit
def _gemm_core(
    X, stride_x_m, stride_x_k,
    W, stride_w_k, stride_w_n,
    WMxScale, stride_w_mx_k, stride_w_mx_n,
    offs_x_m, offs_x_k,
    offs_w_k, offs_w_n,
    offs_w_k_scale, offs_w_n_scale,
    WMxScalePtrs,
    N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr, MASK_K_LIMIT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    W_CACHE_MODIFIER: tl.constexpr,
    mask_m,
):
    """Core GEMM loop: computes X @ W with FP8xFP4 dot_scaled."""
    MX_PACK_DIVISOR: tl.constexpr = 32
    MX_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // MX_PACK_DIVISOR
    W_K_DIVISOR: tl.constexpr = 2
    PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K // W_K_DIVISOR
    NON_K_PRESHUFFLE_BLOCK_SIZE: tl.constexpr = 32
    PACKED_MX_BLOCK: tl.constexpr = MX_SCALE_BLOCK_K * NON_K_PRESHUFFLE_BLOCK_SIZE

    XPtrs = (X + offs_x_m[:, None].to(tl.int64) * stride_x_m
             + offs_x_k[None, :].to(tl.int64) * stride_x_k)
    WPtrs = (W + offs_w_k[:, None].to(tl.int64) * stride_w_k
             + offs_w_n[None, :].to(tl.int64) * stride_w_n)

    x_scales = tl.full((BLOCK_M, MX_SCALE_BLOCK_K), 127, dtype=tl.uint8)

    num_k_iter = tl.cdiv(K, BLOCK_K * SPLIT_K)
    if not EVEN_K:
        num_k_iter -= 1

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(num_k_iter):
        x = tl.load(XPtrs)
        w = tl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)
        w_scales = _unswizzle_mx_scale_cdna4(
            tl.load(WMxScalePtrs, cache_modifier=W_CACHE_MODIFIER),
            BLOCK_N, MX_SCALE_BLOCK_K)
        acc = tl.dot_scaled(x, x_scales, "e4m3", w, w_scales, "e2m1",
                            acc=acc, fast_math=True)
        WMxScalePtrs += (PACKED_MX_BLOCK * SPLIT_K) * stride_w_mx_k
        XPtrs += (BLOCK_K * SPLIT_K) * stride_x_k
        WPtrs += (PACKED_BLOCK_K_W * SPLIT_K) * stride_w_k

    if not EVEN_K:
        mask_x_k = offs_x_k < MASK_K_LIMIT
        mask_w_k = offs_w_k < (MASK_K_LIMIT // W_K_DIVISOR)
        x = tl.load(XPtrs, mask=mask_x_k[None, :], other=0.0)
        w = tl.load(WPtrs, mask=mask_w_k[:, None], other=0,
                    cache_modifier=W_CACHE_MODIFIER)
        w_scales = _unswizzle_mx_scale_cdna4(
            tl.load(WMxScalePtrs, cache_modifier=W_CACHE_MODIFIER),
            BLOCK_N, MX_SCALE_BLOCK_K)
        acc = tl.dot_scaled(x, x_scales, "e4m3", w, w_scales, "e2m1",
                            acc=acc, fast_math=True)
    return acc


@triton.jit
def _fused_moe_persistent(
    # Phase 1 inputs
    X, stride_x_m, stride_x_k,
    W1, stride_w1_e, stride_w1_k, stride_w1_n,
    W1MxScale, stride_w1s_e, stride_w1s_k, stride_w1s_n,
    X_static_scale, Quant_static_scale,
    # Phase 2 inputs
    W2, stride_w2_e, stride_w2_k, stride_w2_n,
    W2MxScale, stride_w2s_e, stride_w2s_k, stride_w2s_n,
    A2_static_scale,
    # Routing
    GatherIndx, Gammas,
    ExptHist, ExptOffs, ExptOffsSum, ExptData,
    # Intermediate buffer
    Intermediate, stride_inter_m, stride_inter_n,
    # Output + scatter
    Out, stride_out_m, stride_out_n,
    ScatterIndx,
    # Barrier
    barrier,
    # Dimensions
    N1, K1,
    N2, K2,
    total_phase1_blocks,
    # Grid
    grid_m, grid_n1, grid_n2,
    # Activation
    alpha, limit,
    # MoE config
    N_EXPTS_ACT: tl.constexpr,
    # Tile config
    BLOCK_M: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_K1: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_K2: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K1: tl.constexpr,
    MASK_K1_LIMIT: tl.constexpr,
    EVEN_K2: tl.constexpr,
    MASK_K2_LIMIT: tl.constexpr,
    W_CACHE_MODIFIER: tl.constexpr,
):
    """Persistent fused MoE: phase1 (W1+SwiGLU+quant) -> barrier -> phase2 (W2+reduce)."""
    pid = tl.program_id(0)

    MX_PACK: tl.constexpr = 32
    MX_SBK1: tl.constexpr = BLOCK_K1 // MX_PACK
    MX_SBK2: tl.constexpr = BLOCK_K2 // MX_PACK
    WKD: tl.constexpr = 2
    PBK1: tl.constexpr = BLOCK_K1 // WKD
    PBK2: tl.constexpr = BLOCK_K2 // WKD
    NKP: tl.constexpr = 32
    PMB1: tl.constexpr = MX_SBK1 * NKP
    PMB2: tl.constexpr = MX_SBK2 * NKP
    SBN1: tl.constexpr = BLOCK_N1 // NKP
    SBN2: tl.constexpr = BLOCK_N2 // NKP
    OUT_BN1: tl.constexpr = BLOCK_N1 // 2

    # ============================================================
    # PHASE 1: Stage1 GEMM (X @ W1 + SwiGLU + FP8 quant)
    # ============================================================
    # Map pid to (pid_m, pid_n) for phase1
    if pid < total_phase1_blocks:
        from triton.language import core as tlc
        pid_m = pid // grid_n1
        pid_n = pid % grid_n1

        expt_data = tl.load(ExptData + pid_m)
        if expt_data != -1:
            expt_id = expt_data & 0x0000FFFF
            block_id = expt_data >> 16
            M = tl.load(ExptHist + expt_id)
            start_m = tl.load(ExptOffs + expt_id)

            offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
            offs_m_wrap = offs_m % M

            # Gather input rows
            GatherIndx_off = GatherIndx + start_m
            offs_x_m = tl.load(GatherIndx_off + offs_m_wrap) // N_EXPTS_ACT
            offs_x_k = tl.arange(0, BLOCK_K1)
            mask_m = offs_m < M

            # W1 pointers
            offs_w_n = pid_n * BLOCK_N1 + tl.arange(0, BLOCK_N1)
            offs_w_k = tl.arange(0, PBK1)
            W1_e = W1 + expt_id.to(tl.int64) * stride_w1_e

            offs_w_n_scale = (pid_n * SBN1 + tl.arange(0, SBN1)) % N1
            offs_w_k_scale = tl.arange(0, PMB1)
            W1S_e = W1MxScale + expt_id.to(tl.int64) * stride_w1s_e
            W1SPtrs = (W1S_e
                       + offs_w_k_scale[None, :].to(tl.int64) * stride_w1s_k
                       + offs_w_n_scale[:, None].to(tl.int64) * stride_w1s_n)

            # GEMM
            acc = _gemm_core(
                X, stride_x_m, stride_x_k,
                W1_e, stride_w1_k, stride_w1_n,
                W1S_e, stride_w1s_k, stride_w1s_n,
                offs_x_m, offs_x_k,
                offs_w_k, offs_w_n, offs_w_k_scale, offs_w_n_scale,
                W1SPtrs,
                N1, K1,
                BLOCK_M, BLOCK_N1, BLOCK_K1,
                EVEN_K1, MASK_K1_LIMIT, 1,
                W_CACHE_MODIFIER,
                mask_m,
            )

            # Static scale compensation
            if X_static_scale is not None:
                acc = acc * tl.load(X_static_scale)

            # SwiGLU
            out = _swiglu(acc, alpha, limit)

            # FP8 quant for stage2
            if Quant_static_scale is not None:
                out = _compute_static_fp8_quant(out, tl.load(Quant_static_scale))

            # Write to intermediate buffer
            offs_inter_n = OUT_BN1 * pid_n + tl.arange(0, OUT_BN1)
            yN1 = N1 // 2
            mask_n = offs_inter_n < yN1
            Inter_ptrs = (Intermediate
                          + (start_m + offs_m).to(tl.int64)[:, None] * stride_inter_m
                          + offs_inter_n[None, :].to(tl.int64) * stride_inter_n)
            mask = mask_m[:, None] & mask_n[None, :]
            tl.store(Inter_ptrs, out, mask=mask)

    # ============================================================
    # GLOBAL BARRIER: wait for all phase1 blocks to finish
    # ============================================================
    # Each block signals completion, then spins until all are done
    if pid < total_phase1_blocks:
        tl.atomic_add(barrier, 1)
    tl.debug_barrier()
    # Spin-wait for all blocks
    while tl.atomic_add(barrier, 0) < total_phase1_blocks:
        pass

    # ============================================================
    # PHASE 2: Stage2 GEMM (Intermediate @ W2 + gammas + reduce)
    # ============================================================
    total_phase2_blocks = grid_m * grid_n2
    if pid < total_phase2_blocks:
        pid_m2 = pid // grid_n2
        pid_n2 = pid % grid_n2

        expt_data2 = tl.load(ExptData + pid_m2)
        if expt_data2 != -1:
            expt_id2 = expt_data2 & 0x0000FFFF
            block_id2 = expt_data2 >> 16
            M2 = tl.load(ExptHist + expt_id2)
            start_m2 = tl.load(ExptOffs + expt_id2)

            offs_m2 = BLOCK_M * block_id2 + tl.arange(0, BLOCK_M)
            offs_m2_wrap = offs_m2 % M2
            mask_m2 = offs_m2 < M2

            # A2 = intermediate (FP8), read in expert-sorted order
            offs_a2_m = start_m2 + offs_m2_wrap
            offs_a2_k = tl.arange(0, BLOCK_K2)

            # W2 pointers
            offs_w2_n = pid_n2 * BLOCK_N2 + tl.arange(0, BLOCK_N2)
            offs_w2_k = tl.arange(0, PBK2)
            W2_e = W2 + expt_id2.to(tl.int64) * stride_w2_e

            offs_w2_n_scale = (pid_n2 * SBN2 + tl.arange(0, SBN2)) % N2
            offs_w2_k_scale = tl.arange(0, PMB2)
            W2S_e = W2MxScale + expt_id2.to(tl.int64) * stride_w2s_e
            W2SPtrs = (W2S_e
                       + offs_w2_k_scale[None, :].to(tl.int64) * stride_w2s_k
                       + offs_w2_n_scale[:, None].to(tl.int64) * stride_w2s_n)

            # GEMM
            acc2 = _gemm_core(
                Intermediate, stride_inter_m, stride_inter_n,
                W2_e, stride_w2_k, stride_w2_n,
                W2S_e, stride_w2s_k, stride_w2s_n,
                offs_a2_m, offs_a2_k,
                offs_w2_k, offs_w2_n, offs_w2_k_scale, offs_w2_n_scale,
                W2SPtrs,
                N2, K2,
                BLOCK_M, BLOCK_N2, BLOCK_K2,
                EVEN_K2, MASK_K2_LIMIT, 1,
                W_CACHE_MODIFIER,
                mask_m2,
            )

            # A2 static scale
            if A2_static_scale is not None:
                acc2 = acc2 * tl.load(A2_static_scale)

            # Apply routing weights (gammas)
            if Gammas is not None:
                gammas = tl.load(Gammas + start_m2 + offs_m2, mask=mask_m2, other=0.0)
                acc2 *= gammas[:, None]

            # Write to output (expert-sorted order, will be reduced later)
            offs_out_n = BLOCK_N2 * pid_n2 + tl.arange(0, BLOCK_N2)
            mask_n2 = offs_out_n < N2
            Out_ptrs = (Out
                        + (start_m2 + offs_m2).to(tl.int64)[:, None] * stride_out_m
                        + offs_out_n[None, :].to(tl.int64) * stride_out_n)
            mask2 = mask_m2[:, None] & mask_n2[None, :]
            tl.store(Out_ptrs, acc2, mask=mask2)


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
    """Fused persistent MoE: single kernel launch for W1+SwiGLU+W2+reduce."""
    from aiter.ops.triton.moe_routing.routing import routing as aiter_routing
    from aiter.ops.triton.quant_moe import downcast_to_static_fp8
    from aiter.ops.triton.moe.moe_op_gemm_a8w4 import (
        moe_gemm_a8w4, reduce_grouped, get_kernel_config, allocate_output,
    )
    from aiter.ops.triton.moe_routing.routing import RoutingData

    assert quant_config is not None

    w1_data = w1.storage.data if hasattr(w1, 'storage') else w1
    w2_data = w2.storage.data if hasattr(w2, 'storage') else w2

    w1_scale = quant_config.w1_precision.weight_scale.storage.data
    w2_scale = quant_config.w2_precision.weight_scale.storage.data
    a1_scale = quant_config.w1_precision.flex_ctx.lhs_data.scale
    a2_scale = quant_config.w2_precision.flex_ctx.lhs_data.scale

    gammas_data = None

    # Routing (same as existing path)
    routing_data, gather_idx, scatter_idx = aiter_routing(
        gating_output, topk, sm_first=not renormalize)
    gammas_data = routing_data.gate_scal
    if apply_router_weight_on_input:
        gammas_stage2 = None
    else:
        gammas_stage2 = gammas_data

    # Downcast activations to FP8
    hidden_fp8 = downcast_to_static_fp8(hidden_states, a1_scale)

    M = gather_idx.shape[0]
    K1, N1 = hidden_states.shape[-1], w1_data.shape[-1]
    N2 = w2_data.shape[-1]
    inter_dim = N1 // 2
    K2 = inter_dim

    # Handle unpadded dimensions
    unpadded_N1 = getattr(quant_config, 'unpadded_N_w1', None) or N1
    unpadded_K1 = getattr(quant_config, 'unpadded_K_w1', None) or K1
    unpadded_N2 = getattr(quant_config, 'unpadded_N_w2', None) or N2
    unpadded_K2 = getattr(quant_config, 'unpadded_K_w2', None) or K2

    # Use existing config heuristics for tile sizes
    block_m = routing_data.block_m
    config1 = get_kernel_config(M, N1, K1, routing_data)
    config2 = get_kernel_config(M, N2, K2, routing_data)

    BLOCK_M = config1["block_m"]
    BLOCK_N1 = config1["block_n"]
    BLOCK_K1 = config1["block_k"]
    BLOCK_N2 = config2["block_n"]
    BLOCK_K2 = config2["block_k"]

    # Grid: max of phase1 and phase2 blocks
    grid_m = routing_data.n_blocks(M, BLOCK_M)
    grid_n1 = triton.cdiv(N1, BLOCK_N1)
    grid_n2 = triton.cdiv(N2, BLOCK_N2)
    total_phase1 = grid_m * grid_n1
    total_phase2 = grid_m * grid_n2
    total_blocks = max(total_phase1, total_phase2)

    # Allocate intermediate buffer (FP8)
    intermediate = torch.empty(
        (M, inter_dim), device=hidden_states.device, dtype=torch.float8_e4m3fn)

    # Allocate output (expert-sorted, before reduction)
    y_stage2 = torch.zeros(
        (1, M, N2), device=hidden_states.device, dtype=torch.bfloat16)
    y_final = torch.empty(
        (hidden_states.shape[0], N2), device=hidden_states.device,
        dtype=hidden_states.dtype)

    # Barrier counter
    barrier_counter = torch.zeros(1, device=hidden_states.device, dtype=torch.int32)

    # Expert metadata
    expt_data = routing_data.expt_data
    expt_hist = expt_data.hist if expt_data else None
    expt_offs = expt_data.token_offs_raw if expt_data else None
    expt_offs_sum = expt_data.token_offs_pad[-1] if expt_data else None
    expt_block_pid = expt_data.block_pid_map if expt_data else None

    _fused_moe_persistent[(total_blocks,)](
        # Phase 1
        hidden_fp8, hidden_fp8.stride(0), hidden_fp8.stride(1),
        w1_data, w1_data.stride(0), w1_data.stride(1), w1_data.stride(2),
        w1_scale, w1_scale.stride(0), w1_scale.stride(1), w1_scale.stride(2),
        a1_scale, a2_scale,
        # Phase 2
        w2_data, w2_data.stride(0), w2_data.stride(1), w2_data.stride(2),
        w2_scale, w2_scale.stride(0), w2_scale.stride(1), w2_scale.stride(2),
        a2_scale,
        # Routing
        gather_idx, gammas_stage2,
        expt_hist, expt_offs, expt_offs_sum, expt_block_pid,
        # Intermediate
        intermediate, intermediate.stride(0), intermediate.stride(1),
        # Output
        y_stage2, y_stage2.stride(1), y_stage2.stride(2),
        scatter_idx,
        # Barrier
        barrier_counter,
        # Dimensions
        N1, K1, N2, K2,
        total_phase1,
        grid_m, grid_n1, grid_n2,
        # Activation
        swiglu_alpha, swiglu_limit,
        # MoE config
        N_EXPTS_ACT=topk,
        # Tile config
        BLOCK_M=BLOCK_M,
        BLOCK_N1=BLOCK_N1, BLOCK_K1=BLOCK_K1,
        BLOCK_N2=BLOCK_N2, BLOCK_K2=BLOCK_K2,
        GROUP_M=config1["group_m"],
        EVEN_K1=(K1 % BLOCK_K1 == 0),
        MASK_K1_LIMIT=(K1 % BLOCK_K1),
        EVEN_K2=(K2 % BLOCK_K2 == 0),
        MASK_K2_LIMIT=(K2 % BLOCK_K2),
        W_CACHE_MODIFIER=config1["w_cache_modifier"],
        num_warps=config1["num_warps"],
        num_stages=config1["num_stages"],
    )

    # Reduce phase2 output (same as existing path)
    group_indx = scatter_idx.view(-1, topk)
    y_final = reduce_grouped(
        y_stage2, group_indx, y_final,
        False, swiglu_alpha, swiglu_limit, 1,
        out_dtype=hidden_states.dtype,
    )

    return y_final
