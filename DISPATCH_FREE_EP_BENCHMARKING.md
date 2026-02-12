# Dispatch-Free MORI EP: Benchmarking & Optimization Guide

## Branch

```
Repository: https://github.com/maeehart/vllm
Branch:     feat/dispatch-free-mori
Base:       https://github.com/mpashkovskii/vllm (feat/dispatch-free-mori)
```

## What This Branch Does

In standard vLLM with TP8, every GPU holds all 256 experts but with
TP-sharded weights (intermediate_size / 8 = 256 per GPU). Each GPU
computes all 8 topk experts per token, touching all 256 expert weight
matrices. This thrashes the L2 cache at large batch sizes.

**Dispatch-free EP** changes this: each GPU owns 32 experts with full
(unsharded) weights (intermediate_size = 2048). Since all GPUs already
hold the full token batch after TP all-reduce, there is no need for
All-to-All dispatch. Each GPU:

1. Passes all tokens + original global topk_ids to AITER
2. AITER's `expert_mask` skips GEMMs for non-local experts
3. Each GPU computes only ~1 expert per token (effective topk=1/GPU)
4. All-reduce combines partial results across EP ranks

This reduces per-GPU MoE compute by ~8x and keeps the active weight
working set within L2 cache.

## Commits on This Branch

| Commit | Description |
|--------|-------------|
| `fa9c196ed` | fix: Enable dispatch-free EP path (fix config gating, make mori_op optional, fix import) |
| `9354ab850` | perf: Use TP group for EP all-reduce (CUDAGraph-safe custom all-reduce) |
| `70c5558f4` | perf: Let AITER expert_mask skip non-local expert GEMMs (8x compute reduction) |

### Files Modified

| File | What changed |
|------|-------------|
| `vllm/model_executor/layers/fused_moe/config.py` | Added `use_mori_dispatch_free` property (gates on `ep_size > 1`) |
| `vllm/model_executor/layers/fused_moe/all2all_utils.py` | Dispatch-free path created before `use_all2all_kernels` gate; uses TP group for all-reduce |
| `vllm/model_executor/layers/fused_moe/mori_prepare_finalize.py` | `_prepare_dispatch_free` is pure pass-through; `_finalize_dispatch_free` uses GroupCoordinator.all_reduce |
| `vllm/compilation/passes/fusion/rocm_aiter_fusion.py` | Fixed broken import (`activation_quant_fusion` -> `act_quant_fusion`) |
| `benchmarks/kernels/test_mori_ep_kernel_markus.py` | Benchmark script with `kernel` and `serve` sub-commands |
| `benchmarks/kernels/bench_mori_ep_serve_markus.sh` | Shell wrapper for serve benchmarks |

## Previous Benchmark Results

### Environment

- **Hardware**: 8x AMD Instinct MI300X (192GB HBM3 each)
- **Model**: DeepSeek-R1 (671B params, FP8 quantized, 256 experts, topk=8)
- **Software**: vLLM v0.1.dev (editable install), AITER MoE enabled
- **Workload**: 200 requests, ISL=512, OSL=128, request_rate=10

### Results BEFORE expert_mask optimization (commit `9354ab850`)

| Metric | TP8 baseline | TP8+EP8 dispatch-free | Delta |
|--------|-------------|----------------------|-------|
| Output throughput (tok/s) | 1,027 | 1,019 | -0.8% |
| Total throughput (tok/s) | 5,127 | 5,086 | -0.8% |
| Mean TPOT (ms) | 83.6 | 95.2 | +14% |
| Median TPOT (ms) | 94.0 | 101.4 | +8% |
| Mean TTFT (ms) | 523 | 693 | +32% |
| Request throughput (req/s) | 8.02 | 7.96 | -0.7% |

**Note**: These results do NOT yet include the expert_mask optimization
(commit `70c5558f4`). At this point, AITER was still computing all 8
topk GEMMs per token and zeroing the non-local results. The cache
thrashing benefit was not realized.

### Results AFTER expert_mask optimization (commit `70c5558f4`)

**Measured 2025-02-12** on the same hardware/workload as above (8x MI300X,
ISL=512, OSL=128, 200 requests, rate=10). Second runs (JIT caches warm).

| Metric | TP8 baseline | TP8+EP8 dispatch-free | Delta |
|--------|-------------|----------------------|-------|
| Output throughput (tok/s) | 998.86 | 1,019.52 | **+2.1%** |
| Total throughput (tok/s) | 4,986.51 | 5,089.63 | **+2.1%** |
| Request throughput (req/s) | 7.80 | 7.96 | **+2.1%** |
| Mean TTFT (ms) | 657.52 | 597.50 | **-9.1%** |
| Median TTFT (ms) | 520.43 | 393.91 | **-24.3%** |
| P99 TTFT (ms) | 2,830.65 | 3,173.57 | +12.1% |
| Mean TPOT (ms) | 100.60 | 87.32 | **-13.2%** |
| Median TPOT (ms) | 105.61 | 94.53 | **-10.5%** |
| P99 TPOT (ms) | 145.91 | 114.92 | **-21.2%** |
| Mean ITL (ms) | 100.60 | 87.32 | **-13.2%** |
| Median ITL (ms) | 49.40 | 46.20 | **-6.5%** |
| P99 ITL (ms) | 261.07 | 212.97 | **-18.4%** |
| Peak output throughput (tok/s) | 4,018 | 2,519 | -37.3% |

**Key observations:**

1. **Throughput**: +2.1% improvement across all throughput metrics.
2. **Decode latency (TPOT)**: 13.2% mean / 21.2% P99 improvement -- the
   biggest win, directly attributable to reduced MoE compute via
   `expert_mask`.
3. **Prefill latency (TTFT)**: Mean improved 9.1%, median improved 24.3%,
   but P99 regressed 12.1% (outlier prefills pay the all-reduce cost).
4. **Peak throughput regression**: -37.3% lower peak burst -- the EP
   all-reduce serializes at the peak.
5. **Compared to pre-expert_mask**: The optimization turned a -0.8%
   throughput / +14% TPOT regression into a +2.1% throughput / -13.2%
   TPOT improvement.

**What still needs investigation:**
- The improvements are modest given the theoretical 8x MoE compute
  reduction. Profiling is needed to understand where the remaining time
  is spent (all-reduce overhead, CUDAGraph overhead, non-MoE layers).
- Peak throughput regression suggests the all-reduce is a bottleneck
  under high concurrency.
- Larger batch / longer sequence benchmarks (ISL=4096) may show more
  benefit since cache thrashing is worse at higher batch sizes.

## How to Run Benchmarks

### Prerequisites

```bash
# Ensure AITER MoE is enabled
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1

# Verify the branch
cd /workspace/dev/vllm_matvei_ep
git branch --show-current  # should be feat/dispatch-free-mori
git log --oneline -5       # verify commits
```

### End-to-End Serving Benchmark (recommended)

The benchmark script handles starting/stopping servers automatically:

```bash
# Run both TP8 baseline and EP8 dispatch-free sequentially
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 \
python benchmarks/kernels/test_mori_ep_kernel_markus.py serve \
  --mode both \
  --model deepseek-ai/DeepSeek-R1 \
  --tp 8 \
  --max-model-len 4096 \
  --isl 512 --osl 128 \
  --num-prompts 200 \
  --request-rate 10

# Run only EP8 dispatch-free
python benchmarks/kernels/test_mori_ep_kernel_markus.py serve \
  --mode ep \
  --model deepseek-ai/DeepSeek-R1 \
  --tp 8 \
  --max-model-len 4096 \
  --isl 512 --osl 128 \
  --num-prompts 200 \
  --request-rate 10

# Run only TP8 baseline
python benchmarks/kernels/test_mori_ep_kernel_markus.py serve \
  --mode tp \
  --model deepseek-ai/DeepSeek-R1 \
  --tp 8 \
  --max-model-len 4096 \
  --isl 512 --osl 128 \
  --num-prompts 200 \
  --request-rate 10
```

### Longer sequences (closer to production workload)

```bash
# ISL=4096, OSL=300 (closer to DeepSeek R1 production)
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 \
python benchmarks/kernels/test_mori_ep_kernel_markus.py serve \
  --mode both \
  --model deepseek-ai/DeepSeek-R1 \
  --tp 8 \
  --max-model-len 8192 \
  --isl 4096 --osl 300 \
  --num-prompts 50 \
  --request-rate 5
```

### Micro-benchmark (kernel-level, no server)

```bash
# EP8 dispatch-free kernel benchmark (8 GPUs)
torchrun --nproc_per_node=8 \
  benchmarks/kernels/test_mori_ep_kernel_markus.py kernel

# AITER-only baseline (single GPU)
python benchmarks/kernels/test_mori_ep_kernel_markus.py kernel --no-ep
```

### Manual Server Launch (for debugging or profiling)

```bash
# TP8 baseline
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 \
vllm serve deepseek-ai/DeepSeek-R1 \
  --tensor-parallel-size 8 \
  --max-model-len 4096 \
  --trust-remote-code

# TP8+EP8 dispatch-free
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 \
vllm serve deepseek-ai/DeepSeek-R1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend mori \
  --max-model-len 4096 \
  --trust-remote-code

# Then benchmark against the running server
vllm bench serve \
  --base-url http://localhost:8000 \
  --model deepseek-ai/DeepSeek-R1 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 200 \
  --request-rate 10
```

## Profiling

### Dynamic profiling (recommended)

Start server with profiler config, let it finish CUDAGraph capture,
then use `--profile` flag on `vllm bench serve`:

```bash
# Start server with profiler config
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 \
VLLM_RPC_TIMEOUT=1800000 \
vllm serve deepseek-ai/DeepSeek-R1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend mori \
  --max-model-len 4096 \
  --trust-remote-code \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/tmp/vllm_profile", "torch_profiler_with_stack": true}'

# Wait for server to be ready, then run with --profile
vllm bench serve \
  --base-url http://localhost:8000 \
  --model deepseek-ai/DeepSeek-R1 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 64 \
  --num-prompts 5 \
  --request-rate inf \
  --profile
```

Traces are saved to `/tmp/vllm_profile/` and can be viewed at
https://ui.perfetto.dev/

### Analyzing profiler output

The `profiler_out_*.txt` files contain aggregated CUDA kernel timing.
The `.pt.trace.json.gz` files contain full traces per rank.

Key kernels to look for:
- `ck::kernel_moe_gemm` -- AITER MoE compute (should be smaller with EP)
- `ck_tile::MoeSortingKernel` -- AITER token sorting
- `cross_device_reduce` / `vllm::all_reduce` -- EP all-reduce
- `rocprim` -- sorting (from sampler, not MoE)

## Known Issues and Gotchas

### First run is slow (JIT compilation)

AITER JIT-compiles MoE ASM kernels on first use. This can take 5-10
minutes. Subsequent runs reuse the cache. The server startup timeout
in the benchmark script is 1200s to accommodate this.

### VLLM_ROCM_USE_AITER_MOE must be set

Without this, vLLM uses the Triton MoE backend which has a separate
compilation error with EP expert counts. Always set both:
```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
```

### Import fix required

The branch includes a fix for a broken import in
`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`
(`activation_quant_fusion` -> `act_quant_fusion`). Without this fix,
AITER MoE fails to load.

### CUDAGraphs work but with a caveat

The dispatch-free path is CUDAGraph-compatible (all ops are static-shape).
However, the EP all-reduce uses the TP group's custom all-reduce (not the
EP group's) because the EP group doesn't initialize `CustomAllreduce`
(it's gated on `"tp" in unique_name` in `cuda_communicator.py`).

### Disable dispatch-free for debugging

```bash
export VLLM_MORI_EP_DISPATCH_FREE=0
```

This falls back to the old behavior (would need standard MORI dispatch
which requires DP > 1).

## Architecture Overview

```
TP8 baseline:
  All GPUs have ALL 256 experts (TP-sharded: I=256)
  → topk=8 experts computed per token per GPU
  → TP all-reduce after MoE layer

EP8 dispatch-free:
  Each GPU has 32 experts (full weights: I=2048)
  → AITER expert_mask filters to ~1 local expert per token
  → EP all-reduce after MoE layer (via TP group)
  → No All-to-All dispatch needed
```

## What to Investigate Next

1. **Measure the expert_mask optimization** -- commit `70c5558f4` should
   show significant MoE compute reduction. Compare TPOT and throughput
   for various batch sizes.

2. **Large batch prefill** -- the cache thrashing benefit should be most
   visible at large batches (ISL=4096+). Run with `--isl 4096 --osl 300`.

3. **Profile after expert_mask** -- re-profile to verify that
   `ck::kernel_moe_gemm` time is reduced and `rocprim` sort time is
   proportionally smaller.

4. **Multi-node EP** -- the current implementation uses TP group
   all-reduce which only works when TP and EP groups have the same
   ranks (single-node). For multi-node, need to either enable
   `CustomAllreduce` for EP group or use NCCL directly.

5. **Proper EP group all-reduce** -- option 2 from our investigation:
   enable `CustomAllreduce` for the EP group in `cuda_communicator.py`.
   Initial attempt caused GPU coredumps (buffer conflicts with TP group).
   Needs investigation into separate P2P buffer allocation.
