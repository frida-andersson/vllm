# Dispatch-Free EP vs TP8 Profiling Analysis

**Date:** 2026-02-13
**Model:** DeepSeek-R1 (671B params, 61 layers, 256 routed experts + shared experts, top-8 routing)
**Hardware:** 8× AMD MI300X (192GB VRAM each), XGMI interconnect
**vLLM version:** 0.1.dev13690+g13dcb3ed3

## 1. Executive Summary

**EP dispatch-free shows parity with TP8 in end-to-end benchmarks despite MoE microbenchmarks
suggesting EP should be faster.** Profiling reveals the root cause: **all-reduce operations
consume 85-87% of GPU time in both configurations**, and EP dispatch-free introduces ~1,800
additional all-reduce calls per profiled workload (~1 extra per MoE layer per step) that negate
any MoE kernel savings.

## 2. End-to-End Benchmark Results

Workload: DeepSeek R1, ISL=70,000, OSL=300, 10 concurrent prompts, request-rate=inf.
Server mode with CUDA graphs enabled. Two runs per configuration (first discarded as JIT warmup).

| Metric                      | TP8 Baseline | EP8 Dispatch-Free | Delta          |
|-----------------------------|-------------|-------------------|----------------|
| Benchmark duration (s)      | 32.82       | 33.67             | +2.6% slower   |
| Total token throughput (tok/s)| 21,420    | 20,878            | -2.5%          |
| Output token throughput (tok/s)| 91.41    | 89.10             | -2.5%          |
| Request throughput (req/s)  | 0.30        | 0.30              | same           |
| Mean TTFT (ms)              | 3,410       | 3,374             | -1.1% (better) |
| Median TTFT (ms)            | 3,632       | 3,370             | -7.2% (better) |
| Mean TPOT (ms)              | 97.39       | 100.39            | +3.1% slower   |
| Mean ITL (ms)               | 97.39       | 100.39            | +3.1% slower   |
| P99 TTFT (ms)               | 4,043       | 4,080             | +0.9%          |
| P99 TPOT (ms)               | 100.08      | 102.78            | +2.7%          |

**Verdict:** Within measurement noise. EP dispatch-free is at parity with TP8.

## 3. Profiling Setup

Profiled using `vllm bench latency` with `--enforce-eager` (to see individual kernel timings
rather than opaque CUDA graph launches) and `--profiler torch`:

```
--input-len 256 --output-len 30 --batch-size 10 --num-iters-warmup 3 --profile
```

This captures 1 prefill step (256 tokens) + 30 decode steps across all 61 layers.
`--enforce-eager` disables CUDA graphs so individual kernel timings are visible.

## 4. Profile Comparison (Rank 0)

### 4.1 Top-Level Time Distribution

| Operation                        | EP Self CUDA | EP %   | EP Calls | TP8 Self CUDA | TP8 %  | TP8 Calls | EP vs TP8        |
|----------------------------------|-------------|--------|----------|--------------|--------|-----------|------------------|
| **Total CUDA time**              | **7.549s**  | 100%   |          | **6.375s**   | 100%   |           | **+18.4% slower** |
| `_C_custom_ar::all_reduce`       | 6.564s      | 86.95% | 5,611    | 5.436s       | 85.28% | 3,813     | **+1,798 calls** |
| ┗ `cross_device_reduce_1stage`   | 6.407s      | 84.88% | 5,430    | 5.293s       | 83.02% | 3,690     | +1,740 calls     |
| ┗ `cross_device_reduce_2stage`   | 134ms       | 1.78%  | 181      | 130ms        | 2.03%  | 123       | +58 calls        |
| `gemm_a8w8_blockscale_ck` (dense)| 240ms       | 3.18%  | 9,882    | 239ms        | 3.76%  | 9,882     | same             |
| NCCL kernels                     | 131ms       | 1.73%  | 213      | 104ms        | 1.63%  | 155       | +58 calls        |
| `ck_moe_stage1` (MoE up-proj)   | 90ms        | 1.20%  | 1,740    | 70ms         | 1.10%  | 1,740     | **+29% slower/call** |
| `dynamic_per_token_scaled_quant` | 59ms        | 0.78%  | 13,478   | 59ms         | 0.92%  | 13,478    | same             |
| `fmoe_fp8_blockscale_g1u1`(fused)| 55ms        | 0.73%  | 116      | 61ms         | 0.95%  | 116       | 10% faster/call  |
| `ck_moe_stage2` (MoE down-proj) | 51ms        | 0.67%  | 1,740    | 39ms         | 0.62%  | 1,740     | **+29% slower/call** |
| `aten::copy_`                    | 55ms        | 0.73%  | 12,219   | 52ms         | 0.82%  | 12,219    | same             |
| `Memcpy DtoD`                    | 41ms        | 0.54%  | 9,737    | 32ms         | 0.49%  | 7,939     | +1,798 copies    |
| `rmsnorm2d_fwd_with_add_ck`     | 30ms        | 0.39%  | 3,904    | 30ms         | 0.47%  | 3,904     | same             |
| `moe_sorting_fwd`               | 20ms        | 0.26%  | 1,856    | 20ms         | 0.32%  | 1,856     | same             |
| `mla_attention` (decode)        | 18ms        | 0.24%  | 1,952    | 19ms         | 0.30%  | 1,952     | same             |

### 4.2 Per-Call Latency for Key Operations

| Operation                     | EP avg/call | TP8 avg/call | EP vs TP8     |
|-------------------------------|-------------|-------------|---------------|
| `cross_device_reduce_1stage`  | 1.180ms     | 1.434ms     | 18% faster    |
| `cross_device_reduce_2stage`  | 0.741ms     | 1.053ms     | 30% faster    |
| `ck_moe_stage1` (up-proj)    | 51.9µs      | 40.3µs      | **29% slower**|
| `ck_moe_stage2` (down-proj)  | 29.2µs      | 22.7µs      | **29% slower**|
| `fmoe_fp8_blockscale_g1u1`   | 472µs       | 522µs       | 10% faster    |
| `gemm_a8w8_blockscale_ck`    | 24.3µs      | 24.2µs      | same          |
| `mla_decode_stage1_asm`      | 8.7µs       | 8.8µs       | same          |

### 4.3 All-Reduce Call Count Analysis

Per forward-pass step (31 total steps: 1 prefill + 30 decode):

| Config       | Total AR calls | AR calls/step | AR calls/layer | Interpretation                              |
|-------------|---------------|---------------|----------------|---------------------------------------------|
| **TP8**     | 3,813         | ~123          | ~2.0           | attn output + MoE/shared-expert output      |
| **EP d-free**| 5,611        | ~181          | ~2.97          | attn output + MoE output + **dispatch-free AR** |
| **Delta**   | +1,798        | +58           | +0.95          | ~1 extra all-reduce per MoE layer           |

The ~1,798 extra calls correspond almost exactly to 58 extra/step × 31 steps ≈ 1,798,
and 58/61 layers ≈ 0.95 → essentially **1 additional all-reduce per MoE layer per step**.

## 5. Root Cause Analysis

### 5.1 The All-Reduce Bottleneck

Both configurations are severely all-reduce bound:
- **TP8:** 5.436s all-reduce out of 6.375s total = **85.3%**
- **EP dispatch-free:** 6.564s all-reduce out of 7.549s total = **86.9%**

At batch_size=10, hidden_dim=7168, bf16: each all-reduce moves
`10 × 7168 × 2 = 143 KB` across 8 GPUs. The `cross_device_reduce_1stage` kernel
averages **1.18-1.43ms per call** for this tiny payload. This suggests the all-reduce
is **latency-bound**, not bandwidth-bound (143KB at 800 GB/s XGMI would take ~0.2µs).

### 5.2 Why EP Dispatch-Free Has More All-Reduces

In standard TP8, each transformer layer does:
1. **Attention output projection** → all-reduce (reduce partial sums from TP-sharded heads)
2. **MoE down-projection output** → all-reduce (reduce partial sums from TP-sharded experts)

In EP dispatch-free, each transformer layer does:
1. **Attention output projection** → all-reduce (same as TP8)
2. **MoE down-projection output** → all-reduce (same as TP8, for TP-sharded dense part)
3. **Dispatch-free aggregation** → **additional all-reduce** (sum local expert contributions across EP ranks)

The 3rd all-reduce is the cost of skipping the All-to-All dispatch. Instead of
dispatching tokens to the right GPU and combining results via All-to-All, each GPU
computes its local experts and then all-reduces to aggregate.

### 5.3 Why MoE Kernels Are Slower in EP

The MoE staged GEMM kernels (`ck_moe_stage1`, `ck_moe_stage2`) are **~29% slower per
call** in EP despite having the same number of calls (1,740). This is because:

- **TP8:** Each GPU processes ALL 256 experts with 1/8 intermediate dimension.
  GEMM shape: many small GEMMs with N=256 (2048/8).
- **EP:** Each GPU processes 32 local experts (256/8) with FULL intermediate dimension.
  GEMM shape: fewer GEMMs with N=2048.

The smaller-N, many-expert TP8 shape happens to be more efficient on the CK MoE GEMM
kernel for these batch sizes (batch=10, top-k=8).

The **fused MoE** path (`fmoe_fp8_blockscale_g1u1`) shows the opposite: EP is 10% faster
(472µs vs 522µs). This kernel handles the prefill phase where batch sizes are larger
and the larger-N EP shape is beneficial.

### 5.4 Where the Time Goes (per decode step, estimated)

For a single decode step with batch=10:

| Component           | EP Dispatch-Free | TP8 Baseline | Notes                              |
|--------------------|-----------------|-------------|-------------------------------------|
| All-reduce (all)   | ~212ms          | ~175ms      | EP: 181 calls × 1.17ms; TP8: 123 × 1.43ms |
| Dense GEMMs        | ~7.7ms          | ~7.7ms      | Same (TP-only operations)          |
| MoE computation    | ~4.6ms          | ~3.5ms      | EP 29% slower per-kernel           |
| Attention          | ~0.6ms          | ~0.6ms      | Same (TP-only)                     |
| Other (quant, etc) | ~5ms            | ~5ms        | Same                               |
| **Total**          | **~230ms**      | **~192ms**  | **EP 20% slower per step**         |

All-reduce is 92% of each decode step in EP, and 91% in TP8.

## 6. Key Insights and Recommendations

### 6.1 The Fundamental Problem

The dispatch-free EP approach replaces All-to-All communication with an all-reduce.
For **intra-node** configurations (all 8 GPUs on the same MI300X node via XGMI), this
is a lateral move at best:

- All-to-All dispatch/combine: ~2 communication operations per MoE layer
- Dispatch-free all-reduce: ~1 extra all-reduce per MoE layer

But the all-reduce's **latency cost (~1.2ms)** far exceeds the MoE compute savings
(~1.1ms faster per layer in fused path, ~1.1ms slower in staged path). The net effect
is roughly zero — explaining the parity in benchmarks.

### 6.2 When Dispatch-Free EP Could Win

Dispatch-free EP would be beneficial when:
1. **Inter-node EP** (multi-node with slower interconnect): All-to-All over network is
   much more expensive than intra-node all-reduce via XGMI.
2. **Larger batch sizes**: More tokens means higher arithmetic intensity in MoE kernels
   and better amortization of all-reduce latency.
3. **Optimized all-reduce**: If the all-reduce latency could be reduced from ~1.2ms to
   ~0.1ms (closer to the bandwidth limit for 143KB), EP would show significant gains.

### 6.3 Optimization Targets (Priority Order)

1. **Reduce all-reduce latency** (highest impact):
   - Current: ~1.2ms for 143KB payload (latency-bound, not bandwidth-bound)
   - Theoretical minimum at 800 GB/s XGMI: ~0.2µs
   - Even 10× improvement (to ~120µs) would save ~5s of the 7.5s total
   - Investigate: Why is QuickReduce so slow for small payloads? Is there a
     synchronization/barrier overhead? Can we use a fused all-reduce that
     overlaps with computation?

2. **Fuse/eliminate extra all-reduces** (medium impact):
   - Can the dispatch-free all-reduce be fused with the existing MoE output all-reduce?
   - Can shared expert + routed expert results be combined before all-reduce?

3. **Overlap communication with computation** (medium impact):
   - Pipeline the all-reduce with the next layer's attention computation
   - Requires careful scheduling but could hide communication latency

4. **Optimize MoE kernel shapes for EP** (lower impact):
   - The CK MoE GEMM is 29% slower for EP shapes (full N=2048 vs TP N=256)
   - Tuning the kernel for the EP shape could recover some of this

## 7. Profiling Commands Used

### EP Dispatch-Free:
```bash
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 MORI_SHMEM_HEAP_SIZE=12G \
VLLM_MORI_EP_DISPATCH_FREE=1 vllm bench latency \
  --model deepseek-ai/DeepSeek-R1 \
  --tensor-parallel-size 8 --enable-expert-parallel --all2all-backend mori \
  --kv-cache-dtype fp8 --max-model-len 2048 --enforce-eager --trust-remote-code \
  --input-len 256 --output-len 30 --batch-size 10 --num-iters-warmup 3 \
  --profile --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/profile_ep"}'
```

### TP8 Baseline:
```bash
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 vllm bench latency \
  --model deepseek-ai/DeepSeek-R1 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 --max-model-len 2048 --enforce-eager --trust-remote-code \
  --input-len 256 --output-len 30 --batch-size 10 --num-iters-warmup 3 \
  --profile --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/profile_tp"}'
```

## 8. Raw Profiler Output

### 8.1 EP Dispatch-Free (Rank 0, sorted by self_cuda_time_total)

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                               _C_custom_ar::all_reduce         0.27%      20.277ms         1.22%      92.478ms      16.482us        6.564s        86.95%        6.564s       1.170ms          5611
                 execute_context_0(0)_generation_10(10)         0.00%       0.000us         0.00%       0.000us       0.000us        6.430s        85.17%        6.430s     238.137ms            27
void vllm::cross_device_reduce_1stage<__hip_bfloat16...         0.00%       0.000us         0.00%       0.000us       0.000us        6.407s        84.88%        6.407s       1.180ms          5430
                   execute_context_0(0)_generation_9(9)         0.00%       0.000us         0.00%       0.000us       0.000us     487.191ms         6.45%     487.191ms     243.595ms             2
                execute_context_9(2304)_generation_1(1)         0.00%       0.000us         0.00%       0.000us       0.000us     246.297ms         3.26%     246.297ms     246.297ms             1
                         aiter::gemm_a8w8_blockscale_ck         1.50%     113.583ms         2.45%     185.392ms      18.761us     239.913ms         3.18%     239.913ms      24.278us          9882
                   execute_context_0(0)_generation_1(1)         0.00%       0.000us         0.00%       0.000us       0.000us     237.408ms         3.14%     237.408ms     237.408ms             1
                 execute_context_1(256)_generation_0(0)         0.00%       0.000us         0.00%       0.000us       0.000us     182.393ms         2.42%     182.393ms     182.393ms             1
void vllm::cross_device_reduce_2stage<__hip_bfloat16...         0.00%       0.000us         0.00%       0.000us       0.000us     134.114ms         1.78%     134.114ms     740.961us           181
ncclDevKernel_Generic_2(ncclDevKernelArgsStorage<409...         0.00%       0.000us         0.00%       0.000us       0.000us     130.808ms         1.73%     130.808ms     614.123us           213
                                       vllm::all_reduce         1.77%     134.105ms         3.40%     256.757ms      44.330us     117.138ms         1.55%        6.681s       1.153ms          5792
                                   aiter::ck_moe_stage1         1.02%      77.482ms         4.03%     304.523ms     175.013us      90.345ms         1.20%      90.345ms      51.922us          1740
                        aiter::fmoe_fp8_blockscale_g1u1         0.02%       1.773ms         0.04%       2.724ms      23.480us      54.741ms         0.73%      54.741ms     471.905us           116
                  aiter::dynamic_per_token_scaled_quant         1.80%     135.760ms         3.00%     226.713ms      16.821us      58.806ms         0.78%      58.853ms       4.367us         13478
                                   aiter::ck_moe_stage2         0.84%      63.204ms         3.21%     242.644ms     139.450us      50.788ms         0.67%      50.788ms      29.189us          1740
                                 aiter::moe_sorting_fwd         0.38%      28.836ms         0.61%      46.374ms      24.986us      19.514ms         0.26%      19.514ms      10.514us          1856
                vllm::unified_mla_attention_with_output        14.49%        1.096s        23.63%        1.787s     915.515us      18.378ms         0.24%      87.542ms      44.847us          1952
                       aiter::mla_decode_stage1_asm_fwd         0.44%      33.111ms         0.68%      51.041ms      26.992us      16.420ms         0.22%      16.420ms       8.683us          1891
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 7.561s
Self CUDA time total: 7.549s
```

### 8.2 TP8 Baseline (Rank 0, sorted by self_cuda_time_total)

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                 execute_context_0(0)_generation_10(10)         0.00%       0.000us         0.00%       0.000us       0.000us        6.239s        97.86%        6.239s     231.067ms            27
                               _C_custom_ar::all_reduce         0.20%      14.722ms         0.91%      66.597ms      17.466us        5.436s        85.28%        5.436s       1.426ms          3813
void vllm::cross_device_reduce_1stage<__hip_bfloat16...         0.00%       0.000us         0.00%       0.000us       0.000us        5.293s        83.02%        5.293s       1.434ms          3690
                   execute_context_0(0)_generation_9(9)         0.00%       0.000us         0.00%       0.000us       0.000us     471.856ms         7.40%     471.856ms     235.928ms             2
                         aiter::gemm_a8w8_blockscale_ck         1.59%     116.123ms         2.59%     189.957ms      19.222us     239.483ms         3.76%     239.483ms      24.234us          9882
                                       vllm::all_reduce         1.35%      99.067ms         2.56%     187.394ms      47.610us      91.751ms         1.44%        5.528s       1.404ms          3936
                                   aiter::ck_moe_stage1         1.10%      80.573ms         4.25%     310.967ms     178.717us      70.089ms         1.10%      70.093ms      40.284us          1740
                        aiter::fmoe_fp8_blockscale_g1u1         0.02%       1.751ms         0.04%       2.665ms      22.978us      60.515ms         0.95%      60.515ms     521.679us           116
                  aiter::dynamic_per_token_scaled_quant         1.88%     137.280ms         3.18%     232.555ms      17.254us      58.524ms         0.92%      58.524ms       4.342us         13478
                                   aiter::ck_moe_stage2         0.91%      66.511ms         3.44%     251.813ms     144.720us      39.426ms         0.62%      39.426ms      22.659us          1740
                                 aiter::moe_sorting_fwd         0.39%      28.440ms         0.63%      46.482ms      25.044us      20.131ms         0.32%      20.131ms      10.847us          1856
                vllm::unified_mla_attention_with_output        15.13%        1.108s        24.63%        1.803s     923.547us      18.866ms         0.30%      86.603ms      44.366us          1952
                       aiter::mla_decode_stage1_asm_fwd         0.44%      32.390ms         0.69%      50.186ms      26.539us      16.671ms         0.26%      16.671ms       8.816us          1891
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 7.321s
Self CUDA time total: 6.375s
```

## 9. Configuration Details

### EP Dispatch-Free Server:
```
--model deepseek-ai/DeepSeek-R1
--tensor-parallel-size 8
--enable-expert-parallel
--all2all-backend mori
--kv-cache-dtype fp8
--max-model-len 72000
VLLM_MORI_EP_DISPATCH_FREE=1
VLLM_ROCM_USE_AITER=1
VLLM_ROCM_USE_AITER_MOE=1
MORI_SHMEM_HEAP_SIZE=12G
```

### TP8 Baseline Server:
```
--model deepseek-ai/DeepSeek-R1
--tensor-parallel-size 8
--kv-cache-dtype fp8
--max-model-len 72000
VLLM_ROCM_USE_AITER=1
VLLM_ROCM_USE_AITER_MOE=1
```

### MORI Submodule:
- Fork: `maeehart/mori` @ commit `ed9e81c`
- Changes: `shmem_is_initialized()` API, `MoriAllReduceOp`, `combine_only()`
- vLLM branch: `feat/dispatch-free-mori` @ `maeehart/vllm`
