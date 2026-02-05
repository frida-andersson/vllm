# MORI-EP Code Walkthrough

## Overview

This document provides a comprehensive walkthrough of the **MORI-EP (Expert Parallelism)** backend implementation for vLLM on AMD MI300X GPUs. MORI-EP enables efficient Expert Parallelism for Mixture-of-Experts (MoE) models by providing optimized dispatch/combine primitives for All-to-All communication.

**Author**: Markus Hartikainen (mahartik@amd.com)  
**Target Hardware**: AMD MI300X (8-GPU nodes connected via XGMI)  
**Model Reference**: DeepSeek-R1 (256 experts, topk=8, 7168 hidden dim)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [Data Flow](#data-flow)
4. [File-by-File Walkthrough](#file-by-file-walkthrough)
5. [Configuration & Environment Variables](#configuration--environment-variables)
6. [Memory Management](#memory-management)
7. [Performance Characteristics](#performance-characteristics)
8. [Known Issues & Solutions](#known-issues--solutions)
9. [Testing & Debugging](#testing--debugging)

---

## Architecture Overview

### What is Expert Parallelism (EP)?

Expert Parallelism (EP) is a parallelization strategy for MoE models that distributes **experts** across GPUs, rather than sharding weight dimensions.

### Tensor Parallelism (TP) vs Expert Parallelism (EP)

Understanding the difference between TP and EP is crucial:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    TENSOR PARALLELISM (TP) - Traditional Approach               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Each GPU has ALL 256 experts, but weights are SHARDED along intermediate dim  │
│                                                                                 │
│  Expert FFN: hidden(7168) → intermediate(2048) → hidden(7168)                  │
│                                                                                 │
│  With TP=8, each GPU stores:                                                   │
│    W1: [256 experts, 2048/8=256, 7168]  ← intermediate dim sharded            │
│    W2: [256 experts, 7168, 2048/8=256]  ← intermediate dim sharded            │
│                                                                                 │
│  ✓ No token routing needed (all experts local)                                 │
│  ✗ All GPUs store all 256 experts (high memory per GPU)                        │
│  ✗ Requires all-reduce after W2 to combine partial results                     │
│                                                                                 │
│  GPU 0: [256 experts × 256 intermediate slice]                                 │
│  GPU 1: [256 experts × 256 intermediate slice]                                 │
│  ...                                                                           │
│  GPU 7: [256 experts × 256 intermediate slice]                                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                    EXPERT PARALLELISM (EP) - MORI Approach                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Each GPU has SUBSET of experts (32), but FULL intermediate dimension          │
│                                                                                 │
│  Expert FFN: hidden(7168) → intermediate(2048) → hidden(7168)                  │
│                                                                                 │
│  With EP=8, each GPU stores:                                                   │
│    W1: [32 experts, 2048, 7168]  ← full intermediate dim, fewer experts       │
│    W2: [32 experts, 7168, 2048]  ← full intermediate dim, fewer experts       │
│                                                                                 │
│  ✓ 8x memory reduction per GPU (32 vs 256 experts)                            │
│  ✓ No all-reduce needed (each expert computes full result)                    │
│  ✗ Requires All-to-All to route tokens to expert-owning GPUs                  │
│                                                                                 │
│  GPU 0: [experts 0-31   × full 2048 intermediate]                              │
│  GPU 1: [experts 32-63  × full 2048 intermediate]                              │
│  ...                                                                           │
│  GPU 7: [experts 224-255 × full 2048 intermediate]                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** 
- **TP shards the computation** (each GPU computes a partial result, then all-reduce)
- **EP shards the experts** (each GPU computes full results for its experts, then All-to-All routes tokens)

**Memory Comparison (DeepSeek-R1, BF16):**
| Parallelism | Experts/GPU | W1 shape | W2 shape | Memory/GPU |
|-------------|-------------|----------|----------|------------|
| TP=8 only   | 256 | [256, 256, 7168] | [256, 7168, 256] | ~1.9 GB |
| EP=8 only   | 32  | [32, 2048, 7168] | [32, 7168, 2048] | ~1.9 GB |

Note: Total memory similar, but EP enables larger intermediate dimensions or more experts.

**Why EP with MORI?**
- MORI provides optimized All-to-All primitives for the token routing
- XGMI on MI300X gives 800 GB/s aggregate bandwidth (low communication overhead)
- Enables scaling to more experts without memory explosion

### How TP (Attention) Connects to EP (MoE)

A transformer layer has both attention and MoE. In our setup:
- **Attention uses TP=8**: Q, K, V weights sharded across 8 GPUs
- **MoE uses EP=8**: Experts distributed across 8 GPUs

Here's how data flows through a transformer layer:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER LAYER: TP ATTENTION → EP MOE                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUT: [batch, seq, hidden=7168]                                              │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    ATTENTION (Tensor Parallel = 8)                      │   │
│  │                                                                         │   │
│  │  Each GPU has: Q,K,V sharded [hidden, hidden/8]                        │   │
│  │  Each GPU computes: partial attention output [batch, seq, hidden/8]    │   │
│  │                                                                         │   │
│  │                         ALL-REDUCE                                      │   │
│  │                            ↓                                            │   │
│  │  After all-reduce: ALL GPUs have IDENTICAL [batch, seq, hidden]        │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                       │
│         │  ← All 8 GPUs now have IDENTICAL hidden states                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    ROUTER (runs identically on all GPUs)                │   │
│  │                                                                         │   │
│  │  Input: [batch, seq, hidden] ← SAME on all GPUs                        │   │
│  │  Output: topk_ids [batch*seq, 8] ← SAME expert selections on all GPUs  │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                       │
│         │  ← All 8 GPUs have SAME tokens and SAME routing decisions            │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    MOE (Expert Parallel = 8)                            │   │
│  │                                                                         │   │
│  │  MORI DISPATCH: Route tokens to expert-owning GPUs                     │   │
│  │    - Each GPU sends its tokens to GPUs that own selected experts        │   │
│  │    - Because all GPUs have SAME tokens, same token sent 8× to same dest│   │
│  │                                                                         │   │
│  │  AITER COMPUTE: Each GPU processes received tokens                     │   │
│  │    - GPU 0 computes experts 0-31 on received tokens                    │   │
│  │    - GPU 1 computes experts 32-63 on received tokens                   │   │
│  │    - ...                                                                │   │
│  │                                                                         │   │
│  │  MORI COMBINE: Return results to original token owners                 │   │
│  │    - Each GPU receives partial results from all 8 expert-owning GPUs   │   │
│  │    - Results summed: output = Σ expert_i(x) * weight_i                 │   │
│  │                                                                         │   │
│  │  After combine: ALL GPUs have IDENTICAL [batch, seq, hidden]           │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                       │
│         ▼                                                                       │
│  OUTPUT: [batch, seq, hidden=7168]  ← IDENTICAL on all GPUs                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key Point: The "Replicated Input" Pattern**

Because attention uses TP with all-reduce:
1. All 8 GPUs end up with **IDENTICAL** hidden states after attention
2. All 8 GPUs run the **SAME** router computation → **SAME** expert selections
3. All 8 GPUs call MORI dispatch with **IDENTICAL** data

This means **the same token is dispatched 8 times** (once from each GPU) to the same destination!

**Why doesn't this break things?**
- MORI combine correctly routes results back to each source GPU
- Each GPU receives the same final result (since all started with same data)
- The replicated computation is "wasted" but mathematically correct

**Note on TP+EP Efficiency:**
In a pure EP setup (without TP attention), each GPU would have different tokens, 
and dispatch would be more efficient. The TP+EP combination means:
- 8× redundant dispatch (same token from 8 sources)
- Potentially 8× redundant expert computation for tokens with local experts
- This is a known trade-off when combining TP attention with EP MoE

### MORI-EP Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MORI-EP DISPATCH/COMBINE FLOW                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: [M tokens, H=7168]      Router: topk_ids [M, K=8]                  │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MORI DISPATCH (All-to-All)                       │   │
│  │  • Each GPU sends tokens to GPUs that own selected experts          │   │
│  │  • Optional FP8 quantization for 2x bandwidth savings               │   │
│  │  • XGMI transport (800 GB/s aggregate)                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    AITER EXPERT COMPUTATION                         │   │
│  │  • Each GPU processes only its LOCAL 32 experts                     │   │
│  │  • W1 (gate+up) → SiLU → W2 (down)                                  │   │
│  │  • AMD-optimized assembly kernels                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MORI COMBINE (All-to-All)                        │   │
│  │  • Return expert outputs to original token owners                   │   │
│  │  • Weighted summation across topk experts                           │   │
│  │  • BF16 output                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         ↓                                                                   │
│  OUTPUT: [M tokens, H=7168]    (reduced across K=8 experts)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. MoriPrepareAndFinalize (`mori_prepare_finalize.py`)

The main class implementing the dispatch/combine logic. Extends `mk.FusedMoEPrepareAndFinalize`.

**Key Methods:**
- `prepare_async()` / `prepare()`: Execute MORI dispatch
- `finalize_async()` / `finalize()`: Execute MORI combine
- `_receiver()`: Process dispatch results, handle expert ID mapping

### 2. MoriEpConfig & create_mori_ep_op (`mori_utils.py`)

Configuration and operator creation utilities.

**Key Features:**
- `MoriEpConfig`: Dataclass holding all EP configuration
- `create_mori_ep_op()`: Creates/caches MORI EP operators
- `_ensure_mori_shmem_initialized()`: Thread-safe shmem initialization

### 3. Oracle Integration (`oracle/unquantized.py`)

Backend selection and kernel creation.

**Key Additions:**
- `UnquantizedMoeBackend.AITER_MORI_EP`: New backend enum
- `make_unquantized_moe_kernel()`: Creates MoriPrepareAndFinalize + AiterExperts

### 4. SharedFusedMoE (`shared_fused_moe.py`)

Handles shared expert computation alongside routed experts.

**MORI-specific Handling:**
- Shared experts must be reduced separately (MORI combine only reduces routed)
- Detection via `all2all_backend == "mori_ep"`

---

## Data Flow

### The Key Transformation: Before and After `ep_op.dispatch`

This is the critical state change that MORI dispatch performs:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        BEFORE ep_op.dispatch()                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  STATE: Each GPU has its OWN tokens, needs to send them to expert owners       │
│                                                                                 │
│  GPU 0 (owns experts 0-31):                                                    │
│    tokens: [M, 7168]     ← M tokens that THIS GPU is responsible for           │
│    topk_ids: [M, 8]      ← each token selected 8 experts (global IDs 0-255)    │
│    weights: [M, 8]       ← router weights for each expert selection            │
│                                                                                 │
│  Example token on GPU 0:                                                       │
│    token[0] selected experts [3, 45, 67, 102, 150, 178, 201, 230]              │
│    → Expert 3 is LOCAL (0-31), experts 45,67,102,150,178,201,230 are REMOTE   │
│    → This token needs to be SENT to GPUs 1,2,3,4,5,6,7 for those experts      │
│                                                                                 │
│  WEIGHT STATE: Each GPU has 32 experts with FULL intermediate dimension        │
│    W1: [32, 2048, 7168]  ← only experts 0-31 on GPU 0                          │
│    W2: [32, 7168, 2048]  ← only experts 0-31 on GPU 0                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │  ep_op.dispatch()
                                      │  (All-to-All communication)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        AFTER ep_op.dispatch()                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  STATE: Each GPU has RECEIVED tokens that need ITS local experts               │
│                                                                                 │
│  GPU 0 (owns experts 0-31):                                                    │
│    recv_tokens: [N_recv, 7168]  ← tokens from ALL GPUs needing experts 0-31   │
│    recv_topk_ids: [N_recv, 8]   ← expert selections (still global IDs)        │
│    recv_weights: [N_recv, 8]    ← weights for each selection                  │
│                                                                                 │
│  N_recv ≠ M! GPU 0 receives tokens from GPUs 0,1,2,3,4,5,6,7 that need        │
│  experts 0-31. Could be more or fewer than M depending on routing.            │
│                                                                                 │
│  Example received token on GPU 0:                                              │
│    Originally from GPU 5, selected experts [3, 28, 67, 102, 150, 178, 201, 230]│
│    → Experts 3, 28 are LOCAL to GPU 0 → GPU 0 will compute these              │
│    → Other experts handled by other GPUs                                       │
│                                                                                 │
│  NOW GPU 0 CAN COMPUTE: Run AITER on recv_tokens using local experts 0-31     │
│    - Full intermediate dimension available (2048)                              │
│    - Only tokens that NEED our experts are here                                │
│    - No wasted computation on irrelevant experts                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Summary of the transformation:**

| Aspect | Before Dispatch | After Dispatch |
|--------|-----------------|----------------|
| Tokens | `[M, H]` - this GPU's own tokens | `[N_recv, H]` - tokens needing our experts |
| Token source | All from this GPU | From ALL GPUs in the cluster |
| Expert IDs | Global (0-255) | Still global (0-255), but filtered to our experts |
| What to compute | Unknown - tokens scattered across experts | Clear - all received tokens need our 32 experts |
| Weight sharding | Full intermediate dim, 32 experts | Same (unchanged by dispatch) |

**Why this matters:**
- Before dispatch: GPU 0 has tokens needing experts scattered across all 8 GPUs
- After dispatch: GPU 0 has ONLY tokens that need experts 0-31 (its local experts)
- Now AITER can compute efficiently using full intermediate dimension weights

### Dispatch Phase (prepare) - Code

```python
# 1. Input arrives
a1: [M, 7168]           # Token embeddings
topk_ids: [M, 8]        # Selected global expert IDs (0-255)
topk_weights: [M, 8]    # Router weights

# 2. Optional FP8 quantization (Strategy A)
if use_fp8_dispatch:
    a1q, a1q_scale = quantize(a1)  # 2x bandwidth savings

# 3. MORI dispatch - THE KEY TRANSFORMATION!
result = ep_op.dispatch(input=a1q, weights=topk_weights, 
                        scales=a1q_scale, indices=topk_ids)

# 4. Result unpacking (FIXED-SIZE BUFFERS!)
recv_x = result[0]              # [max_tokens, 7168]
recv_weights = result[1]        # [max_tokens, 8]
recv_scale = result[2]          # [max_tokens, scale_dim] 
recv_topk_ids = result[3]       # [max_tokens, 8]
total_recv_tokens = result[4]   # GPU scalar: actual count

# 5. CRITICAL: Slice to valid tokens only!
num_valid = total_recv_tokens.item()
expert_x = recv_x[:num_valid]
recv_weights = recv_weights[:num_valid]
recv_topk_ids = recv_topk_ids[:num_valid]
```

### Expert ID Mapping

MORI returns GLOBAL expert IDs (0-255). For AITER with expert_map=None, we convert to LOCAL IDs (0-31):

```python
# Global → Local conversion
local_topk_ids = global_topk_ids - rank_expert_offset

# Mask non-local experts (will have 0 weight)
is_local = (local_topk_ids >= 0) & (local_topk_ids < num_local_experts)
recv_weights = recv_weights * is_local.float()

# Clamp to valid range (non-local have 0 weight anyway)
expert_topk_ids = local_topk_ids.clamp(0, num_local_experts - 1)
```

### Combine Phase (finalize)

```python
# 1. After AITER computation
fused_expert_output: [N_recv, 7168]  # Expert outputs

# 2. Apply weight reduction (if needed)
fused_expert_output = weight_and_reduce_impl.apply(...)

# 3. MORI combine - use ORIGINAL topk_ids, not received!
combine_result = ep_op.combine(
    input=fused_expert_output,
    weights=None,  # AITER already applied weights
    indices=original_topk_ids.to(torch.int32),  # This rank's original tokens
    call_reset=True,
)

# 4. Slice to actual batch size
combined_x = combine_result[0]
output.copy_(combined_x[:num_tokens])
```

---

## File-by-File Walkthrough

### `mori_prepare_finalize.py` (686 lines)

**Purpose**: Main dispatch/combine implementation

**Key Sections:**

1. **Lines 1-76**: Imports and MORI availability check
   - Imports MORI ops: `EpDispatchCombineOp`, `EpDispatchCombineConfig`
   - DBO (Disaggregated Batched Operations) support for microbatching
   - Graceful fallback if MORI not installed

2. **Lines 77-163**: `MoriPrepareAndFinalize.__init__`
   - Stores EP operator, expert configuration
   - Initializes dispatch metadata storage for DBO
   - Logs configuration for debugging

3. **Lines 164-192**: Required properties
   - `output_is_reduced() → True`: MORI combine produces fully reduced output
   - `activation_format → Standard`: AITER uses [N, H] format
   - `supports_async() → True`: Enables compute/comm overlap

4. **Lines 193-252**: `_do_dispatch()`
   - Calls `ep_op.dispatch(input, weights, scales, indices)`
   - Stores metadata for combine phase
   - Returns lambda for async execution

5. **Lines 254-432**: `_receiver()`
   - **Critical section**: Unpacks dispatch results
   - Slices fixed-size buffers to valid tokens (CUDA graph limitation!)
   - Converts global → local expert IDs
   - Masks non-local expert weights to 0
   - Post-dispatch quantization (Strategy B)

6. **Lines 434-521**: `prepare_async()` / `prepare()`
   - Public API for dispatch
   - Handles quantization strategy selection (A vs B)
   - `expert_map` NOT used with MORI (MORI handles routing)

7. **Lines 522-686**: `_finalize_impl()` / `finalize_async()` / `finalize()`
   - Retrieves ORIGINAL topk_ids from dispatch metadata
   - Applies weight reduction
   - Calls `ep_op.combine()`
   - Slices combine output to actual batch size

### `mori_utils.py` (448 lines)

**Purpose**: Configuration and operator management

**Key Sections:**

1. **Lines 1-66**: Imports and module-level state
   - `_MORI_SHMEM_INITIALIZED`: Global flag for shmem init
   - `_MORI_EP_OP_CACHE`: Shared cache for EP operators

2. **Lines 68-120**: `MoriEpConfig` dataclass
   - All configuration parameters for MORI EP
   - Computed properties: `num_experts_per_rank`, `scale_dim`

3. **Lines 122-134**: `get_kernel_type()`
   - Maps string → MORI enum (`IntraNode`, `InterNode`, etc.)

4. **Lines 136-208**: `_ensure_mori_shmem_initialized()`
   - **Thread-safe** initialization using lock
   - Broadcasts unique ID from rank 0 (like NCCL)
   - Uses vLLM's world group (has CPU backend for object broadcast)

5. **Lines 210-340**: `create_mori_ep_op()`
   - **Caches operators** to avoid exhausting symmetric heap
   - Warns if `MORI_SHMEM_HEAP_SIZE` not set
   - Creates `EpDispatchCombineConfig` and `EpDispatchCombineOp`

6. **Lines 364-448**: Helper functions
   - `get_mori_ep_config_for_model()`: Extract config from vLLM model config
   - `compute_num_local_experts()`: 256 / 8 = 32
   - `compute_rank_expert_offset()`: rank * 32

### `oracle/unquantized.py` (234 lines)

**Purpose**: Backend selection and kernel creation

**Key Changes:**

1. **Line 32**: Added `AITER_MORI_EP` to `UnquantizedMoeBackend` enum

2. **Lines 167-225**: `make_unquantized_moe_kernel()` AITER_MORI_EP branch
   - Imports `MoriPrepareAndFinalize`, `MoriEpConfig`, etc.
   - Creates MORI EP operator via `create_mori_ep_op()`
   - Returns `FusedMoEModularKernel(MoriPrepareAndFinalize, AiterExperts)`

### `shared_fused_moe.py` (120 lines)

**Purpose**: Shared expert handling with MORI-EP

**Key Logic:**

```python
# MORI combine reduces ROUTED expert output, NOT shared expert output
# So we MUST reduce shared output separately when using MORI-EP
uses_mori_ep = (
    self.moe_parallel_config is not None
    and self.moe_parallel_config.all2all_backend == "mori_ep"
)
should_reduce = must_reduce or uses_mori_ep

if shared_out is not None and tp_size > 1 and should_reduce:
    shared_out = tensor_model_parallel_all_reduce(shared_out)
```

### `rocm_aiter_fused_moe.py` - AiterExperts class

**Key Addition (Lines 327-347)**: Dynamic activation scale support

```python
# For MORI-EP with FP8, activation scales are computed dynamically
# during dispatch. Override static scale with dynamic scale.
if a1q_scale is not None:
    quant_config = FusedMoEQuantConfig.make(
        ...
        a1_scale=a1q_scale,  # Use dynamic scale from MORI!
        ...
    )
```

---

## Configuration & Environment Variables

### Required Environment Variables

```bash
# CRITICAL: Set heap size before server start
export MORI_SHMEM_HEAP_SIZE=12G  # 12GB for EP8 with DeepSeek R1

# Optional: Enable FP8 dispatch for 2x bandwidth
export VLLM_MORI_EP_USE_FP8_DISPATCH=1

# Optional: Enable debug logging
export VLLM_MORI_DEBUG=1
```

### vLLM Configuration

```python
# Enable via MoE parallel config
moe_parallel_config = FusedMoEParallelConfig(
    all2all_backend="mori_ep",  # Select MORI-EP backend
    ep_size=8,
    tp_size=8,
    ...
)
```

### Heap Size Calculation

```
Memory per rank ≈ max_tokens × hidden_dim × 16 bytes
For DeepSeek R1 (max_tokens=8192, hidden=7168):
  = 8192 × 7168 × 16 = 940 MB per rank
  
Total for EP8 = 940 × 8 = 7.5 GB minimum
Recommended: 12GB (MORI_SHMEM_HEAP_SIZE=12G)
```

---

## Memory Management

### Operator Caching

MORI EP operators are **cached and shared** across all MoE layers:

```python
# In mori_utils.py
_MORI_EP_OP_CACHE: dict[tuple, EpDispatchCombineOp] = {}

def create_mori_ep_op(config):
    cache_key = _make_cache_key(config)
    if cache_key in _MORI_EP_OP_CACHE:
        return _MORI_EP_OP_CACHE[cache_key]  # Reuse!
    
    op = EpDispatchCombineOp(mori_config)
    _MORI_EP_OP_CACHE[cache_key] = op
    return op
```

**Why?** Each operator allocates symmetric heap memory. Without caching:
- 61 MoE layers × 1 operator = 61 operators
- Would exhaust symmetric heap

### Fixed-Size Buffers

MORI returns **fixed-size buffers** [max_tokens, hidden]:

```python
# MORI dispatch returns:
recv_x: [max_tokens, 7168]  # e.g., [8192, 7168]
recv_weights: [max_tokens, 8]
recv_topk_ids: [max_tokens, 8]
total_recv_tokens: scalar (actual valid count)

# CRITICAL: Only positions 0..total_recv_tokens-1 are valid!
# Solution: Slice buffers
num_valid = total_recv_tokens.item()  # NOTE: Breaks CUDA graph!
expert_x = recv_x[:num_valid]
```

---

## Performance Characteristics

### Benchmark Results (8× MI300X, XGMI)

| Operation | Tokens | Latency | Bandwidth |
|-----------|--------|---------|-----------|
| Dispatch | 128 | ~35 µs | 307 GB/s |
| Dispatch | 4096 | ~180 µs | scales linearly |
| Combine | 128 | ~47 µs | 330 GB/s |
| Combine | 4096 | ~220 µs | scales linearly |

### Communication vs Compute

At high batch sizes:
- Expert compute dominates (~85-90% of MoE layer time)
- Communication overhead ~10-15%

### FP8 Dispatch Benefit

With `VLLM_MORI_EP_USE_FP8_DISPATCH=1`:
- 2x bandwidth savings during dispatch
- Quantization adds small overhead
- Net benefit at high token counts

---

## Known Issues & Solutions

### Issue 1: CUDA Graph Capture

**Problem**: `total_recv_tokens.item()` transfers GPU→CPU, breaking CUDA graph capture.

**Current Solution**: Disabled for CUDA graph mode. Need to investigate alternatives.

**Potential Fix**: Use GPU tensor comparison instead of `.item()`:
```python
# Mask instead of slice (CUDA-graph safe)
valid_mask = torch.arange(max_tokens, device=device) < total_recv_tokens
expert_x = expert_x * valid_mask.unsqueeze(-1)
```

### Issue 2: TP+EP Deduplication

**Problem**: With TP=8, all ranks have identical tokens. Same token dispatched 8× to same expert-owning rank.

**Previous Attempt**: Deduplication by local token index.

**Current State**: Deduplication **disabled** (was causing output corruption). MORI may handle this internally. TODO: Investigate if there's wasted compute.

### Issue 3: Symmetric Heap Exhaustion

**Problem**: "Out of symmetric heap memory" errors.

**Solution**: 
1. Set `MORI_SHMEM_HEAP_SIZE=12G` (or higher)
2. EP operators are cached to avoid multiple allocations

---

## Testing & Debugging

### Running the Benchmark

```bash
# Set environment
export MORI_SHMEM_HEAP_SIZE=12G
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import torch; print(torch.__path__[0])")/lib

# Run benchmark
./scripts/bench_mori_ep_serve_markus.sh
```

### Debug Logging

```bash
# Enable verbose MORI logging
export VLLM_MORI_DEBUG=1

# Run server
vllm serve deepseek-ai/DeepSeek-R1 --tensor-parallel-size 8
```

### Key Debug Points

1. **Dispatch output**: Check `recv_x.shape`, `total_recv_tokens`
2. **Expert mapping**: Verify `local_topk_ids` in valid range [0, 31]
3. **Combine input**: Check `fused_expert_output.shape` matches expected
4. **Output values**: Compare against reference implementation

---

## Commit History Summary

| Commit | Description |
|--------|-------------|
| `512f62a8c` | Initial MORI-EP dispatch/combine implementation |
| `60254f63d` | Integrate MORI-EP backend into vLLM oracle |
| `4cf9ce742` | Add benchmark script |
| `2c21c6d25` | Cache and share EP operator across MoE layers |
| `f6ef570db` | Fix AITER integration: global→local expert IDs |
| `2c92a5b06` | Pass dynamic activation scales to AITER |
| `cc411b876` | Remove buggy dedup logic |
| `2d403089a` | Enable torch.compile, cleanup debug logging |

---

## Future Work

1. **CUDA Graph Support**: Investigate GPU-only slicing approach
2. **TP+EP Deduplication**: Profile to understand actual overhead
3. **Multi-Node**: Test InterNode kernel type for RDMA transport
4. **FP8 Combine**: Currently BF16 only; investigate FP8 output

---

## References

- [MORI Repository](https://github.com/ROCm/mori)
- [vLLM MoE Documentation](https://docs.vllm.ai/)
- [DeepSeek-R1 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1)
