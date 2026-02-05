# MORI-EP Dispatch-Free Optimization Plan

## Overview

This document outlines a potential optimization for MORI-EP when used in combination with Tensor Parallelism (TP). The optimization eliminates the unnecessary dispatch All-to-All communication when all GPUs already have identical data.

**Status**: Proposed (Recommended Approach)  
**Author**: Markus Hartikainen (mahartik@amd.com)  
**Created**: 2026-02-05

---

## Why Dispatch-Free Over Partitioned Dispatch?

Two optimizations were considered for the TP+EP redundant dispatch problem:

| Approach | Dispatch | Combine | Extra Step | Savings | Complexity |
|----------|----------|---------|------------|---------|------------|
| Partitioned Dispatch | 1/8 (reduced) | 1/8 | + All-gather | ~81% | Medium |
| **Dispatch-Free** | **0 (eliminated)** | All-reduce | None | **~100%** | Medium |

**Dispatch-free is the recommended approach because:**
1. **Better performance**: Eliminates dispatch entirely vs. just reducing it
2. **Similar complexity**: Both require similar implementation effort
3. **Simpler model**: No MORI dispatch metadata to manage
4. **Cleaner design**: All-reduce is a well-understood primitive

```
Partitioned:   dispatch(1/8) → compute → combine(1/8) → all-gather
Dispatch-free: filter(local)  → compute → all-reduce
               ↑ no communication!        ↑ ~same cost as all-gather
```

---

## Problem Statement

### Current TP+EP Flow (Inefficient)

When combining TP attention with EP MoE:

1. **Attention (TP=8)** produces identical hidden states on all GPUs after all-reduce
2. **Router** runs identically on all GPUs → same topk selections
3. **MORI Dispatch** sends tokens to expert-owning GPUs
4. **AITER Compute** processes received tokens
5. **MORI Combine** returns results to original GPUs

**The Problem**: In step 3, dispatch moves data that doesn't need to move!

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CURRENT FLOW (WASTEFUL)                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  After TP all-reduce, ALL 8 GPUs have IDENTICAL data:                          │
│                                                                                 │
│    GPU 0: tokens [M, H], topk_ids [M, 8]   ─┐                                  │
│    GPU 1: tokens [M, H], topk_ids [M, 8]    │  ALL IDENTICAL!                  │
│    GPU 2: tokens [M, H], topk_ids [M, 8]    │                                  │
│    ...                                      │                                  │
│    GPU 7: tokens [M, H], topk_ids [M, 8]   ─┘                                  │
│                                                                                 │
│  DISPATCH: Each GPU sends its tokens to expert owners                          │
│    → Same token sent 8× from 8 different GPUs to same destination!             │
│    → This is redundant communication                                           │
│                                                                                 │
│  Example: Token T needs expert 50 (on GPU 1)                                   │
│    GPU 0 sends T to GPU 1  ─┐                                                  │
│    GPU 1 sends T to GPU 1   │  8 copies of same data!                          │
│    GPU 2 sends T to GPU 1   │  7 are unnecessary!                              │
│    ...                      │                                                  │
│    GPU 7 sends T to GPU 1  ─┘                                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Wasted Resources

- **Bandwidth**: Dispatch moves 8× more data than necessary
- **Latency**: All-to-All adds ~35µs per dispatch
- **Compute**: If deduplication isn't perfect, may compute same token multiple times

---

## Proposed Optimization

### Dispatch-Free Flow for TP+EP

Since all GPUs have identical data, each GPU already has all the tokens it needs. We can skip dispatch entirely:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       OPTIMIZED FLOW (DISPATCH-FREE)                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  After TP all-reduce, ALL 8 GPUs have IDENTICAL data:                          │
│                                                                                 │
│    GPU 0: tokens [M, H], topk_ids [M, 8]                                       │
│    GPU 1: tokens [M, H], topk_ids [M, 8]                                       │
│    ...                                                                         │
│                                                                                 │
│  STEP 1: LOCAL FILTERING (no communication!)                                   │
│    Each GPU identifies tokens needing its local experts:                       │
│                                                                                 │
│    GPU 0 (experts 0-31):   filter tokens where any topk_id in [0, 31]         │
│    GPU 1 (experts 32-63):  filter tokens where any topk_id in [32, 63]        │
│    ...                                                                         │
│                                                                                 │
│  STEP 2: LOCAL COMPUTE                                                         │
│    Each GPU runs AITER on filtered tokens with its local experts               │
│                                                                                 │
│  STEP 3: COMBINE ONLY (reduce-scatter or custom gather)                        │
│    Gather partial results from all GPUs                                        │
│    Sum contributions: output[i] = Σ expert_j(token_i) * weight_j              │
│                                                                                 │
│  RESULT: Same output, but NO dispatch All-to-All!                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Expected Benefits

| Metric | Current | Optimized | Improvement |
|--------|---------|-----------|-------------|
| Dispatch bandwidth | ~300 GB/s × M tokens | 0 | 100% savings |
| Dispatch latency | ~35µs | 0 | 100% savings |
| Combine bandwidth | ~330 GB/s × M tokens | Same | No change |
| Total comm time | dispatch + combine | combine only | ~45% reduction |

---

## Implementation Options

### Option A: MORI API Extension

Work with MORI team to add a "combine-only" mode:

```python
# Hypothetical new API
class EpDispatchCombineOp:
    def combine_only(
        self,
        expert_outputs: torch.Tensor,  # [N_local, H] from local computation
        token_indices: torch.Tensor,   # Which tokens these outputs correspond to
        topk_ids: torch.Tensor,        # Original routing for all tokens
        topk_weights: torch.Tensor,    # Weights for summation
    ) -> torch.Tensor:
        """Combine expert outputs without prior dispatch."""
        ...
```

**Pros**: Clean API, MORI handles all communication details  
**Cons**: Requires MORI library changes, coordination with ROCm team

### Option B: Custom Implementation (Bypass MORI Dispatch)

Implement dispatch-free path in vLLM directly:

```python
class MoriPrepareAndFinalizeOptimized(mk.FusedMoEPrepareAndFinalize):
    """Optimized EP for TP+EP case where all GPUs have identical data."""
    
    def prepare(self, a1, topk_weights, topk_ids, ...):
        # SKIP DISPATCH - all GPUs already have all tokens
        
        # 1. Filter to tokens needing local experts
        local_expert_mask = self._get_local_expert_mask(topk_ids)
        token_needs_local = local_expert_mask.any(dim=1)
        
        local_tokens = a1[token_needs_local]
        local_topk_ids = topk_ids[token_needs_local]
        local_weights = topk_weights[token_needs_local]
        
        # 2. Convert global → local expert IDs, mask non-local weights
        local_topk_ids, local_weights = self._convert_to_local(
            local_topk_ids, local_weights
        )
        
        # 3. Store metadata for combine
        self._token_indices = token_needs_local.nonzero().squeeze()
        
        return local_tokens, None, None, local_topk_ids, local_weights
    
    def finalize(self, output, expert_output, ...):
        # Use all-gather or reduce-scatter to combine results
        # Each GPU contributes its partial results for its local experts
        
        # Option 1: All-gather expert outputs, then local reduce
        all_expert_outputs = all_gather(expert_output)
        output = self._weighted_sum(all_expert_outputs, ...)
        
        # Option 2: Reduce-scatter (more efficient)
        # Requires careful indexing to scatter results correctly
```

**Pros**: No MORI changes needed, full control  
**Cons**: More complex implementation, must handle edge cases

### Option C: Hybrid Approach

Use MORI dispatch but with single-source optimization:

```python
def prepare(self, a1, topk_weights, topk_ids, ...):
    # Only rank 0 dispatches, others skip
    if self.ep_rank == 0:
        dispatch_result = self.ep_op.dispatch(...)
    else:
        # Don't dispatch, but still call dispatch with empty data
        # to maintain MORI internal state
        dispatch_result = self.ep_op.dispatch(empty_tensor, ...)
    
    # All ranks receive from rank 0's dispatch
    ...
```

**Pros**: Minimal changes, uses existing MORI combine  
**Cons**: Still some dispatch overhead, may need MORI support

---

## Implementation Plan

### Phase 1: Prototype & Benchmark

1. **Implement Option B** as a prototype in a separate class
2. Add feature flag: `VLLM_MORI_EP_DISPATCH_FREE=1`
3. Benchmark against current implementation:
   - Measure dispatch time savings
   - Verify correctness against reference
   - Test with various batch sizes

### Phase 2: Integration

1. Detect TP+EP case automatically (all GPUs have identical data)
2. Select optimized path when applicable
3. Fall back to standard dispatch for non-TP cases (EP-only, EP+DP)

### Phase 3: MORI API Discussion

1. Share findings with ROCm/MORI team
2. Propose API extension if beneficial
3. Potentially upstream the optimization

---

## Code Changes Required

### Files to Modify

1. **`mori_prepare_finalize.py`**
   - Add `MoriPrepareAndFinalizeDispatchFree` class
   - Or add dispatch-free path to existing class

2. **`mori_utils.py`**
   - Add detection for TP+EP identical-data case
   - Configuration for optimization

3. **`oracle/unquantized.py`**
   - Select optimized class when applicable

4. **`shared_fused_moe.py`**
   - Ensure shared expert handling still works

### New Files

1. **`mori_dispatch_free.py`** (if separate implementation)
   - Self-contained optimized implementation

---

## Testing Plan

### Correctness Tests

1. Compare output against standard MORI-EP path (bit-exact or within tolerance)
2. Test with various batch sizes: 1, 128, 1024, 8192
3. Test with different topk values: 1, 2, 8
4. Test edge cases: all tokens to one expert, uniform distribution

### Performance Tests

1. Measure end-to-end MoE layer time
2. Measure dispatch/combine time separately
3. Profile memory usage
4. Test at different sequence lengths

### Integration Tests

1. Full model inference (DeepSeek-R1)
2. Benchmark script validation
3. CUDA graph compatibility (if applicable)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| MORI combine depends on dispatch metadata | High | Prototype first to verify feasibility |
| Edge cases in token filtering | Medium | Comprehensive test suite |
| Performance regression in non-TP cases | Medium | Only enable for TP+EP, feature flag |
| CUDA graph incompatibility | Medium | Test early, may need separate path |

---

## Timeline Estimate

| Phase | Tasks | Estimate |
|-------|-------|----------|
| Phase 1 | Prototype, benchmark | 1-2 weeks |
| Phase 2 | Integration, testing | 1-2 weeks |
| Phase 3 | MORI discussion, upstream | 2-4 weeks |

---

## References

- [MORI Repository](https://github.com/ROCm/mori)
- [MORI-EP Code Walkthrough](./MORI_EP_CODE_WALKTHROUGH.md)
- [vLLM MoE Implementation](../vllm/model_executor/layers/fused_moe/)

---

## Appendix: Pseudocode for Dispatch-Free Path

```python
class MoriPrepareAndFinalizeDispatchFree(mk.FusedMoEPrepareAndFinalize):
    """
    Optimized MORI-EP for TP+EP case.
    
    When all GPUs have identical data (after TP all-reduce), we can skip
    the dispatch All-to-All entirely. Each GPU:
    1. Filters to tokens needing its local experts (no communication)
    2. Computes with AITER on filtered tokens
    3. Participates in combine to gather all results
    """
    
    def __init__(self, ...):
        super().__init__(...)
        self.is_tp_ep_mode = True  # Detected or configured
    
    def prepare(
        self,
        a1: torch.Tensor,           # [M, H] - identical on all GPUs
        topk_weights: torch.Tensor, # [M, K] - identical on all GPUs
        topk_ids: torch.Tensor,     # [M, K] - identical on all GPUs
        ...
    ) -> PrepareResultType:
        """Skip dispatch, just filter to local experts."""
        
        # 1. Identify tokens needing ANY local expert
        # topk_ids contains K expert IDs per token (global: 0-255)
        # Check if any selected expert is in our local range
        local_start = self.rank_expert_offset
        local_end = local_start + self.num_local_experts
        
        is_local_expert = (topk_ids >= local_start) & (topk_ids < local_end)
        token_needs_local = is_local_expert.any(dim=1)  # [M] bool
        
        # 2. Filter to relevant tokens only
        local_token_indices = token_needs_local.nonzero(as_tuple=True)[0]
        num_local_tokens = local_token_indices.shape[0]
        
        if num_local_tokens == 0:
            # No tokens need our experts - return empty
            return self._empty_result(a1)
        
        local_a1 = a1[local_token_indices]                    # [N_local, H]
        local_topk_ids = topk_ids[local_token_indices]        # [N_local, K]
        local_topk_weights = topk_weights[local_token_indices] # [N_local, K]
        
        # 3. Convert global → local expert IDs, zero non-local weights
        local_expert_ids = local_topk_ids - self.rank_expert_offset
        is_local = (local_expert_ids >= 0) & (local_expert_ids < self.num_local_experts)
        
        local_topk_weights = local_topk_weights * is_local.float()
        local_expert_ids = local_expert_ids.clamp(0, self.num_local_experts - 1)
        
        # 4. Store for combine phase
        self._local_token_indices = local_token_indices
        self._original_batch_size = a1.shape[0]
        
        # 5. Quantize if needed (same as before)
        if self.use_fp8_dispatch:
            local_a1, local_scale = self._quantize(local_a1)
        else:
            local_scale = None
        
        return (local_a1, local_scale, None, local_expert_ids, local_topk_weights)
    
    def finalize(
        self,
        output: torch.Tensor,           # [M, H] output buffer
        expert_output: torch.Tensor,    # [N_local, H] from AITER
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        ...
    ) -> None:
        """Combine results from all GPUs using all-reduce or custom gather."""
        
        # 1. Create buffer for this GPU's contribution
        # Shape: [M, H] with zeros, filled at local_token_indices
        local_contribution = torch.zeros_like(output)
        
        if expert_output.numel() > 0:
            local_contribution[self._local_token_indices] = expert_output
        
        # 2. All-reduce to sum contributions from all GPUs
        # Each GPU contributes results for its local experts
        # Sum gives: output[i] = Σ_j expert_j(token_i) * weight_j
        torch.distributed.all_reduce(
            local_contribution, 
            op=torch.distributed.ReduceOp.SUM,
            group=self.ep_group
        )
        
        # 3. Copy to output
        output.copy_(local_contribution)
```

---

## Open Questions

1. **Does MORI combine require dispatch metadata?**
   - Need to test if combine can work independently
   - May need MORI team input

2. **All-reduce vs reduce-scatter for combine?**
   - All-reduce: simpler, all GPUs get full result
   - Reduce-scatter: more bandwidth-efficient if only partial result needed

3. **How to handle non-TP cases?**
   - Need runtime detection of TP+EP vs pure EP
   - Fall back to standard dispatch for non-identical data

4. **CUDA graph compatibility?**
   - The local filtering may have variable output size
   - May need fixed-size buffers with masking
