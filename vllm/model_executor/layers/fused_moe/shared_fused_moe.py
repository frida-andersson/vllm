# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


# TODO(bnell): Add shared + fused combo function? e.g. +
class SharedFusedMoE(FusedMoE):
    """
    A FusedMoE operation that also computes shared experts alongside routed experts.

    DeepSeek-style MoE models have TWO types of experts:
    1. ROUTED experts: Selected by router (topk per token)
    2. SHARED experts: Applied to ALL tokens (always active)

    This class handles both, with optional compute/communication overlap:
    - Shared expert computation can run DURING MORI dispatch (overlap)
    - Or run sequentially if overlap is disabled

    MORI-EP Specific Handling:
    ==========================
    When using MORI-EP (all2all_backend="mori_ep"), special care is needed:

    - MORI combine ONLY reduces ROUTED expert outputs
    - Shared expert outputs are NOT reduced by MORI
    - We MUST explicitly reduce shared outputs with tensor_model_parallel_all_reduce()

    Without this explicit reduction:
    - Routed output: correctly reduced by MORI combine
    - Shared output: NOT reduced → incorrect final result

    The fix (in forward()):
        uses_mori_ep = self.moe_parallel_config.all2all_backend == "mori_ep"
        if tp_size > 1 and (must_reduce or uses_mori_ep):
            shared_out = tensor_model_parallel_all_reduce(shared_out)

    Forward Output:
    ===============
    Returns tuple: (shared_expert_output, routed_expert_output)
    Caller is responsible for combining these (typically: shared + routed).
    """

    def __init__(
        self,
        shared_experts: torch.nn.Module | None,
        gate: torch.nn.Module | None = None,
        use_overlapped: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._shared_experts = shared_experts

        # Disable shared expert overlap if:
        #   - we are using eplb with non-default backend, because of correctness issues
        #   - we are using flashinfer with DP, since there nothint to gain
        #   - we are using marlin kernels
        backend = self.moe_parallel_config.all2all_backend
        self.use_overlapped = (
            use_overlapped
            and not (
                (self.enable_eplb and backend != "allgather_reducescatter")
                or (self.moe_config.use_flashinfer_cutlass_kernels and self.dp_size > 1)
            )
            and self._shared_experts is not None
        )

        self._gate = gate

    @property
    def shared_experts(self) -> torch.nn.Module | None:
        return self._shared_experts if self.use_overlapped else None

    @property
    def gate(self) -> torch.nn.Module | None:
        return self._gate if self.use_overlapped else None

    @property
    def is_internal_router(self) -> bool:
        return self.gate is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing both shared and routed expert outputs.

        Args:
            hidden_states: [batch, hidden] input tensor
            router_logits: [batch, num_experts] router scores

        Returns:
            tuple: (shared_expert_output, routed_expert_output)
                Both tensors have shape [batch, hidden].
                Caller combines these (typically: shared + routed).

        MORI-EP Reduction Logic:
        ========================
        With MORI-EP, the combine phase ONLY reduces routed expert outputs.
        Shared experts are computed locally and need explicit TP reduction.

        Flow diagram:
            Routed path:  dispatch → compute → combine (MORI reduces)
            Shared path:  compute locally → manual all_reduce (WE reduce)

        Without the manual reduction of shared output, the final result
        would be incorrect (routed correct, shared not reduced).
        """
        if not self.use_overlapped:
            # Sequential mode: compute shared experts, then routed experts
            if self._shared_experts is not None:
                shared_out = self._shared_experts(hidden_states)

                # ============================================================
                # CRITICAL: Reduce shared expert outputs when using MORI-EP
                # ============================================================
                # The shared expert MLP was created with reduce_results=False
                # to allow the main MoE layer to handle reduction.
                #
                # However, with MORI-EP:
                # - MORI combine reduces ROUTED expert outputs automatically
                # - Shared expert outputs are NOT touched by MORI
                # - We MUST explicitly reduce shared outputs here
                #
                # Without this: routed output correct, shared output 1/tp_size!
                # ============================================================
                tp_size = get_tensor_model_parallel_world_size()
                must_reduce = self.must_reduce_shared_expert_outputs()
                uses_mori_ep = (
                    self.moe_parallel_config is not None
                    and self.moe_parallel_config.all2all_backend == "mori_ep"
                )
                should_reduce = must_reduce or uses_mori_ep
                if tp_size > 1 and should_reduce:
                    shared_out = tensor_model_parallel_all_reduce(shared_out)
            else:
                shared_out = None

            # Compute routed experts (may use MORI dispatch/combine internally)
            fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        else:
            # Overlapped mode: shared experts computed during MORI dispatch
            # Parent class handles the overlap scheduling
            shared_out, fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

            # ============================================================
            # CRITICAL: Reduce shared expert outputs when using MORI-EP
            # ============================================================
            # Same logic as sequential mode - MORI combine only reduces
            # routed outputs, so we must manually reduce shared outputs.
            #
            # This runs AFTER the overlapped computation completes, so
            # we get the benefits of overlap AND correct reduction.
            # ============================================================
            tp_size = get_tensor_model_parallel_world_size()
            
            # Check if we need to reduce shared output:
            # 1. Standard check via must_reduce_shared_expert_outputs()
            # 2. OR if MORI-EP is used (which reduces routed output but not shared)
            must_reduce = self.must_reduce_shared_expert_outputs()
            uses_mori_ep = (
                self.moe_parallel_config is not None
                and self.moe_parallel_config.all2all_backend == "mori_ep"
            )
            should_reduce = must_reduce or uses_mori_ep
            
            if (
                shared_out is not None
                and tp_size > 1
                and should_reduce
            ):
                shared_out = tensor_model_parallel_all_reduce(shared_out)

        return shared_out, fused_out
