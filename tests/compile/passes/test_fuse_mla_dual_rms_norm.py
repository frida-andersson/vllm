# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit test for the MLADualRMSNormFusionPass.

The pass fuses paired q/kv RMS norms in MLA attention into a single
fused_mla_dual_rms_norm op backed by AITER's fused_qk_rmsnorm kernel.
"""

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from vllm._aiter_ops import (
    check_aiter_fused_qk_rmsnorm_group_quant,
    is_aiter_found_and_supported,
    rocm_aiter_ops,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform

# MLA attention geometry for DeepSeek-V3 / Kimi-K2
Q_DIM = 1536
KV_C_DIM = 512
K_PE_DIM = 64
EPS = 1e-6
GROUP_SIZE = 128

FP8_DTYPE = current_platform.fp8_dtype()


def _group_dequant(
    x_fp8: torch.Tensor, scale: torch.Tensor, group_size: int = GROUP_SIZE
) -> torch.Tensor:
    """Dequantize an FP8 group-quantized tensor with ``(M, N//g)`` scales."""
    m, n = x_fp8.shape
    n_groups = scale.shape[1]
    x = x_fp8.to(torch.float32).reshape(m, n_groups, group_size)
    x = x * scale.reshape(m, n_groups, 1)
    return x.reshape(m, n).to(torch.bfloat16)


class MLADualRMSNormTestModel(torch.nn.Module):
    """
    Minimal model reproducing the MLA dual RMS norm pattern:
        linear -> split([q_dim, kv_dim])
            +-- q_c (getitem 0) -> rms_norm(q_w, eps) -> linear
            +-- kv_lora (getitem 1) -> split([kv_c_dim, k_pe_dim])
                    +-- kv_c (getitem 0) -> rms_norm(kv_w, eps)
                    +-- k_pe
    """

    def __init__(
        self,
        hidden_size: int,
        q_dim: int = Q_DIM,
        kv_c_dim: int = KV_C_DIM,
        k_pe_dim: int = K_PE_DIM,
        eps: float = EPS,
    ):
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_c_dim + k_pe_dim
        self.kv_c_dim = kv_c_dim
        self.k_pe_dim = k_pe_dim

        self.proj = torch.nn.Linear(hidden_size, q_dim + self.kv_dim, bias=False)
        self.q_norm = RMSNorm(q_dim, eps=eps)
        self.kv_norm = RMSNorm(kv_c_dim, eps=eps)
        self.q_b_proj = torch.nn.Linear(q_dim, hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Avoid graph input being a direct arg to a matched pattern node
        x = torch.relu(x)

        projected = self.proj(x)

        q_c, kv_lora = projected.split([self.q_dim, self.kv_dim], dim=-1)
        kv_c, k_pe = kv_lora.split([self.kv_c_dim, self.k_pe_dim], dim=-1)

        q_normed = self.q_norm(q_c)
        kv_normed = self.kv_norm(kv_c)

        q_out = self.q_b_proj(q_normed)
        return q_out, kv_normed, k_pe

    def ops_in_model_before(self):
        return [torch.ops.vllm_ir.rms_norm.default]

    def ops_in_model_after(self):
        return [torch.ops.vllm.fused_mla_dual_rms_norm.default]


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
def test_fuse_mla_dual_rms_norm(
    dtype: torch.dtype,
    hidden_size: int,
    monkeypatch: pytest.MonkeyPatch,
):
    torch._dynamo.reset()

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm"],
            pass_config=PassConfig(
                fuse_mla_dual_rms_norm=True,
                eliminate_noops=True,
            ),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
            MLADualRMSNormFusionPass,
        )

        torch.set_default_device("cuda")
        torch.set_default_dtype(dtype)
        torch.manual_seed(42)

        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()

        fusion_pass = MLADualRMSNormFusionPass(vllm_config)
        passes = [
            NoOpEliminationPass(vllm_config),
            fusion_pass,
            PostCleanupPass(vllm_config),
        ]
        backend = TestBackend(*passes)
        model = MLADualRMSNormTestModel(hidden_size)

        x = torch.randn(1, hidden_size)
        torch._dynamo.mark_dynamic(x, 0)

        outputs_unfused = model(x)

        model_fused = torch.compile(model, backend=backend)
        outputs_fused = model_fused(x)

        torch.testing.assert_close(outputs_unfused, outputs_fused, atol=1e-2, rtol=1e-2)

        assert fusion_pass.matched_count == 1, (
            f"Expected 1 fused pair, got {fusion_pass.matched_count}"
        )

        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())


class MLADualRMSNormFp8GroupTestModel(torch.nn.Module):
    """
    MLA FP8 attention path with *group* (block-scale) quant, single q consumer:
        linear -> split([q_dim, kv_dim])
            +-- q_c -> rocm_aiter_rmsnorm_fp8_group_quant -> dequant
            +-- kv_lora -> split([kv_c_dim, k_pe_dim])
                    +-- kv_c -> rms_norm (bf16)
                    +-- k_pe
    """

    def __init__(
        self,
        hidden_size: int,
        q_dim: int = Q_DIM,
        kv_c_dim: int = KV_C_DIM,
        k_pe_dim: int = K_PE_DIM,
        eps: float = EPS,
    ):
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_c_dim + k_pe_dim
        self.kv_c_dim = kv_c_dim
        self.k_pe_dim = k_pe_dim
        self.eps = eps

        self.proj = torch.nn.Linear(hidden_size, q_dim + self.kv_dim, bias=False)
        self.q_weight = torch.nn.Parameter(torch.ones(q_dim))
        self.kv_norm = RMSNorm(kv_c_dim, eps=eps)

    def forward(self, x: torch.Tensor):
        x = torch.relu(x)
        projected = self.proj(x)

        q_c, kv_lora = projected.split([self.q_dim, self.kv_dim], dim=-1)
        kv_c, k_pe = kv_lora.split([self.kv_c_dim, self.k_pe_dim], dim=-1)

        q_fp8, q_scale = torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant(
            q_c, self.q_weight, self.eps, GROUP_SIZE
        )
        kv_normed = self.kv_norm(kv_c)

        return _group_dequant(q_fp8, q_scale), kv_normed, k_pe

    def ops_in_model_before(self):
        return [
            torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant.default,
            torch.ops.vllm_ir.rms_norm.default,
        ]

    def ops_in_model_after(self):
        return [torch.ops.vllm.fused_mla_dual_rms_norm_group_quant.default]


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.skipif(
    not is_aiter_found_and_supported()
    or not check_aiter_fused_qk_rmsnorm_group_quant(),
    reason=(
        "Only test on ROCm with AITER (incl. fused_qk_rmsnorm_group_quant) "
        "installed and supported"
    ),
)
def test_fuse_mla_dual_rms_norm_fp8_group(
    dtype: torch.dtype,
    hidden_size: int,
    monkeypatch: pytest.MonkeyPatch,
):
    torch._dynamo.reset()

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm"],
            pass_config=PassConfig(
                fuse_mla_dual_rms_norm=True,
                eliminate_noops=True,
            ),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
            MLADualRMSNormFusionPass,
        )

        torch.set_default_device("cuda")
        torch.set_default_dtype(dtype)
        torch.manual_seed(42)

        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()

        fusion_pass = MLADualRMSNormFusionPass(vllm_config)
        passes = [
            NoOpEliminationPass(vllm_config),
            fusion_pass,
            PostCleanupPass(vllm_config),
        ]
        backend = TestBackend(*passes)
        model = MLADualRMSNormFp8GroupTestModel(hidden_size)

        x = torch.randn(4, hidden_size)
        torch._dynamo.mark_dynamic(x, 0)

        with torch.inference_mode():
            outputs_unfused = model(x)

            model_fused = torch.compile(model, backend=backend)
            outputs_fused = model_fused(x)

        q1_u, kv_u, k_pe_u = outputs_unfused
        q1_f, kv_f, k_pe_f = outputs_fused
        q_pairs = [(q1_u, q1_f)]

        torch.testing.assert_close(k_pe_u, k_pe_f, atol=0, rtol=0)
        torch.testing.assert_close(kv_u, kv_f, atol=1e-2, rtol=1e-2)

        E4M3_STEP = 0.125
        for q_u, q_f in q_pairs:
            exact_frac = (q_u == q_f).float().mean().item()
            assert exact_frac > 0.99, (
                f"q: only {exact_frac:.4f} of elements bit-exact; scales differ"
            )
            torch.testing.assert_close(q_u, q_f, atol=1e-2, rtol=E4M3_STEP)

        assert fusion_pass.matched_count == 1, (
            f"Expected 1 fused pair, got {fusion_pass.matched_count}"
        )

        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())
