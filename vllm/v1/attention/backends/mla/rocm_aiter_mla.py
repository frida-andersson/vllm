# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionCGSupport, AttentionLayer, MultipleOf
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# Enable FP8 MLA prefill via mla_prefill_ps_asm_fwd + mla_reduce_v1
# instead of flash_attn_varlen_func.  Requires gfx950 (MI355X).
_use_fp8_mla_prefill = os.environ.get("VLLM_ROCM_FP8_MLA", "0") == "1"



class AiterMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1]

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA"

    @staticmethod
    def get_impl_cls() -> type["AiterMLAImpl"]:
        return AiterMLAImpl

    @staticmethod
    def get_builder_cls() -> type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder


@dataclass
class AiterMLADecodeMetadata(MLACommonDecodeMetadata):
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: torch.Tensor | None = None
    # The page indices of the paged kv cache
    paged_kv_indices: torch.Tensor | None = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: torch.Tensor | None = None
    # The query indptr, shape : [num_decode + 1]
    qo_indptr: torch.Tensor | None = None
    # The dtype of MLA out tensor
    attn_out_dtype: torch.dtype = torch.bfloat16
    # The max query output length: int
    max_qo_len: int | None = None

    # Persistent MLA kernel metadata
    work_metadata: torch.Tensor | None = None
    work_info_set: torch.Tensor | None = None
    work_indptr: torch.Tensor | None = None
    reduce_indptr: torch.Tensor | None = None
    reduce_final_map: torch.Tensor | None = None
    reduce_partial_map: torch.Tensor | None = None


class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]):
    pass


# Tile size used by the mla_prefill_ps_asm_fwd assembly kernel.
_FP8_PREFILL_TILE_Q = 256


class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):
    # TODO(luka, lucas): audit this as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, AiterMLAMetadata
        )

        gpu = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(gpu)
        cu_num = device_properties.multi_processor_count

        self.compilation_config = vllm_config.compilation_config
        self.decode_attn_out_dtype = vllm_config.model_config.dtype
        # kernel block size is always 1.
        max_num_pages_per_req = vllm_config.model_config.max_model_len
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req

        max_seqlen_qo = (
            1
            if vllm_config.speculative_config is None
            else vllm_config.speculative_config.num_speculative_tokens
        )

        max_qo_tiles_per_batch = int(
            math.ceil(max_seqlen_qo * self.num_heads / 128)
        )
        self.work_metadata = torch.empty(
            [10], dtype=torch.uint64, device="cuda"
        )
        self.work_indptr = torch.empty(
            [cu_num + 1], dtype=torch.int32, device="cuda"
        )
        self.work_info_set = torch.empty(
            [max_num_reqs * max_qo_tiles_per_batch * cu_num, 8],
            dtype=torch.int32,
            device="cuda",
        ).fill_(-1)
        self.reduce_indptr = torch.empty(
            [max_num_reqs * max_qo_tiles_per_batch + 1],
            dtype=torch.int32,
            device="cuda",
        )
        self.reduce_final_map = torch.empty(
            [max_num_reqs * max_qo_tiles_per_batch, 2],
            dtype=torch.int32,
            device="cuda",
        )
        self.reduce_partial_map = torch.empty(
            [max_num_reqs * max_qo_tiles_per_batch * cu_num],
            dtype=torch.int32,
            device="cuda",
        )

        # Preparing persistent buffers
        # TODO: we can disambiguate between decode and mixed-prefill decode here
        # so we can only use the persistent buffer if a cudagraph is actually
        # being used.

        # paged_kv_last_page_len is always 1s (kernel block size is always 1),
        # so we create it once and reuse slices in both eager and cudagraph modes.
        self.paged_kv_last_page_len = torch.ones(
            max_num_reqs, dtype=torch.int32, device=device
        )

        # Persistent buffer for paged_kv_indices to avoid blocking boolean mask
        # indexing (block_table_tensor[mask]) which has data-dependent output size.
        self.paged_kv_indices = torch.zeros(
            max_num_pages, dtype=torch.int32, device=device
        )

        # Pre-allocate FP8 MLA prefill PS metadata buffers.
        self._fp8_prefill_enabled = _use_fp8_mla_prefill
        if self._fp8_prefill_enabled:
            # The PS metadata describes how to partition work for a single
            # prefill batch.  The max Q-length per request in any batch is
            # bounded by max_num_batched_tokens (chunked prefill), which is
            # typically much smaller than max_model_len.  Using the tighter
            # bound reduces the pre-allocated work_info buffer by ~20×.
            max_prefill_qlen = min(
                vllm_config.model_config.max_model_len,
                vllm_config.scheduler_config.max_num_batched_tokens,
            )
            self._init_fp8_prefill_ps_buffers(
                max_num_reqs, max_prefill_qlen, device
            )

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )

            self.qo_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )

    def _init_fp8_prefill_ps_buffers(
        self,
        max_num_reqs: int,
        max_prefill_qlen: int,
        device: torch.device,
    ) -> None:
        """Pre-allocate persistent buffers for FP8 MLA prefill PS metadata.

        Uses ``get_ps_metadata_info_v1`` with max values so the buffers are
        large enough for any batch.  ``get_ps_metadata_v1`` fills them
        per-batch in ``build()``.

        Args:
            max_num_reqs: Maximum number of concurrent requests.
            max_prefill_qlen: Maximum Q-length for a single request in one
                prefill batch.  Should be ``min(max_model_len,
                max_num_batched_tokens)`` — the chunked-prefill scheduler
                never emits more than ``max_num_batched_tokens`` new tokens
                per batch.
            device: Target device for the buffers.
        """
        from aiter import get_ps_metadata_info_v1

        # After kv_b_proj decompression, K has num_heads heads (same as Q).
        # So gqa_ratio=1 and num_head_k=num_heads for the PS kernel.
        num_head_k = self.num_heads
        gqa_ratio = 1
        qlen_granularity = _FP8_PREFILL_TILE_Q // max(gqa_ratio, 1)

        (
            (wm_size, wm_dtype),
            (wi_size, wi_dtype),
            (wis_size, wis_dtype),
            (ri_size, ri_dtype),
            (rfm_size, rfm_dtype),
            (rpm_size, rpm_dtype),
        ) = get_ps_metadata_info_v1(
            batch_size=max_num_reqs,
            num_head_k=num_head_k,
            max_qlen=max_prefill_qlen,
            qlen_granularity=qlen_granularity,
        )

        self.fp8_ps_work_metadata = torch.empty(
            wm_size, dtype=wm_dtype, device=device
        )
        self.fp8_ps_work_indptr = torch.empty(
            wi_size, dtype=wi_dtype, device=device
        )
        self.fp8_ps_work_info = torch.empty(
            *wis_size, dtype=wis_dtype, device=device
        )
        self.fp8_ps_reduce_indptr = torch.empty(
            ri_size, dtype=ri_dtype, device=device
        )
        self.fp8_ps_reduce_final_map = torch.empty(
            *rfm_size, dtype=rfm_dtype, device=device
        )
        self.fp8_ps_reduce_partial_map = torch.empty(
            rpm_size, dtype=rpm_dtype, device=device
        )

        logger.info(
            "FP8 MLA prefill PS buffers allocated "
            "(max_batch=%d, max_qlen=%d, num_head_k=%d)",
            max_num_reqs,
            max_prefill_qlen,
            num_head_k,
        )

    def _build_fp8_prefill_ps_metadata(
        self,
        metadata: AiterMLAMetadata,
    ) -> None:
        """Build per-batch FP8 MLA prefill PS metadata and attach to *metadata*.

        Called from ``build()`` when prefill tokens are present and
        ``VLLM_ROCM_FP8_MLA=1``.
        """
        from aiter import get_ps_metadata_v1

        prefill = metadata.prefill
        qo_indptr = prefill.query_start_loc
        kv_indptr = qo_indptr  # new tokens: KV length == Q length

        # get_ps_metadata_v1 reads CPU tensors.
        qo_indptr_cpu = qo_indptr.to("cpu", dtype=torch.int32)
        kv_indptr_cpu = qo_indptr_cpu.clone()
        seq_lens_cpu = (qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).to(torch.int32)

        gqa_ratio = 1
        num_head_k = self.num_heads
        qhead_granularity = max(gqa_ratio, 1)
        qlen_granularity = _FP8_PREFILL_TILE_Q // qhead_granularity
        kvlen_granularity = 128
        block_size = 1  # non-paged: each "page" is one token

        get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_indptr_cpu,
            seq_lens_cpu,
            gqa_ratio,
            num_head_k,
            self.fp8_ps_work_metadata,
            self.fp8_ps_work_indptr,
            self.fp8_ps_work_info,
            self.fp8_ps_reduce_indptr,
            self.fp8_ps_reduce_final_map,
            self.fp8_ps_reduce_partial_map,
            qhead_granularity=qhead_granularity,
            qlen_granularity=qlen_granularity,
            kvlen_granularity=kvlen_granularity,
            block_size=block_size,
            is_causal=True,
        )

        total_prefill_tokens = qo_indptr_cpu[-1].item()
        kv_indices = torch.arange(
            total_prefill_tokens, device=qo_indptr.device, dtype=torch.int32
        )

        # Attach PS metadata to the metadata object so forward_mha can read it.
        metadata.fp8_prefill_qo_indptr = qo_indptr
        metadata.fp8_prefill_kv_indptr = kv_indptr
        metadata.fp8_prefill_kv_indices = kv_indices
        metadata.fp8_prefill_work_indptr = self.fp8_ps_work_indptr
        metadata.fp8_prefill_work_info_set = self.fp8_ps_work_info
        metadata.fp8_prefill_reduce_indptr = self.fp8_ps_reduce_indptr
        metadata.fp8_prefill_reduce_final_map = self.fp8_ps_reduce_final_map
        metadata.fp8_prefill_reduce_partial_map = self.fp8_ps_reduce_partial_map
        metadata.fp8_prefill_max_q_len = prefill.max_query_len

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build
        )
        if self._fp8_prefill_enabled and metadata.prefill is not None:
            self._build_fp8_prefill_ps_metadata(metadata)
        return metadata

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> AiterMLADecodeMetadata:
        # kernel block size is always 1, although the kv block size is not 1.
        device = self.device
        num_reqs = seq_lens_device.size(0)

        # kernel block size is always 1, so each page has exactly 1 token.
        # last_page_len is always 1 - just slice the pre-initialized buffer.
        paged_kv_last_page_len = self.paged_kv_last_page_len[:num_reqs]

        paged_kv_indptr = torch.cat(
            [
                torch.zeros(1, dtype=seq_lens_device.dtype, device=device),
                seq_lens_device.cumsum(dim=0, dtype=torch.int32),
            ]
        )
        qo_len = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_qo_len = qo_len.max().item()

        kv_indptr = torch.zeros(
            [query_start_loc_cpu.size(0)], dtype=torch.int32, device="cuda"
        )
        torch.cumsum(seq_lens_device, dim=0, out=kv_indptr[1:])

        import aiter

        aiter.get_mla_metadata_v1(
            query_start_loc_device,
            kv_indptr,
            paged_kv_last_page_len,
            self.num_heads // self.kv_cache_spec.num_kv_heads,
            self.kv_cache_spec.num_kv_heads,
            True,
            self.work_metadata,
            self.work_info_set,
            self.work_indptr,
            self.reduce_indptr,
            self.reduce_final_map,
            self.reduce_partial_map,
            kv_granularity=max(self.kv_cache_spec.block_size, 16),
            max_seqlen_qo=max_qo_len,
            uni_seqlen_qo=max_qo_len,
            fast_mode=True,
        )

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indices.fill_(-1)
        _copy_page_indices_kernel[(num_reqs,)](
            self.paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            BLOCK_SIZE=1024,
        )
        paged_kv_indices = self.paged_kv_indices

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr[: 1 + num_reqs].copy_(
                paged_kv_indptr, non_blocking=True
            )
            self.paged_kv_indptr[1 + num_reqs :].fill_(paged_kv_indptr[-1])
            paged_kv_indptr = self.paged_kv_indptr[: 1 + num_reqs]

            # paged_kv_last_page_len already uses the pre-initialized buffer slice
            # (set above), so no copy needed - buffer is always 1s.

            self.qo_indptr[: 1 + num_reqs].copy_(
                query_start_loc_device, non_blocking=True
            )
            self.qo_indptr[1 + num_reqs :] = query_start_loc_device[-1]
            qo_indptr = self.qo_indptr[: 1 + num_reqs]

        else:
            qo_indptr = torch.arange(
                0, num_reqs + 1, step=1, dtype=torch.int32, device=device
            )

        attn_metadata = AiterMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            qo_indptr=qo_indptr,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            max_qo_len=max_qo_len,
            attn_out_dtype=self.decode_attn_out_dtype,
            work_metadata=self.work_metadata,
            work_info_set=self.work_info_set,
            work_indptr=self.work_indptr,
            reduce_indptr=self.reduce_indptr,
            reduce_final_map=self.reduce_final_map,
            reduce_partial_map=self.reduce_partial_map,
        )

        return attn_metadata


@triton.jit
def _copy_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy block table rows into a flat page_indices buffer using indptr.
    Avoids blocking boolean mask indexing (tensor[mask]) which has
    data-dependent output size and forces sync.
    This is the same kernel as introduced in backends/flashinfer.py.
    """
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=i + offset < num_blocks,
        )


class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )
        _valid_heads = num_heads in (4, 8) or (
            num_heads % 16 == 0 and 16 <= num_heads <= 128
        )
        assert _valid_heads, (
            f"Aiter MLA supports num_heads of 4, 8, or multiples of 16 "
            f"in [16, 128].\n"
            f"Provided {num_heads} number of heads.\n"
            "Try adjusting tensor_parallel_size value."
        )
        self._needs_head_repeat = num_heads < 16
        self._head_repeat_factor = 16 // num_heads if num_heads < 16 else 1
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "Aiter MLA does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        from aiter import flash_attn_varlen_func

        self.flash_attn_varlen_func = flash_attn_varlen_func
        self._decode_out = None

        # FP8 MLA prefill kernel imports (lazy, only when enabled).
        self._fp8_prefill_enabled = _use_fp8_mla_prefill
        if self._fp8_prefill_enabled:
            from aiter import mla_prefill_ps_asm_fwd, mla_reduce_v1

            self._mla_prefill_ps_asm_fwd = mla_prefill_ps_asm_fwd
            self._mla_reduce_v1 = mla_reduce_v1

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        output = self.flash_attn_varlen_func(  # type: ignore[call-arg]
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            return_lse=return_softmax_lse,
            **kwargs,
        )

        return output

    def _mla_fp8_prefill_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
    ) -> torch.Tensor:
        """Run FP8 MLA prefill via mla_prefill_ps_asm_fwd + mla_reduce_v1.

        Q, K, V are already decompressed (post-kv_b_proj).  After
        decompression K and V have ``num_heads`` heads (same as Q), so
        gqa_ratio = 1 which matches the Gqa=1 assembly kernel.

        Args:
            q: [total_tokens, num_heads, qk_head_dim]
            k: [total_tokens, num_heads, qk_head_dim]
            v: [total_tokens, num_heads, v_head_dim]
            attn_metadata: Must have fp8_prefill_* attributes set by the
                metadata builder.

        Returns:
            Output tensor [total_tokens, num_heads, v_head_dim] in the
            model's working dtype (e.g., bfloat16).
        """
        from vllm.platforms import current_platform

        fp8_dtype = current_platform.fp8_dtype()
        total_q = q.shape[0]
        nhead = self.num_heads
        v_head_dim = self.v_head_dim
        tile_q = _FP8_PREFILL_TILE_Q
        out_dtype = q.dtype if q.dtype != fp8_dtype else torch.bfloat16

        # Cast to FP8 if not already.
        if q.dtype != fp8_dtype:
            q = q.to(fp8_dtype)
        if k.dtype != fp8_dtype:
            k = k.to(fp8_dtype)
        if v.dtype != fp8_dtype:
            v = v.to(fp8_dtype)

        one_scale = torch.ones((), dtype=torch.float32, device=q.device)

        reduce_partial_map = attn_metadata.fp8_prefill_reduce_partial_map

        # The actual number of active partial tiles for this batch is stored
        # in reduce_indptr[-1].  Using reduce_partial_map.size(0) instead
        # would allocate the *maximum* pre-allocated size, which can be tens
        # of GiB for large max_model_len values.
        num_partial_tiles = int(
            attn_metadata.fp8_prefill_reduce_indptr[-1].item()
        )

        # Intermediate buffers for the two-phase PS kernel.
        logits = torch.empty(
            (num_partial_tiles * tile_q, nhead, v_head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        attn_lse = torch.empty(
            (num_partial_tiles * tile_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )
        final_lse = torch.empty(
            (total_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )
        output = torch.empty(
            (total_q, nhead, v_head_dim),
            dtype=out_dtype,
            device=q.device,
        )

        # Phase 1: persistent-scheduling assembly prefill kernel.
        self._mla_prefill_ps_asm_fwd(
            q,
            k,
            v,
            attn_metadata.fp8_prefill_qo_indptr,
            attn_metadata.fp8_prefill_kv_indptr,
            attn_metadata.fp8_prefill_kv_indices,
            attn_metadata.fp8_prefill_work_indptr,
            attn_metadata.fp8_prefill_work_info_set,
            attn_metadata.fp8_prefill_max_q_len,
            self.scale,
            True,  # is_causal
            logits,
            attn_lse,
            output,
            one_scale,
            one_scale,
            one_scale,
        )

        # Phase 2: reduction across KV splits.
        self._mla_reduce_v1(
            logits,
            attn_lse,
            attn_metadata.fp8_prefill_reduce_indptr,
            attn_metadata.fp8_prefill_reduce_final_map,
            attn_metadata.fp8_prefill_reduce_partial_map,
            tile_q,
            output,
            final_lse,
        )

        return output

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Override to use FP8 MLA prefill when VLLM_ROCM_FP8_MLA=1.

        Falls back to the parent (flash_attn_varlen_func) when:
        - FP8 MLA prefill is not enabled
        - There is chunked context (prior KV cache to merge)
        - PS metadata was not built (e.g., pure decode batch)
        """
        if not self._fp8_prefill_enabled or not hasattr(
            attn_metadata, "fp8_prefill_qo_indptr"
        ):
            return super().forward_mha(
                q, kv_c_normed, k_pe, kv_c_and_k_pe_cache,
                attn_metadata, k_scale, output,
            )

        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        has_context = prefill_metadata.chunked_context is not None

        if has_context:
            # Chunked context requires merge_attn_states; fall back to parent
            # which handles the two-pass context + suffix merge.
            return super().forward_mha(
                q, kv_c_normed, k_pe, kv_c_and_k_pe_cache,
                attn_metadata, k_scale, output,
            )

        # Decompress KV: kv_c_normed → k_nope, v via kv_b_proj
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        # FP8 MLA prefill: Q, K, V → mla_prefill_ps_asm_fwd + mla_reduce_v1
        output_prefill = self._mla_fp8_prefill_attn(
            q, k, v, attn_metadata
        )

        # Handle v_head_dim padding if present.
        if self._pad_v:
            output_prefill = output_prefill[..., : v.shape[-1]]

        output.copy_(output_prefill.flatten(start_dim=-2))

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None
        assert attn_metadata.decode.max_qo_len is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]

        if self._needs_head_repeat:
            q = q.repeat_interleave(self._head_repeat_factor, dim=1)
            kernel_num_heads = 16
        else:
            kernel_num_heads = self.num_heads

        dtype = attn_metadata.decode.attn_out_dtype
        if (
            self._decode_out is None
            or self._decode_out.shape[0] < B
            or self._decode_out.dtype != dtype
        ):
            self._decode_out = torch.zeros(
                B,
                kernel_num_heads,
                self.kv_lora_rank,
                dtype=dtype,
                device=q.device,
            )
        o = self._decode_out[:B]

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        rocm_aiter_ops.mla_decode_fwd(
            q,
            kv_buffer,
            o,
            self.scale,
            attn_metadata.decode.qo_indptr,
            attn_metadata.decode.max_qo_len,
            attn_metadata.decode.paged_kv_indptr,
            attn_metadata.decode.paged_kv_indices,
            attn_metadata.decode.paged_kv_last_page_len,
            q_scale=layer._q_scale,
            kv_scale=layer._k_scale,
            work_meta_data=attn_metadata.decode.work_metadata,
            work_indptr=attn_metadata.decode.work_indptr,
            work_info_set=attn_metadata.decode.work_info_set,
            reduce_indptr=attn_metadata.decode.reduce_indptr,
            reduce_final_map=attn_metadata.decode.reduce_final_map,
            reduce_partial_map=attn_metadata.decode.reduce_partial_map,
        )

        if self._needs_head_repeat:
            o = o[:, :: self._head_repeat_factor, :]

        return o, None
