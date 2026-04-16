# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    QueryLenSupport,
    get_mla_dims,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
    AiterMLABackend,
    AiterMLADecodeMetadata,
    AiterMLAImpl,
    AiterMLAMetadata,
    AiterMLAMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
logger = init_logger(__name__)


@triton.jit
def fetch_id_to_ragged_kernel(
    in_tensor_ptr,  # [num_seq, topk]
    cumsum_ptr,  # [num_seq + 1]
    out_tensor_ptr,  # [max_num_seq * topk]
    in_tensor_ptr_stride,
    TOPK: tl.constexpr,
    TOKEN_NUM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    token_start = tl.load(cumsum_ptr + seq_id)
    token_end = tl.load(cumsum_ptr + seq_id + 1)
    token_num = token_end - token_start
    row_offset = block_id * BLOCK_SIZE
    if row_offset >= token_num:
        return
    in_tensor_offset = seq_id * in_tensor_ptr_stride + row_offset + offset
    in_tensor_mask = (row_offset + offset) < TOPK
    in_tensor_val = tl.load(in_tensor_ptr + in_tensor_offset, mask=in_tensor_mask)
    out_tensor_offset = token_start + row_offset + offset
    out_tensor_mask = (out_tensor_offset < token_end) & in_tensor_mask
    tl.store(out_tensor_ptr + out_tensor_offset, in_tensor_val, mask=out_tensor_mask)


def fetch_id_to_ragged_triton(
    in_tensor: torch.Tensor, cumsum: torch.Tensor, out_tensor: torch.Tensor, topk
):
    num_tokens = in_tensor.size(0)
    block_size = 64
    num_block_per_row = triton.cdiv(topk, block_size)
    grid = (
        num_tokens,
        num_block_per_row,
    )
    fetch_id_to_ragged_kernel[grid](
        in_tensor, cumsum, out_tensor, in_tensor.stride(0), topk, num_tokens, block_size
    )


@triton.jit
def generate_sparse_seqlen_kernel(
    seq_len_ptr,
    cu_query_lens_ptr,
    out_ptr,
    topk_token: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    query_offset = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    query_start = tl.load(cu_query_lens_ptr + seq_id)
    query_end = tl.load(cu_query_lens_ptr + seq_id + 1)
    if query_start + tl.program_id(1) * BLOCK_SIZE > query_end:
        return
    query_len = query_end - query_start
    query_mask = query_offset + query_start < query_end
    seq_len = tl.load(seq_len_ptr + seq_id)
    if seq_len == 0:
        return
    context_start_point = seq_len - query_len
    sparse_seqlen = context_start_point + query_offset
    sparse_seqlen_masked = tl.where(
        sparse_seqlen + 1 < topk_token, sparse_seqlen + 1, topk_token
    )
    tl.store(
        out_ptr + query_start + query_offset, sparse_seqlen_masked, mask=query_mask
    )


def generate_sparse_seqlen_triton(
    seq_lens: torch.Tensor,
    cu_query_lens: torch.Tensor,
    topk_token: int,
    num_tokens: int,
    max_query_len: int,
):
    num_seqs = seq_lens.size(0)
    out = torch.zeros([num_tokens], dtype=torch.int32, device=seq_lens.device)
    block_size = 64
    num_block_per_row = triton.cdiv(max_query_len, block_size)
    grid = (num_seqs, num_block_per_row)
    generate_sparse_seqlen_kernel[grid](
        seq_lens, cu_query_lens, out, topk_token, block_size,
    )
    return out


@dataclass
class ROCMAiterMLASparseMetadata(AiterMLAMetadata):
    """Extends AiterMLAMetadata with sparse-specific fields for decode."""

    # Sparse decode fields
    sparse_req_id_per_token: torch.Tensor | None = None
    sparse_topk_tokens: int = 2048
    sparse_qo_indptr: torch.Tensor | None = None
    sparse_paged_kv_last_page_len: torch.Tensor | None = None
    sparse_paged_kv_indices: torch.Tensor | None = None
    sparse_paged_kv_indptr: torch.Tensor | None = None


class ROCMAiterMLASparseBackend(AiterMLABackend):
    """Sparse MLA backend that inherits prefill (MHA) from AiterMLABackend
    and uses sparse decode via mla_decode_fwd with topk index selection."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA_SPARSE"

    @staticmethod
    def get_metadata_cls() -> type["ROCMAiterMLASparseMetadata"]:
        return ROCMAiterMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["ROCMAiterMLASparseMetadataBuilder"]:
        return ROCMAiterMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["ROCMAiterMLASparseImpl"]:
        return ROCMAiterMLASparseImpl

    @classmethod
    def is_sparse(cls) -> bool:
        return True


class ROCMAiterMLASparseMetadataBuilder(
    AiterMLAMetadataBuilder
):
    """Metadata builder that inherits prefill + decode building from
    AiterMLAMetadataBuilder and adds sparse-specific fields for decode."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Override metadata class so parent builder creates our sparse metadata
        self.metadata_cls = ROCMAiterMLASparseMetadata

        # AiterMLAMetadataBuilder.__init__ allocates paged_kv_indices sized for
        # max_num_reqs * max_model_len pages, assuming kernel_block_size=1 (one
        # page per token).  With kernel_block_size=64 that is 64× more entries
        # than needed — and our _build_decode override never uses this buffer at
        # all.  Releasing it saves ~52 MB per attention layer (≈3 GB total for
        # DeepSeek-V3.2's ~60 MLA layers per TP rank), which is critical budget
        # for the indexer's prefill logits allocation.
        self.paged_kv_indices = torch.zeros(1, dtype=torch.int32, device=device)

        # The parent also allocates persistent MLA metadata buffers for
        # aiter.get_mla_metadata_v1() (work_metadata, work_indptr,
        # work_info_set, reduce_indptr, reduce_final_map, reduce_partial_map).
        # Our _build_decode override never calls that function — release them.
        # Largest is work_info_set at ~5 MB on MI355X (304 CUs × 512 reqs × 8).
        _placeholder = torch.empty(0, dtype=torch.int32, device=device)
        self.work_metadata = torch.empty(0, dtype=torch.uint64, device=device)
        self.work_indptr = _placeholder
        self.work_info_set = _placeholder
        self.reduce_indptr = _placeholder
        self.reduce_final_map = _placeholder
        self.reduce_partial_map = _placeholder

        # Sparse-specific fields
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk

        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        self.req_id_per_token_buffer = torch.empty(
            (max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

        # Sparse decode uses separate paged_kv tracking for topk indices
        self.sparse_paged_kv_indices = torch.zeros(
            [max_num_batched_tokens * self.topk_tokens],
            dtype=torch.int32,
            device=device,
        )
        self.sparse_paged_kv_indptr = torch.zeros(
            [max_num_batched_tokens + 1], dtype=torch.int32, device=device
        )
        self.sparse_qo_indptr = torch.arange(
            0, max_num_batched_tokens + 1, dtype=torch.int32, device=device
        )
        self.sparse_paged_kv_last_page_len = torch.ones(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )

    def _build_sparse_fields(self, common_attn_metadata):
        """Build sparse-specific metadata fields for decode."""
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(
            common_attn_metadata.query_start_loc_cpu, dtype=np.int32
        )
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )

        self.sparse_paged_kv_indices.fill_(0)
        self.sparse_paged_kv_indptr.fill_(0)

        seq_lens = common_attn_metadata.seq_lens
        sparse_seqlen = generate_sparse_seqlen_triton(
            seq_lens,
            common_attn_metadata.query_start_loc,
            self.topk_tokens,
            num_tokens,
            common_attn_metadata.max_query_len,
        )
        torch.cumsum(
            sparse_seqlen, dim=0,
            out=self.sparse_paged_kv_indptr[1 : num_tokens + 1]
        )
        self.sparse_paged_kv_indptr[num_tokens + 1 :].fill_(
            self.sparse_paged_kv_indptr[num_tokens]
        )

        return {
            "sparse_req_id_per_token": self.req_id_per_token_buffer[:num_tokens],
            "sparse_topk_tokens": self.topk_tokens,
            "sparse_qo_indptr": self.sparse_qo_indptr[: num_tokens + 1],
            "sparse_paged_kv_last_page_len": (
                self.sparse_paged_kv_last_page_len[:num_tokens]
            ),
            "sparse_paged_kv_indices": (
                self.sparse_paged_kv_indices[: num_tokens * self.topk_tokens]
            ),
            "sparse_paged_kv_indptr": (
                self.sparse_paged_kv_indptr[: num_tokens + 1]
            ),
        }

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
        # The parent's _build_decode calls _copy_page_indices_kernel which
        # iterates seq_len times per row in block_table.  With kernel_block_size
        # = 64 each row has only ceil(seq_len/64) valid entries → OOB GPU read.
        # Our forward_mqa only reads attn_metadata.decode.block_table directly;
        # it never uses paged_kv_indices or the AITER work metadata.
        # Return a minimal decode metadata with just the fields we need.
        return AiterMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        # Build sparse fields first
        sparse_fields = self._build_sparse_fields(common_attn_metadata)

        # Build standard prefill + decode metadata via parent
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build
        )

        # Attach sparse fields to metadata
        for key, value in sparse_fields.items():
            setattr(metadata, key, value)

        return metadata


# Take from
# https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla_prefill.py#L72
def reference_mla_sparse_prefill(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float, d_v: int
) -> tuple[torch.Tensor, torch.Tensor]:
    import math

    def log2sumexp2(a: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(a * math.log(2), dim=dim) * math.log2(math.e)

    skv = kv.shape[0]
    sq = q.shape[0]
    topk = indices.shape[-1]
    dqk = q.shape[-1]
    indices = indices[:, 0, :]  # [s_q, topk]
    invalid_indices_mask = (indices < 0) | (indices >= skv)
    indices[invalid_indices_mask] = 0
    qs = q  # [s_q, h_q, d_qk]
    kvs = kv[:, 0, :][indices].view(sq, topk, dqk)  # [s_q, topk, d_qk]

    attn_score = (qs @ kvs.transpose(1, 2)).float()  # [s_q, h_q, topk]
    attn_score.masked_fill_(invalid_indices_mask.unsqueeze(1), float("-inf"))
    attn_score *= sm_scale * math.log2(math.e)
    lse = log2sumexp2(attn_score, dim=-1)  # [s_q, h_q]
    attn_score = torch.exp2(attn_score - lse.unsqueeze(-1))  # [s_q, h_q, topk]
    result = attn_score.to(q.dtype) @ kvs[:, :, :d_v]
    return (result, lse)


class ROCMAiterMLASparseImpl(AiterMLAImpl):
    """Sparse MLA impl that inherits forward_mha (compute-bound prefill via
    flash_attn_varlen_func) from AiterMLAImpl/MLACommonImpl, and overrides
    forward_mqa for sparse decode via mla_decode_fwd with topk indices."""

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
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            indexer=indexer,
            **mla_args,
        )
        # Sparse-specific: get the topk indices buffer from the indexer
        assert indexer is not None
        self.topk_indices_buffer: torch.Tensor | None = indexer.topk_indices_buffer
        self._decode_out: torch.Tensor | None = None

    def _forward_sparse_mla(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: ROCMAiterMLASparseMetadata,
        layer: AttentionLayer | None = None,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        attn_out_dtype = q.dtype

        is_fp8_kv = self.kv_cache_dtype.startswith("fp8")
        if is_fp8_kv:
            from vllm.platforms import current_platform
            fp8_dtype = current_platform.fp8_dtype()
            q_scale = layer.k_scale if layer is not None else None
            k_scale = layer.k_scale if layer is not None else None
            q = q.to(fp8_dtype)
            kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(fp8_dtype)

        # mla_decode_fwd uses page_size=1 (per-token paging) internally.
        # When kernel_block_size > 1, the KV cache shape is
        # [num_pages, block_size, head_size].  Flatten to
        # [num_pages * block_size, 1, head_size] so that the flat token
        # indices in sparse_paged_kv_indices correctly address each token.
        if kv_c_and_k_pe_cache.shape[1] != 1:
            kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.reshape(
                -1, 1, kv_c_and_k_pe_cache.shape[-1]
            )

        # Slice sparse CSR fields to num_tokens (= num_decode_tokens).
        # _build_sparse_fields builds these for num_actual_tokens (prefill+decode),
        # but forward_mqa only handles decode tokens (q.shape[0]).  In a mixed
        # prefill+decode batch, passing the full qo_indptr/kv_indptr would set
        # bs = num_actual_tokens while o has num_decode_tokens rows, causing the
        # stage2 kernel to write o[cur_qo] for cur_qo >= total_s → GPU OOB fault.
        qo_indptr = attn_metadata.sparse_qo_indptr[:num_tokens + 1]
        kv_indptr = attn_metadata.sparse_paged_kv_indptr[:num_tokens + 1]
        kv_last_page_len = attn_metadata.sparse_paged_kv_last_page_len[:num_tokens]

        if (
            self._decode_out is None
            or self._decode_out.shape[0] < num_tokens
            or self._decode_out.dtype != attn_out_dtype
        ):
            self._decode_out = torch.zeros(
                [num_tokens, self.num_heads, self.kv_lora_rank],
                dtype=attn_out_dtype,
                device=q.device,
            )
        output = self._decode_out[:num_tokens]
        fetch_id_to_ragged_triton(
            topk_indices,
            kv_indptr,
            attn_metadata.sparse_paged_kv_indices,
            attn_metadata.sparse_topk_tokens,
        )

        if is_fp8_kv:
            rocm_aiter_ops.mla_decode_fwd(
                q,
                kv_c_and_k_pe_cache,
                output,
                self.scale,
                qo_indptr,
                1,
                kv_indptr,
                attn_metadata.sparse_paged_kv_indices,
                kv_last_page_len,
                q_scale,
                k_scale,
            )
        else:
            rocm_aiter_ops.mla_decode_fwd(
                q,
                kv_c_and_k_pe_cache,
                output,
                self.scale,
                qo_indptr,
                1,
                kv_indptr,
                attn_metadata.sparse_paged_kv_indices,
                kv_last_page_len,
            )

        return output[:, : self.num_heads, :]

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: ROCMAiterMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # For sparse decode, use MQA 576/512 approach with topk indices

        # Concatenate q if it's a tuple (ql_nope, q_pe)
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        # Get topk indices
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        assert attn_metadata.decode is not None
        # Slice req_id_per_token to decode-only tokens.  _build_sparse_fields
        # builds sparse_req_id_per_token for all tokens (prefill + decode),
        # but forward_mqa is only called with decode q (num_actual_toks rows).
        # triton_convert_req_index_to_global_index uses req_id.shape[0] as the
        # grid dimension and writes out[token_id] — passing the full
        # num_actual_tokens tensor with a num_decode_tokens output would OOB.
        topk_indices_global = triton_convert_req_index_to_global_index(
            attn_metadata.sparse_req_id_per_token[:num_actual_toks],
            attn_metadata.decode.block_table,
            topk_indices,
            BLOCK_SIZE=64,  # block_size=64 for this backend
            NUM_TOPK_TOKENS=attn_metadata.sparse_topk_tokens,
        )

        attn_out = self._forward_sparse_mla(
            q, kv_c_and_k_pe_cache, topk_indices_global, attn_metadata, layer
        )

        return attn_out, None
