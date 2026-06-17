"""MiniMax-M3 HiSparse block swap-in — JIT kernel wrapper.

GPU-side replacement for the CPU block-mode path in
MiniMaxHiSparseKVPool.load_sparse_main_blocks_to_hot.

Public API:

    minimax_hisparse_swap_in(
        topk_idx, seq_lens, req_to_host, req_pool_indices,
        hot_page_table, hot_kv_indices, hot_kv_indices_offset,
        host_locs, hot_locs, next_hot_page, overflow_flag,
        hot_page_offset, hot_page_capacity, num_real_reqs,
    ) -> None

All tensors must reside on CUDA device.  The kernel runs one CTA per
request and is CUDA-graph safe (all outputs are pre-allocated).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


# ---------------------------------------------------------------------------
# M3 compile-time constants
# ---------------------------------------------------------------------------
_BLOCK_THREADS = 256
_PAGE_SIZE = 128


@cache_once
def _jit_module() -> Module:
    """Load (or retrieve cached) the JIT-compiled swap-in kernel."""
    args = make_cpp_args(_BLOCK_THREADS, _PAGE_SIZE)
    return load_jit(
        "minimax_hisparse_swap_in",
        *args,
        cuda_files=["minimax/minimax_hisparse_swap_in.cuh"],
        cuda_wrappers=[
            (
                "minimax_hisparse_swap_in",
                f"minimax_hisparse_swap_in<{args}>",
            ),
        ],
    )


def minimax_hisparse_swap_in(
    *,
    topk_idx: torch.Tensor,              # [Hkv, B, K] int32
    seq_lens: torch.Tensor,               # [B] int64
    req_to_host: torch.Tensor,            # [max_reqs, max_ctx] int64
    req_pool_indices: torch.Tensor,       # [B] int64
    hot_page_table: torch.Tensor,         # [B, max_pages] int32 (output)
    hot_kv_indices: torch.Tensor,         # [total_pages] int32 (output)
    hot_kv_indices_offset: torch.Tensor,  # [B] int32 (output)
    host_locs: torch.Tensor,              # [total_tokens] int64 (output)
    hot_locs: torch.Tensor,               # [total_tokens] int64 (output)
    next_hot_page: torch.Tensor,          # [1] int32 (atomic counter)
    token_counter: torch.Tensor,          # [1] int32 (atomic counter)
    overflow_flag: torch.Tensor,          # [1] int32 (output)
    hot_page_offset: int = 1,
    hot_page_capacity: int = 0,
    num_real_reqs: int = 0,
) -> None:
    """Run GPU-side block dedup, hot-page allocation, and metadata construction.

    This replaces the CPU portion of the block-mode path.  All tensors must
    be pre-allocated on the CUDA device.  After this call:
    - ``hot_page_table`` maps (batch, logical_block) → hot_page_id.
    - ``hot_kv_indices_offset`` stores per-request page counts.
    - ``host_locs`` / ``hot_locs`` provide the H2D copy map.
    - ``overflow_flag`` is set to 1 if hot page capacity was exceeded.

    Args:
        topk_idx: Selected logical block ids, shape ``[Hkv, B, K]`` int32.
            Invalid entries are ``-1``.
        seq_lens: Sequence lengths, shape ``[B]`` int64.
        req_to_host: Maps (req_row, logical_pos) → host loc,
            shape ``[max_reqs, max_ctx]`` int64.
        req_pool_indices: Request pool indices, shape ``[B]`` int64.
        hot_page_table: Output, shape ``[B, max_pages]`` int32.
            Should be pre-filled with ``-1``.
        hot_kv_indices: Output, shape ``[total_pages]`` int32.
            Hot page ids for this request, stored at batch-relative positions.
        hot_kv_indices_offset: Output, shape ``[B]`` int32.
            Per-request count of hot pages (caller does prefix sum afterward).
        host_locs: Output, shape ``[total_tokens]`` int64.
            Host-side token locations for each selected token.
        hot_locs: Output, shape ``[total_tokens]`` int64.
            Hot-side token locations for each selected token.
        next_hot_page: Atomic counter, shape ``[1]`` int32.
            Should be zero-initialized before the first layer.
        overflow_flag: Output, shape ``[1]`` int32.
            Set to 1 if hot page allocation exceeded ``hot_page_capacity``.
        hot_page_offset: First usable hot page id (default 1, page 0 reserved).
        hot_page_capacity: Maximum number of hot pages available.
        num_real_reqs: Number of real (non-padded) requests in the batch.
    """
    assert topk_idx.is_cuda and topk_idx.dtype == torch.int32
    assert topk_idx.ndim == 3, f"topk_idx must be 3-D, got shape {topk_idx.shape}"
    assert seq_lens.is_cuda and seq_lens.dtype == torch.int64
    assert req_to_host.is_cuda and req_to_host.dtype == torch.int64
    assert req_pool_indices.is_cuda and req_pool_indices.dtype == torch.int64
    assert hot_page_table.is_cuda and hot_page_table.dtype == torch.int32
    assert hot_kv_indices.is_cuda and hot_kv_indices.dtype == torch.int32
    assert hot_kv_indices_offset.is_cuda and hot_kv_indices_offset.dtype == torch.int32
    assert host_locs.is_cuda and host_locs.dtype == torch.int64
    assert hot_locs.is_cuda and hot_locs.dtype == torch.int64
    assert next_hot_page.is_cuda and next_hot_page.dtype == torch.int32
    assert token_counter.is_cuda and token_counter.dtype == torch.int32
    assert overflow_flag.is_cuda and overflow_flag.dtype == torch.int32

    # Ensure contiguity (required by JIT kernel).
    tensors = [
        topk_idx, seq_lens, req_to_host, req_pool_indices,
        hot_page_table, hot_kv_indices, hot_kv_indices_offset,
        host_locs, hot_locs, next_hot_page, token_counter, overflow_flag,
    ]
    for t in tensors:
        if not t.is_contiguous():
            t = t.contiguous()

    module = _jit_module()
    module.minimax_hisparse_swap_in(
        topk_idx,
        seq_lens,
        req_to_host,
        req_pool_indices,
        hot_page_table,
        hot_kv_indices,
        hot_kv_indices_offset,
        host_locs,
        hot_locs,
        next_hot_page,
        token_counter,
        overflow_flag,
        int(hot_page_offset),
        int(hot_page_capacity),
        int(num_real_reqs),
    )


def minimax_hisparse_swap_in_available() -> bool:
    """Check whether the JIT swap-in kernel can be loaded on this device.

    Returns ``True`` on supported CUDA devices; ``False`` otherwise
    (the caller should fall back to the Python reference path).
    """
    try:
        _jit_module()
        return True
    except Exception:
        return False
