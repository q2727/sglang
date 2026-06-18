"""MiniMax-M3 HiSparse block swap-in — JIT kernel wrapper (graph-safe).

GPU-side replacement for the CPU block-mode path.  The kernel runs
inline H2D copy (warp PTX from pinned host), so no Python-side .item()
syncs are needed — compatible with CUDA graph capture/replay.

Public API:

    minimax_hisparse_swap_in(
        topk_idx, seq_lens, req_to_host, req_pool_indices,
        host_k_buffer, host_v_buffer, hot_k_buffer, hot_v_buffer,
        hot_page_table, hot_kv_indices, hot_kv_indices_offset,
        next_hot_page, num_real_reqs, overflow_flag,
        hot_page_offset, hot_page_capacity, head_num, head_dim, elem_size_bytes,
    ) -> None
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_BLOCK_THREADS = 256
_PAGE_SIZE = 128


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(_BLOCK_THREADS, _PAGE_SIZE)
    return load_jit(
        "minimax_hisparse_swap_in",
        *args,
        cuda_files=["minimax/minimax_hisparse_swap_in.cuh"],
        cuda_wrappers=[
            ("minimax_hisparse_swap_in", f"minimax_hisparse_swap_in<{args}>"),
        ],
    )


def minimax_hisparse_swap_in(
    *,
    topk_idx: torch.Tensor,              # [Hkv, B, K]  int32 GPU
    seq_lens: torch.Tensor,               # [B]           int64 GPU
    req_to_host: torch.Tensor,            # [max_reqs, max_ctx] int64 GPU
    req_pool_indices: torch.Tensor,       # [B]           int64 GPU
    host_k_buffer: torch.Tensor,          # [host_size, Hkv, D] CPU pinned
    host_v_buffer: torch.Tensor,          # [host_size, Hkv, D] CPU pinned
    hot_k_buffer: torch.Tensor,           # [hot_size,  Hkv, D] GPU
    hot_v_buffer: torch.Tensor,           # [hot_size,  Hkv, D] GPU
    hot_page_table: torch.Tensor,         # [B, max_pages] int32 GPU out
    hot_kv_indices: torch.Tensor,         # [total_pages]  int32 GPU out
    hot_kv_indices_offset: torch.Tensor,  # [B]            int32 GPU out
    next_hot_page: torch.Tensor,          # [1] int32 GPU atomic
    num_real_reqs: torch.Tensor,          # [1] int32 GPU — graph-safe gate
    overflow_flag: torch.Tensor,          # [1] int32 GPU out
    hot_page_offset: int = 1,
    hot_page_capacity: int = 0,
    head_num: int = 4,
    head_dim: int = 128,
    elem_size_bytes: int = 2,
) -> None:
    """Run GPU-side block dedup, hot-page allocation, H2D copy, metadata.

    All output tensors must be pre-allocated on the CUDA device.
    After this call:
    - ``hot_page_table`` maps (batch, logical_block) → hot_page_id.
    - ``hot_kv_indices_offset`` stores per-request page counts.
    - ``hot_kv_indices`` stores hot page ids (batch-relative positions).
    - Hot K/V buffer contains the copied data.
    - ``overflow_flag`` is set to 1 if hot page capacity was exceeded.
    """
    assert topk_idx.is_cuda and topk_idx.dtype == torch.int32
    assert topk_idx.ndim == 3
    assert seq_lens.is_cuda and seq_lens.dtype == torch.int64
    assert req_to_host.is_cuda and req_to_host.dtype == torch.int64
    assert req_pool_indices.is_cuda and req_pool_indices.dtype == torch.int64
    assert hot_k_buffer.is_cuda
    assert hot_v_buffer.is_cuda
    assert hot_page_table.is_cuda and hot_page_table.dtype == torch.int32
    assert hot_kv_indices.is_cuda and hot_kv_indices.dtype == torch.int32
    assert hot_kv_indices_offset.is_cuda and hot_kv_indices_offset.dtype == torch.int32
    assert next_hot_page.is_cuda and next_hot_page.dtype == torch.int32
    assert num_real_reqs.is_cuda and num_real_reqs.dtype == torch.int32
    assert overflow_flag.is_cuda and overflow_flag.dtype == torch.int32

    # host buffers may be CPU (pinned) — no .is_cuda check
    gpu_tensors = [topk_idx, seq_lens, req_to_host, req_pool_indices,
                   hot_k_buffer, hot_v_buffer,
                   hot_page_table, hot_kv_indices, hot_kv_indices_offset,
                   next_hot_page, num_real_reqs, overflow_flag]
    for t in gpu_tensors:
        if not t.is_contiguous():
            t = t.contiguous()

    # host buffers must be contiguous for raw pointer access
    for t in [host_k_buffer, host_v_buffer]:
        if not t.is_contiguous():
            t = t.contiguous()

    module = _jit_module()
    module.minimax_hisparse_swap_in(
        topk_idx, seq_lens, req_to_host, req_pool_indices,
        host_k_buffer, host_v_buffer, hot_k_buffer, hot_v_buffer,
        hot_page_table, hot_kv_indices, hot_kv_indices_offset,
        next_hot_page, num_real_reqs, overflow_flag,
        int(hot_page_offset), int(hot_page_capacity),
        int(head_num), int(head_dim), int(elem_size_bytes),
    )


def minimax_hisparse_swap_in_available() -> bool:
    try:
        _jit_module()
        return True
    except Exception:
        return False
