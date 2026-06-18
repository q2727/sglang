"""MiniMax-M3 HiSparse coordinator.

Manages the request lifecycle for MiniMax-M3 decode-only HiSparse. This
coordinator is intentionally simpler than the DSA HiSparseCoordinator:

- No staging queue (backup is synchronous via set_fused_kv_index_buffer).
- No per-request device buffer (hot buffer is layer-global, rebuilt per step).
- No compress_ratio (M3 does not compress tokens).
- No dual-allocator indirection.

The primary data structure is ``req_to_host``, a 2-D CPU tensor mapping
(request row, logical token position) → host pool slot index in
MiniMaxSparseMainHostPool.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseTokenStats
from sglang.srt.utils import get_device_module

device_module = get_device_module()

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.minimax_hisparse_memory_pool import MiniMaxHiSparseKVPool
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


class MiniMaxHiSparseCoordinator:
    """Minimal coordinator for MiniMax-M3 HiSparse request lifecycle.

    Owns the ``req_to_host`` mapping that translates (req_pool_idx,
    logical_token_pos) → host pool slot. The host pool
    (``MiniMaxSparseMainHostPool``) manages its own free list; the coordinator
    calls ``alloc`` / ``free`` on it for each token slot.

    Scheduler hooks are intentionally thin:
    - **prefill**: after alloc_extend, allocate host slots and write req_to_host.
    - **decode**: append one host slot per new token.
    - **finish/abort**: free host slots, clear req_to_host row.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: MiniMaxHiSparseKVPool,
        token_to_kv_pool_allocator: PagedTokenToKVPoolAllocator,
        device: str,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.device = device

        max_num_reqs = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len

        # Maps (req_pool_idx, logical_token_pos) → host pool slot index.
        # -1 means "not allocated". CPU tensor — consumed by the pool's
        # load_sparse_main_blocks_to_hot block mode which moves it to CPU.
        self.req_to_host = torch.full(
            (max_num_reqs, max_context_len),
            -1,
            dtype=torch.int64,
            device="cpu",
        )
        # Number of allocated host slots per request.
        self.req_to_host_allocated_len = torch.zeros(
            max_num_reqs, dtype=torch.int64, device="cpu"
        )

        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        self._decode_producer_stream: Optional[device_module.Stream] = None

    # ------------------------------------------------------------------
    # Host slot allocation
    # ------------------------------------------------------------------

    def _alloc_host_slots(self, num: int) -> torch.Tensor:
        """Allocate contiguous host slots from the host pool."""
        host_pool = self.token_to_kv_pool.sparse_main_host_pool
        return host_pool.alloc(num)

    def _free_host_slots(self, indices: torch.Tensor) -> None:
        """Return host slots to the host pool free list."""
        if indices.numel() == 0:
            return
        host_pool = self.token_to_kv_pool.sparse_main_host_pool
        host_pool.free(indices)

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def admit_prefill(self, req: Req) -> None:
        """Allocate host pool slots for a prefill request.

        Called after ``alloc_extend`` has allocated GPU slots for dense main
        K/V and sparse index K.  The model forward pass writes sparse main K/V
        to host through ``set_fused_kv_index_buffer``, which calls
        ``backup_sparse_main_to_host`` synchronously.

        After this call, the request is immediately decode-ready (no staging).
        """
        prefill_len = req.fill_len
        if prefill_len <= 0:
            return

        host_locs = self._alloc_host_slots(prefill_len)
        self.req_to_host[req.req_pool_idx, :prefill_len] = host_locs
        self.req_to_host_allocated_len[req.req_pool_idx] = prefill_len

        logger.debug(
            "MiniMaxHiSparse: admitted prefill req %s (len=%d, host_slots=%d..%d)",
            req.rid,
            prefill_len,
            int(host_locs[0].item()),
            int(host_locs[-1].item()),
        )

    def admit_request_into_staging(self, req: Req) -> None:
        """Scheduler-compatible entry point for prefill completion.

        M3 has no staging queue.  This method allocates host slots and marks
        the request as decode-ready immediately (hisparse_staging=False).
        """
        self.admit_prefill(req)
        req.hisparse_staging = False

    def extend_decode(self, req: Req) -> None:
        """Allocate one host slot for the new decode token.

        Called after ``alloc_decode`` has allocated a GPU slot for the new
        token.  The host slot is written into ``req_to_host`` so that the
        block-mode ``load_sparse_main_blocks_to_hot`` can later resolve
        logical positions to host slots.
        """
        seq_len = req.kv_allocated_len
        if seq_len <= 0:
            return

        max_ctx = self.req_to_host.shape[1]
        if seq_len > max_ctx:
            raise RuntimeError(
                f"MiniMaxHiSparse: req {req.rid} seq_len={seq_len} exceeds "
                f"max_context_len={max_ctx}."
            )

        current_allocated = int(self.req_to_host_allocated_len[req.req_pool_idx])
        if seq_len <= current_allocated:
            # Already allocated (e.g. re-prompt or chunked prefill edge case).
            return

        num_new = seq_len - current_allocated
        host_locs = self._alloc_host_slots(num_new)
        self.req_to_host[req.req_pool_idx, current_allocated:seq_len] = host_locs
        self.req_to_host_allocated_len[req.req_pool_idx] = seq_len

        logger.debug(
            "MiniMaxHiSparse: extended req %s decode (seq_len=%d, +%d host slots)",
            req.rid,
            seq_len,
            num_new,
        )

    def request_finished(self, req: Req) -> None:
        """Release host pool resources for a finished request.

        Does NOT free GPU allocator slots — the caller (scheduler's
        ``release_kv_cache`` path) handles that.
        """
        allocated_len = int(self.req_to_host_allocated_len[req.req_pool_idx])
        if allocated_len > 0:
            host_locs = self.req_to_host[req.req_pool_idx, :allocated_len].clone()
            self._free_host_slots(host_locs)

        self.req_to_host[req.req_pool_idx, :] = -1
        self.req_to_host_allocated_len[req.req_pool_idx] = 0

        logger.debug(
            "MiniMaxHiSparse: finished req %s (freed %d host slots)",
            req.rid,
            allocated_len,
        )

    def retract_req(self, req: Req) -> None:
        """Abort a request, releasing host resources."""
        self.request_finished(req)

    # ------------------------------------------------------------------
    # Staging interface (M3 has no staging — all methods return trivially)
    # ------------------------------------------------------------------

    def has_ongoing_staging(self) -> bool:
        """M3 has no async staging DMA queue."""
        return False

    def collect_ready_reqs(self) -> list:
        """M3 requests are decode-ready immediately after prefill."""
        return []

    # ------------------------------------------------------------------
    # Stream management (minimal — backup is synchronous in first phase)
    # ------------------------------------------------------------------

    def set_decode_producer_stream(self, stream) -> None:
        self._decode_producer_stream = stream

    def wait_for_pending_backup(self) -> None:
        """M3 backup is synchronous; no async backup stream to wait on."""
        pass

    # ------------------------------------------------------------------
    # Per-step scheduler hooks (called from ScheduleBatch.prepare_for_decode)
    # ------------------------------------------------------------------

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """No-op for M3: the hot buffer is rebuilt per layer per step.

        The DSA coordinator uses this hook to back up the previous decode
        token to host and grow per-request device buffers.  M3 backup is
        synchronous via ``set_fused_kv_index_buffer`` inside the attention
        backend, and M3 has no per-request device buffer.
        """
        pass

    # ------------------------------------------------------------------
    # Token stats (for observability, pool_stats_observer)
    # ------------------------------------------------------------------

    def get_token_stats(self) -> HiSparseTokenStats:
        """Return HiSparse token usage stats compatible with DSA interface.

        M3 has no per-request device buffer, so ``device_tokens`` reflects
        the standard GPU pool usage (dense_main + sparse_index_k).  Host
        tokens are tracked by the bump allocator in the host pool.
        """
        allocator = self.token_to_kv_pool_allocator
        device_capacity = allocator.size
        device_tokens = device_capacity - allocator.available_size()

        host_pool = self.token_to_kv_pool.sparse_main_host_pool
        host_capacity = host_pool.alloc_size
        host_tokens = host_capacity - host_pool.free_slots

        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    def get_host_usage(self) -> tuple[int, int]:
        """Return (used_host_slots, total_host_slots)."""
        host_pool = self.token_to_kv_pool.sparse_main_host_pool
        used = host_pool.alloc_size - host_pool.free_slots
        return used, host_pool.alloc_size
