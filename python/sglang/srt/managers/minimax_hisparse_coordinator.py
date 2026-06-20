"""MiniMax-M3 HiSparse coordinator (Plan B: two-pool design).

Manages the request lifecycle across two KV pools:

- **standard_kv_pool** (MiniMaxSparseKVPool): authoritative GPU storage for
  prefill, dense K/V, and sparse index K.  After prefill, sparse main K/V is
  backed up from here into the HiSparse host pool.
- **hisparse_kv_pool** (MiniMaxHiSparseKVPool): hot GPU buffer + host backup
  for sparse-layer main K/V during decode.

Data flow::

    prefill (standard pool)
      → backup sparse main K/V to hisparse host
      → decode (standard pool: dense/index K; hisparse: hot sparse main K/V)

No staging queue, no async DMA, no per-request device buffer, no compress_ratio.


GPU-resident tensors for Agent D JIT kernel
--------------------------------------------

``req_to_host`` lives on GPU so Agent D's block-swap kernel can read it
directly without a D2H copy.  Pre-allocated fixed-shape buffers mirror
DSA's ``top_k_device_locs_buffer`` / ``raw_indices_buffer`` pattern and
are graph-safe:

- ``hot_page_table_buffer``: [max_reqs, max_pages_per_req] int32
- ``hot_kv_indices_buffer``: [max_reqs * max_pages_per_req] int32
- ``host_locs_buffer``: [max_reqs * max_pages_per_req * block_size] int64
- ``hot_locs_buffer``:  same shape as host_locs_buffer

These buffers are *not* layer-scoped (unlike DSA's per-layer tensors)
because M3 has a per-layer global hot buffer; the kernel writes into
them and the attention backend reads them for that layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseTokenStats
from sglang.srt.utils import get_device_module

device_module = get_device_module()

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.minimax_hisparse_memory_pool import MiniMaxHiSparseKVPool
    from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool, ReqToTokenPool

logger = logging.getLogger(__name__)


class MiniMaxHiSparseCoordinator:
    """Coordinator for MiniMax-M3 HiSparse with two-pool architecture.

    Owns ``req_to_host``: (req_pool_idx, logical_token_pos) → host pool slot.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: PagedTokenToKVPoolAllocator,
        standard_kv_pool: MiniMaxSparseKVPool,
        hisparse_kv_pool: MiniMaxHiSparseKVPool,
        device: str,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.standard_kv_pool = standard_kv_pool
        self.hisparse_kv_pool = hisparse_kv_pool
        self.device = device

        self._sparse_layer_ids: list[int] = list(
            hisparse_kv_pool.local_sparse_layer_ids
        )

        max_num_reqs = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        block_size = hisparse_kv_pool.page_size
        max_pages_per_req = (max_context_len + block_size - 1) // block_size

        # Primary mapping: (req_pool_idx, logical_token_pos) → host pool slot.
        # GPU-resident so Agent D's JIT block-swap kernel can read it directly.
        self.req_to_host = torch.full(
            (max_num_reqs, max_context_len),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.req_to_host_allocated_len = torch.zeros(
            max_num_reqs, dtype=torch.int64, device="cpu"
        )

        # ------------------------------------------------------------------
        # Graph-safe pre-allocated buffers (M3 equivalents of DSA's
        # top_k_device_locs_buffer / raw_indices_buffer).
        # Agent D's block-swap JIT kernel writes into these each decode step.
        # ------------------------------------------------------------------
        max_selected_tokens = max_num_reqs * max_pages_per_req * block_size
        self.hot_page_table_buffer = torch.full(
            (max_num_reqs, max_pages_per_req),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.hot_kv_indices_buffer = torch.full(
            (max_num_reqs * max_pages_per_req,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.host_locs_buffer = torch.full(
            (max_selected_tokens,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.hot_locs_buffer = torch.full(
            (max_selected_tokens,),
            -1,
            dtype=torch.int64,
            device=device,
        )

        # Scalar: number of real (non-padded) requests, updated before graph replay.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # Async backup stream + sync event (mirrors DSA's decode_backup_stream pattern)
        self.decode_backup_stream = device_module.Stream()
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False
        self._decode_producer_stream: Optional[device_module.Stream] = None

    # ------------------------------------------------------------------
    # Host slot allocation
    # ------------------------------------------------------------------

    def _alloc_host_slots(self, num: int) -> torch.Tensor:
        host_pool = self.hisparse_kv_pool.sparse_main_host_pool
        return host_pool.alloc(num)

    def _free_host_slots(self, indices: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        host_pool = self.hisparse_kv_pool.sparse_main_host_pool
        host_pool.free(indices)

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def admit_prefill(self, req: Req) -> None:
        """After prefill: backup sparse main K/V to hisparse host pool.

        Prefill writes all K/V (dense + sparse main + sparse index) into the
        standard pool on GPU.  This method:
        1. Allocates host slots in the hisparse host pool.
        2. For each sparse layer, reads main K/V from the standard pool and
           backs it up to the hisparse host pool via the bridge method
           ``backup_sparse_main_from_standard_pool``.
        3. Writes the host slot mapping into ``req_to_host``.

        After this call the request is decode-ready.  Dense K/V and sparse index
        K remain in the standard pool; sparse main K/V lives in the hisparse
        host pool (with a per-step hot GPU buffer during decode).
        """
        prefill_len = req.fill_len
        if prefill_len <= 0:
            return

        # 1. Allocate host slots (CPU), write to GPU req_to_host
        host_locs_cpu = self._alloc_host_slots(prefill_len)
        host_locs = host_locs_cpu.to(device=self.device, dtype=torch.int64)
        self.req_to_host[req.req_pool_idx, :prefill_len] = host_locs
        self.req_to_host_allocated_len[req.req_pool_idx] = prefill_len

        # 2. GPU slot indices for this request (standard pool)
        gpu_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefill_len
        ]

        # 3. Backup sparse main K/V from standard pool → hisparse host
        #    Pass CPU host_locs to the bridge (it writes to CPU host pool).
        for layer_id in self._sparse_layer_ids:
            k_cache, v_cache = self.standard_kv_pool.get_kv_buffer(layer_id)
            self.hisparse_kv_pool.backup_sparse_main_from_standard_pool(
                layer_id=layer_id,
                host_locs=host_locs_cpu,
                standard_k_cache=k_cache,
                standard_v_cache=v_cache,
                standard_indices=gpu_indices,
            )

        logger.debug(
            "MiniMaxHiSparse: admitted prefill req %s (len=%d, "
            "backed up %d sparse layers to host)",
            req.rid,
            prefill_len,
            len(self._sparse_layer_ids),
        )

    def admit_request_into_staging(self, req: Req) -> None:
        """Scheduler-compatible entry point.  Delegates to admit_prefill."""
        self.admit_prefill(req)
        req.hisparse_staging = False

    def extend_decode(self, req: Req) -> None:
        """Allocate one host slot for the new decode token."""
        seq_len = req.kv_allocated_len
        if seq_len <= 0:
            return

        max_ctx = self.req_to_host.shape[1]
        if seq_len > max_ctx:
            raise RuntimeError(
                f"MiniMaxHiSparse: req {req.rid} seq_len={seq_len} exceeds "
                f"max_context_len={max_ctx}."
            )

        current_allocated = int(
            self.req_to_host_allocated_len[req.req_pool_idx]
        )
        if seq_len <= current_allocated:
            return

        num_new = seq_len - current_allocated
        host_locs_cpu = self._alloc_host_slots(num_new)
        host_locs = host_locs_cpu.to(device=self.device, dtype=torch.int64)
        self.req_to_host[
            req.req_pool_idx, current_allocated:seq_len
        ] = host_locs
        self.req_to_host_allocated_len[req.req_pool_idx] = seq_len

    def request_finished(self, req: Req) -> None:
        """Release both standard pool GPU slots and hisparse host pool slots."""
        allocated_len = int(
            self.req_to_host_allocated_len[req.req_pool_idx]
        )

        # 1. Free standard pool GPU slots
        if allocated_len > 0:
            gpu_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :allocated_len
            ]
            if gpu_slots.numel() > 0:
                self.token_to_kv_pool_allocator.free(gpu_slots)

        # 2. Free hisparse host pool slots (GPU → CPU for host pool free)
        if allocated_len > 0:
            host_locs = self.req_to_host[
                req.req_pool_idx, :allocated_len
            ].to(device="cpu", copy=True)
            self._free_host_slots(host_locs)

        # 3. Clear bookkeeping
        self.req_to_host[req.req_pool_idx, :] = -1
        self.req_to_host_allocated_len[req.req_pool_idx] = 0

        logger.debug(
            "MiniMaxHiSparse: finished req %s (freed %d GPU + %d host slots)",
            req.rid,
            allocated_len,
            allocated_len,
        )

    def retract_req(self, req: Req) -> None:
        """Abort a request, releasing all resources."""
        self.request_finished(req)

    # ------------------------------------------------------------------
    # Staging interface (M3 has no staging)
    # ------------------------------------------------------------------

    def has_ongoing_staging(self) -> bool:
        return False

    def collect_ready_reqs(self) -> list:
        return []

    # ------------------------------------------------------------------
    # Stream management (mirrors DSA's async backup pattern)
    # ------------------------------------------------------------------

    def set_decode_producer_stream(self, stream) -> None:
        self._decode_producer_stream = stream

    def wait_for_pending_backup(self) -> None:
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False

    # ------------------------------------------------------------------
    # Per-step decode backup (called outside CUDA graph, before graph replay)
    # ------------------------------------------------------------------

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """No-op: per-step backup is in ``async_backup_previous_token``.

        Host slots are allocated by ``extend_decode`` and graph-internal
        backup is handled by Agent E's attention backend.  Async backup
        from the standard pool to the hisparse host pool is triggered by
        ``async_backup_previous_token`` before graph replay.
        """
        pass

    def async_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Async backup of the previous decode token's sparse main K/V.

        Copies sparse main K/V at logical position ``seq_len - 2`` from
        the **standard pool** (GPU) to the **hisparse host pool** (CPU) on
        ``decode_backup_stream``.  The backup overlaps with the main-stream
        forward pass.

        Called outside the CUDA graph — ``prepare_for_graph_replay``
        triggers it before each graph replay.

        References DSA ``_eager_backup_previous_token`` for the async
        stream layering pattern, but is simpler: no compress_ratio, no
        per-request device buffer, no staged skip logic.
        """
        # Identify requests that have a previous token to backup.
        # seq_lens has already been incremented by prepare_for_decode;
        # the token at position seq_len - 2 was stored in the last step.
        backup_indices = []
        prev_positions_list = []
        for i in range(len(seq_lens_cpu)):
            seq_len = int(seq_lens_cpu[i])
            prev_pos = seq_len - 2
            if prev_pos < 0:
                continue
            req_idx = int(req_pool_indices_cpu[i])
            if prev_pos >= int(self.req_to_host_allocated_len[req_idx]):
                continue
            backup_indices.append(i)
            prev_positions_list.append(prev_pos)

        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
        backup_req_indices = req_pool_indices[backup_indices_gpu]
        prev_positions = torch.tensor(
            prev_positions_list, dtype=torch.int64, device=self.device
        )

        # GPU slot in standard pool + host slot for the previous token
        gpu_slots = self.req_to_token_pool.req_to_token[
            backup_req_indices, prev_positions
        ]
        host_locs = self.req_to_host[backup_req_indices, prev_positions]

        # Wait for in-flight backup, then submit async
        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self._decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(
                    self._decode_producer_stream
                )
            for layer_id in self._sparse_layer_ids:
                k_cache, v_cache = self.standard_kv_pool.get_kv_buffer(
                    layer_id
                )
                self.hisparse_kv_pool.backup_sparse_main_from_standard_pool(
                    layer_id=layer_id,
                    host_locs=host_locs,
                    standard_k_cache=k_cache,
                    standard_v_cache=v_cache,
                    standard_indices=gpu_slots,
                )
            self._backup_done_event.record()
            if gpu_slots.is_cuda:
                gpu_slots.record_stream(self.decode_backup_stream)
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    # ------------------------------------------------------------------
    # Graph replay hook (called by model_runner before CUDA graph replay)
    # ------------------------------------------------------------------

    def prepare_for_graph_replay(self, forward_batch) -> None:
        """Update per-batch GPU state before CUDA graph replay.

        1. Triggers ``async_backup_previous_token`` to back up the
           previous token from standard pool → hisparse host on the
           async backup stream (NOT captured in the graph).
        2. Blocks until the async backup completes.
        3. Sets ``num_real_reqs`` so graph-captured kernels skip
           padded batch entries.
        """
        self.async_backup_previous_token(
            seq_lens=forward_batch.seq_lens,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            req_pool_indices_cpu=forward_batch.req_pool_indices_cpu,
        )
        self.wait_for_pending_backup()
        self.num_real_reqs.fill_(forward_batch.batch_size)

    # ------------------------------------------------------------------
    # Token stats
    # ------------------------------------------------------------------

    def get_token_stats(self) -> HiSparseTokenStats:
        allocator = self.token_to_kv_pool_allocator
        device_capacity = allocator.size
        device_tokens = device_capacity - allocator.available_size()

        host_pool = self.hisparse_kv_pool.sparse_main_host_pool
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
        host_pool = self.hisparse_kv_pool.sparse_main_host_pool
        used = host_pool.alloc_size - host_pool.free_slots
        return used, host_pool.alloc_size
