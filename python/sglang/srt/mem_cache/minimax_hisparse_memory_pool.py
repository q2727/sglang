from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MHATokenToKOnlyPool,
    MHATokenToKVPool,
    get_tensor_size_bytes,
    unwrap_write_loc,
)

logger = logging.getLogger(__name__)


@dataclass
class MiniMaxHiSparseLoadResult:
    host_locs: torch.Tensor
    hot_locs: torch.Tensor
    hot_page_table: Optional[torch.Tensor] = None
    hot_kv_indices: Optional[torch.Tensor] = None


def _round_up_to_page_size(size: int, page_size: int) -> int:
    return (size + page_size - 1) // page_size * page_size


def _as_cpu_long(indices: torch.Tensor) -> torch.Tensor:
    return indices.detach().to(device="cpu", dtype=torch.long)


def _check_indices(name: str, indices: torch.Tensor, limit: int) -> None:
    if indices.numel() == 0:
        return
    min_idx = int(indices.min().item())
    max_idx = int(indices.max().item())
    if min_idx < 0 or max_idx >= limit:
        raise IndexError(
            f"{name} out of range: min={min_idx}, max={max_idx}, valid=[0, {limit})"
        )


class MiniMaxSparseMainHostPool:
    """CPU backing store for MiniMax sparse-layer main K/V.

    This pool is intentionally narrow: it stores only sparse main K/V, while
    dense main K/V and sparse index K stay in GPU-resident pools.
    """

    def __init__(
        self,
        *,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        pin_memory: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.device = "cpu"
        self.pin_memory = pin_memory
        self.k_buffer = [self._alloc_buffer() for _ in range(layer_num)]
        self.v_buffer = [self._alloc_buffer() for _ in range(layer_num)]

    @property
    def alloc_size(self) -> int:
        return self.size + self.page_size

    def _alloc_buffer(self) -> torch.Tensor:
        shape = (self.alloc_size, self.head_num, self.head_dim)
        try:
            return torch.zeros(
                shape,
                dtype=self.dtype,
                device="cpu",
                pin_memory=self.pin_memory,
            )
        except RuntimeError:
            if self.pin_memory:
                logger.warning(
                    "MiniMaxHiSparse host pin_memory allocation failed; "
                    "falling back to pageable CPU memory."
                )
                self.pin_memory = False
            return torch.zeros(shape, dtype=self.dtype, device="cpu")

    def get_kv_buffer(self, local_layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_buffer[local_layer_id], self.v_buffer[local_layer_id]

    def set_kv_rows(
        self,
        local_layer_id: int,
        host_locs: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        host_locs_cpu = _as_cpu_long(host_locs)
        _check_indices("host_locs", host_locs_cpu, self.alloc_size)
        if cache_k.shape[0] != host_locs_cpu.numel():
            raise ValueError(
                f"cache_k first dim ({cache_k.shape[0]}) must match "
                f"host_locs length ({host_locs_cpu.numel()})"
            )
        if cache_v.shape[0] != host_locs_cpu.numel():
            raise ValueError(
                f"cache_v first dim ({cache_v.shape[0]}) must match "
                f"host_locs length ({host_locs_cpu.numel()})"
            )
        k_cpu = cache_k.detach().to(
            device="cpu", dtype=self.dtype, non_blocking=self.pin_memory
        )
        v_cpu = cache_v.detach().to(
            device="cpu", dtype=self.dtype, non_blocking=self.pin_memory
        )
        self.k_buffer[local_layer_id][host_locs_cpu] = k_cpu
        self.v_buffer[local_layer_id][host_locs_cpu] = v_cpu

    def get_kv_rows(
        self,
        local_layer_id: int,
        host_locs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        host_locs_cpu = _as_cpu_long(host_locs)
        _check_indices("host_locs", host_locs_cpu, self.alloc_size)
        return (
            self.k_buffer[local_layer_id][host_locs_cpu],
            self.v_buffer[local_layer_id][host_locs_cpu],
        )

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return (
            sum(get_tensor_size_bytes(k) for k in self.k_buffer),
            sum(get_tensor_size_bytes(v) for v in self.v_buffer),
        )


class MiniMaxHiSparseKVPool(KVCache):
    """Experimental MiniMax-M3 HiSparse KV pool.

    Layout:
    - dense_main_pool: full GPU K/V for dense layers.
    - sparse_index_k_pool: full GPU index K for sparse layers.
    - sparse_main_hot_pool: small GPU K/V buffer for selected sparse blocks.
    - sparse_main_host_pool: CPU backing store for sparse main K/V.

    The pool deliberately does not allocate sparse index V; the first target is
    the BF16 MiniMax-M3 K-only sparse-index configuration.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        idx_head_dim: int,
        dense_layer_ids: list[int],
        sparse_layer_ids: list[int],
        device: str,
        *,
        hot_size: Optional[int] = None,
        host_to_device_ratio: float = 2.0,
        disable_value_sparse_layer_ids: Optional[list[int]] = None,
        enable_memory_saver: bool = False,
        index_dtype: Optional[torch.dtype] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        pin_host_memory: Optional[bool] = None,
    ):
        all_layer_ids = dense_layer_ids + sparse_layer_ids
        if start_layer is None:
            start_layer = min(all_layer_ids) if all_layer_ids else 0
        if end_layer is None:
            end_layer = max(all_layer_ids) + 1 if all_layer_ids else start_layer

        local_dense_layer_ids = [
            lid for lid in dense_layer_ids if start_layer <= lid < end_layer
        ]
        local_sparse_layer_ids = [
            lid for lid in sparse_layer_ids if start_layer <= lid < end_layer
        ]

        disable_set = set(disable_value_sparse_layer_ids or [])
        unsupported_index_v_layers = [
            lid for lid in local_sparse_layer_ids if lid not in disable_set
        ]
        if unsupported_index_v_layers:
            raise NotImplementedError(
                "MiniMaxHiSparseKVPool only supports K-only sparse index layers; "
                f"layers requiring index V: {unsupported_index_v_layers}"
            )

        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=len(local_dense_layer_ids) + len(local_sparse_layer_ids),
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        self.head_num = head_num
        self.head_dim = head_dim
        self.idx_head_dim = idx_head_dim
        self.index_dtype = index_dtype if index_dtype is not None else dtype
        self.local_dense_layer_ids = local_dense_layer_ids
        self.local_sparse_layer_ids = local_sparse_layer_ids
        self.dense_layer_id_mapping = {
            gid: i for i, gid in enumerate(local_dense_layer_ids)
        }
        self.sparse_layer_id_mapping = {
            gid: i for i, gid in enumerate(local_sparse_layer_ids)
        }
        self.index_k_layer_id_mapping = dict(self.sparse_layer_id_mapping)
        self.index_kv_layer_id_mapping: dict[int, int] = {}

        hot_size = page_size if hot_size is None else hot_size
        self.hot_size = _round_up_to_page_size(int(hot_size), page_size)
        if self.hot_size <= 0:
            raise ValueError(f"hot_size must be positive, got {hot_size}")
        self.hot_page_offset = 1
        self.hot_page_capacity = self.hot_size // page_size

        host_size = max(int(size * host_to_device_ratio), size)
        self.host_size = _round_up_to_page_size(host_size, page_size)
        if pin_host_memory is None:
            pin_host_memory = torch.device(device).type in ("cuda", "hip")

        self.dense_main_pool = MHATokenToKVPool(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=len(local_dense_layer_ids),
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=0,
            end_layer=len(local_dense_layer_ids),
        )
        self.sparse_index_k_pool = MHATokenToKOnlyPool(
            size=size,
            page_size=page_size,
            dtype=self.index_dtype,
            head_num=1,
            head_dim=idx_head_dim,
            layer_num=len(local_sparse_layer_ids),
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=0,
            end_layer=len(local_sparse_layer_ids),
        )
        self.sparse_main_hot_pool = MHATokenToKVPool(
            size=self.hot_size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=len(local_sparse_layer_ids),
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=0,
            end_layer=len(local_sparse_layer_ids),
        )
        self.sparse_main_host_pool = MiniMaxSparseMainHostPool(
            size=self.host_size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=len(local_sparse_layer_ids),
            pin_memory=pin_host_memory,
        )

        # Compatibility attributes used by the existing MiniMax sparse backend.
        self.main_pool = self.dense_main_pool
        self.index_k_pool = self.sparse_index_k_pool
        self.index_kv_pool = None

        self.mem_usage = (
            self.dense_main_pool.mem_usage
            + self.sparse_index_k_pool.mem_usage
            + self.sparse_main_hot_pool.mem_usage
        )
        self._hot_page_table_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.full_to_hisparse_device_index_mapping: Optional[torch.Tensor] = None

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_to_hisparse_device(self, full_indices: torch.Tensor):
        if self.full_to_hisparse_device_index_mapping is None:
            return full_indices
        return self.full_to_hisparse_device_index_mapping[full_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self.translate_loc_to_hisparse_device(full_indices)

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        return full_indices

    def _wait_for_layer(self, layer_id: int) -> None:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def _dense_local_layer_id(self, layer_id: int) -> int:
        mapped_id = self.dense_layer_id_mapping.get(layer_id)
        if mapped_id is None:
            raise ValueError(
                f"layer_id={layer_id} is not a dense MiniMax layer; "
                f"dense layers: {list(self.dense_layer_id_mapping.keys())}"
            )
        return mapped_id

    def _sparse_local_layer_id(self, layer_id: int) -> int:
        mapped_id = self.sparse_layer_id_mapping.get(layer_id)
        if mapped_id is None:
            raise ValueError(
                f"layer_id={layer_id} is not a sparse MiniMax layer; "
                f"sparse layers: {list(self.sparse_layer_id_mapping.keys())}"
            )
        return mapped_id

    def _is_dense_layer(self, layer_id: int) -> bool:
        return layer_id in self.dense_layer_id_mapping

    def _is_sparse_layer(self, layer_id: int) -> bool:
        return layer_id in self.sparse_layer_id_mapping

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        if self._is_dense_layer(layer_id):
            return self.dense_main_pool.get_key_buffer(
                self._dense_local_layer_id(layer_id)
            )
        return self.get_hot_kv_buffer(layer_id)[0]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        if self._is_dense_layer(layer_id):
            return self.dense_main_pool.get_value_buffer(
                self._dense_local_layer_id(layer_id)
            )
        return self.get_hot_kv_buffer(layer_id)[1]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._wait_for_layer(layer_id)
        if self._is_dense_layer(layer_id):
            return self.dense_main_pool.get_kv_buffer(
                self._dense_local_layer_id(layer_id)
            )
        if self._is_sparse_layer(layer_id):
            return self.get_hot_kv_buffer(layer_id)
        raise ValueError(
            f"layer_id={layer_id} is not managed by MiniMaxHiSparseKVPool"
        )

    def get_hot_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mapped_id = self._sparse_local_layer_id(layer_id)
        return self.sparse_main_hot_pool.get_kv_buffer(mapped_id)

    def get_sparse_main_host_kv_buffer(
        self, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mapped_id = self._sparse_local_layer_id(layer_id)
        return self.sparse_main_host_pool.get_kv_buffer(mapped_id)

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        mapped_id = self._sparse_local_layer_id(layer_id)
        return self.sparse_index_k_pool.get_key_buffer(mapped_id)

    def get_index_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "MiniMaxHiSparseKVPool does not allocate index V for K-only MiniMax-M3 "
            f"sparse layers; requested layer_id={layer_id}."
        )

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        loc, _ = unwrap_write_loc(loc)
        if self._is_dense_layer(layer.layer_id):
            self.dense_main_pool.set_kv_buffer(
                layer,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=self._dense_local_layer_id(layer.layer_id),
            )
            return
        if self._is_sparse_layer(layer.layer_id):
            self.backup_sparse_main_to_host(
                layer.layer_id,
                loc,
                cache_k=cache_k,
                cache_v=cache_v,
            )
            return
        raise ValueError(
            f"layer_id={layer.layer_id} is not managed by MiniMaxHiSparseKVPool"
        )

    def set_index_k_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_idx_k: torch.Tensor,
    ) -> None:
        loc, _ = unwrap_write_loc(loc)
        mapped_id = self._sparse_local_layer_id(layer.layer_id)
        sub_pool = self.sparse_index_k_pool
        if cache_idx_k.dtype != sub_pool.dtype:
            cache_idx_k = cache_idx_k.to(sub_pool.dtype)
        if sub_pool.store_dtype != sub_pool.dtype:
            cache_idx_k = cache_idx_k.view(sub_pool.store_dtype)
        sub_pool.k_buffer[mapped_id][loc] = cache_idx_k

    def set_index_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_idx_k: torch.Tensor,
        cache_idx_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        raise NotImplementedError(
            "MiniMaxHiSparseKVPool does not allocate sparse index V; "
            "use set_index_k_buffer for MiniMax-M3 K-only sparse layers."
        )

    def set_fused_kv_index_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_idx_k: torch.Tensor,
        cache_idx_v: Optional[torch.Tensor],
    ) -> None:
        if cache_idx_v is not None:
            raise NotImplementedError(
                "MiniMaxHiSparseKVPool only supports K-only sparse index layers; "
                "cache_idx_v must be None."
            )
        if not self._is_sparse_layer(layer.layer_id):
            raise ValueError(
                "set_fused_kv_index_buffer is only valid for sparse MiniMax layers; "
                f"got layer_id={layer.layer_id}."
            )
        self.backup_sparse_main_to_host(
            layer.layer_id,
            loc,
            cache_k=cache_k,
            cache_v=cache_v,
        )
        self.set_index_k_buffer(layer, loc, cache_idx_k)

    def backup_sparse_main_to_host(
        self,
        layer_id: int,
        host_locs: torch.Tensor,
        *,
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
        hot_locs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mapped_id = self._sparse_local_layer_id(layer_id)
        if cache_k is None or cache_v is None:
            if hot_locs is None:
                raise ValueError(
                    "backup_sparse_main_to_host requires either cache_k/cache_v "
                    "or hot_locs."
                )
            hot_k, hot_v = self.get_hot_kv_buffer(layer_id)
            hot_locs_device = hot_locs.to(device=hot_k.device, dtype=torch.long)
            _check_indices("hot_locs", _as_cpu_long(hot_locs), hot_k.shape[0])
            cache_k = hot_k[hot_locs_device]
            cache_v = hot_v[hot_locs_device]

        self.sparse_main_host_pool.set_kv_rows(
            mapped_id,
            host_locs,
            cache_k,
            cache_v,
        )
        return _as_cpu_long(host_locs)

    def _load_sparse_main_locs_to_hot(
        self,
        layer_id: int,
        host_locs: torch.Tensor,
        hot_locs: torch.Tensor,
    ) -> MiniMaxHiSparseLoadResult:
        mapped_id = self._sparse_local_layer_id(layer_id)
        host_locs_cpu = _as_cpu_long(host_locs)
        hot_k, hot_v = self.get_hot_kv_buffer(layer_id)
        hot_locs_device = hot_locs.to(device=hot_k.device, dtype=torch.long)
        _check_indices("host_locs", host_locs_cpu, self.sparse_main_host_pool.alloc_size)
        _check_indices("hot_locs", _as_cpu_long(hot_locs), hot_k.shape[0])
        if host_locs_cpu.numel() != hot_locs_device.numel():
            raise ValueError(
                f"host_locs length ({host_locs_cpu.numel()}) must match "
                f"hot_locs length ({hot_locs_device.numel()})"
            )

        k_cpu, v_cpu = self.sparse_main_host_pool.get_kv_rows(
            mapped_id, host_locs_cpu
        )
        hot_k[hot_locs_device] = k_cpu.to(
            device=hot_k.device,
            dtype=hot_k.dtype,
            non_blocking=self.sparse_main_host_pool.pin_memory,
        )
        hot_v[hot_locs_device] = v_cpu.to(
            device=hot_v.device,
            dtype=hot_v.dtype,
            non_blocking=self.sparse_main_host_pool.pin_memory,
        )
        return MiniMaxHiSparseLoadResult(
            host_locs=host_locs_cpu,
            hot_locs=hot_locs_device,
        )

    def load_sparse_main_blocks_to_hot(
        self,
        layer_id: int,
        host_locs: Optional[torch.Tensor] = None,
        hot_locs: Optional[torch.Tensor] = None,
        *,
        req_to_host: Optional[torch.Tensor] = None,
        req_pool_indices: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        block_size: Optional[int] = None,
    ) -> MiniMaxHiSparseLoadResult:
        """Load sparse main K/V rows or selected logical blocks into hot GPU K/V.

        Direct mode passes host_locs and hot_locs. Block mode passes req_to_host,
        req_pool_indices, topk_idx, and seq_lens; topk_idx values remain logical
        block ids and are converted to a layer-scoped hot page table.
        """
        if req_to_host is None:
            if host_locs is None:
                raise ValueError("host_locs is required in direct load mode.")
            if hot_locs is None:
                hot_locs = torch.arange(
                    host_locs.numel(), dtype=torch.long, device=self.device
                )
            return self._load_sparse_main_locs_to_hot(layer_id, host_locs, hot_locs)

        if (
            req_pool_indices is None
            or topk_idx is None
            or seq_lens is None
        ):
            raise ValueError(
                "Block load mode requires req_pool_indices, topk_idx, and seq_lens."
            )
        block_size = self.page_size if block_size is None else block_size
        if block_size != self.page_size:
            raise ValueError(
                f"MiniMax HiSparse hot pages require block_size == page_size; "
                f"got block_size={block_size}, page_size={self.page_size}."
            )
        if topk_idx.ndim != 3:
            raise ValueError(f"topk_idx must be rank-3, got shape={topk_idx.shape}.")

        topk_cpu = topk_idx.detach().to(device="cpu", dtype=torch.int64)
        req_to_host_cpu = req_to_host.detach().to(device="cpu", dtype=torch.long)
        req_pool_cpu = req_pool_indices.detach().to(device="cpu", dtype=torch.long)
        seq_lens_cpu = seq_lens.detach().to(device="cpu", dtype=torch.long)
        batch = int(req_pool_cpu.numel())
        if topk_cpu.shape[1] == batch:
            topk_by_batch = [topk_cpu[:, b, :] for b in range(batch)]
        elif topk_cpu.shape[0] == batch:
            topk_by_batch = [topk_cpu[b, :, :] for b in range(batch)]
        else:
            raise ValueError(
                "topk_idx must be [Hkv, B, K] or [B, Hkv, K]; "
                f"got shape={tuple(topk_idx.shape)} for B={batch}."
            )

        max_pages = int(((seq_lens_cpu.max().item() + block_size - 1) // block_size))
        hot_page_table_cpu = torch.zeros((batch, max_pages), dtype=torch.int32)
        all_host_locs: list[torch.Tensor] = []
        all_hot_locs: list[torch.Tensor] = []
        next_hot_page = 0

        hot_k, hot_v = self.get_hot_kv_buffer(layer_id)
        for batch_id, per_head_blocks in enumerate(topk_by_batch):
            seq_len = int(seq_lens_cpu[batch_id].item())
            num_pages = (seq_len + block_size - 1) // block_size
            req_row = int(req_pool_cpu[batch_id].item())
            selected_blocks: list[int] = []
            seen_blocks: set[int] = set()
            for block_id in per_head_blocks.reshape(-1).tolist():
                if block_id < 0:
                    continue
                if block_id >= num_pages:
                    raise ValueError(
                        f"topk block id {block_id} is outside seq_len={seq_len} "
                        f"(num_pages={num_pages}) for batch index {batch_id}."
                    )
                if block_id not in seen_blocks:
                    selected_blocks.append(block_id)
                    seen_blocks.add(block_id)

            for block_id in selected_blocks:
                if next_hot_page >= self.hot_page_capacity:
                    raise RuntimeError(
                        "MiniMaxHiSparse hot buffer exhausted while loading "
                        f"selected sparse blocks: needed page {next_hot_page + 1}, "
                        f"capacity={self.hot_page_capacity}."
                    )
                start = block_id * block_size
                end = min(start + block_size, seq_len)
                host_block_locs = req_to_host_cpu[req_row, start:end]
                if (host_block_locs < 0).any():
                    raise ValueError(
                        f"req_to_host contains negative locs for req_row={req_row}, "
                        f"logical block={block_id}."
                    )

                hot_page_id = self.hot_page_offset + next_hot_page
                hot_start = hot_page_id * self.page_size
                hot_end = hot_start + (end - start)
                hot_block_locs = torch.arange(hot_start, hot_end, dtype=torch.long)

                page_start = hot_page_id * self.page_size
                page_end = page_start + self.page_size
                hot_k[page_start:page_end].zero_()
                hot_v[page_start:page_end].zero_()
                self._load_sparse_main_locs_to_hot(
                    layer_id,
                    host_block_locs,
                    hot_block_locs,
                )

                hot_page_table_cpu[batch_id, block_id] = hot_page_id
                all_host_locs.append(host_block_locs)
                all_hot_locs.append(hot_block_locs)
                next_hot_page += 1

        hot_kv_indices_cpu = torch.cat(
            [
                hot_page_table_cpu[
                    b, : (int(seq_lens_cpu[b].item()) + block_size - 1) // block_size
                ]
                for b in range(batch)
            ],
            dim=0,
        )
        page_table_device = hot_page_table_cpu.to(
            device=topk_idx.device, non_blocking=False
        )
        kv_indices_device = hot_kv_indices_cpu.to(
            device=topk_idx.device, non_blocking=False
        )
        self._hot_page_table_by_layer[layer_id] = (
            kv_indices_device,
            page_table_device,
        )

        empty = torch.empty((0,), dtype=torch.long)
        return MiniMaxHiSparseLoadResult(
            host_locs=torch.cat(all_host_locs, dim=0) if all_host_locs else empty,
            hot_locs=(
                torch.cat(all_hot_locs, dim=0).to(device=hot_k.device)
                if all_hot_locs
                else empty.to(device=hot_k.device)
            ),
            hot_page_table=page_table_device,
            hot_kv_indices=kv_indices_device,
        )

    def get_hot_page_table(
        self, layer_id: int, *, flattened: bool = True
    ) -> torch.Tensor:
        entry = self._hot_page_table_by_layer.get(layer_id)
        if entry is None:
            raise RuntimeError(
                f"No hot page table has been built for sparse layer {layer_id}."
            )
        hot_kv_indices, hot_page_table = entry
        return hot_kv_indices if flattened else hot_page_table

    def get_kv_size_bytes(self):
        sub_pools = [
            self.dense_main_pool,
            self.sparse_index_k_pool,
            self.sparse_main_hot_pool,
            self.sparse_main_host_pool,
        ]
        sizes = [p.get_kv_size_bytes() for p in sub_pools]
        return sum(k for k, _ in sizes), sum(v for _, v in sizes)

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "PD disaggregation for MiniMaxHiSparseKVPool is not implemented; "
            "sparse main K/V uses host/hot storage and requires a scoped transfer "
            "contract."
        )

    def get_index_k_state_buf_infos(self):
        pool = self.sparse_index_k_pool
        n = pool.layer_num
        data_ptrs = [pool.k_buffer[i].data_ptr() for i in range(n)]
        data_lens = [pool.k_buffer[i].nbytes for i in range(n)]
        item_lens = [pool.k_buffer[i][0].nbytes * pool.page_size for i in range(n)]
        return data_ptrs, data_lens, item_lens

    def maybe_get_custom_mem_pool(self):
        return self.sparse_main_hot_pool.maybe_get_custom_mem_pool()

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        raise NotImplementedError(
            "move_kv_cache is not supported for MiniMaxHiSparseKVPool yet."
        )

    def get_v_head_dim(self):
        return self.head_dim
