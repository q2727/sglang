"""CUDA IPC P2P data plane for decoupled enumeration blocks (NVLink).

Replaces the wire for the drafter -> verifier token block only (the control
plane stays on ZMQ): the drafter writes each round's block into a slot of a
preallocated GPU ring pool; the verifier maps the pool once through its CUDA
IPC handle (peer access crosses the GPUs over NVLink) and scatters rows
straight from the mapped slice into its landing buffer -- no D2H, no
pickling, no H2D. Modeled on utils/cuda_ipc_transport_utils.py (pool + IPC
handle + shm sync flags), specialized to fixed-shape one-round-ahead blocks.

Rendezvous + synchronization ride on one POSIX shared-memory segment whose
name is derived from the drafter's bind endpoint (both sides know it):

    header: magic, layout dims, pickled CUDA IPC handle
    per slot: ready_seq | consumed_seq | batch_size

The drafter writes a block (D2D), synchronizes its device, then bumps
ready_seq: shm visibility after the sync IS the cross-process arrival signal
(the C6 latch). The verifier's reader polls ready_seq != consumed_seq,
consumes the slot on its own stream, synchronizes, then bumps consumed_seq
so the drafter may reuse the slot (the cross-process WAR guard).

Each row is ``[pool_idx, base_committed_len, unit tokens ...]``: routing and
staleness stamps travel on the GPU with the data, so landing needs no host
rid lookup; the sync-mode arrival board takes one tiny D2H per block.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
from multiprocessing import shared_memory
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_MAGIC = 0x53474C444543  # "SGLDEC"
_HEADER_INT64S = 8  # magic, device_index, num_slots, max_rows, row_width, 3 spare
_HANDLE_BYTES = 4096  # pickled _share_cuda_() tuple (a few hundred bytes)
_FLAG_FIELDS = 3  # ready_seq, consumed_seq, batch_size

DEFAULT_NUM_SLOTS = 4


def _shm_name(endpoint: str) -> str:
    digest = hashlib.sha1(endpoint.encode()).hexdigest()[:16]
    return f"sglang_decoupled_enum_{digest}"


def _shm_size(num_slots: int) -> int:
    return _HEADER_INT64S * 8 + _HANDLE_BYTES + num_slots * _FLAG_FIELDS * 8


def _flags_view(shm: shared_memory.SharedMemory, num_slots: int) -> np.ndarray:
    return np.ndarray(
        (num_slots, _FLAG_FIELDS),
        dtype=np.int64,
        buffer=shm.buf,
        offset=_HEADER_INT64S * 8 + _HANDLE_BYTES,
    )


class CudaIpcEnumBlockPool:
    """Drafter-side GPU slot ring + shm rendezvous (the producer)."""

    def __init__(
        self,
        *,
        device: str,
        endpoint: str,
        max_rows: int,
        row_width: int,
        num_slots: int = DEFAULT_NUM_SLOTS,
    ) -> None:
        self.device = device
        self.num_slots = int(num_slots)
        self.max_rows = int(max_rows)
        self.row_width = int(row_width)
        self.pool = torch.zeros(
            (self.num_slots, self.max_rows, self.row_width),
            dtype=torch.int64,
            device=device,
        )
        handle = self.pool.untyped_storage()._share_cuda_()

        name = _shm_name(endpoint)
        try:
            # A stale segment from a killed previous run would otherwise make
            # the create() below fail forever.
            stale = shared_memory.SharedMemory(name=name)
            stale.close()
            stale.unlink()
            logger.warning("Removed stale decoupled-enum shm segment %s", name)
        except FileNotFoundError:
            pass
        self._shm = shared_memory.SharedMemory(
            create=True, size=_shm_size(self.num_slots), name=name
        )
        header = np.ndarray(
            (_HEADER_INT64S,), dtype=np.int64, buffer=self._shm.buf, offset=0
        )
        handle_bytes = pickle.dumps(handle)
        if len(handle_bytes) > _HANDLE_BYTES:
            raise RuntimeError(f"CUDA IPC handle pickle too large: {len(handle_bytes)}")
        self._shm.buf[_HEADER_INT64S * 8 : _HEADER_INT64S * 8 + len(handle_bytes)] = (
            handle_bytes
        )
        self.flags = _flags_view(self._shm, self.num_slots)
        self.flags[:] = 0
        header[1] = self.pool.device.index
        header[2] = self.num_slots
        header[3] = self.max_rows
        header[4] = self.row_width
        # Magic last: a reader that sees the magic sees a complete header.
        header[0] = _MAGIC
        self._next_slot = 0
        # Private push stream + event: the producer-side visibility fence must
        # wait for the SLOT COPY alone. A device-wide sync here waited out the
        # drafter's own pre-queued gate sequence (parked on the NEXT commit),
        # taxing every push with half a verify round trip (push_ms 3.77 vs
        # the evented ZMQ path's 0.05).
        self._push_stream = torch.cuda.Stream(device=device)
        self._push_event = torch.cuda.Event()

    def push(
        self,
        *,
        pool_indices: list[int],
        base_committed_lens: list[int],
        units: torch.Tensor,
        ready_event: Optional[torch.cuda.Event] = None,
    ) -> bool:
        """Write one block into the next free slot; False if the ring is full
        (the verifier then simply falls back that round) or the block exceeds
        the slot capacity.

        ``ready_event`` is the producing pack's completion event: the slot
        copy waits on it (not on the whole stream, which may hold a parked
        gate) so the copied values are the finished block.
        """
        batch_size = len(pool_indices)
        if batch_size == 0:
            return True
        if batch_size > self.max_rows:
            logger.warning(
                "enum block larger than the IPC slot (%d > %d rows); dropped",
                batch_size,
                self.max_rows,
            )
            return False
        slot = self._next_slot
        if self.flags[slot, 0] != self.flags[slot, 1]:
            # Ring full: the consumer is behind by a whole ring. One round
            # ahead means at most one block in flight, so this is exceptional.
            logger.warning("decoupled enum IPC ring full; block dropped")
            return False
        with torch.cuda.stream(self._push_stream):
            if ready_event is not None:
                self._push_stream.wait_event(ready_event)
            meta = torch.tensor(
                [pool_indices, base_committed_lens], dtype=torch.int64
            ).T.to(self.device, non_blocking=True)
            rows = torch.cat([meta, units.view(batch_size, -1)], dim=1)
            self.pool[slot, :batch_size] = rows
            self._push_event.record()
        # The producer-side order guarantee: the block is fully visible on the
        # device before ready_seq becomes visible on the host.
        self._push_event.synchronize()
        self.flags[slot, 2] = batch_size
        self.flags[slot, 0] += 1
        self._next_slot = (slot + 1) % self.num_slots
        return True

    def close(self) -> None:
        try:
            self._shm.close()
            self._shm.unlink()
        except FileNotFoundError:
            pass


class CudaIpcEnumBlockReader:
    """Verifier-side pool mapping + slot polling (the consumer)."""

    def __init__(
        self,
        *,
        device: str,
        endpoint: str,
        attach_timeout_s: float = 300.0,
    ) -> None:
        self.device = device
        name = _shm_name(endpoint)
        deadline = time.monotonic() + attach_timeout_s
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                if self._try_attach(name):
                    return
            except Exception as exc:  # noqa: BLE001
                # A stale segment from a dead producer carries a valid magic
                # but a dead CUDA handle (opening it raises "invalid resource
                # handle"); the live producer replaces the segment at startup,
                # so keep re-opening the shm by name until its handle opens.
                last_error = exc
            time.sleep(0.2)
        raise TimeoutError(
            f"decoupled enum IPC pool did not attach within {attach_timeout_s}s "
            f"(shm name {name}; last error: {last_error!r}); is the drafter up "
            f"with --decoupled-spec-data-transport cuda_ipc?"
        )

    def _try_attach(self, name: str) -> bool:
        """One attach attempt against a FRESH shm open; False = not there yet.

        A pre-recreation mapping must never be reused: after the producer
        unlinks + recreates the segment, an already-open fd still sees the old
        (dead) memory, so every retry re-opens by name.
        """
        try:
            shm = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            return False
        try:
            header = np.ndarray((_HEADER_INT64S,), dtype=np.int64, buffer=shm.buf)
            if header[0] != _MAGIC:
                shm.close()
                return False
            num_slots = int(header[2])
            max_rows = int(header[3])
            row_width = int(header[4])
            handle = pickle.loads(
                bytes(shm.buf[_HEADER_INT64S * 8 : _HEADER_INT64S * 8 + _HANDLE_BYTES])
            )
            # Redirect handle[0] to the consumer's device so _new_shared_cuda's
            # CUDAGuard stays there; peer access handles the cross-GPU open
            # (utils/cuda_ipc_transport_utils.py idiom).
            local_index = torch.device(self.device).index or 0
            redirected_handle = (local_index,) + tuple(handle)[1:]
            with torch.cuda.device(local_index):
                # Force the local context before the IPC open.
                torch.zeros(1, device=f"cuda:{local_index}")
                storage = torch.UntypedStorage._new_shared_cuda(*redirected_handle)
                flat = torch.empty(0, dtype=torch.int64, device=f"cuda:{local_index}")
                flat.set_(
                    storage,
                    storage_offset=0,
                    size=(num_slots * max_rows * row_width,),
                    stride=(1,),
                )
        except Exception:
            shm.close()
            raise
        self._shm = shm
        self.num_slots = num_slots
        self.max_rows = max_rows
        self.row_width = row_width
        self.pool = flat.view(num_slots, max_rows, row_width)
        self.flags = _flags_view(shm, num_slots)
        return True

    def poll(self) -> Optional[tuple[int, torch.Tensor]]:
        """Return (slot, rows [B, row_width] mapped GPU view) for one landed
        block, or None. The caller must finish consuming (and synchronize its
        stream) before calling ack(slot).
        """
        for slot in range(self.num_slots):
            if self.flags[slot, 0] != self.flags[slot, 1]:
                batch_size = int(self.flags[slot, 2])
                return slot, self.pool[slot, :batch_size]
        return None

    def ack(self, slot: int) -> None:
        self.flags[slot, 1] = self.flags[slot, 0]

    def close(self) -> None:
        try:
            self._shm.close()
        except FileNotFoundError:
            pass
