"""Drafter-side IPC thread for decoupled speculative decoding.

Owns the verifier->drafter control inbox and the drafter->verifier outgoing
result queue, moving ``DraftMeshMessage`` envelopes over an injected
``BaseDecoupledSpecTransport``. Message validation and rank routing live here;
the wire lives in the transport.

The loop body is factored into ``_step()`` so it can be driven directly (and
deterministically, no background thread) by the fake-transport integration
tests, while production runs ``_run()`` on a daemon thread.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

import msgspec

from sglang.srt.speculative.decoupled_spec_io import (
    DraftControlBatch,
    DraftControlInbox,
    DraftEnumerationBufferBatch,
    DraftMeshMessage,
    DraftMeshMessageType,
    ReadyDraftControls,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    BaseDecoupledSpecTransport,
    TransportClosed,
)
from sglang.srt.utils.common import set_native_thread_name
from sglang.srt.utils.thread_band_recorder import register_thread_band_recorder

logger = logging.getLogger(__name__)

# Idle floor only: the loop wakes immediately via _wakeup when a result is
# queued; this just bounds the fully-idle sleep before re-polling for controls.
# It doubles as the poll cadence for pending evented blocks' CUDA events.
DRAFTER_IPC_IDLE_WAIT_TIMEOUT_S = 0.0005  # 0.5ms


class EventedDraftBlock(msgspec.Struct):
    """A block handed off before its token payload reached the host.

    The drafter loop enqueues the staging copy on its stream, records
    ``event``, and moves on (the copy_done pattern); this thread completes
    the block off the critical path: event ready -> materialize ``tokens``
    from the pinned buffer -> send -> release the staging slot.
    """

    header: DraftEnumerationBufferBatch  # tokens still empty
    event: Optional[Any]  # duck-typed .query() -> bool; None = ready now
    buffer: Optional[Any]  # pinned flat int64 tensor, >= num_tokens valid
    num_tokens: int
    on_sent: Optional[Callable[[], None]]  # releases the staging slot


class _PushStagingSlot(msgspec.Struct):
    buffer: Optional[Any] = None  # pinned flat int64 tensor, grown on demand


class PushStagingRing:
    """Fixed pool of pinned staging buffers for evented block pushes.

    The drafter loop acquires a slot per push; the IPC thread releases it
    after the send. The pool bounds host-pinned memory and naturally
    backpressures (an empty ring falls back to a synchronous push).
    """

    def __init__(self, *, num_slots: int) -> None:
        self._free: queue.SimpleQueue[_PushStagingSlot] = queue.SimpleQueue()
        for _ in range(num_slots):
            self._free.put(_PushStagingSlot())

    def acquire(self, *, num_tokens: int) -> Optional[_PushStagingSlot]:
        try:
            slot = self._free.get_nowait()
        except queue.Empty:
            return None
        if slot.buffer is None or slot.buffer.numel() < num_tokens:
            import torch

            slot.buffer = torch.empty(num_tokens, dtype=torch.int64, pin_memory=True)
        return slot

    def release(self, slot: _PushStagingSlot) -> None:
        self._free.put(slot)


class DrafterIpcThread:
    """Drafter-side IPC thread for decoupled speculative decoding.

    The injected ``transport`` must be started before the loop runs; ``start()``
    starts it (and the daemon loop) and ``close()`` tears both down.

    Plain class (not a dataclass): a thread controller, not a data container;
    mirrors the sibling ``VerifierIpcThread``.
    """

    def __init__(
        self,
        *,
        transport: BaseDecoupledSpecTransport,
        drafter_rank: int = 0,
        commit_mirror=None,
        on_commits_landed=None,
    ) -> None:
        self.transport = transport
        self.drafter_rank = int(drafter_rank)
        # Optional GPU commit mirror (see DrafterCommitMirror): every routed
        # VerifyCommit also lands its values per seat, and the hook (the
        # manager's arrival-board notify) runs after the batch's landing
        # event is recorded.
        self._commit_mirror = commit_mirror
        self._on_commits_landed = on_commits_landed
        self._control_inbox = DraftControlInbox()
        # Protects _control_inbox (loop writes, scheduler reads).
        self._inbox_lock = threading.Lock()
        self._send_queue: queue.SimpleQueue[DraftEnumerationBufferBatch] = (
            queue.SimpleQueue()
        )
        # Evented blocks: handoff queue (drafter loop -> this thread) plus the
        # thread-local FIFO whose head gates on its CUDA event. Head-first
        # consumption keeps per-seat generation order on the wire.
        self._evented_queue: queue.SimpleQueue[EventedDraftBlock] = queue.SimpleQueue()
        self._evented_fifo: deque[EventedDraftBlock] = deque()
        # Head-wedge watchdog state (see _drain_evented): which head we have
        # been blocked on, and since when.
        self._head_blocked_id: Optional[int] = None
        self._head_blocked_since = 0.0
        self._closed = threading.Event()
        # Wakes the idle loop the instant a result is queued (latency-critical send).
        self._wakeup = threading.Event()
        # Side-band trace recorder (kineto cannot see this thread; the bands
        # are injected into the exported trace). Registered here so the
        # test-driven _step() path works without _run(); the track's tid is
        # stamped by the first band recorded on the worker thread.
        self._bands = register_thread_band_recorder("sgl-draft-ipc")
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-drafter-ipc",
            daemon=True,
        )

    def start(self) -> None:
        self.transport.start()
        if not self._thread.is_alive():
            self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self._wakeup.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                logger.warning("Drafter IPC thread did not exit within 1.0s of close()")
        self.transport.close()

    def collect_ready_draft_controls(
        self,
        collector: Callable[[DraftControlInbox], ReadyDraftControls],
    ) -> ReadyDraftControls:
        """Extract ready controls from the live inbox under the inbox lock."""
        with self._inbox_lock:
            return collector(self._control_inbox)

    def submit_draft_results(self, result_batch: DraftEnumerationBufferBatch) -> None:
        # One block per (dst) verifier; the drafter scheduler produces a fresh
        # block each round and hands it off, so no defensive snapshot is needed.
        if not result_batch.pool_indices:
            return
        self._send_queue.put(result_batch)
        self._wakeup.set()

    def submit_evented_draft_results(self, block: EventedDraftBlock) -> None:
        if not block.header.pool_indices:
            if block.on_sent is not None:
                block.on_sent()
            return
        self._evented_queue.put(block)
        self._wakeup.set()

    def _step(self) -> bool:
        """Run one drain cycle (outgoing results + incoming controls).

        Returns whether any work was done. Safe to call directly from tests.
        """
        did_work = self._drain_send_queue()
        did_work = self._drain_evented() or did_work
        did_work = self._drain_incoming() or did_work
        return did_work

    def _run(self) -> None:
        # pthread name: py-spy / top -H / kernel-side identification.
        set_native_thread_name("sgl-draft-ipc")
        while not self._closed.is_set():
            try:
                if not self._step():
                    self._wakeup.wait(timeout=DRAFTER_IPC_IDLE_WAIT_TIMEOUT_S)
                    self._wakeup.clear()
            except TransportClosed:
                break
            except Exception:
                # Without this, a routing error from _route_* escapes the loop
                # and silently kills the thread for all requests. Die loudly;
                # phase 5c will quarantine the offending request instead.
                logger.exception("Drafter IPC thread terminating on unexpected error")
                break

    def _drain_incoming(self) -> bool:
        # verifier -> drafter controls
        did_work = False
        while (message := self.transport.try_recv()) is not None:
            did_work = True
            control_batch = self._route_control_message(message)
            if control_batch is None:
                continue
            if self._commit_mirror is not None and control_batch.verify_commit_messages:
                # Land -> sync -> publish, ONE COMMIT AT A TIME. The publish
                # releases the pre-launch gate (a host-func on the drafter's
                # forward stream), but the generation values are ASYNC copies
                # on the mirror stream: publishing before they execute lets
                # the gated scatter race ahead, read the stale generation and
                # junk the whole pre-launched round (observed: 195/200
                # junked). The sync must be PER COMMIT, not per batch: a
                # burst's second landing waits on the gated scatter's fence,
                # which only fires after the FIRST landing's publish releases
                # the gate -- a batch-level sync before any publish would
                # deadlock that chain until the gate's timeout.
                for commit in control_batch.verify_commit_messages:
                    with self._bands.band("drafter_ipc.land_commit"):
                        landed_event = self._commit_mirror.land(commit)
                        if landed_event is not None:
                            landed_event.synchronize()
                        if self._on_commits_landed is not None:
                            self._on_commits_landed([commit])
                self._commit_mirror.record_landing()
            with self._inbox_lock, self._bands.band("drafter_ipc.inbox_controls"):
                self._control_inbox.add_control_batch_locked(control_batch)
        return did_work

    def _route_control_message(
        self, message: DraftMeshMessage
    ) -> Optional[DraftControlBatch]:
        """Validate + rank-filter one control message.

        Returns the batch for this drafter, or ``None`` if addressed to another
        drafter rank (fan-out filtering, dropped quietly). Raises on a malformed
        envelope; ``_run`` catches that and terminates loudly (5c will quarantine).
        """
        if not isinstance(message, DraftMeshMessage):
            raise RuntimeError(f"Unexpected draft control message: {message}")
        if (
            message.message_type != DraftMeshMessageType.CONTROL_BATCH
            or message.control_batch is None
        ):
            raise RuntimeError(f"Unexpected draft control message: {message}")
        control_batch = message.control_batch
        if int(control_batch.dst_drafter_rank) != int(self.drafter_rank):
            return None
        return control_batch

    def _drain_send_queue(self) -> bool:
        # drafter -> verifier draft tokens
        did_work = False
        while True:
            try:
                result_batch = self._send_queue.get_nowait()
            except queue.Empty:
                break
            did_work = True
            self._send_draft_results(result_batch)
        return did_work

    def _drain_evented(self) -> bool:
        # drafter -> verifier evented blocks: complete every FIFO-head block
        # whose staging copy has drained (events record in stream order, so
        # head-first never deadlocks); a not-yet-ready head is re-polled on
        # the next cycle (idle floor 0.5ms).
        did_work = False
        while True:
            try:
                self._evented_fifo.append(self._evented_queue.get_nowait())
                did_work = True
            except queue.Empty:
                break
        while self._evented_fifo:
            head = self._evented_fifo[0]
            if head.event is not None and not head.event.query():
                now = time.monotonic()
                if self._head_blocked_id != id(head):
                    self._head_blocked_id = id(head)
                    self._head_blocked_since = now
                if now - self._head_blocked_since <= 1.0:
                    break
                # Watchdog: the staging copy was enqueued >1s ago on a stream
                # that has long since produced later rounds; an event that
                # still reports not-ready is wedged (boot-time race on some
                # platforms, root cause open), not pending. The pinned data
                # is settled -- force the send instead of wedging the FIFO
                # (and with it the whole speculation plane) forever.
                logger.warning(
                    "evented head stuck %.1fs (event never fired); force-sending",
                    now - self._head_blocked_since,
                )
            self._head_blocked_id = None
            self._evented_fifo.popleft()
            batch = head.header
            if head.buffer is not None:
                with self._bands.band("drafter_ipc.read_staging"):
                    batch.tokens = tuple(head.buffer[: head.num_tokens].tolist())
            batch.sent_unix_ts = time.time()
            self._send_draft_results(batch)
            if head.on_sent is not None:
                head.on_sent()
            did_work = True
        return did_work

    def _send_draft_results(self, result_batch: DraftEnumerationBufferBatch) -> None:
        # An enumeration block carries a single dst_verifier_rank (parallel-array
        # message, one verifier per block), so it routes to exactly one peer -- no
        # per-row grouping. A drafter serving M:N verifiers submits one block per
        # verifier, each already addressed.
        if not result_batch.pool_indices:
            return
        with self._bands.band("drafter_ipc.send_block"):
            self.transport.send(
                int(result_batch.dst_verifier_rank),
                DraftMeshMessage.from_enumeration_buffer_batch(result_batch),
            )
