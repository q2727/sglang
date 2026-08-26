"""Verifier-side IPC thread (the recv daemon) for decoupled enumeration spec.

Control batches from the verifier are forwarded to the drafter over an injected
``BaseDecoupledSpecTransport``; enumeration buffer blocks received from the
drafter are landed into the verifier's GPU ``DecoupledEnumBuffer`` (verifier
routing + staleness live in ``DecoupledEnumBuffer.land``; each block row names
its own seat via the pool_idx echoed from DraftSync, so there is no host rid
lookup on this path). Envelope validation lives here; the wire lives in the
transport.

The loop body is factored into ``_step()`` so it can be driven directly (and
deterministically) by the fake-transport integration tests, while production
runs ``_run()`` on a daemon thread.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Optional

import msgspec
import torch

from sglang.srt.speculative.decoupled_spec_io import (
    DraftControlBatch,
    DraftEnumerationBufferBatch,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftSync,
    VerifyCommit,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    BaseDecoupledSpecTransport,
    TransportClosed,
)
from sglang.srt.utils.common import set_native_thread_name
from sglang.srt.utils.thread_band_recorder import register_thread_band_recorder

if TYPE_CHECKING:
    from sglang.srt.speculative.decoupled_enum_buffer import DecoupledEnumBuffer

logger = logging.getLogger(__name__)

# The verifier IPC thread has no send-side wakeup, so a freshly submitted control
# waits up to this long before the loop services the send queue. This bounded
# (<=1ms) control latency is intentional (matches the PR's poll(1ms)).
VERIFIER_IPC_IDLE_WAIT_TIMEOUT_S = 0.001  # 1ms

# A pending evented commit whose DraftSync has not passed through the send
# queue yet is retried this long before its request is poisoned (no further
# commits; the seat rides fallbacks until DraftClose). The genuine race
# window is under one scheduler iteration (controls drain first), so a tight
# bound matters: an unseedable head (e.g. a request that never entered the
# decoupled lifecycle) stalls every commit queued behind it for this long.
EVENTED_COMMIT_LEDGER_WAIT_S = 0.1

# Ring capacity of recently retired request ids (DraftClose seen): commits of
# an overlap tail round (launched before the finish was processed) match here
# and are skipped instantly instead of burning the retry window above.
CLOSED_RID_RING_CAPACITY = 4096


class EventedVerifyCommits(msgspec.Struct):
    """One decode round's commits, handed off at launch (the copy_done
    pattern, symmetric to the drafter's EventedDraftBlock).

    ``result`` is the round's GenerationBatchResult: after its ``copy_done``
    event fires, ``next_token_ids`` / ``accept_lens`` are pinned-CPU tensors
    and row i's accepted run is next_token_ids[i*stride : i*stride+accept[i]]
    -- the exact tokens the batch-result processor appends to output_ids, so
    the wire stream and the scheduler's bookkeeping stay in agreement.
    """

    result: Any  # GenerationBatchResult (copy_done / next_token_ids / accept_lens)
    rids: list[str]
    pool_indices: list[int]
    submitted_ts: float


class VerifierIpcThread:
    """Verifier-side IPC thread (recv daemon) for decoupled enumeration spec.

    The injected ``transport`` must be started before the loop runs; ``start()``
    starts it (and the daemon loop) and ``close()`` tears both down.
    """

    def __init__(
        self,
        *,
        transport: BaseDecoupledSpecTransport,
        enum_buffer: DecoupledEnumBuffer,
        filter_block: Optional[
            Callable[
                [DraftEnumerationBufferBatch], Optional[DraftEnumerationBufferBatch]
            ]
        ] = None,
        on_land: Optional[Callable[[DraftEnumerationBufferBatch], None]] = None,
        on_resync_sent: Optional[Callable[[DraftSync], None]] = None,
        num_drafters: int = 1,
        src_verifier_rank: int = 0,
        land_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self.transport = transport
        # Doorbell topology rule: with device-side gate waits parked on the
        # compute stream, the landing scatter must ride its own stream (a
        # landing kernel enqueued BEHIND the wait could never satisfy it).
        # None = land on this thread's current stream (pre-doorbell shape).
        self._land_stream = land_stream
        # The GPU landing buffer. land() holds verifier_rank and rejects a block
        # routed to another verifier, so this thread does no rank check of its
        # own -- only envelope validation.
        self.enum_buffer = enum_buffer
        # Runs before land(): stale rows must never overwrite the current
        # seat's GPU buffer, even briefly.
        self._filter_block = filter_block
        # Post-land hook (runs on this thread, after the scatter is enqueued);
        # the verify manager mirrors arrival stamps here for the sync-mode gate.
        self._on_land = on_land
        # A desync sync is rewritten from this thread's wire ledger just
        # before send. Notify the manager of that exact high-water mark so
        # its next expected block uses the same base.
        self._on_resync_sent = on_resync_sent
        self._send_queue: queue.SimpleQueue[DraftControlBatch] = queue.SimpleQueue()
        # Evented commits (copy_done pattern): handoff queue plus the
        # thread-local FIFO whose head gates on its round's copy_done event --
        # head-first keeps a request's commits in round order on the wire.
        self._evented_queue: queue.SimpleQueue[EventedVerifyCommits] = (
            queue.SimpleQueue()
        )
        self._evented_fifo: deque[EventedVerifyCommits] = deque()
        # Wire-view ledger of each request's committed total: seeded when this
        # thread forwards the request's DraftSync, advanced by every commit it
        # builds, retired by DraftClose. Owning it here (not mirroring the
        # scheduler's) keeps the wire stream self-consistent by construction.
        self._sent_committed_lens: dict[str, int] = {}
        self._sent_committed_outputs: dict[str, list[int]] = {}
        # Desync re-seed floors (see _drain_send_queue): rid -> the snapshot
        # length its in-flight rounds must clear before commits resume.
        self._resync_floors: dict[str, int] = {}
        # Recently retired rids (bounded ring + set): lets an overlap tail
        # round's commits be dropped instantly instead of stalling the FIFO.
        self._closed_rid_ring: deque[str] = deque(maxlen=CLOSED_RID_RING_CAPACITY)
        self._closed_rids: set[str] = set()
        self.num_drafters = max(1, int(num_drafters))
        self.src_verifier_rank = int(src_verifier_rank)
        self._closed = threading.Event()
        # Side-band trace recorder (kineto is blind to this thread; bands
        # are injected into the exported trace at stop_profile).
        self._bands = register_thread_band_recorder("sgl-verify-ipc")
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-verifier-ipc",
            daemon=True,
        )

    def start(self) -> None:
        self.transport.start()
        if not self._thread.is_alive():
            self._thread.start()

    def close(self) -> None:
        self._closed.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                logger.warning(
                    "Verifier IPC thread did not exit within 1.0s of close()"
                )
        self.transport.close()

    def submit_control_batch(self, batch: DraftControlBatch) -> None:
        # Verifier -> drafter only. The verifier keeps no control mirror in the
        # enumeration design: request lifecycle lives in the scheduler's slot
        # table (assign / remove) and committed length, not on this thread.
        self._send_queue.put(batch)

    def submit_evented_commits(self, pending: EventedVerifyCommits) -> None:
        if not pending.rids:
            return
        self._evented_queue.put(pending)

    def _step(self) -> bool:
        """Run one drain cycle (outgoing controls + evented commits + incoming
        blocks). Controls drain FIRST so a request's DraftSync always hits the
        wire before the commits of its first round (whose copy_done fires at
        least a verify round later anyway).

        Returns whether any work was done. Safe to call directly from tests.
        """
        did_work = self._drain_send_queue()
        did_work = self._drain_evented_commits() or did_work
        did_work = self._drain_incoming() or did_work
        return did_work

    def _run(self) -> None:
        # pthread name for profiler/py-spy/top track labeling.
        set_native_thread_name("sgl-verify-ipc")
        while not self._closed.is_set():
            try:
                if not self._step():
                    self.transport.wait_for_input(VERIFIER_IPC_IDLE_WAIT_TIMEOUT_S)
            except TransportClosed:
                break
            except Exception:
                # Without this, a routing error from _route_* escapes the loop
                # and silently kills the thread for all requests. Die loudly;
                # phase 5c will quarantine the offending request instead.
                logger.exception("Verifier IPC thread terminating on unexpected error")
                break

    def _drain_send_queue(self) -> bool:
        # verifier -> drafter controls
        did_work = False
        while True:
            try:
                batch = self._send_queue.get_nowait()
            except queue.Empty:
                break
            did_work = True
            # The ledger follows the wire: a DraftSync (re-)roots the
            # request's committed-output total, a DraftClose retires it.
            # A desync re-seed keeps the cursor (the chain is intact, the
            # snapshot merely re-tells it to the drafter) and instead sets a
            # send FLOOR: in-flight rounds whose base falls under the
            # snapshot are already inside its committed_outputs -- sending
            # them would double-apply on the re-opened seat.
            for sync in batch.sync_messages:
                if sync.desync_reseed:
                    wire_outputs = self._sent_committed_outputs.get(sync.request_id)
                    if wire_outputs is not None:
                        # Scheduler state and the evented wire ledger advance
                        # independently under overlap. Both contain verified
                        # target tokens, so use their longest common-prefix
                        # high-water. The floor below filters duplicate older
                        # commits while the shorter side catches up.
                        scheduler_outputs = list(sync.committed_outputs)
                        wire_outputs = list(wire_outputs)
                        if len(scheduler_outputs) <= len(wire_outputs) and wire_outputs[
                            : len(scheduler_outputs)
                        ] == scheduler_outputs:
                            sync.committed_outputs = wire_outputs
                        elif len(wire_outputs) <= len(scheduler_outputs) and scheduler_outputs[
                            : len(wire_outputs)
                        ] == wire_outputs:
                            sync.committed_outputs = scheduler_outputs
                        else:
                            logger.error(
                                "desync re-seed prefixes diverged for %s "
                                "(scheduler=%d wire=%d); using wire ledger",
                                sync.request_id,
                                len(scheduler_outputs),
                                len(wire_outputs),
                            )
                            sync.committed_outputs = wire_outputs
                    self._resync_floors[sync.request_id] = len(sync.committed_outputs)
                    if self._on_resync_sent is not None:
                        self._on_resync_sent(sync)
                else:
                    self._sent_committed_lens[sync.request_id] = len(
                        sync.committed_outputs
                    )
                    self._sent_committed_outputs[sync.request_id] = list(
                        sync.committed_outputs
                    )
                    self._resync_floors.pop(sync.request_id, None)
                self._closed_rids.discard(sync.request_id)
            for commit in batch.verify_commit_messages:
                rid = commit.request_id
                pre_len = self._sent_committed_lens.get(rid)
                if pre_len is None or pre_len != int(commit.pre_verify_committed_len):
                    logger.error(
                        "scheduler commit ledger mismatch for %s "
                        "(wire=%s commit_base=%d)",
                        rid,
                        pre_len,
                        int(commit.pre_verify_committed_len),
                    )
                    continue
                self._sent_committed_lens[rid] = pre_len + len(
                    commit.committed_tokens
                )
                self._sent_committed_outputs.setdefault(rid, []).extend(
                    int(token) for token in commit.committed_tokens
                )
            for close in batch.close_messages:
                self._sent_committed_lens.pop(close.request_id, None)
                self._sent_committed_outputs.pop(close.request_id, None)
                self._resync_floors.pop(close.request_id, None)
                if close.request_id not in self._closed_rids:
                    if len(self._closed_rid_ring) == self._closed_rid_ring.maxlen:
                        self._closed_rids.discard(self._closed_rid_ring[0])
                    self._closed_rid_ring.append(close.request_id)
                    self._closed_rids.add(close.request_id)
            with self._bands.band("verifier_ipc.send_controls"):
                self.transport.send(
                    int(batch.dst_drafter_rank),
                    DraftMeshMessage.from_control_batch(batch),
                )
        return did_work

    def _drain_evented_commits(self) -> bool:
        # Complete every FIFO-head round whose copy_done fired: slice the
        # accepted runs from the pinned result tensors, build VerifyCommits,
        # send. Head-first keeps a request's commits in round order.
        did_work = False
        while True:
            try:
                self._evented_fifo.append(self._evented_queue.get_nowait())
                did_work = True
            except queue.Empty:
                break
        while self._evented_fifo:
            head = self._evented_fifo[0]
            # Readiness needs BOTH checks: an assigned-but-not-yet-recorded
            # CUDA event reports query() == True (CUDA treats an unrecorded
            # event as complete), so between the scheduler assigning copy_done
            # and copy_to_cpu() recording it, query alone would let us read
            # the still-GPU tensors -- graph-static storage the next replay
            # overwrites (observed as 0x01010101 accept_lens and index-like
            # "tokens" on long-context runs). copy_to_cpu rebinds the fields
            # to CPU tensors BEFORE recording, so is_cpu AND query together
            # imply the pinned data is complete.
            copy_done = head.result.copy_done
            if (
                copy_done is None
                or not head.result.accept_lens.is_cpu
                or not copy_done.query()
            ):
                break
            missing = [
                rid
                for rid in head.rids
                if rid not in self._sent_committed_lens and rid not in self._closed_rids
            ]
            if missing:
                # The round outran its DraftSync (still queued behind us) --
                # retry briefly; past the window, poisoned rids simply send no
                # further commits (their seats ride fallbacks until close).
                # Closed rids (an overlap tail round) never enter `missing`:
                # the ledger check below drops them instantly.
                if time.monotonic() - head.submitted_ts < EVENTED_COMMIT_LEDGER_WAIT_S:
                    break
                logger.warning(
                    "evented commits dropped for unseeded requests %s "
                    "(DraftSync never passed this thread)",
                    missing[:4],
                )
            self._evented_fifo.popleft()
            with self._bands.band("verifier_ipc.send_round_commits"):
                self._send_round_commits(head)
            did_work = True
        return did_work

    def _send_round_commits(self, pending: EventedVerifyCommits) -> None:
        result = pending.result
        next_token_ids = result.next_token_ids.tolist()
        accept_lens = result.accept_lens.tolist()
        stride = int(result.speculative_num_draft_tokens)
        control_batches: dict[int, DraftControlBatch] = {}
        for i, (rid, pool_idx) in enumerate(zip(pending.rids, pending.pool_indices)):
            pre_len = self._sent_committed_lens.get(rid)
            if pre_len is None:
                continue
            tokens = next_token_ids[i * stride : i * stride + int(accept_lens[i])]
            if not tokens:
                continue
            # The ledger follows RESULTS, not sends: a skipped send (resync
            # floor, negative-token guard) must still advance the cursor, or
            # every later commit gets attributed to the wrong absolute base
            # and the drafter's alignment check passes on corrupt splices.
            self._sent_committed_lens[rid] = pre_len + len(tokens)
            self._sent_committed_outputs.setdefault(rid, []).extend(
                int(token) for token in tokens
            )
            floor = self._resync_floors.get(rid)
            if floor is not None:
                if pre_len + len(tokens) <= floor:
                    # This round predates the desync re-seed: its tokens are
                    # all inside the snapshot's committed_outputs. (The
                    # ledger advance above already moved the cursor, so
                    # later bases stay right.)
                    continue
                if pre_len < floor:
                    # The round STRADDLES the snapshot edge: its head is in
                    # the snapshot, its tail is NEW. Send exactly the tail
                    # (the wire complement of the snapshot) -- skipping the
                    # whole round loses the tail forever, leaving the
                    # drafter one permanent gap behind: every later commit
                    # then drops as misaligned and the seat never recovers
                    # (each 5s re-seed just re-rolls whether the next
                    # snapshot lands on a round boundary).
                    tokens = tokens[floor - pre_len :]
                    pre_len = floor
                self._resync_floors.pop(rid, None)
            if any(token < 0 for token in tokens):
                # A negative id is the verify output's not-accepted padding:
                # it must never reach the wire (the drafter would gather an
                # embedding with it and die on a device assert). Skipping the
                # round costs a base gap the drafter's alignment check turns
                # into staleness fallbacks (healed by the streak resync); the
                # loud log is the probe for whoever produced it.
                logger.error(
                    "evented commit for %s dropped: negative token in accepted "
                    "run %s (accept_len=%d) -- verify output padding leaked",
                    rid,
                    tokens[:8],
                    int(accept_lens[i]),
                )
                continue
            rank = int(pool_idx) % self.num_drafters
            batch = control_batches.get(rank)
            if batch is None:
                batch = DraftControlBatch(dst_drafter_rank=rank)
                control_batches[rank] = batch
            batch.verify_commit_messages.append(
                VerifyCommit(
                    request_id=rid,
                    src_verifier_rank=self.src_verifier_rank,
                    dst_drafter_rank=rank,
                    pre_verify_committed_len=pre_len,
                    committed_tokens=[int(token) for token in tokens],
                    req_pool_idx=int(pool_idx),
                )
            )
        for rank, batch in control_batches.items():
            self.transport.send(rank, DraftMeshMessage.from_control_batch(batch))

    def _drain_incoming(self) -> bool:
        # drafter -> verifier enumeration buffer blocks
        did_work = False
        while (message := self.transport.try_recv()) is not None:
            did_work = True
            block = self._route_enumeration_message(message)
            if self._filter_block is not None:
                block = self._filter_block(block)
                if block is None:
                    continue
            # Verifier routing (wrong-verifier reject), validate(), and the
            # seat-range guard all live in land(); the SYNC scatter runs on the
            # current stream (6.3 moves it to a copy stream), or on the
            # dedicated landing stream when the doorbell is armed.
            with self._bands.band("verifier_ipc.land_block"):
                if self._land_stream is not None:
                    with torch.cuda.stream(self._land_stream):
                        self.enum_buffer.land(block)
                else:
                    self.enum_buffer.land(block)
                if self._on_land is not None:
                    self._on_land(block)
        return did_work

    def _route_enumeration_message(
        self, message: DraftMeshMessage
    ) -> DraftEnumerationBufferBatch:
        """Extract one enumeration buffer block from its envelope.

        Raises on a malformed envelope; ``_run`` catches that and terminates
        loudly (5c will quarantine instead). Semantic validation (verifier
        routing, duplicate rids, K/F dims) is deferred to ``land``.
        """
        if not isinstance(message, DraftMeshMessage):
            raise RuntimeError(
                f"Unexpected message on the verifier IPC thread: {message}"
            )
        if (
            message.message_type != DraftMeshMessageType.ENUMERATION_BUFFER_BATCH
            or message.enumeration_buffer_batch is None
        ):
            raise RuntimeError(
                f"Unexpected message on the verifier IPC thread: {message}"
            )
        return message.enumeration_buffer_batch
