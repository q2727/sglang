"""Verifier-side scheduler collaborator for decoupled enumeration spec.

The scheduler delegates here once per batch result; everything else (wire,
landing, GPU select) lives in the IPC thread and the verify worker. The
manager owns:

- **Control plane bookkeeping** (host): an unseen rid gets a ``DraftSync``
  announcing the prompt, committed outputs, and its seat (req_pool_idx); each
  round's newly committed slice becomes a ``VerifyCommit``; a finished request
  sends ``DraftClose``. A seat change (retraction re-admit) re-syncs the full
  committed prefix -- the drafter-carried pool_idx protocol's only re-sync
  obligation.
- **The C6 launch gate** (``wait_for_select_blocks``, called by the verify
  worker at decode-launch, before the select gather): wait -- bounded --
  until the block each seat's select is about to read has landed. A timeout
  is never an error: that seat degrades to the unified fallback. Under the
  overlap scheduler the wait runs while the previous round still executes on
  the GPU, so up to a full verify round of drafter latency is hidden.
- **Hit / fallback accounting** from the worker's ``select_hits_queue``.

Expected arrival stamps are pure host math: the drafter stamps a block with
its total committed length (prompt + committed outputs) at enumeration time.
The block round M's select reads was enumerated two commits back, so its
stamp equals round M-1's ENTRY seq_lens + 1 -- each gate call arms the next
round's expectation from the batch it just gated; a DraftSync seeds the
first one. The first decode round of a request has no armed expectation
(under overlap its DraftSync has not even been sent when the round
launches) and simply falls back -- one round per request, by design.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.environ import envs
from sglang.srt.speculative.decoupled_doorbell import Doorbell
from sglang.srt.speculative.decoupled_scripted_drafter import ScriptedFakeDrafter
from sglang.srt.speculative.decoupled_spec_io import (
    DecoupledSpecIpcConfig,
    DraftClose,
    DraftControlBatch,
    DraftEnumerationBufferBatch,
    DraftSync,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    DecoupledSpecTransportKind,
    FakeTransportMesh,
    build_transport,
)
from sglang.srt.speculative.verifier_ipc_thread import (
    EventedVerifyCommits,
    VerifierIpcThread,
)
from sglang.srt.utils.nvtx_utils import profile_range

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.speculative.verify_worker import VerifyWorker

logger = logging.getLogger(__name__)

_LOOPBACK_VERIFIER_ENDPOINT = "loopback://decoupled-spec-verifier"
_LOOPBACK_DRAFTER_ENDPOINT = "loopback://decoupled-spec-drafter"

# Per-drafter quarantine (1:N): this many consecutive gate timeouts mark a
# drafter sick; its seats are not waited on for the cooldown (one W burned per
# cooldown instead of per round). Any landed block clears it. Real failover
# is 5c's job -- this only bounds the damage of a dead peer.
_QUARANTINE_TIMEOUT_STREAK = 3
_QUARANTINE_COOLDOWN_S = 5.0


class EnumArrivalBoard:
    """Host mirror of landed stamps per seat (daemon writes, scheduler waits).

    The GPU buffer holds the authoritative stamps; this mirror exists only so
    the sync-mode gate can wait without a device sync.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._stamps: dict[int, int] = {}
        # Speculative (bet) arrivals live on their own EXACT-match lane: a
        # bet's stamp is the FULL-ACCEPT hypothesis (base + K + 1), which is
        # >= every possible expected base -- fed into the GEQ lane it would
        # release every future gate wait unconditionally (one bet landing
        # per seat would free-run the verify loop into permanent staleness
        # fallbacks). Exact-match wakes the gate only when the hypothesis is
        # RIGHT (the select will find the fresh gen-2 block); a wrong bet
        # never wakes anything and the gate waits for the real block as
        # before.
        self._spec_stamps: dict[int, int] = {}

    def record(self, block: DraftEnumerationBufferBatch) -> None:
        if block.speculative:
            with self._cond:
                for pool_idx, stamp in zip(
                    block.pool_indices, block.base_committed_lens
                ):
                    self._spec_stamps[int(pool_idx)] = int(stamp)
                self._cond.notify_all()
            return
        self.record_pairs(block.pool_indices, block.base_committed_lens)

    def record_pairs(self, pool_indices: list[int], stamps: list[int]) -> None:
        with self._cond:
            for pool_idx, stamp in zip(pool_indices, stamps):
                self._stamps[int(pool_idx)] = int(stamp)
            self._cond.notify_all()

    def reset_seat(self, pool_idx: int) -> None:
        # Mirror of DecoupledEnumBuffer.reset_slot: a reused seat's stale
        # stamp can exceed the NEW occupant's expectations (shorter prompt),
        # so a leftover entry makes waits succeed against dead data. Under
        # the doorbell this is a deadlock: the device flag IS reset, so the
        # GPU parks, while this stale mirror convinces the watchdog the
        # block arrived -- nobody force-releases.
        with self._cond:
            self._stamps.pop(int(pool_idx), None)
            self._spec_stamps.pop(int(pool_idx), None)

    def wait_for(self, expected: dict[int, int], timeout_s: float) -> bool:
        """Wait until every seat's landed stamp equals its expected base.

        Returns False on timeout (the verify round then falls back for the
        seats that never arrived).
        """

        def _arrived() -> bool:
            # ">=", not "==": stamps advance monotonically per seat, and a
            # commit merge on the drafter can skip a generation entirely --
            # once the seat moved PAST the expected stamp, waiting longer can
            # never help (the select falls back either way). The speculative
            # lane is exact-match only (see __init__).
            return all(
                self._stamps.get(pool_idx, -1) >= stamp
                or self._spec_stamps.get(pool_idx) == stamp
                for pool_idx, stamp in expected.items()
            )

        deadline = time.monotonic() + timeout_s
        with self._cond:
            while not _arrived():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=remaining)
            return True


class _ReqState:
    def __init__(
        self, *, pool_idx: int, prompt_len: int, committed_len: int, epoch: int
    ) -> None:
        self.pool_idx = pool_idx
        self.prompt_len = prompt_len
        self.committed_len = committed_len  # committed OUTPUT tokens
        self.epoch = epoch

    @property
    def total_committed_len(self) -> int:
        return self.prompt_len + self.committed_len


class DecoupledVerifyManager:
    """Scheduler collaborator: control plane + sync pacing + accounting."""

    def __init__(
        self,
        *,
        ipc_config: DecoupledSpecIpcConfig,
        verify_worker: VerifyWorker,
        data_transport: str = "zmq",
    ) -> None:
        self.ipc_config = ipc_config
        self.verify_worker = verify_worker
        self.arrival_board = EnumArrivalBoard()
        # 1:N seat sharding: each seat is owned by one drafter for its whole
        # occupancy (stable modulo assignment); a seat change re-syncs on the
        # new owner and closes on the old one. Full M:N policy stays out of
        # scope.
        self.num_drafters = max(1, len(ipc_config.connect_endpoints))

        self._rid_states: dict[str, _ReqState] = {}
        # Never delete a seat counter: a late block from a finished request
        # must not become valid when the same physical seat is reused.
        self._seat_epochs: dict[int, int] = {}
        self._stale_epoch_drop_ct = 0
        # Per-seat expected stamp for the NEXT decode round's select (armed by
        # each gate call from the batch it gated; seeded by DraftSync).
        self._gate_expected: dict[int, int] = {}
        # Per-drafter health (1:N): a drafter that keeps timing out the gate
        # is quarantined for a cooldown -- its seats are not waited on (they
        # ride fallbacks), so a dead or wedged drafter costs one W per
        # cooldown instead of one W per round. Any landed block from it
        # clears the quarantine (proof of life).
        self._drafter_timeout_streaks: dict[int, int] = {}
        self._drafter_cooldown_until: dict[int, float] = {}
        self.enum_round_ct = 0
        self.enum_hit_ct = 0
        self.sync_wait_timeout_ct = 0
        self._skip_log_ct = 0
        # Round-timeline profile (SGLANG_DEBUG_DECOUPLED_VERIFY_PROFILE): the
        # verify round seen from this hook. Per on_batch_result call:
        #   loop_ms = entry - previous exit  (verify forward + batch-result +
        #             scheduling, i.e. everything outside this hook)
        #   ctl_ms  = control-plane collect + submit
        #   wait_ms = arrival-board wait
        # transport_ms accumulates (land_time - block.sent_unix_ts) from the
        # IPC thread (same host clock across the two processes).
        self._profile = envs.SGLANG_DEBUG_DECOUPLED_VERIFY_PROFILE.get()
        self._prof_last_exit: Optional[float] = None
        self._prof_round_ct = 0
        self._prof_loop_ms = 0.0
        self._prof_ctl_ms = 0.0
        self._prof_wait_ms = 0.0
        self._prof_transport_ms = 0.0
        self._prof_transport_ct = 0
        # Per-round bound on waiting for the next block: the deterministic
        # sync-mode pacing by default; 0 = pure async pacing (never stall the
        # verifier on the drafter; late blocks fall back).
        self.arrival_wait_s = envs.SGLANG_DECOUPLED_ENUM_WAIT_MS.get() / 1000.0
        # Adaptive gate budget (see _gate_budget_s): EWMA of the MEASURED
        # arrival wait on rounds whose block arrived. Timeouts are censored
        # (never sampled), so a desync episode cannot inflate the budget --
        # only real arrivals, which reflect the drafter's true loop time
        # (commit lag + draft round + transport), can move it.
        self._adaptive_gate = (
            envs.SGLANG_ENABLE_DECOUPLED_ADAPTIVE_GATE_WAIT.get()
            and self.arrival_wait_s > 0
        )
        self._gate_arrival_ewma_s: Optional[float] = None
        self._gate_arrival_ct = 0
        # Anneal escape (see _gate_budget_s): consecutive gate timeouts mean
        # the pair phase-locked in the WRONG phase -- blocks arriving a
        # constant hair after their gate closes, every generation, forever
        # (the frozen-hit basin: overlap defers each round's commit by one
        # round, both loops advance at the same cadence, so the constant
        # phase lag never decays and no tight budget can catch it). One
        # ceiling-budget park lets the in-flight block land INSIDE the gate,
        # which re-enters the good phase -- paying the window once instead
        # of missing forever.
        self._gate_consec_timeouts = 0
        self._gate_anneal_ct = 0
        # Desync resync (the cure the anneal cannot be): under the overlap
        # loop, a commit's readiness flip (copy_to_cpu) is queued on THIS
        # thread BEHIND the next round's gate park, so once a seat slips k
        # generations behind, no wait budget can ever recover it -- the
        # unlock event is sequenced after the gate, every round, forever
        # (observed as whole benches frozen at expected == landed + 2 with
        # 200ms budget). A fresh DraftSync collapses the lag at any depth:
        # the drafter re-opens the seat at the CURRENT committed length
        # (engine.open == retraction re-sync, drops the old prefix KV) and
        # the next block stamps correctly. Control-channel FIFO makes it
        # safe against in-flight commits: stale ones apply to the old seat
        # state before the re-open discards it.
        self._seat_timeout_streaks: dict[int, int] = {}
        self._force_resync_seats: set[int] = set()
        self._seat_resync_last: dict[int, float] = {}
        self._gate_resync_ct = 0
        # Stream gate (see decoupled_stream_gate): the C6 wait as a host-func
        # node ON the forward stream -- the scheduler thread enqueues it and
        # returns, the round's launches pile up behind it, and the IPC
        # thread's landing notifies the same arrival board to release it.
        # Timeout accounting then runs on the driver's callback thread; the
        # counters it touches are GIL-guarded primitives and their readers
        # tolerate one-round staleness.
        self._stream_gate = None
        self._gate_cb_bands = None
        if envs.SGLANG_ENABLE_DECOUPLED_STREAM_GATE.get() and self.arrival_wait_s > 0:
            try:
                from sglang.srt.speculative.decoupled_stream_gate import StreamGate
                from sglang.srt.utils.thread_band_recorder import (
                    register_thread_band_recorder,
                )

                self._stream_gate = StreamGate()
                # The gate's park runs on the driver's host-func callback
                # thread, which kineto cannot see (thread-local callbacks);
                # a side-band recorder puts its true park span into exported
                # traces (tid stamps on first use, i.e. the callback thread).
                self._gate_cb_bands = register_thread_band_recorder(
                    "sgl-c6-gate-callback"
                )
                logger.info("decoupled C6 gate: host-func stream gate enabled")
            except Exception:
                logger.exception(
                    "stream gate unavailable; falling back to the host gate"
                )

        # Doorbell (device-side C6 gate): the scheduler thread enqueues
        # per-seat cuStreamWaitValue32 on its compute stream and launches the
        # round immediately; the landing stream's stamp scatter releases the
        # GPU. A host watchdog force-releases on timeout (the select then
        # falls back naturally -- the doorbell never carries correctness).
        # Requires the landing scatter on a dedicated stream (a landing
        # kernel behind the wait on the SAME stream could never satisfy it).
        self.doorbell: Optional[Doorbell] = None
        self._doorbell_pending: queue.Queue[tuple[dict[int, int], float]] = (
            queue.Queue()
        )
        self._doorbell_closed = threading.Event()
        self._doorbell_watchdog: Optional[threading.Thread] = None
        self.land_stream: Optional[torch.cuda.Stream] = None
        if (
            envs.SGLANG_ENABLE_DECOUPLED_DOORBELL.get()
            and get_tensor_model_parallel_world_size() == 1
        ):
            self.doorbell = Doorbell(flags=verify_worker.enum_buffer.doorbell_flags)
            self.land_stream = torch.cuda.Stream(device=verify_worker.device)
            self._doorbell_watchdog = threading.Thread(
                target=self._doorbell_watchdog_loop,
                name="sglang-decoupled-doorbell-watchdog",
                daemon=True,
            )
            self._doorbell_watchdog.start()

        self.scripted_drafter: Optional[ScriptedFakeDrafter] = None
        loopback_mode = envs.SGLANG_TEST_DECOUPLED_LOOPBACK.get()
        if loopback_mode is not None:
            transport = self._build_loopback(loopback_mode)
        else:
            import zmq

            transport = build_transport(
                kind=DecoupledSpecTransportKind.ZMQ,
                bind_endpoint=ipc_config.bind_endpoint,
                connect_endpoints=ipc_config.connect_endpoints,
                context=zmq.Context(2),
            )
        self.ipc_thread = VerifierIpcThread(
            transport=transport,
            enum_buffer=verify_worker.enum_buffer,
            filter_block=self._filter_block_epoch,
            on_land=self._on_block_landed,
            on_resync_sent=self._on_resync_sent,
            num_drafters=self.num_drafters,
            src_verifier_rank=ipc_config.rank,
            land_stream=self.land_stream,
        )
        self.ipc_thread.start()
        # The worker calls the gate at decode-launch, before its select gather,
        # and relays each round's result for evented commit sending. With the
        # doorbell armed the gate enqueues device-side waits instead of
        # parking the host.
        verify_worker.select_gate = (
            self.enqueue_doorbell_gate
            if self.doorbell is not None
            else self.wait_for_select_blocks
        )
        verify_worker.commit_relay = self._relay_round_commits

        self._ipc_poll_closed = threading.Event()
        self._ipc_poll_thread: Optional[threading.Thread] = None
        if data_transport == "cuda_ipc" and loopback_mode is None:
            self._ipc_poll_thread = threading.Thread(
                target=self._cuda_ipc_poll_loop,
                name="sglang-decoupled-enum-ipc-poll",
                daemon=True,
            )
            self._ipc_poll_thread.start()

    def _cuda_ipc_poll_loop(self) -> None:
        """Consume enumeration blocks from the drafter's CUDA IPC pool.

        Attaches to the shm rendezvous (retrying until the drafter is up),
        then polls the slot flags: mapped rows carry [pool_idx, stamp,
        unit tokens ...], so landing is one device-side scatter; the tiny
        host mirror (2 ints per row) feeds the seat-range guard and the
        sync-mode arrival board.
        """
        from sglang.srt.speculative.cuda_ipc_enum_transport import (
            CudaIpcEnumBlockReader,
        )

        try:
            reader = CudaIpcEnumBlockReader(
                device=self.verify_worker.device,
                # The rendezvous name comes from the DRAFTER's bind endpoint,
                # which is this verifier's (only) connect endpoint.
                endpoint=self.ipc_config.connect_endpoints[0],
            )
        except TimeoutError:
            logger.exception("decoupled enum IPC pool attach failed")
            return
        logger.info("decoupled enum IPC pool attached (cuda_ipc data plane)")
        enum_buffer = self.verify_worker.enum_buffer
        landed_ct = 0
        while not self._ipc_poll_closed.is_set():
            try:
                polled = reader.poll()
            except Exception:
                # A dead poll loop silently starves every seat (landed=None
                # forever); log loudly and keep polling.
                logger.exception("decoupled enum IPC poll failed; retrying")
                time.sleep(0.01)
                continue
            if polled is None:
                time.sleep(0.0002)
                continue
            slot, rows = polled
            try:
                # Landing-stream pinned read: a plain .cpu() here rides the
                # legacy default stream and inherits the forward stream's
                # parked C6 gate (implicit blocking-stream barrier).
                pool_indices, stamps = enum_buffer.read_row_meta(rows)
                if any(p < 1 or p >= enum_buffer.seats for p in pool_indices):
                    logger.error(
                        "decoupled enum IPC block has out-of-range seats; dropped"
                    )
                else:
                    # Stream discipline lives inside land_rows_device (the
                    # buffer's private landing stream + event sync): a
                    # device-wide sync here parked this thread behind the
                    # forward stream's C6 gate, which waits for the very
                    # arrival marking below -- a deadlock loop that made
                    # every block one gate budget late (hit 5/3400).
                    enum_buffer.land_rows_device(rows[:, 0], rows[:, 1], rows[:, 2:])
                    self.arrival_board.record_pairs(pool_indices, stamps)
                    landed_ct += 1
                    if landed_ct <= 3 or landed_ct % 200 == 0:
                        logger.info(
                            "decoupled enum IPC landed #%d: seats=%s stamps=%s",
                            landed_ct,
                            pool_indices,
                            stamps,
                        )
            except Exception:
                logger.exception("decoupled enum IPC landing failed; block dropped")
            finally:
                reader.ack(slot)
        reader.close()

    def _build_loopback(self, mode: str):
        """Single-process loopback: fake mesh + an in-process scripted drafter."""
        mesh = FakeTransportMesh()
        verifier_transport = build_transport(
            kind=DecoupledSpecTransportKind.FAKE,
            bind_endpoint=_LOOPBACK_VERIFIER_ENDPOINT,
            connect_endpoints=[_LOOPBACK_DRAFTER_ENDPOINT],
            mesh=mesh,
        )
        drafter_transport = build_transport(
            kind=DecoupledSpecTransportKind.FAKE,
            bind_endpoint=_LOOPBACK_DRAFTER_ENDPOINT,
            connect_endpoints=[_LOOPBACK_VERIFIER_ENDPOINT],
            mesh=mesh,
        )
        self.scripted_drafter = ScriptedFakeDrafter(
            transport=drafter_transport,
            verifier_rank=self.ipc_config.rank,
            drafter_rank=0,
            num_steps=self.verify_worker.speculative_num_steps,
            fanout=self.verify_worker.speculative_fanout,
            mode=mode,
        )
        self.scripted_drafter.start()
        logger.info(
            "Decoupled-spec loopback: scripted fake drafter (mode=%s) started", mode
        )
        return verifier_transport

    def _close_doorbell(self) -> None:
        self._doorbell_closed.set()
        if self._doorbell_watchdog is not None:
            self._doorbell_watchdog.join(timeout=2.0)

    def close(self) -> None:
        self._close_doorbell()
        if self.scripted_drafter is not None:
            self.scripted_drafter.close()
        self.ipc_thread.close()

    def _relay_round_commits(self, batch: ScheduleBatch, batch_result) -> None:
        """Hand one decode round's result to the IPC thread (copy_done
        pattern): commits hit the wire at forward end + copy, not when the
        scheduler thread gets around to the deferred result processing."""
        self.ipc_thread.submit_evented_commits(
            EventedVerifyCommits(
                result=batch_result,
                rids=[req.rid for req in batch.reqs],
                pool_indices=[int(req.req_pool_idx) for req in batch.reqs],
                submitted_ts=time.monotonic(),
            )
        )

    def _track_drafter_timeouts(
        self, expected: dict[int, int], landed: dict[int, Optional[int]], *, now: float
    ) -> None:
        for seat, stamp in expected.items():
            if (landed.get(seat) or -1) >= stamp:
                self._seat_timeout_streaks.pop(seat, None)
                continue
            streak = self._seat_timeout_streaks.get(seat, 0) + 1
            self._seat_timeout_streaks[seat] = streak
            if streak >= 4 and now - self._seat_resync_last.get(seat, 0.0) >= 5.0:
                # Past the anneal (2): this lag is beyond what a longer wait
                # heals. Queue a full re-seed; the next on_batch_result emits
                # the DraftSync. The 5s cooldown bounds the re-prefill tax:
                # against the readiness-lag basin (commit copy_to_cpu queued
                # on the scheduler thread BEHIND the gate under overlap) a
                # re-seed only re-rolls the lock lottery -- 500ms retriggering
                # was measured to cost more than the desync itself (resync
                # churn, 21-45 tok/s). The real cure for that basin is
                # launch-time result staging (push-model), not more re-seeds.
                self._seat_resync_last[seat] = now
                self._force_resync_seats.add(seat)
                self._gate_resync_ct += 1
                if self._gate_resync_ct <= 10 or self._gate_resync_ct % 50 == 0:
                    logger.info(
                        "decoupled gate: seat %d desynced (%d consecutive "
                        "timeouts, expected=%d landed=%s) -- forcing DraftSync "
                        "re-seed #%d",
                        seat,
                        streak,
                        stamp,
                        landed.get(seat),
                        self._gate_resync_ct,
                    )
        missing_ranks = {
            self._drafter_rank_of(seat)
            for seat, stamp in expected.items()
            if (landed.get(seat) or -1) < stamp
        }
        for rank in missing_ranks:
            streak = self._drafter_timeout_streaks.get(rank, 0) + 1
            self._drafter_timeout_streaks[rank] = streak
            if streak >= _QUARANTINE_TIMEOUT_STREAK:
                self._drafter_cooldown_until[rank] = now + _QUARANTINE_COOLDOWN_S
                logger.warning(
                    "decoupled drafter %d quarantined for %.1fs after %d "
                    "consecutive gate timeouts (its seats ride fallbacks)",
                    rank,
                    _QUARANTINE_COOLDOWN_S,
                    streak,
                )

    def _on_block_landed(self, block: DraftEnumerationBufferBatch) -> None:
        # Proof of life clears the drafter's quarantine.
        rank = int(block.src_drafter_rank)
        if self._drafter_timeout_streaks.get(rank):
            self._drafter_timeout_streaks[rank] = 0
            self._drafter_cooldown_until.pop(rank, None)
        if self._profile and block.sent_unix_ts is not None:
            # Accumulated from the IPC thread; float add races with the
            # scheduler-thread reader are tolerable for debug averages.
            self._prof_transport_ms += 1000.0 * (time.time() - block.sent_unix_ts)
            self._prof_transport_ct += 1
        self.arrival_board.record(block)

    def _filter_block_epoch(
        self, block: DraftEnumerationBufferBatch
    ) -> Optional[DraftEnumerationBufferBatch]:
        """Drop rows emitted before the latest DraftSync for their seat."""
        block.validate()
        keep = [
            i
            for i, (seat, epoch) in enumerate(zip(block.pool_indices, block.epochs))
            if self._seat_epochs.get(int(seat)) == int(epoch)
        ]
        if len(keep) == block.batch_size:
            return block

        dropped = block.batch_size - len(keep)
        self._stale_epoch_drop_ct += dropped
        if self._stale_epoch_drop_ct <= 10 or self._stale_epoch_drop_ct % 100 == 0:
            logger.info(
                "decoupled verifier dropped %d stale-epoch row(s), total=%d: "
                "seats=%s epochs=%s current=%s",
                dropped,
                self._stale_epoch_drop_ct,
                block.pool_indices,
                block.epochs,
                [self._seat_epochs.get(int(seat)) for seat in block.pool_indices],
            )
        if not keep:
            return None

        tokens: list[int] = []
        for i in keep:
            tokens.extend(block.row_tokens(i))
        return replace(
            block,
            pool_indices=[block.pool_indices[i] for i in keep],
            base_committed_lens=[block.base_committed_lens[i] for i in keep],
            epochs=[block.epochs[i] for i in keep],
            tokens=tuple(tokens),
            rids=[block.rids[i] for i in keep] if block.rids else [],
        )

    def _on_resync_sent(self, sync: DraftSync) -> None:
        """Adopt the IPC wire high-water used by the actual re-seed."""
        state = self._rid_states.get(sync.request_id)
        seat = int(sync.req_pool_idx)
        if (
            state is None
            or state.pool_idx != seat
            or state.epoch != int(sync.epoch)
            or self._seat_epochs.get(seat) != int(sync.epoch)
        ):
            return
        state.committed_len = max(state.committed_len, len(sync.committed_outputs))
        self._gate_expected[seat] = state.total_committed_len
        self._seat_timeout_streaks.pop(seat, None)
        rank = self._drafter_rank_of(seat)
        self._drafter_timeout_streaks[rank] = 0
        self._drafter_cooldown_until.pop(rank, None)
        logger.info(
            "decoupled re-seed epoch=%d seat=%d adopted wire base=%d",
            state.epoch,
            seat,
            state.total_committed_len,
        )

    def _force_ahead_resync(
        self, expected: dict[int, int], landed: dict[int, Optional[int]]
    ) -> None:
        """Re-seed only after the expected block fell out of the 2-gen ring."""
        for seat, stamp in expected.items():
            got = landed.get(seat)
            # The enum buffer retains the two newest real generations. When
            # landed == expected + 1, the expected block is still in the other
            # slot and exact-stamp selection remains valid. A gap of two or
            # more means it has been overwritten and cannot heal by waiting.
            if (
                got is None
                or got - stamp < 2
                or seat in self._force_resync_seats
            ):
                continue
            self._force_resync_seats.add(seat)
            self._gate_resync_ct += 1
            logger.info(
                "decoupled gate: seat %d landed ahead (expected=%d landed=%d) "
                "-- forcing epoch re-seed #%d",
                seat,
                stamp,
                got,
                self._gate_resync_ct,
            )

    def _collect_gate_expected(self, batch: ScheduleBatch) -> dict[int, int]:
        """Seats this round's gate covers -> expected stamp. A seat with no
        armed expectation (first decode round: under overlap even its
        DraftSync is still pending) is simply not gated: its select falls
        back if the block is not there, never blocks, never errs. Seats of a
        quarantined drafter are skipped the same way."""
        now = time.monotonic()
        expected: dict[int, int] = {}
        for req in batch.reqs:
            stamp = self._gate_expected.get(req.req_pool_idx)
            if stamp is None:
                continue
            rank = self._drafter_rank_of(req.req_pool_idx)
            if self._drafter_cooldown_until.get(rank, 0.0) > now:
                continue  # quarantined drafter: its seats fall back, no wait
            expected[req.req_pool_idx] = stamp
        return expected

    def enqueue_doorbell_gate(self, batch: ScheduleBatch) -> None:
        """Doorbell C6 gate: enqueue one device-side GEQ wait per gated seat
        on the compute stream and return immediately -- the select and the
        verify launch queue up behind the waits (measured async), and the
        landing stream's stamp scatter releases them. The watchdog
        force-releases on timeout; the select falls back naturally."""
        expected = self._collect_gate_expected(batch)
        if not expected or self.arrival_wait_s <= 0:
            return
        budget_s = self._gate_budget_s()
        stream = torch.cuda.current_stream()
        enqueued: list[tuple[int, int]] = []
        for pool_idx, stamp in expected.items():
            if self.doorbell.enqueue_wait(
                stream=stream, pool_idx=pool_idx, stamp=stamp
            ):
                enqueued.append((pool_idx, stamp))
                continue
            # Driver refused the wait op. The waits already on the stream
            # would park the GPU with nobody committed to releasing them, so
            # force-release those, then run this round as a host gate
            # (correctness identical either way).
            for done_idx, done_stamp in enqueued:
                self.doorbell.force_release(pool_idx=done_idx, stamp=done_stamp)
            self.arrival_board.wait_for(expected, budget_s)
            return
        self._doorbell_pending.put((expected, time.monotonic() + budget_s))

    def _doorbell_watchdog_loop(self) -> None:
        """Companion of enqueue_doorbell_gate: for each armed round, wait on
        the host arrival mirror up to the deadline; force-release the seats
        whose block never landed (GPU unparks, select falls back) and feed
        the same timeout accounting as the host gate."""
        while not self._doorbell_closed.is_set():
            try:
                expected, deadline = self._doorbell_pending.get(timeout=0.5)
            except queue.Empty:
                continue
            remaining = deadline - time.monotonic()
            arrived = remaining > 0 and self.arrival_board.wait_for(expected, remaining)
            if arrived:
                continue
            with self.arrival_board._cond:
                landed = {
                    seat: self.arrival_board._stamps.get(seat) for seat in expected
                }
            now = time.monotonic()
            for pool_idx, stamp in expected.items():
                if (landed.get(pool_idx) or -1) < stamp:
                    self.doorbell.force_release(pool_idx=pool_idx, stamp=stamp)
            self.sync_wait_timeout_ct += 1
            self._track_drafter_timeouts(expected, landed, now=now)
            if self.sync_wait_timeout_ct <= 5 or self._profile:
                logger.info(
                    "decoupled doorbell timeout #%d: expected=%s landed=%s",
                    self.sync_wait_timeout_ct,
                    expected,
                    landed,
                )

    def _gate_budget_s(self) -> float:
        """This round's bounded gate-wait budget.

        A window-sized budget (the raw env bound, 200ms) is self-sustaining
        in the desynced regime: one parked round delays its commits by the
        full window, which starves the drafter and guarantees the NEXT
        round's block is late too -- the pair orbits at window period
        (observed at 397B as the 5-25 tok/s basin, drafter idle pinned at
        the retry cadence). Sizing the budget from MEASURED arrivals keeps
        a desynced round near fallback price, so the pair re-locks within a
        few generations instead of orbiting.

        4x the arrival EWMA + 5ms margin: arrivals reflect the drafter's
        whole loop (commit lag + draft round + transport), so slow-round /
        bootstrap excursions a couple of times the steady arrival still fit
        -- a budget tight enough to clip them would feed the consecutive-
        timeout quarantine and collapse the hit rate (the 0.8B pair failure
        mode of the 2x-verify-round formula this replaced). The env bound
        stays the ceiling, and the bootstrap budget until enough arrivals
        have been sampled.
        """
        if (
            not self._adaptive_gate
            or self._gate_arrival_ct < 20
            or self._gate_arrival_ewma_s is None
        ):
            return self.arrival_wait_s
        if self._gate_consec_timeouts == 2:
            # Anneal: ONE ceiling park per desync episode, for the lag modes
            # a longer wait CAN heal (a time-late block -- drafter capture
            # stall, transport hiccup -- lands inside the park and wait_for
            # returns on arrival). Exactly once: if the ceiling round also
            # misses, the lag is the order-deadlocked kind (see the resync
            # note in __init__) and further parks would burn the window
            # every round for nothing -- drop back to the tight budget and
            # let the seat-streak resync collapse the lag instead.
            return self.arrival_wait_s
        return min(
            self.arrival_wait_s, max(0.008, 4.0 * self._gate_arrival_ewma_s + 0.005)
        )

    def _observe_gate_arrival(self, waited_s: float) -> None:
        """EWMA the measured wait of a round whose block ARRIVED (an
        already-landed block samples ~0). Timeout rounds never sample, so
        the budget can only track the healthy arrival distribution."""
        self._gate_arrival_ct += 1
        self._gate_arrival_ewma_s = (
            waited_s
            if self._gate_arrival_ewma_s is None
            else 0.9 * self._gate_arrival_ewma_s + 0.1 * waited_s
        )

    def _stream_gate_wait(self, expected: dict[int, int], budget_s: float) -> None:
        """Runs on the driver's host-func callback thread, occupying the
        forward stream's timeline while the launch thread keeps going. Same
        wait, same budget, same timeout bookkeeping as the host gate; a
        timeout releases the stream and the gated select reads whatever
        stamp is there -- the natural fallback. No CUDA API in here (spec
        requirement for host funcs)."""
        t_gate = time.monotonic()
        bands = self._gate_cb_bands
        park_cm = (
            bands.band("verifier.c6_stream_gate_park")
            if bands is not None
            else profile_range("verifier.c6_stream_gate")
        )
        with park_cm:
            arrived = self.arrival_board.wait_for(expected, budget_s)
        if arrived:
            self._observe_gate_arrival(time.monotonic() - t_gate)
            self._gate_consec_timeouts = 0
            for seat in expected:
                self._seat_timeout_streaks.pop(seat, None)
            return
        self._gate_consec_timeouts += 1
        if self._gate_consec_timeouts == 2:
            self._gate_anneal_ct += 1
        self.sync_wait_timeout_ct += 1
        with self.arrival_board._cond:
            landed = {seat: self.arrival_board._stamps.get(seat) for seat in expected}
        self._track_drafter_timeouts(expected, landed, now=time.monotonic())
        if self.sync_wait_timeout_ct <= 5 or self._profile:
            logger.info(
                "decoupled stream-gate timeout #%d: expected=%s landed=%s",
                self.sync_wait_timeout_ct,
                expected,
                landed,
            )

    def wait_for_select_blocks(self, batch: ScheduleBatch) -> None:
        """C6 launch gate: called by the verify worker at decode-launch, just
        before the select gather. Waits -- bounded -- for the block each
        seat's select is about to read (its stamp was armed by the LAST
        on_batch_result that ran before this launch; see the arming note in
        ``_collect_req_controls``).
        """
        expected = self._collect_gate_expected(batch)
        t_wait = time.monotonic() if self._profile else 0.0
        if expected and self._skip_log_ct < 30:
            with self.arrival_board._cond:
                snapshot = {
                    seat: self.arrival_board._stamps.get(seat) for seat in expected
                }
            ahead = {
                seat: (exp, snapshot[seat])
                for seat, exp in expected.items()
                if snapshot.get(seat) is not None and snapshot[seat] > exp
            }
            if ahead:
                self._skip_log_ct += 1
                logger.info(
                    "decoupled gate skip-signature #%d (landed AHEAD of "
                    "expected -- drafter merge outran the arming): %s",
                    self._skip_log_ct,
                    ahead,
                )
                self._force_ahead_resync(expected, snapshot)
        if expected and self.arrival_wait_s > 0 and self._stream_gate is not None:
            budget_s = self._gate_budget_s()
            if self._stream_gate.enqueue(
                torch.cuda.current_stream(),
                lambda: self._stream_gate_wait(expected, budget_s),
            ):
                return
            # driver refused the node: fall through to the host gate
        if expected and self.arrival_wait_s > 0:
            t_gate = time.monotonic()
            # Named for the chrome trace: this range IS the bubble on the
            # verifier's compute stream (the CPU parks here while the GPU
            # drains, waiting for the drafter's block to land).
            with profile_range("verifier.c6_gate_wait"):
                arrived = self.arrival_board.wait_for(expected, self._gate_budget_s())
            if arrived:
                self._observe_gate_arrival(time.monotonic() - t_gate)
                self._gate_consec_timeouts = 0
                for seat in expected:
                    self._seat_timeout_streaks.pop(seat, None)
            if not arrived:
                self._gate_consec_timeouts += 1
                if self._gate_consec_timeouts == 2:
                    self._gate_anneal_ct += 1
                self.sync_wait_timeout_ct += 1
                with self.arrival_board._cond:
                    landed = {
                        seat: self.arrival_board._stamps.get(seat) for seat in expected
                    }
                self._track_drafter_timeouts(expected, landed, now=time.monotonic())
                if self.sync_wait_timeout_ct <= 5 or self._profile:
                    # Mismatch probe: a systematic expectation bug shows up in
                    # the first few timeouts (expected vs landed, side by side).
                    logger.info(
                        "decoupled gate timeout #%d: expected=%s landed=%s",
                        self.sync_wait_timeout_ct,
                        expected,
                        landed,
                    )
        if self._profile:
            # NOTE: this wait lies inside the hook-to-hook loop_ms window, so
            # the timeline log's wall sum double-counts it (debug-only).
            self._prof_wait_ms += 1000.0 * (time.monotonic() - t_wait)

    def on_batch_result(self, batch: ScheduleBatch) -> None:
        """Forward this round's lifecycle controls (DraftSync / VerifyCommit /
        DraftClose). Runs on the scheduler thread after the batch-result
        processor appended the round's committed tokens to req.output_ids --
        under the overlap scheduler that is one launch behind the round
        itself, which only delays the drafter's start, never correctness.
        """
        if not batch.reqs:
            return
        t_in = time.monotonic() if self._profile else 0.0
        control_batches: dict[int, DraftControlBatch] = {}
        for req in batch.reqs:
            self._collect_req_controls(
                req, control_batches, overlap=batch.enable_overlap
            )
        for control_batch in control_batches.values():
            if (
                control_batch.sync_messages
                or control_batch.verify_commit_messages
                or control_batch.close_messages
            ):
                self.ipc_thread.submit_control_batch(control_batch)
        self._account_select_hits(batch)
        if self._profile:
            t_out = time.monotonic()
            if self._prof_last_exit is not None:
                self._prof_round_ct += 1
                self._prof_loop_ms += 1000.0 * (t_in - self._prof_last_exit)
                self._prof_ctl_ms += 1000.0 * (t_out - t_in)
                if self._prof_round_ct % 200 == 0:
                    ct = self._prof_round_ct
                    logger.info(
                        "decoupled verify round timeline: ct=%d wall_ms=%.2f | "
                        "loop(verify+sched)=%.2f ctl=%.2f wait=%.2f | "
                        "block transport+land=%.2f (n=%d)",
                        ct,
                        (self._prof_loop_ms + self._prof_ctl_ms + self._prof_wait_ms)
                        / ct,
                        self._prof_loop_ms / ct,
                        self._prof_ctl_ms / ct,
                        self._prof_wait_ms / ct,
                        self._prof_transport_ms / max(1, self._prof_transport_ct),
                        self._prof_transport_ct,
                    )
            self._prof_last_exit = t_out

    def _drafter_rank_of(self, pool_idx: int) -> int:
        return int(pool_idx) % self.num_drafters

    def _control_batch_for(
        self, control_batches: dict[int, DraftControlBatch], drafter_rank: int
    ) -> DraftControlBatch:
        batch = control_batches.get(drafter_rank)
        if batch is None:
            batch = DraftControlBatch(dst_drafter_rank=drafter_rank)
            control_batches[drafter_rank] = batch
        return batch

    def _collect_req_controls(
        self,
        req: Req,
        control_batches: dict[int, DraftControlBatch],
        *,
        overlap: bool,
    ) -> None:
        if req.multimodal_inputs is not None:
            # The drafter is a text-only model fed nothing but token ids: a
            # multimodal prompt's placeholder ids stand for embeddings it never
            # receives (and mrope positions it cannot derive), which took the
            # drafter down with an out-of-bounds gather -- the VLM server
            # warmup alone was enough. Never open a seat for such a request;
            # it rides the plain (unspeculated) verify path.
            return
        state = self._rid_states.get(req.rid)
        if req.finished():
            if state is not None:
                self._control_batch_for(
                    control_batches, self._drafter_rank_of(state.pool_idx)
                ).close_messages.append(
                    DraftClose(
                        request_id=req.rid,
                        src_verifier_rank=self.ipc_config.rank,
                        dst_drafter_rank=self._drafter_rank_of(state.pool_idx),
                        reason="finished",
                    )
                )
                self._rid_states.pop(req.rid, None)
                self._gate_expected.pop(state.pool_idx, None)
                self._seat_timeout_streaks.pop(state.pool_idx, None)
                self._force_resync_seats.discard(state.pool_idx)
            return

        if (
            state is None
            or state.pool_idx != req.req_pool_idx
            or req.req_pool_idx in self._force_resync_seats
        ):
            # New request, a retraction re-admit that moved its seat, or a
            # desync-forced re-seed (see the resync note in __init__): (re-)open
            # with the full committed prefix and poison the seat's stamp so the
            # previous occupant's landed block cannot look fresh. A seat move
            # can also change the owning drafter -- close on the old one.
            desync_reseed = (
                req.req_pool_idx in self._force_resync_seats
                and state is not None
                and state.pool_idx == req.req_pool_idx
            )
            self._force_resync_seats.discard(req.req_pool_idx)
            self._seat_timeout_streaks.pop(req.req_pool_idx, None)
            if state is not None:
                old_rank = self._drafter_rank_of(state.pool_idx)
                self._gate_expected.pop(state.pool_idx, None)
                if old_rank != self._drafter_rank_of(req.req_pool_idx):
                    self._control_batch_for(
                        control_batches, old_rank
                    ).close_messages.append(
                        DraftClose(
                            request_id=req.rid,
                            src_verifier_rank=self.ipc_config.rank,
                            dst_drafter_rank=old_rank,
                            reason="reseated",
                        )
                    )
            self.verify_worker.enum_buffer.reset_slot(req.req_pool_idx)
            self.arrival_board.reset_seat(req.req_pool_idx)
            state = _ReqState(
                pool_idx=req.req_pool_idx,
                prompt_len=len(req.origin_input_ids),
                committed_len=len(req.output_ids),
                epoch=self._seat_epochs.get(int(req.req_pool_idx), 0) + 1,
            )
            self._seat_epochs[int(req.req_pool_idx)] = state.epoch
            self._rid_states[req.rid] = state
            drafter_rank = self._drafter_rank_of(req.req_pool_idx)
            self._control_batch_for(control_batches, drafter_rank).sync_messages.append(
                DraftSync(
                    request_id=req.rid,
                    src_verifier_rank=self.ipc_config.rank,
                    dst_drafter_rank=drafter_rank,
                    req_pool_idx=req.req_pool_idx,
                    epoch=state.epoch,
                    prompt_token_ids=list(req.origin_input_ids),
                    committed_outputs=list(req.output_ids),
                    desync_reseed=desync_reseed,
                )
            )
            # A fresh sync re-roots the drafter: the very next round's select
            # reads the sync-triggered block, so seed the gate with the synced
            # total (subsequent rounds are armed by the gate itself from each
            # gated batch's entry seq_lens).
            self._gate_expected[state.pool_idx] = state.total_committed_len
        else:
            # VerifyCommits ride the evented relay (the IPC thread builds and
            # sends them at forward end + copy_done); this hook only keeps the
            # host bookkeeping the gate and re-syncs are built from.
            #
            # Arm the gate for the NEXT launch. The protocol value is the same
            # in both modes -- the select of round R reads the block stamped
            # two commits back (T_{R-2}) -- but which hook is "the last one
            # before that launch" differs: the synchronous loop processes
            # round M before launching M+1 (this hook arms gate M+1 ->
            # PRE-delta total, T_{M-1}), while the overlap loop launches M+1
            # first and processes M afterwards (this hook arms gate M+2 ->
            # POST-delta total, T_M).
            pre_delta_total = state.total_committed_len
            # The evented commit relay can be one verify result ahead of the
            # scheduler's output_ids. Never regress a wire-ledger high-water
            # adopted by _on_resync_sent.
            state.committed_len = max(state.committed_len, len(req.output_ids))
            self._gate_expected[state.pool_idx] = (
                state.total_committed_len if overlap else pre_delta_total
            )

    def _account_select_hits(self, batch: ScheduleBatch) -> None:
        if not self.verify_worker.select_hits_queue:
            return
        if not batch.forward_mode.is_decode_or_idle():
            return
        pinned_hits, ready = self.verify_worker.select_hits_queue[0]
        if not ready.query():
            # Under the stream gate the round's select may still sit behind
            # its gate; the hook runs every round, so just take it next time.
            return
        self.verify_worker.select_hits_queue.popleft()
        hit_list = pinned_hits.tolist()
        self.enum_round_ct += len(hit_list)
        self.enum_hit_ct += sum(hit_list)
        if self.enum_round_ct and self.enum_round_ct % 200 < len(hit_list):
            logger.info(
                "decoupled enum select: hit_ct=%d round_ct=%d hit_rate=%.3f "
                "sync_wait_timeout_ct=%d gate_budget_ms=%.1f anneal_ct=%d "
                "resync_ct=%d",
                self.enum_hit_ct,
                self.enum_round_ct,
                self.enum_hit_ct / self.enum_round_ct,
                self.sync_wait_timeout_ct,
                1000.0 * self._gate_budget_s(),
                self._gate_anneal_ct,
                self._gate_resync_ct,
            )
