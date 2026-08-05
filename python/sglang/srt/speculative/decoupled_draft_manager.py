"""Drafter-side manager + dedicated event loop for decoupled enumeration spec.

The decoupled drafter serves no user requests: its whole job is answering the
verifier's control plane with enumeration blocks, exactly one round ahead.
Instead of threading mirror requests through the normal scheduler machinery,
the drafter runs this manager's ``run_loop`` as its event loop:

    drain ready controls -> close / open / apply commits -> one enumeration
    round for every touched request -> push blocks -> idle-wait.

Pacing is inherent: one block per DraftSync / VerifyCommit, no backpressure
machinery. The draft model is driven directly by ``EnumDraftEngine``.
"""

from __future__ import annotations

import logging
import os
import time
from functools import partial
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqOutput
from sglang.srt.speculative.decoupled_commit_mirror import DrafterCommitMirror
from sglang.srt.speculative.decoupled_draft_engine import EnumDraftEngine
from sglang.srt.speculative.decoupled_spec_io import (
    DecoupledSpecIpcConfig,
    DraftEnumerationBufferBatch,
    DraftReqKey,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    DecoupledSpecTransportKind,
    build_transport,
)
from sglang.srt.speculative.decoupled_verify_manager import EnumArrivalBoard
from sglang.srt.speculative.drafter_ipc_thread import (
    DrafterIpcThread,
    EventedDraftBlock,
    PushStagingRing,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
    from sglang.srt.managers.scheduler_components.profiler_manager import (
        SchedulerProfilerManager,
    )
    from sglang.srt.managers.scheduler_components.request_receiver import (
        SchedulerRequestReceiver,
    )
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_IDLE_WAIT_S = 0.0005

# Commit backlog (in verify rounds) beyond which the drafter merges the whole
# backlog to catch up instead of producing every generation; <= this depth is
# the overlap scheduler's normal in-flight allowance (one in-flight commit
# plus jitter headroom).
_MERGE_STATS = {"merged": 0, "skipped_rounds": 0, "lockstep": 0}

_CATCH_UP_BACKLOG_ROUNDS = 2

# CUDA IPC slot capacity in block rows; bounds the verifier batch size a
# single push can carry (the verifier's default running cap is far below it).
IPC_POOL_MAX_ROWS = 256


class DrafterControlPlane:
    """The drafter's slice of the scheduler control plane.

    The decoupled drafter answers no user requests, so ``run_loop`` replaces
    the scheduler's request loop wholesale -- and that loop is where the
    profiler hooks live, which is why ``/start_profile`` never reached the
    drafter. This is the missing half: a non-blocking drain of the SAME
    tokenizer / rpc sockets the scheduler drains, serving the profiler
    requests off it (anything else is logged and dropped -- the drafter has
    no request machinery to run it on), plus the round-boundary step of the
    SAME ``SchedulerProfilerManager`` every other engine uses. So the HTTP
    endpoints, ``SGLANG_TORCH_PROFILER_DIR``, the trace file naming and the
    merge all behave on the drafter exactly as they do on a normal server.

    A plain collaborator rather than a ``msgspec.Struct``: it holds live
    wires (sockets, a profiler) and exists for its behavior, like its
    ``SchedulerRequestReceiver`` peer.
    """

    def __init__(
        self,
        *,
        request_receiver: SchedulerRequestReceiver,
        send_to_tokenizer: SenderWrapper,
        profiler_manager: SchedulerProfilerManager,
    ) -> None:
        self.request_receiver = request_receiver
        self.send_to_tokenizer = send_to_tokenizer
        self.profiler_manager = profiler_manager

    def poll(self) -> None:
        """Drain the control sockets (non-blocking) and answer profiler reqs."""
        for recv_req in self.request_receiver.recv_requests():
            if not isinstance(recv_req, ProfileReq):
                logger.warning(
                    "decoupled drafter dropped an unserved control request (%s): "
                    "the drafter runs no user requests",
                    type(recv_req).__name__,
                )
                continue
            self.send_to_tokenizer.send_output(self._profile(recv_req), recv_req)

    def step_profiler(self) -> None:
        """Round-boundary profiler step (the drafter's round == a forward)."""
        self.profiler_manager._profile_forward_ct_predicate()

    def _profile(self, recv_req: ProfileReq) -> ProfileReqOutput:
        """Serve one profiler request, reporting refusals to the HTTP caller
        instead of killing the drafter loop with them."""
        if recv_req.profile_by_stage:
            return ProfileReqOutput(
                success=False,
                message=(
                    "profile_by_stage is not supported on the decoupled drafter: "
                    "its unit of work is one enumeration round, which has no "
                    "prefill / decode stage split to separate traces by."
                ),
            )
        output = self.profiler_manager._profile(recv_req)
        if output is None:
            return ProfileReqOutput(success=True, message="Succeeded")
        return output


class DecoupledDraftManager:
    """Drafter engine driver: controls in, enumeration blocks out."""

    def __init__(
        self,
        *,
        ipc_config: DecoupledSpecIpcConfig,
        model_runner: ModelRunner,
        num_steps: int,
        fanout: int,
        data_transport: str = "zmq",
    ) -> None:
        import zmq

        self.ipc_config = ipc_config
        self.num_steps = int(num_steps)
        self.fanout = int(fanout)
        self.engine = EnumDraftEngine(
            model_runner=model_runner,
            num_steps=self.num_steps,
            fanout=self.fanout,
        )
        transport = build_transport(
            kind=DecoupledSpecTransportKind.ZMQ,
            bind_endpoint=ipc_config.bind_endpoint,
            connect_endpoints=ipc_config.connect_endpoints,
            context=zmq.Context(2),
        )
        # GPU commit mirror + host arrival board (the drafter-side symmetry
        # of the verifier's enum buffer + board): the IPC thread lands every
        # commit's values per seat and notifies the board; the unified bet
        # round's scatter kernel reads the mirror, and the commit gate's
        # callback waits on the board.
        self.commit_mirror = DrafterCommitMirror(
            num_steps=self.num_steps, device=self.engine.device
        )
        self.commit_arrival_board = EnumArrivalBoard()
        self.engine.attach_commit_mirror(self.commit_mirror, self.commit_arrival_board)
        self.ipc_thread = DrafterIpcThread(
            transport=transport,
            drafter_rank=ipc_config.rank,
            commit_mirror=self.commit_mirror,
            on_commits_landed=self._on_commits_landed,
        )
        self.ipc_thread.start()
        self._round_ct = 0
        self._round_time_s = 0.0
        self._push_time_s = 0.0
        # Idle accounting (the prerun / 1:N capacity question: is the drafter
        # starved waiting for commits, or saturated?). Only counted while at
        # least one seat is open, so pre-traffic polling never inflates it.
        self._open_seats = 0
        self._idle_time_s = 0.0
        self._starved_round_ct = 0
        self._idle_since: Optional[float] = None
        # Top-1 prerun rides the ZMQ block message's speculative flag; the
        # CUDA IPC row header has no flag word yet.
        self._enable_top1_prerun = (
            envs.SGLANG_ENABLE_DECOUPLED_TOP1_PRERUN.get() and data_transport == "zmq"
        )
        # Seats eligible for an idle-window bet (filled after each answered
        # commit, consumed by _run_preruns).
        self._prerun_keys: dict[DraftReqKey, None] = {}
        # Prep-ahead (idle-window fast-round skeleton). Mutually exclusive
        # with the prerun bet: both want the idle window, and the bet already
        # subsumes the skeleton when it hits.
        self._prep_ahead = (
            envs.SGLANG_ENABLE_DECOUPLED_PREP_AHEAD.get()
            and not envs.SGLANG_ENABLE_DECOUPLED_TOP1_PRERUN.get()
        )
        self._prep_keys: dict[DraftReqKey, None] = {}
        self._misaligned_ct = 0
        self._mirror_probe_ct = 0
        # Adaptive fanout: keep the round time inside the verifier's enum-wait
        # budget by halving / restoring the engine's effective width. Only
        # meaningful under a positive wait gate (sync pacing).
        wait_ms = envs.SGLANG_DECOUPLED_ENUM_WAIT_MS.get()
        self._adaptive_fanout = (
            envs.SGLANG_ENABLE_DECOUPLED_ADAPTIVE_FANOUT.get()
            and wait_ms > 0
            and self.fanout > 1
        )
        self._fanout_budget_ms = 0.75 * wait_ms
        self._round_ewma_ms: float | None = None
        self._rounds_since_fanout_change = 0
        # Evented push (ZMQ data plane): pinned staging ring + CUDA event
        # consumed on the IPC thread, instead of a blocking D2H here.
        self._push_ring = (
            PushStagingRing(num_slots=4)
            if envs.SGLANG_ENABLE_DECOUPLED_EVENTED_PUSH.get()
            and data_transport == "zmq"
            else None
        )

        self.ipc_block_pool = None
        if data_transport == "cuda_ipc":
            from sglang.srt.speculative.cuda_ipc_enum_transport import (
                CudaIpcEnumBlockPool,
            )

            unit_width = self.num_steps + 1
            self.ipc_block_pool = CudaIpcEnumBlockPool(
                device=model_runner.device,
                # Both sides derive the shm rendezvous name from the drafter's
                # bind endpoint.
                endpoint=ipc_config.bind_endpoint,
                max_rows=IPC_POOL_MAX_ROWS,
                row_width=2 + unit_width * self.fanout * unit_width,
            )

    @property
    def round_ct(self) -> int:
        """Enumeration rounds run so far -- the drafter's forward counter."""
        return self._round_ct

    @staticmethod
    def _die_if_sticky_cuda(exc: Exception) -> None:
        """A sticky CUDA failure (device-side assert / illegal access) poisons
        the context permanently: the belts' continue-on-error would turn this
        process into an unkillable zombie that holds GPU memory and the IPC
        endpoint while spamming the log. Better dead than undead."""
        text = str(exc)
        if "device-side assert" in text or "illegal memory access" in text:
            logger.critical(
                "sticky CUDA failure in the drafter loop; exiting: %s", text
            )
            os._exit(70)

    def _probe_commit_mirror(self, segment) -> None:
        """Debug-only (device sync): when this host apply consumes exactly
        the mirror's LATEST landed commit, its values must match."""
        seat = self.engine.seat_of(segment.draft_key)
        if seat is None:
            return
        new_len = int(segment.pre_verify_committed_len) + len(segment.committed_tokens)
        board = self.commit_arrival_board
        with board._cond:
            latest = board._stamps.get(seat)
        if latest != new_len or len(segment.round_lens) != 1:
            return  # merged/lagged segment: the mirror holds a newer commit
        torch.cuda.synchronize()
        got_len = int(self.commit_mirror.new_committed_lens[seat].item())
        got_delta = int(self.commit_mirror.delta_lens[seat].item())
        got_tokens = self.commit_mirror.tokens[seat, :got_delta].tolist()
        want = [int(t) for t in segment.committed_tokens]
        ok = got_len == new_len and got_tokens == want
        self._mirror_probe_ct += 1
        if not ok or self._mirror_probe_ct <= 3 or self._mirror_probe_ct % 200 == 0:
            log = logger.info if ok else logger.error
            log(
                "commit-mirror probe #%d %s: seat=%d gpu(len=%d, delta=%s) "
                "host(len=%d, delta=%s)",
                self._mirror_probe_ct,
                "OK" if ok else "MISMATCH",
                seat,
                got_len,
                got_tokens[:6],
                new_len,
                want[:6],
            )

    def _on_commits_landed(self, commits) -> None:
        """IPC-thread hook, after the landing event was recorded: publish
        each seat's new committed total to the host board (the commit gate's
        wakeup source)."""
        self.commit_arrival_board.record_pairs(
            [int(c.req_pool_idx) for c in commits],
            [
                int(c.pre_verify_committed_len) + len(c.committed_tokens)
                for c in commits
            ],
        )
        if self.commit_mirror is not None:
            for c in commits:
                seat = int(c.req_pool_idx)
                self.commit_arrival_board.record_generation(
                    seat, self.commit_mirror.host_generation(seat)
                )

    def run_loop(self, *, control_plane: DrafterControlPlane) -> None:
        """The drafter scheduler's event loop (never returns)."""
        logger.info(
            "Decoupled drafter loop started (rank=%d, K=%d, F=%d)",
            self.ipc_config.rank,
            self.num_steps,
            self.fanout,
        )
        while True:
            # Profiler control in / trace window step out. Both run on every
            # iteration, not only on busy ones, so an armed window still
            # closes (and exports) once the traffic that armed it stops.
            control_plane.poll()
            control_plane.step_profiler()
            ready = self.ipc_thread.collect_ready_draft_controls(
                lambda inbox: inbox.extract_ready_controls_locked(
                    self._consumable_commit_len
                )
            )
            if ready.is_empty():
                # Idle window = the verifier's in-flight round: the only time
                # a top-1 prerun may run. Betting inline after a real round
                # would delay draining the next commit and stall the pipeline.
                # Idle-window work must not be able to kill the drafter for
                # every request (same contract as the round path below): a
                # failed bet / flush / prebuild only costs its own benefit.
                try:
                    if self._enable_top1_prerun and self._prerun_keys:
                        self._run_preruns()
                    else:
                        if self._open_seats > 0 and self._idle_since is None:
                            self._idle_since = time.monotonic()
                        # Idle window: only touch the device once the stream
                        # actually drained. Right after a round's CPU returns
                        # the GPU still has ~2ms of tail queued, and a sync
                        # (allocator free's torch.unique, any pageable copy)
                        # issued then blocks THIS LOOP on the full drain --
                        # freezing commit processing. Deferring to the next
                        # idle tick costs nothing: the wait sleep is 50us.
                        if not torch.cuda.current_stream().query():
                            time.sleep(_IDLE_WAIT_S)
                            continue
                        self.engine.flush_scratch_frees()
                        if self._prep_ahead and self._prep_keys:
                            by_verifier: dict[int, list[DraftReqKey]] = {}
                            for draft_key in self._prep_keys:
                                by_verifier.setdefault(
                                    draft_key.src_verifier_rank, []
                                ).append(draft_key)
                            self._prep_keys.clear()
                            for group in by_verifier.values():
                                self.engine.prebuild_fast_round(group)
                                self.engine.pre_launch_extend(group)
                        time.sleep(_IDLE_WAIT_S)
                except Exception as exc:
                    self._die_if_sticky_cuda(exc)
                    logger.exception(
                        "decoupled drafter idle-window task failed; continuing"
                    )
                    time.sleep(_IDLE_WAIT_S)
                continue
            try:
                self._apply_controls_and_draft(ready)
            except Exception as exc:
                self._die_if_sticky_cuda(exc)
                # A bad round must not kill the drafter for every request; the
                # affected verifier rounds simply fall back. TODO(5c-class):
                # quarantine the offending request instead of best-effort.
                logger.exception("decoupled drafter round failed; controls dropped")

    @staticmethod
    def _consumable_commit_len(segment) -> int:
        """Generation lockstep with a catch-up escape hatch.

        Consuming one verify round's delta per drafter round produces EVERY
        block generation, so the verifier's select always finds the one it
        needs -- merging commits (the old unconditional behavior) skips
        generations and each skip costs the verifier a fallback round; under
        the overlap scheduler those fast fallback rounds outrun the drafter
        and the skips cascade. A small backlog is normal there (commits flow
        while a round is in flight); only when the drafter genuinely fell
        behind does merging the whole backlog become right: one jump, one
        fallback, re-locked -- instead of dragging a permanent lag whose gate
        wait would eventually exceed the budget and cascade anyway.
        """
        if segment.pending_rounds > _CATCH_UP_BACKLOG_ROUNDS:
            _MERGE_STATS["merged"] += 1
            _MERGE_STATS["skipped_rounds"] += segment.pending_rounds - 1
            return len(segment.committed_tokens)
        _MERGE_STATS["lockstep"] += 1
        return segment.round_lens[0] if segment.round_lens else 0

    def _apply_controls_and_draft(self, ready) -> None:
        # Close the idle window this batch of controls ended (see the idle
        # accounting fields): everything below is drafter-busy time.
        idle_s = 0.0
        if self._idle_since is not None:
            idle_s = time.monotonic() - self._idle_since
            self._idle_since = None
            self._idle_time_s += idle_s
        stage = self.engine.profiler.stage
        touched: dict[DraftReqKey, None] = {}
        confirmed: dict[DraftReqKey, None] = {}
        with stage("apply-controls"):
            for draft_key in ready.close_keys:
                self.engine.close(draft_key)
                self._open_seats = max(0, self._open_seats - 1)
            for sync in ready.sync_messages:
                self.engine.open(
                    sync.draft_key,
                    req_pool_idx=int(sync.req_pool_idx),
                    prompt_tokens=list(sync.prompt_token_ids),
                    committed_outputs=list(sync.committed_outputs),
                )
                touched[sync.draft_key] = None
                self._open_seats += 1
            for segment in ready.ready_commit_segments:
                if not self.engine.has(segment.draft_key):
                    continue
                self.engine.note_commit_seen(segment.draft_key)
                if not self.engine.commit_base_aligned(
                    segment.draft_key, int(segment.pre_verify_committed_len)
                ):
                    # Stale (pre-resync) or gapped segment: applying it would
                    # corrupt the committed mirror. Drop it -- the seat rides
                    # fallbacks until the verifier's streak resync re-seeds.
                    self._misaligned_ct += 1
                    if self._misaligned_ct <= 5:
                        logger.warning(
                            "decoupled drafter: dropped misaligned commit "
                            "segment #%d for %s (base=%d)",
                            self._misaligned_ct,
                            segment.draft_key.request_id,
                            int(segment.pre_verify_committed_len),
                        )
                    continue
                if envs.SGLANG_DEBUG_DECOUPLED_COMMIT_MIRROR.get():
                    self._probe_commit_mirror(segment)
                if self.engine.apply_commit(
                    segment.draft_key, list(segment.committed_tokens)
                ):
                    # A confirmed top-1 prerun: this seat's next block is
                    # already on the verifier; nothing to draft for this commit.
                    confirmed[segment.draft_key] = None
                else:
                    touched[segment.draft_key] = None
        if not touched and not confirmed:
            return
        # One block per owning verifier (1:1 today: a single peer).
        by_verifier: dict[int, list[DraftReqKey]] = {}
        for draft_key in touched:
            by_verifier.setdefault(draft_key.src_verifier_rank, []).append(draft_key)
        for verifier_rank, draft_keys in by_verifier.items():
            round_start = time.monotonic()
            with stage(f"round[bs={len(draft_keys)}]"):
                packed = self.engine.draft_round(draft_keys)
            round_s = time.monotonic() - round_start
            self._round_ct += 1
            self._round_time_s += round_s
            if idle_s > 0.0:
                self._starved_round_ct += 1
                # Only the first round of a control batch waited; later ones
                # (1:N, multi-verifier) started with work already queued.
                idle_s = 0.0
            self._maybe_adjust_fanout(round_ms=1000.0 * round_s)
            if self._round_ct % 200 == 0:
                logger.info(
                    "decoupled drafter rounds: ct=%d avg_ms=%.1f push_ms=%.2f "
                    "idle_ms=%.2f starved=%.0f%% last_bs=%d fast=%d slow=%d "
                    "eff_fanout=%d prerun_hit=%d prerun_miss=%d merge=%d "
                    "skips=%d lockstep=%d",
                    self._round_ct,
                    1000.0 * self._round_time_s / self._round_ct,
                    1000.0 * self._push_time_s / self._round_ct,
                    1000.0 * self._idle_time_s / self._round_ct,
                    100.0 * self._starved_round_ct / self._round_ct,
                    len(draft_keys),
                    self.engine.hit_ct,
                    self.engine.miss_ct,
                    self.engine.effective_fanout,
                    self.engine.prerun_hit_ct,
                    self.engine.prerun_miss_ct,
                    _MERGE_STATS["merged"],
                    _MERGE_STATS["skipped_rounds"],
                    _MERGE_STATS["lockstep"],
                )
                if self.engine.profiler.enabled:
                    logger.info(
                        "decoupled drafter round breakdown: %s",
                        self.engine.profiler.summary(),
                    )
            with stage("push-block"):
                self._push_block(verifier_rank=verifier_rank, packed=packed)
        if self._enable_top1_prerun:
            for draft_key in list(touched) + list(confirmed):
                self._prerun_keys[draft_key] = None
        if self._prep_ahead:
            # Round-tail restage, hit and miss alike: the skeleton anchors on
            # the POST-round committed prefix, so it is valid for whatever the
            # next commit brings; a miss no longer strands the next fast
            # round without a prebuild. The build rides the engine's
            # dedicated prebuild stream (no barrier on this round's GPU
            # tail), so it does not need the idle window's drain gate -- the
            # idle-window path below stays as a no-op fallback
            # (_prebuilt_fast is already staged).
            by_verifier_prep: dict[int, list[DraftReqKey]] = {}
            for draft_key in touched:
                by_verifier_prep.setdefault(draft_key.src_verifier_rank, []).append(
                    draft_key
                )
            for group in by_verifier_prep.values():
                try:
                    self.engine.prebuild_fast_round(group)
                    self.engine.pre_launch_extend(group)
                except Exception:
                    logger.exception(
                        "decoupled round-tail restage failed; next round "
                        "builds inline"
                    )

    def _maybe_adjust_fanout(self, *, round_ms: float) -> None:
        """Feedback controller for the engine's effective fanout.

        Halve the enumeration width when the round-time EWMA threatens the
        verifier's enum-wait budget (a blown gate collapses the accept length
        batch-wide -- far worse than a narrower block); restore it once rounds
        run comfortably inside the budget. The 0.35 restore threshold plus the
        cooldown gives ~2x hysteresis, so the controller settles instead of
        oscillating around the budget.
        """
        if not self._adaptive_fanout:
            return
        ewma = self._round_ewma_ms
        self._round_ewma_ms = round_ms if ewma is None else 0.7 * ewma + 0.3 * round_ms
        self._rounds_since_fanout_change += 1
        if self._rounds_since_fanout_change < 8:
            return
        current = self.engine.effective_fanout
        new_fanout = current
        if self._round_ewma_ms > self._fanout_budget_ms and current > 1:
            new_fanout = max(1, current // 2)
        elif (
            self._round_ewma_ms < 0.35 * self._fanout_budget_ms
            and current < self.fanout
        ):
            new_fanout = min(self.fanout, current * 2)
        if new_fanout == current:
            return
        logger.info(
            "decoupled adaptive fanout: %d -> %d (round_ewma=%.1fms budget=%.1fms)",
            current,
            new_fanout,
            self._round_ewma_ms,
            self._fanout_budget_ms,
        )
        self.engine.effective_fanout = new_fanout
        self._rounds_since_fanout_change = 0
        self._round_ewma_ms = None  # re-learn at the new width

    def _run_preruns(self) -> None:
        """Idle-window top-1 bets for the seats whose last commit was already
        answered (real block pushed or bet confirmed)."""
        by_verifier: dict[int, list[DraftReqKey]] = {}
        for draft_key in self._prerun_keys:
            by_verifier.setdefault(draft_key.src_verifier_rank, []).append(draft_key)
        self._prerun_keys.clear()
        for verifier_rank, draft_keys in by_verifier.items():
            packed = self.engine.speculative_prerun(draft_keys)
            self._push_block(
                verifier_rank=verifier_rank, packed=packed, speculative=True
            )

    def _push_block(
        self, *, verifier_rank: int, packed, speculative: bool = False
    ) -> None:
        if packed is None:
            return
        push_start = time.monotonic()
        if self.ipc_block_pool is not None:
            # CUDA IPC data plane: D2D into the shared pool; the shm flag
            # bump after the device sync is the arrival signal. (Preruns are
            # ZMQ-only and gated off for this plane.)
            self.ipc_block_pool.push(
                pool_indices=packed["pool_indices"],
                base_committed_lens=packed["base_committed_lens"],
                units=packed["units_device"],
            )
            self._push_time_s += time.monotonic() - push_start
            return
        header = DraftEnumerationBufferBatch(
            src_drafter_rank=self.ipc_config.rank,
            dst_verifier_rank=verifier_rank,
            num_steps=self.num_steps,
            fanout=self.fanout,
            pool_indices=packed["pool_indices"],
            base_committed_lens=packed["base_committed_lens"],
            speculative=speculative,
        )
        units = packed["units_device"]
        if self._push_ring is not None:
            # Evented push: enqueue the pinned staging copy, record its event,
            # and return without waiting for the round's GPU chain to drain --
            # the IPC thread materializes and sends once the event fires.
            num_tokens = units.numel()
            slot = self._push_ring.acquire(num_tokens=num_tokens)
            if slot is not None:
                slot.buffer[:num_tokens].copy_(units.reshape(-1), non_blocking=True)
                event = torch.cuda.Event()
                event.record()
                self.ipc_thread.submit_evented_draft_results(
                    EventedDraftBlock(
                        header=header,
                        event=event,
                        buffer=slot.buffer,
                        num_tokens=num_tokens,
                        on_sent=partial(self._push_ring.release, slot),
                    )
                )
                self._push_time_s += time.monotonic() - push_start
                return
            # Ring exhausted (not expected at one block per round): fall back
            # to an inline D2H, but ride the same FIFO so per-seat generation
            # order is preserved on the wire.
            header.tokens = tuple(units.to("cpu").reshape(-1).tolist())
            self.ipc_thread.submit_evented_draft_results(
                EventedDraftBlock(
                    header=header,
                    event=None,
                    buffer=None,
                    num_tokens=0,
                    on_sent=None,
                )
            )
            self._push_time_s += time.monotonic() - push_start
            return
        header.tokens = tuple(units.to("cpu").reshape(-1).tolist())
        header.sent_unix_ts = time.time()
        self.ipc_thread.submit_draft_results(header)
        self._push_time_s += time.monotonic() - push_start

    def close(self) -> None:
        self.ipc_thread.close()
