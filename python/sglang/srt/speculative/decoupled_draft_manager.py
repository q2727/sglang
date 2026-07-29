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
import time
from functools import partial
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
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
from sglang.srt.speculative.drafter_ipc_thread import (
    DrafterIpcThread,
    EventedDraftBlock,
    PushStagingRing,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_IDLE_WAIT_S = 0.0005

# Commit backlog (in verify rounds) beyond which the drafter merges the whole
# backlog to catch up instead of producing every generation; <= this depth is
# the overlap scheduler's normal in-flight allowance (one in-flight commit
# plus jitter headroom).
_CATCH_UP_BACKLOG_ROUNDS = 2

# CUDA IPC slot capacity in block rows; bounds the verifier batch size a
# single push can carry (the verifier's default running cap is far below it).
IPC_POOL_MAX_ROWS = 256


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
        self.ipc_thread = DrafterIpcThread(
            transport=transport, drafter_rank=ipc_config.rank
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

    def run_loop(self) -> None:
        """The drafter scheduler's event loop (never returns)."""
        logger.info(
            "Decoupled drafter loop started (rank=%d, K=%d, F=%d)",
            self.ipc_config.rank,
            self.num_steps,
            self.fanout,
        )
        while True:
            ready = self.ipc_thread.collect_ready_draft_controls(
                lambda inbox: inbox.extract_ready_controls_locked(
                    self._consumable_commit_len
                )
            )
            if ready.is_empty():
                # Idle window = the verifier's in-flight round: the only time
                # a top-1 prerun may run. Betting inline after a real round
                # would delay draining the next commit and stall the pipeline.
                if self._enable_top1_prerun and self._prerun_keys:
                    self._run_preruns()
                else:
                    if self._open_seats > 0 and self._idle_since is None:
                        self._idle_since = time.monotonic()
                    time.sleep(_IDLE_WAIT_S)
                continue
            try:
                self._apply_controls_and_draft(ready)
            except Exception:
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
            return len(segment.committed_tokens)
        return segment.round_lens[0] if segment.round_lens else 0

    def _apply_controls_and_draft(self, ready) -> None:
        # Close the idle window this batch of controls ended (see the idle
        # accounting fields): everything below is drafter-busy time.
        idle_s = 0.0
        if self._idle_since is not None:
            idle_s = time.monotonic() - self._idle_since
            self._idle_since = None
            self._idle_time_s += idle_s
        for draft_key in ready.close_keys:
            self.engine.close(draft_key)
            self._open_seats = max(0, self._open_seats - 1)
        touched: dict[DraftReqKey, None] = {}
        confirmed: dict[DraftReqKey, None] = {}
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
            if self.engine.apply_commit(
                segment.draft_key, list(segment.committed_tokens)
            ):
                # A confirmed top-1 prerun: this seat's next block is already
                # on the verifier; nothing to draft for this commit.
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
                    "eff_fanout=%d prerun_hit=%d prerun_miss=%d",
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
                )
                if self.engine.profiler.enabled:
                    logger.info(
                        "decoupled drafter round breakdown: %s",
                        self.engine.profiler.summary(),
                    )
            self._push_block(verifier_rank=verifier_rank, packed=packed)
        if self._enable_top1_prerun:
            for draft_key in list(touched) + list(confirmed):
                self._prerun_keys[draft_key] = None

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
