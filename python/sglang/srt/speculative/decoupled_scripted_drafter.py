"""In-process scripted fake drafter for the decoupled-spec loopback lock.

Runs inside the verifier scheduler process, on the fake transport, and answers
the real control plane (DraftSync / VerifyCommit / DraftClose) with
deterministic enumeration blocks -- no draft model. This is the single-process
correctness harness: with ANY drafter behavior the verifier's committed output
must equal the non-speculative baseline token for token (a wrong or absent
chain only ever costs a fallback round), so the scripted behaviors pin the two
deterministic degradation paths:

- ``garbage``: stamps are correct (the block always looks fresh) but every
  guess is a constant token, so the bonus match misses -> every round falls
  back through the guess-miss path.
- ``stale``: chains are the same, stamps lag one behind -> every round falls
  back through the staleness path, before any guess is consulted.

Blocks are paced exactly like the real drafter (one block per sync / commit),
so the sync-mode arrival gate is exercised too.
"""

from __future__ import annotations

import logging
import threading

from sglang.srt.speculative.decoupled_spec_io import (
    DraftEnumerationBufferBatch,
    DraftReqKey,
)
from sglang.srt.speculative.decoupled_spec_transport import BaseDecoupledSpecTransport
from sglang.srt.speculative.drafter_ipc_thread import DrafterIpcThread

logger = logging.getLogger(__name__)

SCRIPTED_DRAFTER_MODES = ("garbage", "stale")

# Any in-vocab token id works: a garbage unit is only consumed if selected,
# selection requires guess == real bonus, and even then verify only accepts
# target-agreeing tokens.
_GARBAGE_TOKEN = 7

_IDLE_WAIT_S = 0.001


class _MirrorState:
    def __init__(self, *, pool_idx: int, total_committed_len: int, epoch: int) -> None:
        self.pool_idx = pool_idx
        self.total_committed_len = total_committed_len
        self.epoch = epoch


class ScriptedFakeDrafter:
    """Deterministic drafter stand-in over an injected (fake) transport."""

    def __init__(
        self,
        *,
        transport: BaseDecoupledSpecTransport,
        verifier_rank: int,
        drafter_rank: int,
        num_steps: int,
        fanout: int,
        mode: str,
    ) -> None:
        if mode not in SCRIPTED_DRAFTER_MODES:
            raise ValueError(
                f"Unknown scripted drafter mode: {mode!r} "
                f"(expected one of {SCRIPTED_DRAFTER_MODES})"
            )
        self.verifier_rank = int(verifier_rank)
        self.drafter_rank = int(drafter_rank)
        self.num_steps = int(num_steps)
        self.fanout = int(fanout)
        self.mode = mode
        self.ipc_thread = DrafterIpcThread(
            transport=transport, drafter_rank=drafter_rank
        )
        self._states: dict[DraftReqKey, _MirrorState] = {}
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="sglang-scripted-fake-drafter", daemon=True
        )

    def start(self) -> None:
        self.ipc_thread.start()
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.ipc_thread.close()

    def _run(self) -> None:
        while not self._closed.is_set():
            ready = self.ipc_thread.collect_ready_draft_controls(
                lambda inbox: inbox.extract_ready_controls_locked(
                    lambda segment: len(segment.committed_tokens)
                )
            )
            if ready.is_empty():
                self._closed.wait(timeout=_IDLE_WAIT_S)
                continue
            touched: list[DraftReqKey] = []
            for draft_key in ready.close_keys:
                self._states.pop(draft_key, None)
            for sync in ready.sync_messages:
                self._states[sync.draft_key] = _MirrorState(
                    pool_idx=int(sync.req_pool_idx),
                    epoch=int(sync.epoch),
                    total_committed_len=(
                        len(sync.prompt_token_ids) + len(sync.committed_outputs)
                    ),
                )
                touched.append(sync.draft_key)
            for segment in ready.ready_commit_segments:
                state = self._states.get(segment.draft_key)
                if state is None:
                    continue
                state.total_committed_len += len(segment.committed_tokens)
                touched.append(segment.draft_key)
            if touched:
                self.ipc_thread.submit_draft_results(self._build_block(touched))

    def _build_block(
        self, draft_keys: list[DraftReqKey]
    ) -> DraftEnumerationBufferBatch:
        pool_indices: list[int] = []
        base_committed_lens: list[int] = []
        epochs: list[int] = []
        seen: set[int] = set()
        for draft_key in draft_keys:
            state = self._states.get(draft_key)
            if state is None or state.pool_idx in seen:
                continue
            seen.add(state.pool_idx)
            pool_indices.append(state.pool_idx)
            epochs.append(state.epoch)
            stamp = state.total_committed_len
            if self.mode == "stale":
                stamp -= 1  # always one behind -> deterministic staleness miss
            base_committed_lens.append(stamp)
        unit_width = self.num_steps + 1
        row_stride = unit_width * self.fanout * unit_width
        tokens = (_GARBAGE_TOKEN,) * (len(pool_indices) * row_stride)
        return DraftEnumerationBufferBatch(
            src_drafter_rank=self.drafter_rank,
            dst_verifier_rank=self.verifier_rank,
            num_steps=self.num_steps,
            fanout=self.fanout,
            pool_indices=pool_indices,
            base_committed_lens=base_committed_lens,
            epochs=epochs,
            tokens=tokens,
        )
