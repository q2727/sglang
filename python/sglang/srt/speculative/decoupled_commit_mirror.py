"""Drafter-side GPU landing of VerifyCommits (the commit mirror).

The drafter's host inbox stays the source of truth for lifecycle and pacing;
this mirror is the VALUE plane the unified bet round reads on GPU: per seat,
the latest commit's delta tokens, its length, the resulting committed total
(the stamp the next block must carry) and a landing generation counter (the
pre-launched scatter kernel's stamp check compares against the generation
expected at enqueue -- a mismatch means the gate timed out and the round
must redirect itself to the junk lane).

Written by the drafter's IPC thread (pinned staging + async copies on a
dedicated landing stream + one event per landing), mirrored on the host by
an arrival board (seat -> new committed total) whose condvar is what the
commit gate's host-func callback waits on. All device work here is pinned /
kernel / event -- the classes of calls the stream-gate probe showed keep
working while a gate blocks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.speculative.decoupled_spec_io import VerifyCommit

logger = logging.getLogger(__name__)

# Seats are verifier req_to_token rows; the decoupled verifier caps running
# requests far below this. Commits routed to seats past the cap only skip the
# MIRROR (the host inbox still applies them) -- log-once and degrade.
_MAX_SEATS = 4096

# Pinned staging slots (a landing writes one slot; reuse is gated on the
# slot's event, which in steady state fired long ago).
_RING = 8


class DrafterCommitMirror:
    """Per-seat GPU mirror of the latest VerifyCommit (see module note)."""

    def __init__(self, *, num_steps: int, device: str) -> None:
        self.width = num_steps + 1
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        # Landing generation per seat: bumped once per landed commit. int32
        # (the scatter kernel's compare key; wraps are unreachable in-life).
        self.generations = torch.zeros(_MAX_SEATS, dtype=torch.int64, device=device)
        # The commit's resulting committed TOTAL (pre_len + len(delta), in
        # output-token terms): the base stamp the next block must carry.
        self.new_committed_lens = torch.zeros(
            _MAX_SEATS, dtype=torch.int64, device=device
        )
        self.delta_lens = torch.zeros(_MAX_SEATS, dtype=torch.int64, device=device)
        self.tokens = torch.zeros(
            _MAX_SEATS, self.width, dtype=torch.int64, device=device
        )
        self._host_generations = [0] * _MAX_SEATS
        self._pin_meta = [
            torch.empty(3, dtype=torch.int64, pin_memory=True) for _ in range(_RING)
        ]
        self._pin_tokens = [
            torch.empty(self.width, dtype=torch.int64, pin_memory=True)
            for _ in range(_RING)
        ]
        self._slot_events = [torch.cuda.Event() for _ in range(_RING)]
        self._slot_used = [False] * _RING
        self._next_slot = 0
        # Recorded after every landing batch on the landing stream; a
        # consumer orders its reads with wait_event (record-before-wait: the
        # gate's host release happens strictly after this record's enqueue).
        self.land_event = torch.cuda.Event()
        self._skipped_ct = 0

    def land(self, commit: VerifyCommit) -> None:
        """Stage one commit's values into the seat's mirror row (async on the
        landing stream). Host generation bookkeeping is immediate; the
        caller notifies its arrival board after the batch's record()."""
        seat = int(commit.req_pool_idx)
        delta = commit.committed_tokens
        if seat < 0 or seat >= _MAX_SEATS or len(delta) > self.width:
            self._skipped_ct += 1
            if self._skipped_ct <= 3:
                logger.warning(
                    "commit mirror skip (seat=%d delta_len=%d width=%d) -- "
                    "host inbox still applies it",
                    seat,
                    len(delta),
                    self.width,
                )
            return
        slot = self._next_slot
        self._next_slot = (slot + 1) % _RING
        if self._slot_used[slot]:
            self._slot_events[slot].synchronize()
        generation = self._host_generations[seat] + 1
        self._host_generations[seat] = generation
        meta = self._pin_meta[slot]
        meta[0] = generation
        meta[1] = int(commit.pre_verify_committed_len) + len(delta)
        meta[2] = len(delta)
        pin_tokens = self._pin_tokens[slot]
        pin_tokens[: len(delta)].copy_(torch.tensor(delta, dtype=torch.int64))
        with torch.cuda.stream(self.stream):
            # Value rows first, generation LAST: the generation write is the
            # release the scatter kernel's stamp check acquires on.
            self.tokens[seat, : len(delta)].copy_(
                pin_tokens[: len(delta)], non_blocking=True
            )
            self.new_committed_lens[seat : seat + 1].copy_(meta[1:2], non_blocking=True)
            self.delta_lens[seat : seat + 1].copy_(meta[2:3], non_blocking=True)
            self.generations[seat : seat + 1].copy_(meta[0:1], non_blocking=True)
            self._slot_events[slot].record(self.stream)
        self._slot_used[slot] = True

    def record_landing(self) -> None:
        """Record the batch-level landing event (call once after a batch of
        land() calls, before notifying the host arrival board)."""
        self.land_event.record(self.stream)

    def host_generation(self, seat: int) -> int:
        return self._host_generations[seat] if 0 <= seat < _MAX_SEATS else 0
