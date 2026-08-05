"""Drafter-side GPU landing of VerifyCommits (the commit mirror).

The drafter's host inbox stays the source of truth for lifecycle and pacing;
this mirror is the VALUE plane the pre-launched round reads on GPU: per seat,
the latest commits' raw wire segments (absolute base + delta tokens).

Self-consistent by construction -- the consumer kernel judges a slot purely
by arithmetic between the slot's OWN pre_len and the committed base sampled
when the consumer was enqueued (exactly the verifier-select stamp recipe):

- pre_len + wire <= base  -> already-consumed / re-sent segment -> junk
- pre_len <= base < pre_len + wire -> the seat's next segment -> consume tail
- pre_len > base          -> a later segment (burst overwrote) -> junk

so a landing that is late, early, or not yet visible degrades to one junked
round (the host redoes it) and can never be double-consumed or torn into a
wrong verdict. No cross-thread generation matching, no landing fence, no
publish-side sync is needed for correctness.

Each seat keeps TWO slots (landing-order alternation) so an in-flight
consumer never races the landing that overwrites the segment it may still
be reading; each slot is a seqlock row

    [seq0 | pre_len | wire_len | tok_0 .. tok_{W-1} | seq1]

written as three ordered copies on the landing stream: seq0 first, payload,
seq1 last. A reader accepts a slot only when seq0 == seq1 > 0, so any
overwrite in progress (which bumps seq0 first) invalidates the slot instead
of tearing it.
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

# Landing-order slot alternation per seat.
_SLOTS = 2

# Pinned staging rows (a landing writes one row; reuse is gated on the row's
# event, which in steady state fired long ago).
_RING = 8


class DrafterCommitMirror:
    """Per-seat GPU mirror of the latest VerifyCommits (see module note)."""

    def __init__(self, *, num_steps: int, device: str) -> None:
        self.width = num_steps + 1
        # Row layout: [seq0, pre_len, wire_len, tok*W, seq1].
        self.row_words = self.width + 4
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.rows = torch.zeros(
            _MAX_SEATS, _SLOTS, self.row_words, dtype=torch.int64, device=device
        )
        # Per-seat landing count: slot alternation + host-side bookkeeping
        # (probes, seat-reuse baselines). NEVER matched by the kernel.
        self._host_generations = [0] * _MAX_SEATS
        self._pin_rows = [
            torch.empty(self.row_words, dtype=torch.int64, pin_memory=True)
            for _ in range(_RING)
        ]
        self._slot_events = [torch.cuda.Event() for _ in range(_RING)]
        self._slot_used = [False] * _RING
        self._next_slot = 0
        # Recorded after every landing batch on the landing stream; a
        # host-synchronized consumer (debug probes) orders its reads with
        # wait_event.
        self.land_event = torch.cuda.Event()
        self._skipped_ct = 0

    def land(self, commit: VerifyCommit) -> torch.cuda.Event | None:
        """Stage one commit's raw wire segment into the seat's next slot
        (async on the landing stream). Host bookkeeping is immediate."""
        seat = int(commit.req_pool_idx)
        pre_len = int(commit.pre_verify_committed_len)
        delta = list(commit.committed_tokens)
        if seat < 0 or seat >= _MAX_SEATS or not delta or len(delta) > self.width:
            self._skipped_ct += 1
            if self._skipped_ct <= 3:
                logger.warning(
                    "commit mirror skip (seat=%d delta_len=%d width=%d) -- "
                    "host inbox still applies it",
                    seat,
                    len(delta),
                    self.width,
                )
            return None
        ring = self._next_slot
        self._next_slot = (ring + 1) % _RING
        if self._slot_used[ring]:
            self._slot_events[ring].synchronize()
        generation = self._host_generations[seat] + 1
        self._host_generations[seat] = generation
        slot = generation % _SLOTS
        pin = self._pin_rows[ring]
        pin[0] = generation
        pin[1] = pre_len
        pin[2] = len(delta)
        pin[3 : 3 + len(delta)].copy_(torch.tensor(delta, dtype=torch.int64))
        pin[self.row_words - 1] = generation
        row = self.rows[seat, slot]
        with torch.cuda.stream(self.stream):
            # Seqlock write order: seq0 first, payload, seq1 last. A reader
            # that observes seq0 == seq1 == g knows the payload belongs to
            # landing g in one piece; any in-progress overwrite differs.
            row[0:1].copy_(pin[0:1], non_blocking=True)
            row[1 : self.row_words - 1].copy_(
                pin[1 : self.row_words - 1], non_blocking=True
            )
            row[self.row_words - 1 :].copy_(
                pin[self.row_words - 1 :], non_blocking=True
            )
            self._slot_events[ring].record(self.stream)
        self._slot_used[ring] = True
        return self._slot_events[ring]

    def record_landing(self) -> None:
        """Record the batch-level landing event (call once after a batch of
        land() calls, before notifying the host arrival board)."""
        self.land_event.record(self.stream)

    def host_generation(self, seat: int) -> int:
        return self._host_generations[seat] if 0 <= seat < _MAX_SEATS else 0
