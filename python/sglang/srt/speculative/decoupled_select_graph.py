"""Post-gate select as ONE CUDA graph for the decoupled verifier.

After the C6 gate releases, the verify round's critical path is the enum
select (a two-level pool-index gather plus ~15 elementwise/gather dust
kernels from ``select_enum_units``) followed by the target-verify graph
launch. This runner captures the whole select -- buffer gather, generation
match, unit pick, fallback blend -- into one graph per (batch size,
read-slot) bucket, so the post-gate host work shrinks to four static-buffer
copies and one ``graph.replay()``.

It is also the doorbell's landing zone: a device-side wait-value node
prepended to this graph (A12 recipe) turns the C6 host wait into a GPU-side
gate, letting the select + verify launch before the enumeration block
arrives.

Graph discipline (pitfalls A10/A11/A13):
- Every per-round input goes through a bucket static buffer: the pool rows
  AND the three select keys.
- The enum buffer's read slot is a PYTHON attribute resolved at record time,
  so buckets are keyed by (bs, read_slot); under double-buffering each slot
  records its own graph.
- Capture records without executing: the first call captures, then falls
  through to the replay that does the round's real work.
- The hits output is a static buffer overwritten by the next replay; the
  accounting consumer runs on a deferred hook, so callers must clone before
  queueing.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class _SelectBucket:
    def __init__(self, *, bs: int, unit_width: int, device: str) -> None:
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.req_rows = torch.zeros(bs, dtype=torch.int64, device=device)
        self.bonus_tokens = torch.zeros(bs, dtype=torch.int32, device=device)
        self.prev_accept_lens = torch.zeros(bs, dtype=torch.int64, device=device)
        self.base_committed_lens = torch.zeros(bs, dtype=torch.int64, device=device)
        self.selected = torch.zeros(bs, unit_width, dtype=torch.int64, device=device)
        self.hits = torch.zeros(bs, dtype=torch.bool, device=device)


class SelectGraphRunner:
    """Verifier-private runner: one captured select graph per (bs, slot)."""

    def __init__(
        self,
        *,
        enum_buffer,
        num_cases: int,
        fanout: int,
        unit_width: int,
        device: str,
    ) -> None:
        self.enum_buffer = enum_buffer
        self.num_cases = num_cases
        self.fanout = fanout
        self.unit_width = unit_width
        self.device = device
        self._buckets: dict[tuple[int, int], _SelectBucket] = {}
        self._failed: set[tuple[int, int]] = set()

    def run(
        self,
        *,
        req_pool_indices: torch.Tensor,
        bonus_tokens: torch.Tensor,
        prev_accept_lens: torch.Tensor,
        base_committed_lens: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """One select round: (selected [bs, unit_width], hits [bs]) from the
        bucket's static outputs, or None when this bucket permanently fell
        back after a capture failure. ``hits`` is overwritten by the next
        call -- clone before deferring."""
        bs = int(req_pool_indices.shape[0])
        key = (bs, self.enum_buffer.read_slot)
        if key in self._failed:
            return None
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _SelectBucket(
                bs=bs, unit_width=self.unit_width, device=self.device
            )
            self._buckets[key] = bucket
        bucket.req_rows.copy_(req_pool_indices)
        bucket.bonus_tokens.copy_(bonus_tokens)
        bucket.prev_accept_lens.copy_(prev_accept_lens)
        bucket.base_committed_lens.copy_(base_committed_lens)
        if bucket.graph is None and not self._try_capture(key=key, bucket=bucket):
            return None
        # First call: capture RECORDED without executing, so this replay is
        # the round's real select. Later calls: plain replay.
        bucket.graph.replay()
        return bucket.selected, bucket.hits

    def _try_capture(self, *, key: tuple[int, int], bucket: _SelectBucket) -> bool:
        from sglang.srt.speculative.verify_worker import select_enum_units

        graph = torch.cuda.CUDAGraph()
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    rows, stamps = self.enum_buffer.gather(bucket.req_rows)
                    selected, hits = select_enum_units(
                        rows,
                        stamps,
                        bonus_tokens=bucket.bonus_tokens,
                        prev_accept_lens=bucket.prev_accept_lens,
                        base_committed_lens=bucket.base_committed_lens,
                        num_cases=self.num_cases,
                        fanout=self.fanout,
                        unit_width=self.unit_width,
                    )
                    bucket.selected.copy_(selected)
                    bucket.hits.copy_(hits)
            torch.cuda.current_stream().wait_stream(stream)
            bucket.graph = graph
            return True
        except Exception:
            logger.exception(
                "select-graph capture failed for (bs, slot)=%s; falling back "
                "to the eager select permanently for this bucket",
                key,
            )
            self._failed.add(key)
            self._buckets.pop(key, None)
            return False
