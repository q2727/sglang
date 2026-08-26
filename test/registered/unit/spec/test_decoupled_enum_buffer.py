"""CPU unit tests for the decoupled-spec verifier-side GPU landing buffer.

DecoupledEnumBuffer is where enumeration blocks land on the verifier: each block
row names its own seat (pool_idx echoed from DraftSync), so landing is
validation + one scatter, and per-seat base_committed_len stamps decide
fresh-vs-fallback on the GPU. torch ops here run on CPU tensors, which drives
the identical code path without a GPU: the scatter/gather routing math, the
stamp lifecycle (sentinel -> landed -> reset), and every ingest reject
(wrong verifier / wrong dims / out-of-range seat).
"""

import unittest

import torch

from sglang.srt.speculative.decoupled_enum_buffer import DecoupledEnumBuffer
from sglang.srt.speculative.decoupled_spec_io import DraftEnumerationBufferBatch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_SEATS = 8


class _FakeReqToTokenPool:
    """Only the seat count (req_to_token.shape[0]) is read by the buffer."""

    def __init__(self, seats: int) -> None:
        self.req_to_token = torch.zeros((seats, 4), dtype=torch.int32)


def _buffer(*, num_steps=2, fanout=2, verifier_rank=0) -> DecoupledEnumBuffer:
    return DecoupledEnumBuffer(
        device="cpu",
        req_to_token_pool=_FakeReqToTokenPool(_SEATS),
        num_steps=num_steps,
        fanout=fanout,
        verifier_rank=verifier_rank,
        enable_overlap=False,
    )


def _block(
    pool_indices,
    bases,
    *,
    num_steps=2,
    fanout=2,
    dst_verifier_rank=0,
    tokens=None,
) -> DraftEnumerationBufferBatch:
    row_stride = (num_steps + 1) * fanout * (num_steps + 1)
    if tokens is None:
        tokens = tuple(range(len(pool_indices) * row_stride))
    return DraftEnumerationBufferBatch(
        src_drafter_rank=0,
        dst_verifier_rank=dst_verifier_rank,
        num_steps=num_steps,
        fanout=fanout,
        pool_indices=list(pool_indices),
        base_committed_lens=list(bases),
        epochs=[1] * len(pool_indices),
        tokens=tokens,
    )


def _seat_stamped(rows, stamps, stamp):
    """Return the row of the generation carrying ``stamp`` for batch entry 0."""
    stamp_list = stamps[0].tolist()
    return rows[0][stamp_list.index(stamp)]


class TestDecoupledEnumBufferLanding(CustomTestCase):
    def test_land_scatters_rows_and_stamps_by_pool_indices(self):
        # Routing rides in the block: row i lands in seat pool_indices[i], and
        # the flat C-order token tuple reshapes so the landed row equals
        # block.row_tokens(i). gather() must return exactly those rows/stamps
        # (on the generation axis) for an arbitrary req_pool_indices order.
        buf = _buffer()
        block = _block([3, 5], [10, 20])
        buf.land(block)

        rows, stamps = buf.gather(torch.tensor([5, 3], dtype=torch.int64))
        self.assertIn(20, stamps[0].tolist())
        self.assertIn(10, stamps[1].tolist())
        self.assertEqual(
            tuple(_seat_stamped(rows[0:1], stamps[0:1], 20).tolist()),
            block.row_tokens(1),
        )
        self.assertEqual(
            tuple(_seat_stamped(rows[1:2], stamps[1:2], 10).tolist()),
            block.row_tokens(0),
        )

    def test_unwritten_seat_gathers_negative_sentinel_stamps(self):
        # The fallback contract for cold seats: a never-written seat's stamps
        # are negative in every generation, so they can never equal a real
        # (>= 0) committed length and the request falls back.
        buf = _buffer()
        _rows, stamps = buf.gather(torch.tensor([1], dtype=torch.int64))
        self.assertTrue(all(s < 0 for s in stamps[0].tolist()))

    def test_reset_slot_invalidates_all_generations_for_reused_seat(self):
        # Seat-reuse lifecycle: when the scheduler reassigns a seat it calls
        # reset_slot, after which NO generation of the previous occupant's
        # landed blocks may look fresh (stamps back to the sentinel).
        buf = _buffer()
        buf.land(_block([3], [10]))
        buf.land(_block([3], [14]))
        buf.reset_slot(3)
        _rows, stamps = buf.gather(torch.tensor([3], dtype=torch.int64))
        self.assertTrue(all(s < 0 for s in stamps[0].tolist()))

    def test_two_newest_blocks_coexist_per_seat(self):
        # Regression (first cross-process e2e, 0-hit): the block serving round
        # r (stamp |P_{r-2}|) must survive the newer push from round r-1's
        # commit (stamp |P_{r-1}|); only the THIRD push may evict it.
        buf = _buffer()
        serving = _block([3], [10])
        buf.land(serving)
        newer = _block([3], [14], tokens=tuple(range(100, 100 + 18)))
        buf.land(newer)
        rows, stamps = buf.gather(torch.tensor([3], dtype=torch.int64))
        self.assertCountEqual(stamps[0].tolist(), [10, 14])
        self.assertEqual(
            tuple(_seat_stamped(rows, stamps, 10).tolist()), serving.row_tokens(0)
        )
        self.assertEqual(
            tuple(_seat_stamped(rows, stamps, 14).tolist()), newer.row_tokens(0)
        )
        # A third push evicts the oldest (10), keeping 14 and 18.
        third = _block([3], [18], tokens=tuple(range(200, 200 + 18)))
        buf.land(third)
        _rows, stamps = buf.gather(torch.tensor([3], dtype=torch.int64))
        self.assertCountEqual(stamps[0].tolist(), [14, 18])

    def test_land_empty_block_is_noop(self):
        buf = _buffer()
        buf.land(_block([], []))  # must not raise

    def test_land_rejects_wrong_verifier(self):
        # Seats are only meaningful within the owning verifier; a misrouted
        # block's pool_indices would otherwise land in unrelated local seats.
        buf = _buffer(verifier_rank=0)
        with self.assertRaises(RuntimeError):
            buf.land(_block([3], [10], dst_verifier_rank=1))

    def test_land_rejects_dims_mismatch(self):
        # A mismatched K/F either shape-errors the scatter or silently mis-lays
        # out the flat [accept_case][guess][step] row if the products coincide.
        buf = _buffer(num_steps=2, fanout=2)
        with self.assertRaises(RuntimeError):
            buf.land(_block([3], [10], num_steps=1, fanout=1))

    def test_land_rejects_out_of_range_pool_idx(self):
        # A peer echoing a seat this verifier never announced must not corrupt
        # an unrelated seat (or crash the scatter with an OOB index).
        buf = _buffer()
        with self.assertRaises(RuntimeError):
            buf.land(_block([_SEATS], [10]))

    def test_land_runs_block_validation(self):
        # Malformed wire input must be caught on the ingest path: parallel
        # arrays out of sync raise ValueError from block.validate() inside
        # land(), not a shape error from deep inside the scatter.
        buf = _buffer()
        malformed = _block([3, 5], [10])  # bases shorter than pool_indices
        with self.assertRaises(ValueError):
            buf.land(malformed)


if __name__ == "__main__":
    unittest.main()
