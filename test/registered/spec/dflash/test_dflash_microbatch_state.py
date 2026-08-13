import unittest

import torch

from sglang.srt.speculative.dflash_info import (
    DFlashDraftInput,
    DFlashMicrobatchInput,
    DFlashVerifyInput,
)


class TestDFlashMicrobatchInput(unittest.TestCase):
    def _draft_input(self) -> DFlashDraftInput:
        return DFlashDraftInput(
            verified_id=torch.tensor([10, 20, 30], dtype=torch.int64),
            target_hidden=torch.tensor(
                [[100.0], [101.0], [200.0], [300.0], [301.0], [302.0]]
            ),
            ctx_lens=torch.tensor([2, 1, 3], dtype=torch.int32),
            draft_seq_lens=torch.tensor([7, 8, 9], dtype=torch.int32),
        )

    def test_split_filter_and_merge_preserve_variable_hidden_segments(self):
        state = DFlashMicrobatchInput.from_draft_input(
            self._draft_input(), draft_token_num=2
        )
        state.filter_batch(torch.tensor([2, 0], dtype=torch.int64))

        merged = state.get_draft_input([0, 1])
        self.assertEqual(merged.verified_id.tolist(), [30, 10])
        self.assertEqual(merged.ctx_lens.tolist(), [3, 2])
        self.assertEqual(merged.draft_seq_lens.tolist(), [9, 7])
        self.assertEqual(
            merged.target_hidden.flatten().tolist(),
            [300.0, 301.0, 302.0, 100.0, 101.0],
        )

    def test_proposals_round_trip_by_request(self):
        state = DFlashMicrobatchInput.from_draft_input(
            self._draft_input(), draft_token_num=2
        )
        proposal = DFlashVerifyInput(
            draft_token=torch.tensor([11, 12, 31, 32], dtype=torch.int64),
            positions=torch.tensor([1, 2, 9, 10], dtype=torch.int64),
            draft_token_num=2,
        )
        state.set_proposal([0, 2], proposal)

        self.assertEqual(state.prepared_indices(), [0, 2])
        self.assertEqual(state.unprepared_indices(), [1])
        rebuilt = state.get_proposal([2, 0])
        self.assertEqual(rebuilt.draft_token.tolist(), [31, 32, 11, 12])
        self.assertEqual(rebuilt.positions.tolist(), [9, 10, 1, 2])

    def test_merge_accepts_new_serial_draft_state(self):
        state = DFlashMicrobatchInput.from_draft_input(
            self._draft_input(), draft_token_num=2
        )
        new_request = DFlashDraftInput(
            verified_id=torch.tensor([40], dtype=torch.int64),
            target_hidden=torch.tensor([[400.0], [401.0]]),
            ctx_lens=torch.tensor([2], dtype=torch.int32),
            draft_seq_lens=torch.tensor([4], dtype=torch.int32),
        )
        state.merge_batch(new_request)

        self.assertEqual(len(state.draft_states), 4)
        self.assertEqual(state.unprepared_indices(), [0, 1, 2, 3])
        self.assertEqual(
            state.get_draft_input([3]).target_hidden.flatten().tolist(),
            [400.0, 401.0],
        )


if __name__ == "__main__":
    unittest.main()
