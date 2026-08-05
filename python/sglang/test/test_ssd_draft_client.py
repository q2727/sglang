from __future__ import annotations

import unittest

from sglang.srt.speculative.ssd_draft_client import (
    DraftCandidate,
    SSDDraftClient,
    SSDDraftClientError,
)


def response(token_ids, top_ids=None):
    meta_info = {
        "output_token_logprobs": [[-0.1, token_id, None] for token_id in token_ids]
    }
    if top_ids is not None:
        meta_info["output_top_logprobs"] = [
            [
                [-float(rank), token_id, None]
                for rank, token_id in enumerate(token_top_ids)
            ]
            for token_top_ids in top_ids
        ]
    return {"meta_info": meta_info}


class FakeSSDDraftClient(SSDDraftClient):
    def __init__(self, replies):
        super().__init__("http://draft.invalid")
        self.replies = list(replies)
        self.calls = []

    def _post_generate(self, input_ids, max_new_tokens, *, top_logprobs_num=0):
        self.calls.append((input_ids, max_new_tokens, top_logprobs_num))
        return self.replies.pop(0)


class FakeDraftSideCacheClient(SSDDraftClient):
    def __init__(self, selected_candidate=None):
        super().__init__("http://draft.invalid")
        self.selected_candidate = selected_candidate
        self.calls = []

    def _post_json(self, path, payload):
        self.calls.append((path, payload))
        if path == "/ssd/build_outcome_cache":
            return {
                "cache_id": payload["cache_id"],
                "branches": sum(payload["fan_outs"]),
                "timing_ms": {
                    "prepare": 0.1,
                    "generate": 12.0,
                    "parse": 0.2,
                    "total": 12.4,
                },
            }
        if path == "/ssd/select_outcome":
            if self.selected_candidate is None:
                return {"hit": False, "lookup_ms": 0.01}
            return {
                "hit": True,
                "candidate": self.candidate_to_payload(self.selected_candidate),
                "lookup_ms": 0.01,
            }
        raise AssertionError(f"unexpected path: {path}")


class TestSSDDraftClient(unittest.TestCase):
    def test_draft_side_cache_uses_compact_build_and_select_protocol(self):
        selected = DraftCandidate(
            tokens=[101, 102],
            recovery_tokens=((111, 112), (), (103,)),
        )
        client = FakeDraftSideCacheClient(selected)
        source = DraftCandidate(
            tokens=[20, 21],
            recovery_tokens=((30, 31), (), (40,)),
        )

        handle = client.prepare_outcome_cache(
            canonical_prefix=[10, 11],
            candidate=source,
            draft_length=2,
            fan_out=(2, 0, 1),
        )
        candidate = client.select_outcome(
            handle,
            outcome_key=(0, 30),
            draft_length=2,
            fan_out=(2, 0, 1),
        )

        self.assertEqual(candidate, selected)
        self.assertEqual(handle.branches, 3)
        self.assertEqual(client.stats.generate_calls, 1)
        self.assertEqual(client.stats.generated_sequences, 3)
        build_path, build_payload = client.calls[0]
        self.assertEqual(build_path, "/ssd/build_outcome_cache")
        self.assertEqual(build_payload["canonical_prefix"], [10, 11])
        self.assertEqual(build_payload["draft_tokens"], [20, 21])
        self.assertEqual(build_payload["recovery_tokens"], [[30, 31], [], [40]])
        self.assertNotIn("branch_prefixes", build_payload)
        self.assertEqual(
            client.calls[1],
            (
                "/ssd/select_outcome",
                {
                    "cache_id": handle.cache_id,
                    "accepted_length": 0,
                    "recovery_token": 30,
                },
            ),
        )

    def test_draft_side_cache_miss_returns_none(self):
        client = FakeDraftSideCacheClient()
        source = DraftCandidate(tokens=[20], recovery_tokens=((30,), (40,)))
        handle = client.prepare_outcome_cache([10], source, 1, 1)
        self.assertIsNone(client.select_outcome(handle, (0, 99), 1, 1))

    def test_build_outcome_cache_uses_saguaro_keys_and_excludes_path(self):
        client = FakeSSDDraftClient(
            [
                [
                    response([101, 102, 103], [[101, 111], [102, 112], [103, 113]]),
                    response([201, 202, 203], [[201, 211], [202, 212], [203, 213]]),
                    response([301, 302, 303], [[301, 311], [302, 312], [303, 313]]),
                ],
            ]
        )

        cache = client.build_outcome_cache(
            canonical_prefix=[10, 11],
            candidate=DraftCandidate(
                tokens=[20, 21],
                recovery_tokens=((30,), (31,), (40,)),
            ),
            draft_length=2,
            fan_out=1,
        )

        self.assertEqual(
            cache,
            {
                (0, 30): DraftCandidate(
                    tokens=[101, 102],
                    recovery_tokens=((111,), (112,), (103,)),
                ),
                (1, 31): DraftCandidate(
                    tokens=[201, 202],
                    recovery_tokens=((211,), (212,), (203,)),
                ),
                (2, 40): DraftCandidate(
                    tokens=[301, 302],
                    recovery_tokens=((311,), (312,), (303,)),
                ),
            },
        )
        self.assertEqual(
            client.calls,
            [
                (
                    [[10, 11, 30], [10, 11, 20, 31], [10, 11, 20, 21, 40]],
                    3,
                    2,
                ),
            ],
        )

    def test_jit_draft_carries_all_endpoint_recoveries(self):
        client = FakeSSDDraftClient(
            [
                [
                    response(
                        [7, 8, 9, 10],
                        [[7, 70], [8, 80], [9, 90], [10, 100]],
                    )
                ]
            ]
        )
        self.assertEqual(
            client.jit_draft([1, 2], 3, 1),
            DraftCandidate(
                tokens=[7, 8, 9],
                recovery_tokens=((70,), (80,), (90,), (10,)),
            ),
        )
        self.assertEqual(client.calls, [([1, 2], 4, 2)])

    def test_fan_out_requires_enough_alternatives(self):
        with self.assertRaises(SSDDraftClientError):
            SSDDraftClient._select_recovery_tokens(
                [20, 20], exclude=20, fan_out=1
            )

    def test_nonuniform_fan_out_allows_zero_endpoints(self):
        client = FakeSSDDraftClient(
            [
                [
                    response([101, 102, 103], [[101, 111, 112], [102], [103, 113, 114]]),
                    response([201, 202, 203], [[201, 211, 212], [202], [203, 213, 214]]),
                    response([301, 302, 303], [[301, 311, 312], [302], [303, 313, 314]]),
                ],
            ]
        )

        cache = client.build_outcome_cache(
            canonical_prefix=[10, 11],
            candidate=DraftCandidate(
                tokens=[20, 21],
                recovery_tokens=((30, 31), (), (40,)),
            ),
            draft_length=2,
            fan_out=(2, 0, 1),
        )

        self.assertEqual(set(cache), {(0, 30), (0, 31), (2, 40)})
        self.assertEqual(
            client.calls,
            [
                (
                    [[10, 11, 30], [10, 11, 31], [10, 11, 20, 21, 40]],
                    3,
                    3,
                ),
            ],
        )

    def test_jit_draft_uses_max_nonuniform_fan_out(self):
        client = FakeSSDDraftClient(
            [
                [
                    response(
                        [7, 8, 9],
                        [[7, 70, 71], [8], [9, 90, 91]],
                    )
                ]
            ]
        )

        self.assertEqual(
            client.jit_draft([1, 2], 2, (2, 0, 1)),
            DraftCandidate(
                tokens=[7, 8],
                recovery_tokens=((70, 71), (), (9,)),
            ),
        )
        self.assertEqual(client.calls, [([1, 2], 3, 3)])


if __name__ == "__main__":
    unittest.main()
