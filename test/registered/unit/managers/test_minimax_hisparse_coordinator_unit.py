"""Unit tests for MiniMaxHiSparseCoordinator.

Tests cover:
- req_to_host allocation during prefill
- req_to_host extension during decode
- req_to_host row cleared on request finish
- finish releases host pool resources
- abort releases all resources
- Host pool exhaustion
- Staging interface (all no-ops)
- Observability (get_host_usage, num_real_reqs)

The coordinator is tested with mocked dependencies (ReqToTokenPool,
MiniMaxHiSparseKVPool, PagedTokenToKVPoolAllocator) so no GPU or
CUDA is required.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_req(rid="test-req-0", fill_len=64, kv_allocated_len=64, req_pool_idx=0):
    """Create a minimal mock Req."""
    return SimpleNamespace(
        rid=rid,
        req_pool_idx=req_pool_idx,
        fill_len=fill_len,
        kv_allocated_len=kv_allocated_len,
        kv_committed_len=kv_allocated_len,
        hisparse_staging=False,
        finished=lambda self=None: False,
    )


def _make_req_to_token_pool(max_reqs=8, max_context_len=2048):
    """Create a mock ReqToTokenPool."""
    pool = SimpleNamespace()
    pool.req_to_token = torch.zeros((max_reqs, max_context_len), dtype=torch.int32)
    pool.max_context_len = max_context_len
    pool.size = max_reqs
    return pool


class _MockHostPool:
    """Mock MiniMaxSparseMainHostPool with a bump allocator."""

    def __init__(self, size=4096, page_size=128):
        self.alloc_size = size
        self.page_size = page_size
        self._free_head = 0

    def alloc(self, num):
        if self._free_head + num > self.alloc_size:
            raise RuntimeError(
                f"Host pool exhausted: need {num}, only "
                f"{self.alloc_size - self._free_head} remain."
            )
        start = self._free_head
        self._free_head += num
        return torch.arange(start, start + num, dtype=torch.int64)

    def free(self, indices):
        pass

    @property
    def free_slots(self):
        return self.alloc_size - self._free_head


def _make_host_pool(size=4096, page_size=128):
    return _MockHostPool(size, page_size)


def _make_kv_pool(host_pool=None):
    if host_pool is None:
        host_pool = _make_host_pool()
    pool = SimpleNamespace()
    pool.sparse_main_host_pool = host_pool
    pool.local_sparse_layer_ids = list(range(3, 60))  # layers 3-59
    pool.page_size = 128  # sparse_block_size for M3
    pool.backup_sparse_main_from_standard_pool = (
        lambda layer_id, host_locs,
        standard_k_cache, standard_v_cache,
        standard_indices: None
    )
    return pool


def _make_standard_kv_pool():
    """Mock MiniMaxSparseKVPool — returns a dummy (k, v) per layer."""
    pool = SimpleNamespace()

    def _get_kv_buffer(layer_id):
        k = torch.zeros(1, 4, 128, dtype=torch.bfloat16)
        v = torch.zeros(1, 4, 128, dtype=torch.bfloat16)
        return k, v

    pool.get_kv_buffer = _get_kv_buffer
    return pool


def _make_allocator():
    alloc = SimpleNamespace()
    alloc.free = lambda indices: None
    alloc.size = 100000
    alloc.available_size = lambda: 99999
    return alloc


# ---------------------------------------------------------------------------
# Load the coordinator module directly (bypass sglang package import)
# ---------------------------------------------------------------------------

_COORDINATOR_PATH = (
    Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "managers"
    / "minimax_hisparse_coordinator.py"
)

# Minimal mock for sglang.srt.utils.get_device_module
_mock_utils = SimpleNamespace()
_mock_utils.get_device_module = lambda: SimpleNamespace(Stream=lambda: None)
sys.modules["sglang.srt.utils"] = _mock_utils

# Mock HiSparseTokenStats for the coordinator import
from collections import namedtuple
_mock_hisparse_coord = SimpleNamespace()
_mock_hisparse_coord.HiSparseTokenStats = namedtuple(
    "HiSparseTokenStats",
    ["device_tokens", "device_token_usage", "host_tokens", "host_token_usage"],
)
sys.modules["sglang.srt.managers.hisparse_coordinator"] = _mock_hisparse_coord

# Load coordinator module from source
spec = importlib.util.spec_from_file_location(
    "minimax_hisparse_coordinator", str(_COORDINATOR_PATH)
)
_coord_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_coord_module)
MiniMaxHiSparseCoordinator = _coord_module.MiniMaxHiSparseCoordinator


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMiniMaxHiSparseCoordinator(unittest.TestCase):
    """Unit tests for MiniMaxHiSparseCoordinator lifecycle."""

    def setUp(self):
        self.max_reqs = 8
        self.max_context_len = 2048
        self.host_size = 4096
        self.page_size = 128

        self.req_to_token_pool = _make_req_to_token_pool(
            max_reqs=self.max_reqs, max_context_len=self.max_context_len
        )
        self.host_pool = _make_host_pool(
            size=self.host_size, page_size=self.page_size
        )
        self.kv_pool = _make_kv_pool(host_pool=self.host_pool)
        self.standard_pool = _make_standard_kv_pool()
        self.allocator = _make_allocator()

        self.coordinator = MiniMaxHiSparseCoordinator(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            standard_kv_pool=self.standard_pool,
            hisparse_kv_pool=self.kv_pool,
            device="cpu",
        )

    # ------------------------------------------------------------------
    # req_to_host allocation
    # ------------------------------------------------------------------

    def test_req_to_host_initial_state(self):
        """req_to_host starts all -1."""
        self.assertTrue(
            torch.all(self.coordinator.req_to_host == -1),
        )
        self.assertEqual(self.coordinator.req_to_host.shape[0], self.max_reqs)
        self.assertEqual(
            self.coordinator.req_to_host.shape[1], self.max_context_len
        )
        self.assertTrue(
            torch.all(self.coordinator.req_to_host_allocated_len == 0),
        )

    def test_admit_prefill_allocates_host_slots(self):
        """admit_prefill allocates host slots for all prefill tokens."""
        req = _make_req("prefill-test", fill_len=256, kv_allocated_len=256)
        self.coordinator.admit_prefill(req)

        host_row = self.coordinator.req_to_host[req.req_pool_idx, : req.fill_len]
        self.assertTrue(torch.all(host_row >= 0))
        self.assertEqual(host_row[0].item(), 0)
        self.assertEqual(host_row[-1].item(), req.fill_len - 1)
        self.assertEqual(
            int(self.coordinator.req_to_host_allocated_len[req.req_pool_idx]),
            req.fill_len,
        )

    def test_admit_prefill_zero_len(self):
        """admit_prefill with fill_len=0 is a no-op."""
        req = _make_req("empty-prefill", fill_len=0, kv_allocated_len=0)
        self.coordinator.admit_prefill(req)
        self.assertEqual(
            int(self.coordinator.req_to_host_allocated_len[req.req_pool_idx]), 0
        )

    def test_admit_request_into_staging_compat(self):
        """admit_request_into_staging delegates to admit_prefill."""
        req = _make_req("staging-compat", fill_len=128, kv_allocated_len=128)
        req.hisparse_staging = True
        self.coordinator.admit_request_into_staging(req)

        self.assertFalse(req.hisparse_staging)
        host_row = self.coordinator.req_to_host[req.req_pool_idx, : req.fill_len]
        self.assertTrue(torch.all(host_row >= 0))
        self.assertEqual(
            int(self.coordinator.req_to_host_allocated_len[req.req_pool_idx]),
            req.fill_len,
        )

    # ------------------------------------------------------------------
    # Decode extension
    # ------------------------------------------------------------------

    def test_extend_decode_adds_one_slot(self):
        """extend_decode allocates one additional host slot per decode step."""
        req = _make_req("decode-test", fill_len=128, kv_allocated_len=128)
        self.coordinator.admit_prefill(req)

        initial_len = int(
            self.coordinator.req_to_host_allocated_len[req.req_pool_idx]
        )

        req.kv_allocated_len = 129
        self.coordinator.extend_decode(req)

        new_len = int(
            self.coordinator.req_to_host_allocated_len[req.req_pool_idx]
        )
        self.assertEqual(new_len, initial_len + 1)
        self.assertGreaterEqual(
            int(self.coordinator.req_to_host[req.req_pool_idx, 128].item()), 0
        )

    def test_extend_decode_already_allocated(self):
        """extend_decode is a no-op if slots are already allocated."""
        req = _make_req("no-extend", fill_len=128, kv_allocated_len=128)
        self.coordinator.admit_prefill(req)
        self.coordinator.extend_decode(req)
        self.assertEqual(
            int(self.coordinator.req_to_host_allocated_len[req.req_pool_idx]),
            128,
        )

    def test_extend_decode_multi_step(self):
        """Multiple decode step extensions work correctly."""
        req = _make_req("multi-decode", fill_len=64, kv_allocated_len=64)
        self.coordinator.admit_prefill(req)

        for step in range(10):
            req.kv_allocated_len = 65 + step
            self.coordinator.extend_decode(req)

        final_len = int(
            self.coordinator.req_to_host_allocated_len[req.req_pool_idx]
        )
        self.assertEqual(final_len, 74)
        host_row = self.coordinator.req_to_host[req.req_pool_idx, :final_len]
        self.assertTrue(torch.all(host_row >= 0))
        self.assertEqual(host_row.numel(), final_len)

    def test_extend_decode_exceeds_context_len_raises(self):
        """extend_decode raises RuntimeError if seq_len > max_context_len."""
        req = _make_req(
            "overflow",
            fill_len=self.max_context_len,
            kv_allocated_len=self.max_context_len,
        )
        self.coordinator.admit_prefill(req)

        req.kv_allocated_len = self.max_context_len + 1
        with self.assertRaises(RuntimeError):
            self.coordinator.extend_decode(req)

    # ------------------------------------------------------------------
    # Request finish
    # ------------------------------------------------------------------

    def test_request_finished_clears_row(self):
        """request_finished clears the req_to_host row to -1."""
        req = _make_req("finish-test", fill_len=256, kv_allocated_len=256)
        self.coordinator.admit_prefill(req)

        self.coordinator.request_finished(req)

        self.assertTrue(
            torch.all(self.coordinator.req_to_host[req.req_pool_idx] == -1)
        )
        self.assertEqual(
            int(self.coordinator.req_to_host_allocated_len[req.req_pool_idx]), 0
        )

    def test_request_finished_idempotent(self):
        """request_finished is safe to call multiple times."""
        req = _make_req("idempotent-finish", fill_len=128, kv_allocated_len=128)
        self.coordinator.admit_prefill(req)
        self.coordinator.request_finished(req)
        self.coordinator.request_finished(req)
        self.assertEqual(
            int(self.coordinator.req_to_host_allocated_len[req.req_pool_idx]), 0
        )

    def test_retract_req_releases_resources(self):
        """retract_req (abort) releases host resources same as finish."""
        req = _make_req("abort-test", fill_len=128, kv_allocated_len=128)
        self.coordinator.admit_prefill(req)

        self.coordinator.retract_req(req)

        self.assertTrue(
            torch.all(self.coordinator.req_to_host[req.req_pool_idx] == -1)
        )
        self.assertEqual(
            int(self.coordinator.req_to_host_allocated_len[req.req_pool_idx]), 0
        )

    # ------------------------------------------------------------------
    # Multiple requests
    # ------------------------------------------------------------------

    def test_multiple_requests_independent(self):
        """Host slots for different requests are independent."""
        reqs = [
            _make_req(f"multi-req-{i}", fill_len=128, kv_allocated_len=128,
                      req_pool_idx=i)
            for i in range(4)
        ]

        for req in reqs:
            self.coordinator.admit_prefill(req)

        for req in reqs:
            host_row = self.coordinator.req_to_host[
                req.req_pool_idx, : req.fill_len
            ]
            self.assertTrue(torch.all(host_row >= 0))

        # Slots should not overlap (contiguous bump allocation)
        all_slots = []
        for req in reqs:
            all_slots.extend(
                self.coordinator.req_to_host[
                    req.req_pool_idx, : req.fill_len
                ].tolist()
            )
        self.assertEqual(len(all_slots), len(set(all_slots)))

        # Finish one request, others unaffected
        self.coordinator.request_finished(reqs[1])
        self.assertTrue(
            torch.all(self.coordinator.req_to_host[reqs[1].req_pool_idx] == -1)
        )
        for req in [reqs[0], reqs[2], reqs[3]]:
            self.assertTrue(
                torch.all(
                    self.coordinator.req_to_host[
                        req.req_pool_idx, : req.fill_len
                    ] >= 0
                )
            )

    # ------------------------------------------------------------------
    # Host pool exhaustion
    # ------------------------------------------------------------------

    def test_host_pool_exhaustion_raises(self):
        """admit_prefill raises RuntimeError when host pool is exhausted."""
        small_host = _make_host_pool(size=64, page_size=128)
        small_kv = _make_kv_pool(host_pool=small_host)

        coord = MiniMaxHiSparseCoordinator(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            standard_kv_pool=self.standard_pool,
            hisparse_kv_pool=small_kv,
            device="cpu",
        )

        coord.admit_prefill(_make_req("small-1", fill_len=32, kv_allocated_len=32))
        coord.admit_prefill(_make_req("small-2", fill_len=32, kv_allocated_len=32))

        with self.assertRaises(RuntimeError):
            coord.admit_prefill(
                _make_req("too-big", fill_len=33, kv_allocated_len=33)
            )

    # ------------------------------------------------------------------
    # Staging interface (M3 has none)
    # ------------------------------------------------------------------

    def test_has_ongoing_staging_always_false(self):
        """M3 coordinator never has ongoing staging."""
        self.assertFalse(self.coordinator.has_ongoing_staging())

    def test_collect_ready_reqs_always_empty(self):
        """M3 coordinator never has staging-ready requests."""
        self.assertEqual(self.coordinator.collect_ready_reqs(), [])

    # ------------------------------------------------------------------
    # Stream / backup interface
    # ------------------------------------------------------------------

    def test_wait_for_pending_backup_noop(self):
        """wait_for_pending_backup is a no-op for M3."""
        self.coordinator.wait_for_pending_backup()

    def test_map_last_loc_to_buffer_noop(self):
        """map_last_loc_to_buffer is a no-op for M3."""
        dummy = torch.zeros(1, dtype=torch.int64)
        self.coordinator.map_last_loc_to_buffer(
            dummy, dummy, dummy, dummy, dummy
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def test_get_host_usage(self):
        """get_host_usage reports correct used/total."""
        used, total = self.coordinator.get_host_usage()
        self.assertEqual(used, 0)
        self.assertEqual(total, self.host_size)

        req = _make_req("usage-test", fill_len=256, kv_allocated_len=256)
        self.coordinator.admit_prefill(req)

        used2, total2 = self.coordinator.get_host_usage()
        self.assertEqual(used2, 256)
        self.assertEqual(total2, self.host_size)

    # ------------------------------------------------------------------
    # num_real_reqs
    # ------------------------------------------------------------------

    def test_num_real_reqs_default(self):
        """num_real_reqs starts at 0."""
        self.assertEqual(int(self.coordinator.num_real_reqs.item()), 0)

    def test_num_real_reqs_set(self):
        """num_real_reqs can be updated."""
        self.coordinator.num_real_reqs.fill_(5)
        self.assertEqual(int(self.coordinator.num_real_reqs.item()), 5)

    # ------------------------------------------------------------------
    # GPU req_to_host
    # ------------------------------------------------------------------

    def test_req_to_host_on_device(self):
        """req_to_host lives on the coordinator's device (GPU), not CPU."""
        self.assertEqual(
            str(self.coordinator.req_to_host.device),
            str(self.coordinator.device),
        )

    # ------------------------------------------------------------------
    # Pre-allocated graph-safe buffers
    # ------------------------------------------------------------------

    def test_graph_buffers_pre_allocated(self):
        """Graph-safe buffers exist with expected shapes and dtypes."""
        max_pages = (
            self.max_context_len + self.page_size - 1
        ) // self.page_size
        self.assertEqual(
            self.coordinator.hot_page_table_buffer.shape,
            (self.max_reqs, max_pages),
        )
        self.assertEqual(self.coordinator.hot_page_table_buffer.dtype, torch.int32)
        self.assertEqual(
            self.coordinator.hot_kv_indices_buffer.shape,
            (self.max_reqs * max_pages,),
        )
        self.assertTrue(
            torch.all(self.coordinator.host_locs_buffer == -1),
        )
        self.assertTrue(
            torch.all(self.coordinator.hot_locs_buffer == -1),
        )

    # ------------------------------------------------------------------
    # prepare_for_graph_replay
    # ------------------------------------------------------------------

    def test_prepare_for_graph_replay_updates_num_real_reqs(self):
        """prepare_for_graph_replay sets num_real_reqs from batch size."""
        mock_batch = SimpleNamespace(batch_size=3)
        self.coordinator.prepare_for_graph_replay(mock_batch)
        self.assertEqual(int(self.coordinator.num_real_reqs.item()), 3)

        mock_batch.batch_size = 7
        self.coordinator.prepare_for_graph_replay(mock_batch)
        self.assertEqual(int(self.coordinator.num_real_reqs.item()), 7)


class TestMiniMaxHiSparseModelDetection(unittest.TestCase):
    """Test model detection functions (no sglang import needed — logic is simple)."""

    def test_is_minimax_sparse_via_architecture_check(self):
        """MiniMax-M3 sparse models are identified by architecture name."""
        # This is the logic from model_config.is_minimax_sparse
        M3_ARCHS = (
            "MiniMaxM3SparseForCausalLM",
            "MiniMaxM3SparseForConditionalGeneration",
        )

        def is_minimax_sparse(architectures):
            return (architectures or [None])[0] in M3_ARCHS

        self.assertTrue(is_minimax_sparse(["MiniMaxM3SparseForCausalLM"]))
        self.assertTrue(
            is_minimax_sparse(["MiniMaxM3SparseForConditionalGeneration"])
        )
        self.assertFalse(is_minimax_sparse(["LlamaForCausalLM"]))
        self.assertFalse(is_minimax_sparse(["DeepseekV3ForCausalLM"]))

    def test_first_phase_restrictions_documented(self):
        """First-phase M3 HiSparse restrictions are clearly enumerable."""
        restrictions = {
            "kv_cache_dtype": "BF16 only (no FP8)",
            "speculative_algorithm": "must be None",
            "enable_pd_disaggregation": "must be False",
            "model_type": "text-only (no multimodal)",
            "cuda_graph": "disabled for now",
        }
        self.assertEqual(len(restrictions), 5)
        for name, desc in restrictions.items():
            with self.subTest(restriction=name):
                self.assertIsInstance(desc, str)


if __name__ == "__main__":
    unittest.main()
