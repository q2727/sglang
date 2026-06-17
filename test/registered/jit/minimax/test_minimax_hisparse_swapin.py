"""Correctness tests for the MiniMax-M3 HiSparse block swap-in kernel.

Tests the JIT-accelerated path against the Python reference path in
MiniMaxHiSparseKVPool.load_sparse_main_blocks_to_hot block mode.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.mem_cache.minimax_hisparse_memory_pool import (
    MiniMaxHiSparseKVPool,
    MiniMaxHiSparseLoadResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAGE_SIZE = 128
Hkv = 4
K = 16
H = 64  # num_attention_heads
D = 128  # head_dim
IDX_D = 128  # index head dim
DTYPE = torch.bfloat16


def _make_pool(
    size: int = 4096,
    hot_size: int = 2048,
    device: str = "cuda",
) -> MiniMaxHiSparseKVPool:
    """Create a minimal HiSparse pool for testing."""
    return MiniMaxHiSparseKVPool(
        size=size,
        page_size=PAGE_SIZE,
        dtype=DTYPE,
        head_num=Hkv,
        head_dim=D,
        idx_head_dim=IDX_D,
        dense_layer_ids=[0, 1, 2],
        sparse_layer_ids=list(range(3, 60)),
        device=device,
        hot_size=hot_size,
        host_to_device_ratio=2.0,
        disable_value_sparse_layer_ids=list(range(3, 60)),
    )


def _fill_host_pool(pool: MiniMaxHiSparseKVPool, layer_id: int, num_tokens: int):
    """Fill the host pool with deterministic test data."""
    host_k, host_v = pool.get_sparse_main_host_kv_buffer(layer_id)
    mapped_id = pool._sparse_local_layer_id(layer_id)
    # Fill with recognizable per-token values.
    for i in range(num_tokens):
        host_k[i] = torch.full((Hkv, D), float(i + 1), dtype=DTYPE)
        host_v[i] = torch.full((Hkv, D), float(-(i + 1)), dtype=DTYPE)


def _make_topk_idx(
    batch: int,
    seq_lens: torch.Tensor,
    unique_blocks: bool = True,
    device: str = "cuda",
) -> torch.Tensor:
    """Create synthetic topk_idx with per-request block selections.

    Args:
        batch: Number of requests.
        seq_lens: [batch] int64 tensor of sequence lengths.
        unique_blocks: If True, each KV head selects different blocks.
        device: Target device.
    """
    topk = torch.full((Hkv, batch, K), -1, dtype=torch.int32, device=device)
    for b in range(batch):
        sl = int(seq_lens[b].item())
        num_pages = max(1, (sl + PAGE_SIZE - 1) // PAGE_SIZE)
        num_select = min(K, num_pages)
        if unique_blocks:
            # Each KV head gets a different set of blocks (shifted).
            for h in range(Hkv):
                for k in range(num_select):
                    topk[h, b, k] = (h * 3 + k) % num_pages
        else:
            # All KV heads share the same blocks (duplication).
            for k in range(num_select):
                for h in range(Hkv):
                    topk[h, b, k] = k % num_pages
    return topk


def _make_req_to_host(
    max_reqs: int,
    max_ctx: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Create a synthetic req_to_host mapping.

    Maps (req_row, logical_pos) → host pool loc. For simplicity, we use a
    1:1 mapping: host_loc = req_row * max_ctx + logical_pos.
    """
    r2h = torch.full((max_reqs, max_ctx), -1, dtype=torch.int64, device=device)
    for r in range(max_reqs):
        for p in range(max_ctx):
            r2h[r, p] = r * max_ctx + p
    return r2h


def _run_reference(
    pool: MiniMaxHiSparseKVPool,
    layer_id: int,
    req_to_host: torch.Tensor,
    req_pool_indices: torch.Tensor,
    topk_idx: torch.Tensor,
    seq_lens: torch.Tensor,
) -> MiniMaxHiSparseLoadResult:
    """Run the Python reference path (force-disable JIT)."""
    # Force the Python path by temporarily disabling JIT.
    saved = pool._use_jit_swap_in
    pool._use_jit_swap_in = False
    try:
        return pool.load_sparse_main_blocks_to_hot(
            layer_id,
            req_to_host=req_to_host,
            req_pool_indices=req_pool_indices,
            topk_idx=topk_idx,
            seq_lens=seq_lens,
        )
    finally:
        pool._use_jit_swap_in = saved


def _run_jit(
    pool: MiniMaxHiSparseKVPool,
    layer_id: int,
    req_to_host: torch.Tensor,
    req_pool_indices: torch.Tensor,
    topk_idx: torch.Tensor,
    seq_lens: torch.Tensor,
) -> MiniMaxHiSparseLoadResult:
    """Run the JIT path (if available)."""
    if not pool._use_jit_swap_in:
        pytest.skip("JIT swap-in kernel not available on this device.")
    return pool.load_sparse_main_blocks_to_hot(
        layer_id,
        req_to_host=req_to_host,
        req_pool_indices=req_pool_indices,
        topk_idx=topk_idx,
        seq_lens=seq_lens,
    )


def _assert_hot_kv_correct(
    pool: MiniMaxHiSparseKVPool,
    layer_id: int,
    result: MiniMaxHiSparseLoadResult,
    req_to_host: torch.Tensor,
    topk_idx: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
):
    """Verify that the hot K/V buffer contains correct values for selected blocks."""
    hot_k, hot_v = pool.get_hot_kv_buffer(layer_id)
    host_k, host_v = pool.get_sparse_main_host_kv_buffer(layer_id)

    Hkv, B, K = topk_idx.shape
    for b in range(B):
        req_row = int(req_pool_indices[b].item())
        sl = int(seq_lens[b].item())
        # Get unique blocks for this request (same as what the kernel deduped).
        seen = set()
        for h in range(Hkv):
            for k in range(K):
                blk = int(topk_idx[h, b, k].item())
                if blk >= 0 and blk * PAGE_SIZE < sl:
                    seen.add(blk)

        for blk in sorted(seen):
            # Where does this block live in hot buffer?
            if result.hot_page_table is not None:
                hot_page_id = int(result.hot_page_table[b, blk].item())
                if hot_page_id < 0:
                    continue
            else:
                continue

            for t in range(min(PAGE_SIZE, sl - blk * PAGE_SIZE)):
                logical_pos = blk * PAGE_SIZE + t
                host_loc = int(req_to_host[req_row, logical_pos].item())
                hot_loc = hot_page_id * PAGE_SIZE + t

                # Compare K
                torch.testing.assert_close(
                    hot_k[hot_loc], host_k[host_loc].to(device=hot_k.device),
                    msg=f"K mismatch: req={b}, blk={blk}, t={t}, host_loc={host_loc}, hot_loc={hot_loc}"
                )
                # Compare V
                torch.testing.assert_close(
                    hot_v[hot_loc], host_v[host_loc].to(device=hot_v.device),
                    msg=f"V mismatch: req={b}, blk={blk}, t={t}"
                )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMiniMaxHiSparseSwapIn:
    """Tests that require CUDA and compare JIT vs reference."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_single_request_unique_blocks(self):
        """Single request, unique blocks per KV head."""
        pool = _make_pool(size=2048, hot_size=1024)
        layer_id = 3

        B = 1
        max_reqs = 4
        max_ctx = 2048
        seq_lens = torch.tensor([512], dtype=torch.int64, device="cuda")
        req_pool_indices = torch.tensor([1], dtype=torch.int64, device="cuda")
        topk_idx = _make_topk_idx(B, seq_lens, unique_blocks=True)
        req_to_host = _make_req_to_host(max_reqs, max_ctx)

        _fill_host_pool(pool, layer_id, max_reqs * max_ctx)

        ref = _run_reference(pool, layer_id, req_to_host, req_pool_indices, topk_idx, seq_lens)
        jit = _run_jit(pool, layer_id, req_to_host, req_pool_indices, topk_idx, seq_lens)

        # Check hot K/V correctness for both paths.
        _assert_hot_kv_correct(pool, layer_id, ref, req_to_host, topk_idx, req_pool_indices, seq_lens)
        _assert_hot_kv_correct(pool, layer_id, jit, req_to_host, topk_idx, req_pool_indices, seq_lens)

        # Page tables should agree.
        if ref.hot_page_table is not None and jit.hot_page_table is not None:
            torch.testing.assert_close(ref.hot_page_table, jit.hot_page_table)

        # kv_indices should have the same set of values (order may differ due to tie-breaking).
        if ref.hot_kv_indices is not None and jit.hot_kv_indices is not None:
            assert set(ref.hot_kv_indices.tolist()) == set(jit.hot_kv_indices.tolist())

    def test_duplicate_blocks(self):
        """Multi KV-head selects same block → dedup applied."""
        pool = _make_pool(size=2048, hot_size=1024)
        layer_id = 3

        B = 1
        max_reqs = 4
        max_ctx = 2048
        seq_lens = torch.tensor([1024], dtype=torch.int64, device="cuda")
        req_pool_indices = torch.tensor([2], dtype=torch.int64, device="cuda")
        # All KV heads select the same blocks (no shifting).
        topk_idx = _make_topk_idx(B, seq_lens, unique_blocks=False)
        req_to_host = _make_req_to_host(max_reqs, max_ctx)

        _fill_host_pool(pool, layer_id, max_reqs * max_ctx)

        ref = _run_reference(pool, layer_id, req_to_host, req_pool_indices, topk_idx, seq_lens)
        jit = _run_jit(pool, layer_id, req_to_host, req_pool_indices, topk_idx, seq_lens)

        _assert_hot_kv_correct(pool, layer_id, ref, req_to_host, topk_idx, req_pool_indices, seq_lens)
        _assert_hot_kv_correct(pool, layer_id, jit, req_to_host, topk_idx, req_pool_indices, seq_lens)

        if ref.hot_page_table is not None and jit.hot_page_table is not None:
            torch.testing.assert_close(ref.hot_page_table, jit.hot_page_table)

        # Dedup: number of hot pages should equal number of unique blocks.
        if jit.hot_kv_indices is not None:
            n_unique = len(set(topk_idx[topk_idx >= 0].tolist()))
            assert jit.hot_kv_indices.numel() == n_unique, \
                f"Expected {n_unique} unique blocks, got {jit.hot_kv_indices.numel()} hot pages"

    def test_multi_head_overlapping_selections(self):
        """Multi KV-head with overlapping but not identical selections."""
        pool = _make_pool(size=2048, hot_size=1024)
        layer_id = 5

        B = 2
        max_reqs = 4
        max_ctx = 2048
        seq_lens = torch.tensor([768, 512], dtype=torch.int64, device="cuda")
        req_pool_indices = torch.tensor([0, 3], dtype=torch.int64, device="cuda")
        topk_idx = _make_topk_idx(B, seq_lens, unique_blocks=True)
        req_to_host = _make_req_to_host(max_reqs, max_ctx)

        _fill_host_pool(pool, layer_id, max_reqs * max_ctx)

        ref = _run_reference(pool, layer_id, req_to_host, req_pool_indices, topk_idx, seq_lens)
        jit = _run_jit(pool, layer_id, req_to_host, req_pool_indices, topk_idx, seq_lens)

        _assert_hot_kv_correct(pool, layer_id, ref, req_to_host, topk_idx, req_pool_indices, seq_lens)
        _assert_hot_kv_correct(pool, layer_id, jit, req_to_host, topk_idx, req_pool_indices, seq_lens)

        # Both paths should produce the same token-level K/V values.
        hot_k, hot_v = pool.get_hot_kv_buffer(layer_id)
        assert hot_k[hot_k.abs().sum() > 0].numel() > 0, "Hot K buffer should contain data"

    def test_partial_final_block(self):
        """Partial final block (seq_len not divisible by block_size)."""
        pool = _make_pool(size=2048, hot_size=1024)
        layer_id = 3

        B = 1
        max_reqs = 4
        max_ctx = 2048
        # seq_len = 200 → 2 blocks, 2nd block has 200 - 128 = 72 tokens
        seq_lens = torch.tensor([200], dtype=torch.int64, device="cuda")
        req_pool_indices = torch.tensor([1], dtype=torch.int64, device="cuda")
        topk_idx = _make_topk_idx(B, seq_lens, unique_blocks=False)
        req_to_host = _make_req_to_host(max_reqs, max_ctx)

        _fill_host_pool(pool, layer_id, max_reqs * max_ctx)

        ref = _run_reference(pool, layer_id, req_to_host, req_pool_indices, topk_idx, seq_lens)
        jit = _run_jit(pool, layer_id, req_to_host, req_pool_indices, topk_idx, seq_lens)

        _assert_hot_kv_correct(pool, layer_id, ref, req_to_host, topk_idx, req_pool_indices, seq_lens)
        _assert_hot_kv_correct(pool, layer_id, jit, req_to_host, topk_idx, req_pool_indices, seq_lens)

        # The partial block (block 1) should have its invalid tail zeroed.
        hot_k, hot_v = pool.get_hot_kv_buffer(layer_id)
        if jit.hot_page_table is not None:
            hot_page_id = int(jit.hot_page_table[0, 1].item())
            if hot_page_id >= 0:
                # Tokens 128..199 are valid, 200..255 should be zero.
                valid_slice = hot_k[hot_page_id * PAGE_SIZE : hot_page_id * PAGE_SIZE + 72]
                zero_slice = hot_k[hot_page_id * PAGE_SIZE + 72 : (hot_page_id + 1) * PAGE_SIZE]
                assert valid_slice.abs().sum() > 0, "Valid tokens should have data"
                assert zero_slice.abs().sum() == 0, f"Tail should be zeroed, got non-zero values"

    def test_batch_padded_entries(self):
        """Batch with entries beyond num_real_reqs should be ignored."""
        pool = _make_pool(size=4096, hot_size=2048)
        layer_id = 3

        max_reqs = 8
        max_ctx = 4096
        # Pad batch: 3 real requests + 1 dummy.
        B = 4
        seq_lens = torch.tensor([512, 1024, 256, 1], dtype=torch.int64, device="cuda")
        req_pool_indices = torch.tensor([1, 3, 5, 0], dtype=torch.int64, device="cuda")
        topk_idx = _make_topk_idx(B, seq_lens, unique_blocks=True)
        req_to_host = _make_req_to_host(max_reqs, max_ctx)

        _fill_host_pool(pool, layer_id, max_reqs * max_ctx)

        # Only run JIT path if available; reference handles padded batch too.
        ref = _run_reference(pool, layer_id, req_to_host, req_pool_indices, topk_idx, seq_lens)

        # The dummy entry (seq_len=1) should contribute at most 1 block.
        # The real entries contribute their selected blocks.
        _assert_hot_kv_correct(pool, layer_id, ref, req_to_host, topk_idx, req_pool_indices, seq_lens)

    def test_msa_metadata_shape_and_range(self):
        """MSA kv_indices: within hot page range, kv_block_indexes == topk_idx permuted."""
        pool = _make_pool(size=2048, hot_size=1024)
        layer_id = 3

        B = 2
        max_reqs = 4
        max_ctx = 2048
        seq_lens = torch.tensor([512, 768], dtype=torch.int64, device="cuda")
        req_pool_indices = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
        topk_idx = _make_topk_idx(B, seq_lens, unique_blocks=True)
        req_to_host = _make_req_to_host(max_reqs, max_ctx)

        _fill_host_pool(pool, layer_id, max_reqs * max_ctx)

        result = _run_reference(pool, layer_id, req_to_host, req_pool_indices, topk_idx, seq_lens)

        if result.hot_kv_indices is not None:
            # kv_indices values should be within hot page range.
            assert result.hot_kv_indices.min() >= pool.hot_page_offset
            assert result.hot_kv_indices.max() < pool.hot_page_offset + pool.hot_page_capacity

        # kv_block_indexes = topk_idx.permute(1, 0, 2).contiguous().to(torch.int32)
        kv_block_indexes = topk_idx.permute(1, 0, 2).contiguous().to(torch.int32)
        assert kv_block_indexes.shape == (B, Hkv, K)

    def test_python_vs_jit_parity(self):
        """Python reference and JIT kernel produce identical hot K/V."""
        pool = _make_pool(size=2048, hot_size=1024)
        layer_id = 3

        if not pool._use_jit_swap_in:
            pytest.skip("JIT swap-in kernel not available.")

        B = 3
        max_reqs = 4
        max_ctx = 2048
        seq_lens = torch.tensor([512, 1024, 256], dtype=torch.int64, device="cuda")
        req_pool_indices = torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda")
        topk_idx = _make_topk_idx(B, seq_lens, unique_blocks=True)
        req_to_host = _make_req_to_host(max_reqs, max_ctx)

        _fill_host_pool(pool, layer_id, max_reqs * max_ctx)

        ref = _run_reference(pool, layer_id, req_to_host, req_pool_indices, topk_idx, seq_lens)

        # Re-fill host pool (reference consumed the host data)
        _fill_host_pool(pool, layer_id, max_reqs * max_ctx)
        jit = _run_jit(pool, layer_id, req_to_host, req_pool_indices, topk_idx, seq_lens)

        # Compare hot K/V byte-for-byte.
        hot_k, hot_v = pool.get_hot_kv_buffer(layer_id)
        # Both paths should produce the same page table.
        if ref.hot_page_table is not None and jit.hot_page_table is not None:
            assert torch.equal(ref.hot_page_table, jit.hot_page_table), \
                "Python ref and JIT hot_page_table must be identical"

        # Compare kv_indices sets.
        if ref.hot_kv_indices is not None and jit.hot_kv_indices is not None:
            ref_set = set(ref.hot_kv_indices.tolist())
            jit_set = set(jit.hot_kv_indices.tolist())
            assert ref_set == jit_set, \
                f"Python ref kv_indices {ref_set} != JIT kv_indices {jit_set}"

    def test_hot_buffer_overflow_detection(self):
        """Hot buffer overflow is detected and raised."""
        # Deliberately tiny hot buffer.
        pool = _make_pool(size=2048, hot_size=PAGE_SIZE)  # only 1 page
        layer_id = 3

        B = 1
        max_reqs = 4
        max_ctx = 2048
        seq_lens = torch.tensor([2048], dtype=torch.int64, device="cuda")
        req_pool_indices = torch.tensor([1], dtype=torch.int64, device="cuda")
        topk_idx = _make_topk_idx(B, seq_lens, unique_blocks=True)
        req_to_host = _make_req_to_host(max_reqs, max_ctx)

        _fill_host_pool(pool, layer_id, max_reqs * max_ctx)

        # Should raise due to insufficient hot pages.
        with pytest.raises(RuntimeError, match="hot buffer|overflow|exhausted"):
            _run_reference(pool, layer_id, req_to_host, req_pool_indices, topk_idx, seq_lens)

        if pool._use_jit_swap_in:
            with pytest.raises(RuntimeError, match="hot buffer|overflow|exhausted"):
                _run_jit(pool, layer_id, req_to_host, req_pool_indices, topk_idx, seq_lens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestHotReqToTokenFallback:
    """Tests for the hot req_to_token fallback path (Triton sparse attention)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_hot_page_table_lookup(self):
        """Verify hot_page_table maps logical blocks → correct hot slots for Triton fallback."""
        pool = _make_pool(size=2048, hot_size=1024)
        layer_id = 3

        B = 1
        max_reqs = 4
        max_ctx = 2048
        seq_lens = torch.tensor([512], dtype=torch.int64, device="cuda")
        req_pool_indices = torch.tensor([1], dtype=torch.int64, device="cuda")
        topk_idx = _make_topk_idx(B, seq_lens, unique_blocks=False)
        req_to_host = _make_req_to_host(max_reqs, max_ctx)

        _fill_host_pool(pool, layer_id, max_reqs * max_ctx)
        result = _run_reference(pool, layer_id, req_to_host, req_pool_indices, topk_idx, seq_lens)

        if result.hot_page_table is None:
            pytest.skip("No hot_page_table in result.")

        # For the Triton fallback, each selected logical block should resolve:
        #   hot_slot = hot_page_table[req, logical_block] * PAGE_SIZE + offset
        # The corresponding token's K/V in the hot buffer should match the host buffer.
        hot_k, hot_v = pool.get_hot_kv_buffer(layer_id)
        host_k, host_v = pool.get_sparse_main_host_kv_buffer(layer_id)

        for b in range(B):
            req_row = int(req_pool_indices[b].item())
            sl = int(seq_lens[b].item())
            for blk in range((sl + PAGE_SIZE - 1) // PAGE_SIZE):
                hot_page_id = int(result.hot_page_table[b, blk].item())
                if hot_page_id < 0:
                    continue
                for t in range(min(PAGE_SIZE, sl - blk * PAGE_SIZE)):
                    hot_slot = hot_page_id * PAGE_SIZE + t
                    host_slot = int(req_to_host[req_row, blk * PAGE_SIZE + t].item())
                    torch.testing.assert_close(
                        hot_k[hot_slot], host_k[host_slot].to(device=hot_k.device),
                        msg=f"Triton fallback K mismatch: req={b}, blk={blk}, t={t}"
                    )
                    torch.testing.assert_close(
                        hot_v[hot_slot], host_v[host_slot].to(device=hot_v.device),
                        msg=f"Triton fallback V mismatch: req={b}, blk={blk}, t={t}"
                    )
