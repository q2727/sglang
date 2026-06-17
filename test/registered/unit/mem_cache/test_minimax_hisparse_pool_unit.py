import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.minimax_hisparse_memory_pool import MiniMaxHiSparseKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layer(layer_id: int):
    return SimpleNamespace(layer_id=layer_id)


def _make_pool(device: str = "cpu", dtype: torch.dtype = torch.float32):
    return MiniMaxHiSparseKVPool(
        size=8,
        page_size=4,
        dtype=dtype,
        head_num=2,
        head_dim=3,
        idx_head_dim=5,
        dense_layer_ids=[0, 1, 2],
        sparse_layer_ids=[3, 4],
        disable_value_sparse_layer_ids=[3, 4],
        device=device,
        hot_size=8,
        host_to_device_ratio=2,
        start_layer=0,
        end_layer=5,
        pin_host_memory=False,
    )


class TestMiniMaxHiSparseKVPool(unittest.TestCase):
    def test_layer_mapping_and_buffer_shapes_cpu(self):
        pool = _make_pool()

        self.assertEqual(pool.dense_layer_id_mapping, {0: 0, 1: 1, 2: 2})
        self.assertEqual(pool.sparse_layer_id_mapping, {3: 0, 4: 1})
        self.assertEqual(pool.index_kv_layer_id_mapping, {})
        self.assertIsNone(pool.index_kv_pool)

        dense_k, dense_v = pool.get_kv_buffer(1)
        self.assertEqual(dense_k.shape, (12, 2, 3))
        self.assertEqual(dense_v.shape, (12, 2, 3))
        self.assertEqual(dense_k.dtype, torch.float32)

        idx_k = pool.get_index_k_buffer(3)
        self.assertEqual(idx_k.shape, (12, 1, 5))

        hot_k, hot_v = pool.get_hot_kv_buffer(3)
        self.assertEqual(hot_k.shape, (12, 2, 3))
        self.assertEqual(hot_v.shape, (12, 2, 3))

        host_k, host_v = pool.get_sparse_main_host_kv_buffer(3)
        self.assertEqual(host_k.shape, (20, 2, 3))
        self.assertEqual(host_v.shape, (20, 2, 3))

    def test_dense_get_kv_buffer_returns_full_gpu_resident_pool(self):
        pool = _make_pool()
        loc = torch.tensor([1, 3], dtype=torch.int64)
        k = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        v = k + 100

        pool.set_kv_buffer(_layer(0), loc, k, v)

        dense_k, dense_v = pool.get_kv_buffer(0)
        self.assertTrue(torch.equal(dense_k[loc], k))
        self.assertTrue(torch.equal(dense_v[loc], v))

    def test_fused_sparse_store_writes_index_k_and_host_main(self):
        pool = _make_pool()
        loc = torch.tensor([4, 5], dtype=torch.int64)
        k = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        v = k + 10
        idx_k = torch.arange(10, dtype=torch.float32).reshape(2, 1, 5)

        pool.set_fused_kv_index_buffer(_layer(3), loc, k, v, idx_k, None)

        got_idx_k = pool.get_index_k_buffer(3)
        self.assertTrue(torch.equal(got_idx_k[loc], idx_k))

        host_k, host_v = pool.get_sparse_main_host_kv_buffer(3)
        self.assertTrue(torch.equal(host_k[loc], k))
        self.assertTrue(torch.equal(host_v[loc], v))

        hot_k, hot_v = pool.get_hot_kv_buffer(3)
        self.assertTrue(torch.equal(hot_k[loc], torch.zeros_like(hot_k[loc])))
        self.assertTrue(torch.equal(hot_v[loc], torch.zeros_like(hot_v[loc])))

    def test_host_backup_and_direct_hot_reload_cpu(self):
        pool = _make_pool()
        host_locs = torch.tensor([6, 7], dtype=torch.int64)
        hot_locs = torch.tensor([4, 5], dtype=torch.int64)
        k = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        v = k + 20

        pool.backup_sparse_main_to_host(3, host_locs, cache_k=k, cache_v=v)
        result = pool.load_sparse_main_blocks_to_hot(
            3, host_locs=host_locs, hot_locs=hot_locs
        )

        hot_k, hot_v = pool.get_hot_kv_buffer(3)
        self.assertTrue(torch.equal(result.host_locs, host_locs))
        self.assertTrue(torch.equal(result.hot_locs.cpu(), hot_locs))
        self.assertTrue(torch.equal(hot_k[hot_locs], k))
        self.assertTrue(torch.equal(hot_v[hot_locs], v))

    def test_topk_block_load_builds_hot_page_table(self):
        pool = _make_pool()
        layer_id = 3
        host_locs = torch.arange(10, 18, dtype=torch.int64)
        k = torch.arange(8 * 2 * 3, dtype=torch.float32).reshape(8, 2, 3)
        v = k + 50
        pool.backup_sparse_main_to_host(layer_id, host_locs, cache_k=k, cache_v=v)

        req_to_host = torch.full((2, 8), -1, dtype=torch.int64)
        req_to_host[1, :8] = host_locs
        topk_idx = torch.tensor([[[1, 0]]], dtype=torch.int32)  # [Hkv, B, K]
        seq_lens = torch.tensor([8], dtype=torch.int64)
        req_pool_indices = torch.tensor([1], dtype=torch.int64)

        result = pool.load_sparse_main_blocks_to_hot(
            layer_id,
            req_to_host=req_to_host,
            req_pool_indices=req_pool_indices,
            topk_idx=topk_idx,
            seq_lens=seq_lens,
        )

        hot_k, hot_v = pool.get_hot_kv_buffer(layer_id)
        page_table = pool.get_hot_page_table(layer_id, flattened=False)
        kv_indices = pool.get_hot_page_table(layer_id)

        # Blocks load in top-k order: logical block 1 -> hot page 1,
        # then logical block 0 -> hot page 2.
        self.assertTrue(torch.equal(page_table, torch.tensor([[2, 1]], dtype=torch.int32)))
        self.assertTrue(torch.equal(kv_indices, torch.tensor([2, 1], dtype=torch.int32)))
        self.assertTrue(torch.equal(result.hot_page_table, page_table))
        self.assertTrue(torch.equal(result.hot_kv_indices, kv_indices))
        self.assertTrue(torch.equal(hot_k[4:8], k[4:8]))
        self.assertTrue(torch.equal(hot_v[4:8], v[4:8]))
        self.assertTrue(torch.equal(hot_k[8:12], k[0:4]))
        self.assertTrue(torch.equal(hot_v[8:12], v[0:4]))

    def test_index_v_unsupported_fails_clearly(self):
        with self.assertRaisesRegex(NotImplementedError, "K-only sparse index"):
            MiniMaxHiSparseKVPool(
                size=8,
                page_size=4,
                dtype=torch.float32,
                head_num=2,
                head_dim=3,
                idx_head_dim=5,
                dense_layer_ids=[0],
                sparse_layer_ids=[1],
                disable_value_sparse_layer_ids=[],
                device="cpu",
                hot_size=4,
                start_layer=0,
                end_layer=2,
                pin_host_memory=False,
            )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required.")
    def test_cuda_host_backup_and_hot_reload(self):
        pool = _make_pool(device="cuda", dtype=torch.bfloat16)
        host_locs = torch.tensor([6, 7], dtype=torch.int64, device="cuda")
        hot_locs = torch.tensor([4, 5], dtype=torch.int64, device="cuda")
        k = torch.arange(12, dtype=torch.bfloat16, device="cuda").reshape(2, 2, 3)
        v = k + 20

        pool.backup_sparse_main_to_host(3, host_locs, cache_k=k, cache_v=v)
        pool.load_sparse_main_blocks_to_hot(3, host_locs=host_locs, hot_locs=hot_locs)

        hot_k, hot_v = pool.get_hot_kv_buffer(3)
        self.assertTrue(torch.equal(hot_k[hot_locs].cpu(), k.cpu()))
        self.assertTrue(torch.equal(hot_v[hot_locs].cpu(), v.cpu()))


if __name__ == "__main__":
    unittest.main()
