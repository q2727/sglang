from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import (
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    get_minimax_sparse_score_type,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.minimax_sparse_ops.decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
)
from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
    minimax_sparse_decode,
    minimax_sparse_prefill,
)
from sglang.srt.mem_cache.minimax_hisparse_memory_pool import MiniMaxHiSparseKVPool
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class MiniMaxSparseAttnBackend(AttentionBackend):
    def __init__(self, runner: ModelRunner):
        assert isinstance(
            runner.token_to_kv_pool, (MiniMaxSparseKVPool, MiniMaxHiSparseKVPool)
        )
        self.kv_pool = runner.token_to_kv_pool
        self.req_to_token = runner.req_to_token_pool.req_to_token
        self.max_context_len = int(runner.model_config.context_len)

        hf_config = runner.model_config.hf_config
        sparse_cfg = get_minimax_sparse_attention_config(hf_config)
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.dense_layer_ids, self.sparse_layer_ids = get_minimax_sparse_layer_ids(
            sparse_cfg
        )
        self.disable_value_layer_ids: set[int] = set(
            get_minimax_sparse_disable_value_layer_ids(sparse_cfg)
        )
        self.score_type: str = get_minimax_sparse_score_type(sparse_cfg)
        # assert self.idx_head_dim == head_dim

        # max_seqlen for the current forward pass, stored as a plain Python int
        # so that it is safe to use inside CUDA graphs (no .item() at graph time).
        # Populated by init_forward_metadata* before each forward.
        self._max_seqlen_q: int = 1
        self._max_seqlen_k: int = 1

        self.block_size_q = 1
        self.block_size_k = sparse_cfg["sparse_block_size"]
        if "sparse_init_block" in sparse_cfg:
            self.init_blocks = sparse_cfg["sparse_init_block"]
        else:
            init_tokens = sparse_cfg["sparse_init_tokens"]
            self.init_blocks = (
                init_tokens + self.block_size_k - 1
            ) // self.block_size_k
        if "sparse_local_block" in sparse_cfg:
            self.local_blocks = sparse_cfg["sparse_local_block"]
        else:
            local_tokens = sparse_cfg["sparse_local_tokens"]
            self.local_blocks = (
                local_tokens + self.block_size_k - 1
            ) // self.block_size_k + 1
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]

        # NVIDIA Blackwell (SM100): use MiniMax's MSA kernel (fmha_sm100) only
        # for the main sparse-attention step when the kernel constraints hold.
        # The lightning indexer remains unchanged; missing fmha_sm100 keeps the
        # existing Triton path.
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
            msa_available,
        )

        # MSA (fmha_sm100) is bf16/fp16-only. With an fp8 main KV cache
        # (--kv-cache-dtype fp8_*) keep the sparse path on Triton (it dequants fp8 on
        # load) rather than feeding fp8 bytes to the bf16 kernel; mirrors vLLM's
        # select_main_impl_cls (fp8 KV -> Triton, never MSA).
        _main_kv_is_fp8 = self.kv_pool.main_pool.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
        self.use_msa = (
            not envs.SGLANG_DISABLE_MSA.get()
            and msa_available()
            and self.block_size_k == 128
            and self.kv_pool.page_size == self.block_size_k
            and self.topk_blocks in (4, 8, 16, 32)
            and not _main_kv_is_fp8
        )
        # Per-forward MSA decode metadata (page table + fmha plan), shared by every
        # sparse layer of a forward; (re)built in init_forward_metadata_out_graph.
        self._msa_dec_meta = None
        if self.use_msa:
            from sglang.srt.layers.dp_attention import get_attention_tp_size

            # Per-rank head counts for the decode plan (== runtime q.shape[1] /
            # k_cache.shape[1]); needed in out_graph where q/k_cache aren't available.
            self.num_q_heads = (
                runner.model_config.num_attention_heads // get_attention_tp_size()
            )
            # KV head count lives on the main sub-pool (== runtime k_cache.shape[1]).
            self.num_kv_heads = self.kv_pool.main_pool.head_num
            # CUDA-graph decode: one persistent plan + page-table buffer per batch
            # size, refreshed in place each step (worklist is length-independent).
            self._msa_nb_max = (
                self.max_context_len + self.block_size_k - 1
            ) // self.block_size_k
            self._msa_cg: dict[int, tuple] = {}

        self.page_size = self.kv_pool.page_size
        self.use_dense_sparse_decode = (
            envs.SGLANG_OPT_USE_MINIMAX_DENSE_SPARSE_DECODE.get()
            and self.block_size_k % self.page_size == 0
        )
        # MSA fmha_sm100 decode is NOT cuda-graph-safe: captured & replayed it returns
        # wrong results that compound across replays (silent ~14% GSM8K loss on B200;
        # masked early by radix-cache prefix reuse, then cliffs under sustained load).
        # Use the MSA decode kernel only when decode does NOT run under cuda graph;
        # otherwise route the decode step through the cuda-graph-safe Triton sparse path.
        # MSA still serves prefill (run eager — prefill cuda graph is disabled), where
        # its long-context speedup matters.
        #
        # Decide from the resolved cuda_graph_config — the same source
        # init_decode_cuda_graph uses to decide capture — not the legacy disable_*
        # server_args flags: the two can disagree under config-native flags, and a
        # mismatch could capture the unsafe MSA decode kernel into a graph.
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            Phase,
            check_cuda_graph_backend,
        )

        _sa = getattr(runner, "server_args", None)
        _decode_cuda_graph = not check_cuda_graph_backend(
            Phase.DECODE, Backend.DISABLED
        )
        self._use_msa_decode = self.use_msa and not _decode_cuda_graph

        # MSA + speculative decode + cuda graph is unsupported: spec verify
        # (TARGET_VERIFY) batches route to forward_extend and are captured into the
        # decode graph, which both dereferences extend metadata absent in the capture
        # batch and would record the MSA prefill kernel into a graph. Fail loudly at
        # startup instead of crashing mid-capture.
        if (
            self.use_msa
            and _decode_cuda_graph
            and getattr(_sa, "speculative_algorithm", None) is not None
        ):
            raise NotImplementedError(
                "MiniMax-M3 MSA attention does not support speculative decoding under "
                "CUDA graph. Use --disable-cuda-graph, set SGLANG_DISABLE_MSA=1, or "
                "disable speculative decoding."
            )
        # MSA owns the main decode step unless dense-sparse-decode does; the dense
        # path only engages when k_cache.shape[1] == 1 (see forward_decode).
        self._msa_owns_decode = self._use_msa_decode and not (
            self.use_dense_sparse_decode and self.kv_pool.main_pool.head_num == 1
        )
        # The page table + effective KV length are allocated and returned by the
        # fused decode top-k kernel each layer, so the backend keeps no metadata.
        self.dense_backend: Optional[AttentionBackend] = None

        # ── Two-pool HiSparse gate ──────────────────────────────────────────
        # When HiSparse is enabled, TWO pools coexist:
        #   standard_kv_pool (MiniMaxSparseKVPool): full GPU K/V for prefill,
        #     dense layers, and sparse index K.
        #   kv_pool / hisparse_kv_pool (MiniMaxHiSparseKVPool): host-backed
        #     sparse main K/V + hot GPU buffer for sparse decode.
        self.standard_kv_pool: Optional["MiniMaxSparseKVPool"] = getattr(
            runner, "standard_kv_pool", None
        )
        self._is_m3_hisparse = (
            self.standard_kv_pool is not None
            and isinstance(self.kv_pool, MiniMaxHiSparseKVPool)
        )

        # Coordinator reference (plumbed from ModelRunner.hisparse_coordinator).
        self._m3_hisparse_coordinator: Optional[
            "MiniMaxHiSparseCoordinator"
        ] = None
        if self._is_m3_hisparse:
            self._m3_hisparse_coordinator = getattr(
                runner, "hisparse_coordinator", None
            )
            if self._m3_hisparse_coordinator is None:
                raise RuntimeError(
                    "MiniMaxHiSparseKVPool + standard_kv_pool in use but "
                    "hisparse_coordinator was not set on ModelRunner."
                )

        # ── CUDA graph policy for HiSparse ──
        # First-phase M3 HiSparse decode does not support CUDA graph capture.
        if self._is_m3_hisparse and _decode_cuda_graph:
            raise RuntimeError(
                "MiniMax-M3 HiSparse does not support CUDA graph decode. "
                "Use --disable-cuda-graph to force eager execution for all layers."
            )

        # GPU mirror of coordinator.req_to_host for JIT swap-in kernel.
        self._hisparse_req_to_host_gpu: Optional[torch.Tensor] = None

        logger.info(
            f"[MiniMaxSparse] Backend initialized "
            f"(score_type={self.score_type!r}, "
            f"main_attn={'MSA' if self.use_msa else 'triton'}, "
            f"hisparse={self._is_m3_hisparse}, "
            f"disable_value_layers={sorted(self.disable_value_layer_ids)})"
        )

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        # cuda-graph replay views are a SimpleNamespace without extend_seq_lens_cpu,
        # and TARGET_VERIFY sets it to None despite is_extend() — getattr covers both.
        # New forward -> invalidate the cached per-forward MSA decode metadata.
        self._msa_dec_meta = None
        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is not None:
            self._max_seqlen_q = int(max(extend_lens))
        else:
            self._max_seqlen_q = 1
        if in_capture and forward_batch.forward_mode.is_decode_or_idle():
            self._max_seqlen_k = self.max_context_len
        else:
            self._max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())

        # Build the MSA decode plan + page table here (eager, outside graph capture)
        # so forward_decode — captured into the graph — only runs device-side ops.
        # Runs at capture, replay, and eager, refreshing the persistent buffers the
        # captured graph reads. Skipped when the dense-sparse-decode path owns decode.
        if self._msa_owns_decode and forward_batch.forward_mode.is_decode_or_idle():
            self._prepare_msa_decode_meta(forward_batch)

    def _prepare_msa_decode_meta(self, forward_batch: ForwardBatch):
        """Refresh the persistent per-batch-size MSA decode plan + page table in place."""
        from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
            build_msa_decode_cg_plan,
            update_msa_decode_cg_meta,
        )

        bs = forward_batch.seq_lens.shape[0]
        if bs == 0:
            return
        entry = self._msa_cg.get(bs)
        if entry is None:
            device = forward_batch.seq_lens.device
            plan = build_msa_decode_cg_plan(
                self.num_q_heads,
                self.num_kv_heads,
                self.block_size_k,
                self.topk_blocks,
                bs,
                device=device,
            )
            kv_indices_buf = torch.zeros(
                bs * self._msa_nb_max, dtype=torch.int32, device=device
            )
            entry = (plan, kv_indices_buf)
            self._msa_cg[bs] = entry
        plan, kv_indices_buf = entry
        update_msa_decode_cg_meta(
            plan,
            kv_indices_buf,
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self.block_size_k,
            self.topk_blocks,
            self.num_q_heads,
            self.num_kv_heads,
        )
        self._msa_dec_meta = (kv_indices_buf, plan)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        pass

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    @staticmethod
    def _is_sparse_kv_cached_by_fusion(
        forward_batch: ForwardBatch, layer_id: int
    ) -> bool:
        layer_ids = forward_batch.minimax_m3_precached_sparse_layers
        return layer_ids is not None and layer_id in layer_ids

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if forward_batch.forward_mode.is_idle():
            idx_q = kwargs.get("idx_q")
            num_idx_heads = idx_q.shape[1]
            disable_value = layer.layer_id in self.disable_value_layer_ids
            idx_out: Optional[torch.Tensor] = (
                None
                if disable_value
                else q.new_zeros(q.shape[0], num_idx_heads * self.idx_head_dim)
            )
            out = q.new_zeros(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
            return idx_out, out
        else:
            return super().forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
    ):
        # ── Prefill: always use the standard pool ──
        # When HiSparse is enabled, prefill writes ALL K/V (dense main,
        # sparse main, sparse index) to the standard GPU pool.  The HiSparse
        # host/hot pool is only consulted during decode.
        p = self.standard_kv_pool if self._is_m3_hisparse else self.kv_pool
        disable_value = layer.layer_id in self.disable_value_layer_ids
        kv_cached_by_fusion = self._is_sparse_kv_cached_by_fusion(
            forward_batch, layer.layer_id
        )
        if not kv_cached_by_fusion:
            p.set_fused_kv_index_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                idx_k,
                None if disable_value else idx_v,
            )
        k_cache, v_cache = p.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = p.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = p.get_index_kv_buffer(layer.layer_id)

        cu_seqlens = torch.cat(
            [
                torch.zeros(
                    1, dtype=torch.int32, device=forward_batch.extend_seq_lens.device
                ),
                forward_batch.extend_seq_lens.to(torch.int32).cumsum(0).to(torch.int32),
            ]
        )
        seq_lens = forward_batch.seq_lens.to(torch.int32)  # prefix + extend
        if forward_batch.extend_prefix_lens is not None:
            prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32)
        else:
            prefix_lens = torch.zeros_like(seq_lens)

        # In DP attention mode, q may be padded beyond the actual token count
        # for collective communication alignment. Trim to actual tokens so
        # the sparse attention kernel sees consistent shapes.
        #
        # Source the token count from CPU-side metadata when available so we do
        # not force a GPU->CPU sync (cu_seqlens[-1].item()) on every sparse
        # layer of every prefill. extend_seq_lens_cpu is a plain list of ints
        # (ForwardBatch sets it from extend_seq_lens.cpu()), so sum() is a host
        # op and the result is identical to cu_seqlens[-1]. Fall back to the
        # device tensor only when CPU metadata is absent.
        if forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
        else:
            actual_num_tokens = int(cu_seqlens[-1].item())
        original_num_tokens = q.shape[0]
        if actual_num_tokens < original_num_tokens:
            q = q[:actual_num_tokens]
            idx_q = idx_q[:actual_num_tokens]

        idx_o, o = minimax_sparse_prefill(
            q,
            k_cache,
            v_cache,
            None,
            idx_q,
            idx_k_cache,
            idx_v_cache,
            None,
            self.req_to_token,
            forward_batch.req_pool_indices,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            self._max_seqlen_q,
            self._max_seqlen_k,
            self.block_size_q,
            self.block_size_k,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
            use_msa=self.use_msa,
            # Host seq-lens let get_cu_seqblocks avoid a per-layer .item() sync.
            seqlens_cpu=forward_batch.extend_seq_lens_cpu,
        )

        # Pad output back to original size for DP communication
        if actual_num_tokens < original_num_tokens:
            pad_len = original_num_tokens - actual_num_tokens
            o = torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)
            if idx_o is not None:
                idx_o = torch.cat(
                    [idx_o, idx_o.new_zeros(pad_len, *idx_o.shape[1:])], dim=0
                )

        return (
            (
                None
                if idx_o is None
                else idx_o.reshape(original_num_tokens, -1).contiguous()
            ),
            o.reshape(original_num_tokens, -1).contiguous(),
        )

    def _dense_sparse_main_decode(
        self,
        q: torch.Tensor,  # [bs, num_q_heads, head_dim]
        page_table: torch.Tensor,  # [bs, max_sparse_pages] int32 (from the indexer)
        real_seq_lens: torch.Tensor,  # [bs] int32, effective KV length per query
        k_cache: torch.Tensor,  # [max_slots, 1, head_dim]
        v_cache: torch.Tensor,  # [max_slots, 1, head_dim]
        layer,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        if isinstance(self.dense_backend, TRTLLMHAAttnBackend):
            import flashinfer

            ps = self.page_size
            nkv = 1
            head_dim = q.size(-1)
            # [max_slots, nkv, D] -> [num_pages, page_size, nkv, D]
            #                     -> [num_pages, nkv, page_size, D] (HND, trtllm default)
            kc = k_cache.view(-1, ps, nkv, head_dim).permute(0, 2, 1, 3)
            vc = v_cache.view(-1, ps, nkv, head_dim).permute(0, 2, 1, 3)
            return flashinfer.decode.trtllm_batch_decode_with_kv_cache(  # type: ignore
                query=q.contiguous(),
                kv_cache=(kc, vc),
                workspace_buffer=self.dense_backend.workspace_buffer,
                block_tables=page_table,
                seq_lens=real_seq_lens,
                max_seq_len=self.topk_blocks * self.block_size_k,
                bmm1_scale=layer.scaling,
                bmm2_scale=1.0,
            )
        raise NotImplementedError(
            "dense sparse decode currently supports trtllm_mha only (fa3 is TODO)"
        )

    # ── HiSparse helpers ───────────────────────────────────────────────────

    def _get_req_to_host_gpu(self) -> torch.Tensor:
        """Return a GPU mirror of coordinator.req_to_host.

        The coordinator owns ``req_to_host`` on CPU.  The JIT swap-in kernel
        requires a GPU copy; this method creates and maintains a lazy GPU
        mirror.  For large contexts this full-tensor copy is a known first-phase
        bottleneck — a future optimization is for the coordinator to maintain
        the GPU mirror incrementally.
        """
        coord = self._m3_hisparse_coordinator
        cpu_tensor = coord.req_to_host  # [max_reqs, max_ctx] int64 CPU
        if cpu_tensor.device.type in ("cuda", "hip"):
            return cpu_tensor

        device = torch.device(self.kv_pool.device)
        if (
            self._hisparse_req_to_host_gpu is None
            or self._hisparse_req_to_host_gpu.shape != cpu_tensor.shape
        ):
            self._hisparse_req_to_host_gpu = cpu_tensor.to(
                device=device, dtype=torch.int64, non_blocking=False
            )
        else:
            self._hisparse_req_to_host_gpu.copy_(cpu_tensor, non_blocking=False)
        return self._hisparse_req_to_host_gpu

    def _get_host_locs_for_decode(
        self, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """Return host pool slot indices for the current decode token per request.

        Reads from ``coordinator.req_to_host[req_pool_idx, seq_len - 1]``.
        Returns a 1-D int64 CPU tensor of length batch_size.
        """
        coord = self._m3_hisparse_coordinator
        batch_size = forward_batch.batch_size
        host_locs = torch.empty(batch_size, dtype=torch.int64)
        # Use CPU-side copies when available to avoid GPU sync;
        # fall back to GPU tensors with .item() for each slot.
        req_pool_cpu = getattr(
            forward_batch, "req_pool_indices_cpu", None
        )
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if req_pool_cpu is None:
            req_pool_cpu = forward_batch.req_pool_indices.cpu()
        if seq_lens_cpu is None:
            seq_lens_cpu = forward_batch.seq_lens.cpu()

        for b in range(batch_size):
            req_idx = int(req_pool_cpu[b].item())
            sl = int(seq_lens_cpu[b].item())
            host_locs[b] = int(coord.req_to_host[req_idx, sl - 1].item())
        return host_locs

    @staticmethod
    def _run_index_branch_decode(
        idx_q: torch.Tensor,
        idx_k_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_ids: torch.Tensor,
        max_seqlen: int,
        block_size_k: int,
        topk_blocks: int,
        init_blocks: int,
        local_blocks: int,
        score_type: str,
        page_size: int,
    ) -> torch.Tensor:
        """Run the MiniMax index branch and return ``topk_idx`` [Hidx, B, K].

        This is the first half of ``minimax_sparse_decode`` — the indexer —
        extracted so the HiSparse path can intercept between index selection
        and main sparse attention.
        """
        _idx_o, topk_idx, _real_seq_lens = flash_decode_with_topk_idx(
            q=idx_q,
            sink=None,
            k_cache=idx_k_cache,
            v_cache=None,  # K-only index; disable_index_value=True
            req_to_token=req_to_token,
            seq_lens=seq_lens,
            max_seqlen=max_seqlen,
            slot_ids=slot_ids,
            block_size=block_size_k,
            topk=topk_blocks,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            score_type=score_type,
            disable_index_value=True,
            use_dense_main_attn=False,
            page_size=page_size,
        )
        return topk_idx  # [Hidx, B, K]

    @staticmethod
    def _reduce_topk_idx(
        topk_idx: torch.Tensor,  # [Hidx, B, K]
        num_idx_heads: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        """Reduce index-head topk to KV-head topk when Hidx > Hkv."""
        idx_group_size = num_idx_heads // num_kv_heads
        if idx_group_size > 1:
            from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
                topk_index_reduce,
            )

            return topk_index_reduce(
                topk_idx.view(num_kv_heads, idx_group_size, -1, topk_idx.shape[-1]),
                dim=1,
            )
        return topk_idx  # shape unchanged when Hidx == Hkv

    def _swap_sparse_main_blocks_to_hot(
        self,
        layer_id: int,
        topk_idx: torch.Tensor,  # [Hkv, B, K] int32
        forward_batch: ForwardBatch,
    ) -> None:
        """Swap selected sparse main K/V blocks into the hot GPU buffer."""
        req_to_host_gpu = self._get_req_to_host_gpu()
        self.kv_pool.load_sparse_main_blocks_to_hot(
            layer_id,
            req_to_host=req_to_host_gpu,
            req_pool_indices=forward_batch.req_pool_indices,
            topk_idx=topk_idx,
            seq_lens=forward_batch.seq_lens,
            block_size=self.block_size_k,
        )

    def _build_hot_msa_meta_for_layer(
        self,
        layer_id: int,
        forward_batch: ForwardBatch,
    ):
        """Build per-layer MSA decode metadata pointing to the hot buffer."""
        from fmha_sm100 import fmha_sm100_plan

        hot_kv_indices = self.kv_pool.get_hot_page_table(
            layer_id, flattened=True
        )
        P = self.block_size_k
        B = forward_batch.seq_lens.shape[0]
        seq_lens_i32 = forward_batch.seq_lens.to(torch.int32)
        plan = fmha_sm100_plan(
            torch.ones(B, dtype=torch.int32),
            seq_lens_i32,
            self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            page_size=P,
            kv_block_num=self.topk_blocks,
            causal=False,
            qo_offset=(seq_lens_i32 - 1).clamp_min(0),
            device=forward_batch.seq_lens.device,
        )
        return hot_kv_indices, plan

    def _build_hot_req_to_token_for_layer(
        self,
        layer_id: int,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Build a layer-scoped hot ``req_to_token`` for the Triton fallback."""
        return self.kv_pool.build_hot_req_to_token(
            layer_id=layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            max_reqs=self.req_to_token.shape[0],
            max_ctx=self.req_to_token.shape[1],
            num_real_reqs=int(forward_batch.batch_size),
        )

    def _run_main_sparse_attention_hisparse(
        self,
        q: torch.Tensor,
        topk_idx: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Run main sparse attention from the *hot* GPU K/V buffer."""
        layer_id = layer.layer_id
        hot_k, hot_v = self.kv_pool.get_hot_kv_buffer(layer_id)
        sm_scale = getattr(layer, "scaling", None)

        if self._use_msa_decode:
            hot_kv_indices, hot_plan = self._build_hot_msa_meta_for_layer(
                layer_id, forward_batch
            )
            from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
                msa_sparse_decode_main,
            )

            o = msa_sparse_decode_main(
                q=q,
                k_cache=hot_k,
                v_cache=hot_v,
                topk_idx=topk_idx,
                req_to_token=self.req_to_token,
                slot_ids=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                block_size_k=self.block_size_k,
                sm_scale=sm_scale,
                kv_indices=hot_kv_indices,
                plan=hot_plan,
            )
        else:
            hot_req_to_token = self._build_hot_req_to_token_for_layer(
                layer_id, forward_batch
            )
            from sglang.srt.layers.attention.minimax_sparse_ops.decode.topk_sparse import (
                flash_decode_with_gqa_share_sparse,
            )

            o = flash_decode_with_gqa_share_sparse(
                q=q,
                sink=None,
                k_cache=hot_k,
                v_cache=hot_v,
                req_to_token=hot_req_to_token,
                seq_lens=forward_batch.seq_lens,
                slot_ids=forward_batch.req_pool_indices,
                block_size=self.block_size_k,
                topk_idx=topk_idx,
                sm_scale=sm_scale,
            )
        return o

    def _forward_decode_hisparse_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        disable_value: bool,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
    ):
        """HiSparse decode path for one sparse MiniMax-M3 layer.

        Two-pool design:
        1. Store main K/V → HiSparse host pool.
        2. Store index K → standard pool (GPU, so the indexer sees full context).
        3. Read index K from standard pool → run indexer → topk_idx.
        4. Swap selected main K/V blocks from host → HiSparse hot GPU buffer.
        5. Run main sparse attention from hot K/V (MSA or Triton fallback).
        """
        hisparse = self.kv_pool  # MiniMaxHiSparseKVPool
        standard = self.standard_kv_pool  # MiniMaxSparseKVPool
        layer_id = layer.layer_id

        # Step 1: Store main K/V → HiSparse host pool.
        # Host locs come from the coordinator's req_to_host mapping
        # (NOT from out_cache_loc, which are GPU slots in the standard pool).
        host_locs = self._get_host_locs_for_decode(forward_batch)
        hisparse.backup_sparse_main_to_host(
            layer_id,
            host_locs,
            cache_k=k,
            cache_v=v,
        )

        # Step 2: Store index K → standard pool (GPU).
        # The standard pool must contain the full index K history so the
        # indexer sees the latest token for local_blocks selection.
        standard.set_index_k_buffer(layer, forward_batch.out_cache_loc, idx_k)

        # Step 3: Read index K from standard pool and run the indexer.
        idx_k_cache = standard.get_index_k_buffer(layer_id)
        topk_idx = self._run_index_branch_decode(
            idx_q=idx_q,
            idx_k_cache=idx_k_cache,
            req_to_token=self.req_to_token,
            seq_lens=forward_batch.seq_lens,
            slot_ids=forward_batch.req_pool_indices,
            max_seqlen=self._max_seqlen_k,
            block_size_k=self.block_size_k,
            topk_blocks=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            score_type=self.score_type,
            page_size=self.page_size,
        )
        num_idx_heads = idx_q.shape[1]
        topk_idx = self._reduce_topk_idx(
            topk_idx,
            num_idx_heads,
            standard.main_pool.head_num,
        )

        # Step 4: Swap selected sparse main K/V blocks into hot GPU buffer.
        self._swap_sparse_main_blocks_to_hot(
            layer_id, topk_idx, forward_batch
        )

        # Step 5: Run main sparse attention from hot K/V.
        o = self._run_main_sparse_attention_hisparse(
            q, topk_idx, layer, forward_batch
        )

        # idx_o is always None for M3 K-only sparse layers.
        return (
            None,
            o.reshape(q.shape[0], -1).contiguous(),
        )

    # ── forward_decode ────────────────────────────────────────────────────

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        **kwargs,
    ):
        assert len(kwargs) == 0
        disable_value = layer.layer_id in self.disable_value_layer_ids

        # ── HiSparse sparse decode path ──
        if (
            self._is_m3_hisparse
            and layer.layer_id in self.sparse_layer_ids
            and forward_batch.forward_mode.is_decode_or_idle()
        ):
            return self._forward_decode_hisparse_sparse(
                q, k, v, layer, forward_batch, disable_value,
                idx_q, idx_k, idx_v,
            )

        # ── Standard / non-HiSparse path (also HiSparse dense layers) ──
        self.kv_pool.set_fused_kv_index_buffer(
            layer,
            forward_batch.out_cache_loc,
            k,
            v,
            idx_k,
            None if disable_value else idx_v,
        )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        attn_fn = None
        if self.use_dense_sparse_decode and k_cache.shape[1] == 1:

            def attn_fn(main_q, page_table, real_seq_lens):
                return self._dense_sparse_main_decode(
                    main_q,
                    page_table,
                    real_seq_lens,
                    k_cache,
                    v_cache,
                    layer,
                    forward_batch,
                )

        msa_kv_indices = msa_plan = None
        if self._use_msa_decode and attn_fn is None:
            if self._msa_dec_meta is not None:
                msa_kv_indices, msa_plan = self._msa_dec_meta
            elif q.shape[0] > 0:
                raise RuntimeError(
                    "MSA decode metadata missing: init_forward_metadata_out_graph "
                    "did not prepare the plan for this forward (gate mismatch)."
                )

        idx_o, o = minimax_sparse_decode(
            q,
            None,
            k_cache,
            v_cache,
            idx_q,
            None,
            idx_k_cache,
            idx_v_cache,
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self._max_seqlen_k,
            1,
            self.block_size_k,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
            dense_main_attn_fn=attn_fn,
            page_size=self.page_size,
            use_msa=self._use_msa_decode,
            msa_kv_indices=msa_kv_indices,
            msa_plan=msa_plan,
        )
        return (
            None if idx_o is None else idx_o.reshape(q.shape[0], -1).contiguous(),
            o.reshape(q.shape[0], -1).contiguous(),
        )


class MiniMaxHybridAttnBackend(AttentionBackend):
    """Combines a dense backend and a sparse backend, routing by call site."""

    def __init__(
        self,
        dense_backend: AttentionBackend,
        sparse_backend: MiniMaxSparseAttnBackend,
        sparse_layer_ids: list[int],
    ):
        self.dense = dense_backend
        self.sparse = sparse_backend
        self.sparse_layer_ids = sparse_layer_ids
        # Let the sparse decode reuse the dense paged backend (page table + workspace).
        self.sparse.dense_backend = dense_backend

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # delegate so the dense (FlashInfer) backend keeps its own eager init.
        self.sparse.init_forward_metadata(forward_batch)
        self.dense.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        self.sparse.init_forward_metadata_out_graph(forward_batch, in_capture)
        self.dense.init_forward_metadata_out_graph(forward_batch, in_capture)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        self.sparse.init_forward_metadata_in_graph(forward_batch)
        self.dense.init_forward_metadata_in_graph(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.dense.init_cuda_graph_state(max_bs, max_num_tokens)
        self.sparse.init_cuda_graph_state(max_bs, max_num_tokens)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.sparse.get_cuda_graph_seq_len_fill_value()

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

        # Dense layers delegate to the stock backend (e.g. flashinfer). Under DP
        # attention the per-rank token block is padded to an even length
        # (prepare_mlp_sync_batch -> ceil_align(num_tokens, attn_cp_size * 2)), but
        # flashinfer builds qo_indptr from extend_seq_lens, so q.shape[0] (padded)
        # != qo_indptr[-1] (real) and the paged-prefill kernel raises. Trim q to
        # the real token count and re-pad the output; k/v stay untrimmed so the
        # KV-cache write stays aligned with out_cache_loc. Prefill-only.
        mode = forward_batch.forward_mode
        if mode.is_extend() and forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
            original_num_tokens = q.shape[0]
            if actual_num_tokens < original_num_tokens:
                o = self.dense.forward(
                    q[:actual_num_tokens],
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache,
                    **kwargs,
                )
                pad_len = original_num_tokens - actual_num_tokens
                return torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)

        return self.dense.forward(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        else:
            return self.dense.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def forward_decode(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        else:
            return self.dense.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
