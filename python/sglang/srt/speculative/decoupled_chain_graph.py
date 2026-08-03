"""One-graph branch chain for the decoupled enumeration drafter.

Captures the fast round's K decode steps -- forward, argmax, feed-next,
seq/pos advance -- as ONE CUDA graph per row-count bucket, replacing K
separate decode-graph replays plus the eager glue between them.

Why this is sound on this stack (all verified on-branch):
- The trtllm graph metadata rebuild is a single fused Triton kernel driven by
  DEVICE seq/req tensors and explicitly recordable into a graph
  (trtllm_mha_backend: "this body is recorded into the CUDA graph"), so the
  in-graph ``seq += 1`` between steps re-drives it correctly.
- The GDN (linear) plane's decode metadata is step-invariant here: the same
  rows and state slots serve every chain step, and the recurrent state
  advances inside the slot itself.
- The chain plan (SGLANG_ENABLE_DECOUPLED_CHAIN_PLAN) pre-writes every
  step's page-table entries and pre-allocates every step's KV slot before
  the chain runs, so the captured region touches no allocator state.

Capture happens on FIRST USE with that round's real inputs. CUDA stream
capture RECORDS WITHOUT executing, so the capture is immediately followed by one
replay -- that replay IS the round's real execution (KV writes, GDN state
advance happen exactly once). The model is already warm by then (bootstrap
and earlier rounds ran identical decode forwards), which is what makes
warmup-free capture viable; any capture failure permanently falls back to
the step-by-step path.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.dp_attention import (
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
)

logger = logging.getLogger(__name__)


class _ChainBucket:
    def __init__(self, *, rows: int, num_steps: int, device: str) -> None:
        self.rows = rows
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.input_ids = torch.zeros(rows, dtype=torch.int64, device=device)
        self.positions = torch.zeros(rows, dtype=torch.int64, device=device)
        self.seq_lens = torch.zeros(rows, dtype=torch.int64, device=device)
        self.req_rows = torch.zeros(rows, dtype=torch.int64, device=device)
        # mrope models (Qwen3.5 family) drive rotary from fb.mrope_positions,
        # not the positions arg: it needs the same static-buffer + in-graph
        # advance treatment or every replay reuses the capture round's
        # positions (the fused-extend path hit the identical trap; see the
        # engine's "runner's replay feeds positions but not the mrope plane").
        self.mrope_positions = torch.zeros(3, rows, dtype=torch.int64, device=device)
        self.out_locs = torch.zeros(num_steps, rows, dtype=torch.int64, device=device)
        self.out_tokens = torch.zeros(num_steps, rows, dtype=torch.int64, device=device)


class ChainGraphRunner:
    """Engine-private runner: one captured graph per chain row count."""

    def __init__(self, *, model_runner, num_steps: int) -> None:
        self.model_runner = model_runner
        self.num_steps = num_steps
        self.device = model_runner.device
        self._buckets: dict[int, _ChainBucket] = {}
        self._failed_rows: set[int] = set()

    def can_replay(self, rows: int) -> bool:
        bucket = self._buckets.get(rows)
        return bucket is not None and bucket.graph is not None

    def _fill(self, bucket: _ChainBucket, plan_steps, first_tokens) -> None:
        """Stage this round's values into the bucket's static buffers.

        ``plan_steps`` is the chain plan's [(fb, slots)] list: fb0 carries the
        round's positions/seq_lens (post-prepare_for_decode for step 0), each
        step its own out_cache_loc.
        """
        fb0 = plan_steps[0][0]
        bucket.input_ids.copy_(first_tokens.to(torch.int64))
        bucket.positions.copy_(fb0.positions)
        bucket.seq_lens.copy_(fb0.seq_lens)
        # Pool rows are PER SEAT: a new request lands on new carrier rows, so
        # the graph must read them through a static buffer (baking request
        # 1's tensor made request 2 read stale rows -- acc 3.95 then 1.0).
        bucket.req_rows.copy_(fb0.req_pool_indices.to(torch.int64))
        if fb0.mrope_positions is not None:
            bucket.mrope_positions.copy_(fb0.mrope_positions)
        for s, (_, slots) in enumerate(plan_steps):
            bucket.out_locs[s].copy_(slots)

    def try_capture_and_run(
        self, *, rows: int, plan_steps, first_tokens: torch.Tensor
    ) -> Optional[list[torch.Tensor]]:
        """First use for this row count: capture the K-step chain while
        EXECUTING it (this round's real work happens inside the capture).
        Returns the chain tokens, or None on capture failure (caller falls
        back; the bucket is then permanently disabled)."""
        if rows in self._failed_rows:
            return None
        bucket = self._buckets.get(rows)
        if bucket is None:
            bucket = _ChainBucket(
                rows=rows, num_steps=self.num_steps, device=self.device
            )
            self._buckets[rows] = bucket
        self._fill(bucket, plan_steps, first_tokens)
        fb0 = plan_steps[0][0]
        attn_backend = self.model_runner.attn_backend
        model = self.model_runner.model
        graph = torch.cuda.CUDAGraph()
        try:
            # Replay-prep half once for the whole chain (host-side metadata;
            # the in-graph fused rebuild handles the per-step device state).
            fb0.input_ids = bucket.input_ids
            fb0.positions = bucket.positions
            fb0.seq_lens = bucket.seq_lens
            fb0.req_pool_indices = bucket.req_rows
            is_mrope = fb0.mrope_positions is not None
            if is_mrope:
                fb0.mrope_positions = bucket.mrope_positions
            attn_backend.init_forward_metadata_out_graph(fb0, in_capture=True)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with forward_context(
                ForwardContext(attn_backend=attn_backend)
            ), torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    # Python-level globals the model forward branches on are
                    # baked in AT RECORD TIME; the chain runs right after the
                    # round's extend, so both must be reset to decode
                    # semantics exactly like the decode runner's run_once.
                    fb0.dp_local_start_pos = None
                    fb0.dp_local_num_tokens = None
                    set_dp_buffer_len(
                        fb0.global_dp_buffer_len,
                        rows,
                        (
                            fb0.dp_padding_mode.is_max_len()
                            if fb0.dp_padding_mode is not None
                            else False
                        ),
                        fb0.global_num_tokens_cpu,
                    )
                    set_is_extend_in_batch(False)
                    for s in range(self.num_steps):
                        fb0.out_cache_loc = bucket.out_locs[s]
                        attn_backend.init_forward_metadata_in_graph(fb0)
                        out = model.forward(bucket.input_ids, bucket.positions, fb0)
                        # model.forward returns the LogitsProcessorOutput
                        # itself (the runner wrapper is bypassed here).
                        logits = out.next_token_logits
                        next_tokens = logits.argmax(dim=-1)
                        bucket.out_tokens[s].copy_(next_tokens)
                        if s + 1 < self.num_steps:
                            bucket.input_ids.copy_(next_tokens)
                            bucket.seq_lens += 1
                            bucket.positions += 1
                            if is_mrope:
                                # Decode steps are text tokens: every mrope
                                # dim advances linearly, mirroring what
                                # init_new derives per step in the eager loop.
                                bucket.mrope_positions += 1
            torch.cuda.current_stream().wait_stream(stream)
            bucket.graph = graph
        except Exception:
            logger.exception(
                "chain-graph capture failed for rows=%d; falling back to the "
                "step-by-step chain permanently for this bucket",
                rows,
            )
            self._failed_rows.add(rows)
            self._buckets.pop(rows, None)
            return None
        # Stream capture RECORDS without executing: this replay is the
        # round's actual execution (side effects happen exactly once here).
        return self.replay(rows=rows, plan_steps=plan_steps, first_tokens=first_tokens)

    def replay(
        self, *, rows: int, plan_steps, first_tokens: torch.Tensor
    ) -> list[torch.Tensor]:
        bucket = self._buckets[rows]
        self._fill(bucket, plan_steps, first_tokens)
        fb0 = plan_steps[0][0]
        fb0.input_ids = bucket.input_ids
        fb0.positions = bucket.positions
        fb0.seq_lens = bucket.seq_lens
        fb0.req_pool_indices = bucket.req_rows
        if fb0.mrope_positions is not None:
            fb0.mrope_positions = bucket.mrope_positions
        self.model_runner.attn_backend.init_forward_metadata_out_graph(
            fb0, in_capture=False
        )
        bucket.graph.replay()
        return [bucket.out_tokens[s] for s in range(self.num_steps)]
