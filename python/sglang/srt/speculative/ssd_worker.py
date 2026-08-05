"""Target-side scheduler worker for same-GPU Saguaro/SSD decoding."""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.ssd_draft_client import (
    DraftCandidate,
    OutcomeCache,
    OutcomeKey,
    SSDDraftClient,
)

logger = logging.getLogger(__name__)


@dataclass
class _RequestState:
    pending: Optional[Future] = None
    pending_kind: Optional[str] = None
    outcome_key: Optional[OutcomeKey] = None
    rounds: int = 0
    hits: int = 0
    misses: int = 0
    jit_drafts: int = 0
    accepted_draft_tokens: int = 0
    draft_wait_ms: float = 0.0
    target_verify_ms: float = 0.0
    outcome_counts: List[int] = field(default_factory=list)
    cacheable_outcome_counts: List[int] = field(default_factory=list)
    recovery_rank_counts: Dict[int, int] = field(default_factory=dict)


class SSDWorker:
    """Verify external drafts while precomputing the next SSD outcome tree.

    This first implementation deliberately supports one greedy request at a
    time.  The target model remains the source of truth, so cache misses only
    add latency and never change generated tokens.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        del dp_rank, moe_ep_rank, attn_cp_rank, moe_dp_rank, nccl_port

        self.server_args = server_args
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.model_config = target_worker.model_runner.model_config
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        self.draft_length = server_args.speculative_num_steps
        self.draft_token_num = server_args.speculative_num_draft_tokens
        self.fan_outs = tuple(server_args.speculative_ssd_fan_outs)
        self.branch_budget = sum(self.fan_outs)
        self.request_timeout = server_args.speculative_ssd_request_timeout
        if self.draft_token_num != self.draft_length + 1:
            raise ValueError(
                "SSD requires speculative_num_draft_tokens == "
                "speculative_num_steps + 1."
            )

        self.client = SSDDraftClient(
            server_args.speculative_ssd_draft_server_url,
            timeout=self.request_timeout,
        )
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ssd-draft"
        )
        self.states: Dict[str, _RequestState] = {}
        self._init_verify_tensors()

        logger.info(
            "Initialized SSD worker: draft_server=%s K=%d fan_outs=%s "
            "branch_budget=%d verify_tokens=%d",
            server_args.speculative_ssd_draft_server_url,
            self.draft_length,
            self.fan_outs,
            self.branch_budget,
            self.draft_token_num,
        )

    def _new_request_state(self, *, jit_drafts: int = 0) -> _RequestState:
        return _RequestState(
            jit_drafts=jit_drafts,
            outcome_counts=[0] * (self.draft_length + 1),
            cacheable_outcome_counts=[0] * (self.draft_length + 1),
        )

    def _init_verify_tensors(self) -> None:
        width = self.draft_token_num
        self.draft_tokens = torch.empty(
            (width,), dtype=torch.int64, device=self.device
        )
        self.positions = torch.empty((width,), dtype=torch.int64, device=self.device)
        self.retrieve_index = torch.empty(
            (1, width), dtype=torch.int64, device=self.device
        )
        self.retrieve_next_token = torch.empty(
            (1, width), dtype=torch.int64, device=self.device
        )
        self.retrieve_next_sibling = torch.empty(
            (1, width), dtype=torch.int64, device=self.device
        )
        self.linear_tree_mask = torch.tril(
            torch.ones((width, width), dtype=torch.bool, device=self.device)
        )

    def clear_cache_pool(self) -> None:
        for state in self.states.values():
            if state.pending is not None:
                state.pending.cancel()
        self.states.clear()

    @staticmethod
    def _canonical_prefix(batch: ScheduleBatch) -> List[int]:
        req = batch.reqs[0]
        return [*req.origin_input_ids, *req.output_ids]

    def _start_initial_draft(
        self, batch: ScheduleBatch, next_token_ids: torch.Tensor
    ) -> None:
        if batch.batch_size() != 1:
            raise ValueError("SSD currently supports batch size 1 only.")
        req = batch.reqs[0]
        next_token_id = int(next_token_ids.reshape(-1)[0].item())
        prefix = [*req.origin_input_ids, *req.output_ids, next_token_id]

        # A new extend replaces any completed request state.  This initial SSD
        # worker admits one active request even though KTransformers needs two
        # request-pool slots internally.
        self.clear_cache_pool()
        state = self._new_request_state(jit_drafts=1)
        state.pending = self.executor.submit(
            self.client.jit_draft, prefix, self.draft_length, self.fan_outs
        )
        state.pending_kind = "initial"
        self.states[req.rid] = state

    def _jit_draft(
        self, prefix: List[int], state: _RequestState
    ) -> DraftCandidate:
        state.jit_drafts += 1
        with torch.cuda.nvtx.range("ssd_jit_draft_wait"):
            return self.client.jit_draft(
                prefix, self.draft_length, self.fan_outs
            )

    def _resolve_draft(
        self, rid: str, prefix: List[int], state: _RequestState
    ) -> DraftCandidate:
        pending = state.pending
        pending_kind = state.pending_kind
        state.pending = None
        state.pending_kind = None

        if pending is None:
            state.misses += 1
            return self._jit_draft(prefix, state)

        begin = time.perf_counter()
        try:
            with torch.cuda.nvtx.range("ssd_wait_outcome_cache"):
                result: Union[DraftCandidate, OutcomeCache] = pending.result(
                    timeout=self.request_timeout + 5.0
                )
        except Exception:
            state.misses += 1
            logger.exception("SSD draft task failed; falling back to JIT drafting.")
            return self._jit_draft(prefix, state)

        wait_ms = (time.perf_counter() - begin) * 1e3
        if pending_kind == "initial":
            logger.debug("SSD initial draft ready after %.3f ms wait", wait_ms)
            if not isinstance(result, DraftCandidate):
                state.misses += 1
                logger.error(
                    "SSD received an invalid initial draft; using JIT drafting."
                )
                return self._jit_draft(prefix, state)
            return result

        if pending_kind != "outcomes" or not isinstance(result, dict):
            state.misses += 1
            logger.error("SSD received an invalid pending result; using JIT drafting.")
            return self._jit_draft(prefix, state)

        cached = result.get(state.outcome_key)
        if cached is None:
            state.misses += 1
            logger.debug(
                "SSD outcome-cache miss: key=%s wait_ms=%.3f",
                state.outcome_key,
                wait_ms,
            )
            return self._jit_draft(prefix, state)

        state.hits += 1
        logger.debug(
            "SSD outcome-cache hit: key=%s wait_ms=%.3f",
            state.outcome_key,
            wait_ms,
        )
        return cached

    def _prepare_verify(
        self,
        batch: ScheduleBatch,
        prefix: List[int],
        candidate: DraftCandidate,
    ) -> None:
        if len(candidate.tokens) != self.draft_length:
            raise ValueError(
                f"SSD expected {self.draft_length} draft tokens, "
                f"got {len(candidate.tokens)}."
            )
        prefix_kv_len = int(batch.seq_lens_cpu[0].item())
        if len(prefix) != prefix_kv_len + 1:
            raise RuntimeError(
                "SSD canonical prefix and target KV length diverged: "
                f"prefix={len(prefix)}, committed_input={prefix_kv_len}."
            )

        candidates = [prefix[-1], *candidate.tokens]
        self.draft_tokens.copy_(
            torch.tensor(candidates, dtype=torch.int64, device=self.device)
        )

        reconstruct_indices_from_tree_mask(
            self.linear_tree_mask.flatten(),
            batch.seq_lens,
            self.positions,
            self.retrieve_index,
            self.retrieve_next_token,
            self.retrieve_next_sibling,
            1,
            self.draft_token_num,
        )

        prefix_mask = torch.ones(
            (self.draft_token_num, prefix_kv_len),
            dtype=torch.bool,
            device=self.device,
        )
        custom_mask = torch.cat((prefix_mask, self.linear_tree_mask), dim=1).flatten()

        # NgramVerifyInput is the target-only tree verifier.  The tree topology
        # is linear here; SSD changes how its tokens are produced, not how the
        # target checks them.
        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = NgramVerifyInput(
            self.draft_tokens,
            custom_mask,
            self.positions,
            self.retrieve_index,
            self.retrieve_next_token,
            self.retrieve_next_sibling,
            self.draft_token_num,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

    def _validate_decode_batch(self, batch: ScheduleBatch) -> None:
        if batch.batch_size() != 1:
            raise ValueError("SSD currently supports exactly one running request.")
        if batch.has_grammar:
            raise ValueError("SSD grammar-constrained decoding is not implemented yet.")
        if not batch.sampling_info.is_all_greedy:
            raise ValueError("SSD currently supports greedy sampling only.")

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend():
            model_worker_batch = batch.get_model_worker_batch()
            result = self.target_worker.forward_batch_generation(model_worker_batch)
            if result.next_token_ids is None:
                raise RuntimeError(
                    "SSD requires a single, non-chunked prefill before decoding."
                )
            self._start_initial_draft(batch, result.next_token_ids)
            return result

        self._validate_decode_batch(batch)
        req = batch.reqs[0]
        prefix = self._canonical_prefix(batch)
        state = self.states.get(req.rid)
        if state is None:
            state = self._new_request_state()
            self.states[req.rid] = state

        draft_begin = time.perf_counter()
        candidate = self._resolve_draft(req.rid, prefix, state)
        state.draft_wait_ms += (time.perf_counter() - draft_begin) * 1e3

        # This is the central SSD overlap: enqueue all next-outcome draft work
        # before launching target verification on the separately partitioned
        # CUDA client.
        state.pending = self.executor.submit(
            self.client.build_outcome_cache,
            list(prefix),
            candidate,
            self.draft_length,
            self.fan_outs,
        )
        state.pending_kind = "outcomes"

        self._prepare_verify(batch, prefix, candidate)
        model_worker_batch = batch.get_model_worker_batch()
        target_begin = time.perf_counter()
        with torch.cuda.nvtx.range("ssd_target_verify"):
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )

        verify_input: NgramVerifyInput = model_worker_batch.spec_info
        logits_output, next_token_ids, num_accepted_tokens = verify_input.verify(
            batch, batch_result.logits_output, self.page_size
        )
        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, verify_input, logits_output)

        accepted_length = int(verify_input.accept_length[0].item())
        state.target_verify_ms += (time.perf_counter() - target_begin) * 1e3
        recovery_token = int(req.output_ids[-1])
        recovery_guesses = candidate.recovery_tokens[accepted_length]
        try:
            recovery_rank = recovery_guesses.index(recovery_token) + 1
        except ValueError:
            recovery_rank = 0
        state.outcome_counts[accepted_length] += 1
        if recovery_rank > 0:
            state.cacheable_outcome_counts[accepted_length] += 1
        state.recovery_rank_counts[recovery_rank] = (
            state.recovery_rank_counts.get(recovery_rank, 0) + 1
        )
        state.outcome_key = (accepted_length, recovery_token)
        state.rounds += 1
        state.accepted_draft_tokens += accepted_length
        if req.finished() or state.rounds % 20 == 0:
            logger.info(
                "SSD request=%s finished=%s rounds=%d hits=%d misses=%d "
                "jit=%d accepted_draft=%d draft_wait_ms=%.3f target_verify_ms=%.3f",
                req.rid,
                req.finished(),
                state.rounds,
                state.hits,
                state.misses,
                state.jit_drafts,
                state.accepted_draft_tokens,
                state.draft_wait_ms,
                state.target_verify_ms,
            )
            logger.info(
                "SSD request=%s fan_outs=%s outcome_hist=%s "
                "cacheable_hist=%s recovery_rank_hist=%s",
                req.rid,
                self.fan_outs,
                state.outcome_counts,
                state.cacheable_outcome_counts,
                dict(sorted(state.recovery_rank_counts.items())),
            )

        batch.forward_mode = ForwardMode.DECODE
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=num_accepted_tokens,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
            accept_lens=verify_input.accept_length,
        )
