"""Decoupled verifier's spec worker: GPU-select a pre-enumerated chain, verify.

``VerifyWorker`` is the verify half of decoupled speculative decoding for the
STANDALONE (token-conditioned) drafter family. Each decode round it selects,
entirely on the GPU, the ``[guess, chain]`` unit of the landed enumeration
block that matches reality -- the previous round's accept case and bonus token
-- pads misses to a bonus-seeded fallback row, and runs the shared eagle
verify path (the exact code colocated STANDALONE verifies with, so committed
outputs match the colocated baseline token for token). Prefill is a plain
target extend. No draft model, no tree construction, no transport: blocks
arrive in the ``DecoupledEnumBuffer`` via the verifier IPC thread.

Select keys ride on ``batch.spec_info`` as an ``EnumSelectInput`` (the
decoupled analog of the colocated ``next_draft_input`` relay, produced here at
the end of every prefill / verify round and re-attached by the scheduler):

- ``bonus_tokens``   -- last committed token per request = this round's root.
- ``prev_accept_lens`` -- last round's accepted-draft count = the accept case.
- ``base_committed_lens`` -- the total committed sequence length (prompt +
  committed outputs) the in-flight enumeration must have been drafted from;
  compared against the landed stamp for fresh-vs-stale, so a late or reused
  block can never be consumed (only ever costs a fallback round).

The unified fallback (stale stamp / bonus miss / never-landed seat) replaces
the selected unit with ``[bonus, bonus, ...]``: the root is the real bonus, so
the row verifies as a plain 1-token decode; the junk tail is safe because
verify only ever accepts tokens the target model itself predicts.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.environ import envs
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_verify_worker import BaseVerifyWorker
from sglang.srt.speculative.decoupled_enum_buffer import DecoupledEnumBuffer
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import default_tree_mask_mode
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    run_eagle_verify,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import get_plan_stream
from sglang.srt.utils.nvtx_utils import profile_range

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.utils import GenerationBatchResult

logger = logging.getLogger(__name__)


@dataclass
class EnumSelectInput(EagleDraftInput):
    """Per-round GPU select keys for the enumeration buffer.

    Subclasses the (grandfathered dataclass) ``EagleDraftInput`` so generic
    spec_info plumbing (SpecInput type checks, idle handling) keeps working;
    ``bonus_tokens`` is inherited. ``filter_batch`` / ``merge_batch`` are
    overridden outright: the parent slices draft-model fields (topk_p /
    hidden_states) this input never carries.
    """

    # [bs] int64: last round's accepted-draft count (0..K) = the accept case.
    prev_accept_lens: Optional[torch.Tensor] = None
    # [bs] int64: expected enumeration-base stamp (total committed seq len).
    base_committed_lens: Optional[torch.Tensor] = None

    def filter_batch(
        self,
        new_indices: torch.Tensor,
        has_been_filtered: bool = True,
        new_indices_cpu=None,
    ):
        # Overlap contract (mirrors EagleDraftInput.filter_batch): survivors'
        # future_indices are re-gathered from the FutureMap next launch, so
        # they slice by position regardless of the has_been_filtered prefix
        # convention below. Unlike the parent we do NOT early-return: the
        # accept-case / stamp keys live only on this object.
        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
        if has_been_filtered:
            # The verify path already dropped finished rows; keep the prefix.
            keep = len(new_indices)
            self.bonus_tokens = self.bonus_tokens[:keep]
            self.prev_accept_lens = self.prev_accept_lens[:keep]
            self.base_committed_lens = self.base_committed_lens[:keep]
        else:
            self.bonus_tokens = self.bonus_tokens[new_indices]
            self.prev_accept_lens = self.prev_accept_lens[new_indices]
            self.base_committed_lens = self.base_committed_lens[new_indices]

    def merge_batch(self, spec_info: EnumSelectInput):
        if self.future_indices is not None and spec_info.future_indices is not None:
            self.future_indices = torch.cat(
                [self.future_indices, spec_info.future_indices]
            )
        self.bonus_tokens = torch.cat(
            [self.bonus_tokens, spec_info.bonus_tokens], axis=0
        )
        self.prev_accept_lens = torch.cat(
            [self.prev_accept_lens, spec_info.prev_accept_lens], axis=0
        )
        self.base_committed_lens = torch.cat(
            [self.base_committed_lens, spec_info.base_committed_lens], axis=0
        )


def select_enum_units(
    rows: torch.Tensor,
    stamps: torch.Tensor,
    *,
    bonus_tokens: torch.Tensor,
    prev_accept_lens: torch.Tensor,
    base_committed_lens: torch.Tensor,
    num_cases: int,
    fanout: int,
    unit_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure GPU select: (selected [bs, unit_width], hits [bs] bool). No host sync.

    ``rows`` / ``stamps`` are the seat-gathered enumeration generations
    ([bs, gen_count, num_cases * fanout * unit_width] and [bs, gen_count]):
    each seat keeps its two newest stamped blocks, because the block serving
    THIS round (enumerated two commits back) coexists with the one the last
    commit already pushed. Three steps: pick the generation whose stamp equals
    the expected base (fresh), then the (accept_case, bonus_guess) unit within
    it -- a hit iff fresh AND the real bonus is among the case's F guesses
    (unit element 0). Miss rows become the bonus-seeded fallback
    ``[bonus, bonus, ...]``: the root is the real bonus, and the junk tail is
    safe because verify only accepts target-agreeing tokens -- the row
    degrades to a plain 1-token decode.
    """
    bs, gen_count = stamps.shape
    units = rows.view(bs, gen_count, num_cases, fanout, unit_width)

    bonus_tokens = bonus_tokens.to(torch.int64)
    gen_matches = stamps.eq(base_committed_lens.unsqueeze(1))  # [bs, gen_count]
    fresh = gen_matches.any(dim=1)
    # First matching generation; 0 (overwritten by the fallback) when none.
    gen_indices = gen_matches.to(torch.int64).argmax(dim=1)
    # Clamp guards a protocol bug from turning into a device-side OOB; a wrong
    # case then simply fails the guess match and falls back.
    cases = prev_accept_lens.clamp(min=0, max=num_cases - 1)

    batch_arange = torch.arange(bs, device=rows.device)
    case_units = units[batch_arange, gen_indices, cases]  # [bs, F, unit_width]
    guesses = case_units[:, :, 0]  # [bs, F]
    guess_matches = guesses.eq(bonus_tokens.unsqueeze(1))  # [bs, F]
    hits = fresh & guess_matches.any(dim=1)
    # First matching guess; 0 (overwritten by the fallback) when none.
    guess_indices = guess_matches.to(torch.int64).argmax(dim=1)
    selected = case_units[batch_arange, guess_indices]  # [bs, unit_width]

    fallback_units = bonus_tokens.unsqueeze(1).expand(bs, unit_width)
    selected = torch.where(hits.unsqueeze(1), selected, fallback_units)
    return selected, hits


class VerifyWorker(BaseVerifyWorker):
    """Target-side verify worker for the decoupled STANDALONE drafter."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker

        # Role validation pinned topk to 1 (linear chains only) and
        # num_draft_tokens to num_steps + 1.
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_fanout = server_args.speculative_fanout
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        assert self.topk == 1 and (
            self.speculative_num_draft_tokens == self.speculative_num_steps + 1
        ), "decoupled verify requires topk == 1 chains (role validator pins this)"

        # Sizes the scheduler's per-decode KV over-alloc, exactly as the
        # colocated standalone worker does.
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        self.tree_mask_mode = default_tree_mask_mode()
        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

        # topk=1 chain constants for build_eagle_verify_input (runtime-invariant;
        # same construction as EagleDraftWorker._rebuild_topk1_chain_buffers).
        num_steps = self.speculative_num_steps
        decode_max_bs = (
            server_args.cuda_graph_config.decode.max_bs
            if server_args.cuda_graph_config is not None
            else None
        )
        max_bs = max(decode_max_bs or 0, server_args.max_running_requests or 0, 1)
        parent_width = num_steps if num_steps > 1 else 0
        self._chain_parents = torch.arange(
            -1, parent_width - 1, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)
        self._chain_score_indices = torch.arange(
            num_steps, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)

        # TP > 1: rank 0 owns the enum plane (buffer contents, C6 gate,
        # select) because gate outcomes follow wall-clock block arrival and
        # must not be decided per replicated rank; the selected units are
        # broadcast so every rank builds the identical verify input
        # (_select_units_tp_safe). Manager injection (select_gate /
        # commit_relay) only happens on rank 0 by construction.
        self._tp_rank = get_tensor_model_parallel_rank()
        self._tp_size = get_tensor_model_parallel_world_size()

        # The GPU landing buffer is sized off req_to_token, which exists only
        # after the scheduler allocates memory pools (alloc_memory_pool below).
        self.enum_buffer: Optional[DecoupledEnumBuffer] = None

        # Select outcomes per decode round ([bs] bool, GPU), FIFO: the manager
        # consumes them for hit / fallback accounting. A queue (not a single
        # slot) because the overlap scheduler defers result processing by one
        # launch, so two rounds' outcomes can be pending at once.
        # Verify prep-ahead (block-independent half before the C6 gate wait):
        # per-bs placeholder unit buffers; the select overwrites them in place.
        self._verify_prep_ahead = envs.SGLANG_ENABLE_DECOUPLED_VERIFY_PREP_AHEAD.get()
        self._placeholder_units: dict[int, torch.Tensor] = {}
        # Post-gate select as one captured graph (TP1 + prep-ahead only);
        # built in alloc_memory_pool once the enum buffer exists. None = the
        # eager select path.
        self._select_graph = None
        # (pinned_hits, ready_event) pairs: the D2H is staged pinned+async
        # at PRODUCTION so the deferred accounting hook never issues a
        # pageable copy -- a pageable D2H waits on the WHOLE device, which
        # under the stream gate means stalling the scheduler thread on the
        # gated forward stream.
        self.select_hits_queue: deque[tuple[torch.Tensor, torch.cuda.Event]] = deque()
        # C6 launch gate, injected by DecoupledVerifyManager: called with the
        # batch at decode-launch, before the select gather, to (boundedly)
        # wait for the enumeration blocks the select is about to read.
        self.select_gate: Optional[Callable[[ScheduleBatch], None]] = None
        # Commit relay, injected by DecoupledVerifyManager: hands each decode
        # round's (batch, result) to the IPC thread at launch; the thread
        # builds and sends the VerifyCommits once the result's copy_done event
        # fires -- forward end + copy, independent of the scheduler thread.
        self.commit_relay: Optional[Callable[[ScheduleBatch, Any], None]] = None

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.enum_buffer = DecoupledEnumBuffer(
            device=self.device,
            req_to_token_pool=req_to_token_pool,
            num_steps=self.speculative_num_steps,
            fanout=self.speculative_fanout,
            verifier_rank=self.server_args.decoupled_spec_rank,
            enable_overlap=False,
        )
        if (
            envs.SGLANG_ENABLE_DECOUPLED_SELECT_GRAPH.get()
            and self._verify_prep_ahead
            # The select is rank-0's job under TP (only rank 0 holds landed
            # blocks); ranks > 0 keep the empty-buffer + broadcast path and
            # never build a runner.
            and self._tp_rank == 0
        ):
            from sglang.srt.speculative.decoupled_select_graph import (
                SelectGraphRunner,
            )

            self._select_graph = SelectGraphRunner(
                enum_buffer=self.enum_buffer,
                num_cases=self.speculative_num_steps + 1,
                fanout=self.speculative_fanout,
                unit_width=self.speculative_num_draft_tokens,
                device=self.device,
            )

    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_target_prefill(batch, on_publish)

        # C6 gate: bounded host wait for the blocks this round's select reads.
        # Under the overlap scheduler this runs while the previous round is
        # still executing on the GPU, hiding the drafter latency behind it.
        prepared_verify = None
        if (
            self._verify_prep_ahead
            and self.select_gate is not None
            and not batch.forward_mode.is_idle()
        ):
            # Block-independent half BEFORE the gate: at topk == 1 the tree
            # mask/positions never depend on token values, so the wait only
            # has to cover select + one in-place copy + launch.
            select_input = batch.spec_info
            verify_input, prepared_verify = self._prepare_verify_prelude(batch)
            self.select_gate(batch)
            selected = self._select_units_tp_safe(batch, select_input)
            verify_input.draft_token.copy_(selected.view(-1))
            batch.spec_info = verify_input
        else:
            if self.select_gate is not None and not batch.forward_mode.is_idle():
                self.select_gate(batch)
            verify_input = self._build_verify_input(batch)
            batch.spec_info = verify_input
        batch_output = self._verify(batch, prepared_verify=prepared_verify)
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)
        if self.commit_relay is not None and not batch.forward_mode.is_idle():
            self.commit_relay(batch, batch_output)
        return batch_output

    def _forward_target_prefill(
        self, batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        # Pure target extend; STANDALONE never captures hidden states.
        batch_output = self.target_worker.forward_batch_generation(
            batch, capture_hidden_mode=CaptureHiddenMode.NULL
        )
        # Spec_v2 convention: batch.seq_lens = length BEFORE this iter's tokens.
        batch_output.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)
        # Seed the first decode round's select keys: the sampled first token is
        # both the committed bonus and round 1's root; the virtual "round 0"
        # accepted 0 drafts; the first enumeration grows from prompt + that
        # token, whose total length is seq_lens + 1.
        batch_output.next_draft_input = EnumSelectInput(
            bonus_tokens=batch_output.next_token_ids.to(torch.int32),
            prev_accept_lens=torch.zeros_like(batch.seq_lens, dtype=torch.int64),
            base_committed_lens=(batch.seq_lens + 1).to(torch.int64),
        )
        return batch_output

    def _build_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                topk=self.topk,
                spec_steps=self.speculative_num_steps,
                num_verify_tokens=self.speculative_num_draft_tokens,
                device=self.device,
            )

        select_input: EnumSelectInput = batch.spec_info
        selected = self._select_units_tp_safe(batch, select_input)

        # The selected unit IS the verify row: [root(=real bonus), K drafts].
        draft_input = EagleDraftInput(
            bonus_tokens=selected[:, 0].to(torch.int32).contiguous()
        )
        return build_eagle_verify_input(
            batch,
            draft_input,
            self._chain_parents[: selected.shape[0]],
            self._chain_score_indices[: selected.shape[0]],
            selected[:, 1:].contiguous(),
            None,  # draft_probs: rejection sampling is out of scope
            target_worker=self.target_worker,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            tree_mask_mode=self.tree_mask_mode,
            device=self.device,
        )

    def _select_enum_units(
        self, batch: ScheduleBatch, select_input: EnumSelectInput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, stamps = self.enum_buffer.gather(batch.req_pool_indices)
        return select_enum_units(
            rows,
            stamps,
            bonus_tokens=select_input.bonus_tokens,
            prev_accept_lens=select_input.prev_accept_lens,
            base_committed_lens=select_input.base_committed_lens,
            num_cases=self.speculative_num_steps + 1,
            fanout=self.speculative_fanout,
            unit_width=self.speculative_num_draft_tokens,
        )

    def _select_units_graph(
        self, batch: ScheduleBatch, select_input: EnumSelectInput
    ) -> Optional[torch.Tensor]:
        """One captured-graph select round (TP1 fast path), or None to fall
        back to the eager select (runner off, or this bucket's capture
        failed)."""
        if self._select_graph is None:
            return None
        with profile_range("verifier.select_graph"):
            got = self._select_graph.run(
                req_pool_indices=batch.req_pool_indices,
                bonus_tokens=select_input.bonus_tokens,
                prev_accept_lens=select_input.prev_accept_lens,
                base_committed_lens=select_input.base_committed_lens,
            )
        if got is None:
            return None
        selected, hits = got
        if envs.SGLANG_DEBUG_DECOUPLED_SELECT_GRAPH_CHECK.get():
            # The select is a pure function of (buffer, keys): re-running it
            # eagerly on the same inputs is side-effect-free, so the graph
            # can be checked in-process, per round.
            ref_selected, ref_hits = self._select_enum_units(batch, select_input)
            if not torch.equal(ref_selected, selected) or not torch.equal(
                ref_hits, hits
            ):
                logger.warning(
                    "select-graph mismatch: graph hits=%s eager hits=%s "
                    "graph sel=%s eager sel=%s",
                    hits.tolist(),
                    ref_hits.tolist(),
                    selected.tolist(),
                    ref_selected.tolist(),
                )
        # The graph's hits output is a static buffer the next replay
        # overwrites; the accounting hook consumes deferred, so queue a copy.
        self._queue_select_hits(hits)
        return selected

    def _queue_select_hits(self, hits: torch.Tensor) -> None:
        """Stage the round's hit flags for the deferred accounting hook:
        pinned + async at production, so the consumer reads host memory and
        never issues a pageable D2H (device-wide wait -- deadly under the
        stream gate)."""
        pinned = torch.empty_like(hits, device="cpu", pin_memory=True)
        pinned.copy_(hits, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record()
        self.select_hits_queue.append((pinned, ready))

    def _select_units_tp_safe(
        self, batch: ScheduleBatch, select_input: EnumSelectInput
    ) -> torch.Tensor:
        """Select on TP rank 0, broadcast the winning units to every rank.

        Only rank 0 holds landed blocks (the manager and its IPC thread exist
        there alone) and only rank 0 ran the C6 gate; replicating the select
        per rank would fork on arrival timing. The broadcast doubles as the
        barrier that holds ranks > 0 while rank 0 finishes its bounded gate
        wait. select_hits_queue is likewise rank-0-only (its consumer, the
        manager's accounting, lives there).
        """
        if self._tp_size == 1:
            selected = self._select_units_graph(batch, select_input)
            if selected is not None:
                return selected
            with profile_range("verifier.select_units"):
                selected, hits = self._select_enum_units(batch, select_input)
            self._queue_select_hits(hits)
            return selected
        if self._tp_rank == 0:
            # The graph replaces only rank 0's COMPUTE; the broadcast below
            # stays -- ranks > 0 are parked at the collective, so an early
            # return here would hang them.
            selected = self._select_units_graph(batch, select_input)
            if selected is None:
                selected, hits = self._select_enum_units(batch, select_input)
                self._queue_select_hits(hits)
        else:
            selected = torch.empty(
                (len(batch.reqs), self.speculative_num_draft_tokens),
                dtype=torch.int64,
                device=self.device,
            )
        return get_tp_group().broadcast(selected, src=0)

    def _prepare_verify_prelude(self, batch: ScheduleBatch):
        """Pre-gate half of the verify round: build the verify input on a
        placeholder unit buffer (zeros are valid token ids; the select
        overwrites the whole row in place) and run eagle_prepare_for_verify.
        Returns (verify_input, (verify_forward_batch, can_run_cuda_graph))."""
        from sglang.srt.speculative.eagle_utils import eagle_prepare_for_verify

        bs = len(batch.reqs)
        width = self.speculative_num_draft_tokens
        placeholder = self._placeholder_units.get(bs)
        if placeholder is None:
            placeholder = torch.zeros(
                (bs, width), dtype=torch.int64, device=self.device
            )
            self._placeholder_units[bs] = placeholder
        draft_input = EagleDraftInput(
            bonus_tokens=placeholder[:, 0].to(torch.int32).contiguous()
        )
        verify_input = build_eagle_verify_input(
            batch,
            draft_input,
            self._chain_parents[:bs],
            self._chain_score_indices[:bs],
            placeholder[:, 1:].contiguous(),
            None,
            target_worker=self.target_worker,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=width,
            tree_mask_mode=self.tree_mask_mode,
            device=self.device,
        )
        batch.spec_info = verify_input
        with self.plan_stream_ctx:
            prepared = eagle_prepare_for_verify(
                verify_input,
                self.req_to_token_pool,
                batch,
                self.target_worker,
            )
        return verify_input, prepared

    def _verify(
        self, batch: ScheduleBatch, *, prepared_verify=None
    ) -> GenerationBatchResult:
        with profile_range("verifier.verify_forward"):
            return self._verify_inner(batch, prepared_verify=prepared_verify)

    def _verify_inner(
        self, batch: ScheduleBatch, *, prepared_verify=None
    ) -> GenerationBatchResult:
        batch_output = run_eagle_verify(
            batch,
            target_worker=self.target_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            plan_stream=self.plan_stream,
            plan_stream_ctx=self.plan_stream_ctx,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,  # identity at topk == 1
            prepared_verify=prepared_verify,
        )
        # Next round's select keys, all GPU-resident (no host sync):
        # - the new bonus is the next root;
        # - accepted drafts (accept_lens includes the bonus) name the case;
        # - the next enumeration grows from this round's committed prefix plus
        #   its root, i.e. entry seq_lens + 1 in total-length terms.
        next_bonus_tokens = batch_output.next_draft_input.bonus_tokens
        batch_output.next_draft_input = EnumSelectInput(
            bonus_tokens=next_bonus_tokens,
            prev_accept_lens=(batch_output.accept_lens - 1).to(torch.int64),
            base_committed_lens=(batch.seq_lens + 1).to(torch.int64),
        )
        return batch_output
