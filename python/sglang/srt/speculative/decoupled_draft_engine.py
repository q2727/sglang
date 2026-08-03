"""Enumeration draft engine: the decoupled drafter's core compute.

Runs the draft model directly through ``ScheduleBatch`` + ``ModelRunner``
(the bench_one_batch harness pattern) -- no user requests, no admission, no
radix cache. Per request it keeps only the committed prefix's KV (token slots
it owns); every round is a re-extension from that prefix:

1. **advance** (extend): compute KV for the newly committed tokens; the last
   position's logits are enumeration node 0.
2. **backbone** (extend + K-1 decodes on scratch rows): greedily draft
   c_1..c_K; the logits at node ``a`` (after c_1..c_a) provide the top-F
   bonus guesses g_{a, 0..F-1} for accept case ``a`` (c_{a+1} == g_{a,0}).
3. **branches** (one batch of (K+1) x F scratch rows, extend + K-1 decodes):
   chain(a, f) = K tokens drafted after prefix + c_1..c_a + g_{a,f}. Nested
   prefixes are shared read-only, never recomputed.
4. **pack**: unit(a, f) = [g_{a,f}, chain_1..chain_K]; the block's stamp is
   the total committed length the tree grew from.

Paged KV (page_size P > 1) changes only WHERE scratch KV lives, under one
rule: rows share only the seat's FULL pages; everything from the round's
page-floor anchor onward is per-row PRIVATE, in engine-owned arena pages.
Sharing a partial page is unsound at P > 1 because a page-table entry maps P
consecutive logical positions to ONE physical page: the span past the
committed prefix mixes positions where all rows agree (delta, backbone c's)
with positions where they diverge (guesses, chains), and rows writing
different tokens through a shared page entry clobber each other. Each private
region is anchored at a page floor, so a private slot's in-page offset equals
its logical position's offset -- the invariant every paged attention backend
derives its page table from, and what lets ``alloc_extend`` / ``alloc_decode``
continue a private page in place. The head of a row's private region is
seeded by a batched K/V copy (COW): glue rows take the seat's boundary tail
before their extend, branch rows take boundary tail + delta + case backbone
from the seat / their glue row after it -- rows x O(P) tokens, two batched
copies per fast round, trivially cheap next to a forward. Frees are
page-granular: committed truncation (prerun rollback, close) releases only
pages fully past the cut, and the round-end scratch free hands back only the
round's THROWAWAY pages -- the allocator frees whole pages, and extend/decode
continuation slots land inside pages the engine's arena still owns. Which
pages those are is bookkept on the HOST as the round allocates (a page comes
off the global free list exactly when the position being filled starts one;
see ``_track_scratch_slots``), never recomputed from the slot ids at the
round tail: reading device-resident slot ids there (unique / boolean mask)
drains the whole round's GPU chain before the CPU may start the next round.
A steady-state paged round hands back nothing at all. At P == 1 the page
floor equals the committed length, boundary tails are always empty, and
every path degenerates to the original full slot-id sharing with zero
copies.

All scratch state (rows + KV slots written for backbone / branch tokens) is
freed at the end of the round: a wrong branch is never selected, and the next
commit re-extends from the committed prefix (keep-winning-branch KV is a
listed future optimization).

Fast rounds can replay the fused extend as ONE captured DRAFT_EXTEND_V2 CUDA
graph (``SGLANG_ENABLE_DECOUPLED_EXTEND_GRAPH``): every fused row is padded to
the static width W = 2K + 1 with repeats of its last real token, the
full-attention plane runs the uniform-W replay contract, and hybrid (GDN)
models feed TRUE per-row lengths through a graph-static device buffer so the
recurrent plane never scans a pad -- see ``_fused_extend_graph_forward`` and
``GDNAttnBackend._forward_draft_extend_v2`` for the two planes' contracts. A
miss round's seat advance replays that same graph in its no-glue degenerate
shape (rows = seats; see ``_advance_graph_forward``). The eager path remains
the mandatory per-round fallback for both.

A miss round (the commit fell outside the last block) collapses to case 0:
the verifier's select missed the same block the same way and falls back, so
the next commit can only be a single bonus -- only the F case-0 chains are
drafted and the dead cells are poisoned (see ``_case0_round``).

Hybrid (GDN / linear-attention) draft models add one twist: besides KV, every
row carries per-request recurrent state (conv + ssm) in a ``MambaPool`` slot
resolved through the pool-row mapping at forward time. The engine owns all of
its slots outright (``_MambaStateArena``): each seat holds a persistent slot
with the state after the committed prefix (only the advance ever runs on it),
carrier rows keep fixed slots for their whole lifetime (so the row->slot
mapping written once at alloc stays valid), and node states are forked
between slots with batched ``MambaPool.copy_from`` before each phase -- see
the per-phase fork comments. Everything is gated on ``self._hybrid``; pure-KV
models take exactly the old paths.
"""

from __future__ import annotations

import copy
import logging
import time
from array import array
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.decoupled_fused_ops import fused_guess_topk
from sglang.srt.speculative.decoupled_spec_io import DraftReqKey
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.nvtx_utils import profile_range

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class _ScratchTreeCache(SimpleNamespace):
    """Allocation-only tree cache stub (no prefix caching), bench-harness style."""

    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return True

    def evict(self, params: EvictParams):
        pass


class _ExtendCaseStaging(msgspec.Struct):
    """One accept case's device staging for the fused-extend graph replay,
    prebuilt in the idle window (everything here depends only on
    (base_lens, delta_len, carrier layout) -- never on token values)."""

    gather: torch.Tensor  # [rows * W] int64 source-gather indices
    out_cache_loc: torch.Tensor  # [rows * W] int64 write slots
    true_lens: torch.Tensor  # [rows] int32 device (GDN plane's true lengths)
    true_lens_host: list[int]
    node_gather: torch.Tensor  # [rows] int64 node-logit offsets
    node_offsets: list[int]


class _PrebuiltFastRound(msgspec.Struct):
    """A fast round's allocation + batch skeleton, built in the idle window.

    The delta is hypothesized at its K+1 maximum: allocation counts and the
    page-table writes are position-stable, so any actual case a consumes the
    per-seat prefix of ``slots`` and returns the leftover page HEADS to
    scratch (a page shared with absorbed prefix slots has its head before the
    leftover range and is never freed -- the ``_track_scratch_slots`` rule).
    Token values are filled post-commit via pinned staging; the placeholder
    input_ids in ``batch`` are never read on the graph path.
    """

    keys: tuple
    base_lens: list[int]
    batch: ScheduleBatch
    slots: torch.Tensor  # [bs * (K+1)] seat-major
    slot_positions: list[int]  # logical positions, aligned with ``slots``
    graph_round: _ExtendGraphRound
    scratch_batches: list
    scratch_slots: list
    scratch_kv_pages: list
    # Glue-head COW + glue mamba fork already ran at idle time: both are
    # case-independent (tail span = [page_floor(base), base); fork source =
    # the seat's post-absorb state, stable until the next round).
    glue_seeded: bool = False
    # bs == 1 only: per-case fused-extend staging, indexed by delta_len - 1.
    case_staging: Optional[list] = None
    # Recorded on the prebuild stream after the skeleton's GPU work; the
    # consuming round waits on it (record-then-wait) before touching any
    # prebuilt tensor.
    ready_event: Optional[torch.cuda.Event] = None


class _RoundProfiler:
    """Per-phase host-time accumulator for the enumeration round.

    Syncs the device at every mark so each phase's wall time is attributed
    exactly -- which also serializes host and GPU work. Numbers are for
    relative breakdown, not absolute round latency (debug only).

    Independently of that (and without any sync), ``stage()`` opens a
    ``profile_range`` so a trace shows named ``drafter.<stage>`` bands over the
    round's kernels, alongside the ``step[...]`` spans the model runner emits.
    ``profile_range`` is the engine-wide annotation entry point: it returns a
    shared no-op unless a torch profiler is active (or nvtx is enabled), so the
    round stays annotated unconditionally and pays one flag check when it isn't
    being profiled.
    """

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.round_ct = 0
        self.phase_ms: dict[str, float] = {}
        self._t_last = 0.0

    @staticmethod
    def stage(name: str):
        """Name a round stage for the trace (no device sync)."""
        return profile_range(f"drafter.{name.replace('-', '_')}")

    def start_round(self) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.round_ct += 1
        self._t_last = time.monotonic()

    def mark(self, phase: str) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize()
        now = time.monotonic()
        self.phase_ms[phase] = self.phase_ms.get(phase, 0.0) + 1000.0 * (
            now - self._t_last
        )
        self._t_last = now

    def summary(self) -> str:
        if self.round_ct == 0:
            return "no profiled rounds"
        parts = [
            f"{phase}={ms / self.round_ct:.2f}"
            for phase, ms in sorted(
                self.phase_ms.items(), key=lambda kv: kv[1], reverse=True
            )
        ]
        total = sum(self.phase_ms.values()) / self.round_ct
        return f"rounds={self.round_ct} total_ms={total:.2f} | " + " ".join(parts)


class _PinnedH2D:
    """Async host-to-device staging for the small index/length tensors a round builds.

    ``torch.tensor(values, device=cuda)`` -- and ``torch.tensor(values).to(cuda,
    non_blocking=True)``, whose ``non_blocking`` is a no-op on unpinned memory --
    both issue a SYNCHRONOUS pageable copy. Such a copy does not merely wait for
    its own bytes: it blocks the host until everything already queued on the
    stream has drained, so each one is a full pipeline barrier. Profiles put them
    at ~500us apiece, 4-5 per round, inside stages that are otherwise a single
    graph replay.

    Staging through pinned memory keeps the copy asynchronous. Buffers rotate so
    one is never rewritten while its copy may still be in flight, and the event
    recorded per slot makes that safe rather than merely likely -- a slot only
    comes back around after ``_RING`` further calls, by which point the wait is a
    no-op in steady state.
    """

    # A round stages ~12 of these (gather, seq_lens x K, true_lens, node_gather,
    # prefix_lens, scratch heads), so a ring of 64 puts a slot's reuse five
    # rounds out and the guard wait stays a no-op (32 still showed ~2 waits of
    # ~80us per round in traces).
    _RING = 64

    def __init__(self, *, device: str) -> None:
        self.device = device
        self.enabled = envs.SGLANG_ENABLE_DECOUPLED_PINNED_H2D.get()
        self._buffers: list[dict[torch.dtype, torch.Tensor]] = [
            {} for _ in range(self._RING)
        ]
        self._events = [torch.cuda.Event() for _ in range(self._RING)]
        self._in_flight = [False] * self._RING
        self._next = 0

    def to_device(self, values, *, dtype: torch.dtype) -> torch.Tensor:
        """Copy a host list (or CPU tensor) to the device without a barrier."""
        host_src = values if isinstance(values, torch.Tensor) else None
        n = host_src.numel() if host_src is not None else len(values)
        if n == 0:
            return torch.empty(0, dtype=dtype, device=self.device)
        if not self.enabled:
            return torch.as_tensor(values, dtype=dtype).to(self.device)

        slot = self._next
        self._next = (slot + 1) % self._RING
        if self._in_flight[slot]:
            self._events[slot].synchronize()

        staging = self._buffers[slot].get(dtype)
        if staging is None or staging.numel() < n:
            staging = torch.empty(max(n, 64), dtype=dtype, pin_memory=True)
            self._buffers[slot][dtype] = staging

        view = staging[:n]
        if host_src is None:
            host_src = torch.as_tensor(values, dtype=dtype)
        view.copy_(host_src)  # host-to-host, no device involvement
        out = view.to(self.device, non_blocking=True)
        self._events[slot].record()
        self._in_flight[slot] = True
        return out


class _DraftReqState:
    def __init__(self, *, req_pool_idx: int, device: torch.device) -> None:
        # The seat on the OWNING VERIFIER (echoed into every block row); this
        # engine's own scratch rows are unrelated and transient.
        self.req_pool_idx = req_pool_idx
        self.committed_tokens: list[int] = []
        # Slot ids of the committed prefix's KV in the drafter's pool
        # (device-resident: the round must never sync on slot bookkeeping).
        self.committed_slots = torch.empty((0,), dtype=torch.int64, device=device)
        # Last round's block, kept for the glue fast path: the winning unit's
        # chain IS the next round's backbone (greedy re-draft is a no-op).
        self.last_units_dev: Optional[torch.Tensor] = None  # [K+1, F, K+1]
        self.last_units_host: Optional[torch.Tensor] = None  # pinned mirror
        self.last_backbone_host: Optional[list[int]] = None  # c_1..c_K
        self.mirror_event: Optional[torch.cuda.Event] = None
        # Top-1 prerun bet: number of speculatively committed tokens (0 = no
        # active bet) + the pre-bet mirror snapshot for rollback.
        self.prerun_len = 0
        self.prerun_snapshot: Optional[tuple] = None
        # Hybrid models only (None on pure-KV): the seat's persistent mamba
        # slot, holding the recurrent state AFTER the committed prefix (node 0
        # post-advance). Only the advance extend ever runs on it; every other
        # phase works on forked copies. Shape [1], mirroring the framework's
        # per-request ``req.mamba_pool_idx`` convention.
        self.mamba_slot: Optional[torch.Tensor] = None
        # Pre-bet copy of the seat state: a prerun's advance contaminates the
        # seat slot with bet tokens, so the stash restores it on a miss (and
        # is simply dropped on a hit).
        self.prerun_mamba_stash: Optional[torch.Tensor] = None

    def pending_delta(self) -> list[int]:
        """Committed tokens whose KV has not been advanced yet."""
        return self.committed_tokens[self.committed_slots.numel() :]


class _CascadeMetadata(msgspec.Struct):
    """Shared-prefix cascade inputs for one branch-decode chain (fa3 decode
    consumes it via ForwardBatch.decoupled_cascade). Page tables/lens are
    int32 per the fa3 convention; tail state advances in place per step."""

    prefix_page_table: torch.Tensor  # [seats, max_prefix_len]
    prefix_lens: torch.Tensor  # [seats]
    tail_page_table: torch.Tensor  # [rows, 2K+2]
    tail_lens: torch.Tensor  # [rows]
    # arange(rows): the row of every per-step tail append, built once with the
    # round's metadata instead of once per decode step.
    row_indices: torch.Tensor  # [rows]


class _SlowRoundPages(msgspec.Struct):
    """Private-page staging for one slow subround at page_size > 1 (mirrors
    the mamba-slot staging): one transient page per backbone row (its
    boundary-tail COW head; extend/decode continuations grow it in place),
    and the carrier-lifetime pages for the glue/branch rows, taken up front
    so the branch extend can target them and the carrier build can adopt
    them (removed from the round's scratch list on donation -- the
    ``scratch_batches.remove`` idiom -- so an aborted subround can't leak
    them). ``*_flats`` are the pages expanded to flat slot ids (column q =
    in-page offset q % P; see ``_page_flat_slots``)."""

    backbone_pages: torch.Tensor  # [bs, 1]
    backbone_flats: torch.Tensor  # [bs, P]
    glue_pages: torch.Tensor  # [bs * K, pages_per_row]
    glue_flats: torch.Tensor  # [bs * K, pages_per_row * P]
    branch_pages: torch.Tensor  # [bs * (K+1) * F, pages_per_row]
    branch_flats: torch.Tensor  # [bs * (K+1) * F, pages_per_row * P]


class _FanoutVariant(msgspec.Struct):
    """Row selections + scatter templates for one per-case column budget.

    Branch rows keep their full-F pool layout (case-major, F rows per case);
    a variant selects the first ``budgets[c]`` rows of case c -- uniformly
    ``f_live`` everywhere for the adaptive-width path, or the skewed per-case
    budget (case K takes the full width, shallower cases less: their bonus is
    a rank-2+ candidate by construction). Two row numberings coexist:
    ``*_pool`` indexes the seat's full branch-row tensor, ``*_sel`` indexes
    the selected (packed) batch order.
    """

    f_live: int  # effective width this variant was built for (its cache key)
    budgets: list[int]  # live guess columns per accept case
    # [1, K+1, F] bool marking the columns PAST each case's budget, or None
    # when the budget is uniform (a plain top-f_live grid needs no masking).
    guess_dead_mask: Optional[torch.Tensor]
    sel_rows_pool: list[int]  # selected per-seat row offsets, full-F layout
    sel_rows_dev: torch.Tensor  # same, device tensor
    br_r_pool: torch.Tensor  # case-prefix scatter rows, full-F layout
    br_r_sel: torch.Tensor  # case-prefix scatter rows, selected numbering
    br_j: torch.Tensor  # case-prefix scatter cols (backbone slot j)
    # Accept case per case-prefix entry; the graph round's chain-source plane
    # (case c's prefix reads glue row c-1 == w_slots plane c).
    br_case: torch.Tensor
    comb_j: torch.Tensor  # glue triangle cols + br_j
    case_of_row: list[int]  # accept case per selected row
    case_of_row_dev: torch.Tensor  # same, device tensor (hybrid fork gather)


class _SeatCarrier:
    """Retained fast-path pool rows + scatter template for ONE seat.

    ``glue_rows``: K one-token extend rows; row g re-materializes backbone
    token c_{g+1} on top of committed + c_1..c_g. Prefixes are slot-shared
    ACROSS ROWS OF THE SAME FORWARD: per layer, the batched KV write precedes
    the attention read, and c_g's KV depends only on its own row, so row g+1
    reads row g's fresh KV exactly as a sequential chain would. (Fused-extend
    mode rides these same rows next to the seat row with delta-prepended
    extends -- see ``_fused_extend_forward``; roles, pool-row content
    maintenance, and mamba slot bindings are unchanged, only the fork-source
    timing moves.)

    ``branch_rows``: (K+1)*F persistent decode rows; each round only seq_lens
    and the pool-row tail entries move, then the K chain steps run as plain
    decode (cuda-graph replays).

    Pool-row content is maintained incrementally: rows carry the committed
    prefix up to ``synced_len``; the region past it is per-round scratch
    mapping (delta slots, then backbone slots) and is rewritten every round.
    Seats are independent so any hit subset of a batch can run the fast path
    (per-seat mixing); rows live until the seat closes.

    At page_size > 1 each row additionally owns arena pages for its whole
    lifetime (``*_private_slots`` are their flat slot ids): ``synced_len`` is
    then always page-aligned (only FULL seat pages are shared), and the
    private region past the round's anchor is re-bound to these pages every
    round -- the fixed page ownership is what lets rows write divergent
    tokens without touching any shared page.
    """

    def __init__(
        self,
        *,
        glue_rows: torch.Tensor,  # [K] device
        branch_rows: torch.Tensor,  # [(K+1)*F] device
        glue_reqs: list,
        branch_reqs: list,
        synced_len: int,
        glue_mamba_slots: Optional[torch.Tensor] = None,  # [K] device
        branch_mamba_slots: Optional[torch.Tensor] = None,  # [(K+1)*F] device
        glue_private_slots: Optional[torch.Tensor] = None,  # [K, W] device
        branch_private_slots: Optional[torch.Tensor] = None,  # [(K+1)*F, W]
        private_kv_pages: Optional[torch.Tensor] = None,  # [K+(K+1)*F, ppr]
    ) -> None:
        self.glue_rows = glue_rows
        self.branch_rows = branch_rows
        # Req stubs owning the pool rows (freed via ReqToTokenPool.free(req)).
        self.glue_reqs = glue_reqs
        self.branch_reqs = branch_reqs
        self.synced_len = synced_len
        # Hybrid models only (None on pure-KV): engine-owned mamba slots bound
        # 1:1 to the carrier rows for the carrier's whole lifetime. The pool's
        # row->slot mapping is written once (at row alloc) and never again --
        # the fixed binding is what keeps every later fast-round forward on
        # these rows reading the right recurrent state; per-round content is
        # refreshed by state FORKS into the slots, not by re-mapping. Glue row
        # g's slot holds node-(g+1) state after each glue forward.
        self.glue_mamba_slots = glue_mamba_slots
        self.branch_mamba_slots = branch_mamba_slots
        # page_size > 1 only (None otherwise): engine-arena pages owned by the
        # rows for the carrier's lifetime, pre-expanded to flat slot ids. The
        # concatenated view is ordered glue-then-branch to match ``all_rows``
        # (the per-round private page-table write indexes them in lockstep).
        self.glue_private_slots = glue_private_slots
        self.branch_private_slots = branch_private_slots
        self.private_kv_pages = private_kv_pages
        self.all_private_slots = (
            torch.cat([glue_private_slots, branch_private_slots])
            if glue_private_slots is not None
            else None
        )
        # All carrier rows sharing the committed prefix (delta broadcast).
        self.all_rows = torch.cat([glue_rows, branch_rows])
        # Combined scatter rows per effective fanout (values/cols come from
        # the engine's per-fanout templates); built lazily, dies with the seat.
        self._comb_rows_cache: dict[int, torch.Tensor] = {}
        # Branch-phase row selection per effective fanout (Req stubs + pool
        # rows). Round-INVARIANT: the variant's row selection depends only on
        # the width, so both the host list and the row gather are built once
        # per seat and width and only rebound each round.
        self._branch_sel_cache: dict[int, tuple[list, torch.Tensor]] = {}

    def comb_rows_for(
        self, *, f_live: int, tri_g: torch.Tensor, br_r_pool: torch.Tensor
    ) -> torch.Tensor:
        rows = self._comb_rows_cache.get(f_live)
        if rows is None:
            rows = torch.cat([self.glue_rows[tri_g], self.branch_rows[br_r_pool]])
            self._comb_rows_cache[f_live] = rows
        return rows

    def branch_sel_for(self, *, variant: _FanoutVariant) -> tuple[list, torch.Tensor]:
        """(Req stubs, pool rows) of this seat's selected branch rows."""
        sel = self._branch_sel_cache.get(variant.f_live)
        if sel is None:
            sel = (
                [self.branch_reqs[row] for row in variant.sel_rows_pool],
                self.branch_rows[variant.sel_rows_dev],
            )
            self._branch_sel_cache[variant.f_live] = sel
        return sel


class _MambaStateArena:
    """Engine-owned mamba slot free-list in front of ``MambaSlotAllocator``.

    Slots are pulled from the pool allocator lazily (grown on demand) and
    then recycled inside the engine forever: persistent takes (seat slots,
    carrier rows) return at seat close/evict, transient takes (slow-path
    backbone, prerun stash) at round end -- steady-state rounds never touch
    the allocator. Ids are virtual (same space as ``req.mamba_pool_idx``);
    translate before physical pool state ops. take/give_back never sync.
    """

    def __init__(self, *, allocator: MambaSlotAllocator, sizing_hint: str) -> None:
        self._allocator = allocator
        self._sizing_hint = sizing_hint
        self._free_slots: Optional[torch.Tensor] = None

    def take(self, num_slots: int) -> torch.Tensor:
        free = self._free_slots
        num_free = 0 if free is None else free.numel()
        if num_slots == 0:
            # Degenerate widths (e.g. K == 0) must not dereference an empty
            # free list.
            return torch.empty((0,), dtype=torch.int64, device=self._allocator.device)
        if num_free < num_slots:
            grown = self._allocator.alloc(num_slots - num_free)
            if grown is None:
                raise RuntimeError(
                    "drafter mamba state pool exhausted "
                    f"(want {num_slots - num_free} more slots, "
                    f"{self._allocator.available_size()} free in the pool "
                    f"allocator); {self._sizing_hint}"
                )
            free = grown if free is None else torch.cat([free, grown])
        taken = free[:num_slots]
        self._free_slots = free[num_slots:]
        return taken

    def give_back(self, slots: torch.Tensor) -> None:
        if slots.numel() == 0:
            return
        slots = slots.reshape(-1)
        self._free_slots = (
            slots if self._free_slots is None else torch.cat([self._free_slots, slots])
        )


class _KVPageArena:
    """Engine-owned KV page free-list in front of the paged allocator
    (page_size > 1 only) -- the mamba arena's twin for physical KV pages.

    Every page the engine binds by hand goes through here: carrier rows need
    PRIVATE pages whose slots the engine writes into page tables and
    out_cache_loc itself, and a graph round's seat pad tail needs throwaway
    junk pages with the same in-page-offset geometry. Recycling them inside
    the engine (rather than allocating and freeing per round) is also what
    keeps the round tail free of device reads: ``PagedTokenToKVPoolAllocator
    .free`` uniques its input on the device, which synchronizes. Ids are
    physical page ids (slot // page_size); take/give_back never sync.
    """

    def __init__(self, *, allocator, page_size: int, sizing_hint: str) -> None:
        self._allocator = allocator
        self._page_size = page_size
        self._sizing_hint = sizing_hint
        self._free_pages: Optional[torch.Tensor] = None

    def take(self, num_pages: int) -> torch.Tensor:
        pages = self.take_optional(num_pages)
        if pages is None:
            raise RuntimeError(
                "drafter KV page arena exhausted "
                f"(want {num_pages} pages, "
                f"{self._allocator.available_size()} tokens free in the "
                f"pool allocator); {self._sizing_hint}"
            )
        return pages

    def take_optional(self, num_pages: int) -> Optional[torch.Tensor]:
        """``take`` for callers with a fallback path (None on exhaustion)."""
        free = self._free_pages
        num_free = 0 if free is None else free.numel()
        if num_pages == 0:
            # Degenerate widths (e.g. K == 0) must not dereference an empty
            # free list.
            return torch.empty((0,), dtype=torch.int64, device=self._allocator.device)
        if num_free < num_pages:
            grow = num_pages - num_free
            slots = self._allocator.alloc(grow * self._page_size)
            if slots is None:
                return None
            # alloc is page-aligned: row k of the [grow, P] view is one page.
            grown = slots.view(grow, self._page_size)[:, 0] // self._page_size
            free = grown if free is None else torch.cat([free, grown])
        taken = free[:num_pages]
        self._free_pages = free[num_pages:]
        return taken

    def give_back(self, pages: torch.Tensor) -> None:
        if pages.numel() == 0:
            return
        pages = pages.reshape(-1)
        self._free_pages = (
            pages if self._free_pages is None else torch.cat([self._free_pages, pages])
        )


class _ExtendGraphRunnerFacade:
    """Read-through view of the drafter's ModelRunner for the fused-extend
    graph stack (runner + dedicated attention backends).

    Two surfaces are overridden; everything else delegates to the real runner:

    - ``server_args``: the drafter role hook nulls
      ``speculative_num_draft_tokens`` (a spec-shaped ModelRunner would
      corrupt the engine's 1-token decode graphs), but the draft-extend
      runner and its backends size their fixed-width buffers from exactly
      that field -- they get a patched copy carrying W and the fused-row
      capture buckets instead.
    - ``graph_shared_output``: the process-shared logits buffer is sized
      max_decode_bs x 1 rows (the drafter is spec-free), while the extend
      graph anchors max_rows x W full-width logits rows; a private buffer
      both fits that and keeps decode replays from ever clobbering
      not-yet-consumed extend logits.

    Attribute WRITES land on the facade (e.g. the runner's replay publishes
    ``war_fastpath_read_done_event`` onto its model runner); that is
    intentional -- the drafter's dedicated event loop never consumes those
    scheduler-side rails.
    """

    def __init__(self, *, model_runner, server_args, graph_shared_output) -> None:
        self._model_runner = model_runner
        self.server_args = server_args
        self.graph_shared_output = graph_shared_output

    def __getattr__(self, name: str):
        return getattr(self._model_runner, name)


class _DraftExtendWorkerShim:
    """Duck-typed ``EagleDraftWorker`` surface for the draft-extend graph
    runner -- exactly the fields its ``__init__`` / capture read (the
    attention-unittest harness precedent,
    test/kits/attention_unittest/runner_modes/speculative_draft_extend_runner).
    STANDALONE keeps the runner's target-hidden-state plumbing off: the enum
    drafter feeds token ids only."""

    def __init__(
        self,
        *,
        draft_runner,
        draft_extend_attn_backend,
        num_steps: int,
        num_draft_tokens: int,
    ) -> None:
        from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

        self.draft_runner = draft_runner
        self.target_worker = SimpleNamespace(model_runner=draft_runner)
        self.draft_extend_attn_backend = draft_extend_attn_backend
        self.topk = 1
        self.speculative_num_steps = num_steps
        self.speculative_num_draft_tokens = num_draft_tokens
        self.server_args = draft_runner.server_args
        self.model_config = draft_runner.model_config
        self.speculative_algorithm = SpeculativeAlgorithm.STANDALONE
        self.eagle_use_aux_hidden_state = False
        self.hot_token_id = None
        EagleDraftWorker._init_dsa_index_share_state(self)


class _ExtendGraphRound(msgspec.Struct):
    """Per-round staging for the fused-extend graph path (fast rounds' fused
    shape and miss rounds' advance-only degenerate shape alike).

    page_size == 1: ``w_slots`` [bs, rows_per_seat, W] gives every graph row a
    private W-slot write window (plane 0 pads the seat row, planes 1..K -- a
    fast round only -- are the glue rows' whole windows). The eager
    duplicate-write trick is off on this path: pad tokens differ across rows,
    so the identical-value argument that makes shared-slot writes benign no
    longer covers the whole window.

    page_size > 1: glue rows already own private arena pages wide enough for
    W (span_max = P - 1 + W by construction); only the seat row's pad tail
    needs ``seat_pad_flats`` [bs, pad_span] -- whole junk pages borrowed from
    the engine's page arena for the round, expanded to flat slots (column q
    holds in-page offset q % P, preserving the offset == logical-position
    invariant paged tables are derived from).
    """

    w_slots: Optional[torch.Tensor]
    seat_pad_flats: Optional[torch.Tensor]


class EnumDraftEngine:
    """Per-request committed KV + one enumeration tree per commit round."""

    def __init__(
        self,
        *,
        model_runner: ModelRunner,
        num_steps: int,
        fanout: int,
        enable_glue_fast_path: bool = True,
        enable_extend_graph: bool = True,
        exclude_dead_guess: Optional[bool] = None,
    ) -> None:
        self.model_runner = model_runner
        self.num_steps = int(num_steps)
        self.fanout = int(fanout)
        # Live enumeration width for the next rounds (<= fanout); the manager's
        # feedback controller lowers it when rounds threaten the verifier's
        # enum-wait budget. Unused guess columns ship poisoned, so the block
        # shape (and the verifier) never changes.
        self.effective_fanout = self.fanout
        self.unit_width = self.num_steps + 1
        self.device = model_runner.device
        # Hybrid (GDN) detection + the engine-owned state-slot arena. Both
        # guards are launch-config errors on the DRAFTER process, caught here
        # rather than as silent corruption mid-round: extra-buffer tracking
        # would allocate radix ping-pong slots per scratch req (leaked -- the
        # engine frees rows via ReqToTokenPool.free, never free_mamba_cache),
        # and the linear-ReplaySSM ring breaks the copy_from fork invariant
        # (sources must be fully flushed checkpoints; the engine forks slots
        # that just ran decode steps).
        self._hybrid = isinstance(model_runner.req_to_token_pool, HybridReqToTokenPool)
        self._mamba_arena: Optional[_MambaStateArena] = None
        if self._hybrid:
            hybrid_pool = model_runner.req_to_token_pool
            if hybrid_pool.enable_mamba_extra_buffer:
                raise RuntimeError(
                    "the decoupled drafter engine requires the plain mamba pool; "
                    "launch the drafter without mamba radix tracking "
                    "(--enable-mamba-extra-buffer)"
                )
            if hybrid_pool.mamba_pool.replayssm_write_pos is not None:
                raise RuntimeError(
                    "the decoupled drafter engine forks mamba state with "
                    "MambaPool.copy_from, which requires fully flushed "
                    "checkpoints; launch the drafter without "
                    "--enable-linear-replayssm"
                )
            # Per-seat worst case: 1 seat + K glue + (K+1)*F branch rows plus
            # 2 transients (slow-path backbone, prerun stash).
            slots_per_seat = 3 + self.num_steps + (self.num_steps + 1) * self.fanout
            self._mamba_arena = _MambaStateArena(
                allocator=hybrid_pool.mamba_allocator,
                sizing_hint=(
                    f"size --max-mamba-cache-size to at least "
                    f"seats x {slots_per_seat} (1 seat + K glue + (K+1)*F "
                    f"branch + 2 transient; K={self.num_steps}, "
                    f"F={self.fanout})"
                ),
            )
        # Paged-KV geometry (init-static; see the module docstring's sharing
        # rule). All paged behavior gates on _paged -- never on a backend
        # name -- and P == 1 keeps every original path bit-identical.
        self._page_size = int(model_runner.token_to_kv_pool_allocator.page_size)
        self._paged = self._page_size > 1
        self._cow_kv_pool: Optional[MHATokenToKVPool] = None
        self._kv_page_arena: Optional[_KVPageArena] = None
        self._pages_per_carrier_row = 0
        if self._paged:
            self._cow_kv_pool = self._resolve_cow_kv_pool(model_runner=model_runner)
            # Worst private span per carrier row: a boundary tail (< P) + a
            # fast-round delta (<= K + 1: the deepest accept case, or a
            # prerun's full bet) + the K backbone/chain positions the blanket
            # page-table write binds to arena slots each round. Decode-step
            # page crossings past this span take throwaway allocator pages,
            # so the arena never needs to cover them.
            span_max = (self._page_size - 1) + (self.num_steps + 1) + self.num_steps
            self._pages_per_carrier_row = (
                span_max + self._page_size - 1
            ) // self._page_size
            # Carrier rows + the slow round's backbone transient + a graph
            # round's per-seat pad tail, which spans a boundary tail plus the
            # static W = 2K + 1 -- the same worst case, so the same page count
            # (see _extend_graph_pad_span).
            pages_per_seat = (
                self.num_steps + (self.num_steps + 1) * self.fanout + 1
            ) * self._pages_per_carrier_row + 1
            self._kv_page_arena = _KVPageArena(
                allocator=model_runner.token_to_kv_pool_allocator,
                page_size=self._page_size,
                sizing_hint=(
                    f"budget ~seats x {pages_per_seat} pages of KV headroom "
                    f"for the engine page arena (K={self.num_steps}, "
                    f"F={self.fanout}, page_size={self._page_size})"
                ),
            )
        self._tree_cache = _ScratchTreeCache(
            page_size=model_runner.server_args.page_size,
            device=model_runner.device,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        )
        # Greedy, never finishing on its own; lifecycle is DraftSync/DraftClose.
        self._sampling_params = SamplingParams(temperature=0, max_new_tokens=1 << 30)
        self._states: dict[DraftReqKey, _DraftReqState] = {}
        self._seat_carriers: dict[DraftReqKey, _SeatCarrier] = {}
        self._enable_glue_fast_path = bool(enable_glue_fast_path)
        self.prerun_hit_ct = 0
        self.prerun_miss_ct = 0
        # Shared-prefix cascade for branch decodes (fa3): only past this
        # prefix length does dedup beat the L2-served re-reads. 0 = off.
        self._cascade_min_prefix_len = (
            envs.SGLANG_DECOUPLED_CASCADE_MIN_PREFIX_LEN.get()
        )
        # Static scatter templates shared by every seat carrier: glue triangle
        # (row g needs c_1..c_g's slots at [L:L+g] INCLUSIVE -- fa3 extend
        # reads the current token's own K/V through the page table too) +
        # branch case prefixes (row (c, f) needs c_1..c_c's slots at [L:L+c);
        # its own entry is written by alloc_for_decode). Entry j's value is
        # backbone slot j.
        num_cases = self.num_steps + 1
        tri_g = [g for g in range(self.num_steps) for j in range(g + 1)]
        tri_j = [j for g in range(self.num_steps) for j in range(g + 1)]
        self._tri_g = torch.tensor(tri_g, dtype=torch.int64, device=self.device)
        self._tri_j = tri_j
        # Hybrid glue rows re-scan their nested chain prefix (see
        # _glue_forward): tri_j doubles as the per-seat gather for row g's
        # input tokens c_1..c_{g+1} and their (shared) backbone KV slots.
        self._tri_j_dev = torch.tensor(tri_j, dtype=torch.int64, device=self.device)
        self._tri_row_lens = [g + 1 for g in range(self.num_steps)]
        # The full uniform grid: every case enumerated at full F. The slow
        # (bootstrap) round always builds exactly this shape, so it keeps its
        # own handle -- a per-case budget must never reach it.
        self._full_grid_variant = self._build_fanout_variant(
            f_live=self.fanout, budgets=[self.fanout] * num_cases
        )
        # Per-effective-fanout row selections / templates (full width seeded
        # here; smaller widths built on first use by the adaptive-fanout
        # controller).
        self._per_case_budgets = self._resolve_per_case_budgets()
        self._fanout_variants: dict[int, _FanoutVariant] = {
            self.fanout: (
                self._full_grid_variant
                if self._per_case_budgets is None
                else self._build_fanout_variant(
                    f_live=self.fanout, budgets=self._per_case_budgets
                )
            )
        }
        # Dead-guess exclusion (see the env var's comment): fast-path rounds
        # mask the backbone token c_{a+1} out of node a's top-F for a < K.
        self._exclude_dead_guess = (
            exclude_dead_guess
            if exclude_dead_guess is not None
            else envs.SGLANG_ENABLE_DECOUPLED_DEAD_GUESS_EXCLUSION.get()
        )
        # Fused fast-path extend: advance + glue as ONE batched forward (see
        # _fused_extend_forward). Env kill switch, read once (init-static);
        # off restores the two-forward advance + glue round.
        self._enable_fused_extend = envs.SGLANG_ENABLE_DECOUPLED_FUSED_EXTEND.get()
        # Per-delta-length row-gather patterns for the fused extend's assembly
        # (host lists; fast-path delta lengths recur, so the cache stays tiny).
        self._fused_gather_patterns: dict[int, list[int]] = {}
        # Fused-extend CUDA graph (fast rounds only): pad every fused row to
        # the static width W = (K+1) + K (the deepest lockstep delta plus the
        # K chain positions) and replay ONE captured DRAFT_EXTEND_V2 graph
        # instead of ~2(K+1) eager kernel launch waves. Eager fused stays the
        # mandatory fallback: per round (catch-up deltas past the window,
        # capture-bucket overflow) and entirely (kill switch, or any
        # construction failure inside _build_extend_graph_runner).
        self._extend_graph_width = 2 * self.num_steps + 1
        self._extend_graph_pad_span = (
            self._page_roundup(self._page_size - 1 + self._extend_graph_width)
            if self._paged
            else 0
        )
        self._extend_graph_pad_pages = self._extend_graph_pad_span // self._page_size
        self._extend_graph_disable_padding = bool(
            model_runner.server_args.disable_cuda_graph_padding
        )
        self._extend_graph_gather_patterns: dict[int, list[int]] = {}
        self._extend_graph_const_cache: dict[int, tuple[torch.Tensor, list[int]]] = {}
        self._extend_graph_capture_rows: frozenset[int] = frozenset()
        self._extend_graph_max_rows = 0
        self._extend_graph_runner = None
        self._extend_graph_backend = None
        if (
            self._enable_fused_extend
            and self.num_steps >= 1
            and enable_extend_graph
            and envs.SGLANG_ENABLE_DECOUPLED_EXTEND_GRAPH.get()
        ):
            self._extend_graph_runner = self._build_extend_graph_runner()
        if self._extend_graph_runner is not None and self._paged:
            # Graph rounds run pad queries at positions past the written
            # region, and paged resolution sends those reads into the REAL
            # page's never-written offsets (their K/V writes went to junk
            # slots): uninitialized pool memory. On processes where that
            # memory holds NaN bit patterns, the pad hidden states go NaN,
            # the pads' K/V (written into the row's own arena page) go NaN,
            # and an additive causal mask then leaks NaN from masked columns
            # into the row's REAL queries. One init-time zero-fill makes
            # every never-written slot read as zeros instead -- finite
            # garbage is harmless (pad outputs are discarded by design).
            for k_layer, v_layer in zip(
                self._cow_kv_pool.k_buffer, self._cow_kv_pool.v_buffer
            ):
                k_layer.zero_()
                v_layer.zero_()
        # Reusable batch shells for assembled fast subrounds (retained from
        # slow rounds; per-round fields are fully rebound before each use).
        self._glue_template: Optional[ScheduleBatch] = None
        self._branch_template: Optional[ScheduleBatch] = None
        self.hit_ct = 0
        self.miss_ct = 0
        self.profiler = _RoundProfiler(
            enabled=envs.SGLANG_DEBUG_DECOUPLED_DRAFT_PROFILE.get()
        )
        self._h2d = _PinnedH2D(device=self.device)
        self._pending_scratch_frees: list[torch.Tensor] = []
        self._fused_topk = envs.SGLANG_ENABLE_DECOUPLED_FUSED_TOPK.get()
        self._prep_ahead = envs.SGLANG_ENABLE_DECOUPLED_PREP_AHEAD.get()
        # Round-invariant per-delta_len staging tensors (see
        # prebuild_fast_round): built lazily once, reused by every restage.
        self._case_staging_static: dict[int, tuple] = {}
        # Restage isolation: the prebuild's pageable H2Ds and staging kernels
        # ride a dedicated stream, so a ROUND-TAIL restage never barriers on
        # the round's still-draining GPU tail (the reason the old idle-window
        # prebuild almost never fired at tight cadence).
        self._prebuild_stream = torch.cuda.Stream()
        self._chain_plan = envs.SGLANG_ENABLE_DECOUPLED_CHAIN_PLAN.get()
        self._chain_graph = None
        if self._chain_plan and envs.SGLANG_ENABLE_DECOUPLED_CHAIN_GRAPH.get():
            from sglang.srt.speculative.decoupled_chain_graph import (
                ChainGraphRunner,
            )

            self._chain_graph = ChainGraphRunner(
                model_runner=self.model_runner, num_steps=self.num_steps
            )
        self._prebuilt_fast: Optional[_PrebuiltFastRound] = None

    # ------------------------------------------------------------------ #
    # Fused-extend CUDA graph: construction (init-time)
    # ------------------------------------------------------------------ #

    def _build_extend_graph_runner(self):
        """Build the DRAFT_EXTEND_V2 graph stack -- dedicated attention
        backend, worker shim, runner (which captures in its __init__). Any
        failure logs once and leaves the engine on the eager fused path."""
        try:
            return self._build_extend_graph_runner_impl()
        except Exception:
            logger.warning(
                "decoupled fused-extend graph disabled: construction failed; "
                "falling back to the eager fused extend",
                exc_info=True,
            )
            return None

    def _build_extend_graph_runner_impl(self):
        from sglang.srt.model_executor.cuda_graph_config import Backend
        from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
        from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
            EAGLEDraftExtendCudaGraphRunner,
        )

        model_runner = self.model_runner
        graph_config = model_runner.server_args.cuda_graph_config
        if (
            graph_config is None
            or graph_config.decode.backend == Backend.DISABLED
            or not graph_config.decode.bs
        ):
            logger.info("decoupled fused-extend graph off: CUDA graphs disabled")
            return None
        if self._hybrid and not self._extend_graph_hybrid_supported():
            return None
        rows_per_seat = self.num_steps + 1
        # Row buckets: a fused fast round always holds seats x (K+1) rows, so
        # bucket exactly on those multiples. Rows are capped at the decode
        # config's max bs: any round able to run its branch phase (seats x
        # (K+1) x f_live rows through the decode graphs) fits it, and the cap
        # bounds the private full-width logits buffer (max_rows x W x vocab).
        row_buckets = sorted(
            {
                seats * rows_per_seat
                for seats in graph_config.decode.bs
                if seats * rows_per_seat <= max(graph_config.decode.bs)
            }
        )
        if not row_buckets:
            logger.info(
                "decoupled fused-extend graph off: no seats x (K+1) row bucket "
                "fits the decode capture list"
            )
            return None
        facade = _ExtendGraphRunnerFacade(
            model_runner=model_runner,
            server_args=self._extend_graph_server_args(row_buckets=row_buckets),
            graph_shared_output=GraphSharedOutput(
                device=torch.device(model_runner.device),
                max_rows=max(row_buckets) * self._extend_graph_width,
            ),
        )
        backend = self._build_extend_graph_backend(runner_facade=facade)
        self._extend_graph_backend = backend
        shim = _DraftExtendWorkerShim(
            draft_runner=facade,
            draft_extend_attn_backend=backend,
            num_steps=self.num_steps,
            num_draft_tokens=self._extend_graph_width,
        )
        runner = EAGLEDraftExtendCudaGraphRunner(
            shim,
            draft_extend_attn_backend=backend,
            speculative_num_steps=self.num_steps,
        )
        self._extend_graph_capture_rows = frozenset(runner.capture_bs)
        self._extend_graph_max_rows = runner.max_bs
        logger.info(
            "decoupled fused-extend graph captured: width=%d row_buckets=%s",
            self._extend_graph_width,
            sorted(self._extend_graph_capture_rows),
        )
        return runner

    def _extend_graph_hybrid_supported(self) -> bool:
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
        from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend

        main_backend = self.model_runner.attn_backend
        if not (
            isinstance(main_backend, HybridLinearAttnBackend)
            and isinstance(main_backend.linear_attn_backend, GDNAttnBackend)
        ):
            # Only the GDN sidecar implements the true-length
            # DRAFT_EXTEND_V2 plane (see GDNAttnBackend); other linear
            # sidecars would silently scan the pads.
            logger.info(
                "decoupled fused-extend graph off: hybrid drafter without a "
                "GDN linear plane (%s)",
                type(main_backend).__name__,
            )
            return False
        if self.model_runner.server_args.enable_page_major_kv_layout:
            # The v2 GDN path runs the conv / recurrent kernels directly on
            # the pool tensors; only the eager gather/scatter path handles
            # the page-major envelope's strided state layout.
            logger.info("decoupled fused-extend graph off: page-major mamba layout")
            return False
        return True

    def _extend_graph_server_args(self, *, row_buckets: list[int]):
        """Patched ServerArgs copy for the dedicated draft-extend graph stack.

        The drafter role hook nulls ``speculative_num_draft_tokens`` (a
        spec-shaped ModelRunner would corrupt the engine's 1-token decode
        graphs, and on hybrid models the field sizes gigantic verify state
        buffers), but the draft-extend runner and every attention backend it
        constructs derive their fixed per-request width from exactly that
        field -- restore it to W on the copy only. The copy also swaps the
        decode capture buckets for the fused-ROW buckets, which is what
        get_batch_sizes_to_capture reads for this runner.
        """
        args = copy.copy(self.model_runner.server_args)
        # The shallow copy shares the provenance lists with the original;
        # rebind fresh ones so the view's override cannot pollute the real
        # args' audit log.
        object.__setattr__(args, "_resolved_overrides", [])
        object.__setattr__(args, "_runtime_mutations", [])
        graph_config = copy.deepcopy(args.cuda_graph_config)
        graph_config.decode.bs = list(row_buckets)
        args.override(
            "decoupled-extend-graph",
            speculative_num_draft_tokens=self._extend_graph_width,
            # The runner reads topk from server_args, not the worker shim; the
            # role validator pins it to 1 on real drafter servers, but harness
            # contexts (the round microbench) build ServerArgs without the
            # decoupled hook and leave it None.
            speculative_eagle_topk=1,
            cuda_graph_config=graph_config,
        )
        return args

    def _build_extend_graph_backend(self, *, runner_facade):
        """DEDICATED attention backend for the fused-extend graphs. The
        engine's main backend must never be reused here: the runner calls
        init_cuda_graph_state on it, which re-allocates the buffers the
        drafter's decode graphs already captured by pointer."""
        from sglang.srt.speculative.draft_utils import DraftBackendFactory

        full_attn_backend = DraftBackendFactory(
            runner_facade.server_args,
            runner_facade,
            1,  # topk: decoupled spec pins chain drafting
            self.num_steps,
        ).create_draft_extend_backend()
        if not self._hybrid:
            return full_attn_backend
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
        from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend

        # A hybrid drafter dispatches EVERY layer through the ForwardContext
        # backend, so the dedicated instance must be a full wrapper. The
        # factory's hybrid draft-extend entry returns only the full-attn
        # plane (MTP draft models have no linear layers); the GDN plane is a
        # fresh sidecar carrying its own DRAFT_EXTEND_V2 metadata buffers.
        gdn_sidecar = GDNAttnBackend(runner_facade)
        # Junk state slot every pad segment scans through (never skipped --
        # a skipped segment's output stays uninitialized and NaN-poisons the
        # row's KV page; see the sidecar field comment). Engine-owned arena
        # slot, held for the engine's lifetime; translated once here -- the
        # engine already rejects the pool variants with unstable translation
        # (extra-buffer, replayssm) at construction.
        pool = self.model_runner.req_to_token_pool
        pad_state_slot = self._mamba_arena.take(1)
        pad_state_phys = pool.translate_mamba_indices(pad_state_slot)
        # Zero it like open() zeroes seat slots: the gating kernel loads the
        # initial state UNCONDITIONALLY from the slot index (only the poison
        # id means "cold"), and a fresh allocator slot is uninitialized
        # memory -- NaN there and every pad scan starts from NaN.
        pool.mamba_pool.clear_slots(pad_state_phys)
        gdn_sidecar.draft_extend_v2_pad_state_slot = int(pad_state_phys.item())
        return HybridLinearAttnBackend(
            full_attn_backend,
            gdn_sidecar,
            self.model_runner.attn_backend.full_attn_layers,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle (control plane)
    # ------------------------------------------------------------------ #

    def open(
        self,
        key: DraftReqKey,
        *,
        req_pool_idx: int,
        prompt_tokens: list[int],
        committed_outputs: list[int],
    ) -> None:
        # Re-open (retraction re-sync) drops the old prefix KV entirely.
        self.close(key)
        state = _DraftReqState(req_pool_idx=req_pool_idx, device=self.device)
        state.committed_tokens = list(prompt_tokens) + list(committed_outputs)
        if self._hybrid:
            # The seat slot must start ZEROED: the GDN extend kernel reads the
            # slot's ssm state as the initial state unconditionally, and an
            # arena-recycled slot still carries the previous seat's state.
            state.mamba_slot = self._mamba_arena.take(1)
            pool = self.model_runner.req_to_token_pool
            pool.mamba_pool.clear_slots(pool.translate_mamba_indices(state.mamba_slot))
        self._states[key] = state

    def apply_commit(self, key: DraftReqKey, committed_tokens: list[int]) -> bool:
        """Apply a real commit; returns True when an active top-1 prerun bet
        matched it exactly (the seat's next block is already on the verifier,
        so the caller skips drafting it this round)."""
        state = self._states.get(key)
        if state is None:
            return False
        delta = [int(t) for t in committed_tokens]
        if any(token < 0 for token in delta):
            # Belt to the verifier-side check: a negative id would reach the
            # embedding gather and kill the whole drafter with a device
            # assert. Dropping the commit desyncs only this seat (staleness
            # fallbacks until the next re-sync), never the process.
            logger.error(
                "commit for %s dropped: negative token in delta %s",
                key.request_id,
                delta[:8],
            )
            return False
        if state.prerun_len > 0:
            bet = state.committed_tokens[-state.prerun_len :]
            if delta == bet:
                state.prerun_len = 0
                state.prerun_snapshot = None
                # Bet confirmed: the seat slot's bet-advanced state is now the
                # real committed state; the pre-bet stash is simply dropped.
                self._drop_prerun_mamba_stash(state)
                self.prerun_hit_ct += 1
                return True
            self._rollback_prerun(state)
            self.prerun_miss_ct += 1
            if self.prerun_miss_ct <= 3:
                # Alignment probe: a systematic construction/compare bug shows
                # up in the first few misses (bet vs real delta side by side).
                logger.info(
                    "prerun miss #%d: bet=%s delta=%s",
                    self.prerun_miss_ct,
                    bet[:8],
                    delta[:8],
                )
        state.committed_tokens.extend(delta)
        return False

    def _rollback_prerun(self, state: _DraftReqState) -> None:
        """Undo a wrong bet: drop the speculative tokens, free their KV, and
        restore the pre-bet mirrors (the real delta must be matched against
        the pre-bet block, not the bet one)."""
        base_len = len(state.committed_tokens) - state.prerun_len
        if state.committed_slots.numel() > base_len:
            # Page-granular truncate: only pages FULLY past the cut are freed
            # (the allocator frees the whole page containing any freed slot,
            # and the boundary page still holds committed KV). Bet slots left
            # in the boundary page are stale but harmless: they sit past the
            # seat's seq_len, and the next advance's alloc_extend resumes at
            # last_loc + 1 -- the exact same slots -- overwriting them before
            # anything can read them. At page_size == 1 the roundup is the
            # identity and this is the original token-granular free.
            free_from = self._page_roundup(base_len)
            if state.committed_slots.numel() > free_from:
                self.model_runner.token_to_kv_pool_allocator.free(
                    state.committed_slots[free_from:]
                )
            state.committed_slots = state.committed_slots[:base_len]
        state.committed_tokens = state.committed_tokens[:base_len]
        if state.prerun_mamba_stash is not None:
            # The bet's advance ran IN PLACE over the seat slot; restore the
            # stashed pre-bet state before the real delta re-advances it.
            self._fork_mamba_states(
                src_slots=state.prerun_mamba_stash, dst_slots=state.mamba_slot
            )
            self._drop_prerun_mamba_stash(state)
        units_dev, units_host_clone, backbone_host, mirror_event = state.prerun_snapshot
        state.last_units_dev = units_dev
        if state.last_units_host is not None and units_host_clone is not None:
            # Restore INTO the pinned buffer so the mirror keeps its identity.
            state.last_units_host.copy_(units_host_clone)
        state.last_backbone_host = backbone_host
        state.mirror_event = mirror_event
        state.prerun_len = 0
        state.prerun_snapshot = None

    def _drop_prerun_mamba_stash(self, state: _DraftReqState) -> None:
        if state.prerun_mamba_stash is not None:
            self._mamba_arena.give_back(state.prerun_mamba_stash)
            state.prerun_mamba_stash = None

    @torch.no_grad()
    def speculative_prerun(self, keys: list[DraftReqKey]) -> Optional[dict]:
        """Bet each seat's most likely next commit (full accept + its own top
        bonus guess g_{K,0}), pre-run that round now, and return the packed
        block to ship speculatively. By construction the bet delta hits the
        glue fast path. A wrong bet is rolled back by apply_commit and only
        cost idle drafter time."""
        ready: list[tuple[DraftReqKey, _DraftReqState]] = []
        for key in keys:
            state = self._states.get(key)
            if (
                state is None
                or state.prerun_len > 0
                or state.last_units_host is None
                # Empty after a case-0 miss round: that block carries no
                # backbone, so there is no full-accept outcome to bet.
                or not state.last_backbone_host
                or key not in self._seat_carriers
            ):
                continue
            ready.append((key, state))
        if not ready:
            return None
        for _, state in ready:
            if state.mirror_event is not None:
                state.mirror_event.synchronize()
            bet_delta = list(state.last_backbone_host) + [
                int(state.last_units_host[self.num_steps, 0, 0])
            ]
            state.prerun_snapshot = (
                state.last_units_dev,
                state.last_units_host.clone(),
                state.last_backbone_host,
                state.mirror_event,
            )
            if self._hybrid:
                # Snapshot the seat state alongside the mirrors: the prerun's
                # advance is about to run the bet tokens over the seat slot.
                state.prerun_mamba_stash = self._mamba_arena.take(1)
                self._fork_mamba_states(
                    src_slots=state.mamba_slot, dst_slots=state.prerun_mamba_stash
                )
            state.committed_tokens.extend(bet_delta)
            state.prerun_len = len(bet_delta)
        try:
            return self.draft_round([key for key, _ in ready])
        except Exception:
            for _, state in ready:
                if state.prerun_len > 0:
                    self._rollback_prerun(state)
            raise

    def close(self, key: DraftReqKey) -> None:
        self.drop_prebuilt_for(key)
        self._evict_seat(key)
        state = self._states.pop(key, None)
        if state is None:
            return
        if state.committed_slots.numel() > 0:
            self.model_runner.token_to_kv_pool_allocator.free(state.committed_slots)
        # Seat + stash slots return to the engine arena (never to the pool
        # allocator: the arena keeps ownership for reuse across seats).
        if state.mamba_slot is not None:
            self._mamba_arena.give_back(state.mamba_slot)
            state.mamba_slot = None
        self._drop_prerun_mamba_stash(state)

    def has(self, key: DraftReqKey) -> bool:
        return key in self._states

    # ------------------------------------------------------------------ #
    # One enumeration round
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def draft_round(self, keys: list[DraftReqKey]) -> Optional[dict]:
        """Draft one enumeration tree per key; returns the packed parallel
        arrays {pool_indices, base_committed_lens, tokens} or None if no key
        is live. Frees every scratch slot before returning.

        Three forms of the same tree, mixed PER SEAT within one call:

        - **glue fast path** (this seat's commit matched a unit of its last
          block): the winning unit's chain IS the new backbone (greedy
          re-draft is deterministic), so one K-row extend re-materializes
          its KV and yields all node logits; the branch phase runs as plain
          decode replays assembled from the seat's retained carrier rows.
          With the fused extend (default, see _fused_extend_forward) the
          advance folds in as one more row per seat, so the whole pre-branch
          phase is a single (K+1)-row-per-seat forward.
        - **case-0 miss round** (commit fell outside the last block, carrier
          exists): a drafter miss mirrors a verifier select miss, so the
          next commit is necessarily a single fallback bonus -- only the
          case-0 rows of this block can ever be read. Enumerate just those
          F chains on the carrier rows and poison the dead cells.
        - **bootstrap** (no carrier yet): the original build-everything
          round for the fresh seats only; it also builds their seat
          carriers and host mirrors.
        """
        keys = [key for key in keys if key in self._states]
        if not keys:
            return None
        scratch_batches: list[ScheduleBatch] = []
        scratch_slots: list[torch.Tensor] = []
        # Hybrid only: mamba slots that must go back to the arena at round end
        # (slow-path transients; carrier-destined slots are appended too and
        # removed on donation, so an aborted slow round cannot leak them).
        scratch_mamba_slots: list[torch.Tensor] = []
        # page_size > 1 only: KV arena pages with the same lifecycle (slow-path
        # backbone transients + carrier-destined pages until donation).
        scratch_kv_pages: list[torch.Tensor] = []
        self.profiler.start_round()
        try:
            hit_keys: list[DraftReqKey] = []
            hit_states: list[_DraftReqState] = []
            selections: list[tuple[int, int]] = []
            case0_keys: list[DraftReqKey] = []
            case0_states: list[_DraftReqState] = []
            slow_keys: list[DraftReqKey] = []
            slow_states: list[_DraftReqState] = []
            for key in keys:
                state = self._states[key]
                selection = self._match_seat(key, state)
                if selection is not None:
                    hit_keys.append(key)
                    hit_states.append(state)
                    selections.append(selection)
                elif key in self._seat_carriers:
                    case0_keys.append(key)
                    case0_states.append(state)
                else:
                    slow_keys.append(key)
                    slow_states.append(state)
            f_live = max(1, min(int(self.effective_fanout), self.fanout))
            if self._prebuilt_fast is not None and not hit_states:
                # A miss/bootstrap round for these seats invalidates the
                # hypothesized skeleton (committed lengths move); fold it in
                # for the round-tail free.
                pre = self._prebuilt_fast
                self._prebuilt_fast = None
                self._scrap_prebuilt_into(
                    pre, scratch_batches, scratch_slots, scratch_kv_pages
                )
            parts: list[dict] = []
            if hit_states:
                self.hit_ct += 1
                parts.append(
                    self._fast_round(
                        hit_keys,
                        hit_states,
                        selections,
                        scratch_batches,
                        scratch_slots,
                        scratch_kv_pages,
                        f_live=f_live,
                    )
                )
            if case0_states or slow_states:
                self.miss_ct += 1
            if case0_states:
                parts.append(
                    self._case0_round(
                        case0_keys,
                        case0_states,
                        scratch_batches,
                        scratch_slots,
                        scratch_kv_pages,
                        f_live=f_live,
                    )
                )
            if slow_states:
                parts.append(
                    self._slow_round(
                        slow_keys,
                        slow_states,
                        scratch_batches,
                        scratch_slots,
                        scratch_mamba_slots,
                        scratch_kv_pages,
                    )
                )
            if len(parts) == 1:
                return parts[0]
            return {
                "pool_indices": [
                    pool_idx for part in parts for pool_idx in part["pool_indices"]
                ],
                "base_committed_lens": [
                    base_len
                    for part in parts
                    for base_len in part["base_committed_lens"]
                ],
                "units_device": torch.cat([part["units_device"] for part in parts]),
            }
        finally:
            with self.profiler.stage("free-scratch"):
                self._free_scratch(
                    scratch_batches,
                    scratch_slots,
                    scratch_mamba_slots,
                    scratch_kv_pages,
                )
            self.profiler.mark("free")

    def _match_seat(
        self, key: DraftReqKey, state: _DraftReqState
    ) -> Optional[tuple[int, int]]:
        """Match one seat's pending delta against its last block; returns the
        winning (accept_case, fanout_index) or None (seat misses)."""
        if not self._enable_glue_fast_path:
            return None
        if key not in self._seat_carriers:
            return None
        if state.last_units_host is None or state.last_backbone_host is None:
            return None
        delta = state.pending_delta()
        case = len(delta) - 1
        if case < 0 or case > self.num_steps:
            return None
        if delta[:case] != state.last_backbone_host[:case]:
            return None
        if state.mirror_event is not None:
            state.mirror_event.synchronize()
        guesses_row = state.last_units_host[case, :, 0].tolist()
        bonus = delta[case]
        if bonus not in guesses_row:
            return None
        return (case, guesses_row.index(bonus))

    def _resolve_per_case_budgets(self) -> Optional[list[int]]:
        """Skewed per-case guess-column budget for fast rounds, or None (one
        uniform width for every case) when the knob is off / F == 1.

        Case a < K is only ever reached by verify REJECTING backbone token
        c_{a+1}, so that case's bonus is a rank-2+ candidate by construction
        (the exclusion already hands it the slot) -- worth one column. Case K's
        bonus is the unconstrained full-accept continuation, worth the whole
        width. Budget: case K -> F, case K-1 -> min(2, F), every shallower
        case -> 1; at K=3, F=4 that is [1, 1, 2, 4] = 8 branch rows, exactly
        the uniform-F2 row cost. The wire block stays (K+1) x F with the
        unbudgeted cells poisoned, so the verifier is oblivious.
        """
        if self.fanout < 2 or not envs.SGLANG_ENABLE_DECOUPLED_PER_CASE_FANOUT.get():
            return None
        budgets = [1] * (self.num_steps + 1)
        budgets[self.num_steps] = self.fanout
        if self.num_steps >= 1:
            budgets[self.num_steps - 1] = min(2, self.fanout)
        return budgets

    def _build_fanout_variant(
        self, *, f_live: int, budgets: list[int]
    ) -> _FanoutVariant:
        """Row selections + scatter templates for one per-case column budget
        (a uniform width f is just ``budgets == [f] * (K+1)``).

        Selected rows keep their FULL-F pool position (case c, column f -> row
        c * F + f), so every budget vector reuses the same carrier rows; only
        which of them a round forwards changes.
        """
        num_cases = self.num_steps + 1
        fanout = self.fanout
        sel_offsets = [sum(budgets[:c]) for c in range(num_cases)]
        cases_cols = [(c, f) for c in range(num_cases) for f in range(budgets[c])]
        sel_rows_pool = [c * fanout + f for c, f in cases_cols]
        br_r_pool = [c * fanout + f for c, f in cases_cols for _ in range(c)]
        br_r_sel = [sel_offsets[c] + f for c, f in cases_cols for _ in range(c)]
        br_j = [j for c, _ in cases_cols for j in range(c)]
        br_c = [c for c, _ in cases_cols for _ in range(c)]
        case_of_row = [c for c, _ in cases_cols]
        guess_dead_mask = None
        if min(budgets) != max(budgets):
            live = torch.zeros((1, num_cases, fanout), dtype=torch.bool)
            for c, budget in enumerate(budgets):
                live[0, c, :budget] = True
            guess_dead_mask = live.logical_not().to(self.device)
        return _FanoutVariant(
            f_live=f_live,
            budgets=list(budgets),
            guess_dead_mask=guess_dead_mask,
            sel_rows_pool=sel_rows_pool,
            sel_rows_dev=torch.tensor(
                sel_rows_pool, dtype=torch.int64, device=self.device
            ),
            br_r_pool=torch.tensor(br_r_pool, dtype=torch.int64, device=self.device),
            br_r_sel=torch.tensor(br_r_sel, dtype=torch.int64, device=self.device),
            br_j=torch.tensor(br_j, dtype=torch.int64, device=self.device),
            br_case=torch.tensor(br_c, dtype=torch.int64, device=self.device),
            comb_j=torch.tensor(
                self._tri_j + br_j, dtype=torch.int64, device=self.device
            ),
            case_of_row=case_of_row,
            case_of_row_dev=torch.tensor(
                case_of_row, dtype=torch.int64, device=self.device
            ),
        )

    def _fanout_variant(self, f_live: int) -> _FanoutVariant:
        """Row selections + scatter templates for an effective fanout: the
        first ``f_live`` of every case's F branch rows -- or the skewed
        per-case budget, seeded at full width when that knob is on (a lowered
        width always falls back to uniform, so the adaptive controller keeps
        its old behavior). The full-F pool layout is untouched, so any width
        <= F reuses the same carrier rows."""
        variant = self._fanout_variants.get(f_live)
        if variant is None:
            variant = self._build_fanout_variant(
                f_live=f_live, budgets=[f_live] * (self.num_steps + 1)
            )
            self._fanout_variants[f_live] = variant
        return variant

    def _fast_round(
        self,
        keys: list[DraftReqKey],
        states: list[_DraftReqState],
        selections: list[tuple[int, int]],
        scratch_batches: list[ScheduleBatch],
        scratch_slots: list[torch.Tensor],
        scratch_kv_pages: list[torch.Tensor],
        *,
        f_live: int,
    ) -> dict:
        num_steps = self.num_steps
        bs = len(states)
        carriers = [self._seat_carriers[key] for key in keys]
        variant = self._fanout_variant(f_live)
        # Guess columns to take per node: the live width, or the FULL width
        # under a per-case budget (the grid is then trimmed by poisoning, so
        # the block keeps its fixed (K+1) x F shape).
        guess_width = f_live if variant.guess_dead_mask is None else self.fanout

        # -- Winning chains: the selected units' chains ARE the new backbone
        # (the host mirror was synced during matching).
        chains: list[torch.Tensor] = []
        new_backbones: list[list[int]] = []
        for state, (case, f) in zip(states, selections):
            chains.append(state.last_units_dev[case, f, 1:])
            new_backbones.append(state.last_units_host[case, f, 1:].tolist())
        # Backbone token matrix for dead-guess exclusion: case a < K is only
        # reached by verify REJECTING c_{a+1}, so that token can never be the
        # case's bonus -- mask it before top-F so a live candidate gets the
        # slot instead.
        chains_mat = torch.stack(chains) if self._exclude_dead_guess else None

        # -- Seat (advance) rows: one fresh pool row per seat extending the
        # committed delta; the only phase that ever runs IN PLACE on the
        # seat's mamba slot. Fused mode uses the batch purely as the
        # ALLOCATION vehicle (pool row + delta KV slots + hybrid seat-slot
        # binding via the preset) -- the forward that actually writes the
        # delta KV is the fused extend below, where the seat row's last
        # position yields node-0 logits. Non-fused mode forwards it here,
        # exactly as on the slow path.
        base_lens = [state.committed_slots.numel() for state in states]
        delta_lens = [
            len(state.committed_tokens) - base_lens[i] for i, state in enumerate(states)
        ]
        # Graph-vs-eager is decided HERE, before any carrier state is
        # written: the two paths bind different page tables / write slots and
        # must never mix within a round.
        seat_input_ids: Optional[torch.Tensor] = None
        glue_preseeded = False
        prestaged_case: Optional[_ExtendCaseStaging] = None
        with self.profiler.stage("alloc-seat"):
            prebuilt = self._prebuilt_fast
            if (
                prebuilt is not None
                and prebuilt.keys == tuple(keys)
                and prebuilt.base_lens == base_lens
                and self._enable_fused_extend
            ):
                # Prebuilt skeleton: allocation + page-table writes already
                # happened (on the prebuild stream); order this round's work
                # after them, then fill the actual delta tokens and go.
                if prebuilt.ready_event is not None:
                    torch.cuda.current_stream().wait_event(prebuilt.ready_event)
                self._prebuilt_fast = None
                glue_preseeded = prebuilt.glue_seeded
                graph_round = prebuilt.graph_round
                advance_batch = prebuilt.batch
                scratch_batches.extend(prebuilt.scratch_batches)
                scratch_slots.extend(prebuilt.scratch_slots)
                scratch_kv_pages.extend(prebuilt.scratch_kv_pages)
                advance_slots, seat_input_ids = self._consume_prebuilt(
                    prebuilt,
                    states=states,
                    base_lens=base_lens,
                    delta_lens=delta_lens,
                    scratch_slots=scratch_slots,
                )
                if (
                    prebuilt.case_staging is not None
                    and bs == 1
                    and 1 <= delta_lens[0] <= len(prebuilt.case_staging)
                ):
                    prestaged_case = prebuilt.case_staging[delta_lens[0] - 1]
            else:
                if prebuilt is not None:
                    self._prebuilt_fast = None
                    self._scrap_prebuilt_into(
                        prebuilt, scratch_batches, scratch_slots, scratch_kv_pages
                    )
                graph_round = self._stage_extend_graph_round(
                    bs=bs,
                    delta_lens=delta_lens,
                    scratch_slots=scratch_slots,
                    scratch_kv_pages=scratch_kv_pages,
                    rows_per_seat=self.num_steps + 1,
                )
                advance_batch, advance_slots = self._extend_batch(
                    token_lists=[state.committed_tokens for state in states],
                    prefix_slots=[state.committed_slots for state in states],
                    tag="advance",
                    mamba_slots=self._seat_mamba_slots(states),
                )
        scratch_batches.append(advance_batch)
        if seat_input_ids is None:
            seat_input_ids = advance_batch.input_ids
        if not self._enable_fused_extend:
            node0_logits = self._forward(advance_batch, tag="advance")
            # Graph-runner logits live in a static output buffer that the NEXT
            # forward overwrites -- consume them (topk) before the glue
            # forward, exactly like the slow path consumes each step's logits
            # immediately. (The in-place mask writes into that same buffer; it
            # is consumed by the topk on the next line and rewritten by the
            # next replay.)
            if chains_mat is not None:
                node0_logits.scatter_(-1, chains_mat[:, :1], float("-inf"))
            node0_guesses = torch.topk(node0_logits, guess_width, dim=-1).indices
        self._absorb_advance_slots(states, advance_slots)

        # -- Carrier pool rows: broadcast the committed delta, then scatter
        # this round's backbone slots into the glue triangle + branch cases.
        # (page_size > 1 dispatches to the private-page rebind instead and
        # returns None -- backbone KV then lives per row, never shared.)
        with self.profiler.stage("page-table-sync"):
            backbone_slots = self._sync_carrier_rows(
                states=states,
                carriers=carriers,
                base_lens=base_lens,
                variant=variant,
                f_live=f_live,
                scratch_slots=scratch_slots,
                graph_round=graph_round,
            )
        if self._paged and not glue_preseeded:
            # Seed the glue rows' private heads before the forward reads them.
            with self.profiler.stage("cow-glue-heads"):
                self._cow_carrier_glue_heads(
                    states=states, carriers=carriers, base_lens=base_lens
                )

        if self._hybrid and not glue_preseeded:
            # All K glue rows fork their state from the SEAT slot before the
            # forward that runs them; what the copy HOLDS differs by mode.
            # Fused: the seat slot still carries the PRE-advance state
            # (nothing ran on it yet -- the fused forward itself advances it
            # by the delta), so each glue row's linear layers re-scan
            # delta + c_1..c_{g+1} from the pre-advance copy. Non-fused: the
            # advance above already left node-0 state, and each glue row
            # re-scans only c_1..c_{g+1} (see _glue_forward). Either way glue
            # row g's slot ends its forward holding node-(g+1) state, so the
            # branch fork below is mode-blind.
            with self.profiler.stage("gdn-fork-glue"):
                self._fork_mamba_states(
                    src_slots=torch.cat(
                        [state.mamba_slot.repeat(num_steps) for state in states]
                    ),
                    dst_slots=torch.cat(
                        [carrier.glue_mamba_slots for carrier in carriers]
                    ),
                )
            if envs.SGLANG_DEBUG_DECOUPLED_EXTEND_GRAPH_NANCHECK.get():
                self._log_mamba_state_nan(
                    tag="post-fork",
                    slots=torch.cat(
                        [states[0].mamba_slot, carriers[0].glue_mamba_slots]
                    ),
                )
        if self._enable_fused_extend:
            # -- Fused extend: seat + glue rows in ONE forward. The glue chain
            # tokens are the matched commit's winning unit -- KNOWN values,
            # never derived from the advance's logits -- so the two phases
            # have no data dependency; per-seat row r yields node-r logits.
            with self.profiler.stage(
                "extend-graph" if graph_round is not None else "extend-eager"
            ):
                fused_logits = self._fused_extend_forward(
                    carriers=carriers,
                    chains=chains,
                    backbone_slots=backbone_slots,
                    seat_reqs=advance_batch.reqs,
                    seat_rows=advance_batch.req_pool_indices,
                    seat_input_ids=seat_input_ids,
                    seat_delta_slots=advance_slots,
                    base_lens=base_lens,
                    delta_lens=delta_lens,
                    graph_round=graph_round,
                    prestaged=prestaged_case,
                )
            if self._hybrid and envs.SGLANG_DEBUG_DECOUPLED_EXTEND_GRAPH_NANCHECK.get():
                self._log_mamba_state_nan(
                    tag="post-fused",
                    slots=torch.cat(
                        [states[0].mamba_slot, carriers[0].glue_mamba_slots]
                    ),
                )
            with self.profiler.stage("guess-topk"):
                # Consume the static logits buffer (mask + topk) before the
                # first branch replay overwrites it. Node a's dead token is
                # c_{a+1}, so rows 0..K-1 mask c_1..c_K; node K keeps its full
                # top-F (a full accept's bonus is unconstrained).
                if self._fused_topk:
                    guesses_stack = fused_guess_topk(
                        fused_logits,
                        chains_mat,
                        nodes=num_steps + 1,
                        width=guess_width,
                    ).view(bs, num_steps + 1, guess_width)
                else:
                    fused_view = fused_logits.view(bs, num_steps + 1, -1)
                    if chains_mat is not None:
                        fused_view[:, :num_steps].scatter_(
                            -1, chains_mat.unsqueeze(-1), float("-inf")
                        )
                    guesses_stack = torch.topk(fused_view, guess_width, dim=-1).indices
        else:
            # -- Glue extend: all K backbone tokens in one forward = node 1..K
            # logits; their KV lands in this round's backbone slots (page 1)
            # or in each glue row's private pages (page > 1).
            glue_logits = self._glue_forward(
                carriers=carriers,
                states=states,
                chains=chains,
                backbone_slots=backbone_slots,
                base_lens=base_lens,
            )
            glue_view = glue_logits.view(bs, num_steps, -1)
            if chains_mat is not None and num_steps >= 2:
                # Node a's dead token is c_{a+1}: glue row g holds node g+1,
                # so rows 0..K-2 mask c_2..c_K; node K (row K-1) keeps its
                # full top-F (a full accept's bonus is unconstrained).
                glue_view[:, : num_steps - 1].scatter_(
                    -1, chains_mat[:, 1:].unsqueeze(-1), float("-inf")
                )
            glue_guesses = torch.topk(glue_view, guess_width, dim=-1).indices
            guesses_stack = torch.cat([node0_guesses.unsqueeze(1), glue_guesses], dim=1)
        if variant.guess_dead_mask is not None:
            # Per-case budget: the top-k ran at FULL width, so poison every
            # column past its case's budget. The block ships those cells dead
            # (-1, matching nothing on either side) and the branch phase below
            # drafts only the budgeted ones.
            guesses_stack = guesses_stack.masked_fill(variant.guess_dead_mask, -1)
            branch_guesses = guesses_stack.reshape(bs, -1)[:, variant.sel_rows_dev]
        else:
            branch_guesses = guesses_stack

        if self._paged:
            # Branch prefixes (boundary tail + delta + case backbone) now
            # exist only in the seat's partial page and the glue rows'
            # private pages -- copy each selected row's head into its own
            # pages before the chain decodes read them.
            with self.profiler.stage("cow-branch-heads"):
                self._cow_carrier_branch_heads(
                    states=states,
                    carriers=carriers,
                    base_lens=base_lens,
                    variant=variant,
                )

        # -- Branch chains: K decode replays on the assembled carrier rows.
        if self._hybrid:
            # Branch case a's rows start from node-a state: the seat slot for
            # a == 0, glue row (a-1)'s slot otherwise (its re-scan ended
            # exactly at node a). One batched fork for all selected rows; the
            # chain decode steps then advance the branch slots in place.
            with self.profiler.stage("gdn-fork-branch"):
                self._fork_mamba_states(
                    src_slots=torch.cat(
                        [
                            torch.cat([state.mamba_slot, carrier.glue_mamba_slots])[
                                variant.case_of_row_dev
                            ]
                            for state, carrier in zip(states, carriers)
                        ]
                    ),
                    dst_slots=torch.cat(
                        [
                            carrier.branch_mamba_slots[variant.sel_rows_dev]
                            for carrier in carriers
                        ]
                    ),
                )
        with self.profiler.stage("branch-chain"):
            chain_steps = self._branch_decode_chain(
                carriers=carriers,
                states=states,
                branch_guesses=branch_guesses,
                backbone_slots=backbone_slots,
                scratch_slots=scratch_slots,
                variant=variant,
            )
        with self.profiler.stage("pack-block"):
            return self._pack_and_mirror(
                states=states,
                guesses_stack=guesses_stack,
                chain_steps=chain_steps,
                new_backbones=new_backbones,
                sel_rows_dev=(
                    None if variant.guess_dead_mask is None else variant.sel_rows_dev
                ),
            )

    def _sync_carrier_rows(
        self,
        *,
        states: list[_DraftReqState],
        carriers: list[_SeatCarrier],
        base_lens: list[int],
        variant: _FanoutVariant,
        f_live: int,
        scratch_slots: list[torch.Tensor],
        graph_round: Optional[_ExtendGraphRound],
    ) -> Optional[torch.Tensor]:
        """Broadcast the committed delta into every carrier pool row, then
        scatter freshly allocated backbone slots into the glue triangle +
        branch case prefixes. Returns the backbone slots as [bs, K].

        page_size > 1 has no shared backbone slots to scatter (every row owns
        private copies) and dispatches to the private-page rebind, returning
        None. Graph rounds at page_size == 1 likewise keep no shared backbone
        slots (every fused row writes its own W-slot window) and dispatch to
        the graph rebind, returning None.
        """
        if self._paged:
            self._sync_carrier_rows_paged(
                states=states,
                carriers=carriers,
                base_lens=base_lens,
                bind_graph_width=graph_round is not None,
            )
            return None
        if graph_round is not None:
            self._sync_carrier_rows_graph(
                states=states,
                carriers=carriers,
                base_lens=base_lens,
                variant=variant,
                f_live=f_live,
                w_slots=graph_round.w_slots,
            )
            return None
        num_steps = self.num_steps
        bs = len(states)
        pool = self.model_runner.req_to_token_pool
        backbone_slots = self.model_runner.token_to_kv_pool_allocator.alloc(
            bs * num_steps
        )
        if backbone_slots is None:
            raise RuntimeError("drafter KV pool exhausted (glue backbone)")
        # P == 1 only (the paged / graph rounds returned above), so a plain
        # slot-list free is the round-end contract here.
        scratch_slots.append(backbone_slots)
        backbone_slots = backbone_slots.view(bs, num_steps)
        for i, (state, carrier) in enumerate(zip(states, carriers)):
            new_len = state.committed_slots.numel()
            synced = min(carrier.synced_len, base_lens[i])
            pool.req_to_token[carrier.all_rows, synced:new_len] = state.committed_slots[
                synced:new_len
            ].to(torch.int32)
            carrier.synced_len = new_len
            slots_i32 = backbone_slots[i].to(torch.int32)
            comb_rows = carrier.comb_rows_for(
                f_live=f_live, tri_g=self._tri_g, br_r_pool=variant.br_r_pool
            )
            pool.req_to_token[comb_rows, variant.comb_j + new_len] = slots_i32[
                variant.comb_j
            ]
        self.profiler.mark("carrier_sync")
        return backbone_slots

    def _sync_carrier_rows_paged(
        self,
        *,
        states: list[_DraftReqState],
        carriers: list[_SeatCarrier],
        base_lens: list[int],
        bind_graph_width: bool,
    ) -> None:
        """Page-aware carrier-row sync (see the module docstring's rule):
        broadcast only the seat's FULL pages (read-only by construction --
        the seat never grows a full page again), then bind the private region
        [anchor, anchor + w) of EVERY carrier row to its own arena pages.

        The anchor is the page floor of the pre-advance committed length, so
        a private slot's in-page offset equals its logical position's offset
        (the alignment paged attention backends and alloc_extend/alloc_decode
        assume). w = boundary tail + delta + K covers every prefix entry any
        row reads this round; entries the chain decodes append past w are
        written by alloc_decode itself. Writing all rows (not just the
        selected fanout subset) keeps every entry a round can read freshly
        rewritten, so no stale mapping from an earlier round -- including
        pages the seat has since freed or re-anchored -- can ever be
        dereferenced.

        A graph round pads every fused row's query window to the static W and
        runs it under a uniform post-write seq_len of base + W, so paged
        backends resolve the pad positions through the page table too: the
        binding then covers [anchor, base + W) = boundary tail + W entries,
        which the carrier arena pages fit by construction
        (span_max = P - 1 + W).
        """
        pool = self.model_runner.req_to_token_pool
        for i, (state, carrier) in enumerate(zip(states, carriers)):
            new_len = state.committed_slots.numel()
            anchor = self._page_floor(base_lens[i])
            synced = min(carrier.synced_len, anchor)
            pool.req_to_token[carrier.all_rows, synced:anchor] = state.committed_slots[
                synced:anchor
            ].to(torch.int32)
            carrier.synced_len = anchor
            if bind_graph_width:
                width = (base_lens[i] - anchor) + self._extend_graph_width
            else:
                width = new_len - anchor + self.num_steps
            pool.req_to_token[carrier.all_rows, anchor : anchor + width] = (
                carrier.all_private_slots[:, :width].to(torch.int32)
            )
        self.profiler.mark("carrier_sync")

    def _sync_carrier_rows_graph(
        self,
        *,
        states: list[_DraftReqState],
        carriers: list[_SeatCarrier],
        base_lens: list[int],
        variant: _FanoutVariant,
        f_live: int,
        w_slots: torch.Tensor,
    ) -> None:
        """page_size == 1 carrier sync for a fused-extend GRAPH round: every
        fused row writes its FULL padded W-token window into its OWN slots
        (``w_slots`` plane), so no two rows share a write target -- the eager
        duplicate-write trick is off because pad tokens differ across rows.
        Reads rewire accordingly: a glue row's page table binds [base,
        base + W) to its own plane, and branch case c's chain-prefix entries
        point into glue row c-1's copies (page1 slot sharing -- the COW-free
        analogue of the paged branch-head copy).

        ``synced_len`` deliberately stays at the PRE-advance length: the glue
        rows' [base, base + W) entries now reference this round's transient
        w_slots (freed at round end), so the next round's shared broadcast
        must re-cover [base, ...) with committed slots. Entries past the next
        round's working set may keep pointing at freed ids -- consistent with
        the freshness discipline, nothing ever reads past the bound region.
        """
        width = self._extend_graph_width
        pool = self.model_runner.req_to_token_pool
        br_plane = variant.br_case * width + variant.br_j
        for i, (state, carrier) in enumerate(zip(states, carriers)):
            new_len = state.committed_slots.numel()
            base = base_lens[i]
            delta = new_len - base
            synced = min(carrier.synced_len, base)
            pool.req_to_token[carrier.all_rows, synced:new_len] = state.committed_slots[
                synced:new_len
            ].to(torch.int32)
            carrier.synced_len = base
            # Glue rows own their whole window (delta re-writes + chain + pads).
            pool.req_to_token[carrier.glue_rows, base : base + width] = w_slots[
                i, 1:
            ].to(torch.int32)
            # Branch case prefixes source the chain from the glue-row copies:
            # case c's entry j holds chain token c_{j+1}, which glue row c-1
            # (= w_slots plane c) wrote at its column delta + j.
            branch_rows = carrier.comb_rows_for(
                f_live=f_live, tri_g=self._tri_g, br_r_pool=variant.br_r_pool
            )[self._tri_g.numel() :]
            branch_vals = w_slots[i].reshape(-1)[br_plane + delta]
            pool.req_to_token[branch_rows, variant.br_j + new_len] = branch_vals.to(
                torch.int32
            )
        self.profiler.mark("carrier_sync")

    def _fused_gather_pattern(self, delta_len: int) -> list[int]:
        """Per-seat gather rows over the [delta | chain] source layout (length
        delta_len + K): the seat row takes the delta, glue row g takes the
        delta then chain[:g + 1]. The pattern depends only on the delta
        length, and fast-path deltas repeat a handful of lengths (<= K + 1),
        so the cache is effectively write-once."""
        pattern = self._fused_gather_patterns.get(delta_len)
        if pattern is None:
            delta_part = list(range(delta_len))
            pattern = list(delta_part)
            for g in range(self.num_steps):
                pattern.extend(delta_part)
                pattern.extend(range(delta_len, delta_len + g + 1))
            self._fused_gather_patterns[delta_len] = pattern
        return pattern

    def _fused_extend_forward(
        self,
        *,
        carriers: list[_SeatCarrier],
        chains: list[torch.Tensor],
        backbone_slots: Optional[torch.Tensor],  # [bs, K]; None at page > 1
        seat_reqs: list,
        seat_rows: torch.Tensor,  # [bs], the advance batch's pool rows
        seat_input_ids: torch.Tensor,  # flat [sum(delta_lens)], device
        seat_delta_slots: torch.Tensor,  # flat [sum(delta_lens)], device
        base_lens: list[int],
        delta_lens: list[int],
        graph_round: Optional[_ExtendGraphRound],
        prestaged: Optional[_ExtendCaseStaging] = None,
    ) -> torch.Tensor:
        """Advance + glue as ONE batched extend (the fast round's fusion).

        Per seat the rows are [seat, glue 0..K-1] (seat FIRST): the seat row
        extends the committed delta (last logits = node 0; on hybrid it also
        advances the seat's state slot in place), and glue row g extends
        delta + c_1..c_{g+1} (last logits = node g+1). Prepending the delta
        is forced by extend contiguity: the delta's KV does not exist before
        this forward, so it cannot sit in a glue row's prefix region.

        KV plumbing (full-attn layers): a glue row's delta positions point
        out_cache_loc at the SEAT's delta slots, and its chain positions at
        the shared backbone slots -- so K+1 rows write each delta slot and
        multiple rows write each backbone slot. Every writer produces the
        SAME values (same token, same absolute position, same prefix
        contents), which makes this the benign same-forward write-then-read
        pattern the glue triangle already relies on: per layer, the batched
        KV write precedes every attention read, so whichever duplicate lands
        is correct by the time any row reads it. At page_size > 1 the
        duplicate-write trick is off (see _fused_out_cache_loc_paged): each
        glue row targets its own private slots, so the extend layout is
        unchanged but no two rows ever write through a shared page.
        """
        graph_logits: Optional[torch.Tensor] = None
        if graph_round is not None:
            graph_logits = self._fused_extend_graph_forward(
                carriers=carriers,
                chains=chains,
                seat_reqs=seat_reqs,
                seat_rows=seat_rows,
                seat_input_ids=seat_input_ids,
                seat_delta_slots=seat_delta_slots,
                base_lens=base_lens,
                delta_lens=delta_lens,
                graph_round=graph_round,
                prestaged=prestaged,
            )
            if not envs.SGLANG_DEBUG_DECOUPLED_EXTEND_GRAPH_DIFF.get():
                return graph_logits
            # Diff probe: fall through to the eager fused body on the SAME
            # staged inputs and report per-node divergence. Safe to double
            # -run: both paths write identical values to every persistent
            # location (KV real slots, GDN seat/glue states), so the second
            # forward is a benign same-value rewrite; the caller consumes
            # the EAGER logits (correct-by-construction reference).
        num_steps = self.num_steps
        bs = len(carriers)
        rows_per_seat = num_steps + 1
        # Generic extend shell (rebind-only, row-count independent) -- the
        # same reuse discipline as _glue_forward.
        fused = self._glue_template
        fused.reqs = [
            req
            for i, carrier in enumerate(carriers)
            for req in [seat_reqs[i]] + carrier.glue_reqs
        ]
        # mrope models (Qwen3.5 family) index multimodal_inputs per row in
        # ForwardBatch.init_new; rebind to the fused row count (text-only).
        fused.multimodal_inputs = [None] * len(fused.reqs)
        fused.req_pool_indices = torch.cat(
            [
                rows
                for i, carrier in enumerate(carriers)
                for rows in (seat_rows[i : i + 1], carrier.glue_rows)
            ]
        )
        fused.extend_logprob_start_lens = [0] * (bs * rows_per_seat)
        fused.extend_lens = [
            row_len
            for i in range(bs)
            for row_len in [delta_lens[i]]
            + [delta_lens[i] + g + 1 for g in range(num_steps)]
        ]
        fused.prefix_lens = [
            base_lens[i] for i in range(bs) for _ in range(rows_per_seat)
        ]
        seq_host = [
            prefix + row_len
            for prefix, row_len in zip(fused.prefix_lens, fused.extend_lens)
        ]
        # Row contents: input ids and out_cache_loc share ONE gather. Per seat
        # the source is [delta | chain] (ids) aligned with
        # [delta slots | backbone slots] (locations), and the row layout
        # repeats the delta then takes a growing chain prefix. (At page > 1
        # write targets are per-row, so only the ids ride the gather; slots
        # come from _fused_out_cache_loc_paged.)
        src_id_pieces: list[torch.Tensor] = []
        src_slot_pieces: list[torch.Tensor] = []
        gather_rows: list[int] = []
        src_offset = 0
        delta_offset = 0
        for i in range(bs):
            delta_len = delta_lens[i]
            src_id_pieces.append(
                seat_input_ids[delta_offset : delta_offset + delta_len]
            )
            src_id_pieces.append(chains[i])
            if not self._paged:
                src_slot_pieces.append(
                    seat_delta_slots[delta_offset : delta_offset + delta_len]
                )
                src_slot_pieces.append(backbone_slots[i])
            gather_rows.extend(
                src_offset + entry for entry in self._fused_gather_pattern(delta_len)
            )
            src_offset += delta_len + num_steps
            delta_offset += delta_len
        gather = self._h2d.to_device(gather_rows, dtype=torch.int64)
        fused.input_ids = torch.cat(src_id_pieces)[gather]
        if self._paged:
            fused.out_cache_loc = self._fused_out_cache_loc_paged(
                carriers=carriers,
                seat_delta_slots=seat_delta_slots,
                base_lens=base_lens,
                delta_lens=delta_lens,
            )
        else:
            fused.out_cache_loc = torch.cat(src_slot_pieces)[gather]
        fused.extend_num_tokens = len(gather_rows)
        seq_cpu = torch.tensor(seq_host, dtype=torch.int64)
        fused.seq_lens = self._h2d.to_device(seq_cpu, dtype=torch.int64)
        fused.seq_lens_cpu = seq_cpu
        fused.seq_lens_sum = sum(seq_host)
        fused.orig_seq_lens = fused.seq_lens.to(torch.int32)
        self.profiler.mark("fused_mut")
        eager_logits = self._forward(fused, tag="fused_extend")
        if graph_logits is not None:
            self._log_extend_graph_diff(
                graph_logits=graph_logits,
                eager_logits=eager_logits,
                base_lens=base_lens,
                delta_lens=delta_lens,
            )
        return eager_logits

    def _log_extend_graph_diff(
        self,
        *,
        graph_logits: torch.Tensor,
        eager_logits: torch.Tensor,
        base_lens: list[int],
        delta_lens: list[int],
    ) -> None:
        """Diff-probe report (SGLANG_DEBUG_DECOUPLED_EXTEND_GRAPH_DIFF): one
        line per fused round with each node row's max-abs logit divergence
        and top-1 agreement. Host syncs are fine here -- diagnosis only."""
        diff = (graph_logits.float() - eager_logits.float()).abs().amax(dim=-1)
        top_same = graph_logits.argmax(dim=-1) == eager_logits.argmax(dim=-1)
        logger.info(
            "extend-graph diff: base=%s delta=%s max_abs=%s top1_same=%s",
            base_lens,
            delta_lens,
            [round(v, 4) for v in diff.tolist()],
            top_same.tolist(),
        )

    def _fused_out_cache_loc_paged(
        self,
        *,
        carriers: list[_SeatCarrier],
        seat_delta_slots: torch.Tensor,
        base_lens: list[int],
        delta_lens: list[int],
    ) -> torch.Tensor:
        """Fused-extend write targets at page_size > 1. The seat row keeps its
        allocator-grown delta slots (standard paged growth on its own pages);
        glue row g's delta + chain positions map to its OWN private slots
        [q0, q0 + dl + g + 1) with q0 = the seat's boundary-tail length --
        the page1 trick of duplicate-writing the seat's delta slots would
        route every glue row's writes through the seat's partial page, whose
        later positions hold divergent tokens across rows. The rows still
        recompute the delta redundantly (extend contiguity forces that); only
        the write targets move."""
        pieces: list[torch.Tensor] = []
        delta_offset = 0
        for i, carrier in enumerate(carriers):
            delta_len = delta_lens[i]
            q0 = base_lens[i] - self._page_floor(base_lens[i])
            pieces.append(seat_delta_slots[delta_offset : delta_offset + delta_len])
            for g in range(self.num_steps):
                pieces.append(
                    carrier.glue_private_slots[g, q0 : q0 + delta_len + g + 1]
                )
            delta_offset += delta_len
        return torch.cat(pieces)

    # ------------------------------------------------------------------ #
    # Fused-extend CUDA graph: per-round path
    # ------------------------------------------------------------------ #

    def _stage_extend_graph_round(
        self,
        *,
        bs: int,
        delta_lens: list[int],
        scratch_slots: list[torch.Tensor],
        scratch_kv_pages: list[torch.Tensor],
        rows_per_seat: int,
    ) -> Optional[_ExtendGraphRound]:
        """Decide graph vs eager for this round and stage the pad slots.

        ``rows_per_seat`` is K+1 for a fast round's fused shape (seat + glue
        rows) and 1 for a miss round's advance-only degenerate shape.

        Every check is host-static so the choice is final before any carrier
        state is written. Per-round fallback triggers: no runner (kill switch
        or construction failure), a delta outside the padded window
        (defensive; catch-up merges miss the glue fast path anyway, and an
        empty delta has no last real token to pad with), a row count over the
        captured buckets, or page exhaustion on the pad staging (the eager
        path needs fewer slots).
        """
        if self._extend_graph_runner is None:
            return None
        rows = bs * rows_per_seat
        if rows > self._extend_graph_max_rows:
            return None
        if (
            self._extend_graph_disable_padding
            and rows not in self._extend_graph_capture_rows
        ):
            return None
        if any(
            delta_len < 1 or delta_len > self.num_steps + 1 for delta_len in delta_lens
        ):
            return None
        if self._paged:
            # Arena pages, not a fresh allocator alloc: the pad tail is pure
            # per-round scratch, and recycling it through the arena is what
            # leaves a steady-state paged round with nothing to hand back to
            # the allocator (and so no device read) at the round tail.
            pad_pages = self._kv_page_arena.take_optional(
                bs * self._extend_graph_pad_pages
            )
            if pad_pages is None:
                return None
            scratch_kv_pages.append(pad_pages)
            return _ExtendGraphRound(
                w_slots=None,
                seat_pad_flats=self._page_flat_slots(
                    pad_pages.view(bs, self._extend_graph_pad_pages)
                ),
            )
        w_slots = self.model_runner.token_to_kv_pool_allocator.alloc(
            rows * self._extend_graph_width
        )
        if w_slots is None:
            return None
        # P == 1 only (the paged branch returned above): plain slot-list free.
        scratch_slots.append(w_slots)
        return _ExtendGraphRound(
            w_slots=w_slots.view(bs, rows_per_seat, self._extend_graph_width),
            seat_pad_flats=None,
        )

    def _extend_graph_gather_pattern(self, delta_len: int) -> list[int]:
        """Graph twin of _fused_gather_pattern over the same [delta | chain]
        source layout: every row padded to W by repeating its LAST REAL index
        -- a valid vocab token, so pad embeddings stay in range; their K/V
        and logits are computed and never read."""
        pattern = self._extend_graph_gather_patterns.get(delta_len)
        if pattern is None:
            width = self._extend_graph_width
            delta_part = list(range(delta_len))
            pattern = delta_part + [delta_len - 1] * (width - delta_len)
            for g in range(self.num_steps):
                row = delta_part + list(range(delta_len, delta_len + g + 1))
                pattern = pattern + row + [row[-1]] * (width - len(row))
            self._extend_graph_gather_patterns[delta_len] = pattern
        return pattern

    def _extend_graph_consts_for(self, rows: int) -> tuple[torch.Tensor, list[int]]:
        """(device int32 [rows] filled with W, host [W] * rows) -- the
        uniform-width constants the full-attn plane's spec_info carries;
        cached per row count (row counts recur across fast rounds)."""
        consts = self._extend_graph_const_cache.get(rows)
        if consts is None:
            consts = (
                torch.full(
                    (rows,),
                    self._extend_graph_width,
                    dtype=torch.int32,
                    device=self.device,
                ),
                [self._extend_graph_width] * rows,
            )
            self._extend_graph_const_cache[rows] = consts
        return consts

    def _extend_graph_seat_pads(
        self,
        *,
        seat: int,
        base_len: int,
        delta_len: int,
        graph_round: _ExtendGraphRound,
    ) -> torch.Tensor:
        """Seat row ``seat``'s W - delta pad slots: its own w_slots plane tail
        at page_size == 1, else the round's throwaway pad pages entered at the
        offset that keeps in-page offset == logical position offset (the
        invariant paged tables are derived from). Both the write targets and
        the page-table entries come from here, so the two can never disagree.
        """
        width = self._extend_graph_width
        if not self._paged:
            return graph_round.w_slots[seat, 0, delta_len:width]
        new_len = base_len + delta_len
        pad_offset = new_len - self._page_floor(new_len)
        return graph_round.seat_pad_flats[
            seat, pad_offset : pad_offset + width - delta_len
        ]

    def _extend_graph_out_cache_loc(
        self,
        *,
        carriers: list[_SeatCarrier],
        seat_delta_slots: torch.Tensor,
        base_lens: list[int],
        delta_lens: list[int],
        graph_round: _ExtendGraphRound,
    ) -> torch.Tensor:
        """W write slots per fused row (row order [seat, glue 0..K-1] per
        seat): the seat row keeps its real allocator delta slots and pads
        into engine-owned junk; glue rows write their WHOLE window into
        private slots (page1: this round's w_slots plane; page>1: the
        carrier's arena pages from the boundary-tail offset). Pad writes are
        junk by construction -- they land in slots only pad queries can ever
        resolve to."""
        width = self._extend_graph_width
        pieces: list[torch.Tensor] = []
        delta_offset = 0
        for i, carrier in enumerate(carriers):
            delta_len = delta_lens[i]
            pieces.append(seat_delta_slots[delta_offset : delta_offset + delta_len])
            pieces.append(
                self._extend_graph_seat_pads(
                    seat=i,
                    base_len=base_lens[i],
                    delta_len=delta_len,
                    graph_round=graph_round,
                )
            )
            if self._paged:
                tail = base_lens[i] - self._page_floor(base_lens[i])
                for g in range(self.num_steps):
                    pieces.append(carrier.glue_private_slots[g, tail : tail + width])
            else:
                pieces.append(graph_round.w_slots[i, 1:].reshape(-1))
            delta_offset += delta_len
        return torch.cat(pieces)

    def _extend_graph_bind_seat_pads(
        self,
        *,
        seat_rows: torch.Tensor,
        base_lens: list[int],
        delta_lens: list[int],
        graph_round: _ExtendGraphRound,
    ) -> None:
        """Map the seat row's pad positions [base + delta, base + W): the
        uniform post-write seq_len (base + W) makes paged backends resolve
        every window position through the page table, so pads must point at
        owned junk -- never left stale (the no-stale-mapping discipline) and
        never routed into a live page. The seat row itself is transient
        (freed with the advance batch), so these entries die with the round."""
        pool = self.model_runner.req_to_token_pool
        width = self._extend_graph_width
        for i in range(len(base_lens)):
            new_len = base_lens[i] + delta_lens[i]
            pads = self._extend_graph_seat_pads(
                seat=i,
                base_len=base_lens[i],
                delta_len=delta_lens[i],
                graph_round=graph_round,
            )
            pool.req_to_token[seat_rows[i], new_len : base_lens[i] + width] = pads.to(
                torch.int32
            )

    def _extend_graph_assemble_rows(
        self,
        *,
        carriers: list[_SeatCarrier],
        chains: list[torch.Tensor],
        seat_reqs: list,
        seat_rows: torch.Tensor,
        seat_input_ids: torch.Tensor,
        seat_delta_slots: torch.Tensor,
        base_lens: list[int],
        delta_lens: list[int],
        graph_round: _ExtendGraphRound,
        prestaged: Optional[_ExtendCaseStaging] = None,
    ) -> tuple[ScheduleBatch, list[int], list[int]]:
        """Bind the fused shell for a graph round (uniform W-token rows) and
        return (shell, per-row true lengths, per-node flat logit offsets).

        Same rows and row semantics as the eager fused extend -- per seat
        [seat(delta), glue g(delta + g + 1)], prefix = pre-write committed
        length -- only padded to the constant width and expressed in the
        DRAFT_EXTEND_V2 prepare contract (pre-write ScheduleBatch seq_lens;
        the caller bumps the ForwardBatch to post-write).
        """
        num_steps = self.num_steps
        width = self._extend_graph_width
        bs = len(carriers)
        rows_per_seat = num_steps + 1
        fused = self._glue_template
        fused.reqs = [
            req
            for i, carrier in enumerate(carriers)
            for req in [seat_reqs[i]] + carrier.glue_reqs
        ]
        # mrope models (Qwen3.5 family) index multimodal_inputs per row in
        # ForwardBatch.init_new; rebind to the fused row count (text-only).
        fused.multimodal_inputs = [None] * len(fused.reqs)
        fused.req_pool_indices = torch.cat(
            [
                rows
                for i, carrier in enumerate(carriers)
                for rows in (seat_rows[i : i + 1], carrier.glue_rows)
            ]
        )
        fused.extend_logprob_start_lens = [0] * (bs * rows_per_seat)
        _, width_list = self._extend_graph_consts_for(bs * rows_per_seat)
        fused.extend_lens = list(width_list)
        fused.prefix_lens = [
            base_lens[i] for i in range(bs) for _ in range(rows_per_seat)
        ]
        # Row contents: one padded gather over the [delta | chain] source per
        # seat; the true row lengths (GDN plane) and the node-logit end
        # offsets ride the same loop. Idle-prebuilt case staging (bs == 1)
        # short-circuits the whole build: only token VALUES flow in here.
        if prestaged is not None:
            fused.input_ids = torch.cat([seat_input_ids, chains[0]])[prestaged.gather]
            fused.out_cache_loc = prestaged.out_cache_loc
            true_lens_host = prestaged.true_lens_host
            node_offsets = prestaged.node_offsets
        else:
            src_id_pieces: list[torch.Tensor] = []
            gather_rows: list[int] = []
            true_lens_host = []
            node_offsets = []
            src_offset = 0
            delta_offset = 0
            for i in range(bs):
                delta_len = delta_lens[i]
                src_id_pieces.append(
                    seat_input_ids[delta_offset : delta_offset + delta_len]
                )
                src_id_pieces.append(chains[i])
                gather_rows.extend(
                    src_offset + entry
                    for entry in self._extend_graph_gather_pattern(delta_len)
                )
                row_base = i * rows_per_seat
                true_lens_host.append(delta_len)
                node_offsets.append(row_base * width + delta_len - 1)
                for g in range(num_steps):
                    true_lens_host.append(delta_len + g + 1)
                    node_offsets.append((row_base + 1 + g) * width + delta_len + g)
                src_offset += delta_len + num_steps
                delta_offset += delta_len
            gather = self._h2d.to_device(gather_rows, dtype=torch.int64)
            fused.input_ids = torch.cat(src_id_pieces)[gather]
            fused.out_cache_loc = self._extend_graph_out_cache_loc(
                carriers=carriers,
                seat_delta_slots=seat_delta_slots,
                base_lens=base_lens,
                delta_lens=delta_lens,
                graph_round=graph_round,
            )
        fused.extend_num_tokens = bs * rows_per_seat * width
        self._extend_graph_bind_seat_pads(
            seat_rows=seat_rows,
            base_lens=base_lens,
            delta_lens=delta_lens,
            graph_round=graph_round,
        )
        # DRAFT_EXTEND_V2 prepare contract: the ScheduleBatch view carries the
        # PRE-write lengths; orig stays post-write like the eager shell.
        seq_cpu = torch.tensor(fused.prefix_lens, dtype=torch.int64)
        fused.seq_lens = self._h2d.to_device(seq_cpu, dtype=torch.int64)
        fused.seq_lens_cpu = seq_cpu
        fused.seq_lens_sum = sum(fused.prefix_lens)
        fused.orig_seq_lens = (fused.seq_lens + width).to(torch.int32)
        return fused, true_lens_host, node_offsets

    def _fused_extend_graph_forward(
        self,
        *,
        carriers: list[_SeatCarrier],
        chains: list[torch.Tensor],
        seat_reqs: list,
        seat_rows: torch.Tensor,
        seat_input_ids: torch.Tensor,
        seat_delta_slots: torch.Tensor,
        base_lens: list[int],
        delta_lens: list[int],
        graph_round: _ExtendGraphRound,
        prestaged: Optional[_ExtendCaseStaging] = None,
    ) -> torch.Tensor:
        """The fused extend as ONE captured DRAFT_EXTEND_V2 graph replay.

        A captured graph replays a CONSTANT per-row query width, so every
        fused row is padded to W = 2K + 1 with repeats of its last real
        token. The dual-plane length contract that makes the pads safe:

        - FULL-ATTENTION plane: uniform W everywhere (extend_seq_lens == W,
          seq_lens == base + W post-write, per-row KV prefix re-derived as
          seq_lens - W == base). Pads are a SUFFIX of each row's window, so
          causality already isolates them: no real query position ever
          attends a pad position, per-position ops never mix positions, and
          the node-logit gather below touches only real end positions. Pad
          K/V writes land in row-owned junk slots
          (_extend_graph_out_cache_loc), pad logits are garbage nobody reads.
        - GDN (recurrent) plane: a pad folded into a row's slot state would
          corrupt it permanently, so the linear plane receives the TRUE
          per-row lengths through spec_info.gdn_true_extend_lens_tensor,
          refreshed into a graph-static device cu_seqlens at every replay
          prep -- the GDN kernels read their loop bounds from device tensors,
          which is what lets variable true lengths replay inside one
          fixed-shape graph (GDNAttnBackend._forward_draft_extend_v2).

        Returns node logits [bs * (K+1), vocab] gathered from the full
        bs * (K+1) * W replay logits at each row's last REAL position -- the
        same contract as the eager fused forward's per-row last logits.
        """
        fused, true_lens_host, node_offsets = self._extend_graph_assemble_rows(
            carriers=carriers,
            chains=chains,
            seat_reqs=seat_reqs,
            seat_rows=seat_rows,
            seat_input_ids=seat_input_ids,
            seat_delta_slots=seat_delta_slots,
            base_lens=base_lens,
            delta_lens=delta_lens,
            graph_round=graph_round,
            prestaged=prestaged,
        )
        return self._extend_graph_replay(
            fused=fused,
            rows=len(carriers) * (self.num_steps + 1),
            true_lens_host=true_lens_host,
            node_offsets=node_offsets,
            base_lens=base_lens,
            delta_lens=delta_lens,
            tag="fused_graph",
            prestaged=prestaged,
        )

    def _extend_graph_replay(
        self,
        *,
        fused: ScheduleBatch,
        rows: int,
        true_lens_host: list[int],
        node_offsets: list[int],
        base_lens: list[int],
        delta_lens: list[int],
        tag: str,
        prestaged: Optional[_ExtendCaseStaging] = None,
    ) -> torch.Tensor:
        """Replay one assembled DRAFT_EXTEND_V2 shell (fused rows or the
        advance-only degenerate shape) and gather the node logits.

        Carries the dual-plane length contract described by
        ``_fused_extend_graph_forward``: the full-attn plane sees the uniform
        padded width W everywhere, the recurrent plane the TRUE per-row scan
        lengths. Row counts below the smallest captured bucket replay padded;
        the runner points those rows at reserved pool row 0 and the GDN plane
        scans them through its junk state slot, so they cannot touch any real
        row's state. ``base_lens`` / ``delta_lens`` are debug-probe context
        only.
        """
        width = self._extend_graph_width
        width_tensor, width_list = self._extend_graph_consts_for(rows)
        spec_info = EagleDraftExtendInput(
            hidden_states=None,  # STANDALONE: the enum drafter feeds token ids only
            num_correct_drafts=width_tensor,
            num_accept_tokens=width_tensor,
            num_tokens_per_req=width,
        )
        # Ad-hoc spec_info attrs, preset before every use (read directly
        # downstream, never defensively): the full-attn plane's uniform width
        # and the GDN plane's true per-row scan lengths.
        spec_info.extend_seq_lens_tensor = width_tensor
        spec_info.extend_seq_lens_cpu = list(width_list)
        if envs.SGLANG_DEBUG_DECOUPLED_EXTEND_GRAPH_FULLSCAN.get():
            # Bisect probe: no pad half-rows at all -- every row scans its
            # full W window on the GDN plane. States and guesses go WRONG
            # (pads fold into row-end states); only for localizing NaN.
            true_lens_host = [width] * rows
        if (
            prestaged is not None
            and not envs.SGLANG_DEBUG_DECOUPLED_EXTEND_GRAPH_FULLSCAN.get()
        ):
            spec_info.gdn_true_extend_lens_tensor = prestaged.true_lens
        else:
            spec_info.gdn_true_extend_lens_tensor = self._h2d.to_device(
                true_lens_host, dtype=torch.int32
            )
        self.profiler.mark(f"{tag}_mut")
        try:
            fused.forward_mode = ForwardMode.DRAFT_EXTEND_V2
            fused.spec_info = spec_info
            forward_batch = ForwardBatch.init_new(
                fused, self.model_runner, return_hidden_states_before_norm=False
            )
        finally:
            # The shell is shared with the eager fused / glue paths; restore
            # its extend identity immediately (the ForwardBatch carries its
            # own references).
            fused.forward_mode = ForwardMode.EXTEND
            fused.spec_info = None
        # v2 prepare contract: the forward sees POST-write lengths (the
        # extend writes W slots per row); mutation stays on the ForwardBatch.
        forward_batch.seq_lens = forward_batch.seq_lens + width
        forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu + width
        forward_batch.seq_lens_sum = fused.seq_lens_sum + rows * width
        runner = self._extend_graph_runner
        assert runner.can_run_graph(forward_batch), (
            "fused-extend graph precheck passed but the runner refused the "
            f"batch (rows={rows})"
        )
        if forward_batch.mrope_positions is not None:
            # The runner's replay feeds positions but not the mrope plane
            # (EAGLE draft models are not mrope models); feed the captured
            # buffer here so Qwen3.5-family drafters read real rotary
            # positions. Text-only rows: mrope == positions replicated x3.
            runner.buffers.mrope_positions[
                :, : forward_batch.mrope_positions.shape[1]
            ].copy_(forward_batch.mrope_positions)
        logits_output = runner.execute(forward_batch)
        self.profiler.mark(f"{tag}_fwd")
        if envs.SGLANG_DEBUG_DECOUPLED_EXTEND_GRAPH_NANCHECK.get():
            # Observation-only probe (no second forward, so no recurrent
            # -state double-advance): per row, which of the W window
            # positions carry NaN logits. Pad positions are expected
            # garbage; NaN at a REAL position (< true_len) is the defect.
            window = logits_output.next_token_logits[: rows * width].view(
                rows, width, -1
            )
            nan_mask = window.isnan().any(dim=-1).tolist()
            logger.info(
                "extend-graph nancheck: base=%s delta=%s true=%s nan_rows=%s",
                base_lens,
                delta_lens,
                true_lens_host,
                ["".join("N" if p else "." for p in row) for row in nan_mask],
            )
            self._log_extend_graph_attn_metadata(rows=rows)
        node_gather = (
            prestaged.node_gather
            if prestaged is not None
            else self._h2d.to_device(node_offsets, dtype=torch.int64)
        )
        # The gather copies out of the runner's private static logits buffer
        # before any later replay could overwrite it.
        return logits_output.next_token_logits[node_gather]

    def _advance_graph_assemble_rows(
        self,
        *,
        seat_reqs: list,
        seat_rows: torch.Tensor,
        seat_input_ids: torch.Tensor,
        seat_delta_slots: torch.Tensor,
        base_lens: list[int],
        delta_lens: list[int],
        graph_round: _ExtendGraphRound,
    ) -> tuple[ScheduleBatch, list[int], list[int]]:
        """Bind the fused shell for an ADVANCE-ONLY graph round: ONE row per
        seat (its committed delta, padded to W), no glue plane.

        Row semantics, write targets and both planes' length contract are the
        fast round's seat row verbatim -- only the glue rows are absent -- so
        the delta's KV lands in the same real allocator slots and a hybrid
        seat's state slot advances by exactly the delta. Returns (shell,
        per-row true lengths, per-row node-0 flat logit offsets).
        """
        width = self._extend_graph_width
        bs = len(base_lens)
        fused = self._glue_template
        fused.reqs = list(seat_reqs)
        # mrope models (Qwen3.5 family) index multimodal_inputs per row in
        # ForwardBatch.init_new; rebind to this round's row count (text-only).
        fused.multimodal_inputs = [None] * bs
        fused.req_pool_indices = seat_rows
        fused.extend_logprob_start_lens = [0] * bs
        _, width_list = self._extend_graph_consts_for(bs)
        fused.extend_lens = list(width_list)
        fused.prefix_lens = list(base_lens)
        gather_rows: list[int] = []
        node_offsets: list[int] = []
        slot_pieces: list[torch.Tensor] = []
        delta_offset = 0
        for i, delta_len in enumerate(delta_lens):
            # The fused gather pattern's FIRST row IS the seat row (the delta
            # padded with repeats of its last real index); the glue rows this
            # round does not have follow it.
            gather_rows.extend(
                delta_offset + entry
                for entry in self._extend_graph_gather_pattern(delta_len)[:width]
            )
            node_offsets.append(i * width + delta_len - 1)
            slot_pieces.append(
                seat_delta_slots[delta_offset : delta_offset + delta_len]
            )
            slot_pieces.append(
                self._extend_graph_seat_pads(
                    seat=i,
                    base_len=base_lens[i],
                    delta_len=delta_len,
                    graph_round=graph_round,
                )
            )
            delta_offset += delta_len
        gather = self._h2d.to_device(gather_rows, dtype=torch.int64)
        fused.input_ids = seat_input_ids[gather]
        fused.out_cache_loc = torch.cat(slot_pieces)
        fused.extend_num_tokens = bs * width
        self._extend_graph_bind_seat_pads(
            seat_rows=seat_rows,
            base_lens=base_lens,
            delta_lens=delta_lens,
            graph_round=graph_round,
        )
        # DRAFT_EXTEND_V2 prepare contract: the ScheduleBatch view carries the
        # PRE-write lengths; orig stays post-write like the eager shell.
        seq_cpu = torch.tensor(base_lens, dtype=torch.int64)
        fused.seq_lens = self._h2d.to_device(seq_cpu, dtype=torch.int64)
        fused.seq_lens_cpu = seq_cpu
        fused.seq_lens_sum = sum(base_lens)
        fused.orig_seq_lens = (fused.seq_lens + width).to(torch.int32)
        return fused, list(delta_lens), node_offsets

    def _advance_forward(
        self,
        *,
        states: list[_DraftReqState],
        base_lens: list[int],
        scratch_batches: list[ScheduleBatch],
        scratch_slots: list[torch.Tensor],
        scratch_kv_pages: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the seats' committed delta on its own (the miss round's
        first phase; a fast round folds the same work into the fused extend).
        Returns (node-0 logits [bs, vocab], the delta's new KV slots).

        The advance rides the SAME captured DRAFT_EXTEND_V2 graph as a fast
        round's fused extend, in its no-glue degenerate shape (rows = seats,
        one W-padded delta window each). Staging happens before any KV is
        written so the choice is final for the round; any check failing keeps
        the eager extend forward.
        """
        delta_lens = [
            len(state.committed_tokens) - base_lens[i] for i, state in enumerate(states)
        ]
        graph_round = self._stage_extend_graph_round(
            bs=len(states),
            delta_lens=delta_lens,
            scratch_slots=scratch_slots,
            scratch_kv_pages=scratch_kv_pages,
            rows_per_seat=1,
        )
        advance_batch, advance_slots = self._extend_batch(
            token_lists=[state.committed_tokens for state in states],
            prefix_slots=[state.committed_slots for state in states],
            tag="advance",
            mamba_slots=self._seat_mamba_slots(states),
        )
        scratch_batches.append(advance_batch)
        if graph_round is None:
            return self._forward(advance_batch, tag="advance"), advance_slots
        node0_logits = self._advance_graph_forward(
            seat_reqs=advance_batch.reqs,
            seat_rows=advance_batch.req_pool_indices,
            seat_input_ids=advance_batch.input_ids,
            seat_delta_slots=advance_slots,
            base_lens=base_lens,
            delta_lens=delta_lens,
            graph_round=graph_round,
        )
        return node0_logits, advance_slots

    def _advance_graph_forward(
        self,
        *,
        seat_reqs: list,
        seat_rows: torch.Tensor,
        seat_input_ids: torch.Tensor,
        seat_delta_slots: torch.Tensor,
        base_lens: list[int],
        delta_lens: list[int],
        graph_round: _ExtendGraphRound,
    ) -> torch.Tensor:
        """A miss round's seat advance as ONE captured DRAFT_EXTEND_V2 replay.

        Reuses the fast round's graph (zero extra capture) in its degenerate
        no-glue shape: rows = seats, each the seat's delta padded to W. Returns
        node-0 logits [bs, vocab] -- the same contract as the eager advance's
        per-row last logits.
        """
        fused, true_lens_host, node_offsets = self._advance_graph_assemble_rows(
            seat_reqs=seat_reqs,
            seat_rows=seat_rows,
            seat_input_ids=seat_input_ids,
            seat_delta_slots=seat_delta_slots,
            base_lens=base_lens,
            delta_lens=delta_lens,
            graph_round=graph_round,
        )
        return self._extend_graph_replay(
            fused=fused,
            rows=len(base_lens),
            true_lens_host=true_lens_host,
            node_offsets=node_offsets,
            base_lens=base_lens,
            delta_lens=delta_lens,
            tag="advance_graph",
        )

    def _glue_forward(
        self,
        *,
        carriers: list[_SeatCarrier],
        states: list[_DraftReqState],
        chains: list[torch.Tensor],
        backbone_slots: Optional[torch.Tensor],
        base_lens: list[int],
    ) -> torch.Tensor:
        num_steps = self.num_steps
        bs = len(states)
        glue = self._glue_template
        # Assemble the subset's rows onto the shared shell (rebind-only).
        glue.reqs = [req for carrier in carriers for req in carrier.glue_reqs]
        # mrope models (Qwen3.5 family) index multimodal_inputs per row in
        # ForwardBatch.init_new; the shell's build-time list goes stale as the
        # row count changes, so rebind it (always text-only rows here).
        glue.multimodal_inputs = [None] * len(glue.reqs)
        glue.req_pool_indices = (
            torch.cat([carrier.glue_rows for carrier in carriers])
            if bs > 1
            else carriers[0].glue_rows
        )
        glue.extend_logprob_start_lens = [0] * (bs * num_steps)
        lens = [state.committed_slots.numel() for state in states]
        seq_host = [lens[i] + g + 1 for i in range(bs) for g in range(num_steps)]
        seq_cpu = torch.tensor(seq_host, dtype=torch.int64)
        if self._hybrid or self._paged:
            # GDN extend scans a row's extend tokens from its OWN slot's state
            # and cannot see node states produced by sibling rows of the same
            # forward, so the one-token stagger doesn't work for the linear
            # layers: row g instead re-scans the whole chain prefix c_1..c_{g+1}
            # from its forked node-0 copy (K(K+1)/2 redundant token-updates --
            # the price of keeping glue ONE forward). Full-attn KV is unchanged
            # semantically: every row writes token c_{j+1}'s K/V to the SAME
            # backbone slot j, and the values are identical across rows (same
            # token, position, and shared prefix), so the duplicate-index
            # scatter is benign. page_size > 1 (pure-KV included) needs the
            # same triangle for a different reason: the stagger reads sibling
            # rows' fresh chain KV through shared slot entries, but a paged
            # page-table entry maps P positions to ONE page and cannot stitch
            # per-token slots from different rows' private pages -- each row
            # must own (and hence re-extend) its whole chain prefix.
            glue.extend_lens = self._tri_row_lens * bs
            glue.extend_num_tokens = bs * len(self._tri_j)
            chains_stack = torch.stack(chains)
            glue.input_ids = chains_stack[:, self._tri_j_dev].reshape(-1)
            if self._paged:
                # Row g's chain positions [new_len, new_len + g + 1) live at
                # its private slots [q0, q0 + g + 1); q0 = boundary tail +
                # delta (the advance ran before this forward, so the anchor
                # is the PRE-advance page floor and the head was COW'd with
                # tail + delta by _cow_carrier_glue_heads).
                pieces: list[torch.Tensor] = []
                for i, carrier in enumerate(carriers):
                    q0 = lens[i] - self._page_floor(base_lens[i])
                    for g in range(num_steps):
                        pieces.append(carrier.glue_private_slots[g, q0 : q0 + g + 1])
                glue.out_cache_loc = torch.cat(pieces)
            else:
                glue.out_cache_loc = backbone_slots[:, self._tri_j_dev].reshape(-1)
            glue.prefix_lens = [lens[i] for i in range(bs) for _ in range(num_steps)]
        else:
            glue.extend_lens = [1] * (bs * num_steps)
            glue.extend_num_tokens = bs * num_steps
            glue.input_ids = torch.cat(chains) if bs > 1 else chains[0]
            glue.out_cache_loc = backbone_slots.view(-1)
            glue.prefix_lens = [s - 1 for s in seq_host]
        glue.seq_lens = self._h2d.to_device(seq_cpu, dtype=torch.int64)
        glue.seq_lens_cpu = seq_cpu
        glue.seq_lens_sum = sum(seq_host)
        glue.orig_seq_lens = glue.seq_lens.to(torch.int32)
        self.profiler.mark("glue_mut")
        return self._forward(glue, tag="glue")

    def _branch_decode_chain(
        self,
        *,
        carriers: list[_SeatCarrier],
        states: list[_DraftReqState],
        branch_guesses: torch.Tensor,
        backbone_slots: Optional[torch.Tensor],
        scratch_slots: list[torch.Tensor],
        variant: _FanoutVariant,
    ) -> list[torch.Tensor]:
        """Run the (case, guess) chains as K decode replays on the seats'
        selected carrier rows. ``branch_guesses`` holds each selected row's
        first token in the batch's row order (case-major, budgeted columns).

        Everything the row SELECTION determines -- the Req stubs, the pool-row
        gather, the mrope row list -- is round-invariant per (seat, width) and
        comes from the seats' caches; only the seq-len family is rebuilt (the
        rebind-only discipline of _glue_template).
        """
        num_steps = self.num_steps
        branch = self._branch_template
        sels = [carrier.branch_sel_for(variant=variant) for carrier in carriers]
        if len(sels) == 1:
            branch.reqs, branch.req_pool_indices = sels[0]
        else:
            branch.reqs = [req for reqs, _ in sels for req in reqs]
            branch.req_pool_indices = torch.cat([rows for _, rows in sels])
        # See _glue_forward: mrope models index this list per row.
        branch.multimodal_inputs = [None] * len(branch.reqs)
        lens = [state.committed_slots.numel() for state in states]
        seq_host = [
            base_len + case for base_len in lens for case in variant.case_of_row
        ]
        seq_cpu = torch.tensor(seq_host, dtype=torch.int64)
        branch.seq_lens = self._h2d.to_device(seq_cpu, dtype=torch.int64)
        branch.seq_lens_cpu = seq_cpu
        branch.seq_lens_sum = None
        branch.orig_seq_lens = branch.seq_lens.to(torch.int32)
        cascade = self._build_branch_cascade(
            states=states,
            lens=lens,
            backbone_slots=backbone_slots,
            variant=variant,
        )
        self.profiler.mark("branch_mut")
        if self._chain_plan and cascade is None:
            # Whole-chain staging: allocation, seq-lens family and the K
            # ForwardBatches are built in one prep pass; the step loop only
            # rebinds the input tokens and replays. (The fa3 cascade path
            # keeps the per-step loop: its tail state advances in place.)
            with self.profiler.stage("branch-plan"):
                plan_steps = self._prepare_chain_steps(
                    branch, first_tokens=branch_guesses.reshape(-1)
                )
            for step, (_, step_slots) in enumerate(plan_steps):
                self._track_scratch_slots(
                    scratch_slots,
                    slots=step_slots,
                    positions=[seq + step for seq in seq_host],
                )
            first_tokens = branch_guesses.reshape(-1)
            rows = int(first_tokens.numel())
            if self._chain_graph is not None:
                with self.profiler.stage("branch-chain-graph"):
                    if self._chain_graph.can_replay(rows):
                        return self._chain_graph.replay(
                            rows=rows,
                            plan_steps=plan_steps,
                            first_tokens=first_tokens,
                        )
                    captured = self._chain_graph.try_capture_and_run(
                        rows=rows,
                        plan_steps=plan_steps,
                        first_tokens=first_tokens,
                    )
                    if captured is not None:
                        return captured
            tokens = first_tokens
            chain_steps: list[torch.Tensor] = []
            for step, (fb, _) in enumerate(plan_steps):
                with self.profiler.stage("branch-step-fwd"):
                    fb.input_ids = tokens
                    logits = self.model_runner.forward(fb).logits_output
                tokens = logits.next_token_logits.argmax(dim=-1)
                chain_steps.append(tokens)
            return chain_steps
        logits, step_slots = self._decode_step(
            branch, branch_guesses.reshape(-1), tag="branch", cascade=cascade
        )
        # Decode step s fills each row's position seq_host[row] + s.
        self._track_scratch_slots(scratch_slots, slots=step_slots, positions=seq_host)
        chain_steps = [logits.argmax(dim=-1)]
        for step in range(1, num_steps):
            logits, step_slots = self._decode_step(
                branch, chain_steps[-1], tag="branch", cascade=cascade
            )
            self._track_scratch_slots(
                scratch_slots,
                slots=step_slots,
                positions=[seq + step for seq in seq_host],
            )
            chain_steps.append(logits.argmax(dim=-1))
        return chain_steps

    def _prepare_chain_steps(
        self, branch: ScheduleBatch, *, first_tokens: torch.Tensor
    ) -> list[tuple[ForwardBatch, torch.Tensor]]:
        """Stage every chain decode step up front.

        Each iteration runs exactly today's per-step mutation
        (``prepare_for_decode``: allocation + table write + seq-lens family)
        and snapshots it into a ForwardBatch -- so the allocator sequence and
        page continuation are byte-identical to the step-by-step loop, just
        hoisted out of the replay gaps. Later steps' page-table entries being
        written early is fine: attention reads at most seq_len positions.
        """
        steps: list[tuple[ForwardBatch, torch.Tensor]] = []
        branch.input_ids = first_tokens.to(torch.int64)
        for _ in range(self.num_steps):
            branch.prepare_for_decode()
            fb = ForwardBatch.init_new(
                branch, self.model_runner, return_hidden_states_before_norm=False
            )
            steps.append((fb, branch.out_cache_loc))
        return steps

    def _build_branch_cascade(
        self,
        *,
        states: list[_DraftReqState],
        lens: list[int],
        backbone_slots: Optional[torch.Tensor],
        variant: _FanoutVariant,
    ) -> Optional[_CascadeMetadata]:
        """Shared-prefix cascade inputs for this round's branch chain, or None
        below the L2 threshold (where per-row re-reads are effectively free
        and the two-call split only adds overhead)."""
        if self._paged or backbone_slots is None:
            # The cascade splits each row into shared-backbone-slot prefix +
            # token-granular private tail for the fa3 cascade kernel; at
            # page_size > 1 -- and on page1 GRAPH rounds, where every fused
            # row owns private window copies -- there ARE no shared backbone
            # slots, so the dedup the cascade exists for is gone.
            return None
        min_prefix = self._cascade_min_prefix_len
        if min_prefix <= 0 or min(lens) < min_prefix:
            return None
        seats = len(states)
        rows_per_seat = len(variant.sel_rows_pool)
        prefix_page_table = torch.zeros(
            (seats, max(lens)), dtype=torch.int32, device=self.device
        )
        for i, state in enumerate(states):
            prefix_page_table[i, : lens[i]] = state.committed_slots.to(torch.int32)
        tail_page_table = torch.zeros(
            (seats * rows_per_seat, 2 * self.num_steps + 2),
            dtype=torch.int32,
            device=self.device,
        )
        for i in range(seats):
            block = tail_page_table[i * rows_per_seat : (i + 1) * rows_per_seat]
            block[variant.br_r_sel, variant.br_j] = backbone_slots[i].to(torch.int32)[
                variant.br_j
            ]
        tail_lens = torch.tensor(
            [case for _ in states for case in variant.case_of_row],
            dtype=torch.int32,
            device=self.device,
        )
        return _CascadeMetadata(
            prefix_page_table=prefix_page_table,
            prefix_lens=self._h2d.to_device(lens, dtype=torch.int32),
            tail_page_table=tail_page_table,
            tail_lens=tail_lens,
            row_indices=torch.arange(
                seats * rows_per_seat, dtype=torch.int64, device=self.device
            ),
        )

    def prebuild_fast_round(self, keys: list[DraftReqKey]) -> None:
        """Idle-window build of the next fast round's allocation + batch
        skeleton (see _PrebuiltFastRound). No-op when one is already staged,
        when the fused-extend graph path is unavailable, or when any key has
        no carrier yet (those seats bootstrap, not fast-round)."""
        if (
            not self._prep_ahead
            or self._prebuilt_fast is not None
            or not self._enable_fused_extend
            or self._extend_graph_runner is None
        ):
            return
        keys = [k for k in keys if k in self._states and k in self._seat_carriers]
        if not keys:
            return
        states = [self._states[k] for k in keys]
        bs = len(states)
        width = self.num_steps + 1
        base_lens = [state.committed_slots.numel() for state in states]
        pre_batches: list = []
        pre_slots: list = []
        pre_pages: list = []
        # The skeleton's GPU work (glue COW / mamba fork) READS state this
        # round's main-stream kernels just wrote (seat post-absorb state,
        # committed KV); order the prebuild stream after them first.
        self._prebuild_stream.wait_stream(torch.cuda.current_stream())
        stream_cm = torch.cuda.stream(self._prebuild_stream)
        stream_cm.__enter__()
        try:
            graph_round = self._stage_extend_graph_round(
                bs=bs,
                delta_lens=[width] * bs,
                scratch_slots=pre_slots,
                scratch_kv_pages=pre_pages,
                rows_per_seat=width,
            )
            if graph_round is None:
                self._scrap_lists(pre_batches, pre_slots, pre_pages)
                return
            batch, slots = self._extend_batch(
                token_lists=[
                    list(state.committed_tokens) + [0] * width for state in states
                ],
                prefix_slots=[state.committed_slots for state in states],
                tag="preadvance",
                mamba_slots=self._seat_mamba_slots(states),
            )
        except Exception:
            stream_cm.__exit__(None, None, None)
            logger.exception("decoupled prep-ahead build failed; falling back inline")
            self._scrap_lists(pre_batches, pre_slots, pre_pages)
            return
        carriers = [self._seat_carriers[k] for k in keys]
        if self._paged:
            self._cow_carrier_glue_heads(
                states=states, carriers=carriers, base_lens=base_lens
            )
        if self._hybrid:
            self._fork_mamba_states(
                src_slots=torch.cat(
                    [state.mamba_slot.repeat(self.num_steps) for state in states]
                ),
                dst_slots=torch.cat([carrier.glue_mamba_slots for carrier in carriers]),
            )
        case_staging = None
        if bs == 1:
            # The gather / true-lens / node-gather tensors are ROUND-INVARIANT
            # (pure functions of delta_len and the static graph shape): built
            # once per delta_len and cached, so a restage does no H2D at all
            # -- which is what lets it run at round tail with the stream
            # still busy. Only out_cache_loc varies per round, and it is
            # composed on-GPU (sync-free).
            case_staging = []
            chunk = slots.view(width)
            for delta_len in range(1, width + 1):
                static = self._case_staging_static.get(delta_len)
                if static is None:
                    pattern = self._extend_graph_gather_pattern(delta_len)
                    true_host = [delta_len] + [
                        delta_len + g + 1 for g in range(self.num_steps)
                    ]
                    w = self._extend_graph_width
                    node_off = [delta_len - 1] + [
                        (1 + g) * w + delta_len + g for g in range(self.num_steps)
                    ]
                    static = (
                        torch.tensor(pattern, dtype=torch.int64, device=self.device),
                        torch.tensor(true_host, dtype=torch.int32, device=self.device),
                        true_host,
                        torch.tensor(node_off, dtype=torch.int64, device=self.device),
                        node_off,
                    )
                    self._case_staging_static[delta_len] = static
                gather, true_lens, true_host, node_gather, node_off = static
                case_staging.append(
                    _ExtendCaseStaging(
                        gather=gather,
                        out_cache_loc=self._extend_graph_out_cache_loc(
                            carriers=carriers,
                            seat_delta_slots=chunk[:delta_len],
                            base_lens=base_lens,
                            delta_lens=[delta_len],
                            graph_round=graph_round,
                        ),
                        true_lens=true_lens,
                        true_lens_host=true_host,
                        node_gather=node_gather,
                        node_offsets=node_off,
                    )
                )
        positions: list[int] = []
        for base in base_lens:
            positions.extend(range(base, base + width))
        ready_event = torch.cuda.Event()
        ready_event.record(self._prebuild_stream)
        stream_cm.__exit__(None, None, None)
        self._prebuilt_fast = _PrebuiltFastRound(
            keys=tuple(keys),
            base_lens=base_lens,
            batch=batch,
            slots=slots,
            slot_positions=positions,
            graph_round=graph_round,
            scratch_batches=pre_batches,
            scratch_slots=pre_slots,
            scratch_kv_pages=pre_pages,
            glue_seeded=True,
            case_staging=case_staging,
            ready_event=ready_event,
        )

    def _scrap_lists(self, batches: list, slots_list: list, kv_pages: list) -> None:
        """Immediate (outside-a-round) release of prebuilt resources. Slot
        tensors must already be page-head filtered (_track_scratch_slots)."""
        for batch in batches:
            for req in batch.reqs:
                if req.req_pool_idx is not None:
                    req.mamba_pool_idx = None
                    self.model_runner.req_to_token_pool.free(req)
        self._pending_scratch_frees.extend(
            s for s in slots_list if s is not None and s.numel() > 0
        )
        for pages in kv_pages:
            self._kv_page_arena.give_back(pages)

    def _scrap_prebuilt_into(
        self,
        pre: _PrebuiltFastRound,
        scratch_batches: list,
        scratch_slots: list,
        scratch_kv_pages: list,
    ) -> None:
        """Fold an unusable prebuilt round into the CURRENT round's scratch
        lists (freed at its tail, page-head rule preserved)."""
        scratch_batches.append(pre.batch)
        scratch_batches.extend(pre.scratch_batches)
        scratch_slots.extend(pre.scratch_slots)
        scratch_kv_pages.extend(pre.scratch_kv_pages)
        self._track_scratch_slots(
            scratch_slots, slots=pre.slots, positions=pre.slot_positions
        )

    def drop_prebuilt_for(self, key: DraftReqKey) -> None:
        """Seat close/reopen invalidation: release the prebuilt round now."""
        pre = self._prebuilt_fast
        if pre is None or key not in pre.keys:
            return
        self._prebuilt_fast = None
        tracked: list = []
        self._track_scratch_slots(
            tracked, slots=pre.slots, positions=pre.slot_positions
        )
        self._scrap_lists(
            [pre.batch] + pre.scratch_batches,
            pre.scratch_slots + tracked,
            pre.scratch_kv_pages,
        )

    def _consume_prebuilt(
        self,
        pre: _PrebuiltFastRound,
        *,
        states: list[_DraftReqState],
        base_lens: list[int],
        delta_lens: list[int],
        scratch_slots: list,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compact the hypothesized-max allocation to the actual deltas and
        stage the real token values. Returns (advance_slots, seat_input_ids)."""
        width = self.num_steps + 1
        chunks = pre.slots.view(len(states), width)
        pieces: list[torch.Tensor] = []
        delta_tokens: list[int] = []
        for i, (state, delta_len) in enumerate(zip(states, delta_lens)):
            pieces.append(chunks[i, :delta_len])
            if delta_len < width:
                self._track_scratch_slots(
                    scratch_slots,
                    slots=chunks[i, delta_len:],
                    positions=list(
                        range(base_lens[i] + delta_len, base_lens[i] + width)
                    ),
                )
            delta_tokens.extend(state.committed_tokens[-delta_len:])
        advance_slots = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
        seat_input_ids = self._h2d.to_device(delta_tokens, dtype=torch.int64)
        return advance_slots, seat_input_ids

    def _absorb_advance_slots(
        self, states: list[_DraftReqState], advance_slots: torch.Tensor
    ) -> None:
        """Newly written KV joins the committed prefix (kept across rounds)."""
        offset = 0
        for state in states:
            new_len = len(state.committed_tokens) - state.committed_slots.numel()
            state.committed_slots = torch.cat(
                [state.committed_slots, advance_slots[offset : offset + new_len]]
            )
            offset += new_len
        self.profiler.mark("commit_slots")

    # ------------------------------------------------------------------ #
    # Paged KV (page_size > 1): geometry, private-page COW
    # ------------------------------------------------------------------ #

    def _page_floor(self, num_tokens: int) -> int:
        return num_tokens - num_tokens % self._page_size

    def _page_roundup(self, num_tokens: int) -> int:
        return -(-num_tokens // self._page_size) * self._page_size

    def _page_flat_slots(self, pages: torch.Tensor) -> torch.Tensor:
        """[rows, n] page ids -> [rows, n * P] flat slot ids. Column q holds
        in-page offset q % P, so anchoring a row's private region at a page
        floor makes slot offset == logical position offset -- the invariant
        paged attention backends derive page tables from, and what lets
        alloc_extend / alloc_decode continue a private page in place."""
        offsets = torch.arange(self._page_size, dtype=torch.int64, device=pages.device)
        return (pages.unsqueeze(-1) * self._page_size + offsets).reshape(
            pages.shape[0], -1
        )

    def _resolve_cow_kv_pool(self, *, model_runner: ModelRunner) -> MHATokenToKVPool:
        """Full-attention K/V pool for the boundary/branch-head COW copies.

        Hybrid (GDN) models only have KV for their full-attn layers, wrapped
        behind ``HybridLinearKVPool.full_kv_pool`` (linear-layer state is
        forked through the mamba arena instead); MLA pools have no (k, v)
        slot rows to move, so a paged MLA drafter fails here at launch rather
        than as mid-round corruption. ``move_kv_cache`` is the layout-aware
        copy primitive (HND / page-major / native / fused-tiled); its plain
        NHD tiled path needs a copy config the configurator only arms for
        colocated spec (the drafter role clears speculative_algorithm), so
        arm it here exactly when that base implementation is the active one.
        """
        kvcache = model_runner.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kvcache, HybridLinearKVPool):
            kvcache = kvcache.full_kv_pool
        if not isinstance(kvcache, MHATokenToKVPool):
            raise RuntimeError(
                "the decoupled drafter at page_size > 1 copies boundary-tail "
                "K/V between pages and requires an MHA-family KV pool; got "
                f"{type(kvcache).__name__}"
            )
        uses_tiled_copy = (
            type(kvcache)._move_kv_cache_impl is MHATokenToKVPool._move_kv_cache_impl
            and not kvcache.use_hnd
            and not kvcache.use_native_move_kv_cache
        )
        if uses_tiled_copy and kvcache._kv_copy_config is None:
            kvcache._init_kv_copy_and_warmup()
        return kvcache

    def _cow_kv(
        self,
        *,
        src_pieces: list[torch.Tensor],
        dst_pieces: list[torch.Tensor],
        tag: str,
    ) -> None:
        """Batched private-head K/V copy (COW) over the full-attn layers: ONE
        (src, dst) index list per call, executed by the pool's layout-aware
        ``move_kv_cache``. Volume is rows x O(page tail) tokens per round --
        trivial next to a forward."""
        if not src_pieces:
            return
        self._cow_kv_pool.move_kv_cache(
            tgt_loc=torch.cat(dst_pieces), src_loc=torch.cat(src_pieces)
        )
        self.profiler.mark(f"{tag}_cow")

    def _cow_carrier_glue_heads(
        self,
        *,
        states: list[_DraftReqState],
        carriers: list[_SeatCarrier],
        base_lens: list[int],
    ) -> None:
        """Seed every glue row's private head before the forward that reads
        it. Fused mode copies only the boundary tail [anchor, base) -- the
        delta's KV is written by the glue rows themselves inside the fused
        extend. Non-fused mode copies [anchor, new_len): the advance already
        materialized the delta's KV in the seat's pages and the glue extend
        carries only chain tokens. Either way a glue row's head ends the
        forward holding boundary tail + delta + its chain prefix, which is
        what lets branch heads source from glue rows uniformly."""
        src_pieces: list[torch.Tensor] = []
        dst_pieces: list[torch.Tensor] = []
        for i, (state, carrier) in enumerate(zip(states, carriers)):
            anchor = self._page_floor(base_lens[i])
            head_end = (
                base_lens[i]
                if self._enable_fused_extend
                else state.committed_slots.numel()
            )
            span = head_end - anchor
            if span == 0:
                continue
            src_pieces.append(
                state.committed_slots[anchor:head_end].repeat(self.num_steps)
            )
            dst_pieces.append(carrier.glue_private_slots[:, :span].reshape(-1))
        self._cow_kv(src_pieces=src_pieces, dst_pieces=dst_pieces, tag="glue_head")

    def _cow_carrier_branch_heads(
        self,
        *,
        states: list[_DraftReqState],
        carriers: list[_SeatCarrier],
        base_lens: list[int],
        variant: _FanoutVariant,
    ) -> None:
        """Copy each selected branch row's prefix head -- boundary tail +
        delta + c_1..c_case -- into its own private pages, after the fused /
        glue forward wrote the chain KV. Case 0 sources from the seat (its
        pages hold tail + delta once the advance KV landed); case a >= 1
        sources from glue row a - 1, whose private head holds exactly
        tail + delta + c_1..c_a. Unselected (dead-fanout) rows keep stale KV:
        they are never forwarded, and their page-table entries are rewritten
        before any later round reads them."""
        src_pieces: list[torch.Tensor] = []
        dst_pieces: list[torch.Tensor] = []
        for i, (state, carrier) in enumerate(zip(states, carriers)):
            anchor = self._page_floor(base_lens[i])
            head = state.committed_slots.numel() - anchor  # tail + delta
            seat_src = state.committed_slots[anchor:]
            for row, case in zip(variant.sel_rows_pool, variant.case_of_row):
                span = head + case
                if span == 0:
                    continue
                src_pieces.append(
                    seat_src
                    if case == 0
                    else carrier.glue_private_slots[case - 1, :span]
                )
                dst_pieces.append(carrier.branch_private_slots[row, :span])
        self._cow_kv(src_pieces=src_pieces, dst_pieces=dst_pieces, tag="branch_head")

    def _case0_sync_paged(
        self,
        *,
        states: list[_DraftReqState],
        carriers: list[_SeatCarrier],
        f_live: int,
    ) -> None:
        """Case-0 carrier sync at page_size > 1. The advance materialized the
        delta's KV before this runs and the seat writes nothing else this
        round, so the shareable region extends to the page floor of the NEW
        committed length; only the boundary tail is private, COW'd into the
        live case-0 rows (the chain decodes then grow their pages via
        alloc_decode). The private rebind still covers ALL rows so no row
        retains a stale mapping."""
        pool = self.model_runner.req_to_token_pool
        src_pieces: list[torch.Tensor] = []
        dst_pieces: list[torch.Tensor] = []
        for state, carrier in zip(states, carriers):
            new_len = state.committed_slots.numel()
            anchor = self._page_floor(new_len)
            synced = min(carrier.synced_len, anchor)
            pool.req_to_token[carrier.all_rows, synced:anchor] = state.committed_slots[
                synced:anchor
            ].to(torch.int32)
            carrier.synced_len = anchor
            tail = new_len - anchor
            if tail == 0:
                continue
            pool.req_to_token[carrier.all_rows, anchor:new_len] = (
                carrier.all_private_slots[:, :tail].to(torch.int32)
            )
            src_pieces.append(state.committed_slots[anchor:].repeat(f_live))
            dst_pieces.append(carrier.branch_private_slots[:f_live, :tail].reshape(-1))
        self._cow_kv(src_pieces=src_pieces, dst_pieces=dst_pieces, tag="case0_head")

    def _take_slow_round_pages(
        self, *, bs: int, scratch_kv_pages: list[torch.Tensor]
    ) -> _SlowRoundPages:
        """Stage a slow subround's private pages up front (the mamba-slot
        staging idiom): transient backbone pages ride ``scratch_kv_pages``
        back to the arena at round end; the glue/branch pages are
        carrier-destined and removed from the list on donation."""
        ppr = self._pages_per_carrier_row
        rows_per_seat = (self.num_steps + 1) * self.fanout
        backbone_pages = self._kv_page_arena.take(bs).view(bs, 1)
        glue_pages = self._kv_page_arena.take(bs * self.num_steps * ppr).view(
            bs * self.num_steps, ppr
        )
        branch_pages = self._kv_page_arena.take(bs * rows_per_seat * ppr).view(
            bs * rows_per_seat, ppr
        )
        scratch_kv_pages += [backbone_pages, glue_pages, branch_pages]
        return _SlowRoundPages(
            backbone_pages=backbone_pages,
            backbone_flats=self._page_flat_slots(backbone_pages),
            glue_pages=glue_pages,
            glue_flats=self._page_flat_slots(glue_pages),
            branch_pages=branch_pages,
            branch_flats=self._page_flat_slots(branch_pages),
        )

    def _paged_backbone_prefixes(
        self,
        *,
        states: list[_DraftReqState],
        backbone_flats: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Per-seat backbone-row prefix at page_size > 1: the seat's full
        pages + a private COW of the boundary tail. The backbone extend then
        continues the private page in place (alloc_extend resumes at the last
        prefix slot + 1) and later crossings take throwaway allocator pages,
        both freed correctly by the round's page-aware scratch free. Copies
        the tail K/V before returning."""
        prefixes: list[torch.Tensor] = []
        src_pieces: list[torch.Tensor] = []
        dst_pieces: list[torch.Tensor] = []
        for i, state in enumerate(states):
            new_len = state.committed_slots.numel()
            anchor = self._page_floor(new_len)
            tail = new_len - anchor
            if tail == 0:
                prefixes.append(state.committed_slots)
                continue
            head = backbone_flats[i, :tail]
            prefixes.append(torch.cat([state.committed_slots[:anchor], head]))
            src_pieces.append(state.committed_slots[anchor:])
            dst_pieces.append(head)
        self._cow_kv(src_pieces=src_pieces, dst_pieces=dst_pieces, tag="backbone_head")
        return prefixes

    def _paged_branch_prefixes(
        self,
        *,
        states: list[_DraftReqState],
        backbone_slots_dev: torch.Tensor,
        branch_flats: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Slow-round branch prefixes at page_size > 1, in the batch's
        case-major full-F row order: shared full pages + a private COW head
        of boundary tail + c_1..c_case (the case backbone lives in the
        backbone row's pages -- copied, never shared, since those pages are
        transient scratch). The guess extend then continues each private page
        in place. The heads land in the rows' carrier-lifetime pages, so the
        donated carrier starts with valid private mappings."""
        num_cases, fanout = self.num_steps + 1, self.fanout
        prefixes: list[torch.Tensor] = []
        src_pieces: list[torch.Tensor] = []
        dst_pieces: list[torch.Tensor] = []
        row = 0
        for i, state in enumerate(states):
            new_len = state.committed_slots.numel()
            anchor = self._page_floor(new_len)
            tail = new_len - anchor
            shared = state.committed_slots[:anchor]
            for case in range(num_cases):
                span = tail + case
                src = (
                    torch.cat(
                        [state.committed_slots[anchor:], backbone_slots_dev[i, :case]]
                    )
                    if span > 0
                    else None
                )
                for _ in range(fanout):
                    if span == 0:
                        prefixes.append(shared)
                    else:
                        head = branch_flats[row, :span]
                        prefixes.append(torch.cat([shared, head]))
                        src_pieces.append(src)
                        dst_pieces.append(head)
                    row += 1
        self._cow_kv(src_pieces=src_pieces, dst_pieces=dst_pieces, tag="branch_head")
        return prefixes

    def _case0_round(
        self,
        keys: list[DraftReqKey],
        states: list[_DraftReqState],
        scratch_batches: list[ScheduleBatch],
        scratch_slots: list[torch.Tensor],
        scratch_kv_pages: list[torch.Tensor],
        *,
        f_live: int,
    ) -> dict:
        """Miss round collapsed to case 0 (the dead-cell theorem).

        The drafter and the verifier judge the same (block, delta) pair, so a
        drafter miss means the verifier's select missed too and fell back to a
        plain decode: the NEXT commit is a single case-0 bonus, and every
        case >= 1 cell of this block is dead. (If the fallback's junk drafts
        happen to be target-agreeing the next delta can still be longer; it
        then simply misses again -- one wasted round, never a wrong token.)

        So: advance the delta, take the top-F node-0 guesses, and run ONE
        F-row decode chain on the carrier's case-0 rows (which carry no
        backbone prefix -- their sequences start at the committed length).
        No backbone, no glue, no carrier rebuild; dead guess cells are
        poisoned with -1 (matches nothing on either side) and dead chains
        with 0 (never read behind a poisoned guess).
        """
        num_steps = self.num_steps
        bs = len(states)
        carriers = [self._seat_carriers[key] for key in keys]
        pool = self.model_runner.req_to_token_pool

        # -- Advance the committed prefix; last logits = node 0 --------------
        # No dead-guess exclusion here: the fallback round this block serves
        # commits a freely decoded bonus (no rejection constrains it).
        base_lens = [state.committed_slots.numel() for state in states]
        node0_logits, advance_slots = self._advance_forward(
            states=states,
            base_lens=base_lens,
            scratch_batches=scratch_batches,
            scratch_slots=scratch_slots,
            scratch_kv_pages=scratch_kv_pages,
        )
        # Consume the graph runner's static logits buffer before the next
        # forward overwrites it.
        node0_guesses = torch.topk(node0_logits, f_live, dim=-1).indices  # [bs, f]
        self._absorb_advance_slots(states, advance_slots)

        # -- Carrier rows only need the committed delta (no backbone) --------
        if self._paged:
            self._case0_sync_paged(states=states, carriers=carriers, f_live=f_live)
        else:
            for i, (state, carrier) in enumerate(zip(states, carriers)):
                new_len = state.committed_slots.numel()
                synced = min(carrier.synced_len, base_lens[i])
                pool.req_to_token[carrier.all_rows, synced:new_len] = (
                    state.committed_slots[synced:new_len].to(torch.int32)
                )
                carrier.synced_len = new_len
        self.profiler.mark("case0_sync")

        # -- Case-0 chains: per seat, the first f_live carrier rows (case 0's
        # rows lead the full-F pool layout, so the slice stays contiguous).
        branch = self._branch_template
        branch.reqs = [
            req for carrier in carriers for req in carrier.branch_reqs[:f_live]
        ]
        # See _glue_forward: mrope models index this list per row.
        branch.multimodal_inputs = [None] * len(branch.reqs)
        branch.req_pool_indices = (
            torch.cat([carrier.branch_rows[:f_live] for carrier in carriers])
            if bs > 1
            else carriers[0].branch_rows[:f_live]
        )
        seq_host = [
            state.committed_slots.numel() for state in states for _ in range(f_live)
        ]
        seq_cpu = torch.tensor(seq_host, dtype=torch.int64)
        branch.seq_lens = self._h2d.to_device(seq_cpu, dtype=torch.int64)
        branch.seq_lens_cpu = seq_cpu
        branch.seq_lens_sum = None
        branch.orig_seq_lens = branch.seq_lens.to(torch.int32)
        if self._hybrid:
            # All case-0 chains start from node 0: fork the freshly advanced
            # seat state into every live case-0 row before the first decode.
            self._fork_mamba_states(
                src_slots=torch.cat(
                    [state.mamba_slot.repeat(f_live) for state in states]
                ),
                dst_slots=torch.cat(
                    [carrier.branch_mamba_slots[:f_live] for carrier in carriers]
                ),
            )
        self.profiler.mark("case0_mut")
        if self._chain_plan:
            with self.profiler.stage("case0-plan"):
                plan_steps = self._prepare_chain_steps(
                    branch, first_tokens=node0_guesses.reshape(-1)
                )
            tokens = node0_guesses.reshape(-1)
            chain_steps: list[torch.Tensor] = []
            for step, (fb, step_slots) in enumerate(plan_steps):
                self._track_scratch_slots(
                    scratch_slots,
                    slots=step_slots,
                    positions=[seq + step for seq in seq_host],
                )
                with self.profiler.stage("case0-step-fwd"):
                    fb.input_ids = tokens
                    logits = self.model_runner.forward(fb).logits_output
                tokens = logits.next_token_logits.argmax(dim=-1)
                chain_steps.append(tokens)
        else:
            logits, step_slots = self._decode_step(
                branch, node0_guesses.reshape(-1), tag="case0"
            )
            # Decode step s fills each row's position seq_host[row] + s.
            self._track_scratch_slots(
                scratch_slots, slots=step_slots, positions=seq_host
            )
            chain_steps = [logits.argmax(dim=-1)]
            for step in range(1, num_steps):
                logits, step_slots = self._decode_step(
                    branch, chain_steps[-1], tag="case0"
                )
                self._track_scratch_slots(
                    scratch_slots,
                    slots=step_slots,
                    positions=[seq + step for seq in seq_host],
                )
                chain_steps.append(logits.argmax(dim=-1))

        # No backbone this round: only a case-0 match can hit next round, and
        # a case-0 hit reads its chain from the units mirror, not the backbone.
        # (_pack_and_mirror pads the single live case out to the full block.)
        return self._pack_and_mirror(
            states=states,
            guesses_stack=node0_guesses.unsqueeze(1),  # [bs, 1, f]
            chain_steps=chain_steps,
            new_backbones=[[] for _ in states],
        )

    def _slow_round(
        self,
        keys: list[DraftReqKey],
        states: list[_DraftReqState],
        scratch_batches: list[ScheduleBatch],
        scratch_slots: list[torch.Tensor],
        scratch_mamba_slots: list[torch.Tensor],
        scratch_kv_pages: list[torch.Tensor],
    ) -> dict:
        num_steps, fanout = self.num_steps, self.fanout
        num_cases = num_steps + 1
        bs = len(states)
        # A slow subround rebuilds its seats' carriers at the end; evicting
        # them up front bounds the subround's peak pool-row usage.
        for key in keys:
            self._evict_seat(key)

        # page_size > 1: stage the subround's private pages (see the struct's
        # docstring); None at page 1, where nested prefixes share raw slots.
        slow_pages = (
            self._take_slow_round_pages(bs=bs, scratch_kv_pages=scratch_kv_pages)
            if self._paged
            else None
        )

        # Hybrid: stage the subround's state slots up front. The backbone
        # advances ONE slot per seat in place; the node checkpoints it passes
        # through are copied into the future carrier glue slots (glue slot g
        # <- node-(g+1) state, the same invariant a fast round's glue forward
        # establishes). All takes ride scratch_mamba_slots so an aborted
        # subround can't leak them; the carrier build removes the two
        # persistent tensors on donation (the scratch_batches.remove idiom).
        glue_mamba_flat = branch_mamba_flat = backbone_mamba_slots = None
        if self._hybrid:
            backbone_mamba_slots = self._mamba_arena.take(bs)
            glue_mamba_flat = self._mamba_arena.take(bs * num_steps)
            branch_mamba_flat = self._mamba_arena.take(bs * num_cases * fanout)
            scratch_mamba_slots += [
                backbone_mamba_slots,
                glue_mamba_flat,
                branch_mamba_flat,
            ]
        glue_ckpt_slots = (
            glue_mamba_flat.view(bs, num_steps) if glue_mamba_flat is not None else None
        )

        # -- Phase 1: advance the committed prefix; last logits = node 0 ----
        node_logits: list[torch.Tensor] = []
        advance_batch, advance_slots = self._extend_batch(
            token_lists=[state.committed_tokens for state in states],
            prefix_slots=[state.committed_slots for state in states],
            tag="advance",
            mamba_slots=self._seat_mamba_slots(states),
        )
        scratch_batches.append(advance_batch)
        logits = self._forward(advance_batch, tag="advance")
        node_logits.append(logits)
        self._absorb_advance_slots(states, advance_slots)
        # Committed length per seat after the absorb == the position every
        # phase below starts writing at (see _track_scratch_slots).
        new_lens = [state.committed_slots.numel() for state in states]

        # -- Phase 2: backbone c_1..c_K + per-node top-F guesses ------------
        # guesses[a]: [bs, F] int64; backbone_tokens[j]: [bs] (c_{j+1}).
        guesses = [torch.topk(node_logits[0], fanout, dim=-1).indices]
        backbone_tokens: list[torch.Tensor] = [guesses[0][:, 0]]
        backbone_slot_steps: list[torch.Tensor] = []
        if num_steps >= 1:
            backbone_prefixes = (
                self._paged_backbone_prefixes(
                    states=states, backbone_flats=slow_pages.backbone_flats
                )
                if self._paged
                else [state.committed_slots for state in states]
            )
            backbone_batch, first_slots = self._extend_batch(
                token_lists=[
                    state.committed_tokens + [int(backbone_tokens[0][i])]
                    for i, state in enumerate(states)
                ],
                prefix_slots=backbone_prefixes,
                tag="backbone",
                mamba_slots=backbone_mamba_slots,
            )
            scratch_batches.append(backbone_batch)
            # One extend token per seat: c_1, at the committed length.
            self._track_scratch_slots(
                scratch_slots, slots=first_slots, positions=new_lens
            )
            backbone_slot_steps.append(first_slots)
            if self._hybrid:
                # The backbone decodes over node-0 state without touching the
                # seat slot itself: fork it into the dedicated backbone slot.
                self._fork_mamba_states(
                    src_slots=self._seat_mamba_slots(states),
                    dst_slots=backbone_mamba_slots,
                )
            logits = self._forward(backbone_batch, tag="backbone")
            node_logits.append(logits)
            guesses.append(torch.topk(logits, fanout, dim=-1).indices)
            if self._hybrid and num_steps >= 2:
                # Node-1 checkpoint (fork source for case-1 branch rows); the
                # extend just advanced the backbone slot past c_1.
                self._fork_mamba_states(
                    src_slots=backbone_mamba_slots, dst_slots=glue_ckpt_slots[:, 0]
                )
            for step_idx in range(num_steps - 1):
                next_tokens = guesses[-1][:, 0]
                backbone_tokens.append(next_tokens)
                logits, step_slots = self._decode_step(
                    backbone_batch, next_tokens, tag="backbone"
                )
                self._track_scratch_slots(
                    scratch_slots,
                    slots=step_slots,
                    positions=[new_len + 1 + step_idx for new_len in new_lens],
                )
                backbone_slot_steps.append(step_slots)
                node_logits.append(logits)
                guesses.append(torch.topk(logits, fanout, dim=-1).indices)
                if self._hybrid and step_idx < num_steps - 2:
                    # Node-(step_idx+2) checkpoint. The final decode needs no
                    # copy: case K forks straight from the backbone slot,
                    # which ends this loop holding node-K state (glue slot
                    # K-1 stays stale until the next fast round's fork
                    # rewrites it -- it is never read before that).
                    self._fork_mamba_states(
                        src_slots=backbone_mamba_slots,
                        dst_slots=glue_ckpt_slots[:, step_idx + 1],
                    )

        # -- Phase 3: branch chains for every (case, guess) -----------------
        # Row order: (req 0: (a0,f0), (a0,f1) ... (aK,fF-1)), (req 1: ...).
        guesses_stack = torch.stack(guesses, dim=1)  # [bs, K+1, F]
        guesses_cpu = guesses_stack.tolist()
        backbone_cpu = [
            [int(backbone_tokens[j][i]) for j in range(num_steps)] for i in range(bs)
        ]
        backbone_slots_dev = (
            torch.stack(backbone_slot_steps, dim=1)
            if backbone_slot_steps
            else torch.empty((bs, 0), dtype=torch.int64, device=self.device)
        )  # [bs, K]
        branch_token_lists: list[list[int]] = []
        for i, state in enumerate(states):
            for case in range(num_cases):
                for f in range(fanout):
                    branch_token_lists.append(
                        state.committed_tokens
                        + backbone_cpu[i][:case]
                        + [guesses_cpu[i][case][f]]
                    )
        if self._paged:
            branch_prefix_slots = self._paged_branch_prefixes(
                states=states,
                backbone_slots_dev=backbone_slots_dev,
                branch_flats=slow_pages.branch_flats,
            )
        else:
            branch_prefix_slots = [
                torch.cat([state.committed_slots, backbone_slots_dev[i, :case]])
                for i, state in enumerate(states)
                for case in range(num_cases)
                for _ in range(fanout)
            ]
        self.profiler.mark("branch_lists")
        branch_batch, branch_first_slots = self._extend_batch(
            token_lists=branch_token_lists,
            prefix_slots=branch_prefix_slots,
            tag="branch",
            mamba_slots=branch_mamba_flat,
        )
        scratch_batches.append(branch_batch)
        # One extend token per branch row (its guess), in the batch's
        # case-major full-F row order: position = committed length + case.
        branch_row_positions = [
            new_len + case
            for new_len in new_lens
            for case in range(num_cases)
            for _ in range(fanout)
        ]
        self._track_scratch_slots(
            scratch_slots, slots=branch_first_slots, positions=branch_row_positions
        )
        if self._hybrid:
            # Fork node states into every branch row before its extend reads
            # them: case 0 <- seat slot, cases 1..K-1 <- the checkpoints,
            # case K <- the backbone slot itself (it ended at node K).
            case_of_row_dev = self._full_grid_variant.case_of_row_dev
            self._fork_mamba_states(
                src_slots=torch.cat(
                    [
                        torch.cat(
                            [
                                states[i].mamba_slot,
                                glue_ckpt_slots[i, : num_steps - 1],
                                backbone_mamba_slots[i : i + 1],
                            ]
                        )[case_of_row_dev]
                        for i in range(bs)
                    ]
                ),
                dst_slots=branch_mamba_flat,
            )
        logits = self._forward(branch_batch, tag="branch")
        chain_steps: list[torch.Tensor] = [logits.argmax(dim=-1)]
        for step in range(num_steps - 1):
            logits, step_slots = self._decode_step(
                branch_batch, chain_steps[-1], tag="branch"
            )
            self._track_scratch_slots(
                scratch_slots,
                slots=step_slots,
                positions=[pos + 1 + step for pos in branch_row_positions],
            )
            chain_steps.append(logits.argmax(dim=-1))

        packed = self._pack_and_mirror(
            states=states,
            guesses_stack=guesses_stack,
            chain_steps=chain_steps,
            new_backbones=backbone_cpu,
        )
        self._build_seat_carriers(
            keys=keys,
            states=states,
            branch_batch=branch_batch,
            scratch_batches=scratch_batches,
            scratch_slots=scratch_slots,
            scratch_mamba_slots=scratch_mamba_slots,
            scratch_kv_pages=scratch_kv_pages,
            backbone_cpu=backbone_cpu,
            backbone_slots_dev=backbone_slots_dev,
            glue_mamba_flat=glue_mamba_flat,
            branch_mamba_flat=branch_mamba_flat,
            slow_pages=slow_pages,
        )
        return packed

    # ------------------------------------------------------------------ #
    # Packing, mirrors, carriers
    # ------------------------------------------------------------------ #

    def _pack_and_mirror(
        self,
        *,
        states: list[_DraftReqState],
        guesses_stack: torch.Tensor,
        chain_steps: list[torch.Tensor],
        new_backbones: list[list[int]],
        sel_rows_dev: Optional[torch.Tensor] = None,
    ) -> dict:
        """Pack units [guess, chain_1..chain_K] and arm the fast path.

        chains: [bs * (K+1) * F, K] -> [bs, K+1, F, K]; stays on device so
        the CUDA IPC data plane can push it D2D (the ZMQ path D2Hs it). Each
        seat keeps the block on device (next round's glue input) plus an
        async pinned host mirror (next round's hit test). A partial unit grid
        (case-0 miss round, adaptive fanout, per-case budget) is padded to the
        full block here.
        """
        num_cases = self.num_steps + 1
        bs = len(states)
        guesses_stack, chain_steps = self._expand_units(
            guesses_stack=guesses_stack,
            chain_steps=chain_steps,
            bs=bs,
            sel_rows_dev=sel_rows_dev,
        )
        chains = torch.stack(chain_steps, dim=1).view(
            bs, num_cases, self.fanout, self.num_steps
        )
        guesses_col = guesses_stack.unsqueeze(-1)  # [bs, K+1, F, 1]
        units_device = torch.cat([guesses_col, chains], dim=-1)  # [bs, K+1, F, K+1]
        self.profiler.mark("pack")
        mirror_event = torch.cuda.Event()
        for i, state in enumerate(states):
            state.last_units_dev = units_device[i]
            if state.last_units_host is None:
                state.last_units_host = torch.empty(
                    units_device[i].shape, dtype=units_device.dtype, pin_memory=True
                )
            state.last_units_host.copy_(units_device[i], non_blocking=True)
            state.last_backbone_host = list(new_backbones[i])
            state.mirror_event = mirror_event
        mirror_event.record()
        self.profiler.mark("mirror")
        return {
            "pool_indices": [state.req_pool_idx for state in states],
            "base_committed_lens": [len(state.committed_tokens) for state in states],
            "units_device": units_device,
        }

    def _expand_units(
        self,
        *,
        guesses_stack: torch.Tensor,  # [bs, cases_live, f_live]
        chain_steps: list[torch.Tensor],  # each [bs * rows_live]
        bs: int,
        sel_rows_dev: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Pad a partial unit grid (fewer live cases and/or guesses) to the
        full (K+1) x F block: dead guess cells are poisoned with -1 (matches
        no vocab token on either side) and dead chains with 0 (never read
        behind a poisoned guess). The wire/mirror shape never changes, so the
        verifier is oblivious to the case-0 collapse, the adaptive fanout and
        the per-case budget alike.

        ``sel_rows_dev`` (per-case budget rounds) makes the chain steps scatter
        by full-grid ROW instead of by rectangular slice -- their guess grid
        already arrives full-width, poisoned outside the budget."""
        num_cases, fanout = self.num_steps + 1, self.fanout
        if sel_rows_dev is not None:
            return guesses_stack, self._scatter_chain_rows(
                chain_steps=chain_steps, bs=bs, sel_rows_dev=sel_rows_dev
            )
        cases_live, f_live = guesses_stack.shape[1], guesses_stack.shape[2]
        if cases_live == num_cases and f_live == fanout:
            return guesses_stack, chain_steps
        full_guesses = torch.full(
            (bs, num_cases, fanout),
            -1,
            dtype=guesses_stack.dtype,
            device=self.device,
        )
        full_guesses[:, :cases_live, :f_live] = guesses_stack
        full_steps: list[torch.Tensor] = []
        for step in chain_steps:
            full_step = torch.zeros(
                (bs, num_cases, fanout), dtype=step.dtype, device=self.device
            )
            full_step[:, :cases_live, :f_live] = step.view(bs, cases_live, f_live)
            full_steps.append(full_step.view(-1))
        return full_guesses, full_steps

    def _scatter_chain_rows(
        self,
        *,
        chain_steps: list[torch.Tensor],  # each [bs * rows_live]
        bs: int,
        sel_rows_dev: torch.Tensor,  # full-grid row of each live row
    ) -> list[torch.Tensor]:
        """Scatter a ragged (per-case budget) round's chain steps into the full
        (K+1) x F row grid. Unbudgeted rows stay 0 -- never read, since their
        guess cell ships poisoned."""
        rows = (self.num_steps + 1) * self.fanout
        full_steps: list[torch.Tensor] = []
        for step in chain_steps:
            full_step = torch.zeros((bs, rows), dtype=step.dtype, device=self.device)
            full_step[:, sel_rows_dev] = step.view(bs, -1)
            full_steps.append(full_step.view(-1))
        return full_steps

    def _build_seat_carriers(
        self,
        *,
        keys: list[DraftReqKey],
        states: list[_DraftReqState],
        branch_batch: ScheduleBatch,
        scratch_batches: list[ScheduleBatch],
        scratch_slots: list[torch.Tensor],
        scratch_mamba_slots: list[torch.Tensor],
        scratch_kv_pages: list[torch.Tensor],
        backbone_cpu: list[list[int]],
        backbone_slots_dev: torch.Tensor,
        glue_mamba_flat: Optional[torch.Tensor],
        branch_mamba_flat: Optional[torch.Tensor],
        slow_pages: Optional[_SlowRoundPages],
    ) -> None:
        """Donate this slow subround's branch rows + a freshly built glue
        batch to per-seat carriers (pool rows persist until the seat closes;
        KV slots stay per-round scratch). The built batches double as the
        assembly shells for later fast subrounds."""
        if not self._enable_glue_fast_path:
            return
        # The branch batch's pool rows survive the round; its KV slots are
        # already tracked in scratch_slots and freed as usual. On hybrid the
        # rows' mamba slots survive WITH the rows (the fixed row->slot binding
        # is what keeps the mapping written at alloc valid), so they move from
        # the scratch list to carrier ownership here -- identity-filtered, not
        # list.remove, which would trip tensor __eq__. The paged private pages
        # (glue + branch rows' carrier-lifetime pages) move the same way.
        scratch_batches.remove(branch_batch)
        if glue_mamba_flat is not None:
            scratch_mamba_slots[:] = [
                slots
                for slots in scratch_mamba_slots
                if slots is not glue_mamba_flat and slots is not branch_mamba_flat
            ]
        if slow_pages is not None:
            scratch_kv_pages[:] = [
                pages
                for pages in scratch_kv_pages
                if pages is not slow_pages.glue_pages
                and pages is not slow_pages.branch_pages
            ]
        if self._paged:
            # Placeholder rows (no forward ever runs on this batch): a
            # page-aligned prefix keeps the throwaway alloc from continuing
            # the seat's partial page -- the round-end free releases whole
            # pages, so continuation slots there would drag the seat's live
            # page into the free.
            glue_build_prefixes = [
                state.committed_slots[: self._page_floor(state.committed_slots.numel())]
                for state in states
                for _ in range(self.num_steps)
            ]
            # Row (i, g) re-extends [page floor, committed + g + 1).
            glue_out_positions = [
                pos
                for state in states
                for g in range(self.num_steps)
                for pos in range(
                    self._page_floor(state.committed_slots.numel()),
                    state.committed_slots.numel() + g + 1,
                )
            ]
        else:
            glue_build_prefixes = [
                torch.cat([state.committed_slots, backbone_slots_dev[i, :g]])
                for i, state in enumerate(states)
                for g in range(self.num_steps)
            ]
            glue_out_positions = []
        glue_batch, glue_slots = self._extend_batch(
            token_lists=[
                state.committed_tokens + backbone_cpu[i][: g + 1]
                for i, state in enumerate(states)
                for g in range(self.num_steps)
            ],
            prefix_slots=glue_build_prefixes,
            tag="glue_build",
            mamba_slots=glue_mamba_flat,
        )
        # Build-time extend slots are placeholders (no forward ran); the fast
        # path re-points out_cache_loc at each round's backbone slots.
        self._track_scratch_slots(
            scratch_slots, slots=glue_slots, positions=glue_out_positions
        )
        bs = len(keys)
        num_steps = self.num_steps
        rows_per_seat = (num_steps + 1) * self.fanout
        glue_rows_all = glue_batch.req_pool_indices.view(bs, num_steps)
        branch_rows_all = branch_batch.req_pool_indices.view(bs, rows_per_seat)
        glue_mamba_all = (
            glue_mamba_flat.view(bs, num_steps) if glue_mamba_flat is not None else None
        )
        branch_mamba_all = (
            branch_mamba_flat.view(bs, rows_per_seat)
            if branch_mamba_flat is not None
            else None
        )
        glue_flats_all = glue_pages_all = None
        branch_flats_all = branch_pages_all = None
        if slow_pages is not None:
            ppr = self._pages_per_carrier_row
            glue_flats_all = slow_pages.glue_flats.view(bs, num_steps, -1)
            branch_flats_all = slow_pages.branch_flats.view(bs, rows_per_seat, -1)
            glue_pages_all = slow_pages.glue_pages.view(bs, num_steps, ppr)
            branch_pages_all = slow_pages.branch_pages.view(bs, rows_per_seat, ppr)
        for i, (key, state) in enumerate(zip(keys, states)):
            self._seat_carriers[key] = _SeatCarrier(
                glue_rows=glue_rows_all[i],
                branch_rows=branch_rows_all[i],
                glue_reqs=glue_batch.reqs[i * num_steps : (i + 1) * num_steps],
                branch_reqs=branch_batch.reqs[
                    i * rows_per_seat : (i + 1) * rows_per_seat
                ],
                # Paged: page-aligned, so the fast-round shared sync starts at
                # a boundary the seat can never free or rewrite; the row
                # entries between the floor and the built length are private
                # already (branch heads) or placeholder (glue rows, never
                # forwarded at build) and every later round rewrites them.
                synced_len=(
                    self._page_floor(state.committed_slots.numel())
                    if self._paged
                    else state.committed_slots.numel()
                ),
                glue_mamba_slots=(
                    glue_mamba_all[i] if glue_mamba_all is not None else None
                ),
                branch_mamba_slots=(
                    branch_mamba_all[i] if branch_mamba_all is not None else None
                ),
                glue_private_slots=(
                    glue_flats_all[i] if glue_flats_all is not None else None
                ),
                branch_private_slots=(
                    branch_flats_all[i] if branch_flats_all is not None else None
                ),
                private_kv_pages=(
                    torch.cat([glue_pages_all[i], branch_pages_all[i]])
                    if glue_pages_all is not None
                    else None
                ),
            )
        self._glue_template = glue_batch
        self._branch_template = branch_batch
        self.profiler.mark("carrier_build")

    def _evict_seat(self, key: DraftReqKey) -> None:
        carrier = self._seat_carriers.pop(key, None)
        if carrier is None:
            return
        for req in carrier.glue_reqs + carrier.branch_reqs:
            if req.req_pool_idx is not None:
                # The carrier's mamba slots are engine-owned (returned to the
                # arena below); null the preset so no pool path can ever treat
                # the row free as a slot free.
                req.mamba_pool_idx = None
                self.model_runner.req_to_token_pool.free(req)
        if carrier.glue_mamba_slots is not None:
            self._mamba_arena.give_back(carrier.glue_mamba_slots)
            self._mamba_arena.give_back(carrier.branch_mamba_slots)
        if carrier.private_kv_pages is not None:
            self._kv_page_arena.give_back(carrier.private_kv_pages)

    # ------------------------------------------------------------------ #
    # Batch plumbing (bench_one_batch harness pattern)
    # ------------------------------------------------------------------ #

    def _seat_mamba_slots(self, states: list[_DraftReqState]) -> Optional[torch.Tensor]:
        """Per-seat state slots for an advance batch (None on pure-KV)."""
        if not self._hybrid:
            return None
        return torch.cat([state.mamba_slot for state in states])

    def _log_extend_graph_attn_metadata(self, *, rows: int) -> None:
        """NANCHECK companion: dump the full-attn plane's DRAFT_EXTEND_V2
        replay metadata views (for micro-vs-e2e diffing). trtllm-only shape;
        other backends just skip. Debug instrumentation only."""
        backend = self._extend_graph_backend
        if backend is None:
            return
        full = backend.full_attn_backend if self._hybrid else backend
        # Debug-only duck check: only the trtllm backend keeps this dict.
        meta_by_bs = getattr(full, "draft_extend_metadata", None)
        if not isinstance(meta_by_bs, dict) or rows not in meta_by_bs:
            return
        meta = meta_by_bs[rows]
        logger.info(
            "extend-graph attn-meta: cache_seqlens=%s cu_q=%s cu_k=%s max_q=%s "
            "pt_head=%s",
            meta.cache_seqlens_int32[:rows].tolist(),
            meta.cu_seqlens_q[: rows + 1].tolist(),
            meta.cu_seqlens_k[: rows + 1].tolist(),
            meta.max_seq_len_q,
            meta.page_table[:rows, :4].tolist(),
        )

    def _log_mamba_state_nan(self, *, tag: str, slots: torch.Tensor) -> None:
        """NANCHECK probe: report NaN presence in the given (virtual) mamba
        slots' conv + temporal state, per slot. Debug only (host syncs)."""
        pool = self.model_runner.req_to_token_pool
        phys = pool.translate_mamba_indices(slots.reshape(-1).contiguous())
        st = pool.mamba_pool.mamba_cache
        conv_nan = [
            [bool(conv[:, p].isnan().any()) for conv in st.conv] for p in phys.tolist()
        ]
        temporal_nan = [bool(st.temporal[:, p].isnan().any()) for p in phys.tolist()]
        logger.info(
            "mamba-state nancheck[%s]: slots=%s conv_nan=%s temporal_nan=%s",
            tag,
            phys.tolist(),
            conv_nan,
            temporal_nan,
        )

    def _fork_mamba_states(
        self, *, src_slots: torch.Tensor, dst_slots: torch.Tensor
    ) -> None:
        """Batched recurrent-state fork (conv + ssm), src[i] -> dst[i].

        Slot ids are virtual (allocator space); translate before the physical
        pool op. Repeated sources are fine, src/dst stay disjoint by
        construction (seat / glue / branch / backbone / stash slots never
        alias). The flatten also materializes strided column views -- the
        fused copy kernel consumes raw index arrays, not strides.
        """
        pool = self.model_runner.req_to_token_pool
        pool.mamba_pool.copy_from(
            src_indices=pool.translate_mamba_indices(
                src_slots.reshape(-1).contiguous()
            ),
            dst_indices=pool.translate_mamba_indices(
                dst_slots.reshape(-1).contiguous()
            ),
        )

    def _extend_batch(
        self,
        *,
        token_lists: list[list[int]],
        prefix_slots: list[torch.Tensor],
        tag: str,
        mamba_slots: Optional[torch.Tensor] = None,
    ) -> tuple[ScheduleBatch, torch.Tensor]:
        """Extend each row's tokens beyond its (slot-shared) prefix.

        ``mamba_slots`` (hybrid only, one engine-owned state slot per row) is
        preset on each Req so ``HybridReqToTokenPool.alloc`` binds the row to
        it instead of fresh-allocating -- a fresh alloc would flag the slot
        for a silent zero on the next forward, destroying the forked state.

        Returns (batch, newly_allocated_slots_flat).
        """
        if self._hybrid:
            assert mamba_slots is not None and mamba_slots.numel() == len(
                token_lists
            ), f"hybrid {tag} batch must preset one mamba slot per row"
        reqs = []
        for i, tokens in enumerate(token_lists):
            req = Req(
                rid=str(i),
                origin_input_text="",
                origin_input_ids=array("q", tokens),
                sampling_params=self._sampling_params,
            )
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.logprob_start_len = -1
            req.prefix_indices = prefix_slots[i].to(self.device)
            req.set_extend_range(prefix_slots[i].numel(), len(tokens))
            if mamba_slots is not None:
                req.mamba_pool_idx = mamba_slots[i]
            reqs.append(req)
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=self._tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        # A non-empty clear list means some row slipped past the preset and
        # fresh-allocated a slot: the next forward would zero it silently
        # (prepare_for_extend already consumed the per-req needs_clear flags
        # into this batch field, so this is the one observable trace).
        assert (
            not self._hybrid or batch.mamba_clear_indices is None
        ), f"hybrid {tag} batch fresh-allocated mamba slots (preset missed)"
        if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        self.profiler.mark(f"{tag}_build")
        return batch, batch.out_cache_loc

    def _forward(self, batch: ScheduleBatch, *, tag: str) -> torch.Tensor:
        forward_batch = ForwardBatch.init_new(
            batch, self.model_runner, return_hidden_states_before_norm=False
        )
        self.profiler.mark(f"{tag}_fb")
        logits_output = self.model_runner.forward(forward_batch).logits_output
        self.profiler.mark(f"{tag}_fwd")
        return logits_output.next_token_logits

    def _decode_step(
        self,
        batch: ScheduleBatch,
        input_tokens: torch.Tensor,
        *,
        tag: str,
        cascade: Optional[_CascadeMetadata] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with self.profiler.stage(f"{tag}-step-prep"):
            batch.input_ids = input_tokens.to(torch.int64)
            batch.prepare_for_decode()
            if cascade is not None:
                # Append this step's KV slot to each row's private tail, then
                # advance the tail lengths so the kernel covers the new token.
                cascade.tail_page_table[
                    cascade.row_indices, cascade.tail_lens.long()
                ] = batch.out_cache_loc.to(torch.int32)
                cascade.tail_lens.add_(1)
            self.profiler.mark(f"{tag}_step_prep")
            forward_batch = ForwardBatch.init_new(
                batch, self.model_runner, return_hidden_states_before_norm=False
            )
            forward_batch.decoupled_cascade = cascade
            self.profiler.mark(f"{tag}_step_fb")
        with self.profiler.stage(f"{tag}-step-fwd"):
            logits_output = self.model_runner.forward(forward_batch).logits_output
            self.profiler.mark(f"{tag}_step_fwd")
        return logits_output.next_token_logits, batch.out_cache_loc

    def _track_scratch_slots(
        self,
        scratch_slots: list[torch.Tensor],
        *,
        slots: torch.Tensor,
        positions: list[int],
    ) -> None:
        """Record one ``alloc_extend`` / ``alloc_decode`` output for the
        round-end free, page-granular at P > 1 -- WITHOUT reading the device.

        ``positions[j]`` is the logical sequence position ``slots[j]`` was
        allocated for (host-known: it is what the engine asked for).

        The page rule, from ``alloc_extend`` / ``alloc_decode``: a page comes
        off the global free list exactly when the position being filled STARTS
        a page; every other position continues the previous slot's page
        (``last_loc + 1``). So ``slots[j]`` heads a freshly allocated page iff
        ``positions[j] % P == 0``, and those heads are the round's complete
        throwaway-page set: any other slot lives in a page the engine already
        owns -- a carrier arena page, the seat's committed page, or a fresh
        page whose own head is in this list. Freeing just the heads therefore
        frees exactly the throwaway pages, which is what the old round-tail
        device set difference (unique + isin over every scratch slot) computed
        at the cost of two synchronizing device reads.

        At P == 1 every position starts a page, so the whole tensor is
        recorded -- the original behavior, without building an index.
        """
        if not self._paged:
            scratch_slots.append(slots)
            return
        # A positions/slots misalignment would free the WRONG pages silently,
        # so pin the one thing that makes the mapping meaningful (host-only).
        assert len(positions) == slots.numel(), (
            f"scratch slot bookkeeping expects one position per slot; got "
            f"{len(positions)} positions for {slots.numel()} slots"
        )
        page_size = self._page_size
        heads = [j for j, pos in enumerate(positions) if pos % page_size == 0]
        if not heads:
            return
        # Only reached on a page crossing (~1 row-step in P), where the free
        # below synchronizes inside the allocator anyway.
        scratch_slots.append(slots[self._h2d.to_device(heads, dtype=torch.int64)])

    def flush_scratch_frees(self) -> None:
        """Release the queued rare scratch slots back to the allocator.

        ``allocator.free()`` hides a device sync (``torch.unique`` of the page
        ids), so ``_free_scratch`` only queues these rare tensors; the manager
        calls this at idle time -- while waiting for commits the GPU is
        drained, so the sync costs nothing.
        """
        pending = self._pending_scratch_frees
        if not pending:
            return
        self._pending_scratch_frees = []
        allocator = self.model_runner.token_to_kv_pool_allocator
        for slots in pending:
            allocator.free(slots)

    def _free_scratch(
        self,
        scratch_batches: list[ScheduleBatch],
        scratch_slots: list[torch.Tensor],
        scratch_mamba_slots: list[torch.Tensor],
        scratch_kv_pages: list[torch.Tensor],
    ) -> None:
        if self._paged:
            # Page-granular free of exactly the round's throwaway pages, as
            # their head slots (see _track_scratch_slots for why the heads are
            # the complete set). A steady-state round records NONE of them --
            # every scratch write lands in an arena page or continues one --
            # so the paged round tail issues no device op at all and the CPU
            # runs straight into the next round while the GPU chain drains.
            live = [s for s in scratch_slots if s is not None and s.numel() > 0]
            if live:
                self._pending_scratch_frees.append(torch.cat(live))
        else:
            for slots in scratch_slots:
                if slots is not None and slots.numel() > 0:
                    self._pending_scratch_frees.append(slots)
        if len(self._pending_scratch_frees) > 32:
            # Cap the hoard: accept one rare in-round sync over unbounded
            # withholding of allocator pages.
            self.flush_scratch_frees()
        for batch in scratch_batches:
            for req in batch.reqs:
                if req.req_pool_idx is not None:
                    # Preset mamba slots are engine-owned (seat slots persist,
                    # transients return via scratch_mamba_slots below); null
                    # them so nothing downstream can ever double-free one.
                    req.mamba_pool_idx = None
                    self.model_runner.req_to_token_pool.free(req)
        for slots in scratch_mamba_slots:
            self._mamba_arena.give_back(slots)
        for pages in scratch_kv_pages:
            self._kv_page_arena.give_back(pages)
