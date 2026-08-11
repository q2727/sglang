"""Fused device ops for the decoupled enumeration drafter.

``fused_guess_topk`` replaces the guess-topk stage's dead-guess scatter +
``torch.topk`` pair (~12 launches: scatter + mbtopk's multi-pass radix
kernels) with two launches: a chunked local top-W scan and a tiny merge.
Top-F for the enumeration is tiny (W <= 8), so W sequential argmax passes
with a register-resident exclusion vector beat a general radix top-k at this
size; chunking the vocab across the grid restores parallelism (a single CTA
per row leaves the GPU >95% idle at enumeration batch sizes).

The dead-token exclusion (node a < K may never guess c_{a+1}) is computed
in-kernel from the row index -- no host-side mask preparation at all.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_CHUNKS = 16


@triton.jit
def _local_topk_kernel(
    logits_ptr,
    chains_ptr,
    cand_val_ptr,
    cand_idx_ptr,
    V,
    row_stride,
    CHUNK: tl.constexpr,
    HAS_DEAD: tl.constexpr,
    NODES: tl.constexpr,  # K + 1 rows per seat
    W: tl.constexpr,  # guess width (top-F)
    W_POW2: tl.constexpr,  # register-vector width (tl shapes need pow2)
    NUM_CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    # Node a < K's dead token is c_{a+1} = chains[b, a]; node K has none.
    dead = tl.full((), -1, tl.int64)
    if HAS_DEAD:
        b = row // NODES
        a = row % NODES
        is_dead_row = a < NODES - 1
        safe_off = b * (NODES - 1) + tl.where(is_dead_row, a, 0)
        dead = tl.where(is_dead_row, tl.load(chains_ptr + safe_off).to(tl.int64), -1)
    base = logits_ptr + row * row_stride
    lo = chunk * CHUNK
    hi = tl.minimum(lo + CHUNK, V)
    # Selected indices stay in a register vector: positional-where update,
    # broadcast-compare exclusion. (Round-tripping them through memory between
    # passes miscompiled; -1 slots never match a token id.)
    found = tl.full((W_POW2,), -1, tl.int64)
    out_base = (row * NUM_CHUNKS + chunk) * W_POW2
    for p in range(W):
        best_v = tl.full((), float("-inf"), tl.float32)
        best_i = tl.full((), -1, tl.int64)
        for start in range(lo, hi, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < hi
            vals = tl.load(base + cols, mask=mask, other=float("-inf")).to(tl.float32)
            cols64 = cols.to(tl.int64)
            vals = tl.where(cols64 == dead, float("-inf"), vals)
            taken = tl.max((found[:, None] == cols64[None, :]).to(tl.int32), axis=0)
            vals = tl.where(taken > 0, float("-inf"), vals)
            block_i = tl.argmax(vals, axis=0)
            block_v = tl.max(vals, axis=0)
            take = block_v > best_v
            best_i = tl.where(take, (start + block_i).to(tl.int64), best_i)
            best_v = tl.where(take, block_v, best_v)
        found = tl.where(tl.arange(0, W_POW2) == p, best_i, found)
        tl.store(cand_val_ptr + out_base + p, best_v)
        tl.store(cand_idx_ptr + out_base + p, best_i)
    for p in range(W, W_POW2):
        tl.store(cand_val_ptr + out_base + p, float("-inf"))
        tl.store(cand_idx_ptr + out_base + p, -1)


@triton.jit
def _merge_topk_kernel(
    cand_val_ptr,
    cand_idx_ptr,
    out_ptr,
    W: tl.constexpr,
    CANDS: tl.constexpr,  # NUM_CHUNKS * W_POW2, power of 2
):
    row = tl.program_id(0).to(tl.int64)
    vals = tl.load(cand_val_ptr + row * CANDS + tl.arange(0, CANDS))
    idxs = tl.load(cand_idx_ptr + row * CANDS + tl.arange(0, CANDS))
    big = tl.full((CANDS,), 2**62, tl.int64)
    for p in range(W):
        # torch.topk resolves value ties by lower index; candidates from
        # different chunks may tie, so pick (max value, min index).
        best_v = tl.max(vals, axis=0)
        tie_idxs = tl.where(vals == best_v, idxs, big)
        best_i = tl.min(tie_idxs, axis=0)
        slot = tl.argmin(tie_idxs, axis=0)
        tl.store(out_ptr + row * W + p, best_i)
        vals = tl.where(tl.arange(0, CANDS) == slot, float("-inf"), vals)


def fused_guess_topk(
    logits: torch.Tensor,  # [rows, V], rows = bs * (K + 1), row-contiguous
    chains_mat: Optional[torch.Tensor],  # [bs, K] int64 backbone tokens, or None
    *,
    nodes: int,  # K + 1
    width: int,  # top-F
) -> torch.Tensor:
    """Top-``width`` indices per row with per-node dead-token exclusion.

    Equivalent to ``logits.scatter_(dead, -inf); torch.topk(logits, width)``
    -- without mutating ``logits``. Returns [rows, width] int64.
    """
    rows, V = logits.shape
    assert logits.stride(-1) == 1, "fused_guess_topk needs row-contiguous logits"
    assert width <= 8, "enumeration width is tiny by design"
    w_pow2 = triton.next_power_of_2(width)
    cands = _CHUNKS * w_pow2
    cand_vals = torch.empty((rows, cands), dtype=torch.float32, device=logits.device)
    cand_idxs = torch.empty((rows, cands), dtype=torch.int64, device=logits.device)
    out = torch.empty((rows, width), dtype=torch.int64, device=logits.device)
    _local_topk_kernel[(rows, _CHUNKS)](
        logits,
        chains_mat if chains_mat is not None else logits,  # unused when HAS_DEAD=0
        cand_vals,
        cand_idxs,
        V,
        logits.stride(0),
        CHUNK=triton.cdiv(V, _CHUNKS),
        HAS_DEAD=chains_mat is not None,
        NODES=nodes,
        W=width,
        W_POW2=w_pow2,
        NUM_CHUNKS=_CHUNKS,
        BLOCK=_BLOCK,
        num_warps=4,
    )
    _merge_topk_kernel[(rows,)](
        cand_vals,
        cand_idxs,
        out,
        W=width,
        CANDS=cands,
        num_warps=1,
    )
    return out


@triton.jit
def _mirror_snapshot(
    rows_ptr,
    seat,
    base_out_len,
    ROW_WORDS: tl.constexpr,
    SLOTS: tl.constexpr,
    W_PAD: tl.constexpr,
):
    """Select and snapshot the seat's CONSUMABLE slot: the seqlock-coherent
    slot whose wire segment [pre_len, pre_len + wire) straddles the caller's
    committed base. Judgement is pure arithmetic between the slot's OWN
    pre_len and the caller-sampled base (the verifier-select stamp recipe):
    a landing that is late, early, not yet visible, or mid-overwrite simply
    fails the check and the caller junks the round -- there is nothing to
    match exactly and nothing to tear.

    Returns (seg_ok, skip, dlen, wire, toks[W_PAD]) with toks snapshotted to
    registers and seqlock-rechecked AFTER the reads (the recheck is what
    makes the snapshot immune to an overwrite racing the loads)."""
    sel_taken = 0
    sel_seq = -1
    sel_pre = 0
    sel_wire = 0
    sel_off = 0
    for s in tl.static_range(SLOTS):
        off = (seat * SLOTS + s) * ROW_WORDS
        s1 = tl.load(rows_ptr + off + ROW_WORDS - 1, volatile=True)
        pre = tl.load(rows_ptr + off + 1, volatile=True)
        wire = tl.load(rows_ptr + off + 2, volatile=True)
        s0 = tl.load(rows_ptr + off + 0, volatile=True)
        coherent = (s0 == s1) & (s1 > 0)
        skip_s = base_out_len - pre
        dlen_s = pre + wire - base_out_len
        fits = (
            coherent
            & (skip_s >= 0)
            & (dlen_s >= 1)
            & (dlen_s <= wire)
            & (wire >= 1)
            & (wire <= ROW_WORDS - 4)
        )
        take = fits & (s1 > sel_seq)
        sel_taken = tl.where(take, 1, sel_taken)
        sel_seq = tl.where(take, s1, sel_seq)
        sel_pre = tl.where(take, pre, sel_pre)
        sel_wire = tl.where(take, wire, sel_wire)
        sel_off = tl.where(take, off, sel_off)
    lanes = tl.arange(0, W_PAD)
    toks = tl.load(
        rows_ptr + sel_off + 3 + lanes,
        mask=lanes < tl.maximum(sel_wire, 1),
        other=0,
        volatile=True,
    )
    # Seqlock recheck AFTER the payload loads: an overwrite in progress
    # bumped seq0 first, so a mismatch here means the token snapshot may be
    # torn -- invalidate instead of consuming.
    s1b = tl.load(rows_ptr + sel_off + ROW_WORDS - 1, volatile=True)
    s0b = tl.load(rows_ptr + sel_off + 0, volatile=True)
    seg_ok = (sel_taken == 1) & (s0b == sel_seq) & (s1b == sel_seq)
    skip = base_out_len - sel_pre
    dlen = sel_pre + sel_wire - base_out_len
    return seg_ok, skip, dlen, sel_wire, sel_pre, toks, sel_off


@triton.jit
def _tok_at(toks, lanes, idx):
    return tl.sum(tl.where(lanes == idx, toks, 0))


@triton.jit
def _commit_match_kernel(
    rows_ptr,
    units_ptr,
    backbone_ptr,
    out_ptr,
    latch_ptr,
    probe_ptr,
    seat,
    backbone_len,
    base_out_len,
    K: tl.constexpr,
    F: tl.constexpr,
    W_UNIT: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    SLOTS: tl.constexpr,
    W_PAD: tl.constexpr,
    LATCH: tl.constexpr,
):
    seg_ok, skip, dlen, wire, pre_len, toks, sel_off = _mirror_snapshot(
        rows_ptr, seat, base_out_len, ROW_WORDS, SLOTS, W_PAD
    )
    lanes = tl.arange(0, W_PAD)
    case = dlen - 1
    prefix_ok = seg_ok & (case >= 0) & (case <= K) & (case <= backbone_len)
    skip_c = tl.minimum(tl.maximum(skip, 0), ROW_WORDS - 5)
    fail_i = -1
    for i in tl.static_range(K):
        d = _tok_at(toks, lanes, tl.minimum(skip_c + i, W_PAD - 1))
        b = tl.load(backbone_ptr + i)
        prefix_ok &= (i >= case) | (d == b)
        fail_i = tl.where((i < case) & (d != b) & (fail_i < 0), i, fail_i)
    case_c = tl.minimum(tl.maximum(case, 0), K)
    bonus = _tok_at(toks, lanes, tl.minimum(skip_c + case_c, W_PAD - 1))
    f_found = -1
    for f in tl.static_range(F):
        g = tl.load(units_ptr + (case_c * F + f) * W_UNIT)
        f_found = tl.where((g == bonus) & (f_found < 0), f, f_found)
    hit = prefix_ok & (f_found >= 0)
    # 0 = no consumable segment (nothing landed for this base yet, an old /
    # re-sent / later segment, or a torn snapshot): state untouched, the
    # host round redoes everything.
    verdict = tl.where(~seg_ok, 0, tl.where(hit, 2, 1))
    tl.store(out_ptr + 0, verdict.to(tl.int64))
    tl.store(out_ptr + 1, case_c.to(tl.int64))
    tl.store(out_ptr + 2, f_found.to(tl.int64))
    tl.store(out_ptr + 3, (pre_len + wire).to(tl.int64))
    if LATCH:
        # Early-judge mode: extend the verdict with the full judgment frame
        # and snapshot the row's tokens, so the gated scatter (which runs
        # only after the previous armed sequence drains) can replay THIS
        # judgment instead of re-reading the live mirror -- a burst landing
        # between the two reads would otherwise give host and kernel two
        # different truths (the D10 two-coordinate-system trap).
        tl.store(out_ptr + 4, tl.where(seg_ok, 1, 0).to(tl.int64))
        tl.store(out_ptr + 5, skip.to(tl.int64))
        tl.store(out_ptr + 6, dlen.to(tl.int64))
        tl.store(out_ptr + 7, wire.to(tl.int64))
        tl.store(latch_ptr + lanes, toks)
        # Debug taps: what THIS execution actually loaded (the host-side
        # dump of the same tensors is useless here -- its tolist syncs the
        # main stream and reads the NEXT generation).
        for i in tl.static_range(3):
            tl.store(
                out_ptr + 8 + i,
                tl.load(backbone_ptr + tl.minimum(i, K - 1)).to(tl.int64),
            )
        tl.store(out_ptr + 11, tl.load(units_ptr + (case_c * F + 0) * W_UNIT))
        tl.store(
            out_ptr + 12,
            tl.load(units_ptr + (case_c * F + tl.minimum(1, F - 1)) * W_UNIT),
        )
        # Generation fingerprint: the scatter verdict buffer's CURRENT
        # (verdict, dlen). If these match THIS commit at judge time, the
        # same-generation scatter has already run -- the seq's wait on the
        # judge event did not hold.
        tl.store(out_ptr + 13, tl.load(probe_ptr + 0))
        tl.store(out_ptr + 14, tl.load(probe_ptr + 3))
        tl.store(out_ptr + 15, tl.extra.cuda.globaltimer())
        # Prefix forensics (acc 3.77 hunt): the exact segment tokens this
        # execution compared, the bonus it looked up, and the first prefix
        # position that failed -- the audit lines these up against the host
        # committed-token chain to tell WHICH side is the wrong generation.
        for i in tl.static_range(3):
            tl.store(
                out_ptr + 16 + i,
                _tok_at(toks, lanes, tl.minimum(skip_c + i, W_PAD - 1)),
            )
        tl.store(out_ptr + 19, bonus)
        tl.store(out_ptr + 20, fail_i.to(tl.int64))
        # Slot hologram: the base this execution was handed, the slot it
        # picked, and BOTH slots' full headers as it saw them -- the final
        # discriminator between "picked the wrong slot", "slot holds stale
        # payload under a fresh header", and "was handed a stale base".
        tl.store(out_ptr + 21, base_out_len)
        tl.store(out_ptr + 22, sel_off)
        for s in tl.static_range(SLOTS):
            hoff = (seat * SLOTS + s) * ROW_WORDS
            tl.store(out_ptr + 24 + s * 4 + 0, tl.load(rows_ptr + hoff + 0))
            tl.store(out_ptr + 24 + s * 4 + 1, tl.load(rows_ptr + hoff + 1))
            tl.store(out_ptr + 24 + s * 4 + 2, tl.load(rows_ptr + hoff + 2))
            tl.store(
                out_ptr + 24 + s * 4 + 3,
                tl.load(rows_ptr + hoff + ROW_WORDS - 1),
            )


def commit_match(
    *,
    mirror,
    seat: int,
    units_dev: torch.Tensor,  # [K+1, F, K+1] block (or [C*F] guess plane)
    backbone_dev: torch.Tensor,  # [K] device backbone twin
    backbone_len: int,
    base_out_len: int,
    num_steps: int,
    fanout: int,
    unit_stride: Optional[int] = None,  # per-unit stride; default K+1 (block)
    out_buf: Optional[torch.Tensor] = None,  # [>=8] i64 static (judge mode)
    latch: Optional[torch.Tensor] = None,  # [W_PAD] i64 static (judge mode)
    probe: Optional[torch.Tensor] = None,  # debug generation fingerprint
) -> torch.Tensor:
    """The drafter-side GPU replica of ``_match_seat``, fed from the commit
    mirror: out = [verdict(0=no consumable segment, 1=miss, 2=hit), case, f,
    new_total]. Self-consistent by construction -- the consumable segment is
    picked by pure arithmetic between each slot's own pre_len and the
    caller-sampled committed base, exactly the verifier-select stamp
    recipe. With ``out_buf``+``latch`` it doubles as the early-judge kernel:
    out[4..7] = (seg_ok, skip, dlen, wire) and the row tokens latch."""
    out = (
        out_buf
        if out_buf is not None
        else torch.empty(4, dtype=torch.int64, device=units_dev.device)
    )
    _commit_match_kernel[(1,)](
        mirror.rows,
        units_dev,
        backbone_dev,
        out,
        latch if latch is not None else out,
        probe if probe is not None else out,
        seat,
        backbone_len,
        base_out_len,
        K=num_steps,
        F=fanout,
        W_UNIT=unit_stride if unit_stride is not None else num_steps + 1,
        ROW_WORDS=mirror.row_words,
        SLOTS=2,
        W_PAD=triton.next_power_of_2(mirror.width),
        LATCH=latch is not None,
    )
    return out


@triton.jit
def _commit_seg_kernel(
    rows_ptr,
    out_ptr,
    seat,
    base_out_len,
    ROW_WORDS: tl.constexpr,
    SLOTS: tl.constexpr,
    W_PAD: tl.constexpr,
):
    """The SEGMENT half of the judge: seg_ok/skip/dlen/wire from the mirror
    alone -- no guesses, no backbone. This is the only bit of the judgment
    the host truly has to wait for ("did the GPU consume this commit"); it
    is pure arithmetic over the landed row and the armed base, so it can sit
    directly behind the commit gate with NO ordering on the previous
    sequence's products. The (case, f) match half stays on the judge stream
    behind the inputs event and is consumed as an after-the-fact audit."""
    seg_ok, skip, dlen, wire, pre_len, toks, sel_off = _mirror_snapshot(
        rows_ptr, seat, base_out_len, ROW_WORDS, SLOTS, W_PAD
    )
    tl.store(out_ptr + 0, tl.where(seg_ok, 1, 0).to(tl.int64))
    tl.store(out_ptr + 1, skip.to(tl.int64))
    tl.store(out_ptr + 2, dlen.to(tl.int64))
    tl.store(out_ptr + 3, wire.to(tl.int64))
    # Phase anchor: seg runs the moment dispatch launches it (empty stream,
    # no waits), so this stamps "dispatch N" on the same device clock the
    # match half stamps into its pin[15] -- their delta is the match's true
    # execution lag with no host/GPU clock calibration.
    tl.store(out_ptr + 4, tl.extra.cuda.globaltimer())


def commit_seg(
    *,
    mirror,
    seat: int,
    base_out_len: int,
    out_buf: torch.Tensor,  # [>=5] i64 static
) -> None:
    _commit_seg_kernel[(1,)](
        mirror.rows,
        out_buf,
        seat,
        base_out_len,
        ROW_WORDS=mirror.row_words,
        SLOTS=2,
        W_PAD=triton.next_power_of_2(mirror.width),
    )


@triton.jit
def _pack_guesses_kernel(
    plane_ptr,  # [C*F] i64 persistent per-seat guess plane
    guesses_ptr,  # [C, GW] i64 topk-graph static output
    rowmap_ptr,  # [C*F] i32: full-grid position -> chain row, -1 dead
    verdict_ptr,  # [>=1] i64 from commit_scatter: [0]=verdict
    GW,
    C: tl.constexpr,
    F: tl.constexpr,
):
    """The guess half of _pack_units_kernel, run right after the topk so the
    NEXT round's early judge can match against exactly the values the pack
    will publish (same live-mask poisoning), a whole chain graph earlier
    than the packed block exists."""
    pid = tl.program_id(0)
    c = pid // F
    f = pid % F
    verdict = tl.load(verdict_ptr)
    hit = verdict == 2
    row = tl.load(rowmap_ptr + pid)
    live = (row >= 0) & (hit | (c == 0))
    g = tl.load(guesses_ptr + c * GW + tl.minimum(f, GW - 1))
    tl.store(plane_ptr + pid, tl.where((f < GW) & live, g, -1))


def pack_guesses(
    *,
    plane: torch.Tensor,  # [C*F] i64
    guesses: torch.Tensor,  # [C, GW] i64
    rowmap: torch.Tensor,  # [C*F] i32
    verdict: torch.Tensor,
    num_steps: int,
    fanout: int,
) -> None:
    num_cases = num_steps + 1
    _pack_guesses_kernel[(num_cases * fanout,)](
        plane,
        guesses,
        rowmap,
        verdict,
        guesses.shape[-1],
        C=num_cases,
        F=fanout,
    )


@triton.jit
def _pack_units_kernel(
    units_ptr,  # [C, F, K+1] i64 persistent per-seat block buffer
    guesses_ptr,  # [C, GW] i64 topk-graph static output
    chains_ptr,  # [K, ROWS] i64 (stacked chain-graph static outputs)
    rowmap_ptr,  # [C*F] i32: full-grid position -> chain output row, -1 dead
    verdict_ptr,  # [>=4] i64 from commit_scatter: [0]=verdict
    GW,  # guess tensor width (f_live, or F for pre-poisoned budget grids)
    ROWS,  # chain output rows
    C: tl.constexpr,
    F: tl.constexpr,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
):
    """Device-side block pack + miss poisoning.

    One program per (case, f) lane. Replaces the dispatch-time host pack
    (units cat/stack + poisoning) AND the whole host case-0
    re-draft: on a miss the armed grid's case-0 lanes are already the case-0
    round's product (scatter feeds -1 chains to the topk on a miss, so its
    node-0 guesses are exclusion-free, and the chain graph forked the case-0
    branches from the post-advance seat state) -- the only thing the host
    path added was poisoning the dead lanes, which is this kernel's `live`
    mask. Enqueued at arm behind the gate: zero host work at round time.
    """
    pid = tl.program_id(0)
    c = pid // F
    f = pid % F
    verdict = tl.load(verdict_ptr)
    hit = verdict == 2
    row = tl.load(rowmap_ptr + pid)
    live = (row >= 0) & (hit | (c == 0))
    g = tl.load(guesses_ptr + c * GW + tl.minimum(f, GW - 1))
    g = tl.where((f < GW) & live, g, -1)
    tl.store(units_ptr + pid * (K + 1), g)
    i = tl.arange(0, K_POW2)
    mi = i < K
    row_c = tl.maximum(row, 0)
    cv = tl.load(chains_ptr + i * ROWS + tl.minimum(row_c, ROWS - 1), mask=mi, other=0)
    cv = tl.where(live, cv, 0)
    tl.store(units_ptr + pid * (K + 1) + 1 + i, cv, mask=mi)
    # The backbone is NOT written here: the next round's prefix key is the
    # winning chain of the block this round's scatter MATCHED (one round
    # older than the block being packed), and commit_scatter already emits
    # exactly that as chains_out -- the caller copies it. Writing the fresh
    # chains here instead was an off-by-one-round bug (acc 3.94 -> 2.55).


def pack_units(
    *,
    units_buf: torch.Tensor,  # [C, F, K+1] i64
    guesses: torch.Tensor,  # [C, GW] i64
    chains_stacked: torch.Tensor,  # [K, ROWS] i64
    rowmap: torch.Tensor,  # [C*F] i32
    verdict: torch.Tensor,  # commit_scatter verdict buffer
    num_steps: int,
    fanout: int,
) -> None:
    num_cases = num_steps + 1
    _pack_units_kernel[(num_cases * fanout,)](
        units_buf,
        guesses,
        chains_stacked,
        rowmap,
        verdict,
        guesses.shape[-1],
        chains_stacked.shape[-1],
        C=num_cases,
        F=fanout,
        K=num_steps,
        K_POW2=triton.next_power_of_2(max(num_steps, 1)),
    )


@triton.jit
def _out_loc_stacks_kernel(
    out_ptr,  # [S, C, R*W] int64: per (stale-shift, accept-case) write slots
    alloc_ptr,  # [ALLOC_W] int64: the skeleton's widened delta allocation
    pad_flats_ptr,  # [PAD_SPAN] int64: the seat's junk pad plane (paged layout)
    glue_flats_ptr,  # [R-1, SPAN_MAX] int64: glue rows' private flat slots
    base_len,  # skeleton base at build (stale; +s gives the consume base)
    S: tl.constexpr,  # shifts (K + 2)
    C: tl.constexpr,  # accept cases (K + 1)
    R: tl.constexpr,  # fused rows per seat (K + 1: seat + K glue)
    W: tl.constexpr,  # padded row width (2K + 1)
    PAGE: tl.constexpr,
    ALLOC_W: tl.constexpr,
    PAD_SPAN: tl.constexpr,
    SPAN_MAX: tl.constexpr,
    W_POW2: tl.constexpr,
):
    """One-launch replacement for the restage's out_cache_loc composition.

    The host used to assemble S*C rows out of ~6 tiny cat/slice ops each
    (~200 dispatches per restage); every entry is pure index arithmetic over
    four inputs, so one program per (s, c, r) writes its W-slot row directly:

      seat row (r == 0): delta slots straight from the widened allocation at
        offset s (alloc slots are position-aligned, so the stale-consume
        shift IS the fix), then junk pads at in-page offset (base+s+dlen)%P;
      glue row r >= 1: the row's private window at in-page offset
        (base+s)%P -- an arena page hosts any offset, tail+W <= SPAN_MAX.
    """
    pid = tl.program_id(0)
    s = pid // (C * R)
    c = (pid % (C * R)) // R
    r = pid % R
    dlen = c + 1
    w = tl.arange(0, W_POW2)
    mask = w < W
    if r == 0:
        pad_offset = (base_len + s + dlen) % PAGE
        pad_idx = tl.minimum(pad_offset + tl.maximum(w - dlen, 0), PAD_SPAN - 1)
        pads = tl.load(pad_flats_ptr + pad_idx, mask=mask, other=0)
        alloc_idx = tl.minimum(s + w, ALLOC_W - 1)
        delta = tl.load(alloc_ptr + alloc_idx, mask=mask, other=0)
        val = tl.where(w < dlen, delta, pads)
    else:
        tail = (base_len + s) % PAGE
        val = tl.load(
            glue_flats_ptr + (r - 1) * SPAN_MAX + tail + w, mask=mask, other=0
        )
    tl.store(out_ptr + ((s * C + c) * R + r) * W + w, val, mask=mask)


def build_out_loc_stacks(
    *,
    alloc_slots: torch.Tensor,  # [ALLOC_W] int64
    pad_flats: torch.Tensor,  # [PAD_SPAN] int64
    glue_flats: torch.Tensor,  # [R-1, SPAN_MAX] int64 (contiguous)
    base_len: int,
    num_steps: int,
    width: int,
    page_size: int,
) -> torch.Tensor:
    """All (shift, case) out_cache_loc rows in one launch -> [S, C, R*W]."""
    num_shifts = num_steps + 2
    num_cases = num_steps + 1
    rows = num_steps + 1
    out = torch.empty(
        num_shifts,
        num_cases,
        rows * width,
        dtype=torch.int64,
        device=alloc_slots.device,
    )
    _out_loc_stacks_kernel[(num_shifts * num_cases * rows,)](
        out,
        alloc_slots,
        pad_flats,
        glue_flats,
        base_len,
        S=num_shifts,
        C=num_cases,
        R=rows,
        W=width,
        PAGE=page_size,
        ALLOC_W=alloc_slots.numel(),
        PAD_SPAN=pad_flats.numel(),
        SPAN_MAX=glue_flats.shape[1],
        W_POW2=triton.next_power_of_2(width),
    )
    return out


@triton.jit
def _commit_scatter_kernel(
    # mirror (seat-indexed seqlock rows)
    rows_ptr,
    # last block
    units_ptr,
    backbone_ptr,
    # per-case stacks (restage-prebuilt)
    gather_stack_ptr,  # [K+1, ROWS_W] int64 (source-gather indices per delta)
    out_loc_stack_ptr,  # [K+1, ROWS_W] int64
    true_stack_ptr,  # [K+1, ROWS] int32
    node_stack_ptr,  # [K+1, ROWS] int64 (node-logit gather offsets per delta)
    pad_flats_ptr,  # [PADS] int64 (this seat's pad plane, paged layout)
    # outputs
    verdict_ptr,  # [4] int64: verdict, case, f, new_total(delta only here)
    input_out_ptr,  # [ROWS_W] int64
    out_loc_out_ptr,  # [ROWS_W] int64
    true_out_ptr,  # [ROWS] int32
    node_out_ptr,  # [ROWS] int64
    chains_out_ptr,  # [K] int64
    table_ptr,  # page-table row base (real or shadow); pads land in its
    # columns [base_len + delta, base_len + WIDTH)
    base_lens_ptr,  # [seats] int64, seat-indexed: committed base length
    # early-judge inputs (read only when FROM_JUDGE; see commit_match LATCH)
    judge_ptr,  # [8] int64: verdict, case, f, new_total, seg_ok, skip, dlen, wire
    latch_ptr,  # [W_PAD] int64: the judged row's token snapshot
    # scalars
    seat,
    backbone_len,
    base_out_len,
    seq_no,  # monotone arm counter; written to verdict[10] as a generation tag
    K: tl.constexpr,
    F: tl.constexpr,
    W_UNIT: tl.constexpr,
    ROWS_W: tl.constexpr,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGE: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    SLOTS: tl.constexpr,
    W_PAD: tl.constexpr,
    FROM_JUDGE: tl.constexpr,
):
    if FROM_JUDGE:
        # Early-judge mode: the judgment and the row snapshot were taken by
        # commit_match(latch=...) on the judge stream the moment the commit
        # landed, and the HOST has already routed on them. Replay exactly
        # that frame -- re-reading the live mirror here (a whole armed
        # sequence later) could see a newer generation after a burst and
        # hand the host and this kernel two different truths (D10).
        lanes = tl.arange(0, W_PAD)
        verdict = tl.load(judge_ptr + 0)
        case_c = tl.load(judge_ptr + 1)
        f_found = tl.load(judge_ptr + 2)
        seg_ok = tl.load(judge_ptr + 4) == 1
        skip = tl.load(judge_ptr + 5)
        dlen = tl.load(judge_ptr + 6)
        wire_len = tl.load(judge_ptr + 7)
        pre_len = base_out_len - skip
        toks = tl.load(latch_ptr + lanes)
        hit = verdict == 2
        skip_c = tl.minimum(tl.maximum(skip, 0), W_PAD - 1)
        bonus = _tok_at(toks, lanes, tl.minimum(skip_c + case_c, W_PAD - 1))
    else:
        seg_ok, skip, dlen, wire_len, pre_len, toks, _sel_off = _mirror_snapshot(
            rows_ptr, seat, base_out_len, ROW_WORDS, SLOTS, W_PAD
        )
        lanes = tl.arange(0, W_PAD)
        case = dlen - 1
        prefix_ok = seg_ok & (case >= 0) & (case <= K) & (case <= backbone_len)
        skip_c = tl.minimum(tl.maximum(skip, 0), W_PAD - 1)
        for i in tl.static_range(K):
            d = _tok_at(toks, lanes, tl.minimum(skip_c + i, W_PAD - 1))
            b = tl.load(backbone_ptr + i)
            prefix_ok &= (i >= case) | (d == b)
        case_c = tl.minimum(tl.maximum(case, 0), K)
        bonus = _tok_at(toks, lanes, tl.minimum(skip_c + case_c, W_PAD - 1))
        f_found = -1
        for f in tl.static_range(F):
            g = tl.load(units_ptr + (case_c * F + f) * W_UNIT)
            f_found = tl.where((g == bonus) & (f_found < 0), f, f_found)
        hit = prefix_ok & (f_found >= 0)
        # 0 = no consumable segment for this base (nothing landed yet / old /
        # later / torn): the junk lane below leaves the seat untouched and the
        # host round redoes everything.
        verdict = tl.where(~seg_ok, 0, tl.where(hit, 2, 1))
    tl.store(verdict_ptr + 0, verdict.to(tl.int64))
    tl.store(verdict_ptr + 1, case_c.to(tl.int64))
    tl.store(verdict_ptr + 2, f_found.to(tl.int64))
    tl.store(verdict_ptr + 3, dlen.to(tl.int64))
    # Debug taps (holo): what THIS execution actually loaded.
    tl.store(verdict_ptr + 4, bonus.to(tl.int64))
    g0_dbg = tl.load(units_ptr + (case_c * F + 0) * W_UNIT)
    g1_dbg = tl.load(units_ptr + (case_c * F + tl.minimum(1, F - 1)) * W_UNIT)
    tl.store(verdict_ptr + 5, g0_dbg.to(tl.int64))
    tl.store(verdict_ptr + 6, g1_dbg.to(tl.int64))
    tl.store(verdict_ptr + 7, tl.where(seg_ok, 1, 0).to(tl.int64))
    tl.store(verdict_ptr + 8, pre_len.to(tl.int64))
    tl.store(verdict_ptr + 9, wire_len.to(tl.int64))
    tl.store(verdict_ptr + 10, seq_no)
    stale_row = verdict == 0
    f_c = tl.maximum(f_found, 0)
    # E: winning chain (junk zeros on miss -- its cells are dead by theorem)
    for i in tl.static_range(K):
        c = tl.load(units_ptr + (case_c * F + f_c) * W_UNIT + 1 + i)
        # -1 on miss/stale: the guess-tail's dead-token exclusion COMPARES
        # against these values, and no real token is -1 (0 would wrongly
        # exclude a live vocab id from the node-0 top-F on miss rounds).
        tl.store(chains_out_ptr + i, tl.where(hit & ~stale_row, c, -1))
    # A: input assembly out[j] = src[gather[case, j]], src = delta ++ chain
    for j in tl.static_range(ROWS_W):
        gidx = tl.load(gather_stack_ptr + case_c * ROWS_W + j)
        from_delta = gidx < dlen
        dval = _tok_at(toks, lanes, tl.minimum(skip_c + gidx, W_PAD - 1))
        cidx = tl.minimum(tl.maximum(gidx - dlen, 0), K - 1)
        craw = tl.load(units_ptr + (case_c * F + f_c) * W_UNIT + 1 + cidx)
        cval = tl.where(hit, craw, 0)
        val = tl.where(from_delta, dval, cval)
        tl.store(input_out_ptr + j, tl.where(stale_row, 0, val))
    # B / C: per-case staging selection
    for j in tl.static_range(ROWS_W):
        v = tl.load(out_loc_stack_ptr + case_c * ROWS_W + j)
        tl.store(out_loc_out_ptr + j, v)
    for r in tl.static_range(ROWS):
        v = tl.load(true_stack_ptr + case_c * ROWS + r)
        # Stale -> zero-length GDN scan: the seat's recurrent state is left
        # untouched by a junk round (the timeout junk lane's key property).
        tl.store(true_out_ptr + r, tl.where(stale_row, 0, v))
    for r in tl.static_range(ROWS):
        nv = tl.load(node_stack_ptr + case_c * ROWS + r)
        tl.store(node_out_ptr + r, nv)
    # D: seat-row pad table entries, columns [base+delta, base+WIDTH).
    # The base is read from the seat-indexed device buffer, not passed down as
    # a host scalar: the column a pad lands at is a "where", and every "where"
    # in this kernel already comes from the device (dlen from the mirror). That
    # is what lets the skeleton stop depending on the host knowing the base.
    base_len = tl.load(base_lens_ptr + seat)
    new_len = base_len + dlen
    pad_offset = new_len - (new_len // PAGE) * PAGE
    for i in tl.static_range(WIDTH):
        col = dlen + i  # relative to base
        in_pad = (col >= dlen) & (col < WIDTH)
        pval = tl.load(pad_flats_ptr + tl.minimum(pad_offset + i, PAGE + WIDTH - 2))
        tl.store(
            table_ptr + base_len + col,
            pval.to(tl.int32),
            mask=in_pad & ~stale_row,
        )


def commit_scatter(
    *,
    mirror,
    seat: int,
    units_dev: torch.Tensor,
    backbone_dev: torch.Tensor,
    backbone_len: int,
    gather_stack: torch.Tensor,
    out_loc_stack: torch.Tensor,
    true_stack: torch.Tensor,
    node_stack: torch.Tensor,
    pad_flats: torch.Tensor,
    table_row: torch.Tensor,  # 1-D view of the seat row (real or shadow)
    base_lens_dev: torch.Tensor,  # [req_pool] int64, seat-indexed base length
    base_out_len: int,
    num_steps: int,
    fanout: int,
    page_size: int,
    extend_width: int,
    outs: tuple,  # (verdict[4]i64, input[ROWS_W]i64, out_loc[ROWS_W]i64, true[ROWS]i32, node[ROWS]i64, chains[K]i64)
    judge: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # (judge_buf, latch)
    seq_no: int = 0,
) -> None:
    """One-launch commit consumption: the on-GPU match plus every per-case
    value the fused-extend replay needs (input assembly, out_cache_loc /
    GDN true-lens selection, seat-pad page-table suffix), fed entirely from
    the commit mirror. Stale generation writes only the verdict -- the
    caller's junk-lane / fallback handles the rest. With ``judge`` the match
    half is skipped: the early-judge kernel already took the judgment and
    the row snapshot the moment the commit landed, and this kernel replays
    that exact frame."""
    verdict, input_out, out_loc_out, true_out, node_out, chains_out = outs
    rows = true_stack.shape[1]
    rows_w = gather_stack.shape[1]
    judge_buf, latch = judge if judge is not None else (verdict, verdict)
    _commit_scatter_kernel[(1,)](
        mirror.rows,
        units_dev,
        backbone_dev,
        gather_stack,
        out_loc_stack,
        true_stack,
        node_stack,
        pad_flats,
        verdict,
        input_out,
        out_loc_out,
        true_out,
        node_out,
        chains_out,
        table_row,
        base_lens_dev,
        judge_buf,
        latch,
        seat,
        backbone_len,
        base_out_len,
        seq_no,
        K=num_steps,
        F=fanout,
        W_UNIT=num_steps + 1,
        ROWS_W=rows_w,
        ROWS=rows,
        WIDTH=extend_width,
        PAGE=page_size,
        ROW_WORDS=mirror.row_words,
        SLOTS=2,
        W_PAD=triton.next_power_of_2(mirror.width),
        FROM_JUDGE=judge is not None,
    )


@triton.jit(
    # Per-round host ints: triton's default divisibility specialization
    # would recompile per {==1, %16, other} class combination -- dozens of
    # mid-run compile stalls riding the dispatch tail (each one a chance
    # for the verifier to fall back). One compile per W_POW2 bucket only,
    # all prewarmed at engine init (see prewarm_armed_cow_locs).
    do_not_specialize=[
        "anchor",
        "tail",
        "slot_shift",
        "span_sup",
        "alloc_w",
        "span_max",
        "flats_w",
    ]
)
def _armed_cow_locs_kernel(
    src_out_ptr,  # [TOTAL] int64 (flat COW source slots, row-major)
    dst_out_ptr,  # [TOTAL] int64 (flat COW target slots, aligned with src)
    committed_ptr,  # [>= anchor + tail] int64 (seat committed flat slots)
    alloc_ptr,  # [alloc_w] int64 (the skeleton's widened delta allocation)
    glue_flats_ptr,  # [K, span_max] int64 (glue rows' private flat slots)
    branch_flats_ptr,  # [ROWS_POOL, flats_w] int64 (branch private slots)
    sel_rows_ptr,  # [ROWS_SEL] int64 (selected row -> carrier pool row)
    case_ptr,  # [ROWS_SEL] int64 (accept case per selected row)
    case_prefix_ptr,  # [ROWS_SEL] int64 (exclusive prefix sum of case)
    anchor,  # page_floor(base_len)
    tail,  # base_len - anchor
    slot_shift,  # stale-skeleton re-base into the widened allocation
    span_sup,  # tail + K + 1 (case-0 span; case c spans span_sup + c)
    alloc_w,
    span_max,
    flats_w,
    W_POW2: tl.constexpr,
):
    """One-launch replacement for the armed branch-head COW's host assembly
    (a per-row python loop of slice views + two big cats). One program per
    selected row writes its span of the flat (src, dst) index lists:

      case 0: seat boundary tail (committed[anchor:]) then this round's K+1
        delta candidates inside the widened allocation (slot_shift re-bases
        a stale skeleton's window);
      case c >= 1: glue row c-1's private head, span_sup + c slots.

    Row r's output offset = r * span_sup + prefix(case) -- the case prefix
    sum is variant-static and cached on device."""
    row = tl.program_id(0)
    case = tl.load(case_ptr + row)
    pool_row = tl.load(sel_rows_ptr + row)
    span = span_sup + case
    out0 = row * span_sup + tl.load(case_prefix_ptr + row)
    w = tl.arange(0, W_POW2)
    mask = w < span
    if case == 0:
        boundary = tl.load(committed_ptr + anchor + w, mask=mask & (w < tail), other=0)
        alloc_idx = tl.minimum(slot_shift + w - tail, alloc_w - 1)
        delta = tl.load(alloc_ptr + alloc_idx, mask=mask & (w >= tail), other=0)
        src = tl.where(w < tail, boundary, delta)
    else:
        src = tl.load(glue_flats_ptr + (case - 1) * span_max + w, mask=mask, other=0)
    dst = tl.load(branch_flats_ptr + pool_row * flats_w + w, mask=mask, other=0)
    tl.store(src_out_ptr + out0 + w, src, mask=mask)
    tl.store(dst_out_ptr + out0 + w, dst, mask=mask)


def build_armed_cow_locs(
    *,
    committed: torch.Tensor,  # [>= base_len] int64
    alloc: torch.Tensor,  # [ALLOC_W] int64, contiguous
    glue_flats: torch.Tensor,  # [K, SPAN_MAX] int64 (contiguous)
    branch_flats: torch.Tensor,  # [ROWS_POOL, FLATS_W] int64 (contiguous)
    sel_rows: torch.Tensor,  # [ROWS_SEL] int64 (variant-static, device)
    case_of_row: torch.Tensor,  # [ROWS_SEL] int64 (variant-static, device)
    case_prefix: torch.Tensor,  # [ROWS_SEL] int64 (variant-static, device)
    case_sum: int,  # variant-static sum(case_of_row)
    anchor: int,
    tail: int,
    slot_shift: int,
    num_steps: int,
) -> torch.Tensor:
    """The armed branch-head COW's (src, dst) flat slot lists in one launch
    -> [2, TOTAL] (row 0 = src, row 1 = dst), ready for ``move_kv_cache``.
    Enqueued pre-gate: every input it reads at gate release must be pinned
    by the caller's keepalive (committed rebinding) or outlive the round
    (variant/carrier tensors)."""
    rows_sel = sel_rows.numel()
    span_sup = tail + num_steps + 1
    out = torch.empty(
        2, rows_sel * span_sup + case_sum, dtype=torch.int64, device=committed.device
    )
    _armed_cow_locs_kernel[(rows_sel,)](
        out[0],
        out[1],
        committed,
        alloc,
        glue_flats,
        branch_flats,
        sel_rows,
        case_of_row,
        case_prefix,
        anchor,
        tail,
        slot_shift,
        span_sup,
        alloc.numel(),
        glue_flats.shape[1],
        branch_flats.shape[1],
        W_POW2=triton.next_power_of_2(span_sup + num_steps),
    )
    return out


def prewarm_armed_cow_locs(
    *, device: torch.device, num_steps: int, page_size: int
) -> None:
    """Compile every W_POW2 bucket the armed COW kernel can hit (span_sup =
    tail + K + 1, tail in [0, page)) at engine init: a mid-run Triton
    compile rides the dispatch tail for ~100ms -- long past the commit
    interval, so each first-hit bucket would be a fallback-capture window
    (D25 family). With do_not_specialize on every host int, these launches
    are the kernel's ONLY compiles for the engine's lifetime."""
    width = page_size + 2 * num_steps + 2  # >= any span the loop launches
    dummy = torch.zeros(width, dtype=torch.int64, device=device)
    glue = torch.zeros((max(num_steps, 1), width), dtype=torch.int64, device=device)
    branch = torch.zeros((1, width), dtype=torch.int64, device=device)
    row = torch.zeros(1, dtype=torch.int64, device=device)
    seen: set[int] = set()
    for tail in range(page_size):
        span_sup = tail + num_steps + 1
        w_pow2 = triton.next_power_of_2(span_sup + num_steps)
        if w_pow2 in seen:
            continue
        seen.add(w_pow2)
        out = torch.empty(2, span_sup, dtype=torch.int64, device=device)
        _armed_cow_locs_kernel[(1,)](
            out[0],
            out[1],
            dummy,
            dummy,
            glue,
            branch,
            row,
            row,
            row,
            0,
            0,
            0,
            span_sup,
            dummy.numel(),
            glue.shape[1],
            branch.shape[1],
            W_POW2=w_pow2,
        )


@triton.jit(
    # Per-round host ints (see _armed_cow_locs_kernel): without this the
    # divisibility specializer would add compile classes the old
    # device-vector signature never had.
    do_not_specialize=["base_len", "tail"]
)
def _chain_meta_kernel(
    verdict_ptr,  # [>=4] int64 from commit_scatter: [0]=verdict, [3]=dlen
    case_ptr,  # [ROWS_SEL] int64 (variant-static: accept case per row)
    sel_rows_ptr,  # [ROWS_SEL] int64 (selected row -> carrier pool row)
    flats_ptr,  # [ROWS_POOL, FLATS_W] int64 (branch rows' private flat slots)
    seq_lens_ptr,  # [ROWS_SEL] int64 (chain bucket static)
    positions_ptr,  # [ROWS_SEL] int64 (chain bucket static)
    mrope_ptr,  # [3, ROWS_SEL] int64 (chain bucket static)
    out_locs_ptr,  # [K, ROWS_SEL] int64 (chain bucket static)
    base_len,  # pre-commit base (host int at arm time; no H2D staging)
    tail,  # base_len - page_floor(base_len)
    FLATS_W: tl.constexpr,
    K: tl.constexpr,
    ROWS_SEL: tl.constexpr,
    ROWS_PAD: tl.constexpr,
):
    """Arithmetic completion of the pre-launched chain's metadata: everything
    per-dlen about the chain is a +dlen shift (seq/pos/mrope) or a +dlen
    offset into the rows' private flat-slot table (out_locs), so ONE kernel
    reading the scatter's dlen tap finishes what the arm staged. The former
    arm-static device vectors were pure arithmetic over (base_len, tail,
    case) -- scalars + the variant's cached case vector replace their two
    pinned H2Ds. Junk verdict (0) degrades to dlen=0: reads stay inside the
    pre-commit base, writes stay inside the rows' private scratch slots --
    harmless by layout."""
    lanes = tl.arange(0, ROWS_PAD)
    mask = lanes < ROWS_SEL
    verdict = tl.load(verdict_ptr + 0)
    dlen = tl.load(verdict_ptr + 3)
    dlen = tl.where(verdict == 0, 0, dlen)
    dlen = tl.minimum(tl.maximum(dlen, 0), K + 1)
    case = tl.load(case_ptr + lanes, mask=mask, other=0)
    seq = base_len + case + dlen
    tl.store(seq_lens_ptr + lanes, seq, mask=mask)
    tl.store(positions_ptr + lanes, seq - 1, mask=mask)
    for d in tl.static_range(3):
        tl.store(mrope_ptr + d * ROWS_SEL + lanes, seq - 1, mask=mask)
    pool_row = tl.load(sel_rows_ptr + lanes, mask=mask, other=0)
    for s in tl.static_range(K):
        off = tail + case + s + dlen
        off = tl.minimum(tl.maximum(off, 0), FLATS_W - 1)
        loc = tl.load(flats_ptr + pool_row * FLATS_W + off, mask=mask, other=0)
        tl.store(out_locs_ptr + s * ROWS_SEL + lanes, loc, mask=mask)


def chain_meta_fill(
    *,
    verdict: torch.Tensor,
    case_of_row: torch.Tensor,
    sel_rows: torch.Tensor,
    flats: torch.Tensor,
    seq_lens_out: torch.Tensor,
    positions_out: torch.Tensor,
    mrope_out: torch.Tensor,
    out_locs_out: torch.Tensor,
    base_len: int,
    tail: int,
    num_steps: int,
) -> None:
    """Finish the armed chain's static buffers from the scatter's dlen tap
    (see _chain_meta_kernel). Enqueue AFTER commit_scatter on the same
    stream; every output is a chain-bucket static buffer the queued replay
    reads."""
    rows_sel = case_of_row.numel()
    _chain_meta_kernel[(1,)](
        verdict,
        case_of_row,
        sel_rows,
        flats,
        seq_lens_out,
        positions_out,
        mrope_out,
        out_locs_out,
        base_len,
        tail,
        FLATS_W=flats.shape[1],
        K=num_steps,
        ROWS_SEL=rows_sel,
        ROWS_PAD=triton.next_power_of_2(rows_sel),
    )


@triton.jit
def _shift_replay_fb_kernel(
    positions_ptr,  # [N_POS] per-token rotary positions
    mrope_ptr,  # [3, N_POS] (HAS_MROPE only)
    seq_lens_ptr,  # [N_ROWS]
    orig_seq_lens_ptr,  # [N_ROWS]
    prefix_lens_ptr,  # [N_ROWS]
    d,
    n_pos,
    n_rows,
    HAS_MROPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Add d to every base-derived field of the replay fb in ONE launch (the
    five separate ``.add_(d)`` calls cost a launch each on the round-serial
    restage path). Sizes are tiny (rows x width tokens), one program."""
    lanes = tl.arange(0, BLOCK)
    m_pos = lanes < n_pos
    tl.store(
        positions_ptr + lanes,
        tl.load(positions_ptr + lanes, mask=m_pos, other=0) + d,
        mask=m_pos,
    )
    m_rows = lanes < n_rows
    tl.store(
        seq_lens_ptr + lanes,
        tl.load(seq_lens_ptr + lanes, mask=m_rows, other=0) + d,
        mask=m_rows,
    )
    tl.store(
        orig_seq_lens_ptr + lanes,
        tl.load(orig_seq_lens_ptr + lanes, mask=m_rows, other=0) + d,
        mask=m_rows,
    )
    tl.store(
        prefix_lens_ptr + lanes,
        tl.load(prefix_lens_ptr + lanes, mask=m_rows, other=0) + d,
        mask=m_rows,
    )
    if HAS_MROPE:
        m_mrope = lanes < 3 * n_pos
        tl.store(
            mrope_ptr + lanes,
            tl.load(mrope_ptr + lanes, mask=m_mrope, other=0) + d,
            mask=m_mrope,
        )


def shift_replay_fb(
    *,
    positions: torch.Tensor,
    mrope_positions: Optional[torch.Tensor],
    seq_lens: torch.Tensor,
    orig_seq_lens: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    d: int,
) -> None:
    """One-launch replacement for the replay fb's device-side +d family
    (positions / mrope twin / seq-lens family / extend prefix)."""
    n_pos = positions.numel()
    n_rows = seq_lens.numel()
    has_mrope = mrope_positions is not None
    _shift_replay_fb_kernel[(1,)](
        positions,
        mrope_positions if has_mrope else positions,
        seq_lens,
        orig_seq_lens,
        extend_prefix_lens,
        d,
        n_pos,
        n_rows,
        HAS_MROPE=has_mrope,
        BLOCK=triton.next_power_of_2(max(3 * n_pos if has_mrope else n_pos, n_rows)),
    )
