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
    return seg_ok, skip, dlen, sel_wire, sel_pre, toks


@triton.jit
def _tok_at(toks, lanes, idx):
    return tl.sum(tl.where(lanes == idx, toks, 0))


@triton.jit
def _commit_match_kernel(
    rows_ptr,
    units_ptr,
    backbone_ptr,
    out_ptr,
    seat,
    backbone_len,
    base_out_len,
    K: tl.constexpr,
    F: tl.constexpr,
    W_UNIT: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    SLOTS: tl.constexpr,
    W_PAD: tl.constexpr,
):
    seg_ok, skip, dlen, wire, pre_len, toks = _mirror_snapshot(
        rows_ptr, seat, base_out_len, ROW_WORDS, SLOTS, W_PAD
    )
    lanes = tl.arange(0, W_PAD)
    case = dlen - 1
    prefix_ok = seg_ok & (case >= 0) & (case <= K) & (case <= backbone_len)
    skip_c = tl.minimum(tl.maximum(skip, 0), ROW_WORDS - 5)
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
    # 0 = no consumable segment (nothing landed for this base yet, an old /
    # re-sent / later segment, or a torn snapshot): state untouched, the
    # host round redoes everything.
    verdict = tl.where(~seg_ok, 0, tl.where(hit, 2, 1))
    tl.store(out_ptr + 0, verdict.to(tl.int64))
    tl.store(out_ptr + 1, case_c.to(tl.int64))
    tl.store(out_ptr + 2, f_found.to(tl.int64))
    tl.store(out_ptr + 3, (pre_len + wire).to(tl.int64))


def commit_match(
    *,
    mirror,
    seat: int,
    units_dev: torch.Tensor,  # [K+1, F, K+1] this seat's last block (contiguous)
    backbone_dev: torch.Tensor,  # [K] device backbone twin
    backbone_len: int,
    base_out_len: int,
    num_steps: int,
    fanout: int,
) -> torch.Tensor:
    """The drafter-side GPU replica of ``_match_seat``, fed from the commit
    mirror: out = [verdict(0=no consumable segment, 1=miss, 2=hit), case, f,
    new_total]. Self-consistent by construction -- the consumable segment is
    picked by pure arithmetic between each slot's own pre_len and the
    caller-sampled committed base, exactly the verifier-select stamp
    recipe."""
    out = torch.empty(4, dtype=torch.int64, device=units_dev.device)
    _commit_match_kernel[(1,)](
        mirror.rows,
        units_dev,
        backbone_dev,
        out,
        seat,
        backbone_len,
        base_out_len,
        K=num_steps,
        F=fanout,
        W_UNIT=num_steps + 1,
        ROW_WORDS=mirror.row_words,
        SLOTS=2,
        W_PAD=triton.next_power_of_2(mirror.width),
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
    table_ptr,  # page-table row base (real or shadow); pads land at
    table_col0,  # columns [table_col0 + delta - ? ...]: col0 = base offset
    # scalars
    seat,
    backbone_len,
    base_len,
    base_out_len,
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
):
    seg_ok, skip, dlen, wire_len, pre_len, toks = _mirror_snapshot(
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
    # D: seat-row pad table entries, columns [base+delta, base+WIDTH)
    new_len = base_len + dlen
    pad_offset = new_len - (new_len // PAGE) * PAGE
    for i in tl.static_range(WIDTH):
        col = dlen + i  # relative to base
        in_pad = (col >= dlen) & (col < WIDTH)
        pval = tl.load(pad_flats_ptr + tl.minimum(pad_offset + i, PAGE + WIDTH - 2))
        tl.store(
            table_ptr + table_col0 + col,
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
    table_col0: int,
    base_len: int,
    base_out_len: int,
    num_steps: int,
    fanout: int,
    page_size: int,
    extend_width: int,
    outs: tuple,  # (verdict[4]i64, input[ROWS_W]i64, out_loc[ROWS_W]i64, true[ROWS]i32, node[ROWS]i64, chains[K]i64)
) -> None:
    """One-launch commit consumption: the on-GPU match plus every per-case
    value the fused-extend replay needs (input assembly, out_cache_loc /
    GDN true-lens selection, seat-pad page-table suffix), fed entirely from
    the commit mirror. Stale generation writes only the verdict -- the
    caller's junk-lane / fallback handles the rest."""
    verdict, input_out, out_loc_out, true_out, node_out, chains_out = outs
    rows = true_stack.shape[1]
    rows_w = gather_stack.shape[1]
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
        table_col0,
        seat,
        backbone_len,
        base_len,
        base_out_len,
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
    )
