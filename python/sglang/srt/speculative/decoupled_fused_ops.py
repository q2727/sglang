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
