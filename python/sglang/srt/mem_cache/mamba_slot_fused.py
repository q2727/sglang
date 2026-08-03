"""Fused Triton kernels that clear / copy state-pool slots across all state
tensors of a hybrid (mamba-style) pool in a single launch.

``MambaPool.clear_slots`` / ``copy_from`` otherwise loop over every conv-state
tensor (models with several short-conv streams have a handful, each a distinct
scattered-index kernel), and the speculative-decode draft worker replays that
loop across every draft head's pool — so one fresh-request / radix-COW event
fans out into many tiny launch-bound kernels on the forward stream. These
kernels fold the whole tensor list into one launch.

The temporal (SSM) state is covered by the same descriptor mechanism: the
PyTorch reference ``temporal[:, dst] = temporal[:, src]`` is a gather that
materializes an intermediate plus a scatter — two kernels and twice the
bandwidth — while the fused kernel copies src->dst rows in one pass.

The tensors are heterogeneous only in their trailing feature size and share
the leading ``[num_layers, pool_size]`` dims + dtype, so they are addressed via a
per-tensor pointer / stride / feature-length array. The kernel reads each
tensor's real ``layer_stride`` / ``slot_stride``, so it is layout-general; it
only requires the per-slot feature block to be contiguous.
"""

from __future__ import annotations

from typing import List, NamedTuple

import torch
import triton
import triton.language as tl

_BLOCK = 1024

# constexpr dtype selector: a Triton pointer cast target must be compile-time,
# so each dtype compiles its own specialization of the kernels below.
_DTYPE_IDS = {torch.bfloat16: 0, torch.float32: 1, torch.float16: 2}


class SlotDescriptor(NamedTuple):
    ptr: torch.Tensor  # [T] int64 base byte-addresses
    feat: torch.Tensor  # [T] int64 per-slot feature length (elements)
    layer_stride: torch.Tensor  # [T] int64 element stride between layers
    slot_stride: torch.Tensor  # [T] int64 element stride between slots
    num_layers: int
    max_feat_blocks: int
    dtype_id: int


# Back-compat alias (pre-temporal name).
ConvSlotDescriptor = SlotDescriptor


# The feature dimension is parallelized on the grid (axis 2 encodes
# layer x feature-block): the temporal (SSM) state has ~5e5 elements per
# (slot, layer) row, and a serial in-program loop over that many blocks both
# under-parallelizes by orders of magnitude and blows up compile time when
# statically unrolled.


@triton.jit
def _fused_slot_clear_kernel(
    ptr_arr,
    feat_arr,
    layer_stride_arr,
    slot_stride_arr,
    index_arr,
    DTYPE_ID: tl.constexpr,
    NUM_FEAT_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    iid = tl.program_id(0)
    tid = tl.program_id(1)
    lid = tl.program_id(2) // NUM_FEAT_BLOCKS
    fb = tl.program_id(2) % NUM_FEAT_BLOCKS
    base_addr = tl.load(ptr_arr + tid)
    cols = fb * BLOCK + tl.arange(0, BLOCK)
    mask = cols < tl.load(feat_arr + tid)
    off = (
        lid * tl.load(layer_stride_arr + tid)
        + tl.load(index_arr + iid) * tl.load(slot_stride_arr + tid)
        + cols
    )
    if DTYPE_ID == 0:
        base = base_addr.to(tl.pointer_type(tl.bfloat16))
        tl.store(base + off, tl.zeros((BLOCK,), dtype=tl.bfloat16), mask=mask)
    elif DTYPE_ID == 1:
        base = base_addr.to(tl.pointer_type(tl.float32))
        tl.store(base + off, tl.zeros((BLOCK,), dtype=tl.float32), mask=mask)
    else:
        base = base_addr.to(tl.pointer_type(tl.float16))
        tl.store(base + off, tl.zeros((BLOCK,), dtype=tl.float16), mask=mask)


@triton.jit
def _fused_slot_copy_kernel(
    ptr_arr,
    feat_arr,
    layer_stride_arr,
    slot_stride_arr,
    src_arr,
    dst_arr,
    DTYPE_ID: tl.constexpr,
    NUM_FEAT_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    iid = tl.program_id(0)
    tid = tl.program_id(1)
    lid = tl.program_id(2) // NUM_FEAT_BLOCKS
    fb = tl.program_id(2) % NUM_FEAT_BLOCKS
    base_addr = tl.load(ptr_arr + tid)
    cols = fb * BLOCK + tl.arange(0, BLOCK)
    mask = cols < tl.load(feat_arr + tid)
    layer_off = lid * tl.load(layer_stride_arr + tid)
    slot_stride = tl.load(slot_stride_arr + tid)
    src_off = layer_off + tl.load(src_arr + iid) * slot_stride + cols
    dst_off = layer_off + tl.load(dst_arr + iid) * slot_stride + cols
    if DTYPE_ID == 0:
        base = base_addr.to(tl.pointer_type(tl.bfloat16))
        tl.store(base + dst_off, tl.load(base + src_off, mask=mask), mask=mask)
    elif DTYPE_ID == 1:
        base = base_addr.to(tl.pointer_type(tl.float32))
        tl.store(base + dst_off, tl.load(base + src_off, mask=mask), mask=mask)
    else:
        base = base_addr.to(tl.pointer_type(tl.float16))
        tl.store(base + dst_off, tl.load(base + src_off, mask=mask), mask=mask)


def slot_fuse_supported(t: torch.Tensor) -> bool:
    """Whether one state tensor qualifies for the fused slot ops: CUDA, a
    supported dtype, and a contiguous per-slot feature block."""
    return (
        t.is_cuda
        and t.dtype in _DTYPE_IDS
        and t.numel() > 0
        and t[0, 0].is_contiguous()
    )


def build_slot_descriptor(tensors: List[torch.Tensor]) -> SlotDescriptor:
    """Build the pool-stable addressing descriptor for a state-tensor list.

    Requires same-dtype tensors sharing the leading (num_layers, pool_size) dims
    with a contiguous per-slot feature block (the kernel reads each tensor's real
    strides, so the block may sit inside a larger strided envelope). Cache the
    result and reuse it — pool tensors don't move after allocation.
    """
    t0 = tensors[0]
    num_layers = t0.shape[0]
    device = t0.device
    dtype_id = _DTYPE_IDS[t0.dtype]
    ptr, feat, layer_stride, slot_stride = [], [], [], []
    max_feat = 0
    for t in tensors:
        assert t.dtype == t0.dtype, "fused slot ops need a uniform dtype per descriptor"
        assert t.shape[0] == num_layers, "state tensors must share num_layers"
        assert t.device == device
        assert t[0, 0].is_contiguous(), "per-slot feature block must be contiguous"
        ptr.append(t.data_ptr())
        feat.append(t[0, 0].numel())
        layer_stride.append(t.stride(0))
        slot_stride.append(t.stride(1))
        max_feat = max(max_feat, t[0, 0].numel())
    to_i64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device=device)
    return SlotDescriptor(
        ptr=to_i64(ptr),
        feat=to_i64(feat),
        layer_stride=to_i64(layer_stride),
        slot_stride=to_i64(slot_stride),
        num_layers=num_layers,
        max_feat_blocks=triton.cdiv(max_feat, _BLOCK),
        dtype_id=dtype_id,
    )


# Back-compat alias (pre-temporal name).
build_conv_slot_descriptor = build_slot_descriptor


def fused_clear_slots(desc: SlotDescriptor, indices: torch.Tensor):
    """Zero ``indices`` slots (dim 1) across every tensor in one launch."""
    if desc.ptr.numel() == 0 or indices.numel() == 0:
        return
    index_arr = indices.to(torch.int64)
    # Slot count on the unbounded grid axis (gridDim.y/z cap at 65535).
    grid = (index_arr.numel(), desc.ptr.numel(), desc.num_layers * desc.max_feat_blocks)
    _fused_slot_clear_kernel[grid](
        desc.ptr,
        desc.feat,
        desc.layer_stride,
        desc.slot_stride,
        index_arr,
        DTYPE_ID=desc.dtype_id,
        NUM_FEAT_BLOCKS=desc.max_feat_blocks,
        BLOCK=_BLOCK,
    )


def fused_copy_slots(
    desc: SlotDescriptor, src_indices: torch.Tensor, dst_indices: torch.Tensor
):
    """Copy state from ``src`` slots to ``dst`` slots across every tensor in
    one launch.

    ``src`` and ``dst`` must be disjoint (the COW invariant: radix-checkpoint
    slots copied into freshly-allocated slots). Unlike the gather-then-scatter
    reference, this kernel reads and writes in one pass, so overlapping ranges
    would race.
    """
    if desc.ptr.numel() == 0 or src_indices.numel() == 0:
        return
    src_arr = src_indices.to(torch.int64)
    dst_arr = dst_indices.to(torch.int64)
    grid = (src_arr.numel(), desc.ptr.numel(), desc.num_layers * desc.max_feat_blocks)
    _fused_slot_copy_kernel[grid](
        desc.ptr,
        desc.feat,
        desc.layer_stride,
        desc.slot_stride,
        src_arr,
        dst_arr,
        DTYPE_ID=desc.dtype_id,
        NUM_FEAT_BLOCKS=desc.max_feat_blocks,
        BLOCK=_BLOCK,
    )


# Back-compat aliases (pre-temporal names).
fused_clear_conv_slots = fused_clear_slots
fused_copy_conv_slots = fused_copy_slots
