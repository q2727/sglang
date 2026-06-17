# MiniMax-M3 Sparse Attention Contract for HiSparse

This document is the Agent A contract for MiniMax-M3 decode-only HiSparse work.
It documents the existing MiniMax sparse attention data flow and the tensor
interfaces that pool, swap-in, and backend agents must preserve.

MiniMax-M3 is not DSA. The sparse layers have a MiniMax-specific two-branch
attention path:

- Layers `0-2` use dense attention and require full main K/V on GPU.
- Layers `3-59` use MiniMax sparse attention.
- The index branch computes selected logical sparse blocks from full index K.
- The main branch attends over the selected blocks.
- MSA is preserved. It may replace only the main sparse attention step.
- First-phase HiSparse keeps sparse-layer index K fully GPU resident.
- First-phase HiSparse moves only sparse-layer main K/V through host-backed
  storage plus a hot GPU buffer.

## Names and IDs

Use these terms consistently:

- Logical token position: zero-based position inside one request sequence.
- Logical block id: `floor(logical_token_position / sparse_block_size)`.
- Physical cache slot: row in the normal SGLang K/V cache pools.
- Hot buffer slot: row in the HiSparse hot GPU main K/V buffer.
- Request row id: row in `req_to_token`; in sparse op signatures this is named
  `slot_ids`, but it is `forward_batch.req_pool_indices`, not a cache slot.

For MiniMax-M3, `sparse_block_size = 128`, `sparse_topk_blocks = 16`,
`sparse_num_index_heads = 4`, `sparse_index_dim = 128`,
`num_attention_heads = 64`, `num_key_value_heads = 4`, and `head_dim = 128`
before tensor-parallel sharding.

## Decode Sequence Diagram

```mermaid
sequenceDiagram
    participant Layer as MiniMaxM3Attention
    participant Backend as MiniMaxSparseAttnBackend.forward_decode
    participant Pool as MiniMaxSparseKVPool
    participant Indexer as flash_decode_with_topk_idx
    participant Reduce as topk_index_reduce
    participant HiSparse as HiSparse swap-in (future)
    participant Main as MSA or Triton main sparse attention

    Layer->>Layer: qkv_proj or fused_qkv_index_proj(hidden_states)
    Layer->>Layer: main q/k norm+rope; v remains value head
    Layer->>Layer: index_qkv_proj or fused split
    Layer->>Layer: idx_q/idx_k norm+rope; idx_v absent for M3 K-only sparse layers
    Layer->>Backend: self.attn(q, k, v, idx_q, idx_k, idx_v)
    Backend->>Pool: set_fused_kv_index_buffer(out_cache_loc, k, v, idx_k, idx_v=None)
    Pool-->>Backend: k_cache, v_cache, idx_k_cache
    Backend->>Indexer: minimax_sparse_decode -> flash_decode_with_topk_idx(idx_q, idx_k_cache, req_to_token, req_pool_indices, seq_lens)
    Indexer-->>Backend: topk_idx [index heads, batch, topk], invalid=-1
    Backend->>Reduce: reduce index heads to KV heads when required
    Reduce-->>Backend: topk_idx [kv heads, batch, topk]
    Backend->>HiSparse: optional: union selected logical blocks and load sparse main K/V to hot GPU pages
    HiSparse-->>Backend: hot K/V plus hot page table or hot req_to_token equivalent
    Backend->>Main: consume topk_idx in main sparse attention
    Main-->>Backend: main attention output o
    Backend-->>Layer: idx_o, o
    Layer->>Layer: o_proj(o); add index_o_proj(idx_o) only if index V is enabled
```

## Decode Data Flow

Main `q/k/v` are produced in `MiniMaxM3Attention.forward_prepare` from
`qkv_proj` or the fused `fused_qkv_index_proj`. Main `q` and `k` are normalized
and RoPE-applied. Main `v` is split from the value projection and is not RoPE
transformed. In `forward_core`, the tensors are reshaped to
`[tokens, heads, head_dim]` and passed to `RadixAttention`.

Index `idx_q/idx_k` are produced only for sparse layers. They come from
`index_qkv_proj` or the index slice of `fused_qkv_index_proj`. `idx_q` and
`idx_k` are normalized and RoPE-applied through the index norm/rope path.
For MiniMax-M3 sparse layers, `sparse_disable_index_value = 1`, so `idx_v` is
`None` and the index branch is K-only.

During decode, `MiniMaxSparseAttnBackend.forward_decode` stores the current
token's main K/V and index K through:

```text
kv_pool.set_fused_kv_index_buffer(
    layer,
    forward_batch.out_cache_loc,
    k,
    v,
    idx_k,
    None,
)
```

Then the backend reads:

```text
k_cache, v_cache = kv_pool.get_kv_buffer(layer_id)
idx_k_cache = kv_pool.get_index_k_buffer(layer_id)
```

The indexer call is inside `minimax_sparse_decode`:

```text
flash_decode_with_topk_idx(
    q=idx_q,
    k_cache=idx_k_cache,
    v_cache=None,
    req_to_token=req_to_token,
    seq_lens=forward_batch.seq_lens,
    slot_ids=forward_batch.req_pool_indices,
    block_size=128,
    topk=16,
    init_blocks=0,
    local_blocks=1,
    disable_index_value=True,
)
```

The returned `topk_idx` is consumed by the main sparse attention step. If MSA is
active, `msa_sparse_decode_main` consumes it as `kv_block_indexes`. Otherwise
`flash_decode_with_gqa_share_sparse` consumes it directly in the Triton fallback.
MSA and Triton replace only the main sparse attention step; they do not replace
the indexer.

## Tensor Contract

Symbols:

- `B`: decode batch size.
- `Hq`: per-rank query heads, `layer.num_heads`.
- `Hkv`: per-rank KV heads, `layer.num_kv_heads`.
- `Hidx`: per-rank index query heads, `layer.num_idx_heads`.
- `D`: main `head_dim`, 128 for MiniMax-M3.
- `Di`: index head dim, 128 for MiniMax-M3.
- `S`: maximum allocated token slots in the cache pool.
- `Lmax`: maximum context length.
- `P`: page size and sparse block size for MSA, 128.
- `K`: sparse top-k blocks, 16 for MiniMax-M3.

| Tensor | Decode shape | Dtype | Device | Semantics |
| --- | --- | --- | --- | --- |
| `q` | `[B, Hq, D]` | `torch.bfloat16` or `torch.float16` | CUDA/HIP device | Current decode query for the main branch after Q norm and RoPE. |
| `k` | `[B, Hkv, D]` | same compute dtype as `q` | CUDA/HIP device | Current token main K after K norm and RoPE, before/while storing into the main K cache. |
| `v` | `[B, Hkv, D]` | same compute dtype as `q` | CUDA/HIP device | Current token main V, stored with `k` into the main V cache. |
| `idx_q` | `[B, Hidx, Di]` | same compute dtype as `q` | CUDA/HIP device | Current decode query for the sparse index branch after index Q norm and RoPE. |
| `idx_k` | `[B, 1, Di]` | same compute dtype as `idx_q` | CUDA/HIP device | Current token index K after index K norm and RoPE. This is stored for all sparse layers and remains full GPU resident in first-phase HiSparse. |
| `idx_v` | `None` for MiniMax-M3 sparse layers | N/A | N/A | Index value is disabled by `sparse_disable_index_value=1`. Other MiniMax sparse variants may have `[B, 1, Di]`. |
| `k_cache` | `[S, Hkv, D]` | main KV cache dtype; BF16/FP16 for MSA, Triton may also handle HIP fp8 cache paths | CUDA/HIP device | Main K cache addressed by physical cache slot. Under HiSparse this pointer may instead be the hot main K buffer if the mapping resolves to hot slots. |
| `v_cache` | `[S, Hkv, D]` | same as `k_cache` | CUDA/HIP device | Main V cache addressed by physical cache slot. Under HiSparse this pointer may instead be the hot main V buffer. |
| `idx_k_cache` | `[S, 1, Di]` | index cache dtype; target MiniMax-M3 BF16 | CUDA/HIP device | Full sparse-layer index K cache. The indexer scans this full GPU-resident cache to select blocks. It is not offloaded in the first design. |
| `req_to_token` | `[max_reqs + 1, Lmax]` | `torch.int32` | CUDA/HIP device | Baseline mapping from request row and logical token position to physical cache slot. Row 0 is padding/dummy for CUDA graph padded batches. |
| `slot_ids` | `[B]` | integer tensor, usually `torch.int64` from `ForwardBatch.req_pool_indices` | CUDA/HIP device | Sparse-op name for request row ids into `req_to_token`. This is not a physical cache slot. |
| `seq_lens` | `[B]` | `torch.int64` in `ForwardBatch`, converted to `torch.int32` by some kernels/plans | CUDA/HIP device | Cached sequence length per request, including the current decode token after cache allocation. |
| `out_cache_loc` | `[B]` for decode | integer tensor | CUDA/HIP device | Physical cache slot for the current token write. This is used before index selection to store current main K/V and index K. |
| `topk_idx` | `[Hkv, B, K]` for MiniMax-M3 after index-head reduction | `torch.int32` | CUDA/HIP device | Selected logical block ids for each KV head and request. Entries are 0-based logical block ids, not token ids or cache slots. Invalid padding is `-1`. |
| MSA `kv_indices` | eager helper: `[sum_i ceil(seq_lens[i] / P)]`; persistent decode buffer: `[B * max_pages]` with valid prefix | `torch.int32` | CUDA/HIP device | Flattened physical page ids for each request's logical pages, in request order. Current helper derives each page from `req_to_token[row, logical_page * P] // P`. Under HiSparse this must point to hot physical pages for sparse main K/V. |
| MSA `kv_block_indexes` | `[B, Hkv, K]` in decode | `torch.int32` | CUDA/HIP device | MSA's selected block index tensor, built as `topk_idx.permute(1, 0, 2).contiguous().to(torch.int32)`. Values remain logical block ids; `-1` is invalid. |

## `topk_idx` Semantics

`topk_idx` is a logical block id. It is not a token id, physical cache slot, or
hot buffer slot.

For block id `b`, the corresponding logical token positions are:

```text
start = b * sparse_block_size
end = min((b + 1) * sparse_block_size, seq_lens[request])
```

The main attention implementation then resolves logical token positions to
physical data. In the existing Triton path this is done through
`req_to_token[request_row, logical_position]`. In the existing MSA path,
`topk_idx` becomes `kv_block_indexes` and indexes the per-request MSA page table.

`flash_decode_with_topk_idx` first produces per-index-head block ids with shape
`[num_idx_heads, B, K]`. `minimax_sparse_decode` reduces index heads to KV heads
when `num_idx_heads > num_kv_heads` by taking the union and padding with `-1`.
For the MiniMax-M3 target config, per-rank index heads and KV heads match, so the
post-reduction decode layout is `[Hkv, B, 16]`.

Invalid entries are `-1`. Valid entries are left-packed by the top-k kernels.
Consumers count or mask `topk_idx >= 0`.

Local/init behavior is part of the indexer contract:

- `init_blocks` forces the first logical blocks into selection by assigning
  very large block scores. MiniMax-M3 has `sparse_init_block = 0`, so no init
  blocks are forced in the target config.
- `local_blocks` forces the last logical blocks into selection. MiniMax-M3 has
  `sparse_local_block = 1`, so the current tail block must remain selectable.
- If `ceil(seq_len / sparse_block_size) <= topk`, all valid logical blocks are
  selected and the remaining entries are `-1`.

Do not sort, reinterpret, or compress block ids into token ids or cache slots.
Any HiSparse union/swap logic must treat them as logical block ids until it
explicitly builds a separate hot mapping.

## MSA Path Contract

MSA must be preserved. Do not set `SGLANG_DISABLE_MSA=1` as a compatibility fix.
MSA replaces only the main sparse attention step after `topk_idx` has been
computed by the MiniMax index branch.

The existing backend enables MSA only when all kernel constraints hold:

- `msa_available()` is true on supported NVIDIA Blackwell SM100-family devices.
- `sparse_block_size == 128`.
- `kv_pool.page_size == sparse_block_size`.
- `sparse_topk_blocks` is one of `4, 8, 16, 32`.
- Main K/V cache dtype is BF16/FP16, not fp8.
- For decode, the current code avoids MSA when decode is captured by CUDA graph;
  it falls back to the CUDA-graph-safe Triton sparse path instead. This is not an
  MSA disable switch and must not change index semantics.

MSA expects paged main K/V:

```text
k_cache/v_cache: [max_slots, Hkv, D]
max_slots % page_size == 0
view -> [num_phys_pages, page_size, Hkv, D]
permute -> [num_phys_pages, Hkv, page_size, D]
```

The current `_build_page_table` builds MSA `kv_indices` from the full cache:

1. Compute `n_pages[i] = ceil(seq_lens[i] / page_size)`.
2. Pack pages request by request.
3. For packed page `(request i, logical page p)`, read the first token slot:
   `physical_slot = req_to_token[slot_ids[i], p * page_size]`.
4. Store `physical_page = physical_slot // page_size` as `int32`.

`kv_block_indexes` is built per sparse layer from that layer's `topk_idx`.
Because top-k selection is layer-dependent, `kv_block_indexes` cannot be shared
across sparse layers.

The following metadata can be prepared outside CUDA graph capture when it does
not depend on per-layer `topk_idx`:

- MSA plan for the current batch size, head counts, page size, and top-k count.
- Length-dependent plan tensors derived from `seq_lens`.
- Baseline or hot `kv_indices` page-table buffers if their contents are known.
- Static hot-buffer workspace and fixed-shape output buffers.

Metadata that depends on `topk_idx` is produced after the indexer in each sparse
layer. If a HiSparse implementation needs CPU-side selected-block union,
H2D copy submission, or hot page-table construction from `topk_idx`, decode
CUDA graph must be disabled for that path or the work must be implemented as
fixed-shape graph-safe device operations.

When sparse main K/V comes from a hot HiSparse buffer:

- The hot buffer must expose the same paged K/V layout to MSA:
  `[num_hot_pages, Hkv, 128, D]`.
- `topk_idx` and MSA `kv_block_indexes` remain logical block ids.
- `kv_indices` must resolve those logical block ids to hot physical page ids.
  The existing `_build_page_table` cannot be used unmodified if it still points
  to full main-cache pages whose data has been offloaded.
- It is valid to build a layer/path-specific hot page table or an equivalent
  mapping, but the MSA kernel must load exactly the same K/V values that the
  baseline full-cache path would load for each selected logical block.
- Index K is not hot-swapped in the first design; `idx_k_cache` stays full GPU
  resident and uses the baseline `req_to_token` mapping for index selection.

## Triton Fallback Path Contract

The Triton fallback is `flash_decode_with_gqa_share_sparse`. It consumes:

```text
q:        [B, Hq, D]
k_cache: [S, Hkv, D]
v_cache: [S, Hkv, D]
topk_idx:[Hkv, B, K]
req_to_token, slot_ids, seq_lens
```

For each selected logical block id, the kernel computes logical positions
`block_id * block_size + offset` and resolves each token through:

```text
slot = req_to_token[slot_ids[batch], logical_position]
k = k_cache[slot, kv_head, :]
v = v_cache[slot, kv_head, :]
```

The kernel has no concept of host storage or hot buffers. A HiSparse fallback
integration must therefore provide one of these equivalent contracts:

- Pass hot `k_cache/v_cache` pointers and a layer/path-specific
  `req_to_token`-equivalent mapping whose selected logical positions resolve to
  hot buffer slots.
- Or modify/wrap the Triton path with an equivalent hot logical-position to hot
  slot mapping.

Do not globally rewrite the baseline `req_to_token` in a way that breaks dense
layers, the indexer, or other sparse layers. If the fallback uses rewritten
metadata, it must be scoped to the sparse main attention step after index
selection.

The Triton fallback output must be mathematically equivalent to the baseline
full-cache MiniMax sparse main attention:

```text
for every selected logical block b and token offset t:
    hot_kv[hot_slot(b, t)] == full_main_kv[req_to_token[row, b * 128 + t]]
```

Invalid `topk_idx == -1` entries must remain invalid. The fallback expects valid
entries to be left-packed, matching the existing top-k kernels.

## Dense Main Decode Optimization Caveat

`flash_decode_with_topk_idx` has an optional `use_dense_main_attn` mode that
returns a dense backend page table in the variable named `topk_idx`, plus
`real_seq_lens`. In that special path, the returned tensor is not the logical
block-id contract described above.

HiSparse agents should not use that special page-table mode as the MiniMax-M3
top-k block contract. If an implementation supports it, document it as a
separate dense-main optimization and preserve equivalent block selection.

## Invariants Other Agents Must Not Violate

- MSA is preserved. Do not disable MSA to make HiSparse work.
- The MiniMax index branch is preserved. MSA or Triton may replace only the main
  sparse attention step.
- Sparse layer index K remains full GPU resident in the first design.
- Dense layer main K/V remains full GPU resident.
- Only sparse layer main K/V is eligible for first-phase HiSparse offload.
- Current token main K/V and index K must be stored before the indexer runs, so
  the current tail block can participate in `local_blocks=1` selection.
- `topk_idx` contains logical block ids, not token ids, physical cache slots, or
  hot buffer slots.
- `req_to_token` maps logical token positions to physical cache slots in the
  baseline cache. Any hot mapping must be explicit and scoped to main sparse
  attention.
- `slot_ids` in MiniMax sparse ops means request row ids
  (`forward_batch.req_pool_indices`), not cache slots.
- MSA page size must stay equal to MiniMax sparse block size, 128.
- The hot K/V buffer must expose the same per-token K/V values and head layout
  that the baseline main K/V cache would expose for selected logical blocks.
- Invalid selected blocks are represented by `-1` and must not be remapped to a
  real block or slot.
- Do not assume MiniMax-M3 is DSA or reuse DSA HiSparse contracts without this
  MiniMax-specific index/main branch separation.
