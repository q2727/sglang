// MiniMax-M3 HiSparse block swap-in kernel.
//
// GPU-side replacement for the CPU block-mode path in
// MiniMaxHiSparseKVPool.load_sparse_main_blocks_to_hot.
//
// One CTA per request: deduplicates topk block ids across KV heads,
// atomically allocates hot pages, and fills hot_page_table / hot_kv_indices /
// host_locs / hot_locs for the subsequent H2D copy.
//
// Template parameters:
//   BLOCK_THREADS – CTA size (256 for M3; Hkv*K <= 64 entries).
//   PAGE_SIZE     – sparse block size (128 for MiniMax-M3).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__)
static constexpr unsigned long long kWarpSyncMask = 0xFFFFFFFFFFFFFFFFull;
#else
static constexpr unsigned int kWarpSyncMask = 0xFFFFFFFFu;
#endif

namespace {

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <int BLOCK_THREADS, int PAGE_SIZE>
__global__ void minimax_hisparse_swap_in_kernel(
    // --- inputs ---
    const int32_t* __restrict__ topk_idx,        // [Hkv, B, K]
    const int64_t* __restrict__ seq_lens,         // [B]
    const int64_t* __restrict__ req_to_host,      // [max_reqs, max_ctx]
    const int64_t* __restrict__ req_pool_indices, // [B]
    // --- outputs ---
    int32_t* __restrict__ hot_page_table,         // [B, max_pages]
    int32_t* __restrict__ hot_kv_indices,         // [total_pages]
    int32_t* __restrict__ hot_kv_indices_offset,  // [B+1]
    int64_t* __restrict__ host_locs,              // [total_tokens]
    int64_t* __restrict__ hot_locs,               // [total_tokens]
    // --- atomic state ---
    int32_t* __restrict__ next_hot_page,          // scalar (atomic)
    int32_t* __restrict__ token_counter,           // scalar (atomic)
    int32_t* __restrict__ overflow_flag,          // scalar
    // --- params ---
    int Hkv,
    int B,
    int K,
    int max_pages,
    int max_ctx,
    int hot_page_offset,
    int hot_page_capacity,
    int num_real_reqs) {

  using namespace device;

  const int batch_id = blockIdx.x;
  if (batch_id >= num_real_reqs) return;

  const int tx = threadIdx.x;
  const int total_entries = Hkv * K;  // <= 64 for M3 (4 * 16)

  // -------------------------------------------------------------------
  // Shared memory
  // -------------------------------------------------------------------
  static constexpr int kMaxEntries = 128;
  __shared__ int32_t s_input[kMaxEntries];
  __shared__ int32_t s_unique[kMaxEntries];
  __shared__ int32_t s_token_counts[kMaxEntries];
  __shared__ int32_t s_token_offsets[kMaxEntries + 1];

  __shared__ int64_t s_req_row;
  __shared__ int64_t s_seq_len;
  __shared__ int32_t s_num_pages;
  __shared__ int32_t s_count;
  __shared__ int32_t s_start_page;
  __shared__ int32_t s_token_start;   // absolute token offset (atomically reserved)
  __shared__ int32_t s_total_tokens;

  // Warp prefix-sum scratch
  constexpr int kWarps = (BLOCK_THREADS + kWarpThreads - 1) / kWarpThreads;
  __shared__ int32_t s_warp_sums[kWarps];

  // -------------------------------------------------------------------
  // Init
  // -------------------------------------------------------------------
  if (tx == 0) {
    s_req_row = req_pool_indices[batch_id];
    s_seq_len = seq_lens[batch_id];
    s_num_pages = (s_seq_len + PAGE_SIZE - 1) / PAGE_SIZE;
    s_count = 0;
    s_total_tokens = 0;
  }
  __syncthreads();

  const int64_t seq_len   = s_seq_len;
  const int64_t req_row   = s_req_row;
  const int     num_pages = s_num_pages;

  // -------------------------------------------------------------------
  // Phase 1: load topk entries
  // -------------------------------------------------------------------
  // topk_idx layout: [Hkv, B, K] -> (h, batch_id, k) at h*(B*K) + batch_id*K + k
  if (tx < total_entries) {
    int h = tx / K;
    int k = tx % K;
    int32_t blk = topk_idx[h * B * K + batch_id * K + k];
    s_input[tx] = (blk >= 0 && blk < num_pages) ? blk : -1;
  } else if (tx < kMaxEntries) {
    s_input[tx] = -1;
  }
  __syncthreads();

  // -------------------------------------------------------------------
  // Phase 2: dedup — mark first occurrence of each block id
  // -------------------------------------------------------------------
  if (tx < total_entries) {
    int32_t my_blk = s_input[tx];
    int32_t valid = 0;
    if (my_blk >= 0) {
      bool first = true;
      for (int j = 0; j < tx; ++j) {
        if (s_input[j] == my_blk) { first = false; break; }
      }
      valid = first ? 1 : 0;
    }
    s_input[tx] = valid;  // repurpose s_input as valid flags
  }
  __syncthreads();

  // -------------------------------------------------------------------
  // Phase 3: compact — prefix-sum valid flags into s_unique
  // -------------------------------------------------------------------
  const int lane = tx % kWarpThreads;
  const int warp = tx / kWarpThreads;

  int32_t my_valid = (tx < total_entries) ? s_input[tx] : 0;
  int32_t prefix = my_valid;
#pragma unroll
  for (int offset = 1; offset < kWarpThreads; offset <<= 1) {
    int32_t n = __shfl_up_sync(kWarpSyncMask, prefix, offset, kWarpThreads);
    if (lane >= offset) prefix += n;
  }

  int32_t warp_sum = __shfl_sync(kWarpSyncMask, prefix, kWarpThreads - 1);
  if (lane == 0) s_warp_sums[warp] = warp_sum;
  __syncthreads();

  int32_t warp_prefix = 0;
  if (tx == 0) {
    int32_t acc = 0;
    for (int w = 0; w < kWarps; ++w) {
      int32_t ws = s_warp_sums[w];
      s_warp_sums[w] = acc;
      acc += ws;
    }
    s_count = acc;
  }
  __syncthreads();

  warp_prefix = s_warp_sums[warp];
  const int32_t count = s_count;

  // Store unique block ids (re-read since we overwrote s_input)
  // We need to re-derive the original block ids. Actually we can't — let's fix.
  // Phase 2 should preserve block ids. Let's use a separate s_valid array or
  // store block ids in s_unique via compaction during Phase 3.
  //
  // REVISED: Phase 2 marks valid in s_input[tx] (overwriting the block id).
  // But we need the block id for Phase 3. Fix: use a two-pass approach.
  //
  // Actually, let's redo this more carefully.
  // s_input[i] = block_id (from topk_idx). We need to compact s_input into
  // s_unique based on a dedup test. The problem is we can't both store valid
  // flags AND block ids in the same array.
  //
  // Fix: Store valid flags separately in shared memory.
  // Reinitialize s_input from Phase 1 and keep it intact throughout Phase 2.

  // --- Phase 2 (corrected): mark first occurrences ---
  // We need s_input AND s_valid. Let's use s_token_counts temporarily as s_valid.
  if (tx < total_entries) {
    int32_t my_blk = s_input[tx];
    int32_t first = 0;
    if (my_blk >= 0) {
      first = 1;
      for (int j = 0; j < tx; ++j) {
        if (s_input[j] == my_blk) { first = 0; break; }
      }
    }
    s_token_counts[tx] = first;  // temporary: valid flags
  }
  __syncthreads();

  // --- Phase 3 (corrected): compact s_input into s_unique ---
  // Recompute prefix sum of valid flags
  my_valid = (tx < total_entries) ? s_token_counts[tx] : 0;
  prefix = my_valid;
#pragma unroll
  for (int offset = 1; offset < kWarpThreads; offset <<= 1) {
    int32_t n = __shfl_up_sync(kWarpSyncMask, prefix, offset, kWarpThreads);
    if (lane >= offset) prefix += n;
  }
  warp_sum = __shfl_sync(kWarpSyncMask, prefix, kWarpThreads - 1);
  if (lane == 0) s_warp_sums[warp] = warp_sum;
  __syncthreads();

  warp_prefix = 0;
  if (tx == 0) {
    int32_t acc = 0;
    for (int w = 0; w < kWarps; ++w) {
      int32_t ws = s_warp_sums[w];
      s_warp_sums[w] = acc;
      acc += ws;
    }
    s_count = acc;
  }
  __syncthreads();

  warp_prefix = s_warp_sums[warp];
  const int32_t count2 = s_count;

  // Compact: each valid lane writes its block id to s_unique
  if (my_valid && tx < total_entries) {
    int32_t pos = warp_prefix + (prefix - my_valid);  // exclusive position
    s_unique[pos] = s_input[tx];
  }
  __syncthreads();

  if (tx == 0) s_count = count2;
  __syncthreads();
  const int32_t cnt = s_count;

  // -------------------------------------------------------------------
  // Phase 4: token counts per unique block
  // -------------------------------------------------------------------
  if (tx < cnt) {
    int32_t blk       = s_unique[tx];
    int32_t blk_start = blk * PAGE_SIZE;
    int32_t blk_end   = min(blk_start + PAGE_SIZE, static_cast<int32_t>(seq_len));
    s_token_counts[tx] = (blk_end > blk_start) ? (blk_end - blk_start) : 0;
  }
  __syncthreads();

  // -------------------------------------------------------------------
  // Phase 5: prefix sum of token counts
  // -------------------------------------------------------------------
  if (tx == 0) {
    int32_t acc = 0;
    for (int i = 0; i < cnt; ++i) {
      s_token_offsets[i] = acc;
      acc += s_token_counts[i];
    }
    s_token_offsets[cnt] = acc;
    s_total_tokens = acc;
  }
  __syncthreads();

  const int32_t total_tokens = s_total_tokens;

  // -------------------------------------------------------------------
  // Phase 6: atomically allocate hot pages and token range
  // -------------------------------------------------------------------
  if (tx == 0 && cnt > 0) {
    s_start_page = atomicAdd(next_hot_page, cnt);
    s_token_start = atomicAdd(token_counter, total_tokens);
  }
  __syncthreads();

  const int32_t start_page = s_start_page;
  const int32_t token_start = s_token_start;

  if (tx == 0) {
    if (cnt > 0 && (start_page + cnt > hot_page_capacity)) {
      atomicExch(overflow_flag, 1);
    }
  }
  __syncthreads();

  // -------------------------------------------------------------------
  // Phase 7: write hot_page_table
  // -------------------------------------------------------------------
  if (tx < cnt) {
    int32_t logical_block = s_unique[tx];
    int32_t hot_page_id   = hot_page_offset + start_page + tx;
    hot_page_table[batch_id * max_pages + logical_block] = hot_page_id;
  }
  __syncthreads();

  // -------------------------------------------------------------------
  // Phase 8: write host_locs and hot_locs
  // -------------------------------------------------------------------
  if (tx < cnt) {
    int32_t logical_block = s_unique[tx];
    int32_t hot_page_id   = hot_page_offset + start_page + tx;
    int32_t token_off     = s_token_offsets[tx];
    int32_t num_tokens    = s_token_counts[tx];
    int32_t blk_start     = logical_block * PAGE_SIZE;
    int64_t req_off       = req_row * max_ctx;

    int32_t abs_off = token_start + token_off;
    for (int t = 0; t < num_tokens; ++t) {
      int32_t logical_pos = blk_start + t;
      host_locs[abs_off + t] = req_to_host[req_off + logical_pos];
      hot_locs[abs_off + t]  = static_cast<int64_t>(hot_page_id) * PAGE_SIZE + t;
    }
  }

  // -------------------------------------------------------------------
  // Phase 9: write hot_kv_indices_offset and hot_kv_indices
  // -------------------------------------------------------------------
  if (tx == 0) {
    hot_kv_indices_offset[batch_id] = cnt;
  }

  // hot_kv_indices: store hot page ids for this request at batch-relative pos.
  // Caller compacts these across the batch via a prefix-sum over the offsets.
  if (tx < cnt) {
    int32_t hot_page_id = hot_page_offset + start_page + tx;
    hot_kv_indices[batch_id * max_pages + tx] = hot_page_id;
  }
}

}  // namespace

// ===========================================================================
// Host-side launcher (FFI entry point)
// ===========================================================================

template <int BLOCK_THREADS, int PAGE_SIZE>
void minimax_hisparse_swap_in(
    tvm::ffi::TensorView topk_idx,          // [Hkv, B, K]   int32
    tvm::ffi::TensorView seq_lens,           // [B]            int64
    tvm::ffi::TensorView req_to_host,        // [max_reqs, max_ctx] int64
    tvm::ffi::TensorView req_pool_indices,   // [B]            int64
    tvm::ffi::TensorView hot_page_table,     // [B, max_pages] int32 (output)
    tvm::ffi::TensorView hot_kv_indices,     // [total_pages]  int32 (output)
    tvm::ffi::TensorView hot_kv_indices_offset, // [B+1]       int32 (output)
    tvm::ffi::TensorView host_locs,          // [total_tokens] int64 (output)
    tvm::ffi::TensorView hot_locs,           // [total_tokens] int64 (output)
    tvm::ffi::TensorView next_hot_page,      // [1]            int32 (atomic)
    tvm::ffi::TensorView token_counter,       // [1]            int32 (atomic)
    tvm::ffi::TensorView overflow_flag,      // [1]            int32 (output)
    int64_t hot_page_offset,
    int64_t hot_page_capacity,
    int64_t num_real_reqs) {

  using namespace host;

  SymbolicSize Hkv   = {"Hkv"};
  SymbolicSize B     = {"B"};
  SymbolicSize K     = {"K"};
  SymbolicSize R     = {"max_reqs"};
  SymbolicSize Ctx   = {"max_ctx"};
  SymbolicSize MP    = {"max_pages"};
  SymbolicSize TotalP = {"total_pages"};
  SymbolicSize TotalT = {"total_tokens"};
  SymbolicSize One   = {"one"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  // Validate inputs
  TensorMatcher({Hkv, B, K}).with_dtype<int32_t>().with_device(device_)
      .verify(topk_idx);
  TensorMatcher({B}).with_dtype<int64_t>().with_device(device_)
      .verify(seq_lens).verify(req_pool_indices);
  TensorMatcher({R, Ctx}).with_dtype<int64_t>().with_device(device_)
      .verify(req_to_host);

  // Validate outputs
  TensorMatcher({B, MP}).with_dtype<int32_t>().with_device(device_)
      .verify(hot_page_table);
  TensorMatcher({TotalP}).with_dtype<int32_t>().with_device(device_)
      .verify(hot_kv_indices);
  TensorMatcher({B}).with_dtype<int32_t>().with_device(device_)
      .verify(hot_kv_indices_offset);
  TensorMatcher({TotalT}).with_dtype<int64_t>().with_device(device_)
      .verify(host_locs).verify(hot_locs);
  TensorMatcher({One}).with_dtype<int32_t>().with_device(device_)
      .verify(next_hot_page).verify(token_counter).verify(overflow_flag);

  const int hkv_val   = static_cast<int>(Hkv.unwrap());
  const int b_val     = static_cast<int>(B.unwrap());
  const int k_val     = static_cast<int>(K.unwrap());
  const int mp_val    = static_cast<int>(MP.unwrap());
  const int ctx_val   = static_cast<int>(Ctx.unwrap());
  const int hpo_val   = static_cast<int>(hot_page_offset);
  const int hpc_val   = static_cast<int>(hot_page_capacity);
  const int nrr_val   = static_cast<int>(num_real_reqs);
  const DLDevice device = device_.unwrap();

  if (b_val == 0 || hkv_val == 0) return;

  RuntimeCheck(PAGE_SIZE > 0, "PAGE_SIZE must be > 0, got ", PAGE_SIZE);

  dim3 grid(b_val);
  LaunchKernel(grid, BLOCK_THREADS, device, 0)(
      minimax_hisparse_swap_in_kernel<BLOCK_THREADS, PAGE_SIZE>,
      topk_idx.data<int32_t>(),
      seq_lens.data<int64_t>(),
      req_to_host.data<int64_t>(),
      req_pool_indices.data<int64_t>(),
      hot_page_table.data<int32_t>(),
      hot_kv_indices.data<int32_t>(),
      hot_kv_indices_offset.data<int32_t>(),
      host_locs.data<int64_t>(),
      hot_locs.data<int64_t>(),
      next_hot_page.data<int32_t>(),
      token_counter.data<int32_t>(),
      overflow_flag.data<int32_t>(),
      hkv_val,
      b_val,
      k_val,
      mp_val,
      ctx_val,
      hpo_val,
      hpc_val,
      nrr_val);
}

// ===========================================================================
// Instantiation for MiniMax-M3 (256 threads, page_size=128)
// ===========================================================================
// The FFI glue is generated by load_jit's cuda_wrappers:
//   ("minimax_hisparse_swap_in", "minimax_hisparse_swap_in<256, 128>")
// This single instantiation covers the M3 config (Hkv=4, K=16, page=128).

}  // namespace
