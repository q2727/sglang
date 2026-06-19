// MiniMax-M3 HiSparse block swap-in kernel — CUDA-graph-safe.
//
// One CTA per request: dedup → atomic hot-page alloc → page table →
// inline H2D copy (warp PTX from pinned host).
//
// Graph safety:
//   - num_real_reqs: GPU scalar pointer (.fill_() before replay), NOT kernel arg
//   - H2D copy inline (no Python .item() sync)
//   - All outputs pre-allocated fixed-shape

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

static constexpr int kWarpThreads = 32;

// ---- warp bulk copy: pinned host → GPU (same as DSA hisparse.cuh) ----
__device__ __forceinline__ void
transfer_item_warp(const void* src, void* dst, int item_size_bytes) {
  const int lane_id = threadIdx.x % kWarpThreads;
  const int total_pairs = item_size_bytes / 16;
  if (total_pairs > 0) {
    const uint64_t* src64 = static_cast<const uint64_t*>(src);
    uint64_t*       dst64 = static_cast<uint64_t*>(dst);
    for (int j = lane_id; j < total_pairs; j += kWarpThreads) {
      uint64_t lo, hi;
      const uint64_t* s = src64 + j * 2;
      asm volatile("ld.global.nc.v2.b64 {%0,%1}, [%2];"
                   : "=l"(lo), "=l"(hi) : "l"(s) : "memory");
      uint64_t* d = dst64 + j * 2;
      asm volatile("st.global.cs.v2.b64 [%0], {%1,%2};"
                   :: "l"(d), "l"(lo), "l"(hi) : "memory");
    }
  }
  const int tail_8B = (item_size_bytes - total_pairs * 16) / 8;
  if (tail_8B > 0 && lane_id < tail_8B) {
    const uint64_t* src8 = reinterpret_cast<const uint64_t*>(
        static_cast<const char*>(src) + total_pairs * 16);
    uint64_t* dst8 = reinterpret_cast<uint64_t*>(
        static_cast<char*>(dst) + total_pairs * 16);
    uint64_t tmp;
    asm volatile("ld.global.nc.b64 %0, [%1];" : "=l"(tmp) : "l"(src8 + lane_id) : "memory");
    asm volatile("st.global.cs.b64 [%0], %1;" :: "l"(dst8 + lane_id), "l"(tmp) : "memory");
  }
}

// ===================================================================
template <int BLOCK_THREADS, int PAGE_SIZE>
__global__ void minimax_hisparse_swap_in_kernel(
    const int32_t* __restrict__ topk_idx,
    const int64_t* __restrict__ seq_lens,
    const int64_t* __restrict__ req_to_host,
    const int64_t* __restrict__ req_pool_indices,
    const void* __restrict__ host_k_ptr,
    const void* __restrict__ host_v_ptr,
    void* __restrict__ hot_k_ptr,
    void* __restrict__ hot_v_ptr,
    int32_t* __restrict__ hot_page_table,
    int32_t* __restrict__ hot_kv_indices,
    int32_t* __restrict__ hot_kv_indices_offset,
    int32_t* __restrict__ next_hot_page,
    const int32_t* __restrict__ num_real_reqs_ptr,  // GPU scalar
    int32_t* __restrict__ overflow_flag,
    int Hkv, int B, int K,
    int max_pages, int max_ctx,
    int hot_page_offset, int hot_page_capacity,
    int head_num, int head_dim, int elem_size_bytes) {

  using namespace device;
  const int batch_id = blockIdx.x;
  if (batch_id >= num_real_reqs_ptr[0]) return;   // graph-safe gate

  const int tx = threadIdx.x;
  const int total_entries = Hkv * K;
  static constexpr int kMaxEntries = 128;
  constexpr int kWarps = (BLOCK_THREADS + kWarpThreads - 1) / kWarpThreads;

  __shared__ int32_t s_input[kMaxEntries], s_valid[kMaxEntries];
  __shared__ int32_t s_unique[kMaxEntries], s_token_counts[kMaxEntries];
  __shared__ int32_t s_token_offsets[kMaxEntries + 1], s_warp_sums[kWarps];
  __shared__ int64_t s_req_row, s_seq_len;
  __shared__ int32_t s_num_pages, s_count, s_start_page, s_total_tokens;
  __shared__ int32_t s_has_local, s_local_idx;    // local-block skip

  if (tx == 0) {
    s_req_row=req_pool_indices[batch_id]; s_seq_len=seq_lens[batch_id];
    s_num_pages=(s_seq_len+PAGE_SIZE-1)/PAGE_SIZE; s_count=0; s_total_tokens=0;
    s_has_local=0; s_local_idx=-1;
  }
  __syncthreads();
  const int64_t seq_len=s_seq_len, req_row=s_req_row;
  const int num_pages=s_num_pages;
  const int32_t local_block_id = num_pages - 1;  // last logical block = local

  // P1 load
  if (tx < total_entries) {
    int h=tx/K, k=tx%K;
    int32_t blk=topk_idx[h*B*K + batch_id*K + k];
    s_input[tx]=(blk>=0 && blk<num_pages)?blk:-1;
  } else if(tx<kMaxEntries) s_input[tx]=-1;
  __syncthreads();

  // P2 dedup
  if (tx < total_entries) {
    int32_t blk=s_input[tx], first=(blk>=0)?1:0;
    if(first) for(int j=0;j<tx;++j) if(s_input[j]==blk){first=0;break;}
    s_valid[tx]=first;
  } else if(tx<kMaxEntries) s_valid[tx]=0;
  __syncthreads();

  // P3 compact
  const int lane=tx%kWarpThreads, warp=tx/kWarpThreads;
  int32_t mv=(tx<total_entries)?s_valid[tx]:0, pfx=mv;
#pragma unroll
  for(int o=1;o<kWarpThreads;o<<=1){int32_t n=__shfl_up_sync(kWarpSyncMask,pfx,o,kWarpThreads);if(lane>=o)pfx+=n;}
  int32_t ws=__shfl_sync(kWarpSyncMask,pfx,kWarpThreads-1);
  if(lane==0)s_warp_sums[warp]=ws;
  __syncthreads();
  if(tx==0){int32_t a=0;for(int w=0;w<kWarps;++w){int32_t wv=s_warp_sums[w];s_warp_sums[w]=a;a+=wv;}s_count=a;}
  __syncthreads();
  int32_t wpfx=s_warp_sums[warp];
  if(mv&&tx<total_entries)s_unique[wpfx+(pfx-mv)]=s_input[tx];
  __syncthreads();
  const int32_t cnt=s_count;

  // P3.5: identify local block (last logical block), swap to end of s_unique
  if(tx<cnt && s_unique[tx]==local_block_id){s_has_local=1; s_local_idx=tx;}
  __syncthreads();
  if(s_has_local && tx==0){
    int32_t last=cnt-1, li=s_local_idx;
    if(li!=last){int32_t tmp=s_unique[li]; s_unique[li]=s_unique[last]; s_unique[last]=tmp;}
  }
  __syncthreads();
  const int32_t has_local = s_has_local;
  const int32_t nonlocal_cnt = cnt - has_local;  // blocks needing H2D copy

  // P4 token counts (recompute after possible swap)
  if(tx<cnt){int32_t bs=s_unique[tx]*PAGE_SIZE;s_token_counts[tx]=max(0,min(PAGE_SIZE,(int32_t)seq_len-bs));}
  __syncthreads();

  // P5 prefix sum
  if(tx==0){int32_t a=0;for(int i=0;i<cnt;++i){s_token_offsets[i]=a;a+=s_token_counts[i];}s_token_offsets[cnt]=a;s_total_tokens=a;}
  __syncthreads();

  // P6 atomic alloc — only allocate for non-local blocks
  if(tx==0&&nonlocal_cnt>0)s_start_page=atomicAdd(next_hot_page,nonlocal_cnt);
  __syncthreads();
  if(tx==0&&nonlocal_cnt>0&&(s_start_page+nonlocal_cnt>hot_page_capacity))atomicExch(overflow_flag,1);
  __syncthreads();
  const int32_t sp=s_start_page;

  // P7 page table: local→0 (reserved page), others→hot_page_offset+sp+pos
  // s_unique layout after swap: [nonlocal_0, ..., nonlocal_{k-1}, local (at cnt-1)]
  if(tx<cnt){
    int32_t hp = (has_local && tx==cnt-1) ? 0 : hot_page_offset+sp+tx;
    hot_page_table[batch_id*max_pages+s_unique[tx]]=hp;
  }
  __syncthreads();

  // P8 inline H2D copy — skip local block (it lives in page 0, already resident)
  const int k_stride=head_num*head_dim*elem_size_bytes;
  const int64_t req_off=req_row*(int64_t)max_ctx;
  if(tx<nonlocal_cnt){
    int32_t blk=s_unique[tx],hp=hot_page_offset+sp+tx;
    int32_t bs=blk*PAGE_SIZE,nt=s_token_counts[tx];
    for(int t=0;t<nt;++t){
      int64_t hs=req_to_host[req_off+bs+t];
      int64_t ds=(int64_t)hp*PAGE_SIZE+t;
      transfer_item_warp((const char*)host_k_ptr+hs*k_stride,(char*)hot_k_ptr+ds*k_stride,k_stride);
      transfer_item_warp((const char*)host_v_ptr+hs*k_stride,(char*)hot_v_ptr+ds*k_stride,k_stride);
    }
  }

  // P9 kv_indices: local→0, others→hot_page_offset+sp+pos
  if(tx==0) hot_kv_indices_offset[batch_id]=cnt;
  __syncthreads();
  if(tx<cnt){
    int32_t hp;
    if(has_local && tx==cnt-1) hp=0;
    else hp=hot_page_offset+sp+tx;
    hot_kv_indices[batch_id*max_pages+tx]=hp;
  }
}

}  // namespace

// ===================================================================
template <int BLOCK_THREADS, int PAGE_SIZE>
void minimax_hisparse_swap_in(
    tvm::ffi::TensorView topk_idx, tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView req_to_host, tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView host_k_buffer, tvm::ffi::TensorView host_v_buffer,
    tvm::ffi::TensorView hot_k_buffer, tvm::ffi::TensorView hot_v_buffer,
    tvm::ffi::TensorView hot_page_table, tvm::ffi::TensorView hot_kv_indices,
    tvm::ffi::TensorView hot_kv_indices_offset,
    tvm::ffi::TensorView next_hot_page, tvm::ffi::TensorView num_real_reqs,
    tvm::ffi::TensorView overflow_flag,
    int64_t hot_page_offset, int64_t hot_page_capacity,
    int64_t head_num, int64_t head_dim, int64_t elem_size_bytes) {

  using namespace host;
  SymbolicSize Hkv{"Hkv"},B_{"B"},K_{"K"},R{"max_reqs"},Ctx{"max_ctx"};
  SymbolicSize HostN{"host_size"},Heads{"head_num"},D{"head_dim"},HotN{"hot_size"};
  SymbolicSize MP{"max_pages"},TotalP{"total_pages"},One{"one"};
  SymbolicDevice dev; dev.set_options<kDLCUDA>();

  TensorMatcher({Hkv,B_,K_}).with_dtype<int32_t>().with_device(dev).verify(topk_idx);
  TensorMatcher({B_}).with_dtype<int64_t>().with_device(dev).verify(seq_lens).verify(req_pool_indices);
  TensorMatcher({R,Ctx}).with_dtype<int64_t>().with_device(dev).verify(req_to_host);
  TensorMatcher({HostN,Heads,D}).verify(host_k_buffer).verify(host_v_buffer);
  TensorMatcher({HotN,Heads,D}).with_device(dev).verify(hot_k_buffer).verify(hot_v_buffer);
  TensorMatcher({B_,MP}).with_dtype<int32_t>().with_device(dev).verify(hot_page_table);
  TensorMatcher({TotalP}).with_dtype<int32_t>().with_device(dev).verify(hot_kv_indices);
  TensorMatcher({B_}).with_dtype<int32_t>().with_device(dev).verify(hot_kv_indices_offset);
  TensorMatcher({One}).with_dtype<int32_t>().with_device(dev)
      .verify(next_hot_page).verify(num_real_reqs).verify(overflow_flag);

  auto go=[](auto&s){return static_cast<int>(s.unwrap());};
  if(go(B_)==0||go(Hkv)==0)return;
  dim3 g(go(B_));
  LaunchKernel(g, BLOCK_THREADS, dev.unwrap(), 0)(
      minimax_hisparse_swap_in_kernel<BLOCK_THREADS, PAGE_SIZE>,
      topk_idx.data<int32_t>(),seq_lens.data<int64_t>(),
      req_to_host.data<int64_t>(),req_pool_indices.data<int64_t>(),
      host_k_buffer.data<void>(),host_v_buffer.data<void>(),
      hot_k_buffer.data<void>(),hot_v_buffer.data<void>(),
      hot_page_table.data<int32_t>(),hot_kv_indices.data<int32_t>(),
      hot_kv_indices_offset.data<int32_t>(),
      next_hot_page.data<int32_t>(),num_real_reqs.data<int32_t>(),
      overflow_flag.data<int32_t>(),
      go(Hkv),go(B_),go(K_),go(MP),go(Ctx),
      (int)hot_page_offset,(int)hot_page_capacity,
      (int)head_num,(int)head_dim,(int)elem_size_bytes);
}

}  // namespace
