#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cfloat>
#include <climits>

// ═══════════════════════════════════════════════════════════════════════════
// Sophia's sort/merge/pick — extracted from her utils.cuh / search_bitonic.cuh
// ═══════════════════════════════════════════════════════════════════════════

namespace sophia {

#define S_MAX_INDEX UINT_MAX

template <class K, class V>
__device__ inline void swap_if_needed(K& k0, V& v0, const unsigned lane_offset, const bool asc) {
  auto k1 = __shfl_xor_sync(~0u, k0, lane_offset);
  auto v1 = __shfl_xor_sync(~0u, v0, lane_offset);
  if ((k0 != k1) && ((k0 < k1) != asc)) { k0 = k1; v0 = v1; }
}

template <class K, class V>
__device__ inline void swap_if_needed(K& k0, V& v0, K& k1, V& v1, const bool asc) {
  if ((k0 != k1) && ((k0 < k1) != asc)) {
    const auto tmp_k = k0; k0 = k1; k1 = tmp_k;
    const auto tmp_v = v0; v0 = v1; v1 = tmp_v;
  }
}

template <class K, class V, unsigned _N, unsigned warp_size>
__device__ inline void warp_merge_core(K k[2], V v[2], const std::uint32_t range, const bool asc) {
  static_assert((_N & (_N - 1)) == 0, "");
  if constexpr (_N == 2) {
    constexpr unsigned N = 2;
    const auto lane_id = threadIdx.x % warp_size;
    if (range == 1) {
      const auto p = ((lane_id & 1) == 0);
      swap_if_needed<float, uint32_t>(k[0], v[0], k[1], v[1], p);
      return;
    }
    const std::uint32_t b = range;
    for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
      const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
      for (std::uint32_t i = 0; i < N; i++) swap_if_needed<float, uint32_t>(k[i], v[i], c, p);
    }
    const auto p = ((lane_id & b) == 0);
    swap_if_needed<float, uint32_t>(k[0], v[0], k[1], v[1], p);
  } else if constexpr (_N >= 4) {
    constexpr unsigned N = _N;
    const auto lane_id = threadIdx.x % warp_size;
    if (range == 1) {
      for (std::uint32_t b = 2; b <= N; b <<= 1) {
        for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
          for (std::uint32_t i = 0; i < N; i++) {
            std::uint32_t j = i ^ c;
            if (i >= j) continue;
            const auto line_id = i + (N * lane_id);
            const auto p = static_cast<bool>(line_id & b) == static_cast<bool>(line_id & c);
            swap_if_needed(k[i], v[i], k[j], v[j], p);
          }
        }
      }
      return;
    }
    const std::uint32_t b = range;
    for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
      const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
      for (std::uint32_t i = 0; i < N; i++) swap_if_needed(k[i], v[i], c, p);
    }
    const auto p = ((lane_id & b) == 0);
    for (std::uint32_t c = N / 2; c >= 1; c >>= 1) {
      for (std::uint32_t i = 0; i < N; i++) {
        std::uint32_t j = i ^ c;
        if (i >= j) continue;
        swap_if_needed(k[i], v[i], k[j], v[j], p);
      }
    }
  }
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_merge(K k[N], V v[N], unsigned range, const bool asc = true) {
  warp_merge_core<K, V, N, warp_size>(k, v, range, asc);
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_sort(K k[N], V v[N], const bool asc = true) {
  for (std::uint32_t range = 1; range <= warp_size; range <<= 1)
    warp_merge<K, V, N, warp_size>(k, v, range, asc);
}

template <unsigned N_1, unsigned N_2>
__device__ void topk_by_bitonic_sort(
    uint32_t* result_indices_ptr, float* result_distances_ptr,
    uint32_t RESULT_BUFFER_SIZE, uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t INTERNAL_TOPK, bool first) {
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;
  if (warp_id > 0) return;

  float key_1[N_1]; uint32_t val_1[N_1];
  auto cd = result_distances_ptr + INTERNAL_TOPK;
  auto ci = result_indices_ptr + INTERNAL_TOPK;
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = lane_id + (32 * i);
    key_1[i] = (j < CANDIDATE_BUFFER_SIZE) ? cd[j] : FLT_MAX;
    val_1[i] = (j < CANDIDATE_BUFFER_SIZE) ? ci[j] : S_MAX_INDEX;
  }
  warp_sort<float, uint32_t, N_1>(key_1, val_1);
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = (N_1 * lane_id) + i;
    if (j < CANDIDATE_BUFFER_SIZE && j < INTERNAL_TOPK) { cd[j] = key_1[i]; ci[j] = val_1[i]; }
  }

  float key_2[N_2]; uint32_t val_2[N_2];
  if (first) {
    for (unsigned i = 0; i < N_2; i++) {
      unsigned j = lane_id + (32 * i);
      key_2[i] = (j < INTERNAL_TOPK) ? result_distances_ptr[j] : FLT_MAX;
      val_2[i] = (j < INTERNAL_TOPK) ? result_indices_ptr[j] : S_MAX_INDEX;
    }
    warp_sort<float, uint32_t, N_2>(key_2, val_2);
  } else {
    for (unsigned i = 0; i < N_2; i++) {
      unsigned j = (N_2 * lane_id) + i;
      key_2[i] = (j < INTERNAL_TOPK) ? result_distances_ptr[j] : FLT_MAX;
      val_2[i] = (j < INTERNAL_TOPK) ? result_indices_ptr[j] : S_MAX_INDEX;
    }
  }

  for (unsigned i = 0; i < N_2; i++) {
    unsigned j = (N_2 * lane_id) + i;
    unsigned k = INTERNAL_TOPK - 1 - j;
    if (k >= INTERNAL_TOPK || k >= CANDIDATE_BUFFER_SIZE) continue;
    if (key_2[i] > cd[k]) { key_2[i] = cd[k]; val_2[i] = ci[k]; }
  }
  warp_merge<float, uint32_t, N_2>(key_2, val_2, 32);

  for (unsigned i = 0; i < N_2; i++) {
    unsigned j = (N_2 * lane_id) + i;
    if (j < INTERNAL_TOPK) { result_distances_ptr[j] = key_2[i]; result_indices_ptr[j] = val_2[i]; }
  }
}

__device__ uint32_t pick_expanders(int N, uint32_t* output_list, int M, uint32_t* pri_queue) {
  uint32_t num_exp = 0;
  constexpr uint32_t msb = 0x80000000;
  uint32_t itopk_max = M;
  if (itopk_max % 32) itopk_max += 32 - (itopk_max % 32);
  for (uint32_t j = threadIdx.x; j < itopk_max; j += 32) {
    int new_parent = 0;
    if (j < (uint32_t)M) {
      uint32_t index = pri_queue[j];
      if ((index & msb) == 0 && index != S_MAX_INDEX) new_parent = 1;
    }
    const uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
    if (new_parent) {
      const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_exp;
      if (i < (uint32_t)N) { output_list[i] = j; pri_queue[j] |= msb; }
    }
    num_exp += __popc(ballot_mask);
    if (num_exp >= (uint32_t)N) break;
  }
  return num_exp;
}

}  // namespace sophia

// ═══════════════════════════════════════════════════════════════════════════

__global__ void kernel_sophia_bench(
    float* g_cand_dist, uint32_t* g_cand_idx,
    unsigned topk_size, unsigned cand_size, unsigned n_iters)
{
    extern __shared__ char smem[];
    unsigned buf_size = topk_size + cand_size;
    uint32_t* all_idx  = reinterpret_cast<uint32_t*>(smem);
    float*    all_dist = reinterpret_cast<float*>(smem + buf_size * sizeof(uint32_t));
    uint32_t* parent   = reinterpret_cast<uint32_t*>(
        smem + buf_size * (sizeof(uint32_t) + sizeof(float)));
    const unsigned tid = threadIdx.x;

    for (unsigned i = tid; i < topk_size; i += blockDim.x) {
        all_idx[i] = UINT_MAX; all_dist[i] = FLT_MAX;
    }
    __syncthreads();

    for (unsigned it = 0; it < n_iters; it++) {
        for (unsigned i = tid; i < cand_size; i += blockDim.x) {
            all_idx[topk_size + i]  = g_cand_idx[it * cand_size + i];
            all_dist[topk_size + i] = g_cand_dist[it * cand_size + i];
        }
        __syncthreads();
        sophia::topk_by_bitonic_sort<2, 8>(
            all_idx, all_dist, buf_size, cand_size, topk_size, (it == 0));
        __syncthreads();
        if (tid < 32) {
            sophia::pick_expanders(1, parent, topk_size, all_idx);
        }
        __syncthreads();
    }
}

// ═══════════════════════════════════════════════════════════════════════════

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } \
} while(0)

int main() {
    constexpr unsigned TOPK = 256;
    constexpr unsigned CAND = 64;
    constexpr unsigned ITERS = 128;
    constexpr unsigned NBLOCKS = 1024;
    constexpr unsigned NRUNS = 5;
    constexpr unsigned TOTAL = ITERS * CAND;

    std::mt19937 rng(42);
    std::vector<float> h_dist(TOTAL);
    std::vector<uint32_t> h_idx(TOTAL);
    for (unsigned i = 0; i < TOTAL; i++) {
        h_dist[i] = static_cast<float>(rng() % 1000000);
        h_idx[i]  = rng() % 1000000;
    }

    float *d_dist; uint32_t *d_idx;
    CUDA_CHECK(cudaMalloc(&d_dist, TOTAL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx,  TOTAL * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_dist, h_dist.data(), TOTAL * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idx,  h_idx.data(),  TOTAL * sizeof(uint32_t), cudaMemcpyHostToDevice));

    size_t smem = (TOPK + CAND) * (sizeof(uint32_t) + sizeof(float)) + sizeof(uint32_t);
    CUDA_CHECK(cudaFuncSetAttribute(kernel_sophia_bench,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));

    kernel_sophia_bench<<<NBLOCKS, 256, smem>>>(d_dist, d_idx, TOPK, CAND, ITERS);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < NRUNS; r++) {
        kernel_sophia_bench<<<NBLOCKS, 256, smem>>>(d_dist, d_idx, TOPK, CAND, ITERS);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - start).count() / NRUNS;
    printf("sophia merge+pick: %.1f us  (%d blocks x %d iters)\n", us, NBLOCKS, ITERS);

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_idx));
    return 0;
}
