#pragma once
#include <cfloat>
#include <climits>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define FULL_MASK     0xffffffff
#define BLOCK_SIZE    256
#define WARP_SIZE     32
#define LOG_WARP_SIZE 5
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define MAX_INDEX UINT_MAX
#define SHFL_DOWN(val, offset) __shfl_down_sync(FULL_MASK, val, offset)
#define SHFL(val, lane) __shfl_sync(FULL_MASK, val, lane)
#define SHFL_XOR(val, mask) __shfl_xor_sync(FULL_MASK, val, mask)
typedef unsigned long long gpu_long_t; // for counters
#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit( cudaStatus );                                                             \
        }                                                                                   \
    }
#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                      \
    cudaError err = call;                                                      \
    if( cudaSuccess != err) {                                                  \
        fprintf(stderr, "error %d: Cuda error in file '%s' in line %i : %s.\n",\
                err, __FILE__, __LINE__, cudaGetErrorString( err) );           \
        exit(EXIT_FAILURE);                                                    \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);              \

#  define CUDA_SAFE_THREAD_SYNC( ) {                                           \
    cudaError err = CUT_DEVICE_SYNCHRONIZE();                                  \
    if ( cudaSuccess != err) {                                                 \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",          \
                __FILE__, __LINE__, cudaGetErrorString( err) );                \
    } }
template <typename T>
class GraphGPU {
  using vid_t = uint32_t;
protected:
  vid_t nv{0};           // number of vertices
  vid_t maxDeg{0};       // maximum degree
  vid_t* d_edges{nullptr};
public:
  GraphGPU() = default;
  __host__ __forceinline__ GraphGPU(vid_t n, vid_t d, vid_t* d_edges_in)
      : nv(n), maxDeg(d), d_edges(d_edges_in) {}

  inline __device__ __host__ vid_t V() const { return nv; }
  inline __device__ __host__ vid_t get_max_degree() const { return maxDeg; }
  inline __device__ __host__ vid_t* N(vid_t vid) const {
    return d_edges + static_cast<size_t>(vid) * static_cast<size_t>(maxDeg);
  }
};

template <typename T = float>
__device__ __forceinline__ T warp_l2_distance(int dim, const T* a, const T* b) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  T val = 0.; 
  for(int i = thread_lane; i < dim; i += WARP_SIZE)
    val += (a[i] - b[i]) * (a[i] - b[i]);
  T sum = val;
  sum += SHFL_DOWN(sum, 16);
  sum += SHFL_DOWN(sum, 8); 
  sum += SHFL_DOWN(sum, 4); 
  sum += SHFL_DOWN(sum, 2); 
  sum += SHFL_DOWN(sum, 1); 
  sum  = SHFL(sum, 0); 
  return sum;
}
__device__ __forceinline__ float warp_l2_distance_128(const float* __restrict__ a,
                                                      const float* __restrict__ b) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const float4 av = reinterpret_cast<const float4*>(a)[lane];
  const float4 bv = reinterpret_cast<const float4*>(b)[lane];
  float dx0 = av.x - bv.x;
  float dx1 = av.y - bv.y;
  float dx2 = av.z - bv.z;
  float dx3 = av.w - bv.w;
  float sum = fmaf(dx0, dx0, fmaf(dx1, dx1, fmaf(dx2, dx2, dx3 * dx3)));
  sum += __shfl_down_sync(FULL_MASK, sum, 16);
  sum += __shfl_down_sync(FULL_MASK, sum, 8);
  sum += __shfl_down_sync(FULL_MASK, sum, 4);
  sum += __shfl_down_sync(FULL_MASK, sum, 2);
  sum += __shfl_down_sync(FULL_MASK, sum, 1);
  return __shfl_sync(FULL_MASK, sum, 0);
}
__device__ __forceinline__ float subwarp8_l2_distance_128_leader(const float* __restrict__ a,
                                                                 const float* __restrict__ b) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int sublane = lane & 7;               // 0..7 within each 8-lane group
  const int base = sublane * 16;              // 16 floats per thread
  const float4* ap = reinterpret_cast<const float4*>(a + base);
  const float4* bp = reinterpret_cast<const float4*>(b + base);
  float sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    const float4 av = ap[i];
    const float4 bv = bp[i];
    const float dx0 = av.x - bv.x;
    const float dx1 = av.y - bv.y;
    const float dx2 = av.z - bv.z;
    const float dx3 = av.w - bv.w;
    sum = fmaf(dx0, dx0, sum);
    sum = fmaf(dx1, dx1, sum);
    sum = fmaf(dx2, dx2, sum);
    sum = fmaf(dx3, dx3, sum);
  }
  sum += __shfl_down_sync(FULL_MASK, sum, 4, 8);
  sum += __shfl_down_sync(FULL_MASK, sum, 2, 8);
  sum += __shfl_down_sync(FULL_MASK, sum, 1, 8);
  return sum;  // only valid for sublane==0
}

/*-------------------------------------------- sort --------------------------------------------*/
typedef float DATA_T;
typedef uint32_t INDEX_T;
typedef float DISTANCE_T;

template <class K, class V>
__device__ inline void swap_if_needed(K& k0, V& v0, const unsigned lane_offset, const bool asc) {
  auto k1 = __shfl_xor_sync(~0u, k0, lane_offset);
  auto v1 = __shfl_xor_sync(~0u, v0, lane_offset);
  if ((k0 != k1) && ((k0 < k1) != asc)) {
    k0 = k1;
    v0 = v1;
  }
}

template <class K, class V>
__device__ inline void swap_if_needed(K& k0, V& v0, K& k1, V& v1, const bool asc) {
  if ((k0 != k1) && ((k0 < k1) != asc)) {
    const auto tmp_k = k0;
    k0               = k1;
    k1               = tmp_k;
    const auto tmp_v = v0;
    v0               = v1;
    v1               = tmp_v;
  }
}

template <class K, class V, unsigned _N, unsigned warp_size>
__device__ inline void warp_merge_core(K k[2], V v[2], const std::uint32_t range, const bool asc) {
  if (_N == 2) {
    constexpr unsigned N = 2;
    const auto lane_id   = threadIdx.x % warp_size;

    if (range == 1) {
      const auto p = ((lane_id & 1) == 0);
      swap_if_needed<float, uint32_t>(k[0], v[0], k[1], v[1], p);
      return;
    }

    const std::uint32_t b = range;
    for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
      const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
      #pragma unroll
      for (std::uint32_t i = 0; i < N; i++) {
        swap_if_needed<float, uint32_t>(k[i], v[i], c, p);
      }
    }
    const auto p = ((lane_id & b) == 0);
    swap_if_needed<float, uint32_t>(k[0], v[0], k[1], v[1], p);
  } else if (_N == 4) {
    constexpr unsigned N = 4;
    const auto lane_id = threadIdx.x % warp_size;
    if (range == 1) {
      for (std::uint32_t b = 2; b <= N; b <<= 1) {
        for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
          #pragma unroll
          for (std::uint32_t i = 0; i < N; i++) {
            std::uint32_t j = i ^ c;
            if (i >= j) continue;
            const auto line_id = i + (N * lane_id);
            const auto p       = static_cast<bool>(line_id & b) == static_cast<bool>(line_id & c);
            swap_if_needed(k[i], v[i], k[j], v[j], p);
          }
        }
      }
      return;
    }

    const std::uint32_t b = range;
    for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
      const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
      #pragma unroll
      for (std::uint32_t i = 0; i < N; i++) {
        swap_if_needed(k[i], v[i], c, p);
      }
    }
    const auto p = ((lane_id & b) == 0);
    for (std::uint32_t c = N / 2; c >= 1; c >>= 1) {
    #pragma unroll
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
  for (std::uint32_t range = 1; range <= warp_size; range <<= 1) {
    warp_merge<K, V, N, warp_size>(k, v, range, asc);
  }
}

template <unsigned N_1, unsigned N_2>
__device__ void candidate_by_bitonic_sort(
    INDEX_T* candidate_indices,
    DISTANCE_T* candidate_distances,
    uint32_t CANDIDATE_BUFFER_SIZE) {
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;

  // sort 1
  if (warp_id > 0) return;
  if (CANDIDATE_BUFFER_SIZE != 64) {
    printf("CANDIDATE_BUFFER_SIZE must be 64\n");
    assert(false);
  }
  // constexpr unsigned N_1 = 2;
  DISTANCE_T key_1[N_1];
  INDEX_T val_1[N_1];

  /* Candidates -> Reg */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = lane_id + (32 * i);
    if (j < CANDIDATE_BUFFER_SIZE) {
      key_1[i] = candidate_distances[j];
      val_1[i] = candidate_indices[j];
    } else {
      key_1[i] = FLT_MAX;
      val_1[i] = MAX_INDEX;
    }
  }
  /* Sort */
  warp_sort<float, uint32_t, N_1>(key_1, val_1);
  /* Reg -> Temp_itopk */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = (N_1 * lane_id) + i;
    if (j < CANDIDATE_BUFFER_SIZE) {
      candidate_distances[j] = key_1[i];
      candidate_indices[j]   = val_1[i];
    }
  }
}

template <unsigned N_1, unsigned N_2>
__device__ void candidate_by_bitonic_sort_inverse(
    INDEX_T* candidate_indices,
    DISTANCE_T* candidate_distances,
    uint32_t CANDIDATE_BUFFER_SIZE) {
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;

  // sort 1
  if (warp_id > 0) { return; }
  if (CANDIDATE_BUFFER_SIZE != 64) {
    printf("CANDIDATE_BUFFER_SIZE must be 64\n");
    assert(false);
  }
  // constexpr unsigned N_1 = 2;
  DISTANCE_T key_1[N_1];
  INDEX_T val_1[N_1];

  /* Candidates -> Reg */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = lane_id + (32 * i);
    if (j < CANDIDATE_BUFFER_SIZE) {
      key_1[i] = candidate_distances[j];
      val_1[i] = candidate_indices[j];
    } else {
      key_1[i] = FLT_MAX;
      val_1[i] = MAX_INDEX;
    }
  }
  /* Sort */
  warp_sort<float, uint32_t, N_1>(key_1, val_1);
  /* Reg -> Temp_itopk */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = CANDIDATE_BUFFER_SIZE - 1 - ( (N_1 * lane_id) + i );
    if (j < CANDIDATE_BUFFER_SIZE) {
      candidate_distances[j] = key_1[i];
      candidate_indices[j]   = val_1[i];
    }
  }
}

template <unsigned N_1, unsigned N_2>
__device__ void topk_by_bitonic_sort(
    INDEX_T* result_indices_ptr,
    DISTANCE_T* result_distances_ptr,
    uint32_t RESULT_BUFFER_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t INTERNAL_TOPK,
    bool first) {
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;

  // Sort part 1
  if (warp_id > 0) return;
  assert (CANDIDATE_BUFFER_SIZE <= 64);
  
  DISTANCE_T key_1[N_1];
  INDEX_T val_1[N_1];
  auto candidate_distances = result_distances_ptr + INTERNAL_TOPK;
  auto candidate_indices = result_indices_ptr + INTERNAL_TOPK;
  /* Candidates -> Reg */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = lane_id + (32 * i);
    if (j < CANDIDATE_BUFFER_SIZE) {
      key_1[i] = candidate_distances[j];
      val_1[i] = candidate_indices[j];
    } else {
      key_1[i] = FLT_MAX;
      val_1[i] = MAX_INDEX;
    }
  }
  /* Sort */
  warp_sort<float, uint32_t, N_1>(key_1, val_1);
  /* Reg -> Temp_itopk */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = (N_1 * lane_id) + i;
    if (j < CANDIDATE_BUFFER_SIZE && j < INTERNAL_TOPK) {
      candidate_distances[j] = key_1[i];
      candidate_indices[j]   = val_1[i];
    }
  }

  // Sort part 2
  // constexpr unsigned N_2 = 4;
  DISTANCE_T key_2[N_2];
  INDEX_T val_2[N_2];
  if (first) {
    /* Load itopk results */
    for (unsigned i = 0; i < N_2; i++) {
      unsigned j = lane_id + (32 * i);
      if (j < INTERNAL_TOPK) {
        key_2[i] = result_distances_ptr[j];
        val_2[i] = result_indices_ptr[j];
      } else {
        key_2[i] = FLT_MAX;
        val_2[i] = MAX_INDEX;
      }
    }
    /* Warp Sort */
    warp_sort<float, uint32_t, N_2>(key_2, val_2);
  } else {
    /* Load itopk results */
    for (unsigned i = 0; i < N_2; i++) {
      unsigned j = (N_2 * lane_id) + i;
      if (j < INTERNAL_TOPK) {
        key_2[i] = result_distances_ptr[j];
        val_2[i] = result_indices_ptr[j];
      } else {
        key_2[i] = FLT_MAX;
        val_2[i] = MAX_INDEX;
      }
    }
  }
  /* Merge candidates */
  for (unsigned i = 0; i < N_2; i++) {
    unsigned j = (N_2 * lane_id) + i;  // [0:MAX_ITOPK-1]
    unsigned k = INTERNAL_TOPK - 1 - j;
    if (k >= INTERNAL_TOPK || k >= CANDIDATE_BUFFER_SIZE) continue;
    auto candidate_key = candidate_distances[k];
    if (key_2[i] > candidate_key) {
      key_2[i] = candidate_key;
      val_2[i] = candidate_indices[k];
    }
  }
  /* Warp Merge */
  warp_merge<float, uint32_t, N_2>(key_2, val_2, 32);

  /* Store new itopk results */
  for (unsigned i = 0; i < N_2; i++) {
    unsigned j = (N_2 * lane_id) + i;
    if (j < INTERNAL_TOPK) {
      result_distances_ptr[j] = key_2[i];
      result_indices_ptr[j]   = val_2[i];
    }
  }
}

template <typename T>
__device__ inline void print_buffer(T* buffer, uint32_t buffer_size, uint32_t index_mask=0x80000000) {
    if constexpr (std::is_same<T, INDEX_T>::value) {
        for (uint32_t i = 0; i < buffer_size; i++) {
            printf("%d, ", buffer[i] & ~index_mask);
        }
    } else if constexpr (std::is_same<T, DISTANCE_T>::value || std::is_same<T, DATA_T>::value) {
        for (uint32_t i = 0; i < buffer_size; i++) {
            printf("%f, ", buffer[i]);
        }
    } else {
        printf("Unsupported type\n");
        assert(false);
    }
    printf("\n");
}

#ifndef SMALL_HASH_RESET_INTERVAL
#define SMALL_HASH_RESET_INTERVAL 16
#endif

__host__ __device__ inline uint32_t hashtable_getsize(const uint32_t bitlen) {
  return 1 << bitlen;
}

__device__ inline void hashtable_init(INDEX_T* const table, const unsigned bitlen, unsigned FIRST_TID = 0) {
  if (threadIdx.x < FIRST_TID) return;
  for (uint32_t i = threadIdx.x - FIRST_TID; i < hashtable_getsize(bitlen); i += blockDim.x - FIRST_TID) {
    table[i] = MAX_INDEX;
  }
}

__device__ inline uint32_t hashtable_insert(INDEX_T* const table, const unsigned bitlen, const INDEX_T key) {
    const uint32_t size     = hashtable_getsize(bitlen);
    const uint32_t bit_mask = size - 1;
    uint32_t h = static_cast<uint32_t>(key);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    uint32_t index = h & bit_mask;
    uint32_t stride = ((h >> bitlen) | 1u);  // odd stride => full period for power-of-two table
    #pragma unroll 1
    for (unsigned i = 0; i < size; i++) {
        const INDEX_T cur = table[index];
        if (cur == key) return 0;
        if (cur == MAX_INDEX) {
            const INDEX_T old = atomicCAS(&table[index], MAX_INDEX, key);
            if (old == MAX_INDEX) return 1;
            if (old == key) return 0;
        }
        index = (index + stride) & bit_mask;
    }
    return 0;
}

__device__ inline uint32_t hashtable_insert_limited(
    INDEX_T* const table, const unsigned bitlen, const INDEX_T key, const unsigned max_probe) {
    const uint32_t size     = hashtable_getsize(bitlen);
    const uint32_t bit_mask = size - 1;
    uint32_t h = static_cast<uint32_t>(key);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    uint32_t index = h & bit_mask;
    uint32_t stride = ((h >> bitlen) | 1u);
    const unsigned probes = (max_probe == 0) ? size : min(max_probe, size);
    #pragma unroll 1
    for (unsigned i = 0; i < probes; i++) {
        const INDEX_T cur = table[index];
        if (cur == key) return 0;
        if (cur == MAX_INDEX) {
            const INDEX_T old = atomicCAS(&table[index], MAX_INDEX, key);
            if (old == MAX_INDEX) return 1;
            if (old == key) return 0;
        }
        index = (index + stride) & bit_mask;
    }
    return 0;
}

__device__ inline void hashtable_restore(   
    INDEX_T* const table, 
    const unsigned bitlen,
    const INDEX_T* itopk_indices,
    const uint32_t itopk_size,
    const uint32_t first_tid = 0) {
  constexpr INDEX_T index_msb_1_mask = 0x80000000;
  if (threadIdx.x < first_tid) return;
  for (unsigned i = threadIdx.x - first_tid; i < itopk_size; i += blockDim.x - first_tid) {
    auto key = itopk_indices[i] & ~index_msb_1_mask;  // clear most significant bit
    hashtable_insert(table, bitlen, key);
  }
}

static __device__ uint32_t pick_expanders(int N, INDEX_T* output_list, int M, INDEX_T* pri_queue) {
  uint32_t num_exp = 0; // number of expanders actually seleceted; must be <= N
  constexpr INDEX_T index_msb_1_mask = 0x80000000;
  uint32_t itopk_max = M;
  if (itopk_max % 32) { itopk_max += 32 - (itopk_max % 32); }
  for (uint32_t j = threadIdx.x; j < itopk_max; j += 32) {
    INDEX_T index;
    int new_parent = 0;
    if (j < M) {
      index = pri_queue[j];
      if ((index & index_msb_1_mask) == 0) {   // check if most significant bit is set
        new_parent = 1;
      }
    }
    const std::uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
    if (new_parent) {
      const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_exp;
      if (i < N) {
        output_list[i] = j;
        pri_queue[j] |= index_msb_1_mask;
      }
    }
    num_exp += __popc(ballot_mask);
    if (num_exp >= N) { break; }
  }
  return num_exp;
}
