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
//#define FLT_MAX 3.40282347e+38
#define GPU_SEARCH_STATS 1
#define GPU_DISTANCE_TIMING 0

#define MAX_INDEX UINT_MAX

#define SHFL_DOWN(val, offset) __shfl_down_sync(FULL_MASK, val, offset)
#define SHFL(val, lane) __shfl_sync(FULL_MASK, val, lane)
#define SHFL_XOR(val, mask) __shfl_xor_sync(FULL_MASK, val, mask)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && (__CUDA_ARCH__ < 900)
#define GPU_MAX_WARPS_PER_SM 48
#else
#define GPU_MAX_WARPS_PER_SM 64
#endif

#define GPU_MIN_BLOCKS_PER_SM_FOR(block_size) (GPU_MAX_WARPS_PER_SM / WARPS_PER_BLOCK)
#define GPU_LAUNCH_BOUNDS(block_size) __launch_bounds__((block_size), GPU_MIN_BLOCKS_PER_SM_FOR(block_size))

typedef unsigned long long gpu_long_t; // for counters
typedef uint32_t vidType;
using vid_t = vidType;

__device__ __forceinline__ gpu_long_t gpu_globaltimer() {
  gpu_long_t ticks;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ticks));
  return ticks;
}

enum GpuDistanceTimingIndex : int {
  GPU_TIMING_L2_CALLS = 0,
  GPU_TIMING_L2_CYCLES = 1,
  GPU_TIMING_EST_CALLS = 2,
  GPU_TIMING_EST_CYCLES = 3,
  GPU_TIMING_COUNT = 4
};

enum GpuSearchStatIndex : int {
  GPU_SEARCH_STAT_DIST_CMPS = 0,
  GPU_SEARCH_STAT_DIST_EST = 1,
  GPU_SEARCH_STAT_ITERATIONS = 2,
  GPU_SEARCH_STAT_COUNT = 3
};

struct QueryFactors {
  float* rotated_query = nullptr;
  uint8_t* quantized_query = nullptr;
  uint8_t* lut = nullptr;
  float low_val = 0.0f;
  float high_val = 0.0f;
  float width = 0.0f;
  int32_t sum_q = 0;
};

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
protected:
  vidType nv;           // number of vertices
  vidType maxDeg;       // maximun degree
  vidType *d_edges;
public:
  GraphGPU(int n, int d, vidType *h_edges) : nv(n), maxDeg(d) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_edges, n * d * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_edges, h_edges, n * d * sizeof(vidType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
  inline __device__ __host__ vidType V() { return nv; }
  inline __device__ __host__ vidType get_max_degree() { return maxDeg; }
  inline __device__ __host__ vidType* N(vidType vid) { return d_edges + vid * maxDeg; }
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

/*-------------------------------------------- sort --------------------------------------------*/
typedef float DATA_T;
typedef uint32_t INDEX_T;
typedef float DISTANCE_T;

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

#define SMALL_HASH_RESET_INTERVAL 16

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
    // Open addressing is used for collision resolution
    const uint32_t size     = hashtable_getsize(bitlen);
    const uint32_t bit_mask = size - 1;

    // Linear probing
    INDEX_T index             = (key ^ (key >> bitlen)) & bit_mask;
    constexpr uint32_t stride = 1;

    for (unsigned i = 0; i < size; i++) {
        const INDEX_T old = atomicCAS(&table[index], ~static_cast<INDEX_T>(0), key);
        if (old == ~static_cast<INDEX_T>(0)) {
            return 1;
        } else if (old == key) {
            return 0;
        }
        index = (index + stride) & bit_mask;
    }
  return 0;
}

__device__ inline void hashtable_restore(   
    INDEX_T* const table, 
    const unsigned BITLEN,
    const INDEX_T* itopk_indices,
    const uint32_t itopk_size,
    const uint32_t first_tid = 0) {
  constexpr INDEX_T index_msb_1_mask = 0x80000000;
  if (threadIdx.x < first_tid) return;
  for (unsigned i = threadIdx.x - first_tid; i < itopk_size; i += blockDim.x - first_tid) {
    auto key = itopk_indices[i] & ~index_msb_1_mask;  // clear most significant bit
    hashtable_insert(table, BITLEN, key);
  }
}

// select top-N un-expanded nodes in the priority queue of size M
static __device__ uint32_t pick_expanders(int N, INDEX_T* output_list, int M, INDEX_T* pri_queue) {
  uint32_t num_exp = 0; // number of expanders actually seleceted; must be <= N
  constexpr INDEX_T index_msb_1_mask = 0x80000000;
  uint32_t itopk_max = M;
  if (itopk_max % 32) { itopk_max += 32 - (itopk_max % 32); }   // round up to be multiple of 32
  for (uint32_t j = threadIdx.x; j < itopk_max; j += 32) {
    INDEX_T index;
    int new_parent = 0;
    if (j < M) {    // within the priority queue bound
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
        //if (blockIdx.x == 0) printf("pick %d-th node %d as the %d-th parent\n", j, pri_queue[j], i);
        // set most significant bit as used node
        pri_queue[j] |= index_msb_1_mask;
      }
    }
    num_exp += __popc(ballot_mask);
    if (num_exp >= N) { break; }
  }
  return num_exp;
}

/*-------------------------------------------- rabitq --------------------------------------------*/
static __device__ inline void rotate_vector_gpu(const float* src, float* dst,
   const float* signs, int dim, int padded_dim) {

  int tid = threadIdx.x;
  
  for (size_t i = tid; i < dim; i += blockDim.x) {
    dst[i] = src[i] * signs[i];
  }
  for (int i = dim + tid; i < padded_dim; i += blockDim.x) {
    dst[i] = 0.0f;
  }
  __syncthreads();

  // Fast Walsh-Hadamard Transform (in-place, power-of-2 B, B >= 8)
  for (size_t len = 1; len < padded_dim; len <<= 1) {
    int pairs = padded_dim >> 1;
    for (int pair_idx = tid; pair_idx < pairs; pair_idx += blockDim.x) {
      int block = pair_idx / len;
      int j = pair_idx % len;
      int base = block * (len << 1);

      int idx1 = base + j;
      int idx2 = idx1 + len;

      float u = dst[idx1];
      float v = dst[idx2];
      dst[idx1] = u + v;
      dst[idx2] = u - v;
    }
    __syncthreads();
  }
}

static __device__ void query_prepare_gpu(const float* query_raw, QueryFactors& scratch,
    const float* signs_ptr, int dim, int padded_dim) {

  int tid = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;

  // rotate the query
  rotate_vector_gpu(query_raw, scratch.rotated_query, signs_ptr, dim, padded_dim);
  __syncthreads();

  __shared__ float warp_min[WARPS_PER_BLOCK];
  __shared__ float warp_max[WARPS_PER_BLOCK];
  float local_min = FLT_MAX;
  float local_max = -FLT_MAX;

  // data range: find the lowest and highest dimension in the rotated query
  for (size_t i = tid; i < padded_dim; i += blockDim.x) {
      float tmp = scratch.rotated_query[i];
      local_min = fminf(local_min, tmp);
      local_max = fmaxf(local_max, tmp);
  }

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    local_min = fminf(local_min, __shfl_down_sync(FULL_MASK, local_min, offset));
    local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
  }

  if (lane_id == 0) {
    warp_min[warp_id] = local_min;
    warp_max[warp_id] = local_max;
  }
  __syncthreads();

  if (warp_id == 0) {
    local_min = (lane_id < WARPS_PER_BLOCK) ? warp_min[lane_id] : FLT_MAX;
    local_max = (lane_id < WARPS_PER_BLOCK) ? warp_max[lane_id] : -FLT_MAX;

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      local_min = fminf(local_min, __shfl_down_sync(FULL_MASK, local_min, offset));
      local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
    }

    if (lane_id == 0) {
      scratch.low_val = local_min;
      scratch.high_val = local_max;
      scratch.width = (scratch.high_val - scratch.low_val) / ((1 << QG_BQUERY) - 1);
    }
  }
  __syncthreads();

  // quantize the query 
  float inv_width = 1.0F / scratch.width;

  __shared__ int32_t warp_sum[WARPS_PER_BLOCK];
  int32_t local_sum = 0;
  for (size_t i = tid; i < padded_dim; i += blockDim.x) {
      uint8_t cur = static_cast<uint8_t>(lroundf(((scratch.rotated_query[i] - scratch.low_val) * inv_width) + 0.5));
      scratch.quantized_query[i] = cur;
      local_sum += cur;
  }

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
  }

  if (lane_id == 0) {
    warp_sum[warp_id] = local_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    local_sum = (lane_id < WARPS_PER_BLOCK) ? warp_sum[lane_id] : 0;

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
    }

    if (lane_id == 0) {
      scratch.sum_q = local_sum;
    }
  }
  __syncthreads();
}

static __device__ inline DISTANCE_T estimated_distance_gpu(const QueryFactors& qf,
  const uint64_t* packed_code, float triple_x, float factor_dq, float factor_vq,
  float exact_dist, int padded_dim_) {

  const int tid = threadIdx.x;
  const int lane_id = tid % WARP_SIZE;

  if (triple_x == FLT_MAX) return 0.0f;

  int32_t local_sum = 0;
  for (int word = 0; word < padded_dim_ / 64; ++word) {
      uint64_t bits = packed_code[word];
      int base = word * 64;

      int d0 = base + lane_id;
      if ((bits >> lane_id) & 1ULL)
          local_sum += static_cast<int32_t>(qf.quantized_query[d0]);

      int d1 = d0 + 32;
      if (d1 < padded_dim_ && ((bits >> (lane_id + 32)) & 1ULL))
          local_sum += static_cast<int32_t>(qf.quantized_query[d1]);
  }

  // reduce the sum
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }

  float result_float = static_cast<float>((local_sum << 1) - qf.sum_q);
  DISTANCE_T est_dist = triple_x + exact_dist;
  est_dist += factor_dq * qf.width * result_float;
  est_dist += factor_vq * qf.low_val;
  return est_dist;
}
