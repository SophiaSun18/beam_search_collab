#pragma once
#include <cfloat>
#include <cassert>
#include <climits>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

#define FULL_MASK     0xffffffff
#define BLOCK_SIZE    256
#define WARP_SIZE     32
#define LOG_WARP_SIZE 5
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

   
//#define FLT_MAX 3.40282347e+38
#ifndef GPU_SEARCH_STATS
#define GPU_SEARCH_STATS 1
#endif

#ifndef GPU_DISTANCE_TIMING
#define GPU_DISTANCE_TIMING 0
#endif

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
  GPU_TIMING_EST_REGION_CANDIDATES = 4,
  GPU_TIMING_EST_REGION_CYCLES = 5,
  GPU_TIMING_QUERY_PREP_CYCLES = 6,
  GPU_TIMING_SORT_CYCLES = 7,
  GPU_TIMING_PICK_CYCLES = 8,
  GPU_TIMING_VISITED_CYCLES = 9,
  GPU_TIMING_STAGE_CYCLES = 10,
  GPU_TIMING_FINAL_SORT_CYCLES = 11,
  GPU_TIMING_RESULT_POOL_CYCLES = 12,
  GPU_TIMING_TENSOR_QUERY_TILE_CYCLES = 13,
  GPU_TIMING_TENSOR_UNPACK_CYCLES = 14,
  GPU_TIMING_TENSOR_MMA_CYCLES = 15,
  GPU_TIMING_TENSOR_FINAL_FACTOR_CYCLES = 16,
  GPU_TIMING_COUNT = 17
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


#define SMALL_HASH_RESET_INTERVAL 16

static __device__ inline void result_pool_recompute_worst(
    const DISTANCE_T* pool_distance,
    uint32_t pool_size,
    uint32_t* worst_idx,
    DISTANCE_T* worst_dist) {
  if (pool_size == 0) {
    *worst_idx = 0;
    *worst_dist = -FLT_MAX;
    return;
  }
  uint32_t local_worst_idx = 0;
  DISTANCE_T local_worst_dist = pool_distance[0];
  for (uint32_t i = 1; i < pool_size; ++i) {
    if (pool_distance[i] > local_worst_dist) {
      local_worst_dist = pool_distance[i];
      local_worst_idx = i;
    }
  }
  *worst_idx = local_worst_idx;
  *worst_dist = local_worst_dist;
}

static __device__ inline void result_pool_push_unique_unsorted(
    INDEX_T* pool_index,
    DISTANCE_T* pool_distance,
    uint32_t* pool_size,
    uint32_t* worst_idx,
    DISTANCE_T* worst_dist,
    int capacity,
    INDEX_T node,
    DISTANCE_T distance) {
  uint32_t size = *pool_size;

  if (size < static_cast<uint32_t>(capacity)) {
    pool_index[size] = node;
    pool_distance[size] = distance;
    *pool_size = size + 1;
    if (size == 0 || distance > *worst_dist) {
      *worst_idx = size;
      *worst_dist = distance;
    }
    return;
  }

  if (distance >= *worst_dist) return;

  pool_index[*worst_idx] = node;
  pool_distance[*worst_idx] = distance;
  result_pool_recompute_worst(pool_distance, size, worst_idx, worst_dist);
}

static __device__ inline void result_pool_sort_small(
    INDEX_T* pool_index,
    DISTANCE_T* pool_distance,
    uint32_t pool_size) {
  for (uint32_t i = 1; i < pool_size; ++i) {
    INDEX_T key_index = pool_index[i];
    DISTANCE_T key_dist = pool_distance[i];
    int j = static_cast<int>(i) - 1;
    while (j >= 0 && pool_distance[j] > key_dist) {
      pool_distance[j + 1] = pool_distance[j];
      pool_index[j + 1] = pool_index[j];
      --j;
    }
    pool_distance[j + 1] = key_dist;
    pool_index[j + 1] = key_index;
  }
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


static __device__ __forceinline__ int fastscan_lane_pack_index(int lane_mod16) {
  return ((lane_mod16 & 7) << 1) | (lane_mod16 >> 3);
}

static __device__ __forceinline__ uint8_t fastscan_decode_code_gpu(
    const uint8_t* packed_codes_block, int lane_id, int codebook_idx) {
  const int pair_idx = codebook_idx >> 1;
  const int nibble_group = codebook_idx & 1;
  const int lane_mod16 = lane_id & 15;
  const int byte_idx = fastscan_lane_pack_index(lane_mod16);
  const uint8_t packed =
      packed_codes_block[pair_idx * 32 + nibble_group * 16 + byte_idx];
  return (lane_id < 16) ? static_cast<uint8_t>(packed & 0x0f)
                        : static_cast<uint8_t>(packed >> 4);
}

static __device__ inline void pack_lut_gpu(
    int dim, const uint8_t* __restrict__ quantized_query, uint8_t* __restrict__ lut) {
  const int tid = threadIdx.x;
  const int num_codebook = dim >> 2;

  for (int cb = tid; cb < num_codebook; cb += blockDim.x) {
    const uint8_t* query_chunk = quantized_query + (cb << 2);
    uint8_t* lut_chunk = lut + (cb << 4);

    lut_chunk[0] = 0;
    for (int mask = 1; mask < 16; ++mask) {
      const int lowbit = mask & -mask;
      const int pos = (lowbit == 8) ? 0 : ((lowbit == 4) ? 1 : ((lowbit == 2) ? 2 : 3));
      lut_chunk[mask] = static_cast<uint8_t>(lut_chunk[mask - lowbit] + query_chunk[pos]);
    }
  }
}

static __device__ inline void scan_all_neighbors_32_gpu(
    const QueryFactors& qf, const uint8_t* packed_codes_block,
    const float* triple_x, const float* factor_dq, const float* factor_vq,
    float exact_dist, int padded_dim, float* appro_dist, int active_neighbors = WARP_SIZE) {
  const int lane_id = threadIdx.x & (WARP_SIZE - 1);
  if (lane_id >= active_neighbors) return;

  const int num_codebook = padded_dim >> 2;
  uint32_t raw_sum = 0;

  for (int cb = 0; cb < num_codebook; ++cb) {
    const uint8_t code = fastscan_decode_code_gpu(packed_codes_block, lane_id, cb);
    raw_sum += static_cast<uint32_t>(qf.lut[(cb << 4) + code]);
  }

  const float result_float = static_cast<float>((static_cast<int32_t>(raw_sum) << 1) - qf.sum_q);
  float est_dist = triple_x[lane_id] + exact_dist;
  est_dist += factor_dq[lane_id] * qf.width * result_float;
  est_dist += factor_vq[lane_id] * qf.low_val;
  appro_dist[lane_id] = est_dist;
}