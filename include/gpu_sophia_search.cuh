#pragma once
#include "gpu_graph.cuh"      // GPUGraphView, vid_t, Span — graph interface only
#include <cuda_runtime.h>
#include <cfloat>
#include <climits>
#include <cstdint>

/**
 * Sophia's beam search kernel, vendored from the sophia branch (628b125).
 * Only the graph interface (GPUGraphView) comes from our codebase.
 *
 * Each section is copied verbatim from its source file, with minimal
 * replacements annotated inline.
 */

// ── config ─────────────────────────────────────────────────────────────────

struct SophiaSearchCfg {
    unsigned topk_size;       // unused by kernel (hardcoded to 128), kept for CLI compat
    unsigned bitlen;
    vid_t    entry_point;
    int      max_iterations;
    int      reset_interval;
};

// ── vendored types/macros from utils.cuh ───────────────────────────────────
// replaced: #include "utils.cuh" with inline definitions

namespace sophia {

// --- verbatim from (include/utils.cuh 628b125) lines 9-18, 20-28, 30-32, 43-51 ---
#define SOPHIA_FULL_MASK     0xffffffff
#define SOPHIA_BLOCK_SIZE    256
#define SOPHIA_WARP_SIZE     32
#define SOPHIA_WARPS_PER_BLOCK (SOPHIA_BLOCK_SIZE / SOPHIA_WARP_SIZE)

// replaced: GPU_HASH_BITLEN_FOR_TOPK macro inlined
#define SOPHIA_GPU_HASH_BASE_BITLEN 10
#define SOPHIA_GPU_HASH_BITLEN_FOR_TOPK(topk) \
  (((topk) * 4 > 3 * (1 << SOPHIA_GPU_HASH_BASE_BITLEN)) ? (SOPHIA_GPU_HASH_BASE_BITLEN + 1) : SOPHIA_GPU_HASH_BASE_BITLEN)

// replaced: GPU_SEARCH_STATS -> 0 (disabled), GPU_DISTANCE_TIMING -> 0 (disabled)
#define SOPHIA_GPU_SEARCH_STATS 0
#define SOPHIA_GPU_DISTANCE_TIMING 0
#define SOPHIA_GPU_SEARCH_STAT_COUNT 3
#define SOPHIA_GPU_TIMING_COUNT 17

typedef unsigned long long sophia_gpu_long_t;
typedef float    SOPHIA_DATA_T;
typedef uint32_t SOPHIA_INDEX_T;
typedef float    SOPHIA_DISTANCE_T;

#define SOPHIA_MAX_INDEX UINT_MAX

#define SOPHIA_SHFL_DOWN(val, offset) __shfl_down_sync(SOPHIA_FULL_MASK, val, offset)
#define SOPHIA_SHFL(val, lane) __shfl_sync(SOPHIA_FULL_MASK, val, lane)
#define SOPHIA_SHFL_XOR(val, mask) __shfl_xor_sync(SOPHIA_FULL_MASK, val, mask)
// --- end ---

// --- verbatim from (include/utils.cuh 628b125) lines 139-151 ---
template <typename T = float>
__device__ __forceinline__ T warp_l2_distance(int dim, const T* a, const T* b) {
  int thread_lane = threadIdx.x & (SOPHIA_WARP_SIZE-1); // thread index within the warp
  T val = 0.;
  for(int i = thread_lane; i < dim; i += SOPHIA_WARP_SIZE)
    val += (a[i] - b[i]) * (a[i] - b[i]);
  T sum = val;
  sum += SOPHIA_SHFL_DOWN(sum, 16);
  sum += SOPHIA_SHFL_DOWN(sum, 8);
  sum += SOPHIA_SHFL_DOWN(sum, 4);
  sum += SOPHIA_SHFL_DOWN(sum, 2);
  sum += SOPHIA_SHFL_DOWN(sum, 1);
  sum  = SOPHIA_SHFL(sum, 0);
  return sum;
}
// --- end ---

// --- verbatim from (include/utils.cuh 628b125) lines 176-267 ---
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

// replaced: `if (_N == 2)` with `if constexpr (_N == 2)` (original used runtime if)
template <class K, class V, unsigned _N, unsigned warp_size>
__device__ inline void warp_merge_core(K k[2], V v[2], const std::uint32_t range, const bool asc) {
  if constexpr (_N == 2) {
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
  } else if constexpr (_N == 4) {
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
// --- end ---

// --- verbatim from (include/utils.cuh 628b125) lines 354-451 (topk_128_by_bitonic_sort) ---
// renamed: topk_128_by_bitonic_sort -> topk_by_bitonic_sort
// replaced: MAX_INDEX -> SOPHIA_MAX_INDEX
template <unsigned N_1, unsigned N_2>
__device__ void topk_by_bitonic_sort(
    SOPHIA_INDEX_T* result_indices_ptr,
    SOPHIA_DISTANCE_T* result_distances_ptr,
    uint32_t RESULT_BUFFER_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk,
    bool first) {
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;

  // Sort part 1
  if (warp_id > 0) return;
  // assert (CANDIDATE_BUFFER_SIZE <= 64);

  SOPHIA_DISTANCE_T key_1[N_1];
  SOPHIA_INDEX_T val_1[N_1];
  auto candidate_distances = result_distances_ptr + internal_topk;
  auto candidate_indices = result_indices_ptr + internal_topk;
  /* Candidates -> Reg */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = lane_id + (32 * i);
    if (j < CANDIDATE_BUFFER_SIZE) {
      key_1[i] = candidate_distances[j];
      val_1[i] = candidate_indices[j];
    } else {
      key_1[i] = FLT_MAX;
      val_1[i] = SOPHIA_MAX_INDEX;
    }
  }
  /* Sort */
  warp_sort<float, uint32_t, N_1>(key_1, val_1);
  /* Reg -> Temp_itopk */
  for (unsigned i = 0; i < N_1; i++) {
    unsigned j = (N_1 * lane_id) + i;
    if (j < CANDIDATE_BUFFER_SIZE && j < internal_topk) {
      candidate_distances[j] = key_1[i];
      candidate_indices[j]   = val_1[i];
    }
  }

  // Sort part 2
  SOPHIA_DISTANCE_T key_2[N_2];
  SOPHIA_INDEX_T val_2[N_2];
  if (first) {
    /* Load itopk results */
    for (unsigned i = 0; i < N_2; i++) {
      unsigned j = lane_id + (32 * i);
      if (j < internal_topk) {
        key_2[i] = result_distances_ptr[j];
        val_2[i] = result_indices_ptr[j];
      } else {
        key_2[i] = FLT_MAX;
        val_2[i] = SOPHIA_MAX_INDEX;
      }
    }
    /* Warp Sort */
    warp_sort<float, uint32_t, N_2>(key_2, val_2);
  } else {
    /* Load itopk results */
    for (unsigned i = 0; i < N_2; i++) {
      unsigned j = (N_2 * lane_id) + i;
      if (j < internal_topk) {
        key_2[i] = result_distances_ptr[j];
        val_2[i] = result_indices_ptr[j];
      } else {
        key_2[i] = FLT_MAX;
        val_2[i] = SOPHIA_MAX_INDEX;
      }
    }
  }

  /* Merge candidates */
  for (unsigned i = 0; i < N_2; i++) {
    unsigned j = (N_2 * lane_id) + i;  // [0:MAX_ITOPK-1]
    unsigned k = internal_topk - 1 - j;
    if (k >= internal_topk || k >= CANDIDATE_BUFFER_SIZE) continue;
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
    if (j < internal_topk) {
      result_distances_ptr[j] = key_2[i];
      result_indices_ptr[j]   = val_2[i];
    }
  }

  (void)RESULT_BUFFER_SIZE;
  (void)first;
}
// --- end ---

// --- verbatim from (include/utils.cuh 628b125) lines 561-578 ---
// renamed: topk_by_sort_candidates_and_merge (dispatch wrapper)
template <unsigned N_1, unsigned N_2>
__device__ void topk_by_sort_candidates_and_merge(
    SOPHIA_INDEX_T* result_indices_ptr,
    SOPHIA_DISTANCE_T* result_distances_ptr,
    uint32_t RESULT_BUFFER_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk,
    bool first) {
  // original dispatches to merge-path for N_2>4; we only use N_2=4 here
  topk_by_bitonic_sort<N_1, N_2>(
      result_indices_ptr, result_distances_ptr, RESULT_BUFFER_SIZE,
      CANDIDATE_BUFFER_SIZE, internal_topk, first);
}
// --- end ---

// --- verbatim from (include/utils.cuh 628b125) lines 582-612 ---
// replaced: INDEX_T -> SOPHIA_INDEX_T
__host__ __device__ inline uint32_t hashtable_getsize(const uint32_t bitlen) {
  return 1 << bitlen;
}

__device__ inline void hashtable_init(SOPHIA_INDEX_T* const table, const unsigned bitlen, unsigned FIRST_TID = 0) {
  if (threadIdx.x < FIRST_TID) return;
  for (uint32_t i = threadIdx.x - FIRST_TID; i < hashtable_getsize(bitlen); i += blockDim.x - FIRST_TID) {
    table[i] = SOPHIA_MAX_INDEX;
  }
}

__device__ inline uint32_t hashtable_insert(SOPHIA_INDEX_T* const table, const unsigned bitlen, const SOPHIA_INDEX_T key) {
    // Open addressing is used for collision resolution
    const uint32_t size     = hashtable_getsize(bitlen);
    const uint32_t bit_mask = size - 1;

    // Linear probing
    SOPHIA_INDEX_T index             = (key ^ (key >> bitlen)) & bit_mask;
    constexpr uint32_t stride = 1;

    for (unsigned i = 0; i < size; i++) {
        const SOPHIA_INDEX_T old = atomicCAS(&table[index], ~static_cast<SOPHIA_INDEX_T>(0), key);
        if (old == ~static_cast<SOPHIA_INDEX_T>(0)) {
            return 1;
        } else if (old == key) {
            return 0;
        }
        index = (index + stride) & bit_mask;
    }
  return 0;
}
// --- end ---

// --- verbatim from (include/utils.cuh 628b125) lines 631-643 ---
__device__ inline void hashtable_restore(
    SOPHIA_INDEX_T* const table,
    const unsigned BITLEN,
    const SOPHIA_INDEX_T* itopk_indices,
    const uint32_t itopk_size,
    const uint32_t first_tid = 0) {
  constexpr SOPHIA_INDEX_T index_msb_1_mask = 0x80000000;
  if (threadIdx.x < first_tid) return;
  for (unsigned i = threadIdx.x - first_tid; i < itopk_size; i += blockDim.x - first_tid) {
    auto key = itopk_indices[i] & ~index_msb_1_mask;  // clear most significant bit
    hashtable_insert(table, BITLEN, key);
  }
}
// --- end ---

// --- verbatim from (include/utils.cuh 628b125) lines 663-691 ---
static __device__ uint32_t pick_expanders(int N, SOPHIA_INDEX_T* output_list, int M, SOPHIA_INDEX_T* pri_queue) {
  uint32_t num_exp = 0; // number of expanders actually seleceted; must be <= N
  constexpr SOPHIA_INDEX_T index_msb_1_mask = 0x80000000;
  uint32_t itopk_max = M;
  if (itopk_max % 32) { itopk_max += 32 - (itopk_max % 32); }   // round up to be multiple of 32
  for (uint32_t j = threadIdx.x; j < itopk_max; j += 32) {
    SOPHIA_INDEX_T index;
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
        // set most significant bit as used node
        pri_queue[j] |= index_msb_1_mask;
      }
    }
    num_exp += __popc(ballot_mask);
    if (num_exp >= N) { break; }
  }
  return num_exp;
}
// --- end ---

}  // namespace sophia

// ═══════════════════════════════════════════════════════════════════════════
// Search kernel
// ═══════════════════════════════════════════════════════════════════════════

// --- verbatim from (include/search_bitonic.cuh 628b125) lines 5-17 ---
#ifndef SOPHIA_INTERNAL_TOPK
#define SOPHIA_INTERNAL_TOPK   128
#endif
#define SOPHIA_SEARCH_WIDTH    1
#define SOPHIA_BITLEN          SOPHIA_GPU_HASH_BITLEN_FOR_TOPK(SOPHIA_INTERNAL_TOPK)
#define SOPHIA_CANDIDATE_BUFFER_SIZE 64
#define SOPHIA_RESULT_BUFFER_SIZE (SOPHIA_INTERNAL_TOPK + SOPHIA_CANDIDATE_BUFFER_SIZE)
#define SOPHIA_SMALL_HASH_RESET_INTERVAL 16
// --- end ---

// replaced: calculate_shared_mem_size with our version using vendored constants
__host__ __device__ inline size_t sophia_shared_mem_bytes(unsigned /*bitlen*/, unsigned dim) {
    size_t size = 0;
    size += sophia::hashtable_getsize(SOPHIA_BITLEN) * sizeof(sophia::SOPHIA_INDEX_T);
    size += SOPHIA_RESULT_BUFFER_SIZE * sizeof(sophia::SOPHIA_INDEX_T);
    size += SOPHIA_RESULT_BUFFER_SIZE * sizeof(sophia::SOPHIA_DISTANCE_T);
    size += SOPHIA_SEARCH_WIDTH * sizeof(sophia::SOPHIA_INDEX_T);
    size += dim * sizeof(float);
    size += 2 * sizeof(uint32_t);  // counter + hops
    return size;
}

// --- verbatim from (include/search_bitonic.cuh 628b125) lines 45-279 ---
// replaced: template <typename T> with float (we only use float)
// replaced: kernel signature parameters:
//   - GraphGPU<vid_t> gg -> GPUGraphView graph
//   - const T* d_data -> (accessed via graph.vect())
//   - vid_t* d_starting_points -> cfg.entry_point (single entry)
//   - removed: d_total_stats, d_total_timing, d_seed_map, d_sign_bit, debug_mode, npoints
// replaced: gg.get_max_degree() -> graph.degree()
//           gg.N(current_node) -> graph.nbrs(current_node)
//           d_data + node * dim -> graph.vect(node).data()
//           warp_l2_distance -> sophia::warp_l2_distance
// replaced: all bare names with sophia:: namespace prefix
//           (hashtable_init, hashtable_insert, hashtable_restore,
//            topk_by_sort_candidates_and_merge, pick_expanders)
// replaced: BLOCK_SIZE -> SOPHIA_BLOCK_SIZE, WARP_SIZE -> SOPHIA_WARP_SIZE,
//           WARPS_PER_BLOCK -> SOPHIA_WARPS_PER_BLOCK, etc.
// replaced: GPU_LAUNCH_BOUNDS(BLOCK_SIZE) -> (removed, not needed)
// replaced: MAX_INDEX -> SOPHIA_MAX_INDEX
// replaced: INDEX_T/DISTANCE_T/DATA_T -> SOPHIA_INDEX_T/SOPHIA_DISTANCE_T/SOPHIA_DATA_T
// replaced: GPU_SEARCH_STATS -> SOPHIA_GPU_SEARCH_STATS (= 0)
// replaced: GPU_DISTANCE_TIMING -> SOPHIA_GPU_DISTANCE_TIMING (= 0)
// replaced: MAX_ITERATIONS -> cfg.max_iterations
// replaced: SMALL_HASH_RESET_INTERVAL -> cfg.reset_interval
// replaced: seeding loop (lines 118-138): original iterates CANDIDATE_BUFFER_SIZE starting
//           points from d_starting_points with multi-warp L2. Replaced with single entry_point
//           seed to match our convention (d_starting_points is not available).
// removed:  GPU_LAUNCH_BOUNDS attribute (line 46)
// removed:  d_graph_ptr (line 65, unused in bitonic path)
// removed:  debug_mode printing (lines 159-178, 186-198)
// removed:  stats writeback (lines 267-278)

__global__ void kernel_sophia_search(
    GPUGraphView graph, SophiaSearchCfg cfg,
    int nq, int dim,
    const float* __restrict__ d_queries,
    vid_t* __restrict__ d_results, int K)
{
    // basic information about this block and thread
    const int query_id = blockIdx.x;
    if (query_id > nq) return;
    const float* query = d_queries + query_id * dim;
    const uint32_t maxDegree = graph.degree();

    const int tid = threadIdx.x;
    const int warp_id = tid / SOPHIA_WARP_SIZE;
    const int lane_id = tid % SOPHIA_WARP_SIZE;

    // initialize the shared memory
    extern __shared__ char shared_memory[];
    sophia::SOPHIA_INDEX_T* ALL_INDEX = (sophia::SOPHIA_INDEX_T*)shared_memory;
    sophia::SOPHIA_INDEX_T* TOP_K_INDEX      = ALL_INDEX;
    sophia::SOPHIA_INDEX_T* CANDIDATE_INDEX  = ALL_INDEX + SOPHIA_INTERNAL_TOPK;

    sophia::SOPHIA_DISTANCE_T* ALL_DISTANCE = (sophia::SOPHIA_DISTANCE_T*)(ALL_INDEX + SOPHIA_RESULT_BUFFER_SIZE);
    sophia::SOPHIA_DISTANCE_T* TOP_K_DISTANCE      = ALL_DISTANCE;
    sophia::SOPHIA_DISTANCE_T* CANDIDATE_DISTANCE  = ALL_DISTANCE + SOPHIA_INTERNAL_TOPK;

    sophia::SOPHIA_INDEX_T* HASH_TABLE = (sophia::SOPHIA_INDEX_T*)(ALL_DISTANCE + SOPHIA_RESULT_BUFFER_SIZE);
    sophia::SOPHIA_INDEX_T* PARENT_LIST = (sophia::SOPHIA_INDEX_T*)(HASH_TABLE + sophia::hashtable_getsize(SOPHIA_BITLEN));
    sophia::SOPHIA_DATA_T* QUERY_BUFFER = (sophia::SOPHIA_DATA_T*)(PARENT_LIST + SOPHIA_SEARCH_WIDTH);
    uint32_t* counter = (uint32_t*)(QUERY_BUFFER + dim);
    uint32_t* hops = (uint32_t*)(counter + 1);

    // initialize the block-wise total distance computation count
    __shared__ uint32_t keep_expanding;
    if (tid == 0) {
        counter[0] = 0;
        hops[0] = 0;
    }

    // initialize the hash table to record visited (computed) nodes
    sophia::hashtable_init(HASH_TABLE, SOPHIA_BITLEN);

    // load query into shared memory
    for (int i = tid; i < dim; i += blockDim.x) {
        if (i < dim) QUERY_BUFFER[i] = query[i];
    }

    // load the top k list with dummy initial values
    for (int i = tid; i < SOPHIA_INTERNAL_TOPK; i += blockDim.x) {
        TOP_K_INDEX[i] = SOPHIA_MAX_INDEX;
        TOP_K_DISTANCE[i] = FLT_MAX;
    }

    // replaced: original seeding loop (lines 118-138) loaded CANDIDATE_BUFFER_SIZE
    // starting points from d_starting_points with multi-warp L2.
    // We seed only the single entry_point since d_starting_points is not available.
    for (int i = tid; i < SOPHIA_CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
        CANDIDATE_INDEX[i] = SOPHIA_MAX_INDEX;
        CANDIDATE_DISTANCE[i] = FLT_MAX;
    }
    __syncthreads();
    {
        sophia::SOPHIA_DISTANCE_T dist = FLT_MAX;
        if (cfg.entry_point != SOPHIA_MAX_INDEX) {
            dist = sophia::warp_l2_distance(dim, query, graph.vect(cfg.entry_point).data());
        }
        if (lane_id == 0 && warp_id == 0) {
            if (cfg.entry_point != SOPHIA_MAX_INDEX) {
                atomicAdd(&counter[0], 1);
                sophia::hashtable_insert(HASH_TABLE, SOPHIA_BITLEN, cfg.entry_point);
            }
            CANDIDATE_INDEX[0] = cfg.entry_point;
            CANDIDATE_DISTANCE[0] = dist;
        }
    }
    __syncthreads();

    // loop end condition: either entire topK expanded, or reach max iteration limit
    for (int iter = 0; iter < cfg.max_iterations; iter++) {

        // forgettable hash table management according to the CARGA paper
        if ((iter + 1) % cfg.reset_interval == 0) {
            sophia::hashtable_init(HASH_TABLE, SOPHIA_BITLEN);
            __syncthreads();
            sophia::hashtable_restore(HASH_TABLE, SOPHIA_BITLEN, TOP_K_INDEX, SOPHIA_INTERNAL_TOPK);
        }
        __syncthreads();

        // use warp level sort to find the actual closest top K elements and update the topK list
        sophia::topk_by_sort_candidates_and_merge<2, SOPHIA_INTERNAL_TOPK / SOPHIA_WARP_SIZE>(ALL_INDEX, ALL_DISTANCE, SOPHIA_RESULT_BUFFER_SIZE,
            SOPHIA_CANDIDATE_BUFFER_SIZE, SOPHIA_INTERNAL_TOPK, (iter == 0));
        __syncthreads();

        // warp 0 pick the top-N unexpanded nodes and store in shared memory
        // this already marks the expanded nodes MSB to 1, no need to mark again
        if (warp_id == 0) {
            keep_expanding = sophia::pick_expanders(SOPHIA_SEARCH_WIDTH, PARENT_LIST, SOPHIA_INTERNAL_TOPK, TOP_K_INDEX);
        }
        __syncthreads();
        if (!keep_expanding) {
            if (tid == 0) {
                hops[0] = iter;
            }
            break;
        }

        // collect the neighbors of the expanding node and add into candidate list if not visited before
        // make sure to mark the invalid neighbor using dummy large values
        sophia::SOPHIA_INDEX_T current_node = TOP_K_INDEX[PARENT_LIST[0]] & 0x7fffffff;
        // replaced: vid_t* neighbors = gg.N(current_node);
        Span<const vid_t> neighbors = graph.nbrs(current_node);
        // replaced: warp_l2_distance(dim, query, d_data + current_node * dim)
        sophia::SOPHIA_DISTANCE_T exact_dist = sophia::warp_l2_distance(dim, query, graph.vect(current_node).data());
        (void)exact_dist;
        __syncthreads();

        for (int i = tid; i < SOPHIA_CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
            // replaced: (i < maxDegree) ? neighbors[i] : MAX_INDEX
            sophia::SOPHIA_INDEX_T child_id = (i < (int)maxDegree) ? neighbors[i] : SOPHIA_MAX_INDEX;

            if (child_id != SOPHIA_MAX_INDEX) {    // check if visited already
                if (sophia::hashtable_insert(HASH_TABLE, SOPHIA_BITLEN, child_id) == 0) {
                    child_id = SOPHIA_MAX_INDEX;
                    CANDIDATE_DISTANCE[i] = FLT_MAX;
                } else {
                    CANDIDATE_DISTANCE[i] = 0;
                }
            }
            CANDIDATE_INDEX[i] = child_id;
        }
        __syncthreads();

        // all warps work together to compute candidate distance
        for (int i = warp_id; i < SOPHIA_CANDIDATE_BUFFER_SIZE; i += SOPHIA_WARPS_PER_BLOCK) {
            // visited candidates are marked as invalid
            if (CANDIDATE_INDEX[i] == SOPHIA_MAX_INDEX) continue;

            // if current node is new in hash table, compute and store its distance
            // replaced: warp_l2_distance(dim, query, d_data + CANDIDATE_INDEX[i] * dim)
            sophia::SOPHIA_DISTANCE_T dist = sophia::warp_l2_distance(dim, query, graph.vect(CANDIDATE_INDEX[i]).data());
            if (lane_id == 0) {
                atomicAdd(&counter[0], 1);
                CANDIDATE_DISTANCE[i] = dist;
            }
        }
        __syncthreads();
    }

    // final sort before writeback
    sophia::topk_by_sort_candidates_and_merge<2, SOPHIA_INTERNAL_TOPK / SOPHIA_WARP_SIZE>(ALL_INDEX, ALL_DISTANCE, SOPHIA_RESULT_BUFFER_SIZE,
        SOPHIA_CANDIDATE_BUFFER_SIZE, SOPHIA_INTERNAL_TOPK, false);
    __syncthreads();

    // return the top K result
    for (int i = tid; i < K; i += blockDim.x) {
        d_results[query_id * K + i] = TOP_K_INDEX[i] & ~0x80000000;
    }

    (void)hops;
}
// --- end (include/search_bitonic.cuh 628b125) ---
