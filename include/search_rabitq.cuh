#pragma once
#include "utils.cuh"
#include "distance.cuh"
#include <cuda_runtime.h>
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
#include <cuda_pipeline_primitives.h>
#endif
#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK   128
#endif
#define SEARCH_WIDTH    1     // MAX: 2 nodes * 32 degree = 64 neighbors = CANDIDATE_BUFFER_SIZE
// #define BITLEN          GPU_HASH_BITLEN_FOR_TOPK(INTERNAL_TOPK)
#define MAX_ITERATIONS (1 << 15)
// #define TEAM_SIZE       8
// #define HASH_TABLE_CONFIG   1
#define CANDIDATE_BUFFER_SIZE 64
// #define RESULT_BUFFER_SIZE (INTERNAL_TOPK + CANDIDATE_BUFFER_SIZE)

#ifndef GPU_RABITQ_FASTSCAN_SUBWARP_LANES
#define GPU_RABITQ_FASTSCAN_SUBWARP_LANES 32
#endif

#if !GPU_RABITQ_USE_LUT && GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 32
#error "GPU_RABITQ_USE_LUT=0 currently supports only GPU_RABITQ_FASTSCAN_SUBWARP_LANES=32"
#endif

#if GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 2 && GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 4 && \
    GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 8 && GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 16 && \
    GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 32
#error "GPU_RABITQ_FASTSCAN_SUBWARP_LANES must be one of 2, 4, 8, 16, or 32"
#endif

template <typename T>
static __device__ inline void staged_copy_async_begin_or_sync(
    T* __restrict__ dst,
    const T* __restrict__ src,
    int count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        __pipeline_memcpy_async(dst + i, src + i, sizeof(T));
    }
    __pipeline_commit();
#else
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        dst[i] = src[i];
    }
#endif
}

static __device__ inline void staged_copy_async_finish_or_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    __pipeline_wait_prior(0);
#endif
}

static __host__ inline uint32_t calculate_shared_mem_size(int dim, int beam_sz, int result_k,
                                                           int max_deg, int bitlen) {
    auto align_up = [](size_t value, size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    };
    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    size_t code_words = padded_dim / 64;
    const size_t result_buffer_size = static_cast<size_t>(beam_sz) + CANDIDATE_BUFFER_SIZE;
    size_t size = 0;
    size += hashtable_getsize(bitlen) * sizeof(INDEX_T);    // hash table
    size += result_buffer_size * sizeof(INDEX_T);           // top k and candidate index
    size += result_buffer_size * sizeof(DISTANCE_T);        // top k and candidate distance
    size += SEARCH_WIDTH * sizeof(INDEX_T);                 // next expanders / parent buffer
    size += dim * 4;                                        // query buffer size
    size += padded_dim * sizeof(float);                     // rotated query
    size += padded_dim * sizeof(uint8_t);                   // quantized query
#if GPU_RABITQ_USE_LUT
    size += (padded_dim << 2) * sizeof(uint8_t);            // LUT buffer
#endif
    size += 3 * sizeof(float);                              // low/high/width
    size += sizeof(int32_t);                                // sum_q
    size = align_up(size, alignof(uint64_t));               // align staged candidate codes
    size += max_deg * code_words * sizeof(uint64_t);        // staged candidate codes
    size += 3 * max_deg * sizeof(float);                    // staged candidate factors
    size += sizeof(uint32_t);                               // hops
    if (GPU_SEARCH_STATS) {
        size += GPU_SEARCH_STAT_COUNT * sizeof(uint32_t);   // search stats
    }
    size += beam_sz * sizeof(INDEX_T);                      // merged topk scratch index
    size += beam_sz * sizeof(DISTANCE_T);                   // merged topk scratch distance
    size += result_k * sizeof(INDEX_T);                     // result pool index
    size += result_k * sizeof(DISTANCE_T);                  // result pool distance
    size += 2 * sizeof(uint32_t);                           // result pool size and worst idx
    size += sizeof(DISTANCE_T);                             // result pool worst dist
    printf("shared memory size: %ld\n", size);
    return size;
}

static __global__ GPU_LAUNCH_BOUNDS(BLOCK_SIZE)
void QuantizedBeamSearch(
    int K, int nq, int dim, int beam_sz, int bitlen, int max_degree,
    const float* __restrict__ d_queries,
    const float* __restrict__ d_qg_data,
    const float* __restrict__ d_qg_signs,
    vidType* __restrict__ d_results,
    gpu_long_t* d_block_stats,
    vidType entry_point,
    size_t row_offset,
    size_t neighbor_offset,
    size_t code_offset,
    size_t factor_offset,
    bool use_ip
) {
    // basic information about this block and thread
    const int query_id = blockIdx.x;
    if (query_id >= nq) return;
    const float* query = d_queries + query_id * dim;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    uint32_t local_exact_count = 0;
    uint32_t local_dist_est_count = 0;
    
    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    size_t code_words = padded_dim / 64;

    // initialize the shared memory
    extern __shared__ char shared_memory[];
    INDEX_T* ALL_INDEX = (INDEX_T*)shared_memory;
    INDEX_T* TOP_K_INDEX      = ALL_INDEX;
    INDEX_T* CANDIDATE_INDEX  = ALL_INDEX + beam_sz;

    const uint32_t result_buffer_size = static_cast<uint32_t>(beam_sz + CANDIDATE_BUFFER_SIZE);
    DISTANCE_T* ALL_DISTANCE = (DISTANCE_T*)(ALL_INDEX + result_buffer_size);
    DISTANCE_T* TOP_K_DISTANCE      = ALL_DISTANCE;
    DISTANCE_T* CANDIDATE_DISTANCE  = ALL_DISTANCE + beam_sz;

    INDEX_T* HASH_TABLE = (INDEX_T*)(ALL_DISTANCE + result_buffer_size);
    INDEX_T* PARENT_LIST = (INDEX_T*)(HASH_TABLE + hashtable_getsize(bitlen));
    DATA_T* QUERY_BUFFER = (DATA_T*)(PARENT_LIST + SEARCH_WIDTH);

    float* ROTATED_QUERY_BUFFER = reinterpret_cast<float*>(QUERY_BUFFER + dim);
    uint8_t* QUANTIZED_QUERY_BUFFER = reinterpret_cast<uint8_t*>(ROTATED_QUERY_BUFFER + padded_dim);
    uint8_t* LUT_BUFFER = QUANTIZED_QUERY_BUFFER + padded_dim;
    char* query_factor_base = reinterpret_cast<char*>(LUT_BUFFER);
#if GPU_RABITQ_USE_LUT
    query_factor_base += (padded_dim << 2) * sizeof(uint8_t);
#endif
    float* low_val = reinterpret_cast<float*>(query_factor_base);
    float* high_val = low_val + 1;
    float* width = high_val + 1;
    int32_t* sum_q = reinterpret_cast<int32_t*>(width + 1);
    char* staged_base = reinterpret_cast<char*>(sum_q + 1);
    staged_base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(staged_base) + alignof(uint64_t) - 1) & ~(static_cast<uintptr_t>(alignof(uint64_t)) - 1));
    uint64_t* STAGED_CODE_BUFFER = reinterpret_cast<uint64_t*>(staged_base);
    float* STAGED_FACTOR_BUFFER = reinterpret_cast<float*>(STAGED_CODE_BUFFER + max_degree * code_words);

    uint32_t* hops = reinterpret_cast<uint32_t*>(STAGED_FACTOR_BUFFER + 3 * max_degree);
    uint32_t* stats = hops + 1;
    INDEX_T* MERGED_TOPK_INDEX = reinterpret_cast<INDEX_T*>(stats + (GPU_SEARCH_STATS ? GPU_SEARCH_STAT_COUNT : 0));
    DISTANCE_T* MERGED_TOPK_DISTANCE = reinterpret_cast<DISTANCE_T*>(MERGED_TOPK_INDEX + beam_sz);
    INDEX_T* RESULT_POOL_INDEX = reinterpret_cast<INDEX_T*>(MERGED_TOPK_DISTANCE + beam_sz);
    DISTANCE_T* RESULT_POOL_DISTANCE = reinterpret_cast<DISTANCE_T*>(RESULT_POOL_INDEX + K);
    uint32_t* result_pool_size = reinterpret_cast<uint32_t*>(RESULT_POOL_DISTANCE + K);
    uint32_t* result_pool_worst_idx = result_pool_size + 1;
    DISTANCE_T* result_pool_worst_dist = reinterpret_cast<DISTANCE_T*>(result_pool_worst_idx + 1);

    __shared__ QueryFactors qf;
    __shared__ INDEX_T fill_candidate_nodes[WARPS_PER_BLOCK];
    __shared__ DISTANCE_T fill_candidate_dists[WARPS_PER_BLOCK];
    __shared__ INDEX_T fill_parent_node_shared;
    __shared__ uint32_t warp_stat_counts[2 * WARPS_PER_BLOCK];

    // initialize the block-wise total distance computation count
    __shared__ uint32_t keep_expanding;
    __shared__ uint32_t current_expanded;
    if (tid == 0) {
        qf.rotated_query = ROTATED_QUERY_BUFFER;
        qf.quantized_query = QUANTIZED_QUERY_BUFFER;
#if GPU_RABITQ_USE_LUT
        qf.lut = LUT_BUFFER;
#else
        qf.lut = nullptr;
#endif
        qf.low_val = FLT_MAX;
        qf.high_val = -FLT_MAX;
        qf.width = 0.0f;
        qf.sum_q = 0;
        hops[0] = 0;
        if (GPU_SEARCH_STATS) {
            for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                stats[i] = 0;
            }
        }
        result_pool_size[0] = 0;
        result_pool_worst_idx[0] = 0;
        result_pool_worst_dist[0] = -FLT_MAX;
    }

    // initialize the hash table to record expanded nodes
    hashtable_init(HASH_TABLE, bitlen);

    // load query into shared memory
    for (int i = tid; i < dim; i += blockDim.x) {
        QUERY_BUFFER[i] = query[i];
    }

    // load the top k list and candidate list with dummy initial values
    for (int i = tid; i < beam_sz; i += blockDim.x) {
        TOP_K_INDEX[i] = MAX_INDEX;
        TOP_K_DISTANCE[i] = FLT_MAX;
    }
    for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
        CANDIDATE_INDEX[i] = MAX_INDEX;
        CANDIDATE_DISTANCE[i] = FLT_MAX;
    }
    for (int i = tid; i < K; i += blockDim.x) {
        RESULT_POOL_INDEX[i] = MAX_INDEX;
        RESULT_POOL_DISTANCE[i] = FLT_MAX;
    }
    __syncthreads();

    // only the first slot in candidate list is actually initialized with valid entry point
    if (warp_id == 0) {
        DISTANCE_T entry_dist = warp_distance(dim, query, d_qg_data + static_cast<size_t>(entry_point) * row_offset, use_ip);
        if (lane_id == 0) {
            CANDIDATE_INDEX[0] = entry_point;
            CANDIDATE_DISTANCE[0] = entry_dist;
            if (GPU_SEARCH_STATS) local_exact_count++;
        }
    }
    __syncthreads();

    // query preparation, quantize the query
    query_prepare_gpu(QUERY_BUFFER, qf, d_qg_signs, dim, padded_dim);
#if GPU_RABITQ_USE_LUT
    pack_lut_gpu(static_cast<int>(padded_dim), qf.quantized_query, qf.lut);
#endif
    __syncthreads();

    // loop end condition: either entire topK expanded, or reach max iteration limit
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        // forgettable hash table management according to the CARGA paper
        if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
            hashtable_init(HASH_TABLE, bitlen);
            __syncthreads();
            hashtable_restore_expand_only(HASH_TABLE, bitlen, TOP_K_INDEX, beam_sz);
        }
        __syncthreads();

        dispatch_topk_by_sort_candidates_and_merge<2>(
            ALL_INDEX, ALL_DISTANCE, MERGED_TOPK_INDEX, MERGED_TOPK_DISTANCE,
            result_buffer_size, CANDIDATE_BUFFER_SIZE, beam_sz, (iter == 0));
        __syncthreads();

        // clear the candidate buffer before next iteration to avoid stale entries
        for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
            CANDIDATE_INDEX[i] = MAX_INDEX;
            CANDIDATE_DISTANCE[i] = FLT_MAX;
        }
        __syncthreads();

        if (warp_id == 0) {
            keep_expanding = pick_expanders(SEARCH_WIDTH, PARENT_LIST, beam_sz, TOP_K_INDEX);
        }
        __syncthreads();

        if (!keep_expanding) {
            if (tid == 0) {
                hops[0] = iter;
            }
            break;
        }

        INDEX_T current_node = TOP_K_INDEX[PARENT_LIST[0]] & 0x7fffffffu;
        if (tid == 0) {
            current_expanded = hashtable_insert(HASH_TABLE, bitlen, current_node);
        }
        __syncthreads();

        if (!current_expanded) {
            continue;
        }

        const float* current_row = d_qg_data + static_cast<size_t>(current_node) * row_offset;
        const vidType* current_neighbors = reinterpret_cast<const vidType*>(current_row + neighbor_offset);
        const uint64_t* current_codes = reinterpret_cast<const uint64_t*>(current_row + code_offset);
        const float* current_factors = current_row + factor_offset;

        staged_copy_async_begin_or_sync(STAGED_CODE_BUFFER, current_codes, static_cast<int>(max_degree * code_words));
        staged_copy_async_begin_or_sync(STAGED_FACTOR_BUFFER, current_factors, 3 * max_degree);

        DISTANCE_T exact_dist = warp_distance(dim, query, current_row, use_ip);
        if (GPU_SEARCH_STATS && tid == 0) {
            stats[GPU_SEARCH_STAT_ITERATIONS]++;
            local_exact_count++;
        }
        if (tid == 0) {
            result_pool_push_unique_unsorted(RESULT_POOL_INDEX, RESULT_POOL_DISTANCE, result_pool_size,
                result_pool_worst_idx, result_pool_worst_dist, K, current_node, exact_dist);
        }
        for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
            INDEX_T child_id = (i < max_degree) ? current_neighbors[i] : MAX_INDEX;
            if (child_id != MAX_INDEX && hashtable_contains(HASH_TABLE, bitlen, child_id)) {
                child_id = MAX_INDEX;
            }
            CANDIDATE_INDEX[i] = child_id;
            CANDIDATE_DISTANCE[i] = (child_id == MAX_INDEX) ? FLT_MAX : 0;
        }
        __syncthreads();

        // wait for the async copy to finish before move on to computation
        staged_copy_async_finish_or_sync();
        __syncthreads();

        // all warps work together to compute candidate distance
        if constexpr (GPU_RABITQ_USE_LUT) {
            const int bytes_per_neighbor = padded_dim >> 3;
            uint32_t warp_dist_est_count = 0;

#if GPU_RABITQ_FASTSCAN_SUBWARP_LANES == 32
            for (int i = warp_id; i < max_degree; i += WARPS_PER_BLOCK) {
                if (CANDIDATE_INDEX[i] == MAX_INDEX) continue;

                const int batch = i / WARP_SIZE;
                const int neighbor_idx = i & (WARP_SIZE - 1);
                const uint8_t* packed_codes_block =
                    reinterpret_cast<const uint8_t*>(STAGED_CODE_BUFFER) + batch * padded_dim * 4;
                const float* triple_x = STAGED_FACTOR_BUFFER + i;
                const float* factor_dq = STAGED_FACTOR_BUFFER + max_degree + i;
                const float* factor_vq = STAGED_FACTOR_BUFFER + 2 * max_degree + i;

                const DISTANCE_T est_dist = scan_one_neighbor_lanes_gpu<32>(
                    qf, packed_codes_block, neighbor_idx, triple_x, factor_dq, factor_vq,
                    exact_dist, padded_dim, bytes_per_neighbor);
                if (lane_id == 0) {
                    CANDIDATE_DISTANCE[i] = est_dist;
                    if (GPU_SEARCH_STATS) warp_dist_est_count++;
                }
                __syncwarp();
            }
#else
            constexpr int lanes_per_neighbor = GPU_RABITQ_FASTSCAN_SUBWARP_LANES;
            constexpr int groups_per_warp = WARP_SIZE / lanes_per_neighbor;
            const int group_lane = lane_id & (lanes_per_neighbor - 1);
            const int warp_group = lane_id / lanes_per_neighbor;
            uint32_t group_dist_est_count = 0;

            for (int base = warp_id * groups_per_warp; base < max_degree; base += WARPS_PER_BLOCK * groups_per_warp) {
                const int i = base + warp_group;
                const bool valid_candidate = i < max_degree && CANDIDATE_INDEX[i] != MAX_INDEX;
                if (valid_candidate) {
                    const int batch = i / WARP_SIZE;
                    const int neighbor_idx = i & (WARP_SIZE - 1);
                    const uint8_t* packed_codes_block =
                        reinterpret_cast<const uint8_t*>(STAGED_CODE_BUFFER) + batch * padded_dim * 4;
                    const float* triple_x = STAGED_FACTOR_BUFFER + i;
                    const float* factor_dq = STAGED_FACTOR_BUFFER + max_degree + i;
                    const float* factor_vq = STAGED_FACTOR_BUFFER + 2 * max_degree + i;

                    const DISTANCE_T est_dist = scan_one_neighbor_lanes_gpu<lanes_per_neighbor>(
                        qf, packed_codes_block, neighbor_idx, triple_x, factor_dq, factor_vq,
                        exact_dist, padded_dim, bytes_per_neighbor);
                    if (group_lane == 0) {
                        CANDIDATE_DISTANCE[i] = est_dist;
                        if (GPU_SEARCH_STATS) group_dist_est_count++;
                    }
                }
                __syncwarp();
            }

            if (GPU_SEARCH_STATS) {
                warp_dist_est_count = group_dist_est_count;
                for (int offset = lanes_per_neighbor; offset < WARP_SIZE; offset <<= 1) {
                    warp_dist_est_count += __shfl_down_sync(FULL_MASK, warp_dist_est_count, offset);
                }
            }
#endif
            if (lane_id == 0) {
                if (GPU_SEARCH_STATS) local_dist_est_count += warp_dist_est_count;
            }
        } else { // no LUT, all warps compute independently
            uint32_t warp_dist_est_count = 0;
            for (int i = warp_id; i < max_degree; i += WARPS_PER_BLOCK) {
                if (CANDIDATE_INDEX[i] == MAX_INDEX) continue;

                const uint64_t* packed_code = STAGED_CODE_BUFFER + static_cast<size_t>(i) * code_words;
                float triple_x = 0.0f;
                float factor_dq = 0.0f;
                float factor_vq = 0.0f;
                if (lane_id == 0) {
                    triple_x = STAGED_FACTOR_BUFFER[i];
                    factor_dq = STAGED_FACTOR_BUFFER[max_degree + i];
                    factor_vq = STAGED_FACTOR_BUFFER[2 * max_degree + i];
                }
                triple_x = SHFL(triple_x, 0);
                factor_dq = SHFL(factor_dq, 0);
                factor_vq = SHFL(factor_vq, 0);

                const DISTANCE_T est_dist = estimate_one_neighbor_warp_no_lut_gpu(
                    qf, packed_code, triple_x, factor_dq, factor_vq, exact_dist, padded_dim);
                if (lane_id == 0) {
                    CANDIDATE_DISTANCE[i] = isfinite(est_dist) ? est_dist : FLT_MAX;
                    if (GPU_SEARCH_STATS) warp_dist_est_count++;
                }
            }
            if (lane_id == 0) {
                if (GPU_SEARCH_STATS) local_dist_est_count += warp_dist_est_count;
            }
        }
        __syncthreads();
    }

    // final sort before writeback
    dispatch_topk_by_sort_candidates_and_merge<2>(
        ALL_INDEX, ALL_DISTANCE, MERGED_TOPK_INDEX, MERGED_TOPK_DISTANCE,
        result_buffer_size, CANDIDATE_BUFFER_SIZE, beam_sz, false);
    __syncthreads();

    if (tid == 0) {
        result_pool_sort_small(RESULT_POOL_INDEX, RESULT_POOL_DISTANCE, result_pool_size[0]);
        result_pool_recompute_worst(RESULT_POOL_DISTANCE, result_pool_size[0], result_pool_worst_idx, result_pool_worst_dist);
    }
    __syncthreads();

    if (result_pool_size[0] < static_cast<uint32_t>(K)) {
        const uint32_t base_pool_size = result_pool_size[0];
        for (uint32_t parent_idx = 0; parent_idx < base_pool_size; ++parent_idx) {
            if (tid == 0) {
                fill_parent_node_shared = RESULT_POOL_INDEX[parent_idx];
            }
            __syncthreads();

            const INDEX_T parent_node = fill_parent_node_shared;
            if (parent_node == MAX_INDEX) {
                continue;
            }

            const float* parent_row = d_qg_data + static_cast<size_t>(parent_node) * row_offset;
            const vidType* parent_neighbors = reinterpret_cast<const vidType*>(parent_row + neighbor_offset);

            for (int nbr_base = 0; nbr_base < max_degree; nbr_base += WARPS_PER_BLOCK) {
                if (lane_id == 0) {
                    fill_candidate_nodes[warp_id] = MAX_INDEX;
                }
                __syncthreads();

                const int nbr_idx = nbr_base + warp_id;
                INDEX_T child_node = MAX_INDEX;
                uint32_t inserted = 0;
                if (nbr_idx < max_degree) {
                    if (lane_id == 0) {
                        child_node = parent_neighbors[nbr_idx];
                        inserted = hashtable_insert(HASH_TABLE, bitlen, child_node);
                    }
                    child_node = __shfl_sync(FULL_MASK, child_node, 0);
                    inserted = __shfl_sync(FULL_MASK, inserted, 0);
                }

                if (nbr_idx < max_degree && inserted && child_node != MAX_INDEX) {
                    DISTANCE_T child_dist = warp_distance(dim, query, d_qg_data + static_cast<size_t>(child_node) * row_offset, use_ip);
                    if (lane_id == 0) {
                        fill_candidate_nodes[warp_id] = child_node;
                        fill_candidate_dists[warp_id] = child_dist;
                        if (GPU_SEARCH_STATS) {
                            local_exact_count++;
                        }
                    }
                }
                __syncthreads();

                if (tid == 0) {
                    const int chunk_end = min(max_degree - nbr_base, WARPS_PER_BLOCK);
                    for (int chunk_warp = 0; chunk_warp < chunk_end; ++chunk_warp) {
                        const INDEX_T accepted_child = fill_candidate_nodes[chunk_warp];
                        if (accepted_child == MAX_INDEX) {
                            continue;
                        }
                        result_pool_push_unique_unsorted(
                            RESULT_POOL_INDEX, RESULT_POOL_DISTANCE, result_pool_size,
                            result_pool_worst_idx, result_pool_worst_dist, K,
                            accepted_child, fill_candidate_dists[chunk_warp]);
                    }
                }
                __syncthreads();

                if (result_pool_size[0] >= static_cast<uint32_t>(K)) {
                    break;
                }
            }
            if (result_pool_size[0] >= static_cast<uint32_t>(K)) {
                break;
            }
        }
        if (tid == 0) {
            result_pool_sort_small(RESULT_POOL_INDEX, RESULT_POOL_DISTANCE, result_pool_size[0]);
            result_pool_recompute_worst(
                RESULT_POOL_DISTANCE, result_pool_size[0],
                result_pool_worst_idx, result_pool_worst_dist);
        }
        __syncthreads();
    }

    if (tid == 0) {

        const uint32_t pool_size = result_pool_size[0];
        for (int i = 0; i < K; ++i) {
            if (i < static_cast<int>(pool_size)) {
                d_results[query_id * K + i] = RESULT_POOL_INDEX[i];
            } else {
                // if there's not enough value in result pool, fill the rest with other top K
                d_results[query_id * K + i] = TOP_K_INDEX[i] & 0x7fffffff;
            }
        }
    }

    if (GPU_SEARCH_STATS) {
        uint32_t warp_exact_count = local_exact_count;
        uint32_t warp_est_count = local_dist_est_count;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            warp_exact_count += __shfl_down_sync(FULL_MASK, warp_exact_count, offset);
            warp_est_count += __shfl_down_sync(FULL_MASK, warp_est_count, offset);
        }
        if (lane_id == 0) {
            warp_stat_counts[warp_id] = warp_exact_count;
            warp_stat_counts[WARPS_PER_BLOCK + warp_id] = warp_est_count;
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t exact_count = 0;
            uint32_t est_count = 0;
            for (int i = 0; i < WARPS_PER_BLOCK; ++i) {
                exact_count += warp_stat_counts[i];
                est_count += warp_stat_counts[WARPS_PER_BLOCK + i];
            }
            stats[GPU_SEARCH_STAT_DIST_CMPS] = exact_count;
            stats[GPU_SEARCH_STAT_DIST_EST] = est_count;
            if (d_block_stats != nullptr) {
                gpu_long_t* dst = d_block_stats + static_cast<size_t>(query_id) * GPU_SEARCH_STAT_COUNT;
                for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                    dst[i] = static_cast<gpu_long_t>(stats[i]);
                }
            }
        }
    }

    (void)d_qg_signs;
}
