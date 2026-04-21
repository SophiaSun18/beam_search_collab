#pragma once
#include "utils.cuh"
#include "distance.cuh"
#include <cuda_runtime.h>
#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK   128
#endif
#define SEARCH_WIDTH    1     // MAX: 2 nodes * 32 degree = 64 neighbors = CANDIDATE_BUFFER_SIZE
#define BITLEN          10
#define MAX_ITERATIONS  1024
#define TEAM_SIZE       8
#define HASH_TABLE_CONFIG   1
#define CANDIDATE_BUFFER_SIZE 64
#define RESULT_BUFFER_SIZE (INTERNAL_TOPK + CANDIDATE_BUFFER_SIZE)

static __host__ inline uint32_t calculate_shared_mem_size(int dim, int topk, int max_deg, 
                                                           int search_width, int bitlen) {
    auto align_up = [](size_t value, size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    };
    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    size_t code_words = padded_dim / 64;
    size_t size = 0;    
    size += hashtable_getsize(BITLEN) * sizeof(INDEX_T);    // hash table
    size += RESULT_BUFFER_SIZE * sizeof(INDEX_T);           // top k and candidate index
    size += RESULT_BUFFER_SIZE * sizeof(DISTANCE_T);        // top k and candidate distance
    size += SEARCH_WIDTH * sizeof(INDEX_T);                 // next expanders / parent buffer
    size += dim * 4;                                        // query buffer size
    size += padded_dim * sizeof(float);                     // rotated query
    size += padded_dim * sizeof(uint8_t);                   // quantized query
    size += (padded_dim << 2) * sizeof(uint8_t);            // LUT buffer
    size += 3 * sizeof(float);                              // low/high/width
    size += sizeof(int32_t);                                // sum_q
    size = align_up(size, alignof(uint64_t));               // align staged candidate codes
    size += max_deg * code_words * sizeof(uint64_t);        // staged candidate codes
    size += 3 * max_deg * sizeof(float);                    // staged candidate factors
    size += 2 * sizeof(uint32_t);                           // counter and hops
    if (GPU_SEARCH_STATS) {
        size += GPU_SEARCH_STAT_COUNT * sizeof(uint32_t);   // search stats
    }
    size += topk * sizeof(INDEX_T);                         // result pool index
    size += topk * sizeof(DISTANCE_T);                      // result pool distance
    size += 2 * sizeof(uint32_t);                           // result pool size and worst idx
    size += sizeof(DISTANCE_T);                             // result pool worst dist
#if GPU_DISTANCE_TIMING
    size = align_up(size, alignof(gpu_long_t));             // align timing stats
    size += GPU_TIMING_COUNT * sizeof(gpu_long_t);          // timing stats
#endif
    printf("shared memory size: %ld\n", size);
    return size;
}

static __global__ GPU_LAUNCH_BOUNDS(BLOCK_SIZE)
void QuantizedBeamSearch(
    int K, int nq, int dim, int max_degree, size_t npoints,
    const float* __restrict__ d_queries,
    const float* __restrict__ d_qg_data,
    const float* __restrict__ d_qg_signs,
    vidType* __restrict__ d_results,
    gpu_long_t* d_block_stats,
    gpu_long_t* d_block_timing,
    vidType entry_point,
    size_t row_offset,
    size_t neighbor_offset,
    size_t code_offset,
    size_t factor_offset,
    bool debug_mode
) {
    // basic information about this block and thread
    const int query_id = blockIdx.x;
    if (query_id >= nq) return;
    const float* query = d_queries + query_id * dim;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    size_t code_words = padded_dim / 64;

    // initialize the shared memory
    extern __shared__ char shared_memory[];
    INDEX_T* ALL_INDEX = (INDEX_T*)shared_memory;
    INDEX_T* TOP_K_INDEX      = ALL_INDEX;
    INDEX_T* CANDIDATE_INDEX  = ALL_INDEX + INTERNAL_TOPK;

    DISTANCE_T* ALL_DISTANCE = (DISTANCE_T*)(ALL_INDEX + RESULT_BUFFER_SIZE);
    DISTANCE_T* TOP_K_DISTANCE      = ALL_DISTANCE;
    DISTANCE_T* CANDIDATE_DISTANCE  = ALL_DISTANCE + INTERNAL_TOPK;

    INDEX_T* HASH_TABLE = (INDEX_T*)(ALL_DISTANCE + RESULT_BUFFER_SIZE);
    INDEX_T* PARENT_LIST = (INDEX_T*)(HASH_TABLE + hashtable_getsize(BITLEN));
    DATA_T* QUERY_BUFFER = (DATA_T*)(PARENT_LIST + SEARCH_WIDTH);

    float* ROTATED_QUERY_BUFFER = reinterpret_cast<float*>(QUERY_BUFFER + dim);
    uint8_t* QUANTIZED_QUERY_BUFFER = reinterpret_cast<uint8_t*>(ROTATED_QUERY_BUFFER + padded_dim);
    uint8_t* LUT_BUFFER = QUANTIZED_QUERY_BUFFER + padded_dim;
    float* low_val = reinterpret_cast<float*>(LUT_BUFFER + (padded_dim << 2));
    float* high_val = low_val + 1;
    float* width = high_val + 1;
    int32_t* sum_q = reinterpret_cast<int32_t*>(width + 1);
    char* staged_base = reinterpret_cast<char*>(sum_q + 1);
    staged_base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(staged_base) + alignof(uint64_t) - 1) &
                                          ~(static_cast<uintptr_t>(alignof(uint64_t)) - 1));
    uint64_t* STAGED_CODE_BUFFER = reinterpret_cast<uint64_t*>(staged_base);
    float* STAGED_FACTOR_BUFFER = reinterpret_cast<float*>(STAGED_CODE_BUFFER + max_degree * code_words);
    
    uint32_t* counter = reinterpret_cast<uint32_t*>(STAGED_FACTOR_BUFFER + 3 * max_degree);
    uint32_t* hops = counter + 1;
    uint32_t* stats = hops + 1;
    INDEX_T* RESULT_POOL_INDEX = reinterpret_cast<INDEX_T*>(stats + (GPU_SEARCH_STATS ? GPU_SEARCH_STAT_COUNT : 0));
    DISTANCE_T* RESULT_POOL_DISTANCE = reinterpret_cast<DISTANCE_T*>(RESULT_POOL_INDEX + K);
    uint32_t* result_pool_size = reinterpret_cast<uint32_t*>(RESULT_POOL_DISTANCE + K);
    uint32_t* result_pool_worst_idx = result_pool_size + 1;
    DISTANCE_T* result_pool_worst_dist = reinterpret_cast<DISTANCE_T*>(result_pool_worst_idx + 1);
    gpu_long_t* timing = nullptr;
#if GPU_DISTANCE_TIMING
    char* timing_base = reinterpret_cast<char*>(result_pool_worst_dist + 1);
    timing_base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(timing_base) + alignof(gpu_long_t) - 1) &
                                          ~(static_cast<uintptr_t>(alignof(gpu_long_t)) - 1));
    timing = reinterpret_cast<gpu_long_t*>(timing_base);
#endif

    __shared__ QueryFactors qf;

    // initialize the block-wise total distance computation count
    __shared__ uint32_t keep_expanding;
    __shared__ uint32_t current_expanded;
    if (tid == 0) {
        qf.rotated_query = ROTATED_QUERY_BUFFER;
        qf.quantized_query = QUANTIZED_QUERY_BUFFER;
        qf.lut = LUT_BUFFER;
        qf.low_val = FLT_MAX;
        qf.high_val = -FLT_MAX;
        qf.width = 0.0f;
        qf.sum_q = 0;
        counter[0] = 0;
        hops[0] = 0;
        if (GPU_SEARCH_STATS) {
            for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                stats[i] = 0;
            }
        }
        result_pool_size[0] = 0;
        result_pool_worst_idx[0] = 0;
        result_pool_worst_dist[0] = -FLT_MAX;
        if (GPU_DISTANCE_TIMING) {
            for (int i = 0; i < GPU_TIMING_COUNT; i++) {
                timing[i] = 0;
            }
        }
    }

    // initialize the hash table to record expanded nodes
    if (HASH_TABLE_CONFIG) hashtable_init(HASH_TABLE, BITLEN);

    // load query into shared memory
    for (int i = tid; i < dim; i += blockDim.x) {
        if (i < dim) QUERY_BUFFER[i] = query[i];
    }
    
    // load the top k list and candidate list with dummy initial values
    for (int i = tid; i < RESULT_BUFFER_SIZE; i += blockDim.x) {
        ALL_INDEX[i] = MAX_INDEX;
        ALL_DISTANCE[i] = FLT_MAX;
    }
    for (int i = tid; i < K; i += blockDim.x) {
        RESULT_POOL_INDEX[i] = MAX_INDEX;
        RESULT_POOL_DISTANCE[i] = FLT_MAX;
    }

    // only the first slot in candidate list is actually initialized with valid entry point
    if (warp_id == 0) {
        gpu_long_t l2_start = 0;
        if (GPU_DISTANCE_TIMING && lane_id == 0) l2_start = gpu_globaltimer();
        CANDIDATE_INDEX[0] = entry_point;
        CANDIDATE_DISTANCE[0] = warp_l2_distance(dim, query, d_qg_data + static_cast<size_t>(entry_point) * row_offset);
        if (GPU_DISTANCE_TIMING && lane_id == 0) {
            atomicAdd(&timing[GPU_TIMING_L2_CALLS], gpu_long_t(1));
            atomicAdd(&timing[GPU_TIMING_L2_CYCLES], gpu_globaltimer() - l2_start);
        }
    }
    __syncthreads();

    // query preparation, quantize the query
    gpu_long_t query_prep_start = 0;
    if (GPU_DISTANCE_TIMING && tid == 0) query_prep_start = gpu_globaltimer();
    query_prepare_gpu(QUERY_BUFFER, qf, d_qg_signs, dim, padded_dim);
    __syncthreads();
    if (GPU_DISTANCE_TIMING && tid == 0) {
        atomicAdd(&timing[GPU_TIMING_QUERY_PREP_CYCLES], gpu_globaltimer() - query_prep_start);
    }
    
    // loop end condition: either entire topK expanded, or reach max iteration limit
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        // forgettable hash table management according to the CARGA paper
        if (HASH_TABLE_CONFIG) {
            if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
                hashtable_init(HASH_TABLE, BITLEN);
                __syncthreads();
                hashtable_restore_expand_only(HASH_TABLE, BITLEN, TOP_K_INDEX, INTERNAL_TOPK);
            }
            __syncthreads();
        }

        // use warp level sort to find the actual closest top K elements and update the topK list
        gpu_long_t sort_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) sort_start = gpu_globaltimer();
        topk_by_bitonic_sort<2, INTERNAL_TOPK / WARP_SIZE>(ALL_INDEX, ALL_DISTANCE, RESULT_BUFFER_SIZE,
            CANDIDATE_BUFFER_SIZE, INTERNAL_TOPK, (iter == 0));
        __syncthreads();
        if (GPU_DISTANCE_TIMING && tid == 0) {
            atomicAdd(&timing[GPU_TIMING_SORT_CYCLES], gpu_globaltimer() - sort_start);
        }

        if (debug_mode && query_id == 0 && tid == 0) {
            printf("[DEBUG] iteration %d\n", iter + 1);
            printf("  beam state:\n");
            for (int i = 0; i < INTERNAL_TOPK; ++i) {
                const INDEX_T raw = TOP_K_INDEX[i];
                const bool expanded = (raw & 0x80000000u) != 0;
                printf("    idx=%d, node=%u, dist=%.6f, expanded=%s\n",
                       i,
                       raw & 0x7fffffff,
                       TOP_K_DISTANCE[i],
                       expanded ? "true" : "false");
            }
            printf("  candidate buffer:\n");
            for (int i = 0; i < CANDIDATE_BUFFER_SIZE; ++i) {
                printf("    idx=%d, node=%u, dist=%.6f\n",
                       i,
                       CANDIDATE_INDEX[i] & 0x7fffffff,
                       CANDIDATE_DISTANCE[i]);
            }
        }

        // warp 0 pick the top-N unexpanded nodes and store in shared memory
        // this already marks the expanded nodes MSB to 1, no need to mark again
        gpu_long_t pick_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) pick_start = gpu_globaltimer();
        if (warp_id == 0) {
            keep_expanding = pick_expanders(SEARCH_WIDTH, PARENT_LIST, INTERNAL_TOPK, TOP_K_INDEX);
        }
        __syncthreads();
        if (GPU_DISTANCE_TIMING && tid == 0) {
            atomicAdd(&timing[GPU_TIMING_PICK_CYCLES], gpu_globaltimer() - pick_start);
        }

        if (!keep_expanding) {
            if (debug_mode && query_id == 0 && tid == 0) {
                printf("  expanding node: none\n");
            }
            if (tid == 0) {
                hops[0] = iter;
            }
            break;
        }

        // collect the neighbors of the expanding node and add into candidate list if not visited before
        // make sure to mark the invalid neighbor using dummy large values
        INDEX_T current_node = TOP_K_INDEX[PARENT_LIST[0]] & 0x7fffffff;

        if (tid == 0) {
            current_expanded = hashtable_insert(HASH_TABLE, BITLEN, current_node);
        }
        __syncthreads();
        if (!current_expanded) {
            continue;
        }

        if (debug_mode && query_id == 0 && tid == 0) {
            const INDEX_T parent_slot = PARENT_LIST[0];
            const DISTANCE_T parent_dist =
                (parent_slot < INTERNAL_TOPK) ? TOP_K_DISTANCE[parent_slot] : FLT_MAX;
            printf("  expanding node: %u\n", current_node);
            printf("  expanding slot: %u, dist=%.6f\n", parent_slot, parent_dist);
        }
        const float* current_row = d_qg_data + static_cast<size_t>(current_node) * row_offset;
        const vidType* current_neighbors = reinterpret_cast<const vidType*>(current_row + neighbor_offset);
        const uint64_t* current_codes = reinterpret_cast<const uint64_t*>(current_row + code_offset);
        const float* current_factors = current_row + factor_offset;

        gpu_long_t l2_start = 0;
        if (GPU_DISTANCE_TIMING && lane_id == 0) l2_start = gpu_globaltimer();
        DISTANCE_T exact_dist = warp_l2_distance(dim, query, current_row);
        if (GPU_DISTANCE_TIMING && lane_id == 0) {
            atomicAdd(&timing[GPU_TIMING_L2_CALLS], gpu_long_t(1));
            atomicAdd(&timing[GPU_TIMING_L2_CYCLES], gpu_globaltimer() - l2_start);
        }
        if (GPU_SEARCH_STATS && tid == 0) {
            stats[GPU_SEARCH_STAT_ITERATIONS]++;
            stats[GPU_SEARCH_STAT_DIST_CMPS]++;
        }
        gpu_long_t result_pool_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) result_pool_start = gpu_globaltimer();
        if (tid == 0) {
            result_pool_push_unique_unsorted(
                RESULT_POOL_INDEX,
                RESULT_POOL_DISTANCE,
                result_pool_size,
                result_pool_worst_idx,
                result_pool_worst_dist,
                K,
                current_node,
                exact_dist);
        }
        if (GPU_DISTANCE_TIMING && tid == 0) {
            atomicAdd(&timing[GPU_TIMING_RESULT_POOL_CYCLES], gpu_globaltimer() - result_pool_start);
        }
        gpu_long_t visited_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) visited_start = gpu_globaltimer();
        for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
            INDEX_T child_id = (i < max_degree) ? current_neighbors[i] : MAX_INDEX;
            if (child_id != MAX_INDEX && hashtable_contains(HASH_TABLE, BITLEN, child_id)) {
                child_id = MAX_INDEX;
            }
            CANDIDATE_INDEX[i] = child_id;
            CANDIDATE_DISTANCE[i] = (child_id == MAX_INDEX) ? FLT_MAX : 0;
        }
        __syncthreads();
        if (GPU_DISTANCE_TIMING && tid == 0) {
            atomicAdd(&timing[GPU_TIMING_VISITED_CYCLES], gpu_globaltimer() - visited_start);
        }

        // load the packed code and factors for the current row once so the candidate loop can
        // read it from shared memory instead of issuing many global loads
        gpu_long_t stage_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) stage_start = gpu_globaltimer();
        for (int j = tid; j < max_degree * code_words; j += blockDim.x) {
            STAGED_CODE_BUFFER[j] = current_codes[j];
        }
        for (int j = tid; j < 3 * max_degree; j += blockDim.x) {
            STAGED_FACTOR_BUFFER[j] = current_factors[j];
        }
        __syncthreads();
        if (GPU_DISTANCE_TIMING && tid == 0) {
            atomicAdd(&timing[GPU_TIMING_STAGE_CYCLES], gpu_globaltimer() - stage_start);
        }

        // all warps work together to compute candidate distance
        gpu_long_t est_region_start = 0;
        if (GPU_DISTANCE_TIMING && tid == 0) est_region_start = gpu_globaltimer();
        for (int i = warp_id; i < CANDIDATE_BUFFER_SIZE; i += WARPS_PER_BLOCK) {
            // visited candidates are marked as invalid
            if (CANDIDATE_INDEX[i] == MAX_INDEX) continue;

            // if current node is new in hash table, compute and store its distance
            const uint64_t* packed_code = STAGED_CODE_BUFFER + i * code_words;
            const float* factors = STAGED_FACTOR_BUFFER;
            float triple_x = 0.0f;
            float factor_dq = 0.0f;
            float factor_vq = 0.0f;
            if (lane_id == 0) {
                triple_x = factors[i];
                factor_dq = factors[max_degree + i];
                factor_vq = factors[2 * max_degree + i];
            }
            triple_x = __shfl_sync(FULL_MASK, triple_x, 0);
            factor_dq = __shfl_sync(FULL_MASK, factor_dq, 0);
            factor_vq = __shfl_sync(FULL_MASK, factor_vq, 0);
            
            gpu_long_t est_start = 0;
            if (GPU_DISTANCE_TIMING && lane_id == 0) est_start = gpu_globaltimer();
            DISTANCE_T est_dist = estimated_distance_gpu(
                qf, packed_code, triple_x, factor_dq, factor_vq, exact_dist, padded_dim);
            if (lane_id == 0) {
                atomicAdd(&counter[0], 1);
                if (GPU_SEARCH_STATS) {
                    atomicAdd(&stats[GPU_SEARCH_STAT_DIST_EST], 1u);
                }
                if (GPU_DISTANCE_TIMING) {
                    atomicAdd(&timing[GPU_TIMING_EST_CALLS], gpu_long_t(1));
                    atomicAdd(&timing[GPU_TIMING_EST_CYCLES], gpu_globaltimer() - est_start);
                }
                CANDIDATE_DISTANCE[i] = est_dist;
            }
        }
        __syncthreads();
        if (GPU_DISTANCE_TIMING && tid == 0) {
            int active_count = 0;
            for (int i = 0; i < CANDIDATE_BUFFER_SIZE; ++i) {
                if (CANDIDATE_INDEX[i] != MAX_INDEX) {
                    active_count++;
                }
            }
            if (active_count > 0) {
                atomicAdd(&timing[GPU_TIMING_EST_REGION_CANDIDATES], gpu_long_t(active_count));
                atomicAdd(&timing[GPU_TIMING_EST_REGION_CYCLES], gpu_globaltimer() - est_region_start);
            }
        }
    }

    // final sort before writeback
    gpu_long_t final_sort_start = 0;
    if (GPU_DISTANCE_TIMING && tid == 0) final_sort_start = gpu_globaltimer();
    topk_by_bitonic_sort<2, INTERNAL_TOPK / WARP_SIZE>(ALL_INDEX, ALL_DISTANCE, RESULT_BUFFER_SIZE,
        CANDIDATE_BUFFER_SIZE, INTERNAL_TOPK, false);
    __syncthreads();
    if (GPU_DISTANCE_TIMING && tid == 0) {
        atomicAdd(&timing[GPU_TIMING_FINAL_SORT_CYCLES], gpu_globaltimer() - final_sort_start);
    }

    if (tid == 0) {

        if (debug_mode && query_id == 0) {
            printf("[DEBUG] final result pool\n");
            for (uint32_t i = 0; i < result_pool_size[0]; ++i) {
                printf("  idx=%u, node=%u, dist=%.6f\n",
                       i,
                       RESULT_POOL_INDEX[i],
                       RESULT_POOL_DISTANCE[i]);
            }
        }

        const uint32_t pool_size = result_pool_size[0];
        for (int i = 0; i < K; ++i) {
            if (i < static_cast<int>(pool_size)) {
                d_results[query_id * K + i] = RESULT_POOL_INDEX[i];
            } else {
                // if there's not enough value in result pool, fill the rest with other top K
                d_results[query_id * K + i] = TOP_K_INDEX[i] & 0x7fffffff;
            }
        }
        if (GPU_SEARCH_STATS) {
            stats[GPU_SEARCH_STAT_DIST_EST] = counter[0];
            if (d_block_stats != nullptr) {
                gpu_long_t* dst = d_block_stats + static_cast<size_t>(query_id) * GPU_SEARCH_STAT_COUNT;
                for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                    dst[i] = static_cast<gpu_long_t>(stats[i]);
                }
            }
        }
        if (GPU_DISTANCE_TIMING) {
            if (d_block_timing != nullptr) {
                gpu_long_t* dst = d_block_timing + static_cast<size_t>(query_id) * GPU_TIMING_COUNT;
                for (int i = 0; i < GPU_TIMING_COUNT; i++) {
                    dst[i] = timing[i];
                }
            }
        }
    }

    (void)d_qg_signs;
    (void)npoints;
    (void)code_offset;
    (void)factor_offset;
}
