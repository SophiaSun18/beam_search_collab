#pragma once
#include "utils.cuh"
#include "distance.cuh"
#include <cuda_runtime.h>
#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK   128
#endif
#ifndef SEARCH_WIDTH
#define SEARCH_WIDTH    1
#endif
#define MAX_ITERATIONS  (1 << 15)
#define USE_SEED        0
#define PRUNE_CONFIG    0
#define THRESHOLD_CONFIG    0
#define HASH_TABLE_CONFIG   1

static __host__ inline uint32_t calculate_shared_mem_size(int dim, int beam_sz, int max_deg,
                                                           int search_width, int bitlen) {
    const size_t result_buffer_size = static_cast<size_t>(beam_sz) +
        round_up_power2_u32(static_cast<uint32_t>(search_width) * static_cast<uint32_t>(max_deg));
    size_t size = 0;    
    size += hashtable_getsize(bitlen) * sizeof(INDEX_T);    // hash table
    size += result_buffer_size * sizeof(INDEX_T);           // top k and candidate index
    size += result_buffer_size * sizeof(DISTANCE_T);        // top k and candidate distance
    size += search_width * sizeof(INDEX_T);                 // next expanders / parent buffer
    size += dim * 4;                                        // query buffer size
    size += sizeof(uint32_t);                               // hops
    if (GPU_SEARCH_STATS) {
        size += GPU_SEARCH_STAT_COUNT * sizeof(uint32_t);   // search stats
    }
    size += beam_sz * sizeof(INDEX_T);                      // merged topk scratch index
    size += beam_sz * sizeof(DISTANCE_T);                   // merged topk scratch distance
    printf("shared memory size: %ld\n", size);
    return size;
}

static __global__ void init_identity_map(size_t n, vid_t* map) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) map[idx] = idx;
}

static inline std::vector<uint32_t> load_sign_bit(const char* filename) {
    return std::vector<uint32_t>();
}

template <typename T>
static __global__ GPU_LAUNCH_BOUNDS(BLOCK_SIZE)
void BeamSearch(
    int K, int nq, int dim, int beam_sz, int bitlen, size_t npoints,
    const T* __restrict__ d_queries,
    const T* __restrict__ d_data,
    vid_t* __restrict__ d_results,
    gpu_long_t* d_total_stats,
    vid_t* d_starting_points,
    vid_t* d_seed_map,
    uint32_t* d_sign_bit,
    GraphGPU<vid_t> gg,
    bool use_ip
) {
    // basic information about this block and thread
    const int query_id = blockIdx.x;
    if (query_id >= nq) return;
    const T* query = d_queries + query_id * dim;
    const uint32_t maxDegree = gg.get_max_degree();
    const uint32_t candidate_buffer_size = round_up_power2_u32(static_cast<uint32_t>(SEARCH_WIDTH) * maxDegree);
    const uint32_t logical_beam_size = static_cast<uint32_t>(beam_sz);
    const uint32_t padded_beam_size = gpu_effective_sort_beam_size(logical_beam_size);
    INDEX_T* d_graph_ptr = gg.N(0);

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // initialize the shared memory
    extern __shared__ char shared_memory[];
    INDEX_T* ALL_INDEX = (INDEX_T*)shared_memory;
    INDEX_T* TOP_K_INDEX      = ALL_INDEX;
    INDEX_T* CANDIDATE_INDEX  = ALL_INDEX + padded_beam_size;

    const uint32_t result_buffer_size = padded_beam_size + candidate_buffer_size;
    DISTANCE_T* ALL_DISTANCE = (DISTANCE_T*)(ALL_INDEX + result_buffer_size);
    DISTANCE_T* TOP_K_DISTANCE      = ALL_DISTANCE;
    DISTANCE_T* CANDIDATE_DISTANCE  = ALL_DISTANCE + padded_beam_size;

    INDEX_T* HASH_TABLE = (INDEX_T*)(ALL_DISTANCE + result_buffer_size);
    INDEX_T* PARENT_LIST = (INDEX_T*)(HASH_TABLE + hashtable_getsize(bitlen));
    DATA_T* QUERY_BUFFER = (DATA_T*)(PARENT_LIST + SEARCH_WIDTH);
    uint32_t* hops = (uint32_t*)(QUERY_BUFFER + dim);
    uint32_t* stats = hops + 1;
    INDEX_T* MERGED_TOPK_INDEX = reinterpret_cast<INDEX_T*>(stats + (GPU_SEARCH_STATS ? GPU_SEARCH_STAT_COUNT : 0));
    DISTANCE_T* MERGED_TOPK_DISTANCE = reinterpret_cast<DISTANCE_T*>(MERGED_TOPK_INDEX + padded_beam_size);

    // initialize the block-wise total distance computation count
    __shared__ uint32_t keep_expanding;
    __shared__ uint32_t warp_exact_counts[WARPS_PER_BLOCK];
    uint32_t local_exact_count = 0;
    if (tid == 0) {
        hops[0] = 0;
        if (GPU_SEARCH_STATS) {
            for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                stats[i] = 0;
            }
        }
    }
    if (GPU_SEARCH_STATS && tid < WARPS_PER_BLOCK) {
        warp_exact_counts[tid] = 0;
    }

    // initialize the hash table to record visited (computed) nodes
    if (HASH_TABLE_CONFIG) hashtable_init(HASH_TABLE, bitlen);

    // load query into shared memory
    for (int i = tid; i < dim; i += blockDim.x) {
        if (i < dim) QUERY_BUFFER[i] = query[i];
    }
    
    // load the top k list with dummy initial values
    for (int i = tid; i < static_cast<int>(padded_beam_size); i += blockDim.x) {
        TOP_K_INDEX[i] = MAX_INDEX;
        TOP_K_DISTANCE[i] = FLT_MAX;
    }
    __syncthreads();

    // load the candidate buffer with starting points
    for (int i = warp_id; i < candidate_buffer_size; i += WARPS_PER_BLOCK) {
        CANDIDATE_INDEX[i] = d_starting_points[i];
        DISTANCE_T dist = FLT_MAX;
        if (CANDIDATE_INDEX[i] != MAX_INDEX) {
            dist = warp_distance(dim, QUERY_BUFFER, d_data + CANDIDATE_INDEX[i] * dim, use_ip);
        }
        if (lane_id == 0) {     // warp computes, only lane 0 writes
            if (CANDIDATE_INDEX[i] != MAX_INDEX) {
                if (GPU_SEARCH_STATS) local_exact_count++;
                hashtable_insert(HASH_TABLE, bitlen, CANDIDATE_INDEX[i]);
            }
            CANDIDATE_DISTANCE[i] = dist;
        }
    }
    __syncthreads();
    
    // loop end condition: either entire topK expanded, or reach max iteration limit
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        // forgettable hash table management according to the CARGA paper
        if (HASH_TABLE_CONFIG) {
            if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
                hashtable_init(HASH_TABLE, bitlen);
                __syncthreads();
                hashtable_restore(HASH_TABLE, bitlen, TOP_K_INDEX, logical_beam_size);
            }
            __syncthreads();
        }

        // use warp level sort to find the actual closest top K elements and update the topK list
        dispatch_topk_by_sort_candidates_and_merge_runtime(
            ALL_INDEX, ALL_DISTANCE, MERGED_TOPK_INDEX, MERGED_TOPK_DISTANCE,
            result_buffer_size, candidate_buffer_size, padded_beam_size, (iter == 0));
        __syncthreads();

        for (uint32_t i = logical_beam_size + tid; i < padded_beam_size; i += blockDim.x) {
            TOP_K_INDEX[i] = MAX_INDEX;
            TOP_K_DISTANCE[i] = FLT_MAX;
        }
        __syncthreads();


        // warp 0 pick the top-N unexpanded nodes and store in shared memory
        // this already marks the expanded nodes MSB to 1, no need to mark again
        if (warp_id == 0) {
            keep_expanding = pick_expanders(SEARCH_WIDTH, PARENT_LIST, logical_beam_size, TOP_K_INDEX);
        }
        __syncthreads();
        if (!keep_expanding) {
            if (tid == 0) {
                hops[0] = iter;
            }
            break;
        }
        if (GPU_SEARCH_STATS && tid == 0) {
            stats[GPU_SEARCH_STAT_ITERATIONS] += keep_expanding;
        }

        // collect the neighbors of the expanding nodes and add into candidate list if not visited before
        // make sure to mark the invalid neighbor using dummy large values
    for (int i = tid; i < candidate_buffer_size; i += blockDim.x) {
        const bool in_work = static_cast<uint32_t>(i) < static_cast<uint32_t>(SEARCH_WIDTH) * maxDegree;
        const uint32_t parent_idx = in_work ? static_cast<uint32_t>(i) / maxDegree : 0;
        const uint32_t neighbor_idx = in_work ? static_cast<uint32_t>(i) - parent_idx * maxDegree : 0;
        INDEX_T child_id = MAX_INDEX;
        if (in_work && parent_idx < keep_expanding) {
            const INDEX_T current_node = TOP_K_INDEX[PARENT_LIST[parent_idx]] & 0x7fffffffu;
            const vid_t* neighbors = gg.N(current_node);
            child_id = neighbors[neighbor_idx];
        }

            if (child_id != MAX_INDEX) {    // check if visited already
                if (hashtable_insert(HASH_TABLE, bitlen, child_id) == 0) {
                    child_id = MAX_INDEX;
                    CANDIDATE_DISTANCE[i] = FLT_MAX;
                } else {
                    CANDIDATE_DISTANCE[i] = 0;
                }
            }
            CANDIDATE_INDEX[i] = child_id;
        }
        __syncthreads();

        // all warps work together to compute candidate distance
        for (int i = warp_id; i < candidate_buffer_size; i += WARPS_PER_BLOCK) {
            // visited candidates are marked as invalid
            if (CANDIDATE_INDEX[i] == MAX_INDEX) continue;

            // if current node is new in hash table, compute and store its distance
            DISTANCE_T dist = warp_distance(dim, QUERY_BUFFER, d_data + CANDIDATE_INDEX[i] * dim, use_ip);
            if (lane_id == 0) {
                if (GPU_SEARCH_STATS) local_exact_count++;
                CANDIDATE_DISTANCE[i] = dist;
            }
        }
        __syncthreads();
    }

    // final sort before writeback
    dispatch_topk_by_sort_candidates_and_merge_runtime(
        ALL_INDEX, ALL_DISTANCE, MERGED_TOPK_INDEX, MERGED_TOPK_DISTANCE,
        result_buffer_size, candidate_buffer_size, padded_beam_size, false);
    __syncthreads();

    // return the top K result
    for (int i = tid; i < K; i += blockDim.x) {
        d_results[query_id * K + i] = TOP_K_INDEX[i] & ~0x80000000;
    }
    if (GPU_SEARCH_STATS) {
        if (lane_id == 0) {
            warp_exact_counts[warp_id] = local_exact_count;
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t exact_count = 0;
            for (int i = 0; i < WARPS_PER_BLOCK; i++) {
                exact_count += warp_exact_counts[i];
            }
            stats[GPU_SEARCH_STAT_DIST_CMPS] = exact_count;
            stats[GPU_SEARCH_STAT_DIST_EST] = 0;
            gpu_long_t* dst = d_total_stats + static_cast<size_t>(query_id) * GPU_SEARCH_STAT_COUNT;
            for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                dst[i] = static_cast<gpu_long_t>(stats[i]);
            }
        }
    }
}
