#pragma once
#include "utils.cuh"
#include "distance.cuh"
#include <cuda_runtime.h>
#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK   128
#endif
#define SEARCH_WIDTH    1     // MAX: 2 nodes * 32 degree = 64 neighbors = CANDIDATE_BUFFER_SIZE
// #define BITLEN          GPU_HASH_BITLEN_FOR_TOPK(INTERNAL_TOPK)
#define MAX_ITERATIONS  (1 << 15)
#define TEAM_SIZE       8
#define USE_SEED        0
#define PRUNE_CONFIG    0
#define THRESHOLD_CONFIG    0
#define HASH_TABLE_CONFIG   1
#define CANDIDATE_BUFFER_SIZE 64
// #define RESULT_BUFFER_SIZE (INTERNAL_TOPK + CANDIDATE_BUFFER_SIZE)

static __host__ inline uint32_t calculate_shared_mem_size(int dim, int beam_sz, int max_deg,
                                                           int search_width, int bitlen) {
    const size_t result_buffer_size = static_cast<size_t>(beam_sz) + CANDIDATE_BUFFER_SIZE;
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
#if GPU_FRONTIER_AUDIT
    size += beam_sz * sizeof(DISTANCE_T);                   // exact-L2 frontier audit scratch
#endif
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
    double* d_expander_trace_sums,
    gpu_long_t* d_expander_trace_counts,
    double* d_expander_trace_net_descent,
    double* d_query_l2_progress_sum,
    double* d_query_l2_progress_per_expander_sum,
    gpu_long_t* d_query_expander_steps_sum,
    gpu_long_t* d_query_expander_count,
    double* d_query_first_expander_l2_by_query,
    gpu_long_t* d_query_expander_steps_by_query,
    int expander_trace_max_steps,
    AccessTraceRecord* d_access_trace_records,
    uint32_t* d_access_trace_counts,
    int access_trace_max_records_per_query,
    GpuFrontierAuditRecord* d_frontier_audit_records,
    uint32_t* d_frontier_audit_counts,
    int frontier_audit_max_records_per_query,
    GpuQualityHitTraceRecord* d_quality_hit_records,
    uint32_t* d_quality_hit_counts,
    int quality_hit_max_records_per_query,
    const int* d_groundtruth,
    int gt_k,
    uint32_t* d_quality_hit_previous_masks,
    uint32_t* d_quality_hit_seen_masks,
    uint32_t* d_quality_hit_current_masks,
    int quality_hit_mask_words,
    GpuQualityHitTraceRecord* d_quality_beam_hit_records,
    uint32_t* d_quality_beam_hit_counts,
    uint32_t* d_quality_beam_hit_previous_masks,
    uint32_t* d_quality_beam_hit_seen_masks,
    uint32_t* d_quality_beam_hit_current_masks,
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
    INDEX_T* d_graph_ptr = gg.N(0);

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

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
    uint32_t* hops = (uint32_t*)(QUERY_BUFFER + dim);
    uint32_t* stats = hops + 1;
    INDEX_T* MERGED_TOPK_INDEX = reinterpret_cast<INDEX_T*>(stats + (GPU_SEARCH_STATS ? GPU_SEARCH_STAT_COUNT : 0));
    DISTANCE_T* MERGED_TOPK_DISTANCE = reinterpret_cast<DISTANCE_T*>(MERGED_TOPK_INDEX + beam_sz);
    DISTANCE_T* FRONTIER_AUDIT_DISTANCE = MERGED_TOPK_DISTANCE + beam_sz;

    // initialize the block-wise total distance computation count
    __shared__ uint32_t keep_expanding;
    __shared__ uint32_t warp_exact_counts[WARPS_PER_BLOCK];
#if GPU_ACCESS_TRACE
    __shared__ uint32_t access_active_mask;
#endif
    uint32_t local_exact_count = 0;
    double previous_expander_dist = 0.0;
    double first_expander_dist = 0.0;
    double best_expander_dist = 0.0;
    bool has_previous_expander_dist = false;
    int expander_steps = 0;
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
    for (int i = tid; i < beam_sz; i += blockDim.x) {
        TOP_K_INDEX[i] = MAX_INDEX;
        TOP_K_DISTANCE[i] = FLT_MAX;
    }

    // load the candidate buffer with starting points
    for (int i = warp_id; i < CANDIDATE_BUFFER_SIZE; i += WARPS_PER_BLOCK) {
        CANDIDATE_INDEX[i] = d_starting_points[i];
        DISTANCE_T dist = FLT_MAX;
        if (CANDIDATE_INDEX[i] != MAX_INDEX) {
            dist = warp_distance(dim, query, d_data + CANDIDATE_INDEX[i] * dim, use_ip);
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
#if GPU_ACCESS_TRACE
    if (tid == 0 && d_starting_points[0] != MAX_INDEX) {
        record_gpu_access_trace(d_access_trace_records,
                                d_access_trace_counts,
                                access_trace_max_records_per_query,
                                static_cast<uint32_t>(query_id),
                                0,
                                ACCESS_TRACE_BASE_SEED_EXACT,
                                d_starting_points[0],
                                1u);
    }
    __syncthreads();
#endif

    // loop end condition: either entire topK expanded, or reach max iteration limit
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        // forgettable hash table management according to the CARGA paper
        if (HASH_TABLE_CONFIG) {
            if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
                hashtable_init(HASH_TABLE, bitlen);
                __syncthreads();
                hashtable_restore(HASH_TABLE, bitlen, TOP_K_INDEX, beam_sz);
            }
            __syncthreads();
        }

        // use warp level sort to find the actual closest top K elements and update the topK list
        dispatch_topk_by_sort_candidates_and_merge<2>(
            ALL_INDEX, ALL_DISTANCE, MERGED_TOPK_INDEX, MERGED_TOPK_DISTANCE,
            result_buffer_size, CANDIDATE_BUFFER_SIZE, beam_sz, (iter == 0));
        __syncthreads();

        // warp 0 pick the top-N unexpanded nodes and store in shared memory
        // this already marks the expanded nodes MSB to 1, no need to mark again
        if (warp_id == 0) {
            keep_expanding = pick_expanders(SEARCH_WIDTH, PARENT_LIST, beam_sz, TOP_K_INDEX);
        }
        __syncthreads();
        if (iter > 0 && keep_expanding) {
            record_gpu_quality_hit_trace(d_quality_hit_records,
                                         d_quality_hit_counts,
                                         quality_hit_max_records_per_query,
                                         static_cast<uint32_t>(query_id),
                                         static_cast<uint32_t>(iter),
                                         GPU_QUALITY_HIT_STAGE_EXPAND,
                                         TOP_K_INDEX,
                                         beam_sz,
                                         K,
                                         d_groundtruth,
                                         gt_k,
                                         d_quality_hit_previous_masks,
                                         d_quality_hit_seen_masks,
                                         d_quality_hit_current_masks,
                                         quality_hit_mask_words);
            record_gpu_quality_beam_hit_trace(d_quality_beam_hit_records,
                                              d_quality_beam_hit_counts,
                                              quality_hit_max_records_per_query,
                                              static_cast<uint32_t>(query_id),
                                              static_cast<uint32_t>(iter),
                                              GPU_QUALITY_HIT_STAGE_EXPAND,
                                              TOP_K_INDEX,
                                              beam_sz,
                                              K,
                                              d_groundtruth,
                                              gt_k,
                                              d_quality_beam_hit_previous_masks,
                                              d_quality_beam_hit_seen_masks,
                                              d_quality_beam_hit_current_masks,
                                              quality_hit_mask_words);
        }
        if (!keep_expanding) {
            if (tid == 0) {
                hops[0] = iter;
            }
            break;
        }
        if (tid == 0) {
            const int parent_idx = PARENT_LIST[0];
            const double expander_dist =
                static_cast<double>(TOP_K_DISTANCE[parent_idx]);
            record_gpu_expander_trace(d_expander_trace_sums,
                                      d_expander_trace_counts,
                                      expander_trace_max_steps,
                                      expander_steps,
                                      expander_dist,
                                      previous_expander_dist,
                                      has_previous_expander_dist,
                                      first_expander_dist,
                                      best_expander_dist);
            expander_steps++;
        }
        if (GPU_SEARCH_STATS && tid == 0) {
            stats[GPU_SEARCH_STAT_ITERATIONS]++;
        }

        // collect the neighbors of the expanding node and add into candidate list if not visited before
        // make sure to mark the invalid neighbor using dummy large values
        INDEX_T current_node = TOP_K_INDEX[PARENT_LIST[0]] & 0x7fffffff;
        record_gpu_frontier_audit(d_frontier_audit_records,
                                  d_frontier_audit_counts,
                                  frontier_audit_max_records_per_query,
                                  static_cast<uint32_t>(query_id),
                                  static_cast<uint32_t>(iter + 1),
                                  TOP_K_INDEX,
                                  beam_sz,
                                  query,
                                  d_data,
                                  static_cast<size_t>(dim),
                                  dim,
                                  use_ip,
                                  current_node,
                                  FRONTIER_AUDIT_DISTANCE);
        vid_t* neighbors = gg.N(current_node);
        __syncthreads();

        for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
            INDEX_T child_id = (i < maxDegree) ? neighbors[i] : MAX_INDEX;      // only in-bound threads take from the neighbor list

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
#if GPU_ACCESS_TRACE
        if (tid == 0) {
            access_active_mask = 0;
        }
        __syncthreads();
#endif
        for (int i = warp_id; i < CANDIDATE_BUFFER_SIZE; i += WARPS_PER_BLOCK) {
            // visited candidates are marked as invalid
            if (CANDIDATE_INDEX[i] == MAX_INDEX) continue;

            // if current node is new in hash table, compute and store its distance
            DISTANCE_T dist = warp_distance(dim, query, d_data + CANDIDATE_INDEX[i] * dim, use_ip);
            if (lane_id == 0) {
#if GPU_ACCESS_TRACE
                if (i < 32) {
                    atomicOr(&access_active_mask, 1u << i);
                }
#endif
                if (GPU_SEARCH_STATS) local_exact_count++;
                CANDIDATE_DISTANCE[i] = dist;
            }
        }
        __syncthreads();
#if GPU_ACCESS_TRACE
        if (tid == 0) {
            record_gpu_access_trace(d_access_trace_records,
                                    d_access_trace_counts,
                                    access_trace_max_records_per_query,
                                    static_cast<uint32_t>(query_id),
                                    static_cast<uint32_t>(iter),
                                    ACCESS_TRACE_BASE_EXPAND_EXACT,
                                    current_node,
                                    access_active_mask);
        }
        __syncthreads();
#endif
    }

    // final sort before writeback
    if (GPU_EXPANDER_TRACE && tid == 0 && d_expander_trace_net_descent != nullptr &&
        has_previous_expander_dist) {
        atomicAdd(d_expander_trace_net_descent, first_expander_dist - best_expander_dist);
        record_gpu_query_expander_efficiency(
            d_query_l2_progress_sum,
            d_query_l2_progress_per_expander_sum,
            d_query_expander_steps_sum,
            d_query_expander_count,
            d_query_first_expander_l2_by_query,
            d_query_expander_steps_by_query,
            query_id,
            expander_steps,
            first_expander_dist,
            best_expander_dist);
    }

    // final sort before writeback
    dispatch_topk_by_sort_candidates_and_merge<2>(
        ALL_INDEX, ALL_DISTANCE, MERGED_TOPK_INDEX, MERGED_TOPK_DISTANCE,
        result_buffer_size, CANDIDATE_BUFFER_SIZE, beam_sz, false);
    __syncthreads();

    record_gpu_quality_hit_trace(d_quality_hit_records,
                                 d_quality_hit_counts,
                                 quality_hit_max_records_per_query,
                                 static_cast<uint32_t>(query_id),
                                 static_cast<uint32_t>(expander_steps),
                                 GPU_QUALITY_HIT_STAGE_FINAL,
                                 TOP_K_INDEX,
                                 beam_sz,
                                 K,
                                 d_groundtruth,
                                 gt_k,
                                 d_quality_hit_previous_masks,
                                 d_quality_hit_seen_masks,
                                 d_quality_hit_current_masks,
                                 quality_hit_mask_words);
    record_gpu_quality_beam_hit_trace(d_quality_beam_hit_records,
                                      d_quality_beam_hit_counts,
                                      quality_hit_max_records_per_query,
                                      static_cast<uint32_t>(query_id),
                                      static_cast<uint32_t>(expander_steps),
                                      GPU_QUALITY_HIT_STAGE_FINAL,
                                      TOP_K_INDEX,
                                      beam_sz,
                                      K,
                                      d_groundtruth,
                                      gt_k,
                                      d_quality_beam_hit_previous_masks,
                                      d_quality_beam_hit_seen_masks,
                                      d_quality_beam_hit_current_masks,
                                      quality_hit_mask_words);

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
