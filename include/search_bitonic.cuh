#pragma once
#include "utils.cuh"
#include "hash_filter.cuh"
#include "topk.cuh"
#include <cuda_runtime.h>
#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK   128
#endif
#define SEARCH_WIDTH    1     // MAX: 2 nodes * 32 degree = 64 neighbors = CANDIDATE_BUFFER_SIZE
#define BITLEN          10
#define MAX_ITERATIONS  1024
#define TEAM_SIZE       8
#define USE_SEED        0
#define PRUNE_CONFIG    0
#define THRESHOLD_CONFIG    0
#define HASH_TABLE_CONFIG   1
#define CANDIDATE_BUFFER_SIZE 64
#define TOPK_BUFFER_SIZE  BLOCK_SIZE  // must be >= BLOCK_SIZE * REG for SharedBitonicTopK

static __host__ inline uint32_t calculate_shared_mem_size(int dim, int topk, int max_deg,
                                                           int search_width, int bitlen) {
    size_t size = 0;
    size += TOPK_BUFFER_SIZE * sizeof(INDEX_T);             // top k index
    size += TOPK_BUFFER_SIZE * sizeof(DISTANCE_T);          // top k distance
    size += CANDIDATE_BUFFER_SIZE * sizeof(INDEX_T);        // candidate index
    size += CANDIDATE_BUFFER_SIZE * sizeof(DISTANCE_T);     // candidate distance
    size += HashFilter::shared_mem_bytes(BITLEN);            // hash table
    size += SEARCH_WIDTH * sizeof(INDEX_T);                 // next expanders / parent buffer
    size += dim * 4;                                        // query buffer size
    size += 2 * 4;                                          // terminate flag and counter
    if (GPU_SEARCH_STATS) {
        size += GPU_SEARCH_STAT_COUNT * sizeof(uint32_t);   // search stats
    }
    size += GPU_TIMING_COUNT * sizeof(gpu_long_t);          // timing stats
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
    int K, int nq, int dim, size_t npoints,
    const T* __restrict__ d_queries,
    const T* __restrict__ d_data,
    vid_t* __restrict__ d_results,
    gpu_long_t* d_total_stats,
    gpu_long_t* d_total_timing,
    vid_t* d_starting_points,
    vid_t* d_seed_map,
    uint32_t* d_sign_bit,
    GraphGPU<vid_t> gg,
    bool debug_mode
) {
    // basic information about this block and thread
    const int query_id = blockIdx.x;
    if (query_id > nq) return;
    const T* query = d_queries + query_id * dim;
    const uint32_t maxDegree = gg.get_max_degree();
    INDEX_T* d_graph_ptr = gg.N(0);

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // initialize the shared memory
    extern __shared__ char shared_memory[];
    INDEX_T* TOP_K_INDEX = (INDEX_T*)shared_memory;
    DISTANCE_T* TOP_K_DISTANCE = (DISTANCE_T*)(TOP_K_INDEX + TOPK_BUFFER_SIZE);
    INDEX_T* CANDIDATE_INDEX = (INDEX_T*)(TOP_K_DISTANCE + TOPK_BUFFER_SIZE);
    DISTANCE_T* CANDIDATE_DISTANCE = (DISTANCE_T*)(CANDIDATE_INDEX + CANDIDATE_BUFFER_SIZE);
    INDEX_T* HASH_TABLE = (INDEX_T*)(CANDIDATE_DISTANCE + CANDIDATE_BUFFER_SIZE);
    HashFilter hf(HASH_TABLE, BITLEN);
    INDEX_T* PARENT_LIST = (INDEX_T*)(HASH_TABLE + HashFilter::shared_mem_bytes(BITLEN) / sizeof(INDEX_T));
    DATA_T* QUERY_BUFFER = (DATA_T*)(PARENT_LIST + SEARCH_WIDTH);
    uint32_t* counter = (uint32_t*)(QUERY_BUFFER + dim);
    uint32_t* hops = (uint32_t*)(counter + 1);
    uint32_t* stats = hops + 1;
    gpu_long_t* timing = reinterpret_cast<gpu_long_t*>(stats + (GPU_SEARCH_STATS ? GPU_SEARCH_STAT_COUNT : 0));

    SharedBitonicTopK<BLOCK_SIZE, 1> topk(TOPK_BUFFER_SIZE, TOP_K_DISTANCE, TOP_K_INDEX);

    // initialize the block-wise total distance computation count
    __shared__ uint32_t keep_expanding;
    if (tid == 0) {
        counter[0] = 0;
        hops[0] = 0;
        if (GPU_SEARCH_STATS) {
            for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                stats[i] = 0;
            }
        }
        for (int i = 0; i < GPU_TIMING_COUNT; i++) {
            timing[i] = 0;
        }
    }

    // initialize the hash table to record visited (computed) nodes
    if (HASH_TABLE_CONFIG) hf.init();

    // load query into shared memory
    for (int i = tid; i < dim; i += blockDim.x) {
        if (i < dim) QUERY_BUFFER[i] = query[i];
    }

    // load the candidate buffer with starting points
    for (int i = warp_id; i < CANDIDATE_BUFFER_SIZE; i += WARPS_PER_BLOCK) {
        CANDIDATE_INDEX[i] = d_starting_points[i];
        DISTANCE_T dist = FLT_MAX;
        gpu_long_t l2_start = 0;
        if (GPU_DISTANCE_TIMING && lane_id == 0 && CANDIDATE_INDEX[i] != MAX_INDEX) l2_start = gpu_globaltimer();
        if (CANDIDATE_INDEX[i] != MAX_INDEX) {
            dist = warp_l2_distance(dim, query, d_data + CANDIDATE_INDEX[i] * dim);
        }
        if (lane_id == 0) {     // warp computes, only lane 0 writes
            if (CANDIDATE_INDEX[i] != MAX_INDEX) {
                atomicAdd(&counter[0], 1);
                hf.insert(CANDIDATE_INDEX[i]);
                if (GPU_DISTANCE_TIMING) {
                    atomicAdd(&timing[GPU_TIMING_L2_CALLS], gpu_long_t(1));
                    atomicAdd(&timing[GPU_TIMING_L2_CYCLES], gpu_globaltimer() - l2_start);
                }
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
                hf.init();
            }
            __syncthreads();
        }

        // merge candidates into topk buffer
        {
            float    dist[1];
            uint32_t idx[1];
            dist[0] = (tid < CANDIDATE_BUFFER_SIZE) ? CANDIDATE_DISTANCE[tid] : FLT_MAX;
            idx[0]  = (tid < CANDIDATE_BUFFER_SIZE) ? CANDIDATE_INDEX[tid]    : MAX_INDEX;
            topk.merge(dist, idx);
        }

        if (debug_mode && query_id == 0 && tid == 0) {
            printf("[DEBUG] iteration %d\n", iter + 1);
            printf("  beam state:\n");
            for (int i = 0; i < TOPK_BUFFER_SIZE; ++i) {
                const INDEX_T raw = TOP_K_INDEX[i];
                if (raw == MAX_INDEX) continue;
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

        // pick the top-N unexpanded nodes
        {
            uint32_t parent_buf[BLOCK_SIZE];
            unsigned found = topk.pick(parent_buf, SEARCH_WIDTH);
            if (warp_id == 0 && lane_id == 0) {
                keep_expanding = found;
                if (found > 0) PARENT_LIST[0] = parent_buf[0];
            }
        }
        __syncthreads();

        if (debug_mode && query_id == 0 && tid == 0) {
            if (keep_expanding) {
                printf("  expanding node: %u\n", PARENT_LIST[0]);
            } else {
                printf("  expanding node: none\n");
            }
        }
        if (!keep_expanding) {
            if (tid == 0) {
                hops[0] = iter;
            }
            break;
        }
        if (GPU_SEARCH_STATS && tid == 0) {
            stats[GPU_SEARCH_STAT_ITERATIONS]++;
        }

        // collect the neighbors of the expanding node and add into candidate list if not visited before
        // make sure to mark the invalid neighbor using dummy large values
        INDEX_T current_node = PARENT_LIST[0];
        vid_t* neighbors = gg.N(current_node);
        gpu_long_t l2_start = 0;
        if (GPU_DISTANCE_TIMING && lane_id == 0) l2_start = gpu_globaltimer();
        DISTANCE_T exact_dist = warp_l2_distance(dim, query, d_data + current_node * dim);
        if (GPU_DISTANCE_TIMING && lane_id == 0) {
            atomicAdd(&timing[GPU_TIMING_L2_CALLS], gpu_long_t(1));
            atomicAdd(&timing[GPU_TIMING_L2_CYCLES], gpu_globaltimer() - l2_start);
        }
        __syncthreads();

        for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
            INDEX_T child_id = (i < maxDegree) ? neighbors[i] : MAX_INDEX;      // only in-bound threads take from the neighbor list

            if (child_id != MAX_INDEX) {    // check if visited already
                if (hf.insert(child_id) == 0) {
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
        for (int i = warp_id; i < CANDIDATE_BUFFER_SIZE; i += WARPS_PER_BLOCK) {
            // visited candidates are marked as invalid
            if (CANDIDATE_INDEX[i] == MAX_INDEX) continue;

            // if current node is new in hash table, compute and store its distance
            gpu_long_t cand_l2_start = 0;
            if (GPU_DISTANCE_TIMING && lane_id == 0) cand_l2_start = gpu_globaltimer();
            DISTANCE_T dist = warp_l2_distance(dim, query, d_data + CANDIDATE_INDEX[i] * dim);
            if (lane_id == 0) {
                atomicAdd(&counter[0], 1);
                if (GPU_DISTANCE_TIMING) {
                    atomicAdd(&timing[GPU_TIMING_L2_CALLS], gpu_long_t(1));
                    atomicAdd(&timing[GPU_TIMING_L2_CYCLES], gpu_globaltimer() - cand_l2_start);
                }
                CANDIDATE_DISTANCE[i] = dist;
            }
        }
        __syncthreads();
    }

    // final merge before writeback
    {
        float    dist[1];
        uint32_t idx[1];
        dist[0] = (tid < CANDIDATE_BUFFER_SIZE) ? CANDIDATE_DISTANCE[tid] : FLT_MAX;
        idx[0]  = (tid < CANDIDATE_BUFFER_SIZE) ? CANDIDATE_INDEX[tid]    : MAX_INDEX;
        topk.merge(dist, idx);
    }

    // return the top K result (smallest K at tail of descending buffer)
    for (int i = tid; i < K; i += blockDim.x) {
        d_results[query_id * K + i] = TOP_K_INDEX[TOPK_BUFFER_SIZE - K + i] & ~0x80000000u;
    }

    if (tid == 0) {
        if (GPU_SEARCH_STATS) {
            stats[GPU_SEARCH_STAT_DIST_CMPS] = counter[0];
            stats[GPU_SEARCH_STAT_DIST_EST] = 0;
            for (int i = 0; i < GPU_SEARCH_STAT_COUNT; i++) {
                atomicAdd(&d_total_stats[i], gpu_long_t(stats[i]));
            }
        }
        for (int i = 0; i < GPU_TIMING_COUNT; i++) {
            atomicAdd(&d_total_timing[i], timing[i]);
        }
    }
}
