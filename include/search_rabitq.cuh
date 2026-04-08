#pragma once
#include "utils.cuh"
#include "distance.cuh"
#include <cuda_runtime.h>
#define SEARCH_WIDTH    1     // MAX: 2 nodes * 32 degree = 64 neighbors = CANDIDATE_BUFFER_SIZE
#define BITLEN          10
#define MAX_ITERATIONS  1024
#define TEAM_SIZE       8
#define HASH_TABLE_CONFIG   1
#define INTERNAL_TOPK   128
#define CANDIDATE_BUFFER_SIZE 64
#define RESULT_BUFFER_SIZE INTERNAL_TOPK + CANDIDATE_BUFFER_SIZE

enum GpuRabitqStatIndex : int {
    GPU_RABITQ_STAT_EXPANDED = 0,
    GPU_RABITQ_STAT_NEIGHBORS_SCANNED = 1,
    GPU_RABITQ_STAT_CANDIDATES_INSERTED = 2,
    GPU_RABITQ_STAT_DUPLICATE_VISITED_PRUNES = 3,
    GPU_RABITQ_STAT_DIST_CMPS = 4,
    GPU_RABITQ_STAT_DIST_EST = 5,
    GPU_RABITQ_STAT_ITERATIONS = 6,
    GPU_RABITQ_STAT_COUNT = 7
};

static __host__ inline uint32_t calculate_shared_mem_size(int dim, int topk, int max_deg, 
                                                           int search_width, int bitlen) {
    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
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
    size += 2 * sizeof(uint32_t);                           // counter and hops
    size += GPU_RABITQ_STAT_COUNT * sizeof(uint32_t);       // search stats
    printf("shared memory size: %ld\n", size);
    return size;
}

static __global__ void QuantizedBeamSearch(
    int K, int nq, int dim, int max_degree, size_t npoints,
    const float* __restrict__ d_queries,
    const float* __restrict__ d_qg_data,
    const float* __restrict__ d_qg_signs,
    vidType* __restrict__ d_results,
    gpu_long_t* d_total_stats,
    vidType entry_point,
    size_t row_offset,
    size_t neighbor_offset,
    size_t code_offset,
    size_t factor_offset
) {
    // basic information about this block and thread
    const int query_id = blockIdx.x;
    if (query_id >= nq) return;
    const float* query = d_queries + query_id * dim;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));

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

    uint32_t* counter = reinterpret_cast<uint32_t*>(sum_q + 1);
    uint32_t* hops = (uint32_t*)(counter + 1);
    uint32_t* stats = hops + 1;

    __shared__ QueryFactors qf;

    // initialize the block-wise total distance computation count
    __shared__ uint32_t keep_expanding;
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
        for (int i = 0; i < GPU_RABITQ_STAT_COUNT; i++) {
            stats[i] = 0;
        }
    }

    // initialize the hash table to record visited (computed) nodes
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

    // only the first slot in candidate list is actually initialized with valid entry point
    if (warp_id == 0) {
        CANDIDATE_INDEX[0] = entry_point;
        CANDIDATE_DISTANCE[0] = warp_l2_distance(dim, query, d_qg_data + static_cast<size_t>(entry_point) * row_offset);
        hashtable_insert(HASH_TABLE, BITLEN, entry_point);
    }
    __syncthreads();

    // query preparation, quantize the query
    query_prepare_gpu(QUERY_BUFFER, qf, d_qg_signs, dim, padded_dim);
    __syncthreads();
    
    // loop end condition: either entire topK expanded, or reach max iteration limit
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        // forgettable hash table management according to the CARGA paper
        if (HASH_TABLE_CONFIG) {
            if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
                hashtable_init(HASH_TABLE, BITLEN);
                __syncthreads();
                hashtable_restore(HASH_TABLE, BITLEN, TOP_K_INDEX, INTERNAL_TOPK);
            }
            __syncthreads();
        }
      
        // use warp level sort to find the actual closest top K elements and update the topK list
        topk_by_bitonic_sort<2, INTERNAL_TOPK / WARP_SIZE>(ALL_INDEX, ALL_DISTANCE, RESULT_BUFFER_SIZE,
            CANDIDATE_BUFFER_SIZE, INTERNAL_TOPK, (iter == 0));
        __syncthreads();

        // warp 0 pick the top-N unexpanded nodes and store in shared memory
        // this already marks the expanded nodes MSB to 1, no need to mark again
        if (warp_id == 0) {
            keep_expanding = pick_expanders(SEARCH_WIDTH, PARENT_LIST, INTERNAL_TOPK, TOP_K_INDEX);
        }
        __syncthreads();
        if (!keep_expanding) {
            if (tid == 0) {
                hops[0] = iter;
            }
            break;
        }
        if (tid == 0) {
            stats[GPU_RABITQ_STAT_ITERATIONS]++;
        }

        // collect the neighbors of the expanding node and add into candidate list if not visited before
        // make sure to mark the invalid neighbor using dummy large values
        INDEX_T current_node = TOP_K_INDEX[PARENT_LIST[0]] & 0x7fffffff;
        const float* current_row = d_qg_data + static_cast<size_t>(current_node) * row_offset;
        const vidType* neighbors = reinterpret_cast<const vidType*>(current_row + neighbor_offset);
        DISTANCE_T exact_dist = warp_l2_distance(dim, query, current_row);
        if (tid == 0) {
            stats[GPU_RABITQ_STAT_EXPANDED]++;
            stats[GPU_RABITQ_STAT_DIST_CMPS]++;
            stats[GPU_RABITQ_STAT_NEIGHBORS_SCANNED] += static_cast<uint32_t>(max_degree);
        }

        for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
            INDEX_T child_id = (i < max_degree) ? neighbors[i] : MAX_INDEX;      // only in-bound threads take from the neighbor list

            if (child_id != MAX_INDEX) {    // check if visited already
                if (hashtable_insert(HASH_TABLE, BITLEN, child_id) == 0) {
                    child_id = MAX_INDEX;
                    CANDIDATE_DISTANCE[i] = FLT_MAX;
                    atomicAdd(&stats[GPU_RABITQ_STAT_DUPLICATE_VISITED_PRUNES], 1u);
                } else {
                    CANDIDATE_DISTANCE[i] = 0;
                    atomicAdd(&stats[GPU_RABITQ_STAT_CANDIDATES_INSERTED], 1u);
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
            DISTANCE_T est_dist = estimated_distance_gpu(qf, current_node, i, exact_dist, max_degree, padded_dim,
                d_qg_data, row_offset, code_offset, factor_offset);
            if (lane_id == 0) {
                atomicAdd(&counter[0], 1);
                atomicAdd(&stats[GPU_RABITQ_STAT_DIST_EST], 1u);
                CANDIDATE_DISTANCE[i] = est_dist;
            }
        }
        __syncthreads();
    }

    // final sort before writeback
    topk_by_bitonic_sort<2, INTERNAL_TOPK / WARP_SIZE>(ALL_INDEX, ALL_DISTANCE, RESULT_BUFFER_SIZE,
        CANDIDATE_BUFFER_SIZE, INTERNAL_TOPK, false);
    __syncthreads();

    // return the top K result
    for (int i = tid; i < K; i += blockDim.x) {
        d_results[query_id * K + i] = TOP_K_INDEX[i] & ~0x80000000;
    }
    if (tid == 0) {
        stats[GPU_RABITQ_STAT_DIST_EST] = counter[0];
        stats[GPU_RABITQ_STAT_ITERATIONS] = hops[0];
        for (int i = 0; i < GPU_RABITQ_STAT_COUNT; i++) {
            atomicAdd(&d_total_stats[i], gpu_long_t(stats[i]));
        }
    }

    (void)d_qg_signs;
    (void)npoints;
    (void)code_offset;
    (void)factor_offset;
}
