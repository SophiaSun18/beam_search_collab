#pragma once
#include "utils.cuh"
#include "distance.cuh"
#include <cuda_runtime.h>
// #define BEAM_SIZE       128
#define INTERNAL_TOPK   128
#define SEARCH_WIDTH    1     // MAX: 2 nodes * 32 degree = 64 neighbors = CANDIDATE_BUFFER_SIZE
#define BITLEN          10
#define MAX_ITERATIONS  1024
// #define MAX_DEGREE      64
#define TEAM_SIZE       8
// #define HASH_SIZE       MAX_ITERATIONS * SEARCH_WIDTH * MAX_DEGREE * 2
#define USE_SEED        0
#define PRUNE_CONFIG    0
#define THRESHOLD_CONFIG    0
#define HASH_TABLE_CONFIG   1
// #define CANDIDATE_BUFFER_SIZE SEARCH_WIDTH * MAX_DEGREE
#define CANDIDATE_BUFFER_SIZE 64
#define RESULT_BUFFER_SIZE INTERNAL_TOPK + CANDIDATE_BUFFER_SIZE

static __host__ inline uint32_t calculate_shared_mem_size(int dim, int topk, int max_deg, 
                                                           int search_width, int bitlen) {
    size_t size = 0;    
    size += hashtable_getsize(BITLEN) * sizeof(INDEX_T);    // hash table
    size += RESULT_BUFFER_SIZE * sizeof(INDEX_T);           // top k and candidate index
    size += RESULT_BUFFER_SIZE * sizeof(DISTANCE_T);        // top k and candidate distance
    size += SEARCH_WIDTH * sizeof(INDEX_T);                 // next expanders / parent buffer
    size += dim * 4;                                        // query buffer size
    size += 2 * 4;                                          // terminate flag and counter
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
static __global__ void BeamSearch(
    int K, int nq, int dim, size_t npoints,
    const T* __restrict__ d_queries,
    const T* __restrict__ d_data,
    vid_t* __restrict__ d_results,
    gpu_long_t* d_total_count_dc,
    gpu_long_t* d_total_hops,
    vid_t* d_starting_points,
    vid_t* d_seed_map,
    uint32_t* d_sign_bit,
    GraphGPU<vid_t> gg
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
    INDEX_T* ALL_INDEX = (INDEX_T*)shared_memory;
    INDEX_T* TOP_K_INDEX      = ALL_INDEX;
    INDEX_T* CANDIDATE_INDEX  = ALL_INDEX + INTERNAL_TOPK;

    DISTANCE_T* ALL_DISTANCE = (DISTANCE_T*)(ALL_INDEX + RESULT_BUFFER_SIZE);
    DISTANCE_T* TOP_K_DISTANCE      = ALL_DISTANCE;
    DISTANCE_T* CANDIDATE_DISTANCE  = ALL_DISTANCE + INTERNAL_TOPK;

    INDEX_T* HASH_TABLE = (INDEX_T*)(ALL_DISTANCE + RESULT_BUFFER_SIZE);
    INDEX_T* PARENT_LIST = (INDEX_T*)(HASH_TABLE + hashtable_getsize(BITLEN));
    DATA_T* QUERY_BUFFER = (DATA_T*)(PARENT_LIST + SEARCH_WIDTH);
    uint32_t* counter = (uint32_t*)(QUERY_BUFFER + dim);
    uint32_t* hops = (uint32_t*)(counter + 1);

    // initialize the block-wise total distance computation count
    __shared__ uint32_t keep_expanding;
    if (tid == 0) {
        counter[0] = 0;
        hops[0] = 0;
    }

    // initialize the hash table to record visited (computed) nodes
    if (HASH_TABLE_CONFIG) hashtable_init(HASH_TABLE, BITLEN);

    // load query into shared memory
    for (int i = tid; i < dim; i += blockDim.x) {
        if (i < dim) QUERY_BUFFER[i] = query[i];
    }
    
    // load the top k list with dummy initial values
    for (int i = tid; i < INTERNAL_TOPK; i += blockDim.x) {
        TOP_K_INDEX[i] = MAX_INDEX;
        TOP_K_DISTANCE[i] = FLT_MAX;
    }

    // load the candidate buffer with starting points
    for (int i = warp_id; i < CANDIDATE_BUFFER_SIZE; i += WARPS_PER_BLOCK) {
        CANDIDATE_INDEX[i] = d_starting_points[i];
        DISTANCE_T dist = warp_l2_distance(dim, query, d_data + CANDIDATE_INDEX[i] * dim);
        if (lane_id == 0) {     // warp computes, only lane 0 writes
            atomicAdd(&counter[0], 1);
            hashtable_insert(HASH_TABLE, BITLEN, CANDIDATE_INDEX[i]);
            CANDIDATE_DISTANCE[i] = dist;
        }
    }
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

        // collect the neighbors of the expanding node and add into candidate list if not visited before
        // make sure to mark the invalid neighbor using dummy large values
        INDEX_T current_node = TOP_K_INDEX[PARENT_LIST[0]] & 0x7fffffff;
        vid_t* neighbors = gg.N(current_node);

        for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
            INDEX_T child_id = (i < maxDegree) ? neighbors[i] : MAX_INDEX;      // only in-bound threads take from the neighbor list

            if (child_id != MAX_INDEX) {    // check if visited already
                if (hashtable_insert(HASH_TABLE, BITLEN, child_id) == 0) {
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
            DISTANCE_T dist = warp_l2_distance(dim, query, d_data + CANDIDATE_INDEX[i] * dim);
            if (lane_id == 0) {
                atomicAdd(&counter[0], 1);
                CANDIDATE_DISTANCE[i] = dist;
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
        atomicAdd(d_total_count_dc, gpu_long_t(counter[0]));
        atomicAdd(d_total_hops, gpu_long_t(hops[0]));
    }
}