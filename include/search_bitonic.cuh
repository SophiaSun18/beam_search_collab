#pragma once
#include "utils.cuh"
#include <cuda_runtime.h>
#define BEAM_SIZE       100
#define INTERNAL_TOPK   128
#define SEARCH_WIDTH    2      // 4 nodes * 32 degree = 128 neighbors < 256 buffer
#define BITLEN          12
#define MAX_ITERATIONS  2000
#define MAX_DEGREE      32
#define HASH_SIZE       MAX_ITERATIONS * SEARCH_WIDTH * MAX_DEGREE * 2
#define USE_SEED        false
#define PRUNE_CONFIG    false
#define CANDIDATE_BUFFER_SIZE 64

static __host__ inline uint32_t calculate_shared_mem_size(int dim, int topk, int max_deg, 
                                                           int search_width, int bitlen) {
    size_t size = 0;
    size += topk * (sizeof(INDEX_T) + sizeof(DISTANCE_T));  // top k buffer
    size += (search_width * max_deg) * (sizeof(INDEX_T) + sizeof(DISTANCE_T));  // candidate buffer
    size += hashtable_getsize(BITLEN) * sizeof(INDEX_T);    // hash table
    size += SEARCH_WIDTH * sizeof(INDEX_T);                 // next expanders
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
    vid_t* d_starting_points,
    vid_t* d_seed_map,
    uint32_t* d_sign_bit,
    GraphGPU<vid_t> gg
) {
    // basic information about this block and thread
    const int query_id = blockIdx.x;
    if (query_id >= nq) return;
    const T* query = d_queries + query_id * dim;
    const vid_t maxDegree = gg.get_max_degree();

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // initialize the shared memory
    extern __shared__ char shared_memory[];
    INDEX_T* ALL_INDEX = (INDEX_T*)shared_memory;
    INDEX_T* TOP_K_INDEX      = ALL_INDEX;
    INDEX_T* CANDIDATE_INDEX  = ALL_INDEX + INTERNAL_TOPK;

    DISTANCE_T* ALL_DISTANCE = (DISTANCE_T*)(ALL_INDEX + INTERNAL_TOPK + CANDIDATE_BUFFER_SIZE);
    DISTANCE_T* TOP_K_DISTANCE      = ALL_DISTANCE;
    DISTANCE_T* CANDIDATE_DISTANCE  = ALL_DISTANCE + INTERNAL_TOPK;

    INDEX_T* HASH_TABLE = (INDEX_T*)(ALL_DISTANCE + INTERNAL_TOPK + CANDIDATE_BUFFER_SIZE);
    INDEX_T* NEXT_EXPANDERS = (INDEX_T*)(HASH_TABLE + hashtable_getsize(BITLEN));

    // initialize the block-wise total distance computation count
    __shared__ gpu_long_t block_dist_comp;
    if (tid == 0) block_dist_comp = 0;
    __syncthreads();
    gpu_long_t warp_dist_comp = 0;

    // initialize the hash table to record visited (computed) nodes
    hashtable_init(HASH_TABLE, BITLEN);
    
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
            warp_dist_comp++;
            hashtable_insert(HASH_TABLE, BITLEN, CANDIDATE_INDEX[i]);
            CANDIDATE_DISTANCE[i] = dist;
        }
    }
    __syncthreads();

    // use warp level sort to find the actual closest top K elements and update the topK list
    // constexpr uint32_t N_1 = (CANDIDATE_BUFFER_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    // constexpr uint32_t N_2 = (INTERNAL_TOPK + WARP_SIZE - 1) / WARP_SIZE;
    // TODO: currently only support CANDIDATE_BUFFER_SIZE <= 64, INTERNAL_TOPK <= 128
    topk_by_bitonic_sort<2, 4>(ALL_INDEX, ALL_DISTANCE, (uint32_t)INTERNAL_TOPK + CANDIDATE_BUFFER_SIZE,
        (uint32_t)CANDIDATE_BUFFER_SIZE, (uint32_t)INTERNAL_TOPK, true);
    __syncthreads();

    // cpu end condition: either entire topK expanded, or reach max expand limit
    // entire top K expanded: no more result found in pick expander
    // max expand limit: keep track of the expanded node count
    __shared__ uint32_t keep_expanding;
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        // forgettable hash table management according to the CARGA paper
        if (iter > 0 && iter % SMALL_HASH_RESET_INTERVAL == 0) {
            hashtable_init(HASH_TABLE, BITLEN);
            __syncthreads();
            hashtable_restore(HASH_TABLE, BITLEN, TOP_K_INDEX, INTERNAL_TOPK);
            __syncthreads();
        }

        // warp 0 pick the top-N unexpanded nodes and store in shared memory
        // this already marks the expanded nodes MSB to 1, no need to mark again
        if (warp_id == 0) keep_expanding = pick_expanders(SEARCH_WIDTH, NEXT_EXPANDERS, INTERNAL_TOPK, TOP_K_INDEX);
        __syncthreads();
        if (!keep_expanding) break;

        // each working warp take an expander and collect all neighbors into candidate list
        if (warp_id < SEARCH_WIDTH) {
            INDEX_T current_node = TOP_K_INDEX[NEXT_EXPANDERS[warp_id]] & 0x7fffffff;
            vid_t* neighbors = gg.N(current_node);

            for (int i = lane_id; i < maxDegree; i+=WARP_SIZE) {    // handles max degree > warp size
                CANDIDATE_INDEX[warp_id * maxDegree + i] = neighbors[i];
            }
        }
        __syncthreads();

        // all warps work together to compute candidate distance
        for (int i = warp_id; i < CANDIDATE_BUFFER_SIZE; i += WARPS_PER_BLOCK) {
            if (i >= CANDIDATE_BUFFER_SIZE) continue;
            INDEX_T n = CANDIDATE_INDEX[i];

            // lane 0 check if the neighbor has been visited, and broadcast to all lanes
            // if current node already in hash table, then skip and move on to next node
            uint32_t computed;
            if (lane_id == 0) computed = hashtable_insert(HASH_TABLE, BITLEN, n);
            computed = SHFL(computed, 0);
            if (computed) continue;

            // if current node is new in hash table, compute and store its distance
            DISTANCE_T dist = warp_l2_distance(dim, query, d_data + n * dim);
            if (lane_id == 0) {
                warp_dist_comp++;
                CANDIDATE_DISTANCE[i] = dist;
            }
        }
        __syncthreads();

        topk_by_bitonic_sort<2, 4>(ALL_INDEX, ALL_DISTANCE, (uint32_t)INTERNAL_TOPK + CANDIDATE_BUFFER_SIZE,
            (uint32_t)CANDIDATE_BUFFER_SIZE, (uint32_t)INTERNAL_TOPK, false);
        __syncthreads();
    }

    // return the top K result
    for (int i = tid; i < K; i += blockDim.x) {
        d_results[query_id * K + i] = TOP_K_INDEX[i] & ~0x80000000;
    }

    if (lane_id == 0) atomicAdd(&block_dist_comp, warp_dist_comp);
    __syncthreads();
    if (tid == 0) atomicAdd(d_total_count_dc, block_dist_comp);
}