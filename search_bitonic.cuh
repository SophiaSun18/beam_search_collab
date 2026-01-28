#pragma once
#include "utils.cuh"
#include <cuda_runtime.h>
#define BEAM_SIZE       500
#define INTERNAL_TOPK   500
#define SEARCH_WIDTH    4      // 4 nodes * 32 degree = 128 neighbors < 256 buffer
#define BITLEN          12
#define MAX_ITERATIONS  2000
#define HASH_SIZE       4096
#define USE_SEED        false
#define PRUNE_CONFIG    false
#define CANDIDATE_BUFFER_SIZE 64

static __host__ inline uint32_t calculate_shared_mem_size(int dim, int topk, int max_deg, 
                                                           int search_width, int bitlen) {
    return 0;
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
    const int qid = blockIdx.x;
    if (qid >= nq) return;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;  // 256/32 = 8 warps
    const int max_degree = gg.get_max_degree();
    const int beam_size = BEAM_SIZE;
    constexpr INDEX_T kIndexMsbMask = 0x80000000;
    __shared__ INDEX_T s_frontier_idx[BEAM_SIZE];
    __shared__ DISTANCE_T s_frontier_dist[BEAM_SIZE];
    __shared__ int s_frontier_size;
    __shared__ int s_dc_count;
    __shared__ INDEX_T s_hash[HASH_SIZE];
    __shared__ INDEX_T s_neighbor_idx[BEAM_SIZE];
    __shared__ DISTANCE_T s_neighbor_dist[BEAM_SIZE];
    __shared__ int s_num_neighbors;
    __shared__ INDEX_T s_candidate_idx[CANDIDATE_BUFFER_SIZE];
    __shared__ DISTANCE_T s_candidate_dist[CANDIDATE_BUFFER_SIZE];
    __shared__ int s_candidate_count;
    __shared__ INDEX_T s_expand_pos[SEARCH_WIDTH];
    __shared__ int s_num_expanders;
    __shared__ int s_warp_neighbor_base[8];  // Base offset for each warp
    __shared__ int s_warp_candidate_base[8];
    
    const T* query = d_queries + qid * dim;
    hashtable_init(s_hash, BITLEN, 0);
    
    for (int i = tid; i < BEAM_SIZE; i += blockDim.x) {
        s_frontier_idx[i] = MAX_INDEX;
        s_frontier_dist[i] = FLT_MAX;
    }
    for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
        s_candidate_idx[i] = MAX_INDEX;
        s_candidate_dist[i] = FLT_MAX;
    }
    
    if (tid == 0) {
        s_frontier_size = 0;
        s_dc_count = 0;
        s_num_neighbors = 0;
        s_candidate_count = 0;
    }
    if (lane_id == 0) {
        s_warp_neighbor_base[warp_id] = 0;
        s_warp_candidate_base[warp_id] = 0;
    }
    __syncthreads();
    {
        int num_starts = min(K, BEAM_SIZE);
        
        for (int s = warp_id; s < num_starts; s += num_warps) {
            vid_t start_vid = d_starting_points[s % BEAM_SIZE];
            if (start_vid >= npoints) continue;
            uint32_t is_new = 0;
            if (lane_id == 0) {
                is_new = hashtable_insert(s_hash, BITLEN, start_vid);
            }
            is_new = __shfl_sync(FULL_MASK, is_new, 0);
            if (!is_new) continue;
            const T* svec = d_data + start_vid * dim;
            DISTANCE_T dist = warp_l2_distance(dim, query, svec);
            if (lane_id == 0) {
                int pos = atomicAdd(&s_frontier_size, 1);
                if (pos < beam_size) {
                    s_frontier_idx[pos] = start_vid;
                    s_frontier_dist[pos] = dist;
                    atomicAdd(&s_dc_count, 1);
                }
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        int n = min(s_frontier_size, beam_size);
        for (int i = 1; i < n; i++) {
            DISTANCE_T key_d = s_frontier_dist[i];
            INDEX_T key_i = s_frontier_idx[i];
            int j = i - 1;
            while (j >= 0 && s_frontier_dist[j] > key_d) {
                s_frontier_dist[j + 1] = s_frontier_dist[j];
                s_frontier_idx[j + 1] = s_frontier_idx[j];
                j--;
            }
            s_frontier_dist[j + 1] = key_d;
            s_frontier_idx[j + 1] = key_i;
        }
    }
    
    __syncthreads();
    
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        if (iter > 0 && (iter % SMALL_HASH_RESET_INTERVAL) == 0) {
            hashtable_init(s_hash, BITLEN, 0);
            __syncthreads();
            hashtable_restore(s_hash, BITLEN, s_frontier_idx, min(s_frontier_size, beam_size), 0);
            __syncthreads();
        }
        if (warp_id == 0) {
            int num_exp = pick_expanders(
                SEARCH_WIDTH, 
                s_expand_pos, 
                min(s_frontier_size, beam_size), 
                s_frontier_idx
            );
            if (lane_id == 0) {
                s_num_expanders = num_exp;
            }
        }
        __syncthreads();
        if (s_num_expanders == 0) break;
        if (tid == 0) s_num_neighbors = 0;
        __syncthreads();
        for (int e = 0; e < s_num_expanders; e++) {
            INDEX_T expand_pos = s_expand_pos[e];
            if (expand_pos >= beam_size) continue;
            vid_t current = s_frontier_idx[expand_pos] & ~kIndexMsbMask;
            if (current >= npoints) continue;
            vid_t* neighbors = gg.N(current);
            for (int base = 0; base < max_degree; base += 32) {
                int n = base + lane_id;
                vid_t neighbor = (n < max_degree) ? neighbors[n] : UINT_MAX;
                bool valid = (n < max_degree) && (neighbor < npoints);
                uint32_t is_new = 0;
                if (valid) {
                    is_new = hashtable_insert(s_hash, BITLEN, neighbor);
                }
                uint32_t ballot = __ballot_sync(FULL_MASK, is_new);
                int warp_count = __popc(ballot);
                int base_offset;
                if (lane_id == 0 && warp_count > 0) {
                    base_offset = atomicAdd(&s_num_neighbors, warp_count);
                }
                base_offset = __shfl_sync(FULL_MASK, base_offset, 0);
                if (is_new) {
                    int my_offset = __popc(ballot & ((1u << lane_id) - 1));
                    int slot = base_offset + my_offset;
                    if (slot < 256) {
                        s_neighbor_idx[slot] = neighbor;
                    }
                }
            }
        }
        __syncthreads();
        int num_neighbors = min(s_num_neighbors, 256);
        if (num_neighbors == 0) continue;
        for (int n = warp_id; n < num_neighbors; n += num_warps) {
            vid_t neighbor = s_neighbor_idx[n];
            const T* nvec = d_data + neighbor * dim;
            DISTANCE_T dist = warp_l2_distance(dim, query, nvec);
            if (lane_id == 0) {
                s_neighbor_dist[n] = dist;
                atomicAdd(&s_dc_count, 1);
            }
        }
        __syncthreads();
        DISTANCE_T cutoff = (s_frontier_size >= beam_size) ? 
                           s_frontier_dist[beam_size - 1] : FLT_MAX;
        for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
            s_candidate_idx[i] = MAX_INDEX;
            s_candidate_dist[i] = FLT_MAX;
        }
        if (tid == 0) s_candidate_count = 0;
        __syncthreads();
        for (int base = 0; base < num_neighbors; base += 32) {
            int n = base + lane_id;
            bool valid = (n < num_neighbors);
            DISTANCE_T dist = valid ? s_neighbor_dist[n] : FLT_MAX;
            bool keep = valid && (dist < cutoff || s_frontier_size < beam_size);
            uint32_t ballot = __ballot_sync(FULL_MASK, keep);
            int warp_count = __popc(ballot);
            int base_offset;
            if (lane_id == 0 && warp_count > 0) {
                base_offset = atomicAdd(&s_candidate_count, warp_count);
            }
            base_offset = __shfl_sync(FULL_MASK, base_offset, 0);
            if (keep) {
                int my_offset = __popc(ballot & ((1u << lane_id) - 1));
                int slot = base_offset + my_offset;
                if (slot < CANDIDATE_BUFFER_SIZE) {
                    s_candidate_idx[slot] = s_neighbor_idx[n];
                    s_candidate_dist[slot] = dist;
                }
            }
        }
        __syncthreads();
        if (s_candidate_count == 0) continue;
        if (warp_id == 0) {
            candidate_by_bitonic_sort<2, 4>(s_candidate_idx, s_candidate_dist, CANDIDATE_BUFFER_SIZE);
        }
        
        __syncthreads();

        {
            int nf = min(s_frontier_size, beam_size);
            int nc = min(s_candidate_count, CANDIDATE_BUFFER_SIZE);
            int total = nf + nc;
            int output_size = min(total, beam_size);
            for (int out_pos = tid; out_pos < output_size; out_pos += blockDim.x) {
                int lo = max(0, out_pos - nf);
                int hi = min(nc, out_pos + 1);
                while (lo < hi) {
                    int cand_idx = (lo + hi) / 2;
                    int front_idx = out_pos - cand_idx;
                    DISTANCE_T cand_val = (cand_idx > 0) ? s_candidate_dist[cand_idx - 1] : -FLT_MAX;
                    DISTANCE_T front_val = (front_idx < nf) ? s_frontier_dist[front_idx] : FLT_MAX;
                    if (cand_val <= front_val) {
                        lo = cand_idx + 1;
                    } else {
                        hi = cand_idx;
                    }
                }
                int cand_idx = lo;
                int front_idx = out_pos - cand_idx;
                bool cand_valid = (cand_idx < nc);
                bool front_valid = (front_idx >= 0 && front_idx < nf);
                DISTANCE_T cand_dist = cand_valid ? s_candidate_dist[cand_idx] : FLT_MAX;
                DISTANCE_T front_dist = front_valid ? s_frontier_dist[front_idx] : FLT_MAX;
                if (cand_valid && cand_dist <= front_dist) {
                    s_neighbor_dist[out_pos] = cand_dist;
                    s_neighbor_idx[out_pos] = s_candidate_idx[cand_idx];
                } else if (front_valid) {
                    s_neighbor_dist[out_pos] = front_dist;
                    s_neighbor_idx[out_pos] = s_frontier_idx[front_idx];
                } else {
                    s_neighbor_dist[out_pos] = FLT_MAX;
                    s_neighbor_idx[out_pos] = MAX_INDEX;
                }
            }
            __syncthreads();
            for (int i = tid; i < output_size; i += blockDim.x) {
                s_frontier_idx[i] = s_neighbor_idx[i];
                s_frontier_dist[i] = s_neighbor_dist[i];
            }
            if (tid == 0) {
                s_frontier_size = output_size;
            }
        }
        
        __syncthreads();
    }
    for (int i = tid; i < K; i += blockDim.x) {
        if (i < s_frontier_size) {
            d_results[qid * K + i] = s_frontier_idx[i] & ~kIndexMsbMask;
        } else {
            d_results[qid * K + i] = 0;
        }
    }
    if (tid == 0) {
        atomicAdd(d_total_count_dc, (gpu_long_t)s_dc_count);
    }
}
