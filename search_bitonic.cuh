#pragma once
#include "utils.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cstring>

#ifndef BEAM_SIZE
#define BEAM_SIZE       500
#endif
#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK   500
#endif
#ifndef SEARCH_WIDTH
#define SEARCH_WIDTH    4      
#endif
#ifndef BITLEN
#define BITLEN          12
#endif
#ifndef HASH_SIZE
#define HASH_SIZE       (1u << (BITLEN))
#endif
#ifndef MAX_ITERATIONS
#define MAX_ITERATIONS  2000
#endif
#ifndef USE_SEED
#define USE_SEED        false
#endif
#ifndef PRUNE_CONFIG
#define PRUNE_CONFIG    false
#endif
#define CANDIDATE_BUFFER_SIZE 64
#ifndef ENABLE_DGS
#define ENABLE_DGS 0
#endif
#ifndef DGS_TOPN
#define DGS_TOPN 8
#endif
#ifndef DGS_TOPN_MAX
#define DGS_TOPN_MAX 64
#endif
#ifndef USE_SUBWARP_L2_128
#define USE_SUBWARP_L2_128 0
#endif
#ifndef DGS_CAP_EXPAND_WIDTH_BY_CANDIDATE
#define DGS_CAP_EXPAND_WIDTH_BY_CANDIDATE 1
#endif
#ifndef HASH_WARP_DEDUP
#define HASH_WARP_DEDUP 1
#endif
#ifndef PICK_EXPANDERS_BLOCK_WIDE
#define PICK_EXPANDERS_BLOCK_WIDE 1
#endif
#ifndef DGS_SEARCH_WIDTH
#define DGS_SEARCH_WIDTH 8
#endif
#ifndef EXPAND_WIDTH_MAX
#define EXPAND_WIDTH_MAX ((DGS_SEARCH_WIDTH) > (SEARCH_WIDTH) ? (DGS_SEARCH_WIDTH) : (SEARCH_WIDTH))
#endif
#ifndef DGS_EST_ITERS
#define DGS_EST_ITERS (BEAM_SIZE / DGS_SEARCH_WIDTH)
#endif
#ifndef DGS_COOL_RATIO
#define DGS_COOL_RATIO 0.30f
#endif
#ifndef DGS_MAX_DEG
#define DGS_MAX_DEG 256
#endif
#ifndef DGS_MAX_WORDS
#define DGS_MAX_WORDS 16
#endif
#ifndef DGS_DEBUG_PRINT
#define DGS_DEBUG_PRINT 0
#endif
static __device__ __forceinline__ void warp_bitonic_sort_u32_asc(uint32_t& key, uint32_t& val) {
    const unsigned lane = threadIdx.x & 31;
    #pragma unroll
    for (uint32_t k = 2; k <= 32; k <<= 1) {
        #pragma unroll
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            const uint32_t other_key = __shfl_xor_sync(FULL_MASK, key, j);
            const uint32_t other_val = __shfl_xor_sync(FULL_MASK, val, j);
            const bool up = ((lane & k) == 0);       // sort direction for this stage
            const bool low = ((lane & j) == 0);      // lower lane in the pair
            if (up) {
                if (low) {
                    if (key > other_key) { key = other_key; val = other_val; }
                } else {
                    if (key < other_key) { key = other_key; val = other_val; }
                }
            } else {
                if (low) {
                    if (key < other_key) { key = other_key; val = other_val; }
                } else {
                    if (key > other_key) { key = other_key; val = other_val; }
                }
            }
        }
    }
}
#ifndef ENABLE_SQ8
#define ENABLE_SQ8 0
#endif
#ifndef SQ8_PRUNE_SLACK
#define SQ8_PRUNE_SLACK 1.0f
#endif
#if ENABLE_SQ8
// Per-dimension scale and scale^2 for dim=128 (copied from host).
// Keep static to avoid cross-TU linkage issues when included in multiple .cu files.
static __device__ __constant__ float g_sq8_scale[128];
static __device__ __constant__ float g_sq8_scale2[128];

static __device__ __forceinline__ float warp_l2_distance_sq8_128(
    const int8_t* __restrict__ q8,
    const int8_t* __restrict__ x8) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int base = lane << 2;  // lane * 4 dims
    const char4 qv = reinterpret_cast<const char4*>(q8)[lane];
    const char4 xv = reinterpret_cast<const char4*>(x8)[lane];
    const int dq0 = (int)qv.x - (int)xv.x;
    const int dq1 = (int)qv.y - (int)xv.y;
    const int dq2 = (int)qv.z - (int)xv.z;
    const int dq3 = (int)qv.w - (int)xv.w;
    float sum = 0.0f;
    sum = fmaf((float)(dq0 * dq0), g_sq8_scale2[base + 0], sum);
    sum = fmaf((float)(dq1 * dq1), g_sq8_scale2[base + 1], sum);
    sum = fmaf((float)(dq2 * dq2), g_sq8_scale2[base + 2], sum);
    sum = fmaf((float)(dq3 * dq3), g_sq8_scale2[base + 3], sum);
    sum += __shfl_down_sync(FULL_MASK, sum, 16);
    sum += __shfl_down_sync(FULL_MASK, sum, 8);
    sum += __shfl_down_sync(FULL_MASK, sum, 4);
    sum += __shfl_down_sync(FULL_MASK, sum, 2);
    sum += __shfl_down_sync(FULL_MASK, sum, 1);
    return __shfl_sync(FULL_MASK, sum, 0);
}
#endif
#ifndef ENABLE_PARTIAL_L2_128
#define ENABLE_PARTIAL_L2_128 0
#endif
#ifndef PARTIAL_L2_128_LANES
#define PARTIAL_L2_128_LANES 24  // 24 lanes * 4 = 96 dims
#endif
template <int VALID_LANES>
static __device__ __forceinline__ float warp_l2_distance_128_prefix(const float* __restrict__ a,
                                                                    const float* __restrict__ b) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    float sum = 0.0f;
    if (lane < VALID_LANES) {
        const float4 av = reinterpret_cast<const float4*>(a)[lane];
        const float4 bv = reinterpret_cast<const float4*>(b)[lane];
        const float dx0 = av.x - bv.x;
        const float dx1 = av.y - bv.y;
        const float dx2 = av.z - bv.z;
        const float dx3 = av.w - bv.w;
        sum = fmaf(dx0, dx0, fmaf(dx1, dx1, fmaf(dx2, dx2, dx3 * dx3)));
    }
    sum += __shfl_down_sync(FULL_MASK, sum, 16);
    sum += __shfl_down_sync(FULL_MASK, sum, 8);
    sum += __shfl_down_sync(FULL_MASK, sum, 4);
    sum += __shfl_down_sync(FULL_MASK, sum, 2);
    sum += __shfl_down_sync(FULL_MASK, sum, 1);
    return __shfl_sync(FULL_MASK, sum, 0);
}

template <int START_LANE>
static __device__ __forceinline__ float warp_l2_distance_128_suffix(const float* __restrict__ a,
                                                                    const float* __restrict__ b) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    float sum = 0.0f;
    if (lane >= START_LANE) {
        const float4 av = reinterpret_cast<const float4*>(a)[lane];
        const float4 bv = reinterpret_cast<const float4*>(b)[lane];
        const float dx0 = av.x - bv.x;
        const float dx1 = av.y - bv.y;
        const float dx2 = av.z - bv.z;
        const float dx3 = av.w - bv.w;
        sum = fmaf(dx0, dx0, fmaf(dx1, dx1, fmaf(dx2, dx2, dx3 * dx3)));
    }
    sum += __shfl_down_sync(FULL_MASK, sum, 16);
    sum += __shfl_down_sync(FULL_MASK, sum, 8);
    sum += __shfl_down_sync(FULL_MASK, sum, 4);
    sum += __shfl_down_sync(FULL_MASK, sum, 2);
    sum += __shfl_down_sync(FULL_MASK, sum, 1);
    return __shfl_sync(FULL_MASK, sum, 0);
}

static __host__ inline uint32_t calculate_shared_mem_size(int dim, int topk, int max_deg, 
                                                           int search_width, int bitlen) {
    return 0;
}

static __global__ void init_identity_map(size_t n, vid_t* map) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) map[idx] = idx;
}

static inline std::vector<uint32_t> load_sign_bit(const char* filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    std::streamoff bytes = f.tellg();
    if (bytes <= 0) return {};
    if ((bytes % static_cast<std::streamoff>(sizeof(uint32_t))) != 0) return {};
    f.seekg(0, std::ios::beg);
    std::vector<uint32_t> buf(static_cast<size_t>(bytes / sizeof(uint32_t)));
    f.read(reinterpret_cast<char*>(buf.data()), bytes);
    if (!f) return {};
    return buf;
}

static inline bool save_sign_bit(const char* filename, const uint32_t* data, size_t words) {
    std::ofstream f(filename, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(words * sizeof(uint32_t)));
    return static_cast<bool>(f);
}

template <typename T>
static __global__ void BeamSearch(
    int K, int beam_size, int nq, int dim, size_t npoints,
    const T* __restrict__ d_queries,
    const T* __restrict__ d_data,
    const int8_t* __restrict__ d_data_sq8,
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
    if (beam_size <= 0 || beam_size > BEAM_SIZE) return;
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
    __shared__ INDEX_T s_expand_pos[EXPAND_WIDTH_MAX];
    __shared__ int s_num_expanders;
#if ENABLE_DGS
    // Per-warp cached query-direction words (avoid repeated shfl broadcasts).
    __shared__ uint32_t s_dgs_qwords[8][DGS_MAX_WORDS];
#endif
#if ENABLE_SQ8
    __shared__ __align__(4) int8_t s_query_sq8[128];
#endif
    __shared__ int s_pick_counts[WARPS_PER_BLOCK];
    __shared__ int s_pick_num;
    __shared__ int s_pick_round_base;
    const T* query = d_queries + qid * dim;
#if ENABLE_DGS && DGS_DEBUG_PRINT
    if (qid == 0 && tid == 0) {
        printf("DGS kernel: ENABLE_DGS=1, DGS_TOPN=%d, DGS_COOL_RATIO=%f, d_sign_bit=%p\n",
               (int)DGS_TOPN, (double)DGS_COOL_RATIO, (void*)d_sign_bit);
    }
#endif
    hashtable_init(s_hash, BITLEN, 0);

#if ENABLE_SQ8
    if (dim == 128 && d_data_sq8 != nullptr) {
        for (int i = tid; i < 128; i += blockDim.x) {
            const float scale = g_sq8_scale[i];
            const float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
            int q = __float2int_rn(((const float*)query)[i] * inv);
            q = max(-127, min(127, q));
            s_query_sq8[i] = (int8_t)q;
        }
    }
#endif
    
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
    __syncthreads();
    {
        int num_starts = min(K, beam_size);
        
        for (int s = warp_id; s < num_starts; s += num_warps) {
            vid_t start_vid = d_starting_points[s % beam_size];
            if (start_vid >= npoints) continue;
            uint32_t is_new = 0;
            if (lane_id == 0) {
                is_new = hashtable_insert(s_hash, BITLEN, start_vid);
            }
            is_new = __shfl_sync(FULL_MASK, is_new, 0);
            if (!is_new) continue;
            const T* svec = d_data + start_vid * dim;
            DISTANCE_T dist = (dim == 128) ? warp_l2_distance_128(query, svec)
                                           : warp_l2_distance(dim, query, svec);
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
#if ENABLE_DGS
        const int cool_start = (int)((1.0f - (float)DGS_COOL_RATIO) * (float)DGS_EST_ITERS);
        const bool use_wide_expand = (d_sign_bit != nullptr) && (iter < cool_start) && (DGS_TOPN > 0);
        int expand_width = use_wide_expand ? (int)DGS_SEARCH_WIDTH : SEARCH_WIDTH;
#if DGS_CAP_EXPAND_WIDTH_BY_CANDIDATE
        if (use_wide_expand) {
            constexpr int kTopN = (int)DGS_TOPN;
            constexpr int kCand = (int)CANDIDATE_BUFFER_SIZE;
            constexpr int kSafe = (kTopN > 0) ? (kCand / kTopN) : 1;
            const int safe_width = (kSafe > 0) ? kSafe : 1;
            expand_width = min(expand_width, safe_width);
        }
#endif
#else
        const int expand_width = SEARCH_WIDTH;
#endif
#if PICK_EXPANDERS_BLOCK_WIDE
        if (tid == 0) s_pick_num = 0;
        __syncthreads();
        const int M = min(s_frontier_size, beam_size);
        int itopk_max = M;
        if (itopk_max % blockDim.x) itopk_max += blockDim.x - (itopk_max % blockDim.x);
        for (int base = 0; base < itopk_max; base += blockDim.x) {
            const int j = base + tid;
            INDEX_T index = 0;
            bool new_parent = false;
            if (j < M) {
                index = s_frontier_idx[j];
                new_parent = ((index & kIndexMsbMask) == 0);
            }
            const uint32_t ballot = __ballot_sync(FULL_MASK, new_parent);
            const int warp_count = __popc(ballot);
            if (lane_id == 0) s_pick_counts[warp_id] = warp_count;
            __syncthreads();
            if (tid == 0) {
                int total = 0;
                #pragma unroll
                for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                    if (w < num_warps) total += s_pick_counts[w];
                }
                s_pick_round_base = s_pick_num;
                s_pick_num += total;
            }
            __syncthreads();
            int warp_prefix = 0;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                if (w < warp_id) warp_prefix += s_pick_counts[w];
            }
            const uint32_t lane_mask = (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
            const int lane_prefix = __popc(ballot & lane_mask);
            const int pos = s_pick_round_base + warp_prefix + lane_prefix;
            if (new_parent && pos < expand_width) {
                s_expand_pos[pos] = j;
                // mark as expanded
                s_frontier_idx[j] |= kIndexMsbMask;
            }
            __syncthreads();
        }
        if (tid == 0) {
            s_num_expanders = min(s_pick_num, expand_width);
        }
        __syncthreads();
#else
        if (warp_id == 0) {
            int num_exp = pick_expanders(expand_width, s_expand_pos, min(s_frontier_size, beam_size), s_frontier_idx);
            if (lane_id == 0) s_num_expanders = num_exp;
        }
        __syncthreads();
#endif
        if (s_num_expanders == 0) break;
        if (tid == 0) s_num_neighbors = 0;
        __syncthreads();
#if ENABLE_DGS
        if (!use_wide_expand) {
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
#if HASH_WARP_DEDUP
                    const uint32_t active = __ballot_sync(FULL_MASK, valid);
                    const uint32_t same = __match_any_sync(active, (uint32_t)neighbor);
                    const int leader = __ffs(same) - 1;
                    if (valid && lane_id == leader) {
                        is_new = hashtable_insert(s_hash, BITLEN, neighbor);
                    }
#else
                    if (valid) {
                        is_new = hashtable_insert(s_hash, BITLEN, neighbor);
                    }
#endif
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
        } else {
            for (int e = warp_id; e < s_num_expanders && e < expand_width; e += num_warps) {
                INDEX_T expand_pos = s_expand_pos[e];
                if (expand_pos < beam_size) {
                    vid_t current = s_frontier_idx[expand_pos] & ~kIndexMsbMask;
                    if (current < npoints) {
                        vid_t* neighbors = gg.N(current);
                        const int deg = min(max_degree, 32);  // graph-d32 fast path

                        constexpr int cool_start = (int)((1.0f - (float)DGS_COOL_RATIO) * (float)DGS_EST_ITERS);
                        const int W = (dim + 31) >> 5;
                        const bool dgs_active = (d_sign_bit != nullptr) &&
                                                (iter < cool_start) &&
                                                (DGS_TOPN > 0) &&
                                                (DGS_TOPN < deg) &&
                                                (W > 0) && (W <= (int)DGS_MAX_WORDS) &&
                                                (deg == 32);
                        if (dgs_active && lane_id < W) {
                            const T* uvec = d_data + ((size_t)current * (size_t)dim);
                            const int w = lane_id;
                            uint32_t word = 0;
                            #pragma unroll
                            for (int b = 0; b < 32; b++) {
                                const int i = (w << 5) + b;
                                if (i >= dim) break;
                                const T diff = query[i] - uvec[i];
                                word |= ((diff >= (T)0) ? 1u : 0u) << b;
                            }
                            s_dgs_qwords[warp_id][w] = word;
                        }
                        if (dgs_active) __syncwarp();
                        const vid_t nb = (lane_id < deg) ? neighbors[lane_id] : UINT_MAX;
                        uint32_t val = nb;
                        bool keep = (lane_id < deg) && (nb < npoints);

                        if (dgs_active) {
                            uint32_t key0 = 0;  // larger is better
                            if (nb < npoints) {
                                const uint32_t* uv_bits =
                                    d_sign_bit + (((size_t)current * (size_t)max_degree + (size_t)lane_id) * (size_t)W);
                                int diff_cnt = 0;
                                if (W == 4) {
                                    const uint32_t qw0 = s_dgs_qwords[warp_id][0];
                                    const uint32_t qw1 = s_dgs_qwords[warp_id][1];
                                    const uint32_t qw2 = s_dgs_qwords[warp_id][2];
                                    const uint32_t qw3 = s_dgs_qwords[warp_id][3];
                                    const uint4 bits = *reinterpret_cast<const uint4*>(uv_bits);
                                    diff_cnt = __popc(bits.x ^ qw0) +
                                               __popc(bits.y ^ qw1) +
                                               __popc(bits.z ^ qw2) +
                                               __popc(bits.w ^ qw3);
                                } else if (W == 3) {
                                    const uint32_t qw0 = s_dgs_qwords[warp_id][0];
                                    const uint32_t qw1 = s_dgs_qwords[warp_id][1];
                                    const uint32_t qw2 = s_dgs_qwords[warp_id][2];
                                    diff_cnt = __popc(uv_bits[0] ^ qw0) +
                                               __popc(uv_bits[1] ^ qw1) +
                                               __popc(uv_bits[2] ^ qw2);
                                } else if (W == 2) {
                                    const uint32_t qw0 = s_dgs_qwords[warp_id][0];
                                    const uint32_t qw1 = s_dgs_qwords[warp_id][1];
                                    diff_cnt = __popc(uv_bits[0] ^ qw0) +
                                               __popc(uv_bits[1] ^ qw1);
                                } else if (W == 1) {
                                    const uint32_t qw0 = s_dgs_qwords[warp_id][0];
                                    diff_cnt = __popc(uv_bits[0] ^ qw0);
                                } else {
                                    #pragma unroll
                                    for (int w = 0; w < DGS_MAX_WORDS; w++) {
                                        if (w >= W) break;
                                        const uint32_t qw = s_dgs_qwords[warp_id][w];
                                        diff_cnt += __popc(uv_bits[w] ^ qw);
                                    }
                                }
                                const int match_cnt = dim - diff_cnt;
                                key0 = (static_cast<uint32_t>(match_cnt) << 16) | (0xFFFFu - (uint32_t)lane_id);
                            }
                            const int topn = min((int)DGS_TOPN, deg);
                            uint32_t key = ~key0;
                            warp_bitonic_sort_u32_asc(key, val);
                            keep = (val < npoints) && (lane_id < topn);
                        }

                        uint32_t is_new = 0;
#if HASH_WARP_DEDUP
                        const uint32_t active = __ballot_sync(FULL_MASK, keep);
                        const uint32_t same = __match_any_sync(active, val);
                        const int leader = __ffs(same) - 1;
                        if (keep && lane_id == leader) {
                            is_new = hashtable_insert(s_hash, BITLEN, val);
                        }
#else
                        if (keep) {
                            is_new = hashtable_insert(s_hash, BITLEN, val);
                        }
#endif
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
                            if (slot < 256) s_neighbor_idx[slot] = val;
                        }
                    }
                }
            }
        }
#else
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
#if HASH_WARP_DEDUP
                const uint32_t active = __ballot_sync(FULL_MASK, valid);
                const uint32_t same = __match_any_sync(active, (uint32_t)neighbor);
                const int leader = __ffs(same) - 1;
                if (valid && lane_id == leader) {
                    is_new = hashtable_insert(s_hash, BITLEN, neighbor);
                }
#else
                if (valid) {
                    is_new = hashtable_insert(s_hash, BITLEN, neighbor);
                }
#endif
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
#endif
        __syncthreads();
        int num_neighbors = min(s_num_neighbors, 256);
        if (num_neighbors == 0) continue;
        const bool frontier_full = (s_frontier_size >= beam_size);
        const DISTANCE_T cutoff = frontier_full ? s_frontier_dist[beam_size - 1] : FLT_MAX;
#if USE_SUBWARP_L2_128
        if (dim == 128) {
            for (int base = warp_id * 4; base < num_neighbors; base += num_warps * 4) {
                const int group = lane_id >> 3;      // 0..3 (which distance within the warp)
                const int sublane = lane_id & 7;     // 0..7 within the 8-lane group
                const int n = base + group;
                const bool valid = (n < num_neighbors);
                const vid_t neighbor = valid ? (vid_t)s_neighbor_idx[n] : (vid_t)0;
                const float* bptr = valid ? (const float*)(d_data + (size_t)neighbor * (size_t)128)
                                          : (const float*)query;  // safe: yields 0 distance
                const float sum = subwarp8_l2_distance_128_leader((const float*)query, bptr);
                if (sublane == 0 && valid) {
                    s_neighbor_dist[n] = sum;
                }
                if (lane_id == 0) {
                    int remain = num_neighbors - base;
                    int cnt = (remain >= 4) ? 4 : ((remain > 0) ? remain : 0);
                    atomicAdd(&s_dc_count, cnt);
                }
            }
        } else {
            for (int n = warp_id; n < num_neighbors; n += num_warps) {
                vid_t neighbor = s_neighbor_idx[n];
                const T* nvec = d_data + neighbor * dim;
                DISTANCE_T dist = warp_l2_distance(dim, query, nvec);
                if (lane_id == 0) {
                    s_neighbor_dist[n] = dist;
                    atomicAdd(&s_dc_count, 1);
                }
            }
        }
#else
        for (int n = warp_id; n < num_neighbors; n += num_warps) {
            vid_t neighbor = s_neighbor_idx[n];
            const T* nvec = d_data + neighbor * dim;
            DISTANCE_T dist;
            bool computed_full = true;
            if (dim == 128) {
#if ENABLE_SQ8
                if (frontier_full && d_data_sq8 != nullptr) {
                    const int8_t* x8 = d_data_sq8 + (size_t)neighbor * (size_t)128;
                    const float approx = warp_l2_distance_sq8_128(s_query_sq8, x8);
                    if (approx < (float)cutoff * (float)SQ8_PRUNE_SLACK) {
                        dist = warp_l2_distance_128((const float*)query, (const float*)nvec);
                    } else {
                        dist = FLT_MAX;
                        computed_full = false;
                    }
                } else {
#if ENABLE_PARTIAL_L2_128
                    if (frontier_full) {
                        const float partial = warp_l2_distance_128_prefix<PARTIAL_L2_128_LANES>(
                            (const float*)query, (const float*)nvec);
                        if (partial < cutoff) {
                            const float tail = warp_l2_distance_128_suffix<PARTIAL_L2_128_LANES>(
                                (const float*)query, (const float*)nvec);
                            dist = partial + tail;
                        } else {
                            dist = FLT_MAX;
                            computed_full = false;
                        }
                    } else {
                        dist = warp_l2_distance_128((const float*)query, (const float*)nvec);
                    }
#else
                    dist = warp_l2_distance_128((const float*)query, (const float*)nvec);
#endif
                }
#else
#if ENABLE_PARTIAL_L2_128
                if (frontier_full) {
                    const float partial = warp_l2_distance_128_prefix<PARTIAL_L2_128_LANES>(
                        (const float*)query, (const float*)nvec);
                    if (partial < cutoff) {
                        const float tail = warp_l2_distance_128_suffix<PARTIAL_L2_128_LANES>(
                            (const float*)query, (const float*)nvec);
                        dist = partial + tail;
                    } else {
                        dist = FLT_MAX;
                        computed_full = false;
                    }
                } else {
                    dist = warp_l2_distance_128((const float*)query, (const float*)nvec);
                }
#else
                dist = warp_l2_distance_128((const float*)query, (const float*)nvec);
#endif
#endif
            } else {
                dist = warp_l2_distance(dim, query, nvec);
            }
            if (lane_id == 0) {
                s_neighbor_dist[n] = dist;
                if (computed_full) {
                    atomicAdd(&s_dc_count, 1);
                }
            }
        }
#endif
        __syncthreads();
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
