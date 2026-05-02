#pragma once
#include "gpu_base_search.cuh"
#include <cuda_runtime.h>
#include <cfloat>
#include <climits>

/**
 * Sophia's beam search kernel, extracted and adapted to use our graph/config
 * types. One query per block, 256 threads, warp 0 does topk sort.
 *
 * Internal topk = 128, candidate buffer = 64, block size = 256.
 */
namespace sophia {

// ── sort/merge primitives ──────────────────────────────────────────────────

template <class K, class V>
__device__ inline void swap_if_needed(K& k0, V& v0, const unsigned lane_offset, const bool asc) {
    auto k1 = __shfl_xor_sync(~0u, k0, lane_offset);
    auto v1 = __shfl_xor_sync(~0u, v0, lane_offset);
    if ((k0 != k1) && ((k0 < k1) != asc)) { k0 = k1; v0 = v1; }
}

template <class K, class V>
__device__ inline void swap_if_needed(K& k0, V& v0, K& k1, V& v1, const bool asc) {
    if ((k0 != k1) && ((k0 < k1) != asc)) {
        const auto tmp_k = k0; k0 = k1; k1 = tmp_k;
        const auto tmp_v = v0; v0 = v1; v1 = tmp_v;
    }
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_merge(K k[N], V v[N], unsigned range, const bool asc = true) {
    static_assert(N == 2 || N == 4, "");
    if constexpr (N == 2) {
        const auto lane_id = threadIdx.x % warp_size;
        if (range == 1) {
            const auto p = ((lane_id & 1) == 0);
            swap_if_needed<K, V>(k[0], v[0], k[1], v[1], p);
            return;
        }
        const uint32_t b = range;
        for (uint32_t c = b / 2; c >= 1; c >>= 1) {
            const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
            for (uint32_t i = 0; i < N; i++) swap_if_needed<K, V>(k[i], v[i], c, p);
        }
        const auto p = ((lane_id & b) == 0);
        swap_if_needed<K, V>(k[0], v[0], k[1], v[1], p);
    } else if constexpr (N == 4) {
        const auto lane_id = threadIdx.x % warp_size;
        if (range == 1) {
            for (uint32_t b = 2; b <= N; b <<= 1) {
                for (uint32_t c = b / 2; c >= 1; c >>= 1) {
                    for (uint32_t i = 0; i < N; i++) {
                        uint32_t j = i ^ c;
                        if (i >= j) continue;
                        const auto line_id = i + (N * lane_id);
                        const auto p = static_cast<bool>(line_id & b) == static_cast<bool>(line_id & c);
                        swap_if_needed(k[i], v[i], k[j], v[j], p);
                    }
                }
            }
            return;
        }
        const uint32_t b = range;
        for (uint32_t c = b / 2; c >= 1; c >>= 1) {
            const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
            for (uint32_t i = 0; i < N; i++) swap_if_needed(k[i], v[i], c, p);
        }
        const auto p = ((lane_id & b) == 0);
        for (uint32_t c = N / 2; c >= 1; c >>= 1) {
            for (uint32_t i = 0; i < N; i++) {
                uint32_t j = i ^ c;
                if (i >= j) continue;
                swap_if_needed(k[i], v[i], k[j], v[j], p);
            }
        }
    }
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_sort(K k[N], V v[N], const bool asc = true) {
    for (uint32_t range = 1; range <= warp_size; range <<= 1)
        warp_merge<K, V, N, warp_size>(k, v, range, asc);
}

// ── topk merge (warp 0 only) ──────────────────────────────────────────────

template <unsigned N_1, unsigned N_2>
__device__ void topk_by_bitonic_sort(
    uint32_t* result_indices, float* result_distances,
    uint32_t result_buf_size, uint32_t cand_buf_size,
    uint32_t internal_topk, bool first)
{
    const unsigned lane_id = threadIdx.x % 32;
    const unsigned warp_id = threadIdx.x / 32;
    if (warp_id > 0) return;

    float* cd = result_distances + internal_topk;
    uint32_t* ci = result_indices + internal_topk;

    // Sort candidates
    float key_1[N_1]; uint32_t val_1[N_1];
    for (unsigned i = 0; i < N_1; i++) {
        unsigned j = lane_id + (32 * i);
        key_1[i] = (j < cand_buf_size) ? cd[j] : FLT_MAX;
        val_1[i] = (j < cand_buf_size) ? ci[j] : UINT_MAX;
    }
    warp_sort<float, uint32_t, N_1>(key_1, val_1);
    for (unsigned i = 0; i < N_1; i++) {
        unsigned j = (N_1 * lane_id) + i;
        if (j < cand_buf_size && j < internal_topk) { cd[j] = key_1[i]; ci[j] = val_1[i]; }
    }

    // Load / sort topk
    float key_2[N_2]; uint32_t val_2[N_2];
    if (first) {
        for (unsigned i = 0; i < N_2; i++) {
            unsigned j = lane_id + (32 * i);
            key_2[i] = (j < internal_topk) ? result_distances[j] : FLT_MAX;
            val_2[i] = (j < internal_topk) ? result_indices[j] : UINT_MAX;
        }
        warp_sort<float, uint32_t, N_2>(key_2, val_2);
    } else {
        for (unsigned i = 0; i < N_2; i++) {
            unsigned j = (N_2 * lane_id) + i;
            key_2[i] = (j < internal_topk) ? result_distances[j] : FLT_MAX;
            val_2[i] = (j < internal_topk) ? result_indices[j] : UINT_MAX;
        }
    }

    // Merge: compare topk with reversed candidates
    for (unsigned i = 0; i < N_2; i++) {
        unsigned j = (N_2 * lane_id) + i;
        unsigned k = internal_topk - 1 - j;
        if (k >= internal_topk || k >= cand_buf_size) continue;
        if (key_2[i] > cd[k]) { key_2[i] = cd[k]; val_2[i] = ci[k]; }
    }
    warp_merge<float, uint32_t, N_2>(key_2, val_2, 32);

    // Store
    for (unsigned i = 0; i < N_2; i++) {
        unsigned j = (N_2 * lane_id) + i;
        if (j < internal_topk) { result_distances[j] = key_2[i]; result_indices[j] = val_2[i]; }
    }
}

// ── pick expanders (warp 0 only) ──────────────────────────────────────────

__device__ uint32_t pick_expanders(int N, uint32_t* output_list, int M, uint32_t* pri_queue) {
    uint32_t num_exp = 0;
    constexpr uint32_t msb = 0x80000000;
    uint32_t itopk_max = M;
    if (itopk_max % 32) itopk_max += 32 - (itopk_max % 32);
    for (uint32_t j = threadIdx.x; j < itopk_max; j += 32) {
        int new_parent = 0;
        if (j < (uint32_t)M) {
            uint32_t index = pri_queue[j];
            if ((index & msb) == 0 && index != UINT_MAX) new_parent = 1;
        }
        const uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
        if (new_parent) {
            const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_exp;
            if (i < (uint32_t)N) { output_list[i] = j; pri_queue[j] |= msb; }
        }
        num_exp += __popc(ballot_mask);
        if (num_exp >= (uint32_t)N) break;
    }
    return num_exp;
}

// ── hash table ─────────────────────────────────────────────────────────────

__device__ inline void hash_init(uint32_t* table, unsigned bitlen) {
    uint32_t size = 1u << bitlen;
    for (uint32_t i = threadIdx.x; i < size; i += blockDim.x)
        table[i] = UINT_MAX;
}

__device__ inline uint32_t hash_insert(uint32_t* table, unsigned bitlen, uint32_t key) {
    uint32_t size = 1u << bitlen;
    uint32_t mask = size - 1;
    uint32_t index = (key ^ (key >> bitlen)) & mask;
    for (unsigned i = 0; i < size; i++) {
        uint32_t old = atomicCAS(&table[index], UINT_MAX, key);
        if (old == UINT_MAX) return 1;
        if (old == key) return 0;
        index = (index + 1) & mask;
    }
    return 0;
}

__device__ inline void hash_restore(uint32_t* table, unsigned bitlen,
                                     const uint32_t* topk_idx, uint32_t topk_size) {
    for (unsigned i = threadIdx.x; i < topk_size; i += blockDim.x) {
        uint32_t key = topk_idx[i] & ~0x80000000u;
        hash_insert(table, bitlen, key);
    }
}

}  // namespace sophia

// ═══════════════════════════════════════════════════════════════════════════
// Search kernel — one query per block, 256 threads
// ═══════════════════════════════════════════════════════════════════════════

static constexpr unsigned SOPHIA_BLOCK = 256;
static constexpr unsigned SOPHIA_TOPK  = 128;
static constexpr unsigned SOPHIA_CAND  = 64;
static constexpr unsigned SOPHIA_BUF   = SOPHIA_TOPK + SOPHIA_CAND;

__host__ __device__ inline size_t sophia_shared_mem_bytes(unsigned bitlen, unsigned dim) {
    size_t size = 0;
    size += (1u << bitlen) * sizeof(uint32_t);          // hash table
    size += SOPHIA_BUF * sizeof(uint32_t);              // all_idx
    size += SOPHIA_BUF * sizeof(float);                 // all_dist
    size += 1 * sizeof(uint32_t);                       // parent
    size += dim * sizeof(float);                        // query buf
    return size;
}

__global__ void kernel_sophia_search(
    GPUGraphView graph, GPUBaseSearchCfg cfg,
    int nq, int dim,
    const float* __restrict__ d_queries,
    vid_t* __restrict__ d_results, int K)
{
    const int qid = blockIdx.x;
    if (qid >= nq) return;
    const float* query = d_queries + qid * dim;
    const unsigned max_degree = graph.degree();
    const unsigned tid = threadIdx.x;
    const unsigned warp_id = tid / 32;
    const unsigned lane_id = tid % 32;

    extern __shared__ char smem[];
    char* ptr = smem;

    // Hash table
    uint32_t* hash_table = reinterpret_cast<uint32_t*>(ptr);
    ptr += (1u << cfg.bitlen) * sizeof(uint32_t);

    // Topk + candidate buffers (contiguous)
    uint32_t* all_idx  = reinterpret_cast<uint32_t*>(ptr);
    ptr += SOPHIA_BUF * sizeof(uint32_t);
    float*    all_dist = reinterpret_cast<float*>(ptr);
    ptr += SOPHIA_BUF * sizeof(float);

    uint32_t* topk_idx  = all_idx;
    float*    topk_dist = all_dist;
    uint32_t* cand_idx  = all_idx  + SOPHIA_TOPK;
    float*    cand_dist = all_dist + SOPHIA_TOPK;

    uint32_t* parent = reinterpret_cast<uint32_t*>(ptr);
    ptr += sizeof(uint32_t);
    float* query_buf = reinterpret_cast<float*>(ptr);

    // Init
    sophia::hash_init(hash_table, cfg.bitlen);
    for (unsigned i = tid; i < (unsigned)dim; i += SOPHIA_BLOCK)
        query_buf[i] = query[i];
    for (unsigned i = tid; i < SOPHIA_TOPK; i += SOPHIA_BLOCK) {
        topk_idx[i] = UINT_MAX;
        topk_dist[i] = FLT_MAX;
    }
    for (unsigned i = tid; i < SOPHIA_CAND; i += SOPHIA_BLOCK) {
        cand_idx[i] = UINT_MAX;
        cand_dist[i] = FLT_MAX;
    }
    __syncthreads();

    // Seed entry point
    {
        float d = l2_warp(dim, query_buf, graph.vect(cfg.entry_point).data());
        if (lane_id == 0 && warp_id == 0) {
            cand_idx[0] = cfg.entry_point;
            cand_dist[0] = d;
            sophia::hash_insert(hash_table, cfg.bitlen, cfg.entry_point);
        }
    }
    __syncthreads();

    // Main loop
    __shared__ uint32_t keep_expanding;
    for (int iter = 0; iter < cfg.max_iterations; iter++) {

        // Hash reset
        if ((iter + 1) % cfg.reset_interval == 0) {
            sophia::hash_init(hash_table, cfg.bitlen);
            __syncthreads();
            sophia::hash_restore(hash_table, cfg.bitlen, topk_idx, SOPHIA_TOPK);
        }
        __syncthreads();

        // Topk merge (warp 0 only): sort candidates, merge into topk
        sophia::topk_by_bitonic_sort<2, 4>(
            all_idx, all_dist, SOPHIA_BUF, SOPHIA_CAND, SOPHIA_TOPK, (iter == 0));
        __syncthreads();

        // Pick (warp 0 only)
        if (warp_id == 0) {
            keep_expanding = sophia::pick_expanders(1, parent, SOPHIA_TOPK, topk_idx);
        }
        __syncthreads();
        if (!keep_expanding) break;

        // Expand: filter neighbors through hash
        uint32_t current_node = topk_idx[parent[0]] & ~0x80000000u;
        Span<const vid_t> nbrs = graph.nbrs(current_node);

        for (unsigned i = tid; i < SOPHIA_CAND; i += SOPHIA_BLOCK) {
            uint32_t child = (i < max_degree) ? nbrs[i] : UINT_MAX;
            if (child != UINT_MAX && sophia::hash_insert(hash_table, cfg.bitlen, child) == 0)
                child = UINT_MAX;
            cand_idx[i] = child;
            cand_dist[i] = FLT_MAX;
        }
        __syncthreads();

        // L2 distance — all warps participate
        for (unsigned i = warp_id; i < SOPHIA_CAND; i += SOPHIA_BLOCK / 32) {
            if (cand_idx[i] == UINT_MAX) continue;
            float d = l2_warp(dim, query_buf, graph.vect(cand_idx[i]).data());
            if (lane_id == 0) cand_dist[i] = d;
        }
        __syncthreads();
    }

    // Final merge
    sophia::topk_by_bitonic_sort<2, 4>(
        all_idx, all_dist, SOPHIA_BUF, SOPHIA_CAND, SOPHIA_TOPK, false);
    __syncthreads();

    // Write results
    for (unsigned i = tid; i < (unsigned)K; i += SOPHIA_BLOCK) {
        d_results[qid * K + i] = topk_idx[i] & ~0x80000000u;
    }
}
