#pragma once
#include "l2.cuh"
#include "topk.cuh"
#include "hash_filter.cuh"
#include "gpu_graph.cuh"
#include <cuda_runtime.h>
#include <climits>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

typedef uint32_t INDEX_T;
typedef float    DISTANCE_T;
#ifndef MAX_INDEX
#define MAX_INDEX UINT_MAX
#endif

/**
 * @class GPUSplitSearch — split-role L2 beam search visitor.
 *
 * Block-level (BX threads, multiple warps) but the TopK merge/pick is
 * handled exclusively by warp 0 using SharedBitonicTopK<32, REG> with
 * high REG (shuffle-only, no scratch memory). All warps participate in
 * L2 distance computation. This matches Sophia's approach: one warp
 * sorts via registers+shuffles while the rest compute distances.
 *
 * @tparam BX   Block size (e.g. 256).
 * @tparam REG  Registers per thread for topk. topk_size = 32 * REG.
 */
template <unsigned BX, unsigned REG = 8>
class GPUSplitSearch {
    static constexpr unsigned TOPK_BX = WARP_SIZE;  // topk runs on warp 0

    SharedBitonicTopK<TOPK_BX, REG> topk_;
    HashFilter               hf_;
    float*                   query_buf_;    // [dim]
    INDEX_T*                 parent_buf_;   // [1]
    INDEX_T*                 cand_idx_;     // [degree]
    DISTANCE_T*              cand_dist_;    // [degree]

    GPUGraphView             graph_;
    GPUBaseSearchCfg         cfg_;
    unsigned                 dim_;
    unsigned                 max_degree_;

public:
    __device__ GPUSplitSearch(char*& smem, const GPUBaseSearchCfg& cfg,
                              GPUGraphView graph);

    __device__ void search(const float* query, vid_t* results, int K);

    __host__ __device__ static size_t shared_mem_bytes(
        const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree);
};

// ── implementation ──────────────────────────────────────────────────────────

template <unsigned BX, unsigned REG>
__device__ GPUSplitSearch<BX, REG>::GPUSplitSearch(
    char*& smem, const GPUBaseSearchCfg& cfg, GPUGraphView graph)
    : topk_(smem, cfg.topk_size),
      hf_(smem, cfg.bitlen),
      graph_(graph), cfg_(cfg),
      dim_(graph.dim()), max_degree_(graph.degree())
{
    smem = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(smem) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));

    query_buf_  = reinterpret_cast<float*>(smem);
    smem += dim_ * sizeof(float);

    parent_buf_ = reinterpret_cast<INDEX_T*>(smem);
    smem += sizeof(INDEX_T);

    cand_idx_   = reinterpret_cast<INDEX_T*>(smem);
    smem += max_degree_ * sizeof(INDEX_T);

    cand_dist_  = reinterpret_cast<DISTANCE_T*>(smem);
    smem += max_degree_ * sizeof(DISTANCE_T);
}

template <unsigned BX, unsigned REG>
__host__ __device__ size_t GPUSplitSearch<BX, REG>::shared_mem_bytes(
    const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree) {
    size_t size = 0;
    // TopK: BX=32, no scratch (shuffle-only)
    size += SharedBitonicTopK<TOPK_BX, REG>::shared_mem_bytes(cfg.topk_size);
    size += HashFilter::shared_mem_bytes(cfg.bitlen);
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
    size += dim * sizeof(float);                     // query_buf_
    size += sizeof(INDEX_T);                         // parent_buf_
    size += degree * sizeof(INDEX_T);                // cand_idx_
    size += degree * sizeof(DISTANCE_T);             // cand_dist_
    return size;
}

// ── search ──────────────────────────────────────────────────────────────────

template <unsigned BX, unsigned REG>
__device__ void GPUSplitSearch<BX, REG>::search(
    const float* query, vid_t* results, int K)
{
    const unsigned tid = threadIdx.x;
    const unsigned warp_id = tid / WARP_SIZE;
    const unsigned lane_id = tid % WARP_SIZE;
    const int dim = static_cast<int>(dim_);

    // ── load query (all threads) ────────────────────────────────────────
    for (unsigned i = tid; i < dim_; i += BX)
        query_buf_[i] = query[i];

    // ── init candidate buffer (all threads) ─────────────────────────────
    for (unsigned i = tid; i < max_degree_; i += BX) {
        cand_idx_[i]  = MAX_INDEX;
        cand_dist_[i] = FLT_MAX;
    }

    // ── init topk + hash (warp 0 inits topk, all threads init hash) ────
    hf_.init();
    __syncthreads();

    // ── seed entry point (warp 0) ───────────────────────────────────────
    if (warp_id == 0) {
        DISTANCE_T d = l2_warp(dim, query_buf_,
                               graph_.vect(cfg_.entry_point).data());
        if (lane_id == 0) {
            cand_idx_[0]  = cfg_.entry_point;
            cand_dist_[0] = d;
            hf_.insert(cfg_.entry_point);
        }
    }
    __syncthreads();

    __shared__ uint32_t keep_expanding;

    // ── main loop ───────────────────────────────────────────────────────
    for (int iter = 0; iter < cfg_.max_iterations; iter++) {

        // ── hash reset + re-populate (all threads) ──────────────────────
        if ((iter + 1) % cfg_.reset_interval == 0) {
            hf_.init();
            __syncthreads();
            for (unsigned i = tid; i < topk_.capacity(); i += BX) {
                uint32_t node = topk_.id(i);
                if (node != MAX_INDEX) hf_.insert(node);
            }
        }
        __syncthreads();

        // ── merge + pick (warp 0 only) ──────────────────────────────────
        if (warp_id == 0) {
            // Load candidates into REG registers
            float    d[REG];
            uint32_t x[REG];
            #pragma unroll
            for (unsigned j = 0; j < REG; j++) {
                unsigned pos = lane_id * REG + j;
                d[j] = (pos < max_degree_) ? cand_dist_[pos] : FLT_MAX;
                x[j] = (pos < max_degree_) ? cand_idx_[pos]  : MAX_INDEX;
            }
            topk_.merge(d, x);

            // Pick next node
            constexpr unsigned N = TOPK_BX * REG;
            uint32_t pbuf[N];
            #pragma unroll
            for (unsigned i = 0; i < N; ++i) pbuf[i] = MAX_INDEX;
            unsigned found = topk_.pick(pbuf, 1);
            uint32_t picked = MAX_INDEX;
            {
                const bool has_pick = found > 0 && pbuf[0] != MAX_INDEX;
                const uint32_t ballot = __ballot_sync(0xffffffff, has_pick);
                if (ballot != 0) {
                    const int src_lane = __ffs(ballot) - 1;
                    picked = __shfl_sync(0xffffffff, pbuf[0], src_lane);
                }
            }
            if (lane_id == 0) {
                keep_expanding = found;
                if (found > 0) parent_buf_[0] = picked;
            }
        }
        __syncthreads();

        if (!keep_expanding) break;

        // ── expand: hash filter (all threads) ───────────────────────────
        vid_t current_node = parent_buf_[0];
        Span<const vid_t> nbrs = graph_.nbrs(current_node);

        for (unsigned i = tid; i < max_degree_; i += BX) {
            INDEX_T child = nbrs[i];
            if (child != MAX_INDEX && hf_.insert(child) == 0)
                child = MAX_INDEX;
            cand_idx_[i]  = child;
            cand_dist_[i] = FLT_MAX;
        }
        __syncthreads();

        // ── L2 distance (all warps in parallel) ────────────────────────
        for (unsigned i = warp_id; i < max_degree_; i += BX / WARP_SIZE) {
            if (cand_idx_[i] == MAX_INDEX) continue;
            DISTANCE_T d = l2_warp(dim, query_buf_,
                                   graph_.vect(cand_idx_[i]).data());
            if (lane_id == 0)
                cand_dist_[i] = d;
        }
        __syncthreads();
    }

    // ── final merge (warp 0) ────────────────────────────────────────────
    if (warp_id == 0) {
        float    d[REG];
        uint32_t x[REG];
        #pragma unroll
        for (unsigned j = 0; j < REG; j++) {
            unsigned pos = lane_id * REG + j;
            d[j] = (pos < max_degree_) ? cand_dist_[pos] : FLT_MAX;
            x[j] = (pos < max_degree_) ? cand_idx_[pos]  : MAX_INDEX;
        }
        topk_.merge(d, x);
    }
    __syncthreads();

    // ── flush results (all threads for strided write) ───────────────────
    topk_.flush(reinterpret_cast<float*>(cand_dist_), results, K);
}
