#pragma once
#include "l2.cuh"
#include "topk_reg.cuh"
#include "hash_filter.cuh"
#include "gpu_graph.cuh"
#include "gpu_base_search.cuh"
#include <cuda_runtime.h>
#include <climits>

/**
 * @class GPURegSearch — single-warp register-topk beam search visitor.
 *
 * Runs one query per warp (BX=32). TopK buffer lives entirely in
 * per-thread registers via RegBitonicTopK — no shared memory for topk.
 * L2 distances are computed warp-collaboratively (l2_warp). All sync
 * uses __syncwarp.
 *
 * @tparam BX    Block size (must be 32).
 * @tparam REGX  Registers per thread for topk. Buffer size = 32 * REGX.
 * @tparam REGY  Registers per thread for incoming merge batch.
 */
template <unsigned BX, unsigned REGX = 8, unsigned REGY = 2>
class GPURegSearch {
    static_assert(BX == WARP_SIZE, "GPURegSearch requires BX == 32");

    RegBitonicTopK<REGX, REGY> topk_;
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
    __device__ GPURegSearch(char*& smem, const GPUBaseSearchCfg& cfg,
                            GPUGraphView graph);

    __device__ void search(const float* query, vid_t* results, int K);

    __host__ __device__ static size_t shared_mem_bytes(
        const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree);
};

// ── implementation ──────────────────────────────────────────────────────────

template <unsigned BX, unsigned REGX, unsigned REGY>
__device__ GPURegSearch<BX, REGX, REGY>::GPURegSearch(
    char*& smem, const GPUBaseSearchCfg& cfg, GPUGraphView graph)
    : topk_(),
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

template <unsigned BX, unsigned REGX, unsigned REGY>
__host__ __device__ size_t GPURegSearch<BX, REGX, REGY>::shared_mem_bytes(
    const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree) {
    size_t size = 0;
    size += HashFilter::shared_mem_bytes(cfg.bitlen);
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
    size += dim * sizeof(float);           // query_buf_
    size += sizeof(INDEX_T);               // parent_buf_
    size += degree * sizeof(INDEX_T);      // cand_idx_
    size += degree * sizeof(DISTANCE_T);   // cand_dist_
    return size;
}

// ── search ──────────────────────────────────────────────────────────────────

template <unsigned BX, unsigned REGX, unsigned REGY>
__device__ void GPURegSearch<BX, REGX, REGY>::search(
    const float* query, vid_t* results, int K)
{
    const unsigned lane = threadIdx.x;
    const int dim = static_cast<int>(dim_);

    // ── load query ──────────────────────────────────────────────────────
    for (unsigned i = lane; i < dim_; i += 32)
        query_buf_[i] = query[i];

    for (unsigned i = lane; i < max_degree_; i += 32) {
        cand_idx_[i]  = MAX_INDEX;
        cand_dist_[i] = FLT_MAX;
    }

    hf_.init();
    __syncwarp();

    // ── seed entry point ────────────────────────────────────────────────
    {
        DISTANCE_T d = l2_warp(dim, query_buf_,
                               graph_.vect(cfg_.entry_point).data());
        if (lane == 0) {
            cand_idx_[0]  = cfg_.entry_point;
            cand_dist_[0] = d;
            hf_.insert(cfg_.entry_point);
        }
    }
    __syncwarp();

    // ── main loop ───────────────────────────────────────────────────────
    for (int iter = 0; iter < cfg_.max_iterations; iter++) {

        // ── hash reset + re-populate ────────────────────────────────────
        if ((iter + 1) % cfg_.reset_interval == 0) {
            hf_.init();
            __syncwarp();
            for (unsigned i = lane; i < topk_.capacity(); i += 32) {
                uint32_t node = topk_.id(i);
                if (node != MAX_INDEX) hf_.insert(node);
            }
        }
        __syncwarp();

        // ── merge candidates into topk ──────────────────────────────────
        {
            float    d[REGY];
            uint32_t x[REGY];
            #pragma unroll
            for (unsigned j = 0; j < REGY; j++) {
                unsigned pos = lane * REGY + j;
                d[j] = (pos < max_degree_) ? cand_dist_[pos] : FLT_MAX;
                x[j] = (pos < max_degree_) ? cand_idx_[pos]  : MAX_INDEX;
            }
            topk_.merge(d, x);
        }

        // ── pick ────────────────────────────────────────────────────────
        uint32_t pbuf[REGY];
        #pragma unroll
        for (unsigned i = 0; i < REGY; ++i) pbuf[i] = MAX_INDEX;
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
        if (lane == 0 && found > 0)
            parent_buf_[0] = picked;
        __syncwarp();

        if (found == 0) break;

        // ── expand: hash filter ─────────────────────────────────────────
        vid_t current_node = parent_buf_[0];
        Span<const vid_t> nbrs = graph_.nbrs(current_node);

        for (unsigned i = lane; i < max_degree_; i += 32) {
            INDEX_T child = nbrs[i];
            if (child != MAX_INDEX && hf_.insert(child) == 0)
                child = MAX_INDEX;
            cand_idx_[i]  = child;
            cand_dist_[i] = FLT_MAX;
        }
        __syncwarp();

        // ── L2 distance (single warp, sequential over neighbors) ────────
        for (unsigned i = 0; i < max_degree_; i++) {
            if (cand_idx_[i] == MAX_INDEX) continue;
            DISTANCE_T d = l2_warp(dim, query_buf_,
                                   graph_.vect(cand_idx_[i]).data());
            if (lane == 0)
                cand_dist_[i] = d;
        }
        __syncwarp();
    }

    // ── final merge + flush ─────────────────────────────────────────────
    {
        float    d[REGY];
        uint32_t x[REGY];
        #pragma unroll
        for (unsigned j = 0; j < REGY; j++) {
            unsigned pos = lane * REGY + j;
            d[j] = (pos < max_degree_) ? cand_dist_[pos] : FLT_MAX;
            x[j] = (pos < max_degree_) ? cand_idx_[pos]  : MAX_INDEX;
        }
        topk_.merge(d, x);
    }

    topk_.flush(cand_dist_, results, K);
}
