#pragma once
#include "gpu_base_search.cuh"
#include "topk.cuh"
#include "visit_filter.cuh"
#include <cuda_runtime.h>

/**
 * @class GPURegWarpSearch — warp-level beam search with register-resident topk.
 *
 * Same multi-query-per-block model as GPUWarpSearch (NY warps, one query
 * per warp), but the topk buffer lives entirely in per-thread registers
 * via RegBitonicTopK. This removes the shared memory cost of the topk
 * buffer, leaving more shared memory budget for the hash filter and
 * candidate buffers, or allowing more warps per block.
 *
 * @tparam NY    Number of warps (queries) per block.
 * @tparam REGX  Registers per thread for topk buffer. Buffer = 32 * REGX.
 * @tparam REGY  Registers per thread for merge batch. Batch = 32 * REGY.
 * @tparam Filter  Visited-node filter type (HashFilter or BloomFilter).
 */
template <unsigned NY, unsigned REGX = 8, unsigned REGY = 2, typename Filter = HashFilter>
class GPURegWarpSearch {
    static constexpr unsigned BX = WARP_SIZE;  // 32

    RegBitonicTopK<REGX, REGY> topk_;
    Filter                     hf_;
    float*                     query_buf_;
    INDEX_T*                   parent_buf_;
    INDEX_T*                   cand_idx_;
    DISTANCE_T*                cand_dist_;

    GPUGraphView               graph_;
    GPUBaseSearchCfg           cfg_;
    unsigned                   dim_;
    unsigned                   max_degree_;

public:
    __device__ GPURegWarpSearch(char*& smem, const GPUBaseSearchCfg& cfg,
                                GPUGraphView graph);

    __device__ void search(const float* query, vid_t* results, int K);

    __host__ __device__ static size_t shared_mem_bytes_per_warp(
        const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree);

    __host__ __device__ static size_t shared_mem_bytes(
        const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree);
};

// ── constructor ─────────────────────────────────────────────────────────────

template <unsigned NY, unsigned REGX, unsigned REGY, typename Filter>
__device__ GPURegWarpSearch<NY, REGX, REGY, Filter>::GPURegWarpSearch(
    char*& smem, const GPUBaseSearchCfg& cfg, GPUGraphView graph)
    : topk_(),
      hf_(smem, cfg.bitlen),
      graph_(graph),
      cfg_(cfg),
      dim_(graph.dim()),
      max_degree_(graph.degree())
{
    // Plain-pointer buffers: manual per-warp offset within remaining smem.
    smem = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(smem) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));

    const size_t rest = graph.dim() * sizeof(float)       // query_buf_
                      + sizeof(INDEX_T)                   // parent_buf_
                      + max_degree_ * sizeof(INDEX_T)     // cand_idx_
                      + max_degree_ * sizeof(DISTANCE_T); // cand_dist_

    char* my = smem + threadIdx.y * rest;
    query_buf_  = reinterpret_cast<float*>(my);       my += graph.dim() * sizeof(float);
    parent_buf_ = reinterpret_cast<INDEX_T*>(my);     my += sizeof(INDEX_T);
    cand_idx_   = reinterpret_cast<INDEX_T*>(my);     my += max_degree_ * sizeof(INDEX_T);
    cand_dist_  = reinterpret_cast<DISTANCE_T*>(my);

    // Advance past ALL instances so every thread ends at the same position.
    smem += blockDim.y * rest;
}

// ── shared memory size ──────────────────────────────────────────────────────

template <unsigned NY, unsigned REGX, unsigned REGY, typename Filter>
__host__ __device__ size_t GPURegWarpSearch<NY, REGX, REGY, Filter>::shared_mem_bytes_per_warp(
    const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree) {
    size_t size = 0;
    // No topk shared memory — it's in registers.
    size += Filter::shared_mem_bytes(cfg.bitlen);     // visited-node filter
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
    size += dim * sizeof(float);                     // query_buf_
    size += sizeof(INDEX_T);                         // parent_buf_
    size += degree * sizeof(INDEX_T);                // cand_idx_
    size += degree * sizeof(DISTANCE_T);             // cand_dist_
    return size;
}

template <unsigned NY, unsigned REGX, unsigned REGY, typename Filter>
__host__ __device__ size_t GPURegWarpSearch<NY, REGX, REGY, Filter>::shared_mem_bytes(
    const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree) {
    return NY * shared_mem_bytes_per_warp(cfg, dim, degree);
}

// ── search ──────────────────────────────────────────────────────────────────

template <unsigned NY, unsigned REGX, unsigned REGY, typename Filter>
__device__ void GPURegWarpSearch<NY, REGX, REGY, Filter>::search(
    const float* query, vid_t* results, int K)
{
    const unsigned lane = threadIdx.x;
    const int dim = static_cast<int>(dim_);

    // ── load query into shared memory ───────────────────────────────────
    for (unsigned i = lane; i < dim_; i += BX)
        query_buf_[i] = query[i];

    // ── init candidate buffer ───────────────────────────────────────────
    for (unsigned i = lane; i < max_degree_; i += BX) {
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

        // ── forgettable hash table reset + re-populate from frontier ────
        if ((iter + 1) % cfg_.reset_interval == 0) {
            hf_.init();
            __syncwarp();
            for (unsigned i = lane; i < topk_.capacity(); i += BX) {
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

        // ── pick next node to expand ────────────────────────────────────
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

        // ── expand ──────────────────────────────────────────────────────
        vid_t current_node = parent_buf_[0];
        Span<const vid_t> nbrs = graph_.nbrs(current_node);

        for (unsigned i = lane; i < max_degree_; i += BX) {
            INDEX_T child = nbrs[i];
            if (child != MAX_INDEX && hf_.insert(child) == 0)
                child = MAX_INDEX;
            cand_idx_[i]  = child;
            cand_dist_[i] = FLT_MAX;
        }
        __syncwarp();

        // ── compute L2 distances from global memory ─────────────────────
        for (unsigned i = 0; i < max_degree_; i++) {
            if (cand_idx_[i] == MAX_INDEX) continue;
            DISTANCE_T d = l2_warp(
                dim, query_buf_, graph_.vect(cand_idx_[i]).data());
            if (lane == 0)
                cand_dist_[i] = d;
        }
        __syncwarp();
    }

    // ── final merge ─────────────────────────────────────────────────────
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

    // ── write results ───────────────────────────────────────────────────
    topk_.flush(cand_dist_, results, K);
}
