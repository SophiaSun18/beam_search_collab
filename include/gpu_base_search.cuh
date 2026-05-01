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
#define MAX_INDEX UINT_MAX

struct GPUBaseSearchCfg {
    unsigned topk_size;
    unsigned bitlen;
    vid_t    entry_point;
    int      max_iterations;
    int      reset_interval;
};

/**
 * @class GPUBaseSearch — self-contained L2 beam search visitor.
 *
 * Owns mutable search state (beam, hash filter, candidate buffer, query
 * scratch) in shared memory.  Immutable external data (graph view) is
 * bound at construction; query + output are passed to search().
 *
 * @tparam BX  Block size (number of threads).
 */
template <unsigned BX>
class GPUBaseSearch {
    SharedBitonicTopK<BX, 1> topk_;
    HashFilter               hf_;
    float*                   query_buf_;    // dim floats in shared memory
    INDEX_T*                 parent_buf_;   // 1 element in shared memory
    INDEX_T*                 cand_idx_;     // max_degree_ elements
    DISTANCE_T*              cand_dist_;    // max_degree_ elements

    GPUGraphView             graph_;
    GPUBaseSearchCfg         cfg_;
    unsigned                 dim_;
    unsigned                 max_degree_;

public:
    /**
     * @brief Construct search state, carving shared memory from smem.
     *
     * All threads must call this (block-wide init). After return, smem
     * points past all allocations.
     */
    __device__ GPUBaseSearch(char*& smem, const GPUBaseSearchCfg& cfg,
                             GPUGraphView graph);

    /**
     * @brief Run beam search for one query.
     *
     * @param query    Device global memory, dim floats (one query vector).
     * @param results  Device global memory, K elements (output top-K node IDs).
     * @param K        Number of results to return.
     */
    __device__ void search(const float* query, vid_t* results, int K);

    /**
     * @brief Total shared memory bytes needed for construction.
     */
    __host__ __device__ static size_t shared_mem_bytes(
        const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree);
};

// ── implementation ──────────────────────────────────────────────────────────

template <unsigned BX>
__device__ GPUBaseSearch<BX>::GPUBaseSearch(
    char*& smem, const GPUBaseSearchCfg& cfg, GPUGraphView graph)
    : topk_(smem, cfg.topk_size),
      hf_(smem, cfg.bitlen),
      graph_(graph),
      cfg_(cfg),
      dim_(graph.dim()),
      max_degree_(graph.degree())
{
    // align for float/uint32_t
    smem = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(smem) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));

    query_buf_  = reinterpret_cast<float*>(smem);
    smem += graph.dim() * sizeof(float);

    parent_buf_ = reinterpret_cast<INDEX_T*>(smem);
    smem += sizeof(INDEX_T);

    cand_idx_   = reinterpret_cast<INDEX_T*>(smem);
    smem += max_degree_ * sizeof(INDEX_T);

    cand_dist_  = reinterpret_cast<DISTANCE_T*>(smem);
    smem += max_degree_ * sizeof(DISTANCE_T);
}

template <unsigned BX>
__host__ __device__ size_t GPUBaseSearch<BX>::shared_mem_bytes(
    const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree) {
    size_t size = 0;
    size += SharedBitonicTopK<BX, 1>::shared_mem_bytes(cfg.topk_size);
    size += HashFilter::shared_mem_bytes(cfg.bitlen);
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
    size += dim * sizeof(float);                     // query_buf_
    size += sizeof(INDEX_T);                         // parent_buf_
    size += degree * sizeof(INDEX_T);                // cand_idx_
    size += degree * sizeof(DISTANCE_T);             // cand_dist_
    return size;
}

// ── search ──────────────────────────────────────────────────────────────────

template <unsigned BX>
__device__ void GPUBaseSearch<BX>::search(
    const float* query, vid_t* results, int K)
{
    const unsigned tid = threadIdx.x;
    const unsigned warp_id = tid / WARP_SIZE;
    const unsigned lane_id = tid % WARP_SIZE;
    const int dim = static_cast<int>(dim_);

    // ── load query into shared memory ───────────────────────────────────
    for (unsigned i = tid; i < dim_; i += BX)
        query_buf_[i] = query[i];

    // ── init candidate buffer ───────────────────────────────────────────
    for (unsigned i = tid; i < max_degree_; i += BX) {
        cand_idx_[i]  = MAX_INDEX;
        cand_dist_[i] = FLT_MAX;
    }

    hf_.init();
    __syncthreads();

    // ── seed entry point ────────────────────────────────────────────────
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

    // ── shared flag for loop control ────────────────────────────────────
    __shared__ uint32_t keep_expanding;

    // ── main loop ───────────────────────────────────────────────────────
    for (int iter = 0; iter < cfg_.max_iterations; iter++) {

        // ── forgettable hash table reset + re-populate from frontier ────
        if ((iter + 1) % cfg_.reset_interval == 0) {
            hf_.init();
            __syncthreads();
            for (unsigned i = tid; i < topk_.capacity(); i += BX) {
                uint32_t node = topk_.id(i);
                if (node != MAX_INDEX) hf_.insert(node);
            }
        }
        __syncthreads();

        // ── merge candidates into topk ──────────────────────────────────
        {
            float    d[1];
            uint32_t x[1];
            d[0] = (tid < max_degree_) ? cand_dist_[tid] : FLT_MAX;
            x[0] = (tid < max_degree_) ? cand_idx_[tid]  : MAX_INDEX;
            topk_.merge(d, x);
        }

        // ── pick next node to expand ────────────────────────────────────
        {
            uint32_t pbuf[BX];
            #pragma unroll
            for (unsigned i = 0; i < BX; ++i) pbuf[i] = MAX_INDEX;
            unsigned found = topk_.pick(pbuf, 1);
            uint32_t picked = MAX_INDEX;
            if (warp_id == 0) {
                const bool has_pick = found > 0 && pbuf[0] != MAX_INDEX;
                const uint32_t ballot = __ballot_sync(0xffffffff, has_pick);
                if (ballot != 0) {
                    const int src_lane = __ffs(ballot) - 1;
                    picked = __shfl_sync(0xffffffff, pbuf[0], src_lane);
                }
            }
            if (warp_id == 0 && lane_id == 0) {
                keep_expanding = found;
                if (found > 0) parent_buf_[0] = picked;
            }
        }
        __syncthreads();

        if (!keep_expanding) break;

        // ── expand ──────────────────────────────────────────────────────
        vid_t current_node = parent_buf_[0];
        Span<const vid_t> nbrs = graph_.nbrs(current_node);

        // filter neighbors through hash, init candidate buffer
        for (unsigned i = tid; i < max_degree_; i += BX) {
            INDEX_T child = nbrs[i];
            if (child != MAX_INDEX && hf_.insert(child) == 0)
                child = MAX_INDEX;
            cand_idx_[i]  = child;
            cand_dist_[i] = FLT_MAX;
        }
        __syncthreads();

        // all warps compute L2 distances in parallel
        for (unsigned i = warp_id; i < max_degree_; i += BX / WARP_SIZE) {
            if (cand_idx_[i] == MAX_INDEX) continue;
            DISTANCE_T d = l2_warp(dim, query_buf_,
                                              graph_.vect(cand_idx_[i]).data());
            if (lane_id == 0)
                cand_dist_[i] = d;
        }
        __syncthreads();
    }

    // ── final merge (catch unmerged candidates from last iteration) ─────
    {
        float    d[1];
        uint32_t x[1];
        d[0] = (tid < max_degree_) ? cand_dist_[tid] : FLT_MAX;
        x[0] = (tid < max_degree_) ? cand_idx_[tid]  : MAX_INDEX;
        topk_.merge(d, x);
    }

    // ── write results ───────────────────────────────────────────────────
    topk_.flush(reinterpret_cast<float*>(cand_dist_), results, K);
}
