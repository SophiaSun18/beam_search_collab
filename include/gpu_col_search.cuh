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
 * @class GPUColSearch — column-parallel L2 beam search visitor.
 *
 * Each thread computes the full L2 distance to one neighbor (column-parallel),
 * bypassing the candidate buffer entirely. Per-thread results go directly
 * into the topk merge as register values.
 *
 * @tparam BX  Block size (number of threads).
 */
template <unsigned BX>
class GPUColSearch {
    SharedBitonicTopK<BX, 1> topk_;
    HashFilter               hf_;
    float*                   query_buf_;    // dim floats in shared memory
    INDEX_T*                 parent_buf_;   // 1 element in shared memory

    GPUGraphView             graph_;
    GPUBaseSearchCfg         cfg_;
    unsigned                 dim_;
    unsigned                 max_degree_;

public:
    __device__ GPUColSearch(char*& smem, const GPUBaseSearchCfg& cfg,
                            GPUGraphView graph);

    __device__ void search(const float* query, vid_t* results, int K);

    __host__ __device__ static size_t shared_mem_bytes(
        const GPUBaseSearchCfg& cfg, unsigned dim);
};

// ── implementation ──────────────────────────────────────────────────────────

template <unsigned BX>
__device__ GPUColSearch<BX>::GPUColSearch(
    char*& smem, const GPUBaseSearchCfg& cfg, GPUGraphView graph)
    : topk_(smem, cfg.topk_size),
      hf_(smem, cfg.bitlen),
      graph_(graph),
      cfg_(cfg),
      dim_(graph.dim()),
      max_degree_(graph.degree())
{
    smem = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(smem) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));

    query_buf_  = reinterpret_cast<float*>(smem);
    smem += graph.dim() * sizeof(float);

    parent_buf_ = reinterpret_cast<INDEX_T*>(smem);
    smem += sizeof(INDEX_T);
}

template <unsigned BX>
__host__ __device__ size_t GPUColSearch<BX>::shared_mem_bytes(
    const GPUBaseSearchCfg& cfg, unsigned dim) {
    size_t size = 0;
    size += SharedBitonicTopK<BX, 1>::shared_mem_bytes(cfg.topk_size);
    size += HashFilter::shared_mem_bytes(cfg.bitlen);
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
    size += dim * sizeof(float);     // query_buf_
    size += sizeof(INDEX_T);         // parent_buf_
    return size;
}

// ── search ──────────────────────────────────────────────────────────────────

template <unsigned BX>
__device__ void GPUColSearch<BX>::search(
    const float* query, vid_t* results, int K)
{
    const unsigned tid = threadIdx.x;
    const unsigned warp_id = tid / WARP_SIZE;
    const unsigned lane_id = tid % WARP_SIZE;
    const int dim = static_cast<int>(dim_);

    // ── load query into shared memory ───────────────────────────────────
    for (unsigned i = tid; i < dim_; i += BX)
        query_buf_[i] = query[i];

    hf_.init();
    __syncthreads();

    // ── per-thread registers for one candidate ─────────────────────────
    float    my_dist = FLT_MAX;
    uint32_t my_idx  = MAX_INDEX;

    // ── seed entry point (thread 0 only) ───────────────────────────────
    if (tid == 0) {
        my_idx  = cfg_.entry_point;
        my_dist = l2_column(dim, query_buf_,
                                 graph_.vect(cfg_.entry_point).data());
        hf_.insert(cfg_.entry_point);
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

        // ── merge per-thread registers into topk ────────────────────────
        {
            float    d[1] = {my_dist};
            uint32_t x[1] = {my_idx};
            topk_.merge(d, x);
        }

        // clear for next iteration
        my_dist = FLT_MAX;
        my_idx  = MAX_INDEX;

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

        // ── expand: each thread handles one neighbor ────────────────────
        vid_t current_node = parent_buf_[0];
        Span<const vid_t> nbrs = graph_.nbrs(current_node);

        if (tid < max_degree_) {
            vid_t child = nbrs[tid];
            if (hf_.insert(child)) {
                my_idx  = child;
                my_dist = l2_column(dim, query_buf_,
                                         graph_.vect(child).data());
            }
        }
        __syncthreads();
    }

    // ── final merge ─────────────────────────────────────────────────────
    {
        float    d[1] = {my_dist};
        uint32_t x[1] = {my_idx};
        topk_.merge(d, x);
    }

    // ── write results ───────────────────────────────────────────────────
    // Reuse query_buf_ as scratch for flush (need degree floats, query is
    // no longer needed). query_buf_ has dim floats >= degree.
    topk_.flush(query_buf_, results, K);
}
