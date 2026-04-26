#pragma once
#include "utils.cuh"
#include "topk.cuh"
#include "hash_filter.cuh"
#include "gpu_graph.cuh"
#include <cuda_runtime.h>

#define GPU_WARP_SEARCH_CANDIDATE_BUFFER_SIZE 64

/**
 * @class GPUWarpSearch — warp-level L2 beam search visitor.
 *
 * Each warp independently runs one query. Block shape is 2D:
 *   threadIdx.x = lane within warp (0..31)
 *   threadIdx.y = warp / query index (0..NY-1)
 *
 * Shared memory is partitioned: each warp gets its own region for
 * topk buffer, hash filter, query scratch, and candidate buffer.
 *
 * @tparam NY   Number of warps (queries) per block.
 * @tparam REG  Registers per thread for topk merge. topk_size must be >= 32*REG.
 */
template <unsigned NY, unsigned REG = 2>
class GPUWarpSearch {
    static constexpr unsigned BX   = WARP_SIZE;    // 32
    static constexpr unsigned CAND = GPU_WARP_SEARCH_CANDIDATE_BUFFER_SIZE;

    SharedBitonicTopK<BX, REG> topk_;
    HashFilter                 hf_;
    float*                     query_buf_;    // dim floats
    INDEX_T*                   parent_buf_;   // 1 element
    INDEX_T*                   cand_idx_;     // CAND elements
    DISTANCE_T*                cand_dist_;    // CAND elements

    GPUGraphView               graph_;
    GPUBaseSearchCfg           cfg_;
    unsigned                   dim_;
    unsigned                   max_degree_;

public:
    /**
     * @brief Construct per-warp search state from a warp-local shared memory region.
     *
     * @param smem   Pointer to this warp's shared memory region. Advanced past allocations.
     * @param cfg    Search configuration.
     * @param graph  Immutable graph view.
     */
    __device__ GPUWarpSearch(char*& smem, const GPUBaseSearchCfg& cfg,
                             GPUGraphView graph);

    /**
     * @brief Run beam search for one query. Called by all lanes of the warp.
     *
     * @param query    Device global memory, dim floats.
     * @param results  Device global memory, K elements (output top-K node IDs).
     * @param K        Number of results to return.
     */
    __device__ void search(const float* query, vid_t* results, int K);

    /** @brief Shared memory bytes needed per warp. */
    __host__ __device__ static size_t shared_mem_bytes_per_warp(
        const GPUBaseSearchCfg& cfg, unsigned dim);

    /** @brief Total shared memory bytes for the full block (NY warps). */
    __host__ __device__ static size_t shared_mem_bytes(
        const GPUBaseSearchCfg& cfg, unsigned dim);
};

// ── implementation ──────────────────────────────────────────────────────────

template <unsigned NY, unsigned REG>
__device__ GPUWarpSearch<NY, REG>::GPUWarpSearch(
    char*& smem, const GPUBaseSearchCfg& cfg, GPUGraphView graph)
    : topk_(smem, cfg.topk_size),
      hf_(smem, cfg.bitlen),
      query_buf_  (reinterpret_cast<float*>(smem)),
      parent_buf_ (reinterpret_cast<INDEX_T*>(smem + graph.dim() * sizeof(float))),
      cand_idx_   (reinterpret_cast<INDEX_T*>(smem + graph.dim() * sizeof(float)
                       + sizeof(INDEX_T))),
      cand_dist_  (reinterpret_cast<DISTANCE_T*>(smem + graph.dim() * sizeof(float)
                       + sizeof(INDEX_T)
                       + CAND * sizeof(INDEX_T))),
      graph_(graph),
      cfg_(cfg),
      dim_(graph.dim()),
      max_degree_(graph.degree())
{
    smem += graph.dim() * sizeof(float)
          + sizeof(INDEX_T)
          + CAND * sizeof(INDEX_T)
          + CAND * sizeof(DISTANCE_T);
}

template <unsigned NY, unsigned REG>
__host__ __device__ size_t GPUWarpSearch<NY, REG>::shared_mem_bytes_per_warp(
    const GPUBaseSearchCfg& cfg, unsigned dim) {
    size_t size = 0;
    size += SharedBitonicTopK<BX, REG>::shared_mem_bytes(cfg.topk_size);
    size += HashFilter::shared_mem_bytes(cfg.bitlen);
    size += dim * sizeof(float);                     // query_buf_
    size += sizeof(INDEX_T);                         // parent_buf_
    size += CAND * sizeof(INDEX_T);                  // cand_idx_
    size += CAND * sizeof(DISTANCE_T);               // cand_dist_
    return size;
}

template <unsigned NY, unsigned REG>
__host__ __device__ size_t GPUWarpSearch<NY, REG>::shared_mem_bytes(
    const GPUBaseSearchCfg& cfg, unsigned dim) {
    return NY * shared_mem_bytes_per_warp(cfg, dim);
}

// ── search ──────────────────────────────────────────────────────────────────

template <unsigned NY, unsigned REG>
__device__ void GPUWarpSearch<NY, REG>::search(
    const float* query, vid_t* results, int K)
{
    const unsigned lane = threadIdx.x;  // 0..31
    const int dim = static_cast<int>(dim_);

    // ── load query into shared memory ───────────────────────────────────
    for (unsigned i = lane; i < dim_; i += BX)
        query_buf_[i] = query[i];

    // ── init candidate buffer ───────────────────────────────────────────
    for (unsigned i = lane; i < CAND; i += BX) {
        cand_idx_[i]  = MAX_INDEX;
        cand_dist_[i] = FLT_MAX;
    }
    __syncwarp();

    // ── seed entry point ────────────────────────────────────────────────
    {
        DISTANCE_T d = warp_l2_distance(dim, query_buf_,
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

        // forgettable hash table reset
        if ((iter + 1) % cfg_.reset_interval == 0)
            hf_.init();
        __syncwarp();

        // ── merge candidates into topk ──────────────────────────────────
        {
            float    d[REG];
            uint32_t x[REG];
            #pragma unroll
            for (unsigned j = 0; j < REG; j++) {
                unsigned pos = lane * REG + j;
                d[j] = (pos < CAND) ? cand_dist_[pos] : FLT_MAX;
                x[j] = (pos < CAND) ? cand_idx_[pos]  : MAX_INDEX;
            }
            topk_.merge(d, x);
        }

        // ── pick next node to expand ────────────────────────────────────
        uint32_t pbuf[BX * REG];
        unsigned found = topk_.pick(pbuf, 1);
        if (lane == 0 && found > 0)
            parent_buf_[0] = pbuf[0];
        __syncwarp();

        if (found == 0) break;

        // ── expand ──────────────────────────────────────────────────────
        vid_t current_node = parent_buf_[0];
        Span<const vid_t> nbrs = graph_.nbrs(current_node);

        // filter neighbors, init candidate buffer
        for (unsigned i = lane; i < CAND; i += BX) {
            INDEX_T child = (i < max_degree_) ? nbrs[i] : MAX_INDEX;
            if (child != MAX_INDEX && hf_.insert(child) == 0)
                child = MAX_INDEX;
            cand_idx_[i]  = child;
            cand_dist_[i] = FLT_MAX;
        }
        __syncwarp();

        // compute L2 distances — one warp, sequential over candidates
        for (unsigned i = 0; i < CAND; i++) {
            if (cand_idx_[i] == MAX_INDEX) continue;
            DISTANCE_T d = warp_l2_distance(dim, query_buf_,
                                             graph_.vect(cand_idx_[i]).data());
            if (lane == 0)
                cand_dist_[i] = d;
            __syncwarp();
        }
    }

    // ── final merge ─────────────────────────────────────────────────────
    {
        float    d[REG];
        uint32_t x[REG];
        #pragma unroll
        for (unsigned j = 0; j < REG; j++) {
            unsigned pos = lane * REG + j;
            d[j] = (pos < CAND) ? cand_dist_[pos] : FLT_MAX;
            x[j] = (pos < CAND) ? cand_idx_[pos]  : MAX_INDEX;
        }
        topk_.merge(d, x);
    }

    // ── write results ───────────────────────────────────────────────────
    topk_.flush(cand_dist_, results, K);
}
