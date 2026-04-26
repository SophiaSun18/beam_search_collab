#pragma once
#include "utils.cuh"
#include "topk.cuh"
#include "hash_filter.cuh"
#include "gpu_graph.cuh"
#include <cuda_runtime.h>
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
#include <cuda_pipeline_primitives.h>
#endif

#define GPU_WARP_SEARCH_CANDIDATE_BUFFER_SIZE 64

/**
 * @class GPUWarpSearch — warp-level L2 beam search visitor.
 *
 * Each warp independently runs one query. Block shape is 2D:
 *   threadIdx.x = lane within warp (0..31)
 *   threadIdx.y = warp / query index (0..NY-1)
 *
 * Shared memory is partitioned: each warp gets its own region for
 * topk buffer, hash filter, query scratch, candidate buffer, and
 * double-buffered staging for async vector prefetch.
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
    float*                     staged_[2];    // double buffer, dim floats each

    GPUGraphView               graph_;
    GPUBaseSearchCfg           cfg_;
    unsigned                   dim_;
    unsigned                   max_degree_;

    __device__ static void prefetch_begin(float* dst, const float* src, int n);
    __device__ static void prefetch_wait();

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

// ── async copy helpers ──────────────────────────────────────────────────────

template <unsigned NY, unsigned REG>
__device__ void GPUWarpSearch<NY, REG>::prefetch_begin(
    float* dst, const float* src, int n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    for (int i = threadIdx.x; i < n; i += BX)
        __pipeline_memcpy_async(dst + i, src + i, sizeof(float));
    __pipeline_commit();
#else
    for (int i = threadIdx.x; i < n; i += BX)
        dst[i] = src[i];
#endif
}

template <unsigned NY, unsigned REG>
__device__ void GPUWarpSearch<NY, REG>::prefetch_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    __pipeline_wait_prior(0);
#endif
}

// ── constructor ─────────────────────────────────────────────────────────────

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
      staged_{
          reinterpret_cast<float*>(smem + graph.dim() * sizeof(float)
              + sizeof(INDEX_T)
              + CAND * (sizeof(INDEX_T) + sizeof(DISTANCE_T))),
          reinterpret_cast<float*>(smem + graph.dim() * sizeof(float)
              + sizeof(INDEX_T)
              + CAND * (sizeof(INDEX_T) + sizeof(DISTANCE_T))
              + graph.dim() * sizeof(float))
      },
      graph_(graph),
      cfg_(cfg),
      dim_(graph.dim()),
      max_degree_(graph.degree())
{
    smem += graph.dim() * sizeof(float)              // query_buf_
          + sizeof(INDEX_T)                          // parent_buf_
          + CAND * sizeof(INDEX_T)                   // cand_idx_
          + CAND * sizeof(DISTANCE_T)                // cand_dist_
          + 2 * graph.dim() * sizeof(float);         // staged_[2]
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
    size += 2 * dim * sizeof(float);                 // staged_[2]
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
        // prefetch entry point vector into staged_[0]
        prefetch_begin(staged_[0], graph_.vect(cfg_.entry_point).data(), dim);
        prefetch_wait();
        __syncwarp();

        DISTANCE_T d = warp_l2_distance(dim, query_buf_, staged_[0]);
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

        // ── compute L2 with double-buffered prefetch ────────────────────
        // find first two valid candidates
        int cur_valid = -1, next_valid = -1;
        for (int i = 0; i < static_cast<int>(CAND); i++) {
            if (cand_idx_[i] != MAX_INDEX) {
                if (cur_valid < 0) cur_valid = i;
                else if (next_valid < 0) { next_valid = i; break; }
            }
        }

        if (cur_valid >= 0) {
            int cur = 0;
            // prefetch first valid
            prefetch_begin(staged_[cur],
                           graph_.vect(cand_idx_[cur_valid]).data(), dim);

            while (cur_valid >= 0) {
                // wait for current prefetch
                prefetch_wait();
                __syncwarp();

                // start prefetch of next while computing current
                if (next_valid >= 0)
                    prefetch_begin(staged_[1 - cur],
                                   graph_.vect(cand_idx_[next_valid]).data(), dim);

                // compute L2 from staged buffer
                DISTANCE_T d = warp_l2_distance(dim, query_buf_, staged_[cur]);
                if (lane == 0)
                    cand_dist_[cur_valid] = d;

                // advance
                cur_valid = next_valid;
                cur = 1 - cur;
                // find next valid after cur_valid
                next_valid = -1;
                if (cur_valid >= 0) {
                    for (int i = cur_valid + 1; i < static_cast<int>(CAND); i++) {
                        if (cand_idx_[i] != MAX_INDEX) { next_valid = i; break; }
                    }
                }
                __syncwarp();
            }
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
