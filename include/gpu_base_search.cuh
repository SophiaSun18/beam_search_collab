#pragma once
#include "utils.cuh"
#include "topk.cuh"
#include "hash_filter.cuh"
#include "gpu_graph.cuh"
#include <cuda_runtime.h>
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
#include <cuda_pipeline_primitives.h>
#endif

#define GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE 64

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
 * scratch, prefetch staging) in shared memory.  Immutable external data
 * (graph) is bound at construction; query + output are passed to search().
 *
 * @tparam BX  Block size (number of threads).
 */
template <unsigned BX>
class GPUBaseSearch {
    SharedBitonicTopK<BX, 1> topk_;
    HashFilter               hf_;
    float*                   query_buf_;    // dim floats in shared memory
    INDEX_T*                 parent_buf_;   // BX elements (pick output)
    INDEX_T*                 cand_idx_;     // CANDIDATE_BUFFER_SIZE elements
    DISTANCE_T*              cand_dist_;    // CANDIDATE_BUFFER_SIZE elements
    float*                   staged_[2];    // double buffer, dim floats each

    const GPUGraph&          graph_;
    GPUBaseSearchCfg         cfg_;
    unsigned                 dim_;
    unsigned                 max_degree_;

    __device__ static void prefetch_begin(float* dst, const float* src, int n);
    __device__ static void prefetch_wait();

public:
    /**
     * @brief Construct search state, carving shared memory from smem.
     *
     * All threads must call this (block-wide init). After return, smem
     * points past all allocations.
     *
     * @param smem   Reference to shared memory pointer; advanced past all allocations.
     * @param cfg    Search configuration.
     * @param graph  Device-resident graph (vectors + edges). Stored by reference.
     */
    __device__ GPUBaseSearch(char*& smem, const GPUBaseSearchCfg& cfg,
                             const GPUGraph& graph);

    /**
     * @brief Run beam search for one query.
     *
     * @param query    Device global memory, dim floats (one query vector).
     * @param results  Device global memory, K elements (output top-K node IDs, ascending distance).
     * @param K        Number of results to return.
     */
    __device__ void search(const float* query, vid_t* results, int K);

    /**
     * @brief Total shared memory bytes needed for construction.
     */
    __host__ __device__ static size_t shared_mem_bytes(
        const GPUBaseSearchCfg& cfg, unsigned dim);
};

// ── implementation ──────────────────────────────────────────────────────────

template <unsigned BX>
__device__ void GPUBaseSearch<BX>::prefetch_begin(float* dst, const float* src, int n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        __pipeline_memcpy_async(dst + i, src + i, sizeof(float));
    __pipeline_commit();
#else
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        dst[i] = src[i];
#endif
}

template <unsigned BX>
__device__ void GPUBaseSearch<BX>::prefetch_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    __pipeline_wait_prior(0);
#endif
}

template <unsigned BX>
__device__ GPUBaseSearch<BX>::GPUBaseSearch(
    char*& smem, const GPUBaseSearchCfg& cfg, const GPUGraph& graph)
    : topk_(smem, cfg.topk_size),
      hf_(smem, cfg.bitlen),
      query_buf_  (reinterpret_cast<float*>(smem)),
      parent_buf_ (reinterpret_cast<INDEX_T*>(smem + graph.dim() * sizeof(float))),
      cand_idx_   (reinterpret_cast<INDEX_T*>(smem + graph.dim() * sizeof(float)
                       + BX * sizeof(INDEX_T))),
      cand_dist_  (reinterpret_cast<DISTANCE_T*>(smem + graph.dim() * sizeof(float)
                       + BX * sizeof(INDEX_T)
                       + GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE * sizeof(INDEX_T))),
      staged_{
          reinterpret_cast<float*>(smem + graph.dim() * sizeof(float)
              + BX * sizeof(INDEX_T)
              + GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE * (sizeof(INDEX_T) + sizeof(DISTANCE_T))),
          reinterpret_cast<float*>(smem + graph.dim() * sizeof(float)
              + BX * sizeof(INDEX_T)
              + GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE * (sizeof(INDEX_T) + sizeof(DISTANCE_T))
              + graph.dim() * sizeof(float))
      },
      graph_(graph),
      cfg_(cfg),
      dim_(graph.dim()),
      max_degree_(graph.degree())
{
    smem += graph.dim() * sizeof(float)                                          // query_buf_
          + BX * sizeof(INDEX_T)                                                 // parent_buf_
          + GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE * sizeof(INDEX_T)              // cand_idx_
          + GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE * sizeof(DISTANCE_T)           // cand_dist_
          + 2 * graph.dim() * sizeof(float);                                     // staged_[2]
}

template <unsigned BX>
__host__ __device__ size_t GPUBaseSearch<BX>::shared_mem_bytes(
    const GPUBaseSearchCfg& cfg, unsigned dim) {
    size_t size = 0;
    size += SharedBitonicTopK<BX, 1>::shared_mem_bytes(cfg.topk_size);
    size += HashFilter::shared_mem_bytes(cfg.bitlen);
    size += dim * sizeof(float);                                                 // query_buf_
    size += BX * sizeof(INDEX_T);                                                // parent_buf_
    size += GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE * sizeof(INDEX_T);             // cand_idx_
    size += GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE * sizeof(DISTANCE_T);          // cand_dist_
    size += 2 * dim * sizeof(float);                                             // staged_[2]
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
    constexpr unsigned CAND = GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE;
    const int dim = static_cast<int>(dim_);

    // ── load query into shared memory ───────────────────────────────────
    for (unsigned i = tid; i < dim_; i += BX)
        query_buf_[i] = query[i];

    // ── init candidate buffer ───────────────────────────────────────────
    for (unsigned i = tid; i < CAND; i += BX) {
        cand_idx_[i]  = MAX_INDEX;
        cand_dist_[i] = FLT_MAX;
    }
    __syncthreads();

    // ── seed entry point ────────────────────────────────────────────────
    if (warp_id == 0) {
        Span<const float> v = graph_.vect(cfg_.entry_point);
        DISTANCE_T d = warp_l2_distance(dim, query_buf_, v.data());
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

        // forgettable hash table reset
        if ((iter + 1) % cfg_.reset_interval == 0)
            hf_.init();
        __syncthreads();

        // ── merge candidates into topk ──────────────────────────────────
        {
            float    d[1];
            uint32_t x[1];
            d[0] = (tid < CAND) ? cand_dist_[tid] : FLT_MAX;
            x[0] = (tid < CAND) ? cand_idx_[tid]  : MAX_INDEX;
            topk_.merge(d, x);
        }

        // ── pick next node to expand ────────────────────────────────────
        {
            unsigned found = topk_.pick(parent_buf_, 1);
            if (warp_id == 0 && lane_id == 0)
                keep_expanding = found;
        }
        __syncthreads();

        if (!keep_expanding) break;

        // ── expand ──────────────────────────────────────────────────────
        vid_t current_node = parent_buf_[0];
        Span<vid_t> nbrs = graph_.nbrs(current_node);

        // filter neighbors through hash, init candidate buffer
        for (unsigned i = tid; i < CAND; i += BX) {
            INDEX_T child = (i < max_degree_) ? nbrs[i] : MAX_INDEX;
            if (child != MAX_INDEX && hf_.insert(child) == 0)
                child = MAX_INDEX;
            cand_idx_[i]  = child;
            cand_dist_[i] = FLT_MAX;
        }
        __syncthreads();

        // compute L2 distances with double-buffered prefetch
        // warp 0 computes distances sequentially, other warps assist with prefetch
        int first_valid = -1;
        for (int i = 0; i < static_cast<int>(CAND); i++) {
            if (cand_idx_[i] != MAX_INDEX) { first_valid = i; break; }
        }

        if (first_valid >= 0) {
            int cur = 0;  // which staged buffer holds current data

            // prefetch first valid neighbor
            prefetch_begin(staged_[cur],
                           graph_.vect(cand_idx_[first_valid]).data(), dim);

            // find next valid
            int next_valid = -1;
            for (int i = first_valid + 1; i < static_cast<int>(CAND); i++) {
                if (cand_idx_[i] != MAX_INDEX) { next_valid = i; break; }
            }

            int cur_valid = first_valid;
            while (cur_valid >= 0) {
                prefetch_wait();
                __syncthreads();

                // start prefetch of next while computing current
                if (next_valid >= 0)
                    prefetch_begin(staged_[1 - cur],
                                   graph_.vect(cand_idx_[next_valid]).data(), dim);

                // L2 distance — all warps participate via warp_l2_distance
                DISTANCE_T dist = warp_l2_distance(dim, query_buf_, staged_[cur]);
                if (warp_id == 0 && lane_id == 0)
                    cand_dist_[cur_valid] = dist;

                // advance
                cur_valid = next_valid;
                cur = 1 - cur;
                next_valid = -1;
                if (cur_valid >= 0) {
                    for (int i = cur_valid + 1; i < static_cast<int>(CAND); i++) {
                        if (cand_idx_[i] != MAX_INDEX) { next_valid = i; break; }
                    }
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }

    // ── final merge (catch unmerged candidates from last iteration) ─────
    {
        float    d[1];
        uint32_t x[1];
        d[0] = (tid < CAND) ? cand_dist_[tid] : FLT_MAX;
        x[0] = (tid < CAND) ? cand_idx_[tid]  : MAX_INDEX;
        topk_.merge(d, x);
    }

    // ── write results ───────────────────────────────────────────────────
    topk_.flush(reinterpret_cast<float*>(cand_dist_), results, K);
}
