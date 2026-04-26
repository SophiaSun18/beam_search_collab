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
    INDEX_T*                 parent_slot_;  // 1 element in shared memory
    INDEX_T*                 cand_idx_;     // CANDIDATE_BUFFER_SIZE elements
    DISTANCE_T*              cand_dist_;    // CANDIDATE_BUFFER_SIZE elements
    float*                   staged_vec_;   // 2 * dim floats, double buffer

    const GPUGraph&          graph_;
    GPUBaseSearchCfg         cfg_;
    unsigned                 dim_;
    unsigned                 max_degree_;

public:
    /**
     * @brief Construct search state, carving shared memory from smem.
     *
     * @param smem   Reference to shared memory pointer; advanced past all allocations.
     * @param cfg    Search configuration (topk size, hash bitlen, entry point, iteration limits).
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
__device__ GPUBaseSearch<BX>::GPUBaseSearch(
    char*& smem, const GPUBaseSearchCfg& cfg, const GPUGraph& graph)
    : topk_(smem, cfg.topk_size),
      hf_(smem, cfg.bitlen),
      query_buf_(reinterpret_cast<float*>(smem)),
      parent_slot_(reinterpret_cast<INDEX_T*>(smem + graph.dim() * sizeof(float))),
      cand_idx_(reinterpret_cast<INDEX_T*>(
          smem + graph.dim() * sizeof(float) + sizeof(INDEX_T))),
      cand_dist_(reinterpret_cast<DISTANCE_T*>(
          smem + graph.dim() * sizeof(float) + sizeof(INDEX_T)
               + GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE * sizeof(INDEX_T))),
      staged_vec_(reinterpret_cast<float*>(
          smem + graph.dim() * sizeof(float) + sizeof(INDEX_T)
               + GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE * (sizeof(INDEX_T) + sizeof(DISTANCE_T)))),
      graph_(graph),
      cfg_(cfg),
      dim_(graph.dim()),
      max_degree_(graph.degree())
{
    smem += graph.dim() * sizeof(float)                                          // query_buf_
          + sizeof(INDEX_T)                                                      // parent_slot_
          + GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE * sizeof(INDEX_T)              // cand_idx_
          + GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE * sizeof(DISTANCE_T)           // cand_dist_
          + 2 * graph.dim() * sizeof(float);                                     // staged_vec_
}

template <unsigned BX>
__host__ __device__ size_t GPUBaseSearch<BX>::shared_mem_bytes(
    const GPUBaseSearchCfg& cfg, unsigned dim) {
    size_t size = 0;
    size += SharedBitonicTopK<BX, 1>::shared_mem_bytes(cfg.topk_size);
    size += HashFilter::shared_mem_bytes(cfg.bitlen);
    size += dim * sizeof(float);                                                 // query_buf_
    size += sizeof(INDEX_T);                                                     // parent_slot_
    size += GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE * sizeof(INDEX_T);             // cand_idx_
    size += GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE * sizeof(DISTANCE_T);          // cand_dist_
    size += 2 * dim * sizeof(float);                                             // staged_vec_
    return size;
}

// ── async copy helpers ──────────────────────────────────────────────────────

static __device__ inline void prefetch_begin(float* dst, const float* src, int count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        __pipeline_memcpy_async(dst + i, src + i, sizeof(float));
    }
    __pipeline_commit();
#else
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        dst[i] = src[i];
    }
#endif
}

static __device__ inline void prefetch_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    __pipeline_wait_prior(0);
#endif
}

// ── search ──────────────────────────────────────────────────────────────────

template <unsigned BX>
__device__ void GPUBaseSearch<BX>::search(const float* query, vid_t* results, int K) {
    const unsigned tid = threadIdx.x;
    const unsigned warp_id = tid / WARP_SIZE;
    const unsigned lane_id = tid % WARP_SIZE;
    constexpr unsigned CAND_SIZE = GPU_BASE_SEARCH_CANDIDATE_BUFFER_SIZE;

    // ── init topk buffer ────────────────────────────────────────────────────
    for (unsigned i = tid; i < cfg_.topk_size; i += BX) {
        topk_.topk_idx[i]  = MAX_INDEX;
        topk_.topk_dist[i] = FLT_MAX;
    }
    hf_.init();

    // ── load query into shared memory ───────────────────────────────────────
    for (unsigned i = tid; i < dim_; i += BX) {
        query_buf_[i] = query[i];
    }

    // ── init candidate buffer ───────────────────────────────────────────────
    for (unsigned i = tid; i < CAND_SIZE; i += BX) {
        cand_idx_[i]  = MAX_INDEX;
        cand_dist_[i] = FLT_MAX;
    }
    __syncthreads();

    // ── seed entry point ────────────────────────────────────────────────────
    if (warp_id == 0) {
        Span<const float> entry_vec = graph_.vect(cfg_.entry_point);
        DISTANCE_T dist = warp_l2_distance(static_cast<int>(dim_), query_buf_, entry_vec.data());
        if (lane_id == 0) {
            cand_idx_[0]  = cfg_.entry_point;
            cand_dist_[0] = dist;
            hf_.insert(cfg_.entry_point);
        }
    }
    __syncthreads();

    // ── shared flag for loop control ────────────────────────────────────────
    __shared__ uint32_t keep_expanding;

    // ── main loop ───────────────────────────────────────────────────────────
    for (int iter = 0; iter < cfg_.max_iterations; iter++) {

        // forgettable hash table reset
        if ((iter + 1) % cfg_.reset_interval == 0) {
            hf_.init();
        }
        __syncthreads();

        // merge candidates into topk
        {
            float    d[1];
            uint32_t x[1];
            d[0] = (tid < CAND_SIZE) ? cand_dist_[tid] : FLT_MAX;
            x[0] = (tid < CAND_SIZE) ? cand_idx_[tid]  : MAX_INDEX;
            topk_.merge(d, x);
        }

        // pick next node to expand
        {
            uint32_t pbuf[BX];
            unsigned found = topk_.pick(pbuf, 1);
            if (warp_id == 0 && lane_id == 0) {
                keep_expanding = found;
                if (found > 0) parent_slot_[0] = pbuf[0];
            }
        }
        __syncthreads();

        if (!keep_expanding) break;

        // ── expand ──────────────────────────────────────────────────────────
        INDEX_T current_node = topk_.topk_idx[parent_slot_[0]] & 0x7fffffff;
        Span<vid_t> neighbors = graph_.nbrs(current_node);

        // filter neighbors through hash, mark invalid
        for (unsigned i = tid; i < CAND_SIZE; i += BX) {
            INDEX_T child = (i < max_degree_) ? neighbors[i] : MAX_INDEX;
            if (child != MAX_INDEX) {
                if (hf_.insert(child) == 0) {
                    child = MAX_INDEX;
                }
            }
            cand_idx_[i]  = child;
            cand_dist_[i] = FLT_MAX;
        }
        __syncthreads();

        // compute L2 distances with double-buffered prefetch
        // find first valid candidate for initial prefetch
        int first_valid = -1;
        for (int i = 0; i < static_cast<int>(CAND_SIZE); i++) {
            if (cand_idx_[i] != MAX_INDEX) { first_valid = i; break; }
        }

        if (first_valid >= 0) {
            int cur_buf = 0;
            float* buf[2] = { staged_vec_, staged_vec_ + dim_ };

            // prefetch first valid neighbor
            prefetch_begin(buf[cur_buf], graph_.vect(cand_idx_[first_valid]).data(),
                           static_cast<int>(dim_));

            // find next valid after first
            int next_valid = -1;
            for (int i = first_valid + 1; i < static_cast<int>(CAND_SIZE); i++) {
                if (cand_idx_[i] != MAX_INDEX) { next_valid = i; break; }
            }

            int cur_valid = first_valid;
            while (cur_valid >= 0) {
                prefetch_wait();
                __syncthreads();

                // prefetch next while computing current
                if (next_valid >= 0) {
                    prefetch_begin(buf[1 - cur_buf],
                                   graph_.vect(cand_idx_[next_valid]).data(),
                                   static_cast<int>(dim_));
                }

                // compute L2 from staged buffer
                DISTANCE_T dist = warp_l2_distance(
                    static_cast<int>(dim_), query_buf_, buf[cur_buf]);
                if (lane_id == 0 && warp_id == 0) {
                    cand_dist_[cur_valid] = dist;
                }

                // advance
                cur_valid = next_valid;
                cur_buf = 1 - cur_buf;
                next_valid = -1;
                if (cur_valid >= 0) {
                    for (int i = cur_valid + 1; i < static_cast<int>(CAND_SIZE); i++) {
                        if (cand_idx_[i] != MAX_INDEX) { next_valid = i; break; }
                    }
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }

    // ── final merge (catch unmerged candidates from last iteration) ─────────
    {
        float    d[1];
        uint32_t x[1];
        d[0] = (tid < CAND_SIZE) ? cand_dist_[tid] : FLT_MAX;
        x[0] = (tid < CAND_SIZE) ? cand_idx_[tid]  : MAX_INDEX;
        topk_.merge(d, x);
    }

    // ── write results (smallest K at tail of descending buffer) ─────────────
    for (unsigned i = tid; i < static_cast<unsigned>(K); i += BX) {
        results[i] = topk_.topk_idx[cfg_.topk_size - K + i] & ~0x80000000u;
    }
}
