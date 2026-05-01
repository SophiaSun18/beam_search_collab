#pragma once
#include "l2.cuh"
#include "topk.cuh"
#include "hash_filter.cuh"
#include "gpu_graph.cuh"
#include <cuda_runtime.h>
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
#include <cuda_pipeline_primitives.h>
#endif
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
 * @class GPUPipeSearch — pipelined L2 beam search visitor.
 *
 * Same structure as GPUBaseSearch, but each warp double-buffers one
 * neighbor vector at a time: while computing L2 for the current
 * neighbor from shared memory, the next neighbor's vector is being
 * async-loaded into the alternate slot.
 *
 * Staging layout: [NWARPS][2][dim] — each warp owns two vector slots.
 *
 * @tparam BX  Block size (number of threads).
 */
template <unsigned BX>
class GPUPipeSearch {
    static constexpr unsigned NWARPS = BX / WARP_SIZE;

    SharedBitonicTopK<BX, 1> topk_;
    HashFilter               hf_;
    float*                   query_buf_;    // [dim]
    INDEX_T*                 parent_buf_;   // [1]
    INDEX_T*                 cand_idx_;     // [degree]
    DISTANCE_T*              cand_dist_;    // [degree]
    float*                   staged_vecs_;  // [NWARPS * 2 * dim]

    GPUGraphView             graph_;
    GPUBaseSearchCfg         cfg_;
    unsigned                 dim_;
    unsigned                 max_degree_;

public:
    __device__ GPUPipeSearch(char*& smem, const GPUBaseSearchCfg& cfg,
                             GPUGraphView graph);

    __device__ void search(const float* query, vid_t* results, int K);

    __host__ __device__ static size_t shared_mem_bytes(
        const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree);

private:
    __device__ void compute_distances_pipelined();
};

// ── implementation ──────────────────────────────────────────────────────────

template <unsigned BX>
__device__ GPUPipeSearch<BX>::GPUPipeSearch(
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

    smem = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(smem) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));

    staged_vecs_ = reinterpret_cast<float*>(smem);
    smem += NWARPS * 2 * dim_ * sizeof(float);
}

template <unsigned BX>
__host__ __device__ size_t GPUPipeSearch<BX>::shared_mem_bytes(
    const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree) {
    size_t size = 0;
    size += SharedBitonicTopK<BX, 1>::shared_mem_bytes(cfg.topk_size);
    size += HashFilter::shared_mem_bytes(cfg.bitlen);
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
    size += dim * sizeof(float);                     // query_buf_
    size += sizeof(INDEX_T);                         // parent_buf_
    size += degree * sizeof(INDEX_T);                // cand_idx_
    size += degree * sizeof(DISTANCE_T);             // cand_dist_
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
    size += (BX / WARP_SIZE) * 2 * dim * sizeof(float);  // staged_vecs_
    return size;
}

// ── search ──────────────────────────────────────────────────────────────────

template <unsigned BX>
__device__ void GPUPipeSearch<BX>::search(
    const float* query, vid_t* results, int K)
{
    const unsigned tid = threadIdx.x;
    const unsigned warp_id = tid / WARP_SIZE;
    const unsigned lane_id = tid % WARP_SIZE;
    const int dim = static_cast<int>(dim_);

    // ── load query into shared memory ───────────────────────────────────
    for (unsigned i = tid; i < dim_; i += BX)
        query_buf_[i] = query[i];

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

    __shared__ uint32_t keep_expanding;

    // ── main loop ───────────────────────────────────────────────────────
    for (int iter = 0; iter < cfg_.max_iterations; iter++) {

        // ── hash reset + re-populate ────────────────────────────────────
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

        // ── pick ────────────────────────────────────────────────────────
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

        // ── expand: hash filter ─────────────────────────────────────────
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

        compute_distances_pipelined();
        __syncthreads();
    }

    // ── final merge ─────────────────────────────────────────────────────
    {
        float    d[1];
        uint32_t x[1];
        d[0] = (tid < max_degree_) ? cand_dist_[tid] : FLT_MAX;
        x[0] = (tid < max_degree_) ? cand_idx_[tid]  : MAX_INDEX;
        topk_.merge(d, x);
    }

    topk_.flush(reinterpret_cast<float*>(cand_dist_), results, K);
}

/**
 * @brief Pipelined L2 distances for all neighbors in cand_idx_.
 *
 * Per-warp double-buffered vector loading. Each warp owns two dim-sized
 * slots in staged_vecs_. Iteration schedule for warp w (step s):
 *   neighbor index = w + s * NWARPS
 *   Each lane loads dim/WARP_SIZE floats per vector.
 *
 * Pipeline: while computing L2 for step s from slot (s&1), async-load
 * step s+1's vector into slot ((s+1)&1).
 *
 * Reads cand_idx_[0..degree-1] (MAX_INDEX entries are skipped).
 * Writes cand_dist_[i] for each valid neighbor.
 */
template <unsigned BX>
__device__ void GPUPipeSearch<BX>::compute_distances_pipelined()
{
    const unsigned warp_id = threadIdx.x / WARP_SIZE;
    const unsigned lane_id = threadIdx.x % WARP_SIZE;
    const int dim = static_cast<int>(dim_);

    float* my_stage = staged_vecs_ + warp_id * 2 * dim_;
    const unsigned steps = (max_degree_ + NWARPS - 1) / NWARPS;

    // Pre-load first vector into slot 0
    {
        unsigned nbr0 = warp_id;
        vid_t child = (nbr0 < max_degree_) ? cand_idx_[nbr0] : MAX_INDEX;
        if (child != MAX_INDEX) {
            const float* src = graph_.vect(child).data();
            for (unsigned d = lane_id; d < dim_; d += WARP_SIZE) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                __pipeline_memcpy_async(my_stage + d, src + d, sizeof(float));
#else
                my_stage[d] = src[d];
#endif
            }
        }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        __pipeline_commit();
#endif
    }

    for (unsigned s = 0; s < steps; s++) {
        unsigned cur_slot = s & 1;

        // Pre-load next vector into alternate slot
        if (s + 1 < steps) {
            unsigned nxt_nbr = warp_id + (s + 1) * NWARPS;
            vid_t child = (nxt_nbr < max_degree_)
                        ? cand_idx_[nxt_nbr] : MAX_INDEX;
            if (child != MAX_INDEX) {
                const float* src = graph_.vect(child).data();
                float* dst = my_stage + ((s + 1) & 1) * dim_;
                for (unsigned d = lane_id; d < dim_; d += WARP_SIZE) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                    __pipeline_memcpy_async(dst + d, src + d, sizeof(float));
#else
                    dst[d] = src[d];
#endif
                }
            }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
            __pipeline_commit();
            __pipeline_wait_prior(1);
#else
            __syncwarp();
#endif
        } else {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
            __pipeline_wait_prior(0);
#endif
        }
        __syncwarp();

        // Compute L2 from staged vector
        unsigned cur_nbr = warp_id + s * NWARPS;
        if (cur_nbr < max_degree_ && cand_idx_[cur_nbr] != MAX_INDEX) {
            const float* vec = my_stage + cur_slot * dim_;
            DISTANCE_T d = l2_warp(dim, query_buf_, vec);
            if (lane_id == 0)
                cand_dist_[cur_nbr] = d;
        }
        __syncwarp();
    }
}
