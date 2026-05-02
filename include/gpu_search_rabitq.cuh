#pragma once

#include "rabitq.cuh"
#include "l2.cuh"
#include "topk.cuh"
#include "filter_hash.cuh"
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

struct GPURabitqSearchCfg {
    unsigned topk_size;
    unsigned bitlen;
    vid_t    entry_point;
    int      max_iterations;
    int      reset_interval;
};

/**
 *  @class GPURabitqSearch — self-contained RAbitQ beam search visitor.
 *
 *  Owns mutable search state (beam, hash filter, candidate buffer, staged
 *  code/factor buffers) in shared memory.  Immutable external data (graph
 *  view, rabitq view) are bound at construction; the RabitqQuery is
 *  created inside search() once the query is loaded.
 *
 *  @tparam BX  Block size (number of threads).
 */
template <unsigned BX>
class GPURabitqSearch {
    SharedBitonicTopK<BX, 1> topk_;
    HashFilter               hf_;
    float*                   query_buf_;     // dim floats
    INDEX_T*                 parent_buf_;    // 1 element
    INDEX_T*                 cand_idx_;      // max_degree_ elements
    DISTANCE_T*              cand_dist_;     // max_degree_ elements
    char*                    query_smem_;    // saved pointer for RabitqQuery
    uint64_t*                staged_codes_;
    float*                   staged_factors_;

    GPUGraphView             graph_;
    GPURabitqView            rabitq_;
    GPURabitqSearchCfg       cfg_;
    unsigned                 dim_;
    unsigned                 max_degree_;
    unsigned                 bitcode_words_;

    template <typename T>
    __device__ static void stage_begin(T* dst, const T* src, int count);
    __device__ static void stage_wait();

public:
    __device__ GPURabitqSearch(char*& smem, const GPURabitqSearchCfg& cfg,
                               GPUGraphView graph, GPURabitqView rabitq);

    __device__ void search(const float* query, vid_t* results, int K);

    __host__ __device__ static size_t shared_mem_bytes(
        const GPURabitqSearchCfg& cfg, unsigned dim, unsigned degree);
};

// ── async staging helpers ────���─────────────────────────────────────────────

template <unsigned BX>
template <typename T>
__device__ void GPURabitqSearch<BX>::stage_begin(T* dst, const T* src, int count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    for (int i = threadIdx.x; i < count; i += blockDim.x)
        __pipeline_memcpy_async(dst + i, src + i, sizeof(T));
    __pipeline_commit();
#else
    for (int i = threadIdx.x; i < count; i += blockDim.x)
        dst[i] = src[i];
#endif
}

template <unsigned BX>
__device__ void GPURabitqSearch<BX>::stage_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    __pipeline_wait_prior(0);
#endif
}

// ── constructor ───────────────���────────────────────────────��───────────────

template <unsigned BX>
__device__ GPURabitqSearch<BX>::GPURabitqSearch(
    char*& smem, const GPURabitqSearchCfg& cfg,
    GPUGraphView graph, GPURabitqView rabitq)
    : topk_(smem, cfg.topk_size),
      hf_(smem, cfg.bitlen),
      graph_(graph),
      rabitq_(rabitq),
      cfg_(cfg),
      dim_(graph.dim()),
      max_degree_(graph.degree()),
      bitcode_words_(rabitq.padded_dim() / 64)
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

    // reserve region for RabitqQuery (constructed later in search())
    query_smem_ = smem;
    smem += RabitqQuery<BX>::shared_mem_bytes(graph.dim());

    // align for uint64_t before staged codes
    smem = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(smem) + alignof(uint64_t) - 1)
            & ~(static_cast<uintptr_t>(alignof(uint64_t)) - 1));

    staged_codes_ = reinterpret_cast<uint64_t*>(smem);
    smem += static_cast<size_t>(graph.degree()) * bitcode_words_ * sizeof(uint64_t);

    // align for float before staged factors
    smem = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(smem) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));

    staged_factors_ = reinterpret_cast<float*>(smem);
    smem += 3 * static_cast<size_t>(graph.degree()) * sizeof(float);
}

// ── shared_mem_bytes ─────────���─────────────────────────────────────────────

template <unsigned BX>
__host__ __device__ size_t GPURabitqSearch<BX>::shared_mem_bytes(
    const GPURabitqSearchCfg& cfg, unsigned dim, unsigned degree)
{
    int pdim = rabitq_padded_dim(dim);
    unsigned bitcode_words = pdim / 64;

    size_t size = 0;
    size += SharedBitonicTopK<BX, 1>::shared_mem_bytes(cfg.topk_size);
    size += HashFilter::shared_mem_bytes(cfg.bitlen);
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
    size += dim * sizeof(float);                     // query_buf
    size += sizeof(INDEX_T);                         // parent_buf
    size += degree * sizeof(INDEX_T);                 // cand_idx
    size += degree * sizeof(DISTANCE_T);              // cand_dist
    size += RabitqQuery<BX>::shared_mem_bytes(dim);  // RabitqQuery region
    size = (size + alignof(uint64_t) - 1) & ~(alignof(uint64_t) - 1);
    size += static_cast<size_t>(degree) * bitcode_words * sizeof(uint64_t);
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
    size += 3 * static_cast<size_t>(degree) * sizeof(float);
    return size;
}

// ── search ──���────────────────────────────���───────────────────────��─────────

template <unsigned BX>
__device__ void GPURabitqSearch<BX>::search(
    const float* query, vid_t* results, int K)
{
    const unsigned tid     = threadIdx.x;
    const unsigned warp_id = tid / WARP_SIZE;
    const unsigned lane_id = tid % WARP_SIZE;
    const int dim = static_cast<int>(dim_);

    // ── load query into shared memory ────────��──────────────────────
    for (unsigned i = tid; i < dim_; i += BX)
        query_buf_[i] = query[i];
    __syncthreads();

    // ── prepare RAbitQ query (rotate, quantize, build LUT) ──────────
    char* qp = query_smem_;
    RabitqQuery<BX> rq(qp, query_buf_, rabitq_.signs(), dim);
    __syncthreads();

    // ── init candidate buffer ───────────────────────────────────────
    for (unsigned i = tid; i < max_degree_; i += BX) {
        cand_idx_[i]  = MAX_INDEX;
        cand_dist_[i] = FLT_MAX;
    }

    hf_.init();
    __syncthreads();

    // ── seed entry point with L2 distance ────────────────��──────────
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

    // ── shared flags for loop control ─────────��─────────────────────
    __shared__ uint32_t keep_expanding;

    // ── main loop ───────────────────────────────────────────────────
    for (int iter = 0; iter < cfg_.max_iterations; iter++) {

        // forgettable hash table reset + re-populate from frontier
        if ((iter + 1) % cfg_.reset_interval == 0) {
            hf_.init();
            __syncthreads();
            for (unsigned i = tid; i < topk_.capacity(); i += BX) {
                uint32_t node = topk_.id(i);
                if (node != MAX_INDEX) hf_.insert(node);
            }
        }
        __syncthreads();

        // merge candidates into topk
        {
            float d[1] = {(tid < max_degree_) ? cand_dist_[tid] : FLT_MAX};
            uint32_t x[1] = {(tid < max_degree_) ? cand_idx_[tid]  : MAX_INDEX};
            topk_.merge(d, x);
        }

        // pick next node to expand
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

        vid_t current_node = parent_buf_[0];
        hf_.insert(current_node);
        __syncthreads();

        // ── stage codes and factors (async, overlaps with L2) ───────
        stage_begin(staged_codes_,
                    rabitq_.codes(current_node),
                    static_cast<int>(max_degree_ * bitcode_words_));
        stage_begin(staged_factors_,
                    rabitq_.factors(current_node),
                    static_cast<int>(3 * max_degree_));

        // L2 distance to expanding node (all warps collaborate)
        DISTANCE_T exact_dist = l2_warp(
            dim, query_buf_, graph_.vect(current_node).data());

        // fill candidate buffer with neighbors, filtered by hash
        Span<const vid_t> nbrs = graph_.nbrs(current_node);
        for (unsigned i = tid; i < max_degree_; i += BX) {
            INDEX_T child = nbrs[i];
            if (child != MAX_INDEX && hf_.contains(child))
                child = MAX_INDEX;
            cand_idx_[i]  = child;
            cand_dist_[i] = FLT_MAX;
        }
        __syncthreads();

        // wait for staged data
        stage_wait();
        __syncthreads();

        // ── fastscan distance estimation ──────────���─────────────────
        int pdim = rabitq_.padded_dim();
        int num_batches = (max_degree_ + WARP_SIZE - 1) / WARP_SIZE;
        for (int batch = warp_id; batch < num_batches;
             batch += static_cast<int>(BX / WARP_SIZE)) {
            int batch_begin = batch * WARP_SIZE;
            int active = min(static_cast<int>(WARP_SIZE),
                             static_cast<int>(max_degree_) - batch_begin);

            const uint8_t* codes_block =
                reinterpret_cast<const uint8_t*>(staged_codes_)
                + batch * pdim * 4;

            rq.scan_neighbors_32(
                codes_block, staged_factors_,
                static_cast<int>(max_degree_), batch_begin,
                exact_dist, cand_dist_ + batch_begin, active);

            // invalidate visited candidates
            if (lane_id < static_cast<unsigned>(active)) {
                if (cand_idx_[batch_begin + lane_id] == MAX_INDEX)
                    cand_dist_[batch_begin + lane_id] = FLT_MAX;
            }
        }
        __syncthreads();
    }

    // ── final merge ─────────────────────────────────────────────────
    {
        float    d[1];
        uint32_t x[1];
        d[0] = (tid < max_degree_) ? cand_dist_[tid] : FLT_MAX;
        x[0] = (tid < max_degree_) ? cand_idx_[tid]  : MAX_INDEX;
        topk_.merge(d, x);
    }

    // ── write results ───────────────────────────────────────────────
    topk_.flush(reinterpret_cast<float*>(cand_dist_), results, K);
}
