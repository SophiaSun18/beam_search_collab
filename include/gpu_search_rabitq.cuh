#pragma once

#include "rabitq.cuh"
#include "l2.cuh"
#include "topk.cuh"
#include "filter_hash.cuh"
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

struct GPURabitqSearchCfg {
    unsigned topk_size;
    unsigned filter_capacity;
    vid_t    entry_point;
    int      max_iterations;
    int      reset_interval;
};

/**
 *  @class GPURabitqSearch — warp-level RAbitQ beam search visitor.
 *
 *  One warp (32 threads) handles one query. Block shape is 2D:
 *    threadIdx.x = lane within warp (0..31)
 *    threadIdx.y = warp / query index (0..NY-1)
 *
 *  @tparam NY     Number of warps (queries) per block.
 *  @tparam TopK   TopK buffer type (SharedBitonicTopK<32,REG> or RegBitonicTopK).
 *  @tparam REG    Registers per thread for merge batch.
 *  @tparam Filter Visited-node filter type (HashFilter or BloomFilter).
 */
template <unsigned NY, typename TopK, unsigned REG, typename Filter = HashFilter>
class GPURabitqSearch {
    static constexpr unsigned BX = WARP_SIZE;  // 32

    TopK     topk_;
    Filter   hf_;
    float*   query_buf_;      // dim floats
    INDEX_T* cand_idx_;       // max_degree_ elements
    DISTANCE_T* cand_dist_;   // max_degree_ elements
    char*    query_smem_;     // region for RabitqQuery
    uint64_t* staged_codes_;
    float*   staged_factors_;

    GPUGraphView    graph_;
    GPURabitqView   rabitq_;
    GPURabitqSearchCfg cfg_;
    unsigned        dim_;
    unsigned        max_degree_;
    unsigned        bitcode_words_;

public:
    __device__ GPURabitqSearch(char*& smem, const GPURabitqSearchCfg& cfg,
                               GPUGraphView graph, GPURabitqView rabitq);

    __device__ void search(const float* query, vid_t* results, int K);

    __host__ __device__ static size_t shared_mem_bytes_per_warp(
        const GPURabitqSearchCfg& cfg, unsigned dim, unsigned degree);

    __host__ __device__ static size_t shared_mem_bytes(
        const GPURabitqSearchCfg& cfg, unsigned dim, unsigned degree);
};

// ── constructor ─────────────────────────────────────────────────────────────

template <unsigned NY, typename TopK, unsigned REG, typename Filter>
__device__ GPURabitqSearch<NY, TopK, REG, Filter>::GPURabitqSearch(
    char*& smem, const GPURabitqSearchCfg& cfg,
    GPUGraphView graph, GPURabitqView rabitq)
    : topk_(smem, cfg.topk_size),
      hf_(smem, cfg.filter_capacity),
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

    // Plain-pointer buffers: manual per-warp offset
    const size_t rest =
          graph.dim() * sizeof(float)                                    // query_buf
        + max_degree_ * sizeof(INDEX_T)                                  // cand_idx
        + max_degree_ * sizeof(DISTANCE_T)                               // cand_dist
        + RabitqQuery<BX>::shared_mem_bytes(graph.dim())                  // RabitqQuery
        + (alignof(uint64_t) - 1)                                        // alignment
        + static_cast<size_t>(max_degree_) * bitcode_words_ * sizeof(uint64_t) // staged_codes
        + (alignof(float) - 1)                                           // alignment
        + 3 * static_cast<size_t>(max_degree_) * sizeof(float);          // staged_factors

    char* my = smem + threadIdx.y * rest;

    query_buf_  = reinterpret_cast<float*>(my);
    my += graph.dim() * sizeof(float);

    cand_idx_   = reinterpret_cast<INDEX_T*>(my);
    my += max_degree_ * sizeof(INDEX_T);

    cand_dist_  = reinterpret_cast<DISTANCE_T*>(my);
    my += max_degree_ * sizeof(DISTANCE_T);

    query_smem_ = my;
    my += RabitqQuery<BX>::shared_mem_bytes(graph.dim());

    // align for uint64_t
    my = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(my) + alignof(uint64_t) - 1)
            & ~(static_cast<uintptr_t>(alignof(uint64_t)) - 1));
    staged_codes_ = reinterpret_cast<uint64_t*>(my);
    my += static_cast<size_t>(max_degree_) * bitcode_words_ * sizeof(uint64_t);

    // align for float
    my = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(my) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));
    staged_factors_ = reinterpret_cast<float*>(my);

    // Advance past ALL instances
    smem += blockDim.y * rest;
}

// ── shared memory size ──────────────────────────────────────────────────────

template <unsigned NY, typename TopK, unsigned REG, typename Filter>
__host__ __device__ size_t
GPURabitqSearch<NY, TopK, REG, Filter>::shared_mem_bytes_per_warp(
    const GPURabitqSearchCfg& cfg, unsigned dim, unsigned degree)
{
    int pdim = rabitq_padded_dim(dim);
    unsigned bitcode_words = pdim / 64;

    size_t size = 0;
    size += TopK::shared_mem_bytes_per_instance(cfg.topk_size);
    size += Filter::shared_mem_bytes(cfg.filter_capacity);
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
    size += dim * sizeof(float);                                        // query_buf
    size += degree * sizeof(INDEX_T);                                   // cand_idx
    size += degree * sizeof(DISTANCE_T);                                // cand_dist
    size += RabitqQuery<BX>::shared_mem_bytes(dim);                      // RabitqQuery
    size = (size + alignof(uint64_t) - 1) & ~(alignof(uint64_t) - 1);
    size += static_cast<size_t>(degree) * bitcode_words * sizeof(uint64_t); // staged_codes
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
    size += 3 * static_cast<size_t>(degree) * sizeof(float);            // staged_factors
    return size;
}

template <unsigned NY, typename TopK, unsigned REG, typename Filter>
__host__ __device__ size_t
GPURabitqSearch<NY, TopK, REG, Filter>::shared_mem_bytes(
    const GPURabitqSearchCfg& cfg, unsigned dim, unsigned degree)
{
    return NY * shared_mem_bytes_per_warp(cfg, dim, degree);
}

// ── search ──────────────────────────────────────────────────────────────────

template <unsigned NY, typename TopK, unsigned REG, typename Filter>
__device__ void GPURabitqSearch<NY, TopK, REG, Filter>::search(
    const float* query, vid_t* results, int K)
{
    const unsigned lane = threadIdx.x;  // 0..31
    const int dim = static_cast<int>(dim_);

    // load query into shared memory (per-warp, private)
    for (unsigned i = lane; i < dim_; i += BX)
        query_buf_[i] = query[i];
    __syncwarp();

    // prepare RAbitQ query (rotate, quantize, build LUT)
    char* qp = query_smem_;
    RabitqQuery<BX> rq(qp, query_buf_, rabitq_.signs(), dim);
    __syncwarp();

    // compute exact dist of entry point, insert entry point into frontier
    {
        DISTANCE_T d = l2_warp(dim, query_buf_,
                               graph_.vect(cfg_.entry_point).data());
        float    fd[REG];
        uint32_t fx[REG];
        #pragma unroll
        for (unsigned j = 0; j < REG; j++) {
            unsigned pos = lane * REG + j;
            fd[j] = (pos == 0) ? d : FLT_MAX;
            fx[j] = (pos == 0) ? static_cast<uint32_t>(cfg_.entry_point) : MAX_INDEX;
        }
        topk_.merge(fd, fx);
    }

    hf_.init();
    __syncwarp();

    // main loop
    for (int iter = 0; iter < cfg_.max_iterations; iter++) {

        // pick element from frontier
        uint32_t pbuf[1];
        pbuf[0] = MAX_INDEX;
        unsigned found = topk_.template pick<1>(pbuf, 1);
        uint32_t picked = MAX_INDEX;
        {
            const bool has_pick = found > 0 && pbuf[0] != MAX_INDEX;
            const uint32_t ballot = __ballot_sync(0xffffffff, has_pick);
            if (ballot != 0) {
                const int src_lane = __ffs(ballot) - 1;
                picked = __shfl_sync(0xffffffff, pbuf[0], src_lane);
            }
        }

        if (found == 0) break;

        vid_t current_node = picked;

        // try insert into hf_, if duplicate, skip to next iteration
        uint32_t inserted = (lane == 0) ? hf_.insert(current_node) : 0;
        inserted = __shfl_sync(0xffffffff, inserted, 0);
        if (inserted == 0) continue;

        // compute estimated distance between picked node and each neighbor
        // load codes and factors into shared memory
        const uint64_t* src_codes = rabitq_.codes(current_node);
        int code_count = static_cast<int>(max_degree_ * bitcode_words_);
        for (int i = static_cast<int>(lane); i < code_count; i += static_cast<int>(BX))
            staged_codes_[i] = src_codes[i];

        const float* src_factors = rabitq_.factors(current_node);
        int factor_count = static_cast<int>(3 * max_degree_);
        for (int i = static_cast<int>(lane); i < factor_count; i += static_cast<int>(BX))
            staged_factors_[i] = src_factors[i];

        // load neighbor IDs, filter by hash (skip already-expanded nodes)
        Span<const vid_t> nbrs = graph_.nbrs(current_node);
        for (unsigned i = lane; i < max_degree_; i += BX) {
            INDEX_T child = nbrs[i];
            if (child != MAX_INDEX && hf_.contains(child))
                child = MAX_INDEX;
            cand_idx_[i] = child;
        }
        __syncwarp();

        // L2 distance from query to expanding node
        DISTANCE_T exact_dist = l2_warp(
            dim, query_buf_, graph_.vect(current_node).data());

        // fastscan distance estimation
        int pdim = rabitq_.padded_dim();
        int num_batches = (max_degree_ + WARP_SIZE - 1) / WARP_SIZE;
        for (int batch = 0; batch < num_batches; batch++) {
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
            if (lane < static_cast<unsigned>(active)) {
                if (cand_idx_[batch_begin + lane] == MAX_INDEX)
                    cand_dist_[batch_begin + lane] = FLT_MAX;
            }
        }
        __syncwarp();

        // merge computed distance and neighbors into frontier
        {
            float    d[REG];
            uint32_t x[REG];
            #pragma unroll
            for (unsigned j = 0; j < REG; j++) {
                unsigned pos = lane * REG + j;
                d[j] = (pos < max_degree_) ? cand_dist_[pos] : FLT_MAX;
                x[j] = (pos < max_degree_) ? cand_idx_[pos]  : MAX_INDEX;
            }
            topk_.merge(d, x);
        }
    }

    // copy results into output field
    topk_.flush(cand_dist_, results, K);
}
