#pragma once
#include "gpu_base_search.cuh"
#include <cuda_runtime.h>
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
#include <cuda_pipeline_primitives.h>
#endif

/**
 * @class GPUPipeSearch — warp-level pipelined L2 beam search visitor.
 *
 * Same warp-level structure as GPUWarpSearch (dim3(32, NY), each warp
 * handles one query with __syncwarp), but the per-neighbor L2 loop is
 * double-buffered: while computing L2 for the current neighbor from
 * shared memory, the next neighbor's vector is async-loaded into the
 * alternate staging slot via 128-bit cp.async.
 *
 * @tparam NY   Number of warps (queries) per block.
 * @tparam REG  Registers per thread for topk merge.
 */
template <unsigned NY, unsigned REG = 2>
class GPUPipeSearch {
    static constexpr unsigned BX = WARP_SIZE;  // 32

    SharedBitonicTopK<BX, REG> topk_;
    HashFilter                 hf_;
    float*                     query_buf_;     // [dim]
    INDEX_T*                   parent_buf_;    // [1]
    INDEX_T*                   cand_idx_;      // [degree]
    DISTANCE_T*                cand_dist_;     // [degree]
    uint32_t*                  compact_map_;   // [degree]
    float*                     staged_vecs_;   // [2 * dim] double-buffer

    GPUGraphView               graph_;
    GPUBaseSearchCfg           cfg_;
    unsigned                   dim_;
    unsigned                   max_degree_;

public:
    __device__ GPUPipeSearch(char*& smem, const GPUBaseSearchCfg& cfg,
                             GPUGraphView graph);

    __device__ void search(const float* query, vid_t* results, int K);

    __host__ __device__ static size_t shared_mem_bytes_per_warp(
        const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree);

    __host__ __device__ static size_t shared_mem_bytes(
        const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree);

private:
    __device__ void compute_distances_pipelined();
};

// ── constructor ─────────────────────────────────────────────────────────────

template <unsigned NY, unsigned REG>
__device__ GPUPipeSearch<NY, REG>::GPUPipeSearch(
    char*& smem, const GPUBaseSearchCfg& cfg, GPUGraphView graph)
    : topk_(smem, cfg.topk_size),
      hf_(smem, cfg.bitlen),
      graph_(graph), cfg_(cfg),
      dim_(graph.dim()), max_degree_(graph.degree())
{
    // Align smem to 16 bytes so every per-warp region starts aligned.
    smem = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(smem) + 15) & ~static_cast<uintptr_t>(15));

    const size_t plain = graph.dim() * sizeof(float)    // query_buf_
                       + sizeof(INDEX_T)                // parent_buf_
                       + max_degree_ * sizeof(INDEX_T)  // cand_idx_
                       + max_degree_ * sizeof(DISTANCE_T) // cand_dist_
                       + max_degree_ * sizeof(uint32_t);  // compact_map_
    const size_t plain_pad = (plain + 15) & ~static_cast<size_t>(15);
    const size_t staged = 2 * graph.dim() * sizeof(float);
    // Round total per-warp size to 16 so every warp starts 16-byte aligned.
    const size_t rest = (plain_pad + staged + 15) & ~static_cast<size_t>(15);

    char* my = smem + threadIdx.y * rest;
    query_buf_   = reinterpret_cast<float*>(my);       my += graph.dim() * sizeof(float);
    parent_buf_  = reinterpret_cast<INDEX_T*>(my);     my += sizeof(INDEX_T);
    cand_idx_    = reinterpret_cast<INDEX_T*>(my);     my += max_degree_ * sizeof(INDEX_T);
    cand_dist_   = reinterpret_cast<DISTANCE_T*>(my);  my += max_degree_ * sizeof(DISTANCE_T);
    compact_map_ = reinterpret_cast<uint32_t*>(my);
    staged_vecs_ = reinterpret_cast<float*>(smem + threadIdx.y * rest + plain_pad);

    smem += blockDim.y * rest;
}

// ── shared memory size ──────────────────────────────────────────────────────

template <unsigned NY, unsigned REG>
__host__ __device__ size_t GPUPipeSearch<NY, REG>::shared_mem_bytes_per_warp(
    const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree) {
    // topk + hash are allocated before the 16-byte aligned per-warp region
    size_t prefix = 0;
    prefix += SharedBitonicTopK<BX, REG>::shared_mem_bytes_per_instance(cfg.topk_size);
    prefix += (1u << cfg.bitlen) * sizeof(uint32_t);   // hash filter

    size_t plain = dim * sizeof(float)                 // query_buf_
                 + sizeof(INDEX_T)                     // parent_buf_
                 + degree * sizeof(INDEX_T)            // cand_idx_
                 + degree * sizeof(DISTANCE_T)         // cand_dist_
                 + degree * sizeof(uint32_t);          // compact_map_
    size_t plain_pad = (plain + 15) & ~static_cast<size_t>(15);
    size_t staged = 2 * dim * sizeof(float);           // staged_vecs_
    size_t per_warp = (plain_pad + staged + 15) & ~static_cast<size_t>(15);
    return prefix + per_warp;
}

template <unsigned NY, unsigned REG>
__host__ __device__ size_t GPUPipeSearch<NY, REG>::shared_mem_bytes(
    const GPUBaseSearchCfg& cfg, unsigned dim, unsigned degree) {
    // topk and hash handle NY internally; plain+staged are per-warp.
    size_t size = 0;
    size += SharedBitonicTopK<BX, REG>::shared_mem_bytes(cfg.topk_size, NY);
    size += HashFilter::shared_mem_bytes(cfg.bitlen, NY);
    size = (size + 15) & ~static_cast<size_t>(15);  // 16-byte align
    size_t plain = dim * sizeof(float) + sizeof(INDEX_T)
                 + degree * sizeof(INDEX_T) + degree * sizeof(DISTANCE_T)
                 + degree * sizeof(uint32_t);
    size_t plain_pad = (plain + 15) & ~static_cast<size_t>(15);
    size_t staged = 2 * dim * sizeof(float);
    size_t per_warp = (plain_pad + staged + 15) & ~static_cast<size_t>(15);
    size += NY * per_warp;
    return size;
}

// ── search ──────────────────────────────────────────────────────────────────

template <unsigned NY, unsigned REG>
__device__ void GPUPipeSearch<NY, REG>::search(
    const float* query, vid_t* results, int K)
{
    const unsigned lane = threadIdx.x;
    const int dim = static_cast<int>(dim_);

    // ── load query ──────────────────────────────────────────────────────
    for (unsigned i = lane; i < dim_; i += BX)
        query_buf_[i] = query[i];

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

        // ── hash reset + re-populate ────────────────────────────────────
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

        // ── pick ────────────────────────────────────────────────────────
        uint32_t pbuf[BX * REG];
        #pragma unroll
        for (unsigned i = 0; i < BX * REG; ++i) pbuf[i] = MAX_INDEX;
        unsigned found = topk_.pick(pbuf, 1);
        uint32_t picked = MAX_INDEX;
        const bool has_pick = found > 0 && pbuf[0] != MAX_INDEX;
        const uint32_t ballot = __ballot_sync(0xffffffff, has_pick);
        if (ballot != 0) {
            const int src_lane = __ffs(ballot) - 1;
            picked = __shfl_sync(0xffffffff, pbuf[0], src_lane);
        }
        if (lane == 0 && found > 0)
            parent_buf_[0] = picked;
        __syncwarp();

        if (found == 0) break;

        // ── expand: hash filter ─────────────────────────────────────────
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

        // ── pipelined L2 ────────────────────────────────────────────────
        compute_distances_pipelined();
        __syncwarp();
    }

    // ── final merge ─────────────────────────────────────────────────────
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

    topk_.flush(cand_dist_, results, K);
}

// ── compute_distances_pipelined ─────────────────────────────────────────────

/**
 * @brief Pipelined L2 over compacted valid neighbors.
 *
 * Compacts valid entries in cand_idx_ to eliminate pipeline bubbles,
 * then iterates the dense list with double-buffered 128-bit async
 * vector loads. One warp, two dim-sized staging slots.
 */
template <unsigned NY, unsigned REG>
__device__ void GPUPipeSearch<NY, REG>::compute_distances_pipelined()
{
    const unsigned lane = threadIdx.x;
    const int dim = static_cast<int>(dim_);

    // ── compact valid neighbors (warp-collaborative) ────────────────────
    // Each lane checks one or more candidates, ballot to count & place.
    unsigned valid_count = 0;
    for (unsigned base = 0; base < max_degree_; base += WARP_SIZE) {
        unsigned i = base + lane;
        bool valid = (i < max_degree_) && (cand_idx_[i] != MAX_INDEX);
        uint32_t mask = __ballot_sync(0xffffffff, valid);
        unsigned count = __popc(mask);
        unsigned prefix = __popc(mask & ((1u << lane) - 1));
        if (valid)
            compact_map_[valid_count + prefix] = i;
        valid_count += count;
    }

    if (valid_count == 0) return;

    // ── pipelined L2 over compacted list ────────────────────────────────
    constexpr unsigned COPY_WIDTH = 4;  // floats per 128-bit async copy
    const unsigned copies_per_lane = dim_ / (WARP_SIZE * COPY_WIDTH);

    // Pre-load first vector into slot 0
    {
        vid_t child = cand_idx_[compact_map_[0]];
        const float* src = graph_.vect(child).data();
        for (unsigned c = 0; c < copies_per_lane; c++) {
            unsigned d = lane * COPY_WIDTH + c * WARP_SIZE * COPY_WIDTH;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
            __pipeline_memcpy_async(staged_vecs_ + d, src + d,
                                    COPY_WIDTH * sizeof(float));
#else
            for (unsigned k = 0; k < COPY_WIDTH; k++)
                staged_vecs_[d + k] = src[d + k];
#endif
        }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        __pipeline_commit();
#endif
    }

    for (unsigned s = 0; s < valid_count; s++) {
        unsigned cur_slot = s & 1;

        // Pre-load next vector into alternate slot
        if (s + 1 < valid_count) {
            vid_t child = cand_idx_[compact_map_[s + 1]];
            const float* src = graph_.vect(child).data();
            float* dst = staged_vecs_ + ((s + 1) & 1) * dim_;
            for (unsigned c = 0; c < copies_per_lane; c++) {
                unsigned d = lane * COPY_WIDTH + c * WARP_SIZE * COPY_WIDTH;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                __pipeline_memcpy_async(dst + d, src + d,
                                        COPY_WIDTH * sizeof(float));
#else
                for (unsigned k = 0; k < COPY_WIDTH; k++)
                    dst[d + k] = src[d + k];
#endif
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
        const float* vec = staged_vecs_ + cur_slot * dim_;
        DISTANCE_T d = l2_warp(dim, query_buf_, vec);
        if (lane == 0)
            cand_dist_[compact_map_[s]] = d;
        __syncwarp();
    }
}
