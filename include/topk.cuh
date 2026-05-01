#pragma once
#include <concepts>
#include <cstdint>
#include <cfloat>
#include <cuda_runtime.h>

/*
 * TopKBuffer — self-contained warp-collaborative sorted priority queue.
 * Owns its sorted (distance, index) storage internally.
 * All methods called in warp lockstep.
 */
template <typename B, unsigned REG>
concept TopKBuffer = requires(
    B              	buf,
    float           (&dist)[REG],
    uint32_t        (&idx)[REG],
    unsigned       	n
) {
    // Merge per-thread REG candidates (unsorted) into internal storage.
    { buf.merge(dist, idx) } -> std::same_as<void>;

    // Find up to `n` unexpanded nodes (MSB==0), mark them expanded,
    // write their positions to parent_list. Returns count found.
    { buf.pick(idx, n) } -> std::convertible_to<unsigned>;

    // Write internal storage out to result arrays (e.g. global memory).
    { buf.flush((float*)nullptr, (uint32_t*)nullptr, n) } -> std::same_as<void>;
};

/**
 *  @struct BitonicScratchBase — conditional storage for cross-warp scratch.
 *
 *  When HasScratch is true (BX > 32), stores two pointers (16 bytes).
 *  When false (BX <= 32), empty — zero bytes via EBO.
 */
template <bool HasScratch>
struct BitonicScratchBase {
    float*    scratch_d_ = nullptr;
    uint32_t* scratch_i_ = nullptr;
    __device__ void       set_scratch(float* d, uint32_t* i) { scratch_d_ = d; scratch_i_ = i; }
    __device__ float*     get_scratch_d() const { return scratch_d_; }
    __device__ uint32_t*  get_scratch_i() const { return scratch_i_; }
};
template <>
struct BitonicScratchBase<false> {
    __device__ void       set_scratch(float*, uint32_t*) {}
    __device__ float*     get_scratch_d() const { return nullptr; }
    __device__ uint32_t*  get_scratch_i() const { return nullptr; }
};

/**
 *  @class SharedBitonicTopK shared memory topk buffer for bitonic sorting
 *
 *  @tparam BX  Number of threads participating in one instance.
 *  @tparam REG Registers per thread (candidates per thread per merge).
 *
 *  ## Supported thread layouts
 *
 *  | BX       | BY (instances/block) | Block shape      | Scratch | Use case                  |
 *  |----------|----------------------|------------------|---------|---------------------------|
 *  | 32       | >= 1                 | dim3(32, BY)     | No      | Warp-level, independent   |
 *  | > 32     | 1                    | dim3(BX)         | Yes     | Block-level, single query |
 *
 *  BX > 32 with BY > 1 is NOT supported. The __syncthreads used for
 *  cross-warp sort steps would synchronize across independent queries,
 *  causing deadlocks when queries converge at different iterations.
 *
 *  When BX <= 32 (single warp): all sort steps use __shfl_xor_sync,
 *  scratch buffers are unused (nullptr), and sync() uses __syncwarp.
 *  When BX > 32 (multi-warp): cross-warp sort steps use scratch buffers
 *  in shared memory, and sync() uses __syncthreads.
 */
template <unsigned BX, unsigned REG>
class SharedBitonicTopK : private BitonicScratchBase<(BX > 32)> {
    private:
    static constexpr unsigned N = BX * REG;
    unsigned  topk;       // the number of elements in this buffer
    float*    topk_dist;  // points into shared memory, stored descending
    uint32_t* topk_idx;   // points into shared memory, stored descending

    /// Correct sync for the number of participating threads.
    /// BX <= 32 (single warp): __syncwarp.
    /// BX > 32 (multi-warp, BY must be 1): __syncthreads.
    __device__ static void sync();

    public:
    /**
     * @brief Construct a top-k buffer backed by caller-provided shared memory.
     *
     * Initializes all slots to FLT_MAX / 0xFFFFFFFF. All BX threads must call.
     * When BX > 32, scratch_d and scratch_i must point to shared memory arrays
     * of at least N elements each for cross-warp sort steps.
     * When BX <= 32, scratch pointers are ignored (all sorts use shfl).
     *
     * @param topk       Number of elements the buffer holds.
     * @param topk_dist  Shared memory array of distances, size >= topk.
     * @param topk_idx   Shared memory array of indices, size >= topk.
     * @param scratch_d  Scratch for cross-warp sort (nullptr ok when BX <= 32).
     * @param scratch_i  Scratch for cross-warp sort (nullptr ok when BX <= 32).
     */
    __device__ SharedBitonicTopK(unsigned topk, float* topk_dist, uint32_t* topk_idx,
                                  float* scratch_d = nullptr, uint32_t* scratch_i = nullptr);

    /**
     * @brief Self-allocating constructor. Carves all storage from smem.
     *
     * When blockDim.y > 1, each instance (identified by threadIdx.y)
     * gets its own region within the block-wide smem. All instances
     * advance smem to the same final position (past all ny instances).
     *
     * Allocates topk dist/idx arrays, and scratch buffers (if BX > 32).
     * Initializes all slots to FLT_MAX / 0xFFFFFFFF. All BX threads must call.
     */
    __device__ SharedBitonicTopK(char*& smem, unsigned topk);

    /**
     * @brief Shared memory bytes per instance (one topk buffer + scratch).
     */
    __host__ __device__ static size_t shared_mem_bytes_per_instance(unsigned topk) {
        size_t size = alignof(float) - 1;  // worst-case initial alignment
        size += topk * (sizeof(float) + sizeof(uint32_t));
        if constexpr (BX > 32)
            size += N * (sizeof(float) + sizeof(uint32_t));
        return size;
    }

    /**
     * @brief Total shared memory bytes for ny instances.
     *
     * @param topk  Elements per instance.
     * @param ny    Number of instances (blockDim.y). Default 1.
     */
    __host__ __device__ static size_t shared_mem_bytes(unsigned topk, unsigned ny = 1) {
        return ny * shared_mem_bytes_per_instance(topk);
    }

    /**
     * @brief Merge candidates into the descending internal buffer.
     *
     * Input may be unsorted — the buffer sorts it internally. The internal
     * buffer must be sorted descending and its size must be a multiple of N.
     *
     * @param dist  Per-thread REG distances (unsorted).
     * @param idx   Per-thread REG indices, corresponding to dist.
     */
    __device__ void merge(float (&dist)[REG], uint32_t (&idx)[REG]);
    /**
     * @brief Pick up to n unmarked nodes with the smallest distances.
     *
     * Scans the buffer from the tail (smallest distances). Unmarked means
     * MSB of the index is 0 and the index is not 0xFFFFFFFF. Found nodes
     * are marked (MSB set to 1). Only warp 0 (threads 0..31) participates.
     *
     * @param idx   Output array receiving the node IDs of picked nodes (MSB clear).
     * @param n     Maximum number of nodes to pick.
     * @return      Actual number of nodes picked (<= n).
     */
    __device__ unsigned pick(uint32_t (&idx)[N], unsigned n);
    /**
     * @brief Flush the n smallest elements to global memory.
     *
     * Reads from the tail of the descending buffer. Clears the MSB mark
     * on output indices. All BX threads participate with strided access.
     *
     * @param dist  Output distance array in global memory, size >= n.
     * @param idx   Output index array in global memory, size >= n.
     * @param n     Number of elements to write, must be <= topk.
     */
    __device__ void flush(float* dist, uint32_t* idx, unsigned n);
    // -- helpers
    template <bool asc = true>
    __device__ static inline void bitonic_sort_local(
            float (&d)[REG], uint32_t (&i)[REG]);
    template <bool asc = true>
    __device__ static inline void bitonic_sort_block(
            float (&d)[REG], uint32_t (&i)[REG], float* scratch_d, uint32_t* scratch_i);
    __device__ static inline void bitonic_sort_full(
            float (&d)[REG], uint32_t (&i)[REG], float* scratch_d, uint32_t* scratch_i);
    __device__ static inline void butterfly(
            float (&a_d)[REG], uint32_t (&a_i)[REG], float (&b_d)[REG], uint32_t (&b_i)[REG]);
};

// ── sync ────────────────────────────────────────────────────────────────────

template <unsigned BX, unsigned REG>
__device__ void SharedBitonicTopK<BX, REG>::sync() {
    // BX <= 32: single warp, independent of other warps in the block.
    // BX > 32: multi-warp, requires block-wide barrier (BY must be 1).
    if constexpr (BX <= 32) __syncwarp(); else __syncthreads();
}

// ── constructors ────────────────────────────────────────────────────────────

template <unsigned BX, unsigned REG>
__device__ SharedBitonicTopK<BX, REG>::SharedBitonicTopK(
    unsigned topk, float* topk_dist, uint32_t* topk_idx,
    float* scratch_d, uint32_t* scratch_i)
    : topk(topk), topk_dist(topk_dist), topk_idx(topk_idx)
{
    this->set_scratch(scratch_d, scratch_i);
    for (unsigned i = threadIdx.x; i < topk; i += BX) {
        topk_dist[i] = FLT_MAX;
        topk_idx[i]  = 0xFFFFFFFF;
    }
}

template <unsigned BX, unsigned REG>
__device__ SharedBitonicTopK<BX, REG>::SharedBitonicTopK(char*& smem, unsigned topk)
    : topk(topk)
{
    // Each instance (threadIdx.y) gets its own region within block-wide smem.
    const size_t per_inst = shared_mem_bytes_per_instance(topk);
    char* my = smem + threadIdx.y * per_inst;

    // align for float/uint32_t
    my = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(my) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));
    topk_dist = reinterpret_cast<float*>(my);
    topk_idx  = reinterpret_cast<uint32_t*>(my + topk * sizeof(float));
    my += topk * (sizeof(float) + sizeof(uint32_t));

    // Allocate scratch only when BX > 32 (cross-warp sort steps need it).
    // When BX <= 32, all sort steps use __shfl_xor_sync — no scratch needed.
    if constexpr (BX > 32) {
        this->set_scratch(
            reinterpret_cast<float*>(my),
            reinterpret_cast<uint32_t*>(my + N * sizeof(float)));
    }

    // Advance smem past ALL instances so every thread ends at the same position.
    smem += blockDim.y * per_inst;

    for (unsigned i = threadIdx.x; i < topk; i += BX) {
        topk_dist[i] = FLT_MAX;
        topk_idx[i]  = 0xFFFFFFFF;
    }
}

// ── merge ───────────────────────────────────────────────────────────────────

template <unsigned BX, unsigned REG>
__device__ void SharedBitonicTopK<BX, REG>::merge(float (&dist)[REG], uint32_t (&idx)[REG]) {
    const unsigned tid = threadIdx.x;
    const unsigned chunks = topk / N;
    float*    sd = this->get_scratch_d();
    uint32_t* si = this->get_scratch_i();
    // [1] Sort incoming candidates ascending (from arbitrary order).
    bitonic_sort_full(dist, idx, sd, si);
    for (unsigned c = 0; c < chunks; c++) {
        // [2] Load chunk c into registers
        float    dist_[REG];
        uint32_t idx_[REG];
        #pragma unroll
        for (unsigned j = 0; j < REG; j++) {
            unsigned pos = c * N + tid * REG + j;
            dist_[j] = topk_dist[pos];
            idx_[j]  = topk_idx[pos];
        }
        // [3] Butterfly: dist gets min (lower half), dist_ gets max (upper half)
        butterfly(dist, idx, dist_, idx_);
        // [5] If not first chunk, sort upper half descending and flush to previous chunk
        if (c > 0) {
            bitonic_sort_block<false>(dist_, idx_, sd, si);
            #pragma unroll
            for (unsigned j = 0; j < REG; j++) {
                unsigned pos = (c - 1) * N + tid * REG + j;
                topk_dist[pos] = dist_[j];
                topk_idx[pos]  = idx_[j];
            }
            sync();
        }
        // [6] Sort lower half ascending, ready for next chunk
        bitonic_sort_block<true>(dist, idx, sd, si);
    }
    // [7] Flush last chunk descending
    bitonic_sort_block<false>(dist, idx, sd, si);
    #pragma unroll
    for (unsigned j = 0; j < REG; j++) {
        unsigned pos = (chunks - 1) * N + tid * REG + j;
        topk_dist[pos] = dist[j];
        topk_idx[pos]  = idx[j];
    }
    sync();
}

// ── pick ────────────────────────────────────────────────────────────────────

template <unsigned BX, unsigned REG>
__device__ unsigned SharedBitonicTopK<BX, REG>::pick(uint32_t (&out)[N], unsigned n) {
    constexpr uint32_t MSB = 0x80000000;
    const unsigned tid = threadIdx.x;

    unsigned found = 0;

    // Smallest elements at the tail (descending buffer). Scan backwards.
    unsigned base = (topk > 32) ? topk - 32 : 0;
    while (found < n) {
        uint32_t val = 0;
        bool unmarked = false;
        if (tid < 32 && base + tid < topk) {
            val = topk_idx[base + tid];
            unmarked = ((val & MSB) == 0) && (val != 0xFFFFFFFF);
        }

        uint32_t ballot = __ballot_sync(0xffffffff, unmarked);
        if (ballot == 0) {
            if (base == 0) break;
            base = (base > 32) ? base - 32 : 0;
            continue;
        }

        unsigned count = __popc(ballot);
        unsigned lane_prefix = __popc(ballot & ((1u << tid) - 1));

        if (unmarked && found + lane_prefix < n) {
            out[found + lane_prefix] = val;  // node ID (MSB clear)
            topk_idx[base + tid] = val | MSB;
        }

        found += count;
        if (base == 0) break;
        base = (base > 32) ? base - 32 : 0;
    }

    return (found < n) ? found : n;
}

// ── flush ───────────────────────────────────────────────────────────────────

template <unsigned BX, unsigned REG>
__device__ void SharedBitonicTopK<BX, REG>::flush(float* dist, uint32_t* idx, unsigned n) {
    const unsigned tid = threadIdx.x;
    // Copy the n smallest elements (at the tail of the descending buffer).
    for (unsigned i = tid; i < n; i += BX) {
        unsigned pos = topk - n + i;
        dist[i] = topk_dist[pos];
        idx[i]  = topk_idx[pos] & ~0x80000000u;  // clear MSB mark
    }
}

// ── bitonic_sort_full ───────────────────────────────────────────────────────

/**
 * @brief Full bitonic sort of BX*REG key-value pairs from arbitrary order.
 *        Always produces ascending output.
 *
 * @param scratch_d   Shared memory scratch for floats (nullptr when BX <= 32).
 * @param scratch_i   Shared memory scratch for uint32_ts (nullptr when BX <= 32).
 */
template <unsigned BX, unsigned REG>
__device__ inline void SharedBitonicTopK<BX, REG>::bitonic_sort_full(
    float (&d)[REG], uint32_t (&i)[REG],
    float* scratch_d, uint32_t* scratch_i)
{
    const unsigned tid = threadIdx.x;
    constexpr unsigned TOTAL = BX * REG;

    for (unsigned len = 2; len <= TOTAL; len <<= 1) {
        for (unsigned stride = len >> 1; stride >= 1; stride >>= 1) {
            if (stride >= REG) {
                unsigned thread_stride = stride / REG;
                if (thread_stride < 32) {
                    // Intra-warp: use shfl (works for any BX)
                    unsigned partner = tid ^ thread_stride;
                    #pragma unroll
                    for (unsigned j = 0; j < REG; j++) {
                        float    pd = __shfl_xor_sync(0xffffffff, d[j], thread_stride);
                        uint32_t pi = __shfl_xor_sync(0xffffffff, i[j], thread_stride);
                        unsigned gp = tid * REG + j;
                        bool dir = ((gp & len) == 0);
                        bool is_lower = (tid < partner);
                        bool want_smaller = (dir == is_lower);
                        if (want_smaller ? (d[j] > pd) : (d[j] < pd)) {
                            d[j] = pd;
                            i[j] = pi;
                        }
                    }
                } else {
                    // Cross-warp: use scratch (only reached when BX > 32)
                    #pragma unroll
                    for (unsigned j = 0; j < REG; j++) {
                        scratch_d[tid * REG + j] = d[j];
                        scratch_i[tid * REG + j] = i[j];
                    }
                    __syncthreads();
                    unsigned partner = tid ^ thread_stride;
                    #pragma unroll
                    for (unsigned j = 0; j < REG; j++) {
                        float    pd = scratch_d[partner * REG + j];
                        uint32_t pi = scratch_i[partner * REG + j];
                        unsigned gp = tid * REG + j;
                        bool dir = ((gp & len) == 0);
                        bool is_lower = (tid < partner);
                        bool want_smaller = (dir == is_lower);
                        if (want_smaller ? (d[j] > pd) : (d[j] < pd)) {
                            d[j] = pd;
                            i[j] = pi;
                        }
                    }
                    __syncthreads();
                }
            } else {
                #pragma unroll
                for (unsigned k = 0; k < REG; k++) {
                    unsigned j = k ^ stride;
                    if (j > k) {
                        unsigned gp = tid * REG + k;
                        bool dir = ((gp & len) == 0);
                        if ((d[k] > d[j]) == dir) {
                            float    td = d[k]; d[k] = d[j]; d[j] = td;
                            uint32_t ti = i[k]; i[k] = i[j]; i[j] = ti;
                        }
                    }
                }
            }
        }
    }
}

// ── butterfly ───────────────────────────────────────────────────────────────

/**
 * @brief Pairwise min/max split on REG register pairs.
 */
template <unsigned BX, unsigned REG>
__device__ inline void SharedBitonicTopK<BX, REG>::butterfly(
    float (&a_d)[REG], uint32_t (&a_i)[REG],
    float (&b_d)[REG], uint32_t (&b_i)[REG])
{
    #pragma unroll
    for (unsigned j = 0; j < REG; j++) {
        if (a_d[j] > b_d[j]) {
            float    td = a_d[j]; a_d[j] = b_d[j]; b_d[j] = td;
            uint32_t ti = a_i[j]; a_i[j] = b_i[j]; b_i[j] = ti;
        }
    }
}

// ── bitonic_sort_local ──────────────────────────────────────────────────────

/**
 * @brief Bitonic sort of REG key-value pairs within a single thread.
 *        No cross-thread communication.
 */
template <unsigned BX, unsigned REG>
template <bool asc>
__device__ inline void SharedBitonicTopK<BX, REG>::bitonic_sort_local(float (&d)[REG], uint32_t (&i)[REG]) {
    #pragma unroll
    for (unsigned len = 2; len <= REG; len <<= 1) {
        #pragma unroll
        for (unsigned stride = len >> 1; stride >= 1; stride >>= 1) {
            #pragma unroll
            for (unsigned k = 0; k < REG; k++) {
                unsigned j = k ^ stride;
                if (j > k) {
                    bool dir = ((k & len) == 0) == asc;
                    if ((d[k] > d[j]) == dir) {
                        float    td = d[k]; d[k] = d[j]; d[j] = td;
                        uint32_t ti = i[k]; i[k] = i[j]; i[j] = ti;
                    }
                }
            }
        }
    }
}

// ── bitonic_sort_block ──────────────────────────────────────────────────────

/**
 * @brief Bitonic sort of BX*REG key-value pairs distributed across BX threads.
 *
 * Intra-warp steps use __shfl_xor_sync. Cross-warp steps (only when BX > 32)
 * go through the shared-memory scratch buffers.
 *
 * @param scratch_d   Shared memory scratch for floats (nullptr when BX <= 32).
 * @param scratch_i   Shared memory scratch for uint32_ts (nullptr when BX <= 32).
 */
template <unsigned BX, unsigned REG>
template <bool asc>
__device__ inline void SharedBitonicTopK<BX, REG>::bitonic_sort_block(
    float (&d)[REG], uint32_t (&i)[REG],
    float* scratch_d, uint32_t* scratch_i)
{
    const unsigned tid = threadIdx.x;
    constexpr unsigned TOTAL = BX * REG;
    bitonic_sort_local<asc>(d, i);
    for (unsigned len = 2 * REG; len <= TOTAL; len <<= 1) {
        for (unsigned stride = len >> 1; stride >= 1; stride >>= 1) {
            if (stride >= REG) {
                unsigned thread_stride = stride / REG;
                if (thread_stride < 32) {
                    // Intra-warp: use shfl
                    unsigned partner = tid ^ thread_stride;
                    #pragma unroll
                    for (unsigned j = 0; j < REG; j++) {
                        float    pd = __shfl_xor_sync(0xffffffff, d[j], thread_stride);
                        uint32_t pi = __shfl_xor_sync(0xffffffff, i[j], thread_stride);
                        unsigned global_pos = tid * REG + j;
                        bool dir = ((global_pos & len) == 0) == asc;
                        bool is_lower = (tid < partner);
                        bool want_smaller = (dir == is_lower);
                        if (want_smaller ? (d[j] > pd) : (d[j] < pd)) {
                            d[j] = pd;
                            i[j] = pi;
                        }
                    }
                } else {
                    // Cross-warp: use scratch (only reached when BX > 32)
                    #pragma unroll
                    for (unsigned j = 0; j < REG; j++) {
                        scratch_d[tid * REG + j] = d[j];
                        scratch_i[tid * REG + j] = i[j];
                    }
                    __syncthreads();
                    unsigned partner = tid ^ thread_stride;
                    #pragma unroll
                    for (unsigned j = 0; j < REG; j++) {
                        float    pd = scratch_d[partner * REG + j];
                        uint32_t pi = scratch_i[partner * REG + j];
                        unsigned global_pos = tid * REG + j;
                        bool dir = ((global_pos & len) == 0) == asc;
                        bool is_lower = (tid < partner);
                        bool want_smaller = (dir == is_lower);
                        if (want_smaller ? (d[j] > pd) : (d[j] < pd)) {
                            d[j] = pd;
                            i[j] = pi;
                        }
                    }
                    __syncthreads();
                }
            } else {
                #pragma unroll
                for (unsigned k = 0; k < REG; k++) {
                    unsigned j = k ^ stride;
                    if (j > k) {
                        unsigned global_pos_k = tid * REG + k;
                        bool dir = ((global_pos_k & len) == 0) == asc;
                        if ((d[k] > d[j]) == dir) {
                            float    td = d[k]; d[k] = d[j]; d[j] = td;
                            uint32_t ti = i[k]; i[k] = i[j]; i[j] = ti;
                        }
                    }
                }
            }
        }
    }
}
