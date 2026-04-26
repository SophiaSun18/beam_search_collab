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
 *  @class SharedBitonicTopK shared memory topk buffer for bitonic sorting
 *  @tparam BX number of threads in the block
 *  @tparam REG rough register pressure parameter per thread
 */
template <unsigned BX, unsigned REG>
class SharedBitonicTopK {
    private:
    static constexpr unsigned N = BX * REG;
    unsigned  topk;       // the number of elements in this buffer
    float*    topk_dist;  // points into shared memory, stored descending
    uint32_t* topk_idx;   // points into shared memory, stored descending
    public:
    /**
     * @brief Construct a top-k buffer backed by shared memory.
     *
     * The caller is responsible for allocating shared memory arrays of at
     * least `topk` elements each. The buffer must be pre-filled in
     * descending order by distance before the first merge call.
     *
     * @param topk       Number of elements the buffer holds.
     * @param topk_dist  Shared memory array of distances, size >= topk.
     * @param topk_idx   Shared memory array of indices, size >= topk.
     */
    __device__ SharedBitonicTopK(unsigned topk, float* topk_dist, uint32_t* topk_idx)
        : topk(topk), topk_dist(topk_dist), topk_idx(topk_idx) {}
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
     * @param idx   Output array receiving the buffer positions of picked nodes.
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

template <unsigned BX, unsigned REG>
__device__ void SharedBitonicTopK<BX, REG>::merge(float (&dist)[REG], uint32_t (&idx)[REG]) {
    // Scratch for cross-warp bitonic sort steps. When BX <= 32 (single warp),
    // the cross-warp path is dead code and the compiler eliminates these.
    __shared__ float    scratch_d[N];
    __shared__ uint32_t scratch_i[N];
    const unsigned tid = threadIdx.x;
    const unsigned chunks = topk / N;
    // [1] Sort incoming candidates ascending (from arbitrary order).
    bitonic_sort_full(dist, idx, scratch_d, scratch_i);
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
            bitonic_sort_block<false>(dist_, idx_, scratch_d, scratch_i);
            #pragma unroll
            for (unsigned j = 0; j < REG; j++) {
                unsigned pos = (c - 1) * N + tid * REG + j;
                topk_dist[pos] = dist_[j];
                topk_idx[pos]  = idx_[j];
            }
            __syncthreads();
        }
        // [6] Sort lower half ascending, ready for next chunk
        bitonic_sort_block<true>(dist, idx, scratch_d, scratch_i);
    }
    // [7] Flush last chunk descending
    bitonic_sort_block<false>(dist, idx, scratch_d, scratch_i);
    #pragma unroll
    for (unsigned j = 0; j < REG; j++) {
        unsigned pos = (chunks - 1) * N + tid * REG + j;
        topk_dist[pos] = dist[j];
        topk_idx[pos]  = idx[j];
    }
    __syncthreads();
}

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
            out[found + lane_prefix] = base + tid;
            topk_idx[base + tid] = val | MSB;
        }

        found += count;
        if (base == 0) break;
        base = (base > 32) ? base - 32 : 0;
    }

    return (found < n) ? found : n;
}

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

/**
 * @brief Full bitonic sort of BX*REG key-value pairs from arbitrary order.
 *        Always produces ascending output.
 *
 * Unlike bitonic_sort_block (which assumes bitonic input), this runs
 * the complete bitonic sorting network from unsorted data.
 * Direction: ((global_pos & len) == 0) — naturally ascending at top level
 * since global_pos < TOTAL = len at the final pass.
 *
 * @param scratch_d   Shared memory scratch for floats, size >= N.
 * @param scratch_i   Shared memory scratch for uint32_ts, size >= N.
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

/**
 * @brief Pairwise min/max split on REG register pairs.
 *
 * After the call, a[j] holds min(a[j], b[j]) and b[j] holds max(a[j], b[j])
 * for every j in [0, REG). When both inputs are sorted halves of a bitonic
 * sequence, the result is two bitonic subsequences where every element in a
 * is <= every element in b.
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

/**
 * @brief Bitonic sort of REG key-value pairs within a single thread.
 *
 * Applies the full bitonic sorting network over the REG elements held in
 * registers by one thread. No cross-thread communication.
 *
 * @tparam asc  true for ascending output, false for descending.
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

/**
 * @brief Bitonic sort of BX*REG key-value pairs distributed across BX threads.
 *
 * Each thread holds REG elements. The sort first calls bitonic_sort_local,
 * then performs cross-thread merge steps: intra-warp steps use __shfl_xor_sync,
 * cross-warp steps go through the shared-memory scratch buffers.
 *
 * @tparam asc        true for ascending output, false for descending.
 * @param scratch_d   Shared memory scratch for floats, size >= N.
 * @param scratch_i   Shared memory scratch for uint32_ts, size >= N.
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

