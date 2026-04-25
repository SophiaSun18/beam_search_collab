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
    unsigned       	n,
) {
    // Merge per-thread REG sorted candidates into internal storage.
    { buf.merge(dist, idx) } -> std::same_as<void>;

    // Find up to `n` unexpanded nodes (MSB==0), mark them expanded,
    // write their positions to parent_list. Returns count found.
    { buf.pick(idx, n) } -> std::convertible_to<unsigned>;

    // Write internal storage out to result arrays (e.g. global memory).
    { buf.flush((float*)nullptr, (uint32_t*)nullptr, n) } -> std::same_as<void>;
};

/**
 *  @class SharedBitonicTopK shared memory topk buffer for bitonic sorting
 *  @tparam N assumed max number of neighbors (= BX * REG)
 *  @tparam BX number of threads in the block
 *  @tparam REG rough register pressure parameter per thread
 */
template <unsigned N, unsigned BX, unsigned REG>
class SharedBitonicTopK {
    private:
    unsigned  topk;       // the number of elements in this buffer
    float*    topk_dist;  // points into shared memory, stored descending
    uint32_t* topk_idx;   // points into shared memory, stored descending
    public:
    __device__ void merge(float (&dist)[REG], uint32_t (&idx)[REG]);
    __device__ unsigned pick(uint32_t (&idx)[N], unsigned n);
    __device__ void flush(float* dist, uint32_t* idx, unsigned n);
    private: // helpers
    template <bool asc = true>
    __device__ static inline void bitonic_sort_local(
            float (&d)[REG], uint32_t (&i)[REG]);
    template <bool asc = true>
    __device__ static inline void bitonic_sort_block(
            float (&d)[REG], uint32_t (&i)[REG], float* scratch_d, uint32_t* scratch_i);
    __device__ static inline void butterfly(
            float (&a_d)[REG], uint32_t (&a_i)[REG], float (&b_d)[REG], uint32_t (&b_i)[REG]);
};

template <unsigned N, unsigned BX, unsigned REG>
__device__ void SharedBitonicTopK<N, BX, REG>::merge(float (&dist)[REG], uint32_t (&idx)[REG]) {
    static_assert(N == BX * REG, "N must equal BX * REG");
    const unsigned tid = threadIdx.x;
    const unsigned chunks = topk / N;
    // [1] dist/idx arrive ascending (the incoming candidates).
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
            bitonic_sort_block<false>(dist_, idx_, topk_dist, topk_idx);
            #pragma unroll
            for (unsigned j = 0; j < REG; j++) {
                unsigned pos = (c - 1) * N + tid * REG + j;
                topk_dist[pos] = dist_[j];
                topk_idx[pos]  = idx_[j];
            }
            __syncthreads();
        }
        // [6] Sort lower half ascending, ready for next chunk
        bitonic_sort_block<true>(dist, idx, topk_dist, topk_idx);
    }
    // [7] Flush last chunk descending
    bitonic_sort_block<false>(dist, idx, topk_dist, topk_idx);
    #pragma unroll
    for (unsigned j = 0; j < REG; j++) {
        unsigned pos = (chunks - 1) * N + tid * REG + j;
        topk_dist[pos] = dist[j];
        topk_idx[pos]  = idx[j];
    }
    __syncthreads();
}

template <unsigned N, unsigned BX, unsigned REG>
__device__ unsigned SharedBitonicTopK<N, BX, REG>::pick(uint32_t (&out)[N], unsigned n) {
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

template <unsigned N, unsigned BX, unsigned REG>
__device__ void SharedBitonicTopK<N, BX, REG>::flush(float* dist, uint32_t* idx, unsigned n) {
    const unsigned tid = threadIdx.x;
    // Copy the n smallest elements (at the tail of the descending buffer).
    for (unsigned i = tid; i < n; i += BX) {
        unsigned pos = topk - n + i;
        dist[i] = topk_dist[pos];
        idx[i]  = topk_idx[pos] & ~0x80000000u;  // clear MSB mark
    }
}

// Pairwise min/max split.
//   a[j] = min(a[j], b[j]),  b[j] = max(a[j], b[j])
template <unsigned N, unsigned BX, unsigned REG>
__device__ inline void SharedBitonicTopK<N, BX, REG>::butterfly(
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

// Bitonic sort REG elements within a single thread.
template <unsigned N, unsigned BX, unsigned REG>
template <bool asc>
__device__ inline void SharedBitonicTopK<N, BX, REG>::bitonic_sort_local(float (&d)[REG], uint32_t (&i)[REG]) {
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

// Bitonic sort across BX threads, each holding REG elements.
// Total sorted elements = BX * REG.
template <unsigned N, unsigned BX, unsigned REG>
template <bool asc>
__device__ inline void SharedBitonicTopK<N, BX, REG>::bitonic_sort_block(
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

