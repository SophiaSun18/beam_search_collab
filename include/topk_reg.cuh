#pragma once
#include <cstdint>
#include <cfloat>
#include <cuda_runtime.h>

/**
 * @class RegBitonicTopK — register-resident warp-collaborative sorted priority queue.
 *
 * All state lives in per-thread registers (REGX elements per thread,
 * 32*REGX total across the warp). No shared memory is used for sorting
 * or storage — all cross-lane communication uses __shfl_xor_sync.
 *
 * The buffer is stored in descending order: global position 0 is the
 * largest distance, position 32*REGX-1 is the smallest.
 *
 * @tparam REGX  Registers per thread for the topk buffer. Buffer size = 32 * REGX.
 * @tparam REGY  Registers per thread for incoming candidates. Batch size = 32 * REGY.
 *               REGX must be divisible by REGY.
 */
template <unsigned REGX, unsigned REGY>
class RegBitonicTopK {
    static_assert(REGX >= REGY, "REGX must be >= REGY");
    static_assert(REGX % REGY == 0, "REGX must be divisible by REGY");

    float    topk_d_[REGX];
    uint32_t topk_i_[REGX];

public:
    __device__ RegBitonicTopK();

    /**
     * @brief Merge incoming candidates into the descending internal buffer.
     *
     * Input may be unsorted. REGX must be divisible by REGY. The merge
     * sorts incoming, then sweeps through the topk in REGY-sized chunks
     * via butterfly, keeping the smallest 32*REGX elements.
     */
    __device__ void merge(float (&dist)[REGY], uint32_t (&idx)[REGY]);

    /**
     * @brief Pick up to n unmarked nodes with the smallest distances.
     *
     * Scans from the tail (smallest). Unmarked means MSB==0 and not
     * 0xFFFFFFFF. Found nodes are marked (MSB set). Lane 0 receives
     * the picked node IDs in out[0..found-1].
     */
    __device__ unsigned pick(uint32_t (&out)[REGY], unsigned n);

    /**
     * @brief Flush the n smallest elements to memory.
     *
     * Reads from the tail of the descending buffer. Clears MSB marks.
     */
    __device__ void flush(float* dist, uint32_t* idx, unsigned n);

    /** @brief Number of slots in the buffer. */
    __device__ unsigned capacity() const;

    /** @brief Node ID at position pos (MSB cleared). 0xFFFFFFFF for empty. */
    __device__ uint32_t id(unsigned pos) const;

    // -- helpers
    template <bool asc, unsigned N>
    __device__ static void sort_local(float (&d)[N], uint32_t (&i)[N]);

    template <bool asc, unsigned N>
    __device__ static void sort_warp(float (&d)[N], uint32_t (&i)[N]);

    template <bool asc, unsigned N>
    __device__ static void merge_warp(float (&d)[N], uint32_t (&i)[N]);

    template <unsigned N>
    __device__ static void sort_warp_full(float (&d)[N], uint32_t (&i)[N]);

    template <unsigned N>
    __device__ static void butterfly(
        float (&a_d)[N], uint32_t (&a_i)[N],
        float (&b_d)[N], uint32_t (&b_i)[N]);
};

// ── constructor ─────────────────────────────────────────────────────────────

template <unsigned REGX, unsigned REGY>
__device__ RegBitonicTopK<REGX, REGY>::RegBitonicTopK() {
    #pragma unroll
    for (unsigned j = 0; j < REGX; j++) {
        topk_d_[j] = FLT_MAX;
        topk_i_[j] = 0xFFFFFFFF;
    }
}

template <unsigned REGX, unsigned REGY>
__device__ unsigned RegBitonicTopK<REGX, REGY>::capacity() const {
    return 32 * REGX;
}

template <unsigned REGX, unsigned REGY>
__device__ uint32_t RegBitonicTopK<REGX, REGY>::id(unsigned pos) const {
    unsigned src_lane = pos / REGX;
    unsigned src_reg  = pos % REGX;
    uint32_t val = 0xFFFFFFFF;
    for (unsigned r = 0; r < REGX; r++) {
        uint32_t tmp = __shfl_sync(0xffffffff, topk_i_[r], src_lane);
        if (r == src_reg) val = tmp;
    }
    return val & ~0x80000000u;
}

// ── merge ───────────────────────────────────────────────────────────────────

template <unsigned REGX, unsigned REGY>
__device__ void RegBitonicTopK<REGX, REGY>::merge(
    float (&dist)[REGY], uint32_t (&idx)[REGY])
{
    const unsigned tid = threadIdx.x & 31;
    constexpr unsigned chunks = REGX / REGY;
    constexpr unsigned CAND = 32 * REGY;

    // [1] Sort incoming candidates ascending
    sort_warp_full<REGY>(dist, idx);

    // [2] Compare buffer[j] with candidate[j] for j = 0..CAND-1.
    //     Buffer is descending, candidates ascending → result is bitonic.
    //     Candidate at position j = tid*REGX+r is in REGY-layout:
    //       lane  = tid*chunks + r/REGY
    //       reg   = r % REGY
    #pragma unroll
    for (unsigned r = 0; r < REGX; r++) {
        unsigned j = tid * REGX + r;
        unsigned src_lane = tid * chunks + r / REGY;
        float    cd = __shfl_sync(0xffffffff, dist[r % REGY], src_lane);
        uint32_t ci = __shfl_sync(0xffffffff, idx[r % REGY], src_lane);
        if (j < CAND && topk_d_[r] > cd) {
            topk_d_[r] = cd;
            topk_i_[r] = ci;
        }
    }

    // [3] Buffer is now bitonic — single merge pass to fix
    merge_warp<false, REGX>(topk_d_, topk_i_);
}

// ── pick ────────────────────────────────────────────────────────────────────

template <unsigned REGX, unsigned REGY>
__device__ unsigned RegBitonicTopK<REGX, REGY>::pick(
    uint32_t (&out)[REGY], unsigned n)
{
    constexpr uint32_t MSB = 0x80000000;
    constexpr unsigned TOTAL = 32 * REGX;
    const unsigned lane = threadIdx.x & 31;
    unsigned found = 0;

    // Scan 32-position windows from the tail (smallest distances).
    // Each thread reads one position per window via shuffle.
    for (unsigned base = TOTAL - 32; found < n; base -= 32) {
        // Thread tid checks position base + tid (tid=0 → largest in window)
        unsigned pos = base + lane;
        unsigned src_lane = pos / REGX;
        unsigned src_reg  = pos % REGX;

        // Read topk_i_[src_reg] from thread src_lane
        uint32_t val = 0xFFFFFFFF;
        #pragma unroll
        for (unsigned r = 0; r < REGX; r++) {
            uint32_t tmp = __shfl_sync(0xffffffff, topk_i_[r], src_lane);
            if (r == src_reg) val = tmp;
        }

        bool unmarked = ((val & MSB) == 0) && (val != 0xFFFFFFFF);
        uint32_t ballot = __ballot_sync(0xffffffff, unmarked);

        if (ballot == 0) {
            if (base == 0) break;
            continue;
        }

        while (ballot && found < n) {
            // Highest set bit = highest lane = smallest position in window
            int pick_tid = 31 - __clz(ballot);

            // Broadcast picked info to all threads
            uint32_t picked_val = __shfl_sync(0xffffffff, val, pick_tid);
            unsigned picked_src_lane = __shfl_sync(0xffffffff, src_lane, pick_tid);
            unsigned picked_src_reg  = __shfl_sync(0xffffffff, src_reg, pick_tid);

            // Mark: only the owning thread sets MSB on the correct register
            if (lane == picked_src_lane) {
                #pragma unroll
                for (unsigned r = 0; r < REGX; r++) {
                    if (r == picked_src_reg)
                        topk_i_[r] |= MSB;
                }
            }

            if (lane == 0)
                out[found] = picked_val & ~MSB;
            found++;

            ballot &= ~(1u << pick_tid);
        }

        if (base == 0) break;
    }

    return (found < n) ? found : n;
}

// ── flush ───────────────────────────────────────────────────────────────────

template <unsigned REGX, unsigned REGY>
__device__ void RegBitonicTopK<REGX, REGY>::flush(
    float* dist, uint32_t* idx, unsigned n)
{
    const unsigned lane = threadIdx.x & 31;
    const unsigned total = 32 * REGX;
    const unsigned start = total - n;
    // Round up so all 32 threads iterate the same number of times,
    // ensuring every __shfl_sync has full warp participation.
    const unsigned n_rounded = (n + 31) & ~31u;

    for (unsigned i = lane; i < n_rounded; i += 32) {
        unsigned pos = start + i;
        unsigned src_lane = pos / REGX;
        unsigned src_reg  = pos % REGX;
        float    d = 0;
        uint32_t x = 0;
        #pragma unroll
        for (unsigned r = 0; r < REGX; r++) {
            float    td = __shfl_sync(0xffffffff, topk_d_[r], src_lane);
            uint32_t tx = __shfl_sync(0xffffffff, topk_i_[r], src_lane);
            if (r == src_reg) { d = td; x = tx; }
        }
        if (i < n) {
            dist[i] = d;
            idx[i]  = x & ~0x80000000u;
        }
    }
}

/**
 * @brief Bitonic sort of N key-value pairs within a single thread.
 *        No cross-thread communication.
 */
template <unsigned REGX, unsigned REGY>
template <bool asc, unsigned N>
__device__ void RegBitonicTopK<REGX, REGY>::sort_local(
    float (&d)[N], uint32_t (&i)[N])
{
    #pragma unroll
    for (unsigned len = 2; len <= N; len <<= 1) {
        #pragma unroll
        for (unsigned stride = len >> 1; stride >= 1; stride >>= 1) {
            #pragma unroll
            for (unsigned k = 0; k < N; k++) {
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
 * @brief Bitonic sort of 32*N key-value pairs (N per thread).
 *
 * Input must already be locally sorted within each thread.
 * Uses __shfl_xor_sync only.
 */
template <unsigned REGX, unsigned REGY>
template <bool asc, unsigned N>
__device__ void RegBitonicTopK<REGX, REGY>::sort_warp(
    float (&d)[N], uint32_t (&i)[N])
{
    const unsigned tid = threadIdx.x & 31;
    constexpr unsigned TOTAL = 32 * N;

    sort_local<asc, N>(d, i);

    for (unsigned len = 2 * N; len <= TOTAL; len <<= 1) {
        for (unsigned stride = len >> 1; stride >= 1; stride >>= 1) {
            if (stride >= N) {
                unsigned thread_stride = stride / N;
                unsigned partner = tid ^ thread_stride;
                #pragma unroll
                for (unsigned j = 0; j < N; j++) {
                    float    pd = __shfl_xor_sync(0xffffffff, d[j], thread_stride);
                    uint32_t pi = __shfl_xor_sync(0xffffffff, i[j], thread_stride);
                    unsigned gp = tid * N + j;
                    bool dir = ((gp & len) == 0) == asc;
                    bool is_lower = (tid < partner);
                    bool want_smaller = (dir == is_lower);
                    if (want_smaller ? (d[j] > pd) : (d[j] < pd)) {
                        d[j] = pd;
                        i[j] = pi;
                    }
                }
            } else {
                #pragma unroll
                for (unsigned k = 0; k < N; k++) {
                    unsigned j = k ^ stride;
                    if (j > k) {
                        unsigned gp = tid * N + k;
                        bool dir = ((gp & len) == 0) == asc;
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
 * @brief Single merge pass on a bitonic sequence of 32*N elements.
 *
 * After butterfly between ascending and descending sequences, both output
 * halves are bitonic. This resolves a bitonic sequence into sorted order
 * using O(log n) comparator stages instead of the O(log² n) of sort_warp.
 */
template <unsigned REGX, unsigned REGY>
template <bool asc, unsigned N>
__device__ void RegBitonicTopK<REGX, REGY>::merge_warp(
    float (&d)[N], uint32_t (&i)[N])
{
    const unsigned tid = threadIdx.x & 31;
    constexpr unsigned TOTAL = 32 * N;
    // Single pass: len = TOTAL. Since gp < TOTAL, direction is uniform.
    for (unsigned stride = TOTAL >> 1; stride >= 1; stride >>= 1) {
        if (stride >= N) {
            unsigned thread_stride = stride / N;
            unsigned partner = tid ^ thread_stride;
            #pragma unroll
            for (unsigned j = 0; j < N; j++) {
                float    pd = __shfl_xor_sync(0xffffffff, d[j], thread_stride);
                uint32_t pi = __shfl_xor_sync(0xffffffff, i[j], thread_stride);
                bool is_lower = (tid < partner);
                bool want_smaller = (asc == is_lower);
                if (want_smaller ? (d[j] > pd) : (d[j] < pd)) {
                    d[j] = pd;
                    i[j] = pi;
                }
            }
        } else {
            #pragma unroll
            for (unsigned k = 0; k < N; k++) {
                unsigned j = k ^ stride;
                if (j > k) {
                    if ((d[k] > d[j]) == asc) {
                        float    td = d[k]; d[k] = d[j]; d[j] = td;
                        uint32_t ti = i[k]; i[k] = i[j]; i[j] = ti;
                    }
                }
            }
        }
    }
}

/**
 * @brief Full bitonic sort of 32*N key-value pairs from arbitrary order.
 *        Always produces ascending output.
 */
template <unsigned REGX, unsigned REGY>
template <unsigned N>
__device__ void RegBitonicTopK<REGX, REGY>::sort_warp_full(
    float (&d)[N], uint32_t (&i)[N])
{
    const unsigned tid = threadIdx.x & 31;
    constexpr unsigned TOTAL = 32 * N;

    for (unsigned len = 2; len <= TOTAL; len <<= 1) {
        for (unsigned stride = len >> 1; stride >= 1; stride >>= 1) {
            if (stride >= N) {
                unsigned thread_stride = stride / N;
                unsigned partner = tid ^ thread_stride;
                #pragma unroll
                for (unsigned j = 0; j < N; j++) {
                    float    pd = __shfl_xor_sync(0xffffffff, d[j], thread_stride);
                    uint32_t pi = __shfl_xor_sync(0xffffffff, i[j], thread_stride);
                    unsigned gp = tid * N + j;
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
                for (unsigned k = 0; k < N; k++) {
                    unsigned j = k ^ stride;
                    if (j > k) {
                        unsigned gp = tid * N + k;
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

/** @brief Pairwise min/max split on N register pairs. */
template <unsigned REGX, unsigned REGY>
template <unsigned N>
__device__ void RegBitonicTopK<REGX, REGY>::butterfly(
    float (&a_d)[N], uint32_t (&a_i)[N],
    float (&b_d)[N], uint32_t (&b_i)[N])
{
    #pragma unroll
    for (unsigned j = 0; j < N; j++) {
        if (a_d[j] > b_d[j]) {
            float    td = a_d[j]; a_d[j] = b_d[j]; b_d[j] = td;
            uint32_t ti = a_i[j]; a_i[j] = b_i[j]; b_i[j] = ti;
        }
    }
}
