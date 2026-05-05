#pragma once

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

// ── RabitqQuantizer ─────────────────────────────────────────────────────────

/**
 * RaBitQ rotation + scalar quantization.
 *
 * Holds the sign table for the randomized orthogonal rotation.
 * Quantizes a float vector into multi-bit uint8_t codes via
 * sign-flip, Walsh-Hadamard transform, and uniform scalar quantization.
 *
 * @tparam BX Thread count for cooperative execution.
 */
template <unsigned BX>
class RabitqQuantizer {
public:
    /**
     * @param signs Sign table in global memory, [padded_dim] floats.
     * @param seed  Seed for per-dimension uniform dithering.
     */
    __device__ RabitqQuantizer(const float* signs, uint32_t seed);

    /**
     * Quantize a float query into multi-bit codes.
     *
     * @param query        Query vector in shared memory, [dim] floats.
     * @param dim          Original dimension.
     * @param num_bits     Quantization bit-width (e.g. 6).
     * @param scratch      Scratch buffer in shared memory, [padded_dim] floats.
     * @param out          Output buffer in shared memory, [padded_dim] uint8_t.
     * @param low_val_out  Written with quantization floor value.
     * @param width_out    Written with quantization step width.
     *
     * Applies sign-flip + WHT into scratch, then uniform-quantizes into out
     * with per-dimension random dithering: floor(x_normalized + u), where
     * u ~ Uniform(0,1) is drawn from a hash of (seed, dimension index).
     * Output values are in [0, 2^num_bits - 1].
     * padded_dim is the smallest power of 2 >= dim.
     * All BX threads must participate.
     */
    __device__ void quantize(
        const float* query, int dim, int num_bits,
        float* scratch, uint8_t* out,
        float* low_val_out, float* width_out) const;

private:
    const float* signs_;
    uint32_t     seed_;
};

// ── RabitQFastScan ──────────────────────────────────────────────────────────

/**
 * Bit-plane decomposition of a multi-bit query for fast distance
 * estimation against one-bit neighbor vectors via AND + popcount.
 *
 * On construction, calls the quantizer to produce multi-bit codes, then
 * decomposes them into num_bits bit-planes in shared memory.  Stores
 * quantization scalars (low_val, width, sum_q) for the distance formula.
 *
 * @tparam BX       Thread count for cooperative execution.
 * @tparam NUM_BITS Quantization bit-width.
 */
template <unsigned BX, int NUM_BITS = 6>
class RabitQFastScan {
public:
    /**
     * Quantize query and build bit-planes.
     *
     * @param smem       Shared memory pointer, advanced past allocations.
     * @param query      Float query in shared memory, [dim] floats.
     * @param dim        Original dimension.
     * @param quantizer  Provides rotation + scalar quantization.
     *
     * All BX threads must participate.
     */
    __device__ RabitQFastScan(
        char*& smem, const float* query, int dim,
        const RabitqQuantizer<BX>& quantizer);

    /**
     * Estimate distances to neighbors.
     *
     * @param code_neighbors  Flat one-bit codes, [num_neighbors * words_per_plane]
     *                        uint32_t. Each neighbor is words_per_plane consecutive
     *                        words; bit j of word w is dimension w*32+j.
     * @param num_neighbors   Number of neighbors (<= 32, one per lane).
     * @param factors         Factor array: [triple_x(degree) | factor_dq(degree) |
     *                        factor_vq(degree)].
     * @param degree          Stride between factor segments.
     * @param exact_dist      L2 distance from query to expanding parent node.
     * @param output          Estimated distance per neighbor, [num_neighbors].
     *
     * Per lane: computes raw dot product via AND+popcount across bit-planes,
     * then applies: est = tx + exact_dist + fdq * width * (2*raw - sum_q) + fvq * low.
     * Invalid factors (tx == FLT_MAX) or non-finite results are clamped to FLT_MAX.
     */
    __device__ void dot(
        const uint32_t* code_neighbors, int num_neighbors,
        const float* factors, int degree,
        float exact_dist,
        float* output) const;

private:
    uint32_t* bitplanes_;
    float*    low_val_;
    float*    width_;
    int32_t*  sum_q_;
    int       words_per_plane_;
};

// ── padded dim helper (local to this module) ────────────────────────────────

namespace fastscan_detail {
__host__ __device__ inline int padded_dim(int dim)
{
    int p = 1;
    while (p < dim) p <<= 1;
    return p;
}
__host__ __device__ inline uint32_t hash(uint32_t a, uint32_t b)
{
    a ^= b;
    a ^= a >> 16;
    a *= 0x45d9f3b;
    a ^= a >> 16;
    a *= 0x45d9f3b;
    a ^= a >> 16;
    return a;
}

__host__ __device__ inline float uniform_01(uint32_t h)
{
    return static_cast<float>(h & 0xFFFFFF) / 16777216.0f;
}

} // namespace fastscan_detail

// ── RabitqQuantizer implementation ──────────────────────────────────────────

template <unsigned BX>
__device__ inline RabitqQuantizer<BX>::RabitqQuantizer(
    const float* signs, uint32_t seed)
    : signs_(signs), seed_(seed) {}

template <unsigned BX>
__device__ inline void RabitqQuantizer<BX>::quantize(
    const float* query, int dim, int num_bits,
    float* scratch, uint8_t* out,
    float* low_val_out, float* width_out) const
{
    int tid  = threadIdx.x;
    int pdim = fastscan_detail::padded_dim(dim);

    for (int i = tid; i < dim; i += static_cast<int>(BX))
        scratch[i] = query[i] * signs_[i];
    for (int i = dim + tid; i < pdim; i += static_cast<int>(BX))
        scratch[i] = 0.0f;
    if constexpr (BX <= 32) __syncwarp(); else __syncthreads();

    for (int len = 1; len < pdim; len <<= 1) {
        int pairs = pdim >> 1;
        for (int pi = tid; pi < pairs; pi += static_cast<int>(BX)) {
            int blk  = pi / len;
            int j    = pi % len;
            int base = blk * (len << 1);
            int i1   = base + j;
            int i2   = i1 + len;
            float u  = scratch[i1];
            float v  = scratch[i2];
            scratch[i1] = u + v;
            scratch[i2] = u - v;
        }
        if constexpr (BX <= 32) __syncwarp(); else __syncthreads();
    }

    int lane_id = tid & 31;
    float local_min =  FLT_MAX;
    float local_max = -FLT_MAX;
    for (int i = tid; i < pdim; i += static_cast<int>(BX)) {
        float v = scratch[i];
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_min = fminf(local_min,
            __shfl_down_sync(0xffffffff, local_min, off));
        local_max = fmaxf(local_max,
            __shfl_down_sync(0xffffffff, local_max, off));
    }

    if constexpr (BX > 32) {
        constexpr unsigned WARPS = BX / 32;
        int warp_id = tid >> 5;
        __shared__ float s_min[WARPS];
        __shared__ float s_max[WARPS];
        if (lane_id == 0) {
            s_min[warp_id] = local_min;
            s_max[warp_id] = local_max;
        }
        __syncthreads();
        if (warp_id == 0) {
            local_min = (lane_id < static_cast<int>(WARPS))
                ? s_min[lane_id] :  FLT_MAX;
            local_max = (lane_id < static_cast<int>(WARPS))
                ? s_max[lane_id] : -FLT_MAX;
            for (int off = 16; off > 0; off >>= 1) {
                local_min = fminf(local_min,
                    __shfl_down_sync(0xffffffff, local_min, off));
                local_max = fmaxf(local_max,
                    __shfl_down_sync(0xffffffff, local_max, off));
            }
        }
    }

    __shared__ float s_low, s_width;
    if (lane_id == 0 && (BX <= 32 || (tid >> 5) == 0)) {
        s_low   = local_min;
        s_width = (local_max - local_min) / ((1 << num_bits) - 1);
    }
    if constexpr (BX <= 32) __syncwarp(); else __syncthreads();

    float low       = s_low;
    float width     = s_width;
    float inv_width = 1.0f / width;
    int max_val     = (1 << num_bits) - 1;
    for (int i = tid; i < pdim; i += static_cast<int>(BX)) {
        float x = (scratch[i] - low) * inv_width;
        float u = fastscan_detail::uniform_01(
            fastscan_detail::hash(seed_, static_cast<uint32_t>(i)));
        int q = static_cast<int>(floorf(x + u));
        if (q < 0) q = 0;
        if (q > max_val) q = max_val;
        out[i] = static_cast<uint8_t>(q);
    }
    if constexpr (BX <= 32) __syncwarp(); else __syncthreads();

    if (tid == 0) {
        *low_val_out = low;
        *width_out   = width;
    }
    if constexpr (BX <= 32) __syncwarp(); else __syncthreads();
}

// ── RabitQFastScan implementation ───────────────────────────────────────────

template <unsigned BX, int NUM_BITS>
__device__ inline RabitQFastScan<BX, NUM_BITS>::RabitQFastScan(
    char*& smem, const float* query, int dim,
    const RabitqQuantizer<BX>& quantizer)
    : words_per_plane_(fastscan_detail::padded_dim(dim) / 32)
{
    int pdim = words_per_plane_ * 32;

    // carve scratch for quantizer (reused after construction)
    smem = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(smem) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));
    float* scratch = reinterpret_cast<float*>(smem);
    uint8_t* codes = reinterpret_cast<uint8_t*>(smem);

    // carve bit-planes (persistent)
    char* bp_start = smem + pdim * sizeof(float);
    bp_start = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(bp_start) + alignof(uint32_t) - 1)
            & ~(static_cast<uintptr_t>(alignof(uint32_t)) - 1));
    bitplanes_ = reinterpret_cast<uint32_t*>(bp_start);
    char* after_bp = bp_start
        + static_cast<size_t>(NUM_BITS) * words_per_plane_
          * sizeof(uint32_t);

    // carve scalars (persistent)
    after_bp = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(after_bp) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));
    low_val_ = reinterpret_cast<float*>(after_bp);
    after_bp += sizeof(float);
    width_ = reinterpret_cast<float*>(after_bp);
    after_bp += sizeof(float);
    sum_q_ = reinterpret_cast<int32_t*>(after_bp);
    after_bp += sizeof(int32_t);
    smem = after_bp;

    // quantize into scratch, receive scalars
    quantizer.quantize(query, dim, NUM_BITS, scratch, codes,
                       low_val_, width_);

    // build bit-planes and compute sum_q
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x / 32;
    int nwarps  = static_cast<int>(BX) / 32;

    int32_t local_sum = 0;
    for (int group = warp_id; group < words_per_plane_; group += nwarps) {
        uint8_t val = codes[group * 32 + lane];
        local_sum += val;
        for (int b = 0; b < NUM_BITS; ++b) {
            uint32_t word = __ballot_sync(
                0xffffffff, (val >> b) & 1);
            if (lane == 0)
                bitplanes_[b * words_per_plane_ + group] = word;
        }
    }

    // warp-reduce sum_q
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);

    if constexpr (BX > 32) {
        constexpr unsigned WARPS = BX / 32;
        __shared__ int32_t s_sum[WARPS];
        if (lane == 0)
            s_sum[warp_id] = local_sum;
        __syncthreads();
        if (warp_id == 0) {
            local_sum = (lane < static_cast<int>(WARPS))
                ? s_sum[lane] : 0;
            for (int off = 16; off > 0; off >>= 1)
                local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
        }
    }

    if (lane == 0 && (BX <= 32 || warp_id == 0))
        *sum_q_ = local_sum;
    if constexpr (BX <= 32) __syncwarp(); else __syncthreads();
}

template <unsigned BX, int NUM_BITS>
__device__ inline void RabitQFastScan<BX, NUM_BITS>::dot(
    const uint32_t* code_neighbors, int num_neighbors,
    const float* factors, int degree,
    float exact_dist,
    float* output) const
{
    int lane = threadIdx.x & 31;
    if (lane >= num_neighbors) return;

    float tx  = factors[lane];
    float fdq = factors[degree + lane];
    float fvq = factors[2 * degree + lane];

    int wpp = words_per_plane_;
    const uint32_t* nbr = code_neighbors + lane * wpp;
    uint32_t raw_sum = 0;
    for (int b = 0; b < NUM_BITS; ++b) {
        const uint32_t* bp = bitplanes_ + b * wpp;
        uint32_t partial = 0;
        for (int w = 0; w < wpp; ++w)
            partial += __popc(bp[w] & nbr[w]);
        raw_sum += partial << b;
    }

    float w   = *width_;
    float lv  = *low_val_;
    int32_t sq = *sum_q_;

    float ip  = static_cast<float>(
        (static_cast<int32_t>(raw_sum) << 1) - sq);
    float est = tx + exact_dist + fdq * w * ip + fvq * lv;

    if (tx == FLT_MAX || !isfinite(est))
        est = FLT_MAX;

    output[lane] = est;
}
