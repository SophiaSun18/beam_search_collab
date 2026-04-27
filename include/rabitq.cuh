#pragma once

#include "gpu_graph.cuh"
#include "rabitq.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

#define RABITQ_CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(err));                                  \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

// ── Non-owning view of device-resident RAbitQ data ───────────────────────

/**
 *  @class GPURabitqView
 *  @brief Non-owning, immutable, trivially-copyable view of device-resident
 *         RAbitQ auxiliary data.  Safe to pass by value to kernels.
 */
class GPURabitqView {
    const uint64_t* d_codes_;
    const float*    d_factors_;
    const float*    d_signs_;
    int             padded_dim_;
    int             degree_;
    int             bitcode_words_;
public:
    __device__ __host__ GPURabitqView(const uint64_t* codes, const float* factors,
                                      const float* signs, int padded_dim, int degree);
    __device__ __host__ const uint64_t* codes(vid_t node)   const;
    __device__ __host__ const float*    factors(vid_t node)  const;
    __device__ __host__ const float*    signs()              const;
    __device__ __host__ int             padded_dim()         const;
};

__device__ __host__ inline GPURabitqView::GPURabitqView(
    const uint64_t* codes, const float* factors, const float* signs,
    int padded_dim, int degree)
    : d_codes_(codes), d_factors_(factors), d_signs_(signs),
      padded_dim_(padded_dim), degree_(degree),
      bitcode_words_(padded_dim / 64) {}

__device__ __host__ inline const uint64_t* GPURabitqView::codes(vid_t node) const {
    return d_codes_ + static_cast<size_t>(node) * degree_ * bitcode_words_;
}
__device__ __host__ inline const float* GPURabitqView::factors(vid_t node) const {
    return d_factors_ + static_cast<size_t>(node) * 3 * degree_;
}
__device__ __host__ inline const float* GPURabitqView::signs()      const { return d_signs_; }
__device__ __host__ inline int          GPURabitqView::padded_dim() const { return padded_dim_; }

// ── Owning GPU allocations ───────────────────────────────────────────────

/**
 *  @class GPURabitq
 *  @brief Device-resident RAbitQ data.  Owns CUDA allocations for codes,
 *         factors, and sign table.  Non-copyable; freed on destruction.
 */
class GPURabitq {
public:
    explicit GPURabitq(const CPURabitqData& cpu);
    ~GPURabitq();
    GPURabitqView view() const;

    GPURabitq(const GPURabitq&)            = delete;
    GPURabitq& operator=(const GPURabitq&) = delete;

private:
    uint64_t* d_codes_   = nullptr;
    float*    d_factors_ = nullptr;
    float*    d_signs_   = nullptr;
    int       padded_dim_ = 0;
    int       degree_     = 0;
};

inline GPURabitq::GPURabitq(const CPURabitqData& cpu)
    : padded_dim_(cpu.padded_dim), degree_(cpu.degree)
{
    size_t code_bytes   = cpu.codes.size()   * sizeof(uint64_t);
    size_t factor_bytes = cpu.factors.size()  * sizeof(float);
    size_t sign_bytes   = cpu.signs.size()    * sizeof(float);

    RABITQ_CUDA_CHECK(cudaMalloc(&d_codes_,   code_bytes));
    RABITQ_CUDA_CHECK(cudaMemcpy(d_codes_, cpu.codes.data(),
                                 code_bytes, cudaMemcpyHostToDevice));

    RABITQ_CUDA_CHECK(cudaMalloc(&d_factors_, factor_bytes));
    RABITQ_CUDA_CHECK(cudaMemcpy(d_factors_, cpu.factors.data(),
                                 factor_bytes, cudaMemcpyHostToDevice));

    RABITQ_CUDA_CHECK(cudaMalloc(&d_signs_,   sign_bytes));
    RABITQ_CUDA_CHECK(cudaMemcpy(d_signs_, cpu.signs.data(),
                                 sign_bytes, cudaMemcpyHostToDevice));

    RABITQ_CUDA_CHECK(cudaDeviceSynchronize());
}

inline GPURabitq::~GPURabitq() {
    cudaFree(d_codes_);
    cudaFree(d_factors_);
    cudaFree(d_signs_);
}

inline GPURabitqView GPURabitq::view() const {
    return {d_codes_, d_factors_, d_signs_, padded_dim_, degree_};
}

// ── helpers ───────────────────────────────────────────────────────────────

__host__ __device__ inline int rabitq_padded_dim(int dim) {
    int p = 1;
    while (p < dim) p <<= 1;
    return p;
}

// ── RabitqQuery ──────────────────────────────────────────────────────────

/**
 *  @class RabitqQuery
 *  @brief One fully-prepared RAbitQ query, backed by shared memory.
 *
 *  The constructor carves shared memory, then rotates, quantizes, and
 *  builds the fastscan LUT for the given query vector.  After
 *  construction the object is immediately usable — scan_neighbors_32()
 *  may be called in any order, any number of times.
 *
 *  @tparam BX  Block size (number of threads) for block-wide reductions.
 */
template <unsigned BX>
class RabitqQuery {
    float*       rotated_query_;      // [padded_dim] floats, shared mem
    uint8_t*     quantized_query_;    // [padded_dim] bytes, shared mem
    uint8_t*     lut_;                // [padded_dim * 4] bytes, shared mem
    float*       low_val_;            // 1 float, shared mem
    float*       width_;              // 1 float, shared mem
    int32_t*     sum_q_;              // 1 int32_t, shared mem
    int          padded_dim_;

    __device__ void rotate_and_transform(const float* src, const float* signs, int dim);
    __device__ void find_range_and_quantize();
    __device__ void build_lut();
    __device__ static uint8_t fastscan_decode(
        const uint8_t* packed, int lane_id, int codebook_idx);

public:
    /**
     * @brief Carve shared memory and fully prepare the query.
     *
     * Rotates via Walsh-Hadamard, quantizes to RABITQ_BQUERY bits,
     * and builds the fastscan LUT.  All BX threads must participate.
     * padded_dim is computed internally as next power of 2 >= dim.
     *
     * @param smem        Shared memory pointer, advanced past all allocations.
     * @param query_smem  Query vector already in shared memory (dim floats).
     * @param signs       Sign table in device global memory (padded_dim floats).
     * @param dim         Original vector dimension.
     */
    __device__ RabitqQuery(char*& smem, const float* query_smem,
                           const float* signs, int dim);

    /**
     * @brief Estimate distances for up to 32 neighbors via fastscan.
     *
     * One warp lane per neighbor.  Reads fastscan-packed codes, accumulates
     * through the LUT, and writes the estimated distance.  Invalid factors
     * (triple_x == FLT_MAX) or non-finite results are clamped to FLT_MAX.
     *
     * @param packed_codes_block  Fastscan-packed codes for this batch (shared mem).
     * @param staged_factors      Full factor array for the node: [triple_x | factor_dq | factor_vq],
     *                            each segment has `degree` floats, in shared memory.
     * @param degree              Graph degree (stride between factor segments).
     * @param batch_begin         Index of the first neighbor in this batch.
     * @param exact_dist          L2 distance from query to the expanding parent node.
     * @param appro_dist          Output: estimated distance per active lane.
     * @param active_neighbors    Number of valid neighbors in this batch (<= 32).
     */
    __device__ void scan_neighbors_32(
        const uint8_t* packed_codes_block,
        const float* staged_factors, int degree, int batch_begin,
        float exact_dist, float* appro_dist, int active_neighbors);

    /** @brief Shared memory bytes needed for construction (dim → padded internally). */
    __host__ __device__ static size_t shared_mem_bytes(int dim);
};

// ── RabitqQuery implementation ───────────────────────────────────────────

template <unsigned BX>
__device__ void RabitqQuery<BX>::rotate_and_transform(
    const float* src, const float* signs, int dim)
{
    int tid  = threadIdx.x;
    int pdim = padded_dim_;

    for (int i = tid; i < dim; i += static_cast<int>(BX))
        rotated_query_[i] = src[i] * signs[i];
    for (int i = dim + tid; i < pdim; i += static_cast<int>(BX))
        rotated_query_[i] = 0.0f;
    __syncthreads();

    // Fast Walsh-Hadamard Transform (in-place, power-of-2)
    for (int len = 1; len < pdim; len <<= 1) {
        int pairs = pdim >> 1;
        for (int pi = tid; pi < pairs; pi += static_cast<int>(BX)) {
            int blk  = pi / len;
            int j    = pi % len;
            int base = blk * (len << 1);
            int i1   = base + j;
            int i2   = i1 + len;
            float u  = rotated_query_[i1];
            float v  = rotated_query_[i2];
            rotated_query_[i1] = u + v;
            rotated_query_[i2] = u - v;
        }
        __syncthreads();
    }
}

template <unsigned BX>
__device__ void RabitqQuery<BX>::find_range_and_quantize()
{
    constexpr unsigned WARPS = BX / 32;
    int tid     = threadIdx.x;
    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    int pdim    = padded_dim_;

    // ── block-wide min/max ──────────────────────────────────────────
    __shared__ float s_warp_min[WARPS];
    __shared__ float s_warp_max[WARPS];
    float local_min =  FLT_MAX;
    float local_max = -FLT_MAX;

    for (int i = tid; i < pdim; i += static_cast<int>(BX)) {
        float v = rotated_query_[i];
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_min = fminf(local_min, __shfl_down_sync(0xffffffff, local_min, off));
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));
    }
    if (lane_id == 0) {
        s_warp_min[warp_id] = local_min;
        s_warp_max[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_min = (lane_id < static_cast<int>(WARPS)) ? s_warp_min[lane_id] :  FLT_MAX;
        local_max = (lane_id < static_cast<int>(WARPS)) ? s_warp_max[lane_id] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1) {
            local_min = fminf(local_min, __shfl_down_sync(0xffffffff, local_min, off));
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));
        }
        if (lane_id == 0) {
            *low_val_ = local_min;
            *width_   = (local_max - local_min) / ((1 << RABITQ_BQUERY) - 1);
        }
    }
    __syncthreads();

    // ── quantize ────────────────────────────────────────────────────
    float low = *low_val_;
    float inv_width = 1.0f / (*width_);

    __shared__ int32_t s_warp_sum[WARPS];
    int32_t local_sum = 0;
    for (int i = tid; i < pdim; i += static_cast<int>(BX)) {
        uint8_t q = static_cast<uint8_t>(
            lroundf((rotated_query_[i] - low) * inv_width + 0.5f));
        quantized_query_[i] = q;
        local_sum += q;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
    if (lane_id == 0)
        s_warp_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane_id < static_cast<int>(WARPS)) ? s_warp_sum[lane_id] : 0;
        for (int off = 16; off > 0; off >>= 1)
            local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
        if (lane_id == 0)
            *sum_q_ = local_sum;
    }
    __syncthreads();
}

template <unsigned BX>
__device__ void RabitqQuery<BX>::build_lut()
{
    int tid = threadIdx.x;
    int num_codebook = padded_dim_ >> 2;

    for (int cb = tid; cb < num_codebook; cb += static_cast<int>(BX)) {
        const uint8_t* qc = quantized_query_ + (cb << 2);
        uint8_t* lc        = lut_ + (cb << 4);
        lc[0] = 0;
        for (int mask = 1; mask < 16; ++mask) {
            int lowbit = mask & -mask;
            int pos = (lowbit == 8) ? 0
                    : (lowbit == 4) ? 1
                    : (lowbit == 2) ? 2
                    :                 3;
            lc[mask] = static_cast<uint8_t>(lc[mask - lowbit] + qc[pos]);
        }
    }
}

template <unsigned BX>
__device__ __forceinline__ uint8_t RabitqQuery<BX>::fastscan_decode(
    const uint8_t* packed, int lane_id, int codebook_idx)
{
    int pair_idx     = codebook_idx >> 1;
    int nibble_group = codebook_idx & 1;
    int lane16       = lane_id & 15;
    int byte_idx     = ((lane16 & 7) << 1) | (lane16 >> 3);
    uint8_t val      = packed[pair_idx * 32 + nibble_group * 16 + byte_idx];
    return (lane_id < 16) ? static_cast<uint8_t>(val & 0x0f)
                          : static_cast<uint8_t>(val >> 4);
}

// ── constructor ─────────────────────────────────────────────────────────

template <unsigned BX>
__device__ RabitqQuery<BX>::RabitqQuery(
    char*& smem, const float* query_smem,
    const float* signs, int dim)
    : padded_dim_(rabitq_padded_dim(dim))
{
    int padded_dim = padded_dim_;

    // carve shared memory — align before each typed allocation
    smem = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(smem) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));
    rotated_query_   = reinterpret_cast<float*>(smem);
    smem += padded_dim * sizeof(float);

    quantized_query_ = reinterpret_cast<uint8_t*>(smem);
    smem += padded_dim;

    lut_             = reinterpret_cast<uint8_t*>(smem);
    smem += padded_dim * 4;

    // realign after uint8_t allocations
    smem = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(smem) + alignof(float) - 1)
            & ~(static_cast<uintptr_t>(alignof(float)) - 1));
    low_val_         = reinterpret_cast<float*>(smem);
    smem += sizeof(float);

    width_           = reinterpret_cast<float*>(smem);
    smem += sizeof(float);

    sum_q_           = reinterpret_cast<int32_t*>(smem);
    smem += sizeof(int32_t);

    // fully prepare: rotate, quantize, build LUT
    rotate_and_transform(query_smem, signs, dim);
    find_range_and_quantize();
    build_lut();
    __syncthreads();
}

// ── shared_mem_bytes ────────────────────────────────────────────────────

template <unsigned BX>
__host__ __device__ size_t RabitqQuery<BX>::shared_mem_bytes(int dim)
{
    int padded_dim = rabitq_padded_dim(dim);
    size_t size = alignof(float) - 1;      // worst-case initial alignment
    size += padded_dim * sizeof(float);    // rotated_query
    size += padded_dim;                    // quantized_query
    size += padded_dim * 4;                // lut
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);  // realign
    size += sizeof(float);                 // low_val
    size += sizeof(float);                 // width
    size += sizeof(int32_t);               // sum_q
    return size;
}

// ── scan_neighbors_32 ───────────────────────────────────────────────────

template <unsigned BX>
__device__ void RabitqQuery<BX>::scan_neighbors_32(
    const uint8_t* packed_codes_block,
    const float* staged_factors, int degree, int batch_begin,
    float exact_dist, float* appro_dist, int active_neighbors)
{
    int lane = threadIdx.x & 31;
    if (lane >= active_neighbors) return;

    float tx  = staged_factors[batch_begin + lane];
    float fdq = staged_factors[degree + batch_begin + lane];
    float fvq = staged_factors[2 * degree + batch_begin + lane];

    int num_codebook = padded_dim_ >> 2;
    uint32_t raw_sum = 0;
    for (int cb = 0; cb < num_codebook; ++cb) {
        uint8_t code = fastscan_decode(packed_codes_block, lane, cb);
        raw_sum += static_cast<uint32_t>(lut_[(cb << 4) + code]);
    }

    float w  = *width_;
    float lv = *low_val_;
    int32_t sq = *sum_q_;

    float result_float = static_cast<float>(
        (static_cast<int32_t>(raw_sum) << 1) - sq);
    float est = tx + exact_dist;
    est += fdq * w * result_float;
    est += fvq * lv;

    if (tx == FLT_MAX || !isfinite(est))
        est = FLT_MAX;

    appro_dist[lane] = est;
}
