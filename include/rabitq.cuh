#pragma once

#include "rabitq.hpp"
#include <cuda_runtime.h>
#include <cstdint>

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
 *
 *  Codes are 1-bit per dimension: each neighbor has words_per_plane
 *  consecutive uint32_t words, where bit j of word w is dimension w*32+j.
 */
class GPURabitqView {
    const uint32_t* d_codes_;
    const float*    d_factors_;
    const float*    d_signs_;
    int             padded_dim_;
    int             degree_;
    int             words_per_plane_;
public:
    __device__ __host__ GPURabitqView(const uint32_t* codes, const float* factors,
                                      const float* signs, int padded_dim, int degree);
    __device__ __host__ const uint32_t* codes(vid_t node)   const;
    __device__ __host__ const float*    factors(vid_t node)  const;
    __device__ __host__ const float*    signs()              const;
    __device__ __host__ int             padded_dim()         const;
    __device__ __host__ int             words_per_plane()    const;
};

__device__ __host__ inline GPURabitqView::GPURabitqView(
    const uint32_t* codes, const float* factors, const float* signs,
    int padded_dim, int degree)
    : d_codes_(codes), d_factors_(factors), d_signs_(signs),
      padded_dim_(padded_dim), degree_(degree),
      words_per_plane_(padded_dim / 32) {}

__device__ __host__ inline const uint32_t* GPURabitqView::codes(vid_t node) const {
    return d_codes_ + static_cast<size_t>(node) * degree_ * words_per_plane_;
}
__device__ __host__ inline const float* GPURabitqView::factors(vid_t node) const {
    return d_factors_ + static_cast<size_t>(node) * 3 * degree_;
}
__device__ __host__ inline const float* GPURabitqView::signs()          const { return d_signs_; }
__device__ __host__ inline int          GPURabitqView::padded_dim()     const { return padded_dim_; }
__device__ __host__ inline int          GPURabitqView::words_per_plane() const { return words_per_plane_; }

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
    uint32_t* d_codes_   = nullptr;
    float*    d_factors_ = nullptr;
    float*    d_signs_   = nullptr;
    int       padded_dim_ = 0;
    int       degree_     = 0;
};

inline GPURabitq::GPURabitq(const CPURabitqData& cpu)
    : padded_dim_(cpu.padded_dim), degree_(cpu.degree)
{
    size_t code_bytes   = cpu.codes.size()   * sizeof(uint32_t);
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
