#pragma once
#include <cstdint>
#include <cuda_runtime.h>

/**
 * @class HashFilter — shared-memory linear-probing hash set for visited-node tracking.
 *
 * Backed by a caller-provided shared memory buffer.  All device methods are
 * called in block-wide lockstep (init) or per-thread (insert / contains).
 */
class HashFilter {
    uint32_t* table_;
    unsigned  bitlen_;
    uint32_t  size_;
    uint32_t  mask_;

    __host__ __device__ static unsigned log2_ceil(unsigned n) {
        if (n <= 1) return 0;
        unsigned p = 1, b = 0;
        while (p < n) { p <<= 1; b++; }
        return b;
    }

public:
    /**
     * @brief Construct a hash filter backed by shared memory.
     *
     * @param table     Shared memory array, size >= capacity elements.
     * @param capacity  Minimum number of slots (rounded up to power of 2).
     */
    __device__ HashFilter(uint32_t* table, unsigned capacity);

    /**
     * @brief Self-allocating constructor. Carves table from smem.
     *
     * When blockDim.y > 1, each instance (threadIdx.y) gets its own table.
     * Advances smem past all blockDim.y instances.
     */
    __device__ HashFilter(char*& smem, unsigned capacity);

    /**
     * @brief Shared memory bytes for ny instances.
     *
     * @param capacity  Minimum number of slots per instance (rounded up to power of 2).
     * @param ny        Number of instances (blockDim.y). Default 1.
     */
    __host__ __device__ static uint32_t shared_mem_bytes(unsigned capacity, unsigned ny = 1) {
        unsigned bitlen = log2_ceil(capacity);
        return (alignof(uint32_t) - 1) + ny * (1u << bitlen) * sizeof(uint32_t);
    }

    __device__ void init(unsigned first_tid = 0);
    __device__ uint32_t insert(uint32_t key);
    __device__ uint32_t contains(uint32_t key);
};

__device__ inline HashFilter::HashFilter(uint32_t* table, unsigned capacity)
    : table_(table), bitlen_(log2_ceil(capacity)),
      size_(1u << bitlen_), mask_(size_ - 1) {}

__device__ inline HashFilter::HashFilter(char*& smem, unsigned capacity)
    : bitlen_(log2_ceil(capacity)), size_(1u << bitlen_), mask_(size_ - 1)
{
    smem = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(smem) + alignof(uint32_t) - 1)
            & ~(static_cast<uintptr_t>(alignof(uint32_t)) - 1));
    size_t per_inst = size_ * sizeof(uint32_t);
    table_ = reinterpret_cast<uint32_t*>(smem + threadIdx.y * per_inst);
    smem += blockDim.y * per_inst;
}

__device__ inline void HashFilter::init(unsigned first_tid) {
    if (threadIdx.x < first_tid) return;
    for (uint32_t i = threadIdx.x - first_tid; i < size_; i += blockDim.x - first_tid) {
        table_[i] = ~uint32_t(0);
    }
}

__device__ inline uint32_t HashFilter::insert(uint32_t key) {
    uint32_t idx = (key ^ (key >> bitlen_)) & mask_;
    for (unsigned i = 0; i < size_; i++) {
        uint32_t old = atomicCAS(&table_[idx], ~uint32_t(0), key);
        if (old == ~uint32_t(0)) return 1;
        if (old == key)          return 0;
        idx = (idx + 1) & mask_;
    }
    return 0;
}

__device__ inline uint32_t HashFilter::contains(uint32_t key) {
    uint32_t idx = (key ^ (key >> bitlen_)) & mask_;
    for (unsigned i = 0; i < size_; i++) {
        uint32_t slot = table_[idx];
        if (slot == key)          return 1;
        if (slot == ~uint32_t(0)) return 0;
        idx = (idx + 1) & mask_;
    }
    return 0;
}
