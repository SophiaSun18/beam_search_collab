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
public:
    /**
     * @brief Construct a hash filter backed by shared memory.
     *
     * @param table   Shared memory array, size >= (1 << bitlen) elements.
     * @param bitlen  log2 of table size.
     */
    __device__ HashFilter(uint32_t* table, unsigned bitlen)
        : table_(table), bitlen_(bitlen),
          size_(1u << bitlen), mask_((1u << bitlen) - 1) {}

    /**
     * @brief Shared memory footprint in bytes for a given bitlen.
     */
    __host__ __device__ static uint32_t shared_mem_bytes(unsigned bitlen) {
        return (1u << bitlen) * sizeof(uint32_t);
    }

    /**
     * @brief Fill all slots with 0xFFFFFFFF. All threads participate.
     *
     * @param first_tid  Threads with threadIdx.x < first_tid are skipped.
     */
    __device__ void init(unsigned first_tid = 0);

    /**
     * @brief Insert key via atomicCAS with linear probing.
     *
     * @return 1 if key was newly inserted, 0 if already present.
     */
    __device__ uint32_t insert(uint32_t key);

    /**
     * @brief Check membership without modifying the table.
     *
     * @return 1 if key is present, 0 otherwise.
     */
    __device__ uint32_t contains(uint32_t key);
};

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
