#pragma once
#include <cstdint>
#include <cuda_runtime.h>

/**
 * @class BloomFilter — shared-memory bloom filter for visited-node tracking.
 *
 * Uses k=3 independent hash functions over a bit array.
 * Stored as uint32_t words. False positives possible; no false negatives.
 *
 * Same interface as HashFilter for drop-in replacement.
 */
class BloomFilter {
    uint32_t* table_;   // bit array stored as uint32_t words
    unsigned  bitlen_;  // log2 of number of BITS (not words)
    uint32_t  mask_;    // (1 << bitlen) - 1

    __host__ __device__ static unsigned log2_ceil(unsigned n) {
        if (n <= 1) return 0;
        unsigned p = 1, b = 0;
        while (p < n) { p <<= 1; b++; }
        return b;
    }

    __device__ static void hashes(uint32_t key, unsigned bitlen, uint32_t mask,
                                   uint32_t& h0, uint32_t& h1, uint32_t& h2) {
        uint32_t a = key;
        a ^= a >> 16; a *= 0x45d9f3b; a ^= a >> 16;
        h0 = a & mask;

        uint32_t b = key * 0x9e3779b9u;
        b ^= b >> 16; b *= 0x45d9f3b; b ^= b >> 16;
        h1 = b & mask;

        uint32_t c = key * 0x517cc1b7u;
        c ^= c >> 16; c *= 0x45d9f3b; c ^= c >> 16;
        h2 = c & mask;
    }

public:
    /**
     * @brief Construct a bloom filter backed by shared memory.
     *
     * @param table     Shared memory array.
     * @param capacity  Minimum number of bits (rounded up to power of 2, min 32).
     */
    __device__ BloomFilter(uint32_t* table, unsigned capacity);

    /**
     * @brief Self-allocating constructor. Carves table from smem.
     */
    __device__ BloomFilter(char*& smem, unsigned capacity);

    /**
     * @brief Shared memory bytes for ny instances.
     *
     * @param capacity  Minimum number of bits per instance (rounded up to power of 2, min 32).
     * @param ny        Number of instances (blockDim.y). Default 1.
     */
    __host__ __device__ static uint32_t shared_mem_bytes(unsigned capacity, unsigned ny = 1) {
        unsigned bitlen = log2_ceil(capacity);
        if (bitlen < 5) bitlen = 5;
        uint32_t words = 1u << (bitlen - 5);
        return (alignof(uint32_t) - 1) + ny * words * sizeof(uint32_t);
    }

    __device__ void init(unsigned first_tid = 0);
    __device__ uint32_t insert(uint32_t key);
    __device__ uint32_t contains(uint32_t key);
};

__device__ inline BloomFilter::BloomFilter(uint32_t* table, unsigned capacity)
    : table_(table) {
    unsigned bl = log2_ceil(capacity);
    if (bl < 5) bl = 5;
    bitlen_ = bl;
    mask_ = (1u << bl) - 1;
}

__device__ inline BloomFilter::BloomFilter(char*& smem, unsigned capacity) {
    unsigned bl = log2_ceil(capacity);
    if (bl < 5) bl = 5;
    bitlen_ = bl;
    mask_ = (1u << bl) - 1;

    smem = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(smem) + alignof(uint32_t) - 1)
            & ~(static_cast<uintptr_t>(alignof(uint32_t)) - 1));

    uint32_t words = 1u << (bitlen_ - 5);
    size_t per_inst = words * sizeof(uint32_t);
    table_ = reinterpret_cast<uint32_t*>(smem + threadIdx.y * per_inst);
    smem += blockDim.y * per_inst;
}

__device__ inline void BloomFilter::init(unsigned first_tid) {
    if (threadIdx.x < first_tid) return;
    uint32_t words = 1u << (bitlen_ - 5);
    for (uint32_t i = threadIdx.x - first_tid; i < words; i += blockDim.x - first_tid) {
        table_[i] = 0;
    }
}

__device__ inline uint32_t BloomFilter::insert(uint32_t key) {
    uint32_t h0, h1, h2;
    hashes(key, bitlen_, mask_, h0, h1, h2);

    uint32_t w0 = table_[h0 >> 5];
    uint32_t w1 = table_[h1 >> 5];
    uint32_t w2 = table_[h2 >> 5];
    bool present = (w0 & (1u << (h0 & 31))) &&
                   (w1 & (1u << (h1 & 31))) &&
                   (w2 & (1u << (h2 & 31)));

    atomicOr(&table_[h0 >> 5], 1u << (h0 & 31));
    atomicOr(&table_[h1 >> 5], 1u << (h1 & 31));
    atomicOr(&table_[h2 >> 5], 1u << (h2 & 31));

    return present ? 0 : 1;
}

__device__ inline uint32_t BloomFilter::contains(uint32_t key) {
    uint32_t h0, h1, h2;
    hashes(key, bitlen_, mask_, h0, h1, h2);
    uint32_t w0 = table_[h0 >> 5];
    uint32_t w1 = table_[h1 >> 5];
    uint32_t w2 = table_[h2 >> 5];
    return ((w0 & (1u << (h0 & 31))) &&
            (w1 & (1u << (h1 & 31))) &&
            (w2 & (1u << (h2 & 31)))) ? 1 : 0;
}
