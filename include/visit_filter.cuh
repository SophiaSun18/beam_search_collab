#pragma once
#include <concepts>
#include <cstdint>

/*
 * VisitFilter — shared-memory approximate or exact set for visited-node tracking.
 * All methods called in block-wide lockstep (init) or per-thread (insert/contains).
 */
template <typename F>
concept VisitFilter = requires(
    F          filter,
    uint32_t   key,
    unsigned   first_tid
) {
    // Clear all entries. All threads in the block participate.
    // Threads with threadIdx.x < first_tid are skipped.
    { filter.init(first_tid) } -> std::same_as<void>;

    // Insert key. Returns 1 if newly inserted (not previously present),
    // 0 if already present. For approximate filters (bloom), the "already
    // present" check may have false positives.
    { filter.insert(key) } -> std::convertible_to<uint32_t>;

    // Check membership without modifying state.
    // Returns 1 if present, 0 otherwise.
    { filter.contains(key) } -> std::convertible_to<uint32_t>;

    // Shared memory bytes for ny instances at the given bitlen.
    { F::shared_mem_bytes(0u, 1u) } -> std::convertible_to<uint32_t>;
};

#include "hash_filter.cuh"
#include "bloom_filter.cuh"
