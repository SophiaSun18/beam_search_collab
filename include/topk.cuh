#pragma once
#include <concepts>
#include <cstdint>

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

    // Number of slots in the buffer.
    { buf.capacity() } -> std::convertible_to<unsigned>;

    // Node ID at position pos (MSB cleared). 0xFFFFFFFF for empty slots.
    { buf.id(n) } -> std::convertible_to<uint32_t>;
};

#include "topk_shared.cuh"
#include "topk_reg.cuh"
