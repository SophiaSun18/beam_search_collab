#pragma once
#include <concepts>
#include <cstdint>
#include <cstddef>

/*
 * TopKBuffer — self-contained warp-collaborative sorted priority queue.
 * Owns its sorted (distance, index) storage internally.
 * All methods called in warp lockstep.
 *
 * Constructible from (char*& smem, unsigned topk):
 *   SharedBitonicTopK carves storage from smem.
 *   RegBitonicTopK validates topk and leaves smem untouched.
 */
template <typename B, unsigned REG>
concept TopKBuffer = requires(
    B               buf,
    char*           smem_ptr,
    float           (&dist)[REG],
    uint32_t        (&idx)[REG],
    unsigned        n
) {
    // Construct from shared memory pointer and topk size.
    { B(smem_ptr, n) };

    // Shared memory bytes needed per instance.
    { B::shared_mem_bytes_per_instance(n) } -> std::convertible_to<size_t>;

    // Merge per-thread REG candidates (unsorted) into internal storage.
    { buf.merge(dist, idx) } -> std::same_as<void>;

    // Find up to `n` unexpanded nodes (MSB==0), mark them expanded,
    // write node IDs to out[0..found-1]. Returns count found.
    // Template on output array size: pick<N>(out, n).
    { buf.template pick<REG>(idx, n) } -> std::convertible_to<unsigned>;

    // Also requires: template <typename F> __device__ void rerank(F&& f)
    // Recomputes distances via warp-collaborative functor f(uint32_t node_id) -> float,
    // then re-sorts. All lanes must call f in lockstep.

    // Write the n smallest elements to dist/idx arrays (e.g. global memory).
    { buf.flush((float*)nullptr, (uint32_t*)nullptr, n) } -> std::same_as<void>;

    // Number of slots in the buffer.
    { buf.capacity() } -> std::convertible_to<unsigned>;

    // Node ID at position pos (MSB cleared). 0xFFFFFFFF for empty slots.
    { buf.id(n) } -> std::convertible_to<uint32_t>;
};

#include "topk_shared.cuh"
#include "topk_reg.cuh"
