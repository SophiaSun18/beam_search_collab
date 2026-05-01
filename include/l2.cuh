#pragma once
#include <cuda_runtime.h>

/**
 * @file l2.cuh
 * @brief L2 (squared Euclidean) distance computation primitives for CUDA.
 *
 * All functions are prefixed with l2_. Currently provides warp-collaborative
 * distance. Additional shapes (thread, block) can be added here.
 */

#define L2_FULL_MASK 0xffffffff

/**
 * @brief Warp-collaborative squared L2 distance.
 *
 * All 32 lanes of the warp must call this. Each lane processes a strided
 * subset of dimensions, then a warp-wide reduction produces the result
 * in all lanes.
 *
 * @param dim  Number of dimensions.
 * @param a    First vector, dim elements in device memory.
 * @param b    Second vector, dim elements in device memory.
 * @return     Squared L2 distance (same value in all lanes).
 */
template <typename T = float>
__device__ __forceinline__ T l2_warp(int dim, const T* a, const T* b) {
    int lane = threadIdx.x & 31;
    T val = T(0);
    for (int i = lane; i < dim; i += 32)
        val += (a[i] - b[i]) * (a[i] - b[i]);
    T sum = val;
    sum += __shfl_down_sync(L2_FULL_MASK, sum, 16);
    sum += __shfl_down_sync(L2_FULL_MASK, sum, 8);
    sum += __shfl_down_sync(L2_FULL_MASK, sum, 4);
    sum += __shfl_down_sync(L2_FULL_MASK, sum, 2);
    sum += __shfl_down_sync(L2_FULL_MASK, sum, 1);
    sum  = __shfl_sync(L2_FULL_MASK, sum, 0);
    return sum;
}

/**
 * @brief Per-lane squared L2 distance (column-parallel pattern).
 *
 * Each lane independently computes the full squared L2 distance between
 * a shared query and its own vector. All lanes read the same query
 * (shared memory broadcast) but different vectors from global memory.
 *
 * @param dim  Number of dimensions.
 * @param a    Query vector (typically in shared memory), dim elements.
 * @param b    Per-lane vector (typically in global memory), dim elements.
 * @return     Squared L2 distance (independent per lane).
 */
template <typename T = float>
__device__ __forceinline__ T l2_column(int dim, const T* a, const T* b) {
    T sum = T(0);
    for (int i = 0; i < dim; i++) {
        T diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}
