#pragma once

// =============================================================================
// qleaf_device_primitives.cuh
//
// Small warp/device helpers shared by kernel.cuh and the extracted RouteA
// policy headers (qleaf_gate_policies.cuh, qleaf_controller.cuh,
// qleaf_rabitq.cuh). Keep this header free of candidate-buffer writes,
// hash admission, sidecar loading, and host code.
//
// Caller contract: helpers that use `__shfl_sync(FULL_MASK, ...)` must be
// reached by ALL 32 lanes in the warp through the existing kernel call
// structure. Lanes with `valid=false` participate by carrying INT_MAX or
// FLT_MAX as a sentinel so the cooperative reduction stays correct.
// =============================================================================

#include "qleaf_config.cuh"
#include "utils.cuh"

#include <cfloat>
#include <climits>
#include <cstdint>

static __device__ __forceinline__ bool rabitq_lane_keep_topk_smallest_i32(
    int value,
    bool valid,
    int keep_k) {
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    // Invalid lanes carry INT_MAX — detected via value comparison,
    // eliminating the need for a separate valid-flag shuffle.
    const int my_value = valid ? value : INT_MAX;
    int better = 0;
    #pragma unroll
    for (int src = 0; src < WARP_SIZE; ++src) {
        const int other_value = __shfl_sync(FULL_MASK, my_value, src);
        if (other_value >= INT_MAX) {
            continue;
        }
        if ((other_value < my_value) || ((other_value == my_value) && (src < lane_id))) {
            ++better;
        }
    }
    if (!valid || keep_k <= 0) {
        return false;
    }
    return better < keep_k;
}

static __device__ __forceinline__ bool rabitq_lane_keep_topk_smallest_f32(
    float value,
    bool valid,
    int keep_k) {
    // NOTE: do NOT early-return here — all 32 lanes must reach the
    // __shfl_sync(FULL_MASK, ...) calls below.  Invalid lanes participate
    // with FLT_MAX so they never win the topk selection.
    // The valid flag is implicit: invalid lanes carry FLT_MAX, so
    // 'other_value >= FLT_MAX' reliably signals invalidity.
    // This halves the shuffle count (64 → 32 per call).
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const float my_value = valid ? value : FLT_MAX;
    int better = 0;
    #pragma unroll
    for (int src = 0; src < WARP_SIZE; ++src) {
        const float other_value = __shfl_sync(FULL_MASK, my_value, src);
        if (other_value >= FLT_MAX) {
            continue;
        }
        if ((other_value < my_value) || ((other_value == my_value) && (src < lane_id))) {
            ++better;
        }
    }
    if (!valid || keep_k <= 0) {
        return false;
    }
    return better < keep_k;
}

static __device__ __forceinline__ int rabitq_lane_rank_smallest_f32(
    float value,
    bool valid) {
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const float my_value = valid ? value : FLT_MAX;
    int better = 0;
    #pragma unroll
    for (int src = 0; src < WARP_SIZE; ++src) {
        const float other_value = __shfl_sync(FULL_MASK, my_value, src);
        if (other_value >= FLT_MAX) {
            continue;
        }
        if ((other_value < my_value) || ((other_value == my_value) && (src < lane_id))) {
            ++better;
        }
    }
    return valid ? better : INT_MAX;
}
