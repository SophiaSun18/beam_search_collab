#pragma once

// =============================================================================
// qleaf_profile.cuh — Profile counter helpers.
//
// Wraps the repeated `atomicAdd(d_routea_profile_counters + idx, value)`
// pattern. Compiles out to a no-op when ROUTEA_PROFILE_COUNTERS is 0 so
// release builds carry no extra ALU.
//
// Counter index symbols are defined in qleaf_internal_defaults.cuh as
// kRouteAProfile* enum values.
// =============================================================================

#include "qleaf_config.cuh"

__device__ __forceinline__ void routea_profile_add(
    gpu_long_t* counters,
    int idx,
    gpu_long_t value) {
#if ROUTEA_PROFILE_COUNTERS
    if (counters != nullptr && value != (gpu_long_t)0) {
        atomicAdd(&counters[idx], value);
    }
#else
    (void)counters;
    (void)idx;
    (void)value;
#endif
}

__device__ __forceinline__ void routea_profile_max(
    gpu_long_t* counters,
    int idx,
    gpu_long_t value) {
#if ROUTEA_PROFILE_COUNTERS
    if (counters != nullptr) {
        atomicMax(&counters[idx], value);
    }
#else
    (void)counters;
    (void)idx;
    (void)value;
#endif
}
