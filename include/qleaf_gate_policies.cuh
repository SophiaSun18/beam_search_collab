#pragma once

// =============================================================================
// qleaf_gate_policies.cuh — Local-gate keep-decision helpers.
//
// Header-only `__device__ __forceinline__` helpers for the formal local
// gate decisions:
//
//   GATE_MODE=0 (ROUTEA_GATE_EXACT):     no local pruning.
//   GATE_MODE=1 (ROUTEA_GATE_LOCAL_TOPK): bitplane edge score top-k /
//                                         policy-0 adaptive keep.
//
// The non-formal gate modes (lane order, deterministic random, score-all dead-drop) were removed.
//
// The bitplane edge score itself lives in routea_scorers.cuh; this file
// consumes the score for GATE_MODE 1 only.
//
// Caller contract: helpers may be called from divergent lane masks. Any
// helper that uses `__shfl_sync(FULL_MASK, ...)` must still be reached by
// all lanes in the warp through the existing kernel call structure.
// =============================================================================

#include "qleaf_device_primitives.cuh"

// Common gate input/output shape consumed by the kernel candidate-admission
// block. The kernel call sites assemble a `RouteALocalGateInput` once per
// (warp, neighbor) and read back a `RouteALocalGateDecision` from the
// helper that matches the active GATE_MODE / adaptive policy.
struct RouteALocalGateInput {
    bool valid;        // true when the lane has a real, in-range neighbour.
    float approx;      // bitplane approximate score.
    int deg;           // active neighbour degree for this parent (<= warp size).
    int lane_id;       // 0..WARP_SIZE-1.
    int approx_rank;   // 0..WARP_SIZE; rank of approx within the warp; 32 = unused.
    int keep_k;        // caller-provided keep budget.
};

struct RouteALocalGateDecision {
    bool keep;
    int keep_k;
    int approx_rank;
};

// Backward-compat alias used by the policy-0 adaptive kernel block.
using RouteAGateDecision = RouteALocalGateDecision;

// Non-adaptive scored gate decision for GATE_MODE 1 (local top-k). The
// helper sets keep_k = min(REFLOW_TOPN, deg) and runs the warp-cooperative
// top-k smallest selector to pick the kept lanes.
//
// The policy-0 adaptive gate does NOT use this helper. It has its own
// fill/stall active-width schedule and cutoff-density keep decision in the
// kernel, and only reuses `RouteALocalGateDecision` as the final result shape.
__device__ __forceinline__ RouteALocalGateDecision
routea_scored_nonadaptive_gate_decide(const RouteALocalGateInput& in) {
    RouteALocalGateDecision out;
    out.keep_k = min((int)REFLOW_TOPN, in.deg);
    out.approx_rank = in.approx_rank > 0 ? in.approx_rank : 32;
    out.keep = rabitq_lane_keep_topk_smallest_f32(in.approx, in.valid, out.keep_k);
    return out;
}

// Backward-compat thin wrapper. Existing callers that still take the
// `(valid, approx, deg)` triple build a minimal `RouteALocalGateInput`
// and forward to the unified helper.
__device__ __forceinline__ RouteAGateDecision routea_scored_fixed_gate_decide(
    bool valid,
    float approx,
    int deg) {
    RouteALocalGateInput in;
    in.valid = valid;
    in.approx = approx;
    in.deg = deg;
    in.lane_id = (threadIdx.x & (WARP_SIZE - 1));
    in.approx_rank = 32;
    in.keep_k = min((int)REFLOW_TOPN, deg);
    return routea_scored_nonadaptive_gate_decide(in);
}

// GATE_MODE 0 (no local pruning). Trivially `keep = valid`.
__device__ __forceinline__ bool routea_gate_keep_all(bool valid) {
    return valid;
}
