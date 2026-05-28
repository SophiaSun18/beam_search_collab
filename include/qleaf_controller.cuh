#pragma once

// =============================================================================
// qleaf_controller.cuh — Formal Q-LEAF adaptive controller.
//
// The formal Q-LEAF "Adaptive Fast" controller is variant
// `qleaf_adaptive_keep_w16` (placement = adaptive_fast,
// ROUTEA_ADAPTIVE_GATE=1, ROUTEA_ADAPTIVE_POLICY=0).
//
// Q-LEAF supports only POLICY=0. The controller has TWO degrees of
// freedom inside a fixed compile-time maximum SEARCH_WIDTH:
//
//   1. active_width per round — chosen by routea_adaptive_width_policy0_decide
//      based on whether the frontier is full and how long the per-query
//      best-distance has been stalled.
//   2. keep_k per parent — chosen by routea_adaptive_keep_policy0_decide
//      from a cutoff-density-driven `desired_keep`, clamped against
//      adaptive min/max and a per-round candidate-budget keep.
//
// Other adaptive policy IDs (2/3/4/5/6/8/10..21/23..28), the hidden
// cutoff-density disable, the early-broad override, and the small-beam
// guard were all removed: they were exploration artifacts, not formal
// ablation variants.
// =============================================================================

#include "qleaf_config.cuh"

// Q-LEAF supports only POLICY=0. The compile-time check in
// routea_compile_config.cuh is the primary fence; this header reasserts
// it locally so any helper added later cannot silently drift.
static_assert(ROUTEA_ADAPTIVE_POLICY == 0,
              "Q-LEAF supports only ROUTEA_ADAPTIVE_POLICY=0.");

// =============================================================================
// Active-width policy=0 schedule.
//
// Inside the fixed compile-time maximum SEARCH_WIDTH, the controller
// chooses an active expansion width per round:
//
//   - frontier not yet full                 → width_max (fill the beam)
//   - stall_iters >= stall_widen_2          → width_max (escape stall)
//   - stall_iters >= stall_widen_1          → midpoint(base, max)
//   - default                               → width_base
//
// The result is then clamped to [width_min, width_max].
// =============================================================================

struct RouteAAdaptiveWidthInput {
    int width_min;
    int width_base;
    int width_max;
    bool frontier_full;
    int stall_iters;
    int stall_widen_1;
    int stall_widen_2;
};

struct RouteAAdaptiveWidthDecision {
    int active_width;
};

__device__ __forceinline__ RouteAAdaptiveWidthDecision
routea_adaptive_width_policy0_decide(const RouteAAdaptiveWidthInput& in) {
    int active_width = in.width_base;
    if (!in.frontier_full) {
        active_width = in.width_max;
    } else if (in.stall_iters >= in.stall_widen_2) {
        active_width = in.width_max;
    } else if (in.stall_iters >= in.stall_widen_1) {
        active_width = max(in.width_base, (in.width_base + in.width_max + 1) >> 1);
    } else {
        active_width = in.width_base;
    }
    active_width = max(in.width_min, min(active_width, in.width_max));
    RouteAAdaptiveWidthDecision out;
    out.active_width = active_width;
    return out;
}

// =============================================================================
// Adaptive Fast policy=0 keep decision.
//
// This helper is the ONLY place where the policy-0 keep budget clamp is
// computed. The kernel still computes per-parent `approx_rank` and the
// cutoff-density `desired_keep`; this helper applies the adaptive_min/max
// clamp, the candidate-budget clamp, and the max_keep clamp so that no
// future kernel edit can drift the policy-0 semantics independently.
//
// Equivalent inline pseudocode (must stay in sync with the body below):
//   keep_k = clamp(desired_keep, [adaptive_min_keep, adaptive_max_keep]);
//   keep_k = min(keep_k, budget_keep);
//   keep_k = clamp(keep_k, [adaptive_min_keep, max_keep]);
// =============================================================================

struct RouteAAdaptiveKeepInput {
    int desired_keep;        // caller-supplied (cutoff-density-derived)
    int adaptive_min_keep;
    int adaptive_max_keep;
    int budget_keep;
    int max_keep;
};

struct RouteAAdaptiveKeepDecision {
    int keep_k;
};

__device__ __forceinline__ RouteAAdaptiveKeepDecision
routea_adaptive_keep_policy0_decide(const RouteAAdaptiveKeepInput& in) {
    RouteAAdaptiveKeepDecision out;
    int keep_k = max(in.adaptive_min_keep,
                     min(in.desired_keep, in.adaptive_max_keep));
    keep_k = min(keep_k, in.budget_keep);
    keep_k = max(in.adaptive_min_keep, min(keep_k, in.max_keep));
    out.keep_k = keep_k;
    return out;
}
