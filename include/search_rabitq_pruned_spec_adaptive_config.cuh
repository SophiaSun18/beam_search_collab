#pragma once

// Adaptive pruning (AP) configuration map.
//
// Keep AP policy in this file and leave the kernel body focused on execution.
// The knobs below are grouped by the part of the controller they affect:
//   1. Kernel shape: compile-time storage and LUT scan layout.
//   2. Local gate: whether each parent tile is pruned by warp-local ranking.
//   3. Width policy: how many beam parents AP expands in fill, cruise, and escape.
//   4. Candidate budget: the shared candidate-buffer capacity cap.
//   5. Prune policy: degree-aware base p plus fill/escape offsets.
//   6. Progress signals: kth-cutoff margin and tail-stall tolerance.
//

// Build-time kernel shape. AP compiles with SEARCH_WIDTH=16 so the runtime
// active_width can move between fill/cruise/escape without reallocating shared memory.
#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK 128
#endif

#ifndef SEARCH_WIDTH
#define SEARCH_WIDTH 16
#endif

#define MAX_ITERATIONS (1 << 15)

// RaBitQ LUT scan layout. AP uses the sequential-LUT artifact; this controls how
// many lanes cooperate on one neighbor estimate.
#ifndef GPU_RABITQ_FASTSCAN_SUBWARP_LANES
#define GPU_RABITQ_FASTSCAN_SUBWARP_LANES 32
#endif

#if GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 2 && GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 4 && \
    GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 8 && GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 16 && \
    GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 32
#error "GPU_RABITQ_FASTSCAN_SUBWARP_LANES must be one of 2, 4, 8, 16, or 32"
#endif

// Local gate enablement. The gate ranks each parent tile by estimated distance
// before admitting candidates into the compact shared candidate buffer.
#ifndef GPU_RABITQ_PRUNED_LOCAL_GATE_MIN_WIDTH
#define GPU_RABITQ_PRUNED_LOCAL_GATE_MIN_WIDTH 2
#endif

#ifndef GPU_RABITQ_PRUNED_USE_LOCAL_GATE
#define GPU_RABITQ_PRUNED_USE_LOCAL_GATE (SEARCH_WIDTH >= GPU_RABITQ_PRUNED_LOCAL_GATE_MIN_WIDTH)
#endif

// Runtime width policy. Fill starts narrow, cruise uses the normal AP width, and
// escape widens once the full-beam tail has stalled for STALL_WIDEN_1 iterations.
#ifndef RABITQ_ADAPTIVE_WIDTH_MIN
#define RABITQ_ADAPTIVE_WIDTH_MIN 1
#endif

#ifndef RABITQ_ADAPTIVE_WIDTH_MAX
#define RABITQ_ADAPTIVE_WIDTH_MAX SEARCH_WIDTH
#endif

#ifndef RABITQ_ADAPTIVE_WIDTH_START
#define RABITQ_ADAPTIVE_WIDTH_START RABITQ_ADAPTIVE_FILL_WIDTH
#endif

#ifndef RABITQ_ADAPTIVE_WIDTH_STEP
#define RABITQ_ADAPTIVE_WIDTH_STEP 2
#endif

#ifndef RABITQ_ADAPTIVE_FILL_WIDTH
#define RABITQ_ADAPTIVE_FILL_WIDTH 2
#endif

#ifndef RABITQ_ADAPTIVE_CRUISE_WIDTH
#define RABITQ_ADAPTIVE_CRUISE_WIDTH 8
#endif

#ifndef RABITQ_ADAPTIVE_ESCAPE_WIDTH
#define RABITQ_ADAPTIVE_ESCAPE_WIDTH 16
#endif

#ifndef RABITQ_ADAPTIVE_STALL_WIDEN_1
#define RABITQ_ADAPTIVE_STALL_WIDEN_1 4
#endif

// Fixed candidate-buffer budget per iteration. active_width divides this target
// into a per-parent capacity cap. The final keep count is also bounded by the
// current p ratio, so this budget cannot override the paper prune-ratio limit.
#ifndef RABITQ_ADAPTIVE_TARGET_CANDIDATES
#define RABITQ_ADAPTIVE_TARGET_CANDIDATES 32
#endif

// Per-parent keep clamps used after target_candidates / active_width.
#ifndef RABITQ_ADAPTIVE_KEEP_MIN
#define RABITQ_ADAPTIVE_KEEP_MIN 2
#endif

#ifndef RABITQ_ADAPTIVE_KEEP_MAX
#define RABITQ_ADAPTIVE_KEEP_MAX WARP_SIZE
#endif

// Prune-ratio policy. Fill uses base_p - fill_relax; escape uses base_p + escape_tighten.
#ifndef RABITQ_ADAPTIVE_PRUNE_MIN
#define RABITQ_ADAPTIVE_PRUNE_MIN 0.0f
#endif

#ifndef RABITQ_ADAPTIVE_PRUNE_MAX
#define RABITQ_ADAPTIVE_PRUNE_MAX 0.9375f
#endif

#ifndef RABITQ_ADAPTIVE_FILL_PRUNE_RELAX_STEP
#define RABITQ_ADAPTIVE_FILL_PRUNE_RELAX_STEP 0.375f
#endif

#ifndef RABITQ_ADAPTIVE_BASE_PRUNE_RATIO_DEG16
#define RABITQ_ADAPTIVE_BASE_PRUNE_RATIO_DEG16 0.75f
#endif

#ifndef RABITQ_ADAPTIVE_BASE_PRUNE_RATIO_DEG32
#define RABITQ_ADAPTIVE_BASE_PRUNE_RATIO_DEG32 0.875f
#endif

#ifndef RABITQ_ADAPTIVE_ESCAPE_PRUNE_TIGHTEN_STEP
#define RABITQ_ADAPTIVE_ESCAPE_PRUNE_TIGHTEN_STEP 0.0625f
#endif

// Near-cutoff margin for local-gate down-selection. This signal may reduce the
// keep count when few neighbors look competitive, but the p-ratio bound remains the upper limit.
#ifndef RABITQ_ADAPTIVE_MARGIN_ABS
#define RABITQ_ADAPTIVE_MARGIN_ABS 0.0f
#endif

#ifndef RABITQ_ADAPTIVE_MARGIN_REL
#define RABITQ_ADAPTIVE_MARGIN_REL 0.02f
#endif

// Tail-stall detector tolerance. The helper uses max(abs_eps,
// rel_eps * abs(best_tail)): the absolute floor avoids noise resets for small
// distances, while the relative bound scales with large distances.
#ifndef RABITQ_ADAPTIVE_STALL_IMPROVE_EPS_ABS
#define RABITQ_ADAPTIVE_STALL_IMPROVE_EPS_ABS 1e-4f
#endif

#ifndef RABITQ_ADAPTIVE_STALL_IMPROVE_EPS_REL
#define RABITQ_ADAPTIVE_STALL_IMPROVE_EPS_REL 1e-4f
#endif

namespace rabitq_pruned_adaptive_config {
inline constexpr int kInternalTopk = INTERNAL_TOPK;
inline constexpr int kSearchWidth = SEARCH_WIDTH;
inline constexpr int kLutScanLanes = GPU_RABITQ_FASTSCAN_SUBWARP_LANES;
inline constexpr int kLocalGateMinWidth = GPU_RABITQ_PRUNED_LOCAL_GATE_MIN_WIDTH;
inline constexpr int kUseLocalGate = GPU_RABITQ_PRUNED_USE_LOCAL_GATE;

inline constexpr int kWidthMin = RABITQ_ADAPTIVE_WIDTH_MIN;
inline constexpr int kWidthMax = RABITQ_ADAPTIVE_WIDTH_MAX;
inline constexpr int kWidthStart = RABITQ_ADAPTIVE_WIDTH_START;
inline constexpr int kWidthStep = RABITQ_ADAPTIVE_WIDTH_STEP;
inline constexpr int kFillWidth = RABITQ_ADAPTIVE_FILL_WIDTH;
inline constexpr int kCruiseWidth = RABITQ_ADAPTIVE_CRUISE_WIDTH;
inline constexpr int kEscapeWidth = RABITQ_ADAPTIVE_ESCAPE_WIDTH;
inline constexpr int kStallWiden1 = RABITQ_ADAPTIVE_STALL_WIDEN_1;

inline constexpr int kTargetCandidates = RABITQ_ADAPTIVE_TARGET_CANDIDATES;
inline constexpr int kKeepMin = RABITQ_ADAPTIVE_KEEP_MIN;
inline constexpr int kKeepMax = RABITQ_ADAPTIVE_KEEP_MAX;

inline constexpr float kPruneMin = RABITQ_ADAPTIVE_PRUNE_MIN;
inline constexpr float kPruneMax = RABITQ_ADAPTIVE_PRUNE_MAX;
inline constexpr float kFillPruneRelaxStep = RABITQ_ADAPTIVE_FILL_PRUNE_RELAX_STEP;
inline constexpr float kBasePruneRatioDeg16 = RABITQ_ADAPTIVE_BASE_PRUNE_RATIO_DEG16;
inline constexpr float kBasePruneRatioDeg32 = RABITQ_ADAPTIVE_BASE_PRUNE_RATIO_DEG32;
inline constexpr float kEscapePruneTightenStep = RABITQ_ADAPTIVE_ESCAPE_PRUNE_TIGHTEN_STEP;
inline constexpr float kMarginAbs = RABITQ_ADAPTIVE_MARGIN_ABS;
inline constexpr float kMarginRel = RABITQ_ADAPTIVE_MARGIN_REL;
inline constexpr float kStallImproveEpsAbs = RABITQ_ADAPTIVE_STALL_IMPROVE_EPS_ABS;
inline constexpr float kStallImproveEpsRel = RABITQ_ADAPTIVE_STALL_IMPROVE_EPS_REL;

static_assert(kSearchWidth == 16, "Adaptive AP requires SEARCH_WIDTH=16");
static_assert(kWidthMin >= 1, "RABITQ_ADAPTIVE_WIDTH_MIN must be >= 1");
static_assert(kWidthMax >= kWidthMin, "RABITQ_ADAPTIVE_WIDTH_MAX must be >= RABITQ_ADAPTIVE_WIDTH_MIN");
static_assert(kWidthMax <= kSearchWidth, "RABITQ_ADAPTIVE_WIDTH_MAX must be <= SEARCH_WIDTH");
static_assert(kWidthStart >= kWidthMin, "RABITQ_ADAPTIVE_WIDTH_START must be >= RABITQ_ADAPTIVE_WIDTH_MIN");
static_assert(kWidthStart <= kWidthMax, "RABITQ_ADAPTIVE_WIDTH_START must be <= RABITQ_ADAPTIVE_WIDTH_MAX");
static_assert(kFillWidth >= kWidthMin && kFillWidth <= kWidthMax, "RABITQ_ADAPTIVE_FILL_WIDTH must be within adaptive width bounds");
static_assert(kCruiseWidth >= kWidthMin && kCruiseWidth <= kWidthMax, "RABITQ_ADAPTIVE_CRUISE_WIDTH must be within adaptive width bounds");
static_assert(kEscapeWidth >= kWidthMin && kEscapeWidth <= kWidthMax, "RABITQ_ADAPTIVE_ESCAPE_WIDTH must be within adaptive width bounds");
static_assert(kWidthStep >= 1, "RABITQ_ADAPTIVE_WIDTH_STEP must be >= 1");
static_assert(kTargetCandidates > 0, "RABITQ_ADAPTIVE_TARGET_CANDIDATES must be > 0");
static_assert(kKeepMin >= 1, "RABITQ_ADAPTIVE_KEEP_MIN must be >= 1");
static_assert(kKeepMax >= kKeepMin, "RABITQ_ADAPTIVE_KEEP_MAX must be >= RABITQ_ADAPTIVE_KEEP_MIN");
static_assert(kKeepMax <= WARP_SIZE, "RABITQ_ADAPTIVE_KEEP_MAX must be <= WARP_SIZE");
static_assert(kFillPruneRelaxStep >= 0.0f, "RABITQ_ADAPTIVE_FILL_PRUNE_RELAX_STEP must be >= 0");
static_assert(kBasePruneRatioDeg16 >= kPruneMin && kBasePruneRatioDeg16 <= kPruneMax, "RABITQ_ADAPTIVE_BASE_PRUNE_RATIO_DEG16 must be within prune bounds");
static_assert(kBasePruneRatioDeg32 >= kPruneMin && kBasePruneRatioDeg32 <= kPruneMax, "RABITQ_ADAPTIVE_BASE_PRUNE_RATIO_DEG32 must be within prune bounds");
}  // namespace rabitq_pruned_adaptive_config
