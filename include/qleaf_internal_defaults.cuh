#pragma once

#include <cstdint>

// Stable mainline config surface.
// This file now owns both:
// 1) externally overridable build-shape inputs
// 2) stable fixed Route-A / reflow constants

#ifndef BEAM_SIZE
#define BEAM_SIZE       128
#endif
#ifndef SEARCH_WIDTH
#define SEARCH_WIDTH    8
#endif
#ifndef BITLEN
#define BITLEN          11
#endif

// Stable fixed compile-time constants for the current mainline.
#ifndef ROUTEA_HASH_MAX_PROBES
#define ROUTEA_HASH_MAX_PROBES 24
#endif
#define HASH_INSERT_MAX_PROBES ROUTEA_HASH_MAX_PROBES
#define HASH_INSERT_LIMIT_START_FILL 160
#ifndef RABITQ_EARLY_STOP_STALL_ITERS
#define RABITQ_EARLY_STOP_STALL_ITERS 64
#endif
#ifndef RABITQ_EARLY_STOP_WARMUP_ITERS
#define RABITQ_EARLY_STOP_WARMUP_ITERS 8
#endif
#define RABITQ_MAX_ITER_CAP 256
#define RABITQ_STALL_IMPROVE_EPS_ABS 1e-4f
#define RABITQ_STALL_IMPROVE_EPS_REL 1e-4f
#ifndef REFLOW_TOPN
#define REFLOW_TOPN 8
#endif
#ifndef REFLOW_POP_BATCH
#define REFLOW_POP_BATCH 8
#endif
#ifndef ROUTEA_CANDIDATE_BUFFER_SIZE
#define ROUTEA_CANDIDATE_BUFFER_SIZE 128
#endif
#ifndef ROUTEA_GATE_MODE
// 0: no gate (used by exact baselines and NAV=8 edge-key).
// 1: RaBitQ local top-k / adaptive keep gate.
#define ROUTEA_GATE_MODE 1
#endif
#define ROUTEA_GATE_EXACT 0
#define ROUTEA_GATE_LOCAL_TOPK 1
#ifndef NAV_MODE
#define NAV_MODE 0
#endif
#define ROUTEA_NAV_EXACT 0
// Sophia-style edge-residual RaBitQ-as-key with final exact rerank.
// child_nav = exact_L2(parent) + edge_residual_rabitq(parent->child)
// No local pruning; all valid neighbors admitted; final exact rerank.
#define ROUTEA_NAV_EDGE_RABITQ_KEY_FINAL_RERANK 8
#ifndef ROUTEA_GPU_QUERY_PREP
#define ROUTEA_GPU_QUERY_PREP 0
#endif
#ifndef ROUTEA_PROFILE_COUNTERS
#define ROUTEA_PROFILE_COUNTERS 0
#endif
#ifndef ROUTEA_AUDIT_COUNTERS
#define ROUTEA_AUDIT_COUNTERS 0
#endif
#ifndef ROUTEA_ADAPTIVE_GATE
#define ROUTEA_ADAPTIVE_GATE 0
#endif
#ifndef ROUTEA_ADAPTIVE_POLICY
#define ROUTEA_ADAPTIVE_POLICY 0
#endif
#ifndef ROUTEA_ADAPTIVE_MIN_KEEP
#define ROUTEA_ADAPTIVE_MIN_KEEP 4
#endif
#ifndef ROUTEA_ADAPTIVE_BASE_KEEP
#define ROUTEA_ADAPTIVE_BASE_KEEP 8
#endif
#ifndef ROUTEA_ADAPTIVE_MAX_KEEP
#define ROUTEA_ADAPTIVE_MAX_KEEP 32
#endif
#ifndef ROUTEA_ADAPTIVE_WIDTH_MIN
#define ROUTEA_ADAPTIVE_WIDTH_MIN 4
#endif
#ifndef ROUTEA_ADAPTIVE_WIDTH_BASE
#define ROUTEA_ADAPTIVE_WIDTH_BASE 8
#endif
#ifndef ROUTEA_ADAPTIVE_WIDTH_MAX
#define ROUTEA_ADAPTIVE_WIDTH_MAX SEARCH_WIDTH
#endif
#ifndef ROUTEA_ADAPTIVE_TARGET_CANDIDATES
#define ROUTEA_ADAPTIVE_TARGET_CANDIDATES 96
#endif
#ifndef ROUTEA_ADAPTIVE_MARGIN_REL_X1000
#define ROUTEA_ADAPTIVE_MARGIN_REL_X1000 25
#endif
#ifndef ROUTEA_ADAPTIVE_STALL_WIDEN_1
#define ROUTEA_ADAPTIVE_STALL_WIDEN_1 4
#endif
#ifndef ROUTEA_ADAPTIVE_STALL_WIDEN_2
#define ROUTEA_ADAPTIVE_STALL_WIDEN_2 8
#endif

enum RouteAProfileCounterIndex : int {
    kRouteAProfileParentsExpanded = 0,
    kRouteAProfileValidNeighborsScored = 1,
    kRouteAProfileGateKeptPreHash = 2,
    kRouteAProfileHashAccepted = 3,
    kRouteAProfileHashDuplicate = 4,
    kRouteAProfileHashProbeExhausted = 5,
    kRouteAProfileCandidateBufferOverflow = 6,
    kRouteAProfileInitialExactL2 = 7,
    kRouteAProfileCandidateExactL2 = 8,
    kRouteAProfileEarlyStopTriggers = 9,
    kRouteAProfileNoProgressBreaks = 10,
    kRouteAProfileGateExactTopN = 11,
    kRouteAProfileGateHitExactTopN = 12,
    kRouteAProfileGateFalseNegativeTopN = 13,
    kRouteAProfileGateFalseNegativeCutoff = 14,
    kRouteAProfileGateFalseNegativeCutoffMarginX1000 = 15,
    kRouteAProfileParentPickCount = 16,
    kRouteAProfileParentExactRankSum = 17,
    kRouteAProfileParentExactRankMax = 18,
    kRouteAProfileParentMatchesExactBest = 19,
    kRouteAProfileParentDelaysExactBest = 20,
    kRouteAProfileAdaptiveParents = 21,
    kRouteAProfileAdaptiveKeepSum = 22,
    kRouteAProfileAdaptiveKeepLe4 = 23,
    kRouteAProfileAdaptiveKeepLe8 = 24,
    kRouteAProfileAdaptiveKeepLe16 = 25,
    kRouteAProfileAdaptiveKeepGt16 = 26,
    kRouteAProfileAdaptiveWidthRounds = 27,
    kRouteAProfileAdaptiveWidthSum = 28,
    kRouteAProfileAdaptiveThresholdUnder = 29,
    kRouteAProfileAdaptiveExactUseful = 30,
    kRouteAProfileAdaptiveFeedbackNarrow = 31,
    kRouteAProfileAdaptiveFeedbackNormal = 32,
    kRouteAProfileAdaptiveFeedbackBroad = 33,
    kRouteAProfileCounterCount = 34,
};

// Stable compile-time constants used by the fixed mainline.
inline constexpr unsigned kHashSize = (1u << BITLEN);
inline constexpr int kCandidateBufferSize = ROUTEA_CANDIDATE_BUFFER_SIZE;
inline constexpr int kResultBufferSize = BEAM_SIZE + kCandidateBufferSize;
inline constexpr int kNeighborBufferSize = kCandidateBufferSize;
inline constexpr int kNoProgressBreakIters = 8;
inline constexpr int kRouteAMaxNeighborFanout = WARP_SIZE;
inline constexpr int kRouteAStaticKeepBound =
    (REFLOW_TOPN < kRouteAMaxNeighborFanout) ? REFLOW_TOPN : kRouteAMaxNeighborFanout;
inline constexpr int kRouteAAdaptiveKeepBound =
    (ROUTEA_ADAPTIVE_MAX_KEEP > kRouteAStaticKeepBound)
        ? ROUTEA_ADAPTIVE_MAX_KEEP
        : kRouteAStaticKeepBound;
inline constexpr int kRouteAKeepUpperBound =
    ((ROUTEA_GATE_MODE == 0) ||
     (REFLOW_TOPN >= kRouteAMaxNeighborFanout))
        ? kRouteAMaxNeighborFanout
        : (ROUTEA_ADAPTIVE_GATE
               ? ((kRouteAAdaptiveKeepBound < kRouteAMaxNeighborFanout)
                      ? kRouteAAdaptiveKeepBound
                      : kRouteAMaxNeighborFanout)
               : kRouteAStaticKeepBound);
inline constexpr int kRouteAMaxPreHashCandidates = SEARCH_WIDTH * kRouteAKeepUpperBound;

// ── Inlined former routea_compile_config.cuh ─────────────────────────────────
// Semantic compile-time predicates derived from the raw macros above, plus
// the invariants every formal-only Q-LEAF kernel relies on. The QLEAF_*
// guardrails (BITLEN, REFLOW_TOPN, SEARCH_WIDTH, candidate-buffer size)
// live in qleaf_config.cuh and run after this file completes.

inline constexpr bool kRouteAGateModeSupported =
    (ROUTEA_GATE_MODE == ROUTEA_GATE_EXACT) ||
    (ROUTEA_GATE_MODE == ROUTEA_GATE_LOCAL_TOPK);
inline constexpr bool kRouteANavModeSupported =
    (NAV_MODE == ROUTEA_NAV_EXACT) ||
    (NAV_MODE == ROUTEA_NAV_EDGE_RABITQ_KEY_FINAL_RERANK);
inline constexpr bool kRouteAAdaptivePolicySupported =
    (ROUTEA_ADAPTIVE_POLICY == 0);
static_assert(kRouteAGateModeSupported,
              "Unsupported ROUTEA_GATE_MODE. Use ROUTEA_GATE_EXACT (0) or "
              "ROUTEA_GATE_LOCAL_TOPK (1).");
static_assert(kRouteANavModeSupported,
              "Unsupported NAV_MODE. Use ROUTEA_NAV_EXACT (0) or "
              "ROUTEA_NAV_EDGE_RABITQ_KEY_FINAL_RERANK (8).");
static_assert((ROUTEA_ADAPTIVE_GATE == 0) || kRouteAAdaptivePolicySupported,
              "ROUTEA_ADAPTIVE_GATE=1 requires a supported ROUTEA_ADAPTIVE_POLICY.");
static_assert(ROUTEA_ADAPTIVE_POLICY == 0,
              "Q-LEAF supports only the formal POLICY=0 controller. "
              "Use ROUTEA_ADAPTIVE_POLICY=0.");
static_assert(kCandidateBufferSize >= kRouteAMaxPreHashCandidates,
              "ROUTEA_CANDIDATE_BUFFER_SIZE is too small for this configuration. "
              "Increase the candidate buffer.");

inline constexpr bool kRouteAUseNavArray =
    (NAV_MODE == ROUTEA_NAV_EDGE_RABITQ_KEY_FINAL_RERANK);
inline constexpr bool kRouteASortByNav =
    (NAV_MODE == ROUTEA_NAV_EDGE_RABITQ_KEY_FINAL_RERANK);
inline constexpr bool kRouteASkipSurvivorExact =
    (NAV_MODE == ROUTEA_NAV_EDGE_RABITQ_KEY_FINAL_RERANK);
inline constexpr bool kRouteAUseEdgeRabitqNav =
    (NAV_MODE == ROUTEA_NAV_EDGE_RABITQ_KEY_FINAL_RERANK);

inline constexpr bool kRouteANeedsEdgeSidecar =
    (ROUTEA_GATE_MODE == ROUTEA_GATE_LOCAL_TOPK) ||
    kRouteAUseEdgeRabitqNav;
inline constexpr bool kRouteANeedsQueryPrep = kRouteANeedsEdgeSidecar;

static_assert(!kRouteAUseEdgeRabitqNav ||
                  (ROUTEA_GATE_MODE == ROUTEA_GATE_EXACT),
              "NAV_MODE=8 (edge_rabitq) requires ROUTEA_GATE_MODE=0 "
              "(no local pruning).");
static_assert(!kRouteAUseEdgeRabitqNav ||
                  (ROUTEA_ADAPTIVE_GATE == 0),
              "NAV_MODE=8 (edge_rabitq) requires ROUTEA_ADAPTIVE_GATE=0 "
              "(adaptive keep is incompatible with NAV=8 first-layer ablation).");
static_assert(!kRouteAUseEdgeRabitqNav || (REFLOW_TOPN >= 32),
              "NAV_MODE=8 (edge_rabitq) requires REFLOW_TOPN>=32 to admit all "
              "valid neighbours of a degree-32 parent (no local pruning).");
static_assert(!kRouteAUseEdgeRabitqNav || kRouteANeedsEdgeSidecar,
              "NAV_MODE=8 (edge_rabitq) requires the edge sidecar; "
              "kRouteANeedsEdgeSidecar must be true when nav=8.");
// ── End inlined former routea_compile_config.cuh ─────────────────────────────
#ifndef ROUTEA_SCORE_WORDS
#define ROUTEA_SCORE_WORDS 4
#endif
inline constexpr int kStableRabitqScoreWords = ROUTEA_SCORE_WORDS;
inline constexpr int kStableRabitqMaxScoreDims = kStableRabitqScoreWords << 5;

// When kResultBufferSize fits in one block (≤ BLOCK_SIZE=256), use fast
// in-place bitonic sort.  Otherwise fall back to binary merge + scratch.
inline constexpr bool kUseBitonicMerge = (kResultBufferSize <= BLOCK_SIZE);

// ── Shared-memory budget & launch_bounds auto-calculation ──────────────
// Estimate shared memory per block (bytes) to derive optimal occupancy.
//   s_all_{idx,dist}   : (kUseBitonicMerge ? kResultBufferSize : BEAM_SIZE) * 8
//   s_cand_{idx,dist}  : (kUseBitonicMerge ? 0 : kCandidateBufferSize * 8)
//   s_hash             : kHashSize * 4
//   Fixed overhead     : s_query_cache + s_query_bp + s_expand_pos(28)
//                        + scalar shared vars(~60) + s_pick_counts(32) ≈ 728
inline constexpr int kSmemFixedOverhead =
    (kStableRabitqMaxScoreDims * (int)sizeof(float)) +
    (6 * kStableRabitqScoreWords * (int)sizeof(uint32_t)) +
    28 + 60 + 32;
inline constexpr int kSmemPerBlock =
    (kUseBitonicMerge ? kResultBufferSize : BEAM_SIZE) * 8 +
    (kUseBitonicMerge ? 0 : kCandidateBufferSize * 8) +
    (kRouteAUseNavArray ? ((kUseBitonicMerge ? kResultBufferSize : BEAM_SIZE) * 4) : 0) +
    ((kRouteAUseNavArray && !kUseBitonicMerge) ? (kCandidateBufferSize * 4) : 0) +
    (int)kHashSize * 4 +
    kSmemFixedOverhead;

// Derive blocks/SM from shared memory budget.
// RTX 4000 Ada (AD104): 100 KB programmable shared memory per SM.
// Adjust this value if targeting a different GPU architecture.
inline constexpr int kSmemPerSM = 102400;
inline constexpr int kMaxBlocksBySmem = kSmemPerSM / kSmemPerBlock;
// Register pressure caps practical occupancy at 6 blocks/SM
// (65536 regs / 256 threads / 6 blocks ≈ 42 regs/thread).
// For the bitonic path, 6 is empirically proven.
// For the merge path, cap at 6 as well — the extra local_dist[4]/local_idx[4]
// adds ~8 registers but fits within the 42-register budget.
inline constexpr int kMinBlocksPerSM = kUseBitonicMerge ? 6 :
    (kMaxBlocksBySmem > 6 ? 6 :
     kMaxBlocksBySmem < 1 ? 1 : kMaxBlocksBySmem);
