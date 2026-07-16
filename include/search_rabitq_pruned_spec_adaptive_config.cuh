#pragma once

// Adaptive pruning (AP) configuration map.
//
// Keep AP policy in this file and leave the kernel body focused on execution.
// The knobs below are grouped by the part of the controller they affect:
//   1. Kernel shape: compile-time storage and LUT scan layout.
//   2. Local gate: whether each parent tile is pruned by warp-local ranking.
//   3. Width policy: how many beam parents AP expands in Stage I and Stage II.
//   4. Candidate budget: the shared candidate-buffer capacity cap.
//   5. Prune policy: Stage I progress-rho and Stage II fixed rho.
//   6. Progress signals: head-stall transition and exact kth cutoff.
//

// Build-time kernel shape. AP compiles enough parent slots for the largest
// Stage II width we test.
#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK 128
#endif

#ifndef SEARCH_WIDTH
#define SEARCH_WIDTH (2 * WARPS_PER_BLOCK)
#endif

#define MAX_ITERATIONS (1 << 15)

// Recompute the AP policy every N iterations. Between checks, keep w/p fixed
// but refresh the kth cutoff for local gating.
#ifndef RABITQ_ADAPTIVE_CHECK_INTERVAL
#define RABITQ_ADAPTIVE_CHECK_INTERVAL 1
#endif

// Opt-in progress trace for AP policy debugging. This prints from the kernel,
// so keep query/iteration limits small.
#ifndef RABITQ_ADAPTIVE_TRACE
#define RABITQ_ADAPTIVE_TRACE 0
#endif

#ifndef RABITQ_ADAPTIVE_TRACE_QUERY_LIMIT
#define RABITQ_ADAPTIVE_TRACE_QUERY_LIMIT 2
#endif

#ifndef RABITQ_ADAPTIVE_TRACE_ITER_LIMIT
#define RABITQ_ADAPTIVE_TRACE_ITER_LIMIT 128
#endif


#ifndef RABITQ_ADAPTIVE_BEAM_TRACE
#define RABITQ_ADAPTIVE_BEAM_TRACE 0
#endif

#ifndef RABITQ_ADAPTIVE_BEAM_TRACE_LIMIT
#define RABITQ_ADAPTIVE_BEAM_TRACE_LIMIT 20
#endif

#ifndef RABITQ_ADAPTIVE_BEAM_TRACE_TOPN
#define RABITQ_ADAPTIVE_BEAM_TRACE_TOPN 32
#endif

#ifndef RABITQ_ADAPTIVE_TRACE_VALID_PER_EXPANDER
#define RABITQ_ADAPTIVE_TRACE_VALID_PER_EXPANDER 0
#endif

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

// Hard theta/rho bounds. Phase parameters below must stay within these ranges.
#ifndef RABITQ_ADAPTIVE_THETA_MIN
#define RABITQ_ADAPTIVE_THETA_MIN 1
#endif

#ifndef RABITQ_ADAPTIVE_THETA_MAX
#define RABITQ_ADAPTIVE_THETA_MAX WARPS_PER_BLOCK
#endif

#ifndef RABITQ_ADAPTIVE_RHO_MIN
#define RABITQ_ADAPTIVE_RHO_MIN 0.0f
#endif

#ifndef RABITQ_ADAPTIVE_RHO_MAX
#define RABITQ_ADAPTIVE_RHO_MAX 0.99f
#endif

// Candidate-buffer budget used by the AP controller.
#ifndef RABITQ_ADAPTIVE_BUFFER_BOUND
#define RABITQ_ADAPTIVE_BUFFER_BOUND 32
#endif

// Phase policy. Stage I uses progress-based rho tuning from PHASE1_RHO;
// Stage II jumps to PHASE2_THETA/PHASE2_RHO after head-stall detection.
#ifndef RABITQ_ADAPTIVE_PHASE1_THETA
#define RABITQ_ADAPTIVE_PHASE1_THETA RABITQ_ADAPTIVE_THETA_MIN
#endif

#ifndef RABITQ_ADAPTIVE_PHASE1_RHO
#define RABITQ_ADAPTIVE_PHASE1_RHO 0.25f
#endif

#ifndef RABITQ_ADAPTIVE_PHASE2_THETA
#define RABITQ_ADAPTIVE_PHASE2_THETA RABITQ_ADAPTIVE_THETA_MAX
#endif

#ifndef RABITQ_ADAPTIVE_PHASE2_RHO
#define RABITQ_ADAPTIVE_PHASE2_RHO RABITQ_ADAPTIVE_RHO_MAX
#endif

#ifndef RABITQ_ADAPTIVE_TOP1_WARMUP_STALL_ITERS
#define RABITQ_ADAPTIVE_TOP1_WARMUP_STALL_ITERS 4
#endif

#ifndef RABITQ_ADAPTIVE_SEARCH_USE_KTH_CUTOFF
#define RABITQ_ADAPTIVE_SEARCH_USE_KTH_CUTOFF 1
#endif

static_assert(SEARCH_WIDTH >= 1, "Adaptive AP requires SEARCH_WIDTH >= 1");
static_assert(RABITQ_ADAPTIVE_THETA_MIN >= 1, "RABITQ_ADAPTIVE_THETA_MIN must be >= 1");
static_assert(RABITQ_ADAPTIVE_BUFFER_BOUND > 0, "RABITQ_ADAPTIVE_BUFFER_BOUND must be > 0");
static_assert(RABITQ_ADAPTIVE_THETA_MAX >= RABITQ_ADAPTIVE_THETA_MIN,
              "RABITQ_ADAPTIVE_THETA_MAX must be >= RABITQ_ADAPTIVE_THETA_MIN");
static_assert(RABITQ_ADAPTIVE_CHECK_INTERVAL >= 1, "RABITQ_ADAPTIVE_CHECK_INTERVAL must be >= 1");
static_assert(RABITQ_ADAPTIVE_TRACE == 0 || RABITQ_ADAPTIVE_TRACE == 1,
              "RABITQ_ADAPTIVE_TRACE must be 0 or 1");
static_assert(RABITQ_ADAPTIVE_RHO_MIN >= 0.0f && RABITQ_ADAPTIVE_RHO_MIN < 1.0f,
              "RABITQ_ADAPTIVE_RHO_MIN must be in [0, 1)");
static_assert(RABITQ_ADAPTIVE_RHO_MAX >= RABITQ_ADAPTIVE_RHO_MIN && RABITQ_ADAPTIVE_RHO_MAX < 1.0f,
              "RABITQ_ADAPTIVE_RHO_MAX must be in [RHO_MIN, 1)");
static_assert(RABITQ_ADAPTIVE_SEARCH_USE_KTH_CUTOFF == 0 || RABITQ_ADAPTIVE_SEARCH_USE_KTH_CUTOFF == 1,
              "RABITQ_ADAPTIVE_SEARCH_USE_KTH_CUTOFF must be 0 or 1");
static_assert(RABITQ_ADAPTIVE_TOP1_WARMUP_STALL_ITERS >= 1,
              "RABITQ_ADAPTIVE_TOP1_WARMUP_STALL_ITERS must be >= 1");
static_assert(RABITQ_ADAPTIVE_PHASE1_THETA >= RABITQ_ADAPTIVE_THETA_MIN &&
                  RABITQ_ADAPTIVE_PHASE1_THETA <= RABITQ_ADAPTIVE_THETA_MAX,
              "RABITQ_ADAPTIVE_PHASE1_THETA must be within theta bounds");
static_assert(RABITQ_ADAPTIVE_PHASE2_THETA >= RABITQ_ADAPTIVE_THETA_MIN &&
                  RABITQ_ADAPTIVE_PHASE2_THETA <= RABITQ_ADAPTIVE_THETA_MAX,
              "RABITQ_ADAPTIVE_PHASE2_THETA must be within theta bounds");
static_assert(RABITQ_ADAPTIVE_PHASE1_RHO >= RABITQ_ADAPTIVE_RHO_MIN &&
                  RABITQ_ADAPTIVE_PHASE1_RHO <= RABITQ_ADAPTIVE_RHO_MAX,
              "RABITQ_ADAPTIVE_PHASE1_RHO must be within rho bounds");
static_assert(RABITQ_ADAPTIVE_PHASE2_RHO >= RABITQ_ADAPTIVE_RHO_MIN &&
                  RABITQ_ADAPTIVE_PHASE2_RHO <= RABITQ_ADAPTIVE_RHO_MAX,
              "RABITQ_ADAPTIVE_PHASE2_RHO must be within rho bounds");
