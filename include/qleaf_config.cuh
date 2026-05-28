#pragma once

// =============================================================================
// qleaf_config.cuh — Q-LEAF compile-time configuration surface (entry point).
//
// Single header every helper in include/ should pull when it needs the
// formal Q-LEAF compile-time state. Wraps the lower-level
// `qleaf_internal_defaults.cuh` and adds the QLEAF_* public macro
// surface plus ROUTEA_* compatibility aliases on top.
//
// Phase layout (do not reorder):
//   1) Public Q-LEAF macros (QLEAF_*). Users configure these via
//      `-DQLEAF_X=...` on the nvcc command line.
//   2) Compatibility aliases: ROUTEA_* macros are mapped to the
//      corresponding QLEAF_* values so legacy headers (qleaf_internal_defaults.cuh,
//      legacy kernel internals) keep compiling without further edits.
//   3) Pull in `qleaf_internal_defaults.cuh` to materialise the rest of the raw mainline
//      macro defaults, the derived `inline constexpr` constants and the
//      compile-time invariants.
//   4) Static-assert the public Q-LEAF guardrails on the resulting
//      compile-time state.
//
// Formal Q-LEAF supports only:
//   GATE_MODE = ROUTEA_GATE_EXACT (0)        — no local gate
//   GATE_MODE = ROUTEA_GATE_LOCAL_TOPK (1)   — bitplane local top-k / adaptive
//   NAV_MODE  = ROUTEA_NAV_EXACT (0)         — exact ordering
//   NAV_MODE  = ROUTEA_NAV_EDGE_RABITQ_KEY_FINAL_RERANK (8) — Sophia edge key
//   ADAPTIVE_POLICY = 0                      — fixed maximum SEARCH_WIDTH +
//                                              fill/stall active_width +
//                                              adaptive keep ratio
// =============================================================================

// ── Phase 1: Public Q-LEAF macros ────────────────────────────────────────────

#ifndef QLEAF_ENABLE_ADAPTIVE
#define QLEAF_ENABLE_ADAPTIVE 0
#endif

#ifndef QLEAF_CANDIDATE_BUFFER_SIZE
#define QLEAF_CANDIDATE_BUFFER_SIZE 128
#endif

#ifndef QLEAF_HASH_MAX_PROBES
#define QLEAF_HASH_MAX_PROBES 24
#endif

#ifndef QLEAF_ADAPTIVE_MIN_KEEP
#define QLEAF_ADAPTIVE_MIN_KEEP 4
#endif

#ifndef QLEAF_ADAPTIVE_BASE_KEEP
#define QLEAF_ADAPTIVE_BASE_KEEP 8
#endif

#ifndef QLEAF_ADAPTIVE_MAX_KEEP
#define QLEAF_ADAPTIVE_MAX_KEEP 16
#endif

#ifndef QLEAF_ADAPTIVE_TARGET_CANDIDATES
#define QLEAF_ADAPTIVE_TARGET_CANDIDATES 64
#endif

#ifndef QLEAF_ADAPTIVE_STALL_WIDEN_1
#define QLEAF_ADAPTIVE_STALL_WIDEN_1 4
#endif

#ifndef QLEAF_ADAPTIVE_STALL_WIDEN_2
#define QLEAF_ADAPTIVE_STALL_WIDEN_2 8
#endif

// ── Phase 2: Compatibility aliases (ROUTEA_* -> QLEAF_*) ─────────────────────
//
// `#ifndef`-guarded so an explicit `-DROUTEA_X=...` from the command line
// still wins; otherwise the legacy macro picks up the QLEAF_* value.
// Either spelling reaches the same compile-time state.

#ifndef ROUTEA_ADAPTIVE_GATE
#define ROUTEA_ADAPTIVE_GATE QLEAF_ENABLE_ADAPTIVE
#endif
#ifndef ROUTEA_CANDIDATE_BUFFER_SIZE
#define ROUTEA_CANDIDATE_BUFFER_SIZE QLEAF_CANDIDATE_BUFFER_SIZE
#endif
#ifndef ROUTEA_HASH_MAX_PROBES
#define ROUTEA_HASH_MAX_PROBES QLEAF_HASH_MAX_PROBES
#endif
#ifndef ROUTEA_ADAPTIVE_MIN_KEEP
#define ROUTEA_ADAPTIVE_MIN_KEEP QLEAF_ADAPTIVE_MIN_KEEP
#endif
#ifndef ROUTEA_ADAPTIVE_BASE_KEEP
#define ROUTEA_ADAPTIVE_BASE_KEEP QLEAF_ADAPTIVE_BASE_KEEP
#endif
#ifndef ROUTEA_ADAPTIVE_MAX_KEEP
#define ROUTEA_ADAPTIVE_MAX_KEEP QLEAF_ADAPTIVE_MAX_KEEP
#endif
#ifndef ROUTEA_ADAPTIVE_TARGET_CANDIDATES
#define ROUTEA_ADAPTIVE_TARGET_CANDIDATES QLEAF_ADAPTIVE_TARGET_CANDIDATES
#endif
#ifndef ROUTEA_ADAPTIVE_STALL_WIDEN_1
#define ROUTEA_ADAPTIVE_STALL_WIDEN_1 QLEAF_ADAPTIVE_STALL_WIDEN_1
#endif
#ifndef ROUTEA_ADAPTIVE_STALL_WIDEN_2
#define ROUTEA_ADAPTIVE_STALL_WIDEN_2 QLEAF_ADAPTIVE_STALL_WIDEN_2
#endif

// ── Phase 3: Raw mainline defaults + derived constexpr + invariants ──────────
//
// `utils.cuh` defines WARP_SIZE / BLOCK_SIZE / MAX_INDEX / gpu_long_t /
// GraphGPU and the warp-level primitives that qleaf_internal_defaults.cuh and the kernel
// headers reference. Pull it in first so every consumer of qleaf_config.cuh
// sees a complete dependency chain.
#include "utils.cuh"
#include "qleaf_internal_defaults.cuh"

// ── Phase 4: Public Q-LEAF guardrails (per migration plan) ───────────────────
static_assert(BITLEN == 11,
              "Standalone Q-LEAF formal path currently expects BITLEN=11");
static_assert(REFLOW_TOPN > 0 && REFLOW_TOPN <= 32,
              "REFLOW_TOPN must be in [1, 32]");
static_assert(SEARCH_WIDTH > 0 && SEARCH_WIDTH <= 16,
              "SEARCH_WIDTH must be in [1, 16]");
static_assert(kCandidateBufferSize >= kRouteAMaxPreHashCandidates,
              "Q-LEAF candidate buffer must hold all pre-hash candidates "
              "for the selected formal mode.");
