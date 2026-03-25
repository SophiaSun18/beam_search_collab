#pragma once

// Stable mainline config surface.
// This file now owns both:
// 1) externally overridable build-shape inputs
// 2) stable fixed Route-A / reflow constants

#ifndef BEAM_SIZE
#define BEAM_SIZE       500
#endif
#ifndef SEARCH_WIDTH
#define SEARCH_WIDTH    4      
#endif
#ifndef BITLEN
#define BITLEN          12
#endif

// Stable fixed compile-time constants for the current mainline.
#define HASH_INSERT_MAX_PROBES 24
#define HASH_INSERT_LIMIT_START_FILL 160
#define RABITQ_EARLY_STOP_STALL_ITERS 64
#define RABITQ_EARLY_STOP_WARMUP_ITERS 8
#define RABITQ_MAX_ITER_CAP 256
#define RABITQ_STALL_IMPROVE_EPS_ABS 1e-4f
#define RABITQ_STALL_IMPROVE_EPS_REL 1e-4f
#define REFLOW_TOPN 20
#define REFLOW_POP_BATCH 7

// Stable compile-time constants used by the fixed mainline.
inline constexpr unsigned kHashSize = (1u << BITLEN);
inline constexpr int kCandidateBufferSize = 64;
inline constexpr int kNeighborBufferSize = (BEAM_SIZE > 256 ? BEAM_SIZE : 256);
inline constexpr int kNoProgressBreakIters = 8;
inline constexpr int kStableRabitqScoreWords = 4;
