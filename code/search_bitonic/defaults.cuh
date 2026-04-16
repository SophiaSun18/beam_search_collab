#pragma once

// Stable mainline config surface.
// This file now owns both:
// 1) externally overridable build-shape inputs
// 2) stable fixed Route-A / reflow constants

#ifndef BEAM_SIZE
#define BEAM_SIZE       128
#endif
#ifndef SEARCH_WIDTH
#define SEARCH_WIDTH    7
#endif
#ifndef BITLEN
#define BITLEN          11
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
inline constexpr int kCandidateBufferSize = 128;
inline constexpr int kResultBufferSize = BEAM_SIZE + kCandidateBufferSize;
inline constexpr int kNeighborBufferSize = kCandidateBufferSize;
inline constexpr int kNoProgressBreakIters = 8;
inline constexpr int kStableRabitqScoreWords = 4;

// When kResultBufferSize fits in one block (≤ BLOCK_SIZE=256), use fast
// in-place bitonic sort.  Otherwise fall back to binary merge + scratch.
inline constexpr bool kUseBitonicMerge = (kResultBufferSize <= BLOCK_SIZE);

// ── Shared-memory budget & launch_bounds auto-calculation ──────────────
// Estimate shared memory per block (bytes) to derive optimal occupancy.
//   s_all_{idx,dist}   : (kUseBitonicMerge ? kResultBufferSize : BEAM_SIZE) * 8
//   s_cand_{idx,dist}  : (kUseBitonicMerge ? 0 : kCandidateBufferSize * 8)
//   s_hash             : kHashSize * 4
//   Fixed overhead     : s_query_cache(512) + s_query_bp(96) + s_expand_pos(28)
//                        + scalar shared vars(~60) + s_pick_counts(32) ≈ 728
inline constexpr int kSmemFixedOverhead = 728;
inline constexpr int kSmemPerBlock =
    (kUseBitonicMerge ? kResultBufferSize : BEAM_SIZE) * 8 +
    (kUseBitonicMerge ? 0 : kCandidateBufferSize * 8) +
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

