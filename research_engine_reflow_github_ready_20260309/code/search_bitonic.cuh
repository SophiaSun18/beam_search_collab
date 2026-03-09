#pragma once
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cstring>

#ifndef BEAM_SIZE
#define BEAM_SIZE       500
#endif
#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK   500
#endif
#ifndef SEARCH_WIDTH
#define SEARCH_WIDTH    4      
#endif
#ifndef BITLEN
#define BITLEN          12
#endif
#ifndef HASH_SIZE
#define HASH_SIZE       (1u << (BITLEN))
#endif
#ifndef MAX_ITERATIONS
#define MAX_ITERATIONS  2000
#endif
#ifndef USE_SEED
#define USE_SEED        false
#endif
#ifndef PRUNE_CONFIG
#define PRUNE_CONFIG    false
#endif
#ifndef CANDIDATE_BUFFER_SIZE
// Legacy-stable default; can be overridden by compile flags for controlled A/B.
#define CANDIDATE_BUFFER_SIZE 64
#endif
#ifndef CANDIDATE_SELECT_STRICT
#define CANDIDATE_SELECT_STRICT 0
#endif
#ifndef CANDIDATE_SELECT_STRICT_ITERS
#define CANDIDATE_SELECT_STRICT_ITERS 0
#endif
#ifndef CANDIDATE_SELECT_STRICT_ADAPTIVE
#define CANDIDATE_SELECT_STRICT_ADAPTIVE 0
#endif
#ifndef CANDIDATE_SELECT_STRICT_ADAPTIVE_MIN_VALID
#define CANDIDATE_SELECT_STRICT_ADAPTIVE_MIN_VALID 96
#endif
#ifndef CANDIDATE_SELECT_STRICT_FAST
#define CANDIDATE_SELECT_STRICT_FAST 0
#endif
#ifndef CANDIDATE_SELECT_STRICT_FAST_DGS_ONLY
#define CANDIDATE_SELECT_STRICT_FAST_DGS_ONLY 1
#endif
#ifndef STRICT_FAST_DEBUG_PRINT
#define STRICT_FAST_DEBUG_PRINT 0
#endif
#ifndef STRICT_FAST_MODE
#define STRICT_FAST_MODE 0
#endif
#ifndef STRICT_FAST_AUTO_VALID_THRESHOLD
#define STRICT_FAST_AUTO_VALID_THRESHOLD 160
#endif
#ifndef STRICT_FAST_SECONDARY_SORT_MODE
// 0: legacy secondary full bitonic sort (kept for A/B)
// 1: k-way merge of per-warp sorted top-k (no secondary full sort)
#define STRICT_FAST_SECONDARY_SORT_MODE 1
#endif
#ifndef STRICT_FAST_WARP_TOPK_MODE
// 0: per-warp full 32-lane bitonic sort, then keep top-k lanes
// 1: per-warp iterative argmin select (k selections, lower compare cost for k<<32)
#define STRICT_FAST_WARP_TOPK_MODE 1
#endif
#ifndef STRICT_FAST_SECONDARY_MERGE_MODE
// 0: tid0 serial k-way merge (legacy)
// 1: warp0 cooperative k-way merge (lower scalar bottleneck)
#define STRICT_FAST_SECONDARY_MERGE_MODE 1
#endif
#ifndef STRICT_TOPK_PARTIAL_SELECT_MODE
// 0: legacy block-wide full bitonic sort on candidate_tmp
// 1: serial partial top-k select (top64) from candidate_tmp
#define STRICT_TOPK_PARTIAL_SELECT_MODE 0
#endif
#ifndef ENABLE_GHOST_STAGE
#define ENABLE_GHOST_STAGE 0
#endif
#ifndef GHOST_SCALE_FACTOR
#define GHOST_SCALE_FACTOR 10000
#endif
#ifndef GHOST_MIN_NODES
#define GHOST_MIN_NODES 128
#endif
#ifndef GHOST_REUSE_STARTS
#define GHOST_REUSE_STARTS 1
#endif
#ifndef GHOST_GRAPH_DEG
#define GHOST_GRAPH_DEG 16
#endif
#ifndef GHOST_STAGE_M
#define GHOST_STAGE_M 16
#endif
#ifndef GHOST_STAGE_R
#define GHOST_STAGE_R 8
#endif
#ifndef GHOST_STAGE_ITERS
#define GHOST_STAGE_ITERS 3
#endif
#ifndef ENABLE_DGS
#define ENABLE_DGS 0
#endif
#ifndef DGS_TOPN
#define DGS_TOPN 8
#endif
#ifndef DGS_TOPN_EARLY
#define DGS_TOPN_EARLY DGS_TOPN
#endif
#ifndef DGS_TOPN_EARLY_ITERS
#define DGS_TOPN_EARLY_ITERS 0
#endif
#ifndef DGS_TOPN_MAX
#define DGS_TOPN_MAX 64
#endif
#ifndef DGS_TOPK_SELECT_FAST
#define DGS_TOPK_SELECT_FAST 0
#endif
#ifndef DGS_TOPK_SELECT_FAST_MAX
#define DGS_TOPK_SELECT_FAST_MAX 8
#endif
#ifndef DGS_RAW_KEEP
#define DGS_RAW_KEEP 0
#endif
#ifndef USE_SUBWARP_L2_128
#define USE_SUBWARP_L2_128 0
#endif
#ifndef DGS_CAP_EXPAND_WIDTH_BY_CANDIDATE
#define DGS_CAP_EXPAND_WIDTH_BY_CANDIDATE 1
#endif
#ifndef HASH_WARP_DEDUP
#define HASH_WARP_DEDUP 1
#endif
#ifndef SKIP_ATOMIC_WHEN_POOL_FULL
// 1: when staging pool is already full, skip hash insert / append atomics for overflow entries.
#define SKIP_ATOMIC_WHEN_POOL_FULL 1
#endif
#ifndef HASH_INSERT_MAX_PROBES
// 0: full probe loop (legacy); >0: limit probes in high-fill stage to reduce atomicCAS contention.
#define HASH_INSERT_MAX_PROBES 0
#endif
#ifndef HASH_INSERT_LIMIT_START_FILL
// Start limited-probe insertion when staged neighbor count reaches this threshold.
#define HASH_INSERT_LIMIT_START_FILL 160
#endif
#ifndef HASH_INSERT_HIGH_FILL_START
// In very high fill stage, use a tighter probe cap to reduce atomicCAS retries.
#define HASH_INSERT_HIGH_FILL_START 224
#endif
#ifndef HASH_INSERT_MAX_PROBES_HIGH_FILL
// 0: disable high-fill tighter cap; >0: max probes in high-fill stage.
#define HASH_INSERT_MAX_PROBES_HIGH_FILL 0
#endif
#ifndef WARP_L1_DEDUP_ENABLE
// 1: use per-warp local no-atomic hash to pre-dedup candidates before global atomic hash insert.
#define WARP_L1_DEDUP_ENABLE 0
#endif
#ifndef WARP_L1_HASH_BITLEN
// local hash size per warp = 1 << WARP_L1_HASH_BITLEN
#define WARP_L1_HASH_BITLEN 7
#endif
#ifndef PICK_EXPANDERS_BLOCK_WIDE
#define PICK_EXPANDERS_BLOCK_WIDE 1
#endif
#ifndef DGS_SEARCH_WIDTH
#define DGS_SEARCH_WIDTH 8
#endif
#ifndef DGS_DECOUPLE_EXPAND_WIDTH
// 0: legacy behavior, DGS controls both filtering and expand width (DGS_SEARCH_WIDTH)
// 1: decoupled behavior, DGS controls filtering only; expand width uses DGS_DECOUPLED_EXPAND_WIDTH
#define DGS_DECOUPLE_EXPAND_WIDTH 0
#endif
#ifndef DGS_DECOUPLED_EXPAND_WIDTH
#define DGS_DECOUPLED_EXPAND_WIDTH SEARCH_WIDTH
#endif
#ifndef DGS_FORCE_EXPAND_WIDTH
// 0: disabled
// >0: force expand width to a constant value, independent of warmup/use_wide_expand.
//     Useful for warmup mechanism diagnosis (separate filter-effect vs width-effect).
#define DGS_FORCE_EXPAND_WIDTH 0
#endif
#ifndef DGS_FORCE_WIDE_EXPAND
// -1: disabled (default behavior)
//  0: force disable wide-path expansion
//  1: force enable wide-path expansion
// Useful for warmup mechanism diagnosis: keep path stable while toggling DGS filter gate.
#define DGS_FORCE_WIDE_EXPAND -1
#endif
#ifndef DGS_TREE_OFF_MODE
// 0: disabled
// 1: tree-off disables wide path + DGS filter in scheduled iterations (scheme A)
// 2: tree-off disables DGS filter only, keeps wide path unchanged (scheme B)
// 3: keep wide/filter path unchanged; only cap expand width in scheduled iterations
#define DGS_TREE_OFF_MODE 0
#endif
#ifndef DGS_TREE_OFF_PATTERN
// 1: off at t = 0,1,3,7,15,...
// 2: off at t = 0,2,6,14,... (milder, recommended default)
#define DGS_TREE_OFF_PATTERN 2
#endif
#ifndef DGS_TREE_START_ITERS
// Start applying tree schedule after this many active iterations.
#define DGS_TREE_START_ITERS 0
#endif
#ifndef DGS_TREE_MIN_ACTIVE_ITERS
// Disable tree schedule if active window is too short.
#define DGS_TREE_MIN_ACTIVE_ITERS 0
#endif
#ifndef DGS_TREE_RELAX_TOPN
// 0: disabled (legacy hard tree behavior)
// >0: in mode=2 tree-off rounds, keep DGS active and relax topN to this value.
#define DGS_TREE_RELAX_TOPN 0
#endif
#ifndef DGS_TREE_WIDE_CAP
// 0: disabled
// >0: in mode=3 tree-off rounds, cap expand width to this value while staying on wide path.
#define DGS_TREE_WIDE_CAP 0
#endif
#ifndef DGS_TREE_REQUIRE_STALL
// 0: schedule by iteration only
// 1: tree event applies only when stall counter reaches threshold
#define DGS_TREE_REQUIRE_STALL 0
#endif
#ifndef DGS_TREE_MODE2_EARLY_WARMUP_ITERS
// 0: disabled
// >0: in mode=2, force non-wide path for first N iterations (warmup-like width control).
#define DGS_TREE_MODE2_EARLY_WARMUP_ITERS 0
#endif
#ifndef DGS_TREE_STALL_THRESHOLD
#define DGS_TREE_STALL_THRESHOLD 1
#endif
#ifndef DGS_TREE_MAX_HITS
// 0: unlimited tree events in active window
// >0: max number of tree events per query
#define DGS_TREE_MAX_HITS 0
#endif
#ifndef EXPAND_WIDTH_MAX
#define EXPAND_WIDTH_MAX ((DGS_SEARCH_WIDTH) > (SEARCH_WIDTH) ? (DGS_SEARCH_WIDTH) : (SEARCH_WIDTH))
#endif
#ifndef DGS_EST_ITERS
#define DGS_EST_ITERS (BEAM_SIZE / DGS_SEARCH_WIDTH)
#endif
#ifndef NEIGHBOR_BUFFER_SIZE
// Neighbor staging is used in two phases:
// 1) new-neighbor collection (cap=256)
// 2) merged frontier staging (cap=beam_size)
// Keep buffer >= max(256, BEAM_SIZE) to avoid OOB for small beam sizes.
#define NEIGHBOR_BUFFER_SIZE ((BEAM_SIZE) > 256 ? (BEAM_SIZE) : 256)
#endif
#ifndef NO_PROGRESS_BREAK_ITERS
// Break query loop after consecutive no-progress rounds to avoid long tails.
#define NO_PROGRESS_BREAK_ITERS 8
#endif
#ifndef DGS_COOL_RATIO
#define DGS_COOL_RATIO 0.30f
#endif
#ifndef DGS_WARMUP_ITERS
#define DGS_WARMUP_ITERS 0
#endif
#ifndef DGS_SIGNBIT_INVERT
#define DGS_SIGNBIT_INVERT 0
#endif
#ifndef DGS_DIVERSITY_KEEP
#define DGS_DIVERSITY_KEEP 0
#endif
#ifndef DGS_MAX_DEG
#define DGS_MAX_DEG 256
#endif
#ifndef DGS_MAX_WORDS
#define DGS_MAX_WORDS 16
#endif
#ifndef DGS_DEBUG_PRINT
#define DGS_DEBUG_PRINT 0
#endif
#ifndef ENABLE_RABITQ
#define ENABLE_RABITQ 0
#endif
#ifndef RABITQ_TOPN
#define RABITQ_TOPN 8
#endif
#ifndef RABITQ_TOPN_EARLY
#define RABITQ_TOPN_EARLY RABITQ_TOPN
#endif
#ifndef RABITQ_TOPN_EARLY_ITERS
#define RABITQ_TOPN_EARLY_ITERS 0
#endif
#ifndef RABITQ_ADAPTIVE_TOPN_BOOST
// >0: dynamically increase tile keep-count by this value while search is still
// clearly improving (low stall iters). Helps reduce early false-prune recall loss.
#define RABITQ_ADAPTIVE_TOPN_BOOST 0
#endif
#ifndef RABITQ_ADAPTIVE_TOPN_STALL_ITERS
// Adaptive topn boost is active when stall_iters < this threshold.
#define RABITQ_ADAPTIVE_TOPN_STALL_ITERS 2
#endif
#ifndef RABITQ_ADAPTIVE_TOPN_STALL_ITERS_LOW_BEAM
// Optional low-beam override to keep topn boost active longer.
// 0 disables the override.
#define RABITQ_ADAPTIVE_TOPN_STALL_ITERS_LOW_BEAM 0
#endif
#ifndef RABITQ_ADAPTIVE_TOPN_LOW_BEAM_MAX
// Apply low-beam adaptive stall override when BEAM_SIZE <= this value.
// <=0 disables the override.
#define RABITQ_ADAPTIVE_TOPN_LOW_BEAM_MAX 0
#endif
#ifndef RABITQ_TWO_PHASE_EXACT
// 1: GPU-oriented dual-pool approximation:
//    sort staged neighbors by RabitQ approx, exact-evaluate in micro-batches,
//    and update an exact cutoff after each batch.
#define RABITQ_TWO_PHASE_EXACT 0
#endif
#ifndef RABITQ_TWO_PHASE_MIN_NEIGHBORS
// Activate two-phase exact path only when staged neighbors reach this threshold.
#define RABITQ_TWO_PHASE_MIN_NEIGHBORS 64
#endif
#ifndef RABITQ_TWO_PHASE_BATCH
// Exact micro-batch size in two-phase mode.
#define RABITQ_TWO_PHASE_BATCH 16
#endif
#ifndef RABITQ_TWO_PHASE_ANCHORS
// Number of forced-exact anchors per micro-batch.
#define RABITQ_TWO_PHASE_ANCHORS 8
#endif
#ifndef RABITQ_TWO_PHASE_APPROX_RATIO
// Skip exact when approx > exact_cutoff * ratio. <=0 disables this gate.
#define RABITQ_TWO_PHASE_APPROX_RATIO 1.05f
#endif
#ifndef RABITQ_GLOBAL_APPROX_PRESELECT
// 1: keep full tile lanes after RabitQ scoring, then run one global
// approximate preselect before exact distance stage.
// 0: keep legacy per-tile topn prune behavior.
#define RABITQ_GLOBAL_APPROX_PRESELECT 0
#endif
#ifndef RABITQ_GLOBAL_APPROX_PRESELECT_EXTRA
// Extra keep count on top of (expanders * topn) in global approx preselect.
#define RABITQ_GLOBAL_APPROX_PRESELECT_EXTRA 0
#endif
#ifndef RABITQ_GLOBAL_APPROX_PRESELECT_REQUIRE_FRONTIER_FULL
// 1: apply global approx preselect only after frontier is full.
// 0: allow from first iteration.
#define RABITQ_GLOBAL_APPROX_PRESELECT_REQUIRE_FRONTIER_FULL 1
#endif
#ifndef RABITQ_MAX_WORDS
#define RABITQ_MAX_WORDS 32
#endif
#ifndef RABITQ_WARMUP_ITERS
#define RABITQ_WARMUP_ITERS 0
#endif
#ifndef RABITQ_COOL_RATIO
#define RABITQ_COOL_RATIO 0.0f
#endif
#ifndef RABITQ_ACTIVE_ITERS
// 0: use RABITQ_COOL_RATIO to derive active window
// >0: directly set RABITQ active window iterations
#define RABITQ_ACTIVE_ITERS 0
#endif
#ifndef RABITQ_DISABLE_ON_STALL_ITERS
// >0: disable RabitQ filter after this many consecutive no-improvement rounds.
#define RABITQ_DISABLE_ON_STALL_ITERS 0
#endif
#ifndef RABITQ_DISABLE_ON_STALL_ITERS_LOW_BEAM
// Optional low-beam override for stall-based RabitQ disable.
#define RABITQ_DISABLE_ON_STALL_ITERS_LOW_BEAM 0
#endif
#ifndef RABITQ_DISABLE_ON_STALL_LOW_BEAM_MAX
// Apply low-beam stall-disable override when BEAM_SIZE <= this value.
#define RABITQ_DISABLE_ON_STALL_LOW_BEAM_MAX 0
#endif
#ifndef RABITQ_DISABLE_IF_NO_PRUNE_ITERS
// >0: if RabitQ is active but yields zero tile-level pruning for N consecutive
// iterations, disable RabitQ for the rest of this query.
#define RABITQ_DISABLE_IF_NO_PRUNE_ITERS 0
#endif
#ifndef RABITQ_RAW_KEEP
#define RABITQ_RAW_KEEP 0
#endif
#ifndef RABITQ_RAW_KEEP_LOW
// Optional beam-adaptive raw-keep for BEAM_SIZE <= split_beam.
#define RABITQ_RAW_KEEP_LOW 0
#endif
#ifndef RABITQ_RAW_KEEP_HIGH
// Optional beam-adaptive raw-keep for BEAM_SIZE > split_beam.
#define RABITQ_RAW_KEEP_HIGH 0
#endif
#ifndef RABITQ_RAW_KEEP_SPLIT_BEAM
#define RABITQ_RAW_KEEP_SPLIT_BEAM 0
#endif
#ifndef RABITQ_MIN_BEAM
#define RABITQ_MIN_BEAM 0
#endif
#ifndef RABITQ_MAX_BEAM
// <=0 means no upper beam gate.
#define RABITQ_MAX_BEAM 0
#endif
#ifndef RABITQ_MIN_EXPANDERS
#define RABITQ_MIN_EXPANDERS 0
#endif
#ifndef RABITQ_NEIGHBOR_BUDGET
// 0: disabled; >0: cap staged neighbors per iteration while RabitQ gate is active.
#define RABITQ_NEIGHBOR_BUDGET 0
#endif
#ifndef RABITQ_NEIGHBOR_BUDGET_PRESAMPLE
// 1: pre-limit keep-mask to remaining budget before hash insert/atomicAdd.
#define RABITQ_NEIGHBOR_BUDGET_PRESAMPLE 1
#endif
#ifndef RABITQ_NEIGHBOR_BUDGET_REQUIRE_FRONTIER_FULL
// 1: only apply neighbor-budget cap when frontier is already full.
#define RABITQ_NEIGHBOR_BUDGET_REQUIRE_FRONTIER_FULL 0
#endif
#ifndef RABITQ_NEIGHBOR_BUDGET_LOW
// Optional beam-adaptive budget: used when BEAM_SIZE <= split_beam.
#define RABITQ_NEIGHBOR_BUDGET_LOW 0
#endif
#ifndef RABITQ_NEIGHBOR_BUDGET_HIGH
// Optional beam-adaptive budget: used when BEAM_SIZE > split_beam.
#define RABITQ_NEIGHBOR_BUDGET_HIGH 0
#endif
#ifndef RABITQ_NEIGHBOR_BUDGET_SPLIT_BEAM
#define RABITQ_NEIGHBOR_BUDGET_SPLIT_BEAM 0
#endif
#ifndef RABITQ_STAGE_CAP
// 0: disabled; >0: hard cap staged neighbors before exact distance.
#define RABITQ_STAGE_CAP 0
#endif
#ifndef RABITQ_FACTOR_EPS
#define RABITQ_FACTOR_EPS 1e-8f
#endif
#ifndef RABITQ_CPU_ALIGN_FLOW
// 1: CPU-aligned approximate-first flow:
//    allow approx distance to drive candidate expansion without requiring frontier_full.
// 0: keep legacy GPU gating.
#define RABITQ_CPU_ALIGN_FLOW 0
#endif
#ifndef RABITQ_CPU_ALIGN_FORCE_EXACT_DIST
// 1: in CPU-align flow, do not skip exact distance stage by approx values.
// Useful for isolating approximation-stage error vs queue/cutoff error.
#define RABITQ_CPU_ALIGN_FORCE_EXACT_DIST 0
#endif
#ifndef RABITQ_CPU_ALIGN_EXACT_EXPANDER_SQRY
// 1: in CPU-align flow, recompute exact sqr_y for each expander before RabitQ score.
// 0: trust frontier distance as sqr_y (avoid redundant exact L2 on expander).
#define RABITQ_CPU_ALIGN_EXACT_EXPANDER_SQRY 1
#endif
#ifndef RABITQ_CPU_ALIGN_SKIP_TRIVIAL_SCORE
// 1: in CPU-align approx-first mode, if topn==deg (no tile-level RabitQ prune),
// skip heavy RabitQ score/select path and use a lightweight approx score.
#define RABITQ_CPU_ALIGN_SKIP_TRIVIAL_SCORE 0
#endif
#ifndef RABITQ_CPU_ALIGN_TRIVIAL_APPROX_WORDS
// Lightweight approx score word count when RABITQ_CPU_ALIGN_SKIP_TRIVIAL_SCORE=1.
// Clamped to [1, rabitq_score_words].
#define RABITQ_CPU_ALIGN_TRIVIAL_APPROX_WORDS 1
#endif
#ifndef RABITQ_CPU_ALIGN_TILE_FULLSCAN
// 1: in CPU-align flow, force tile keep-count topn=deg (no tile-level prune).
// 0: keep configured RabitQ topn at tile stage.
#define RABITQ_CPU_ALIGN_TILE_FULLSCAN 0
#endif
#ifndef RABITQ_CPU_PARITY_NEIGHBOR_RECHECK
// 1: keep legacy parity behavior that can force neighbor-side exact checks.
// 0: rely on pop-exact stream + realtime exact-cutoff feedback only.
#define RABITQ_CPU_PARITY_NEIGHBOR_RECHECK 0
#endif
#ifndef RABITQ_CPU_PARITY_FEEDBACK_CAP
// 0: use min(K, beam_size) as exact feedback pool cap.
// >0: cap exact feedback pool to this size for earlier/tighter cutoff.
#define RABITQ_CPU_PARITY_FEEDBACK_CAP 0
#endif
#ifndef RABITQ_CPU_PARITY_MODE
// 1: strict CPU-style parity mode:
//    - use approximate pool for expansion scheduling
//    - compute exact L2 primarily at pop/expander points
//    - maintain a dedicated exact top-k result pool
#define RABITQ_CPU_PARITY_MODE 0
#endif
#ifndef RABITQ_CPU_PARITY_EXACT_POOL
// 1: keep parity exact-topk/stream buffers and merge path.
// 0: keep parity flow controls but output from frontier-only path.
#define RABITQ_CPU_PARITY_EXACT_POOL 1
#endif
#ifndef RABITQ_GPU_DUALPOOL_ENGINE
// 1: unified GPU dual-pool engine to align with CPU RabitQ main loop:
//    pop-by-approx -> exact(pop) -> approx-neighbor insert/merge.
// This mode intentionally disables mixed legacy paths that blur semantics.
#define RABITQ_GPU_DUALPOOL_ENGINE 0
#endif
#ifndef RABITQ_GPU_DUALPOOL_ANCHOR_EXACT
// In unified dual-pool mode, exact-evaluate top-M staged neighbors by approx
// as a lightweight calibration anchor. 0 keeps pure approx-neighbor behavior.
#define RABITQ_GPU_DUALPOOL_ANCHOR_EXACT 0
#endif
#ifndef RABITQ_GPU_DUALPOOL_POP_BATCH
// Batch size of popped expanders per outer iteration in unified dual-pool mode.
// 1 is strict CPU-style; larger values are GPU-oriented parallelization.
#define RABITQ_GPU_DUALPOOL_POP_BATCH 1
#endif
// NOTE:
// We intentionally avoid compile-time silent collapse (#undef/#define) of parity
// or approx-push modes. Hidden rewrites make experiment semantics ambiguous.
#ifndef RABITQ_CPU_PARITY_POP_BATCH
// In CPU parity mode, number of popped expanders per outer iteration.
// 1 is strict CPU behavior; >1 is parallelized parity approximation.
#define RABITQ_CPU_PARITY_POP_BATCH 1
#endif
#ifndef RABITQ_MAIN_POP_BATCH
// >0: cap popped expanders per outer iteration when RabitQ gate is active.
// 0: keep normal expand-width policy.
#define RABITQ_MAIN_POP_BATCH 0
#endif
#ifndef RABITQ_MAIN_POP_BATCH_START_ITER
// Iteration to start pop-batch cap. 0 means active from the first iteration.
#define RABITQ_MAIN_POP_BATCH_START_ITER 0
#endif
#ifndef RABITQ_MAIN_POP_BATCH_REQUIRE_FRONTIER_FULL
// 1: apply pop-batch cap only when frontier is full.
// 0: ignore frontier fullness for this cap.
#define RABITQ_MAIN_POP_BATCH_REQUIRE_FRONTIER_FULL 0
#endif
#ifndef RABITQ_MAIN_POP_BATCH_REQUIRE_STALL
// 1: apply pop-batch cap only when stall counter reaches threshold.
// 0: ignore stall counter for this cap.
#define RABITQ_MAIN_POP_BATCH_REQUIRE_STALL 0
#endif
#ifndef RABITQ_MAIN_POP_BATCH_STALL_ITERS
// stall threshold used when RABITQ_MAIN_POP_BATCH_REQUIRE_STALL=1.
#define RABITQ_MAIN_POP_BATCH_STALL_ITERS RABITQ_STALL_TRIGGER
#endif
#ifndef RABITQ_FEEDBACK_MERGE_BATCH
// 0: disabled
// >0: after candidate sort, update a provisional exact cutoff every N candidates
// and early-truncate dominated tail candidates before frontier merge.
#define RABITQ_FEEDBACK_MERGE_BATCH 0
#endif
#ifndef RABITQ_CPU_PARITY_START_CAP
// 0: keep normal start count policy; >0: cap start count in parity mode.
#define RABITQ_CPU_PARITY_START_CAP 0
#endif
#ifndef RABITQ_CPU_PARITY_FORCE_EXPAND_WIDTH
// 0: keep normal expand_width policy and only enforce a minimum pop batch.
// 1: force expand_width = RABITQ_CPU_PARITY_POP_BATCH (legacy parity behavior).
#define RABITQ_CPU_PARITY_FORCE_EXPAND_WIDTH 0
#endif
#ifndef RABITQ_CPU_PARITY_EARLY_EXACT_ITERS
// In parity mode, force exact L2 neighbor distance for first N iterations.
// This stabilizes early search direction before approximation-dominant stage.
#define RABITQ_CPU_PARITY_EARLY_EXACT_ITERS 8
#endif
#ifndef RABITQ_CPU_PARITY_FLUSH_INTERVAL
// In CPU parity mode, merge exact stream into exact top-k every N iterations.
// 0 disables periodic flush (flush only on stream-full / break / finalize).
#define RABITQ_CPU_PARITY_FLUSH_INTERVAL 4
#endif
#ifndef RABITQ_CPU_PARITY_STREAM_CAP
// Per-query exact stream capacity before forced flush.
#define RABITQ_CPU_PARITY_STREAM_CAP BEAM_SIZE
#endif
#ifndef RABITQ_CPU_PARITY_FINAL_FRONTIER_EXACT
// 1: before final output in parity mode, exact-evaluate frontier and merge into
// exact top-k to avoid approximate-distance artifacts in final ranking.
#define RABITQ_CPU_PARITY_FINAL_FRONTIER_EXACT 0
#endif
#ifndef RABITQ_CPU_PARITY_UPDATE_RESULTS
// 1: CPU-style final exact completion from exact-topk one-hop neighbors.
// This helps recover recall when main loop is approximation-dominant.
#define RABITQ_CPU_PARITY_UPDATE_RESULTS 0
#endif
#ifndef RABITQ_CPU_PARITY_USE_EXACT_CUTOFF
// 1: feed exact-topk cutoff back into candidate admission.
// 0: keep candidate admission on approximate frontier cutoff (CPU-like dual-pool behavior).
#define RABITQ_CPU_PARITY_USE_EXACT_CUTOFF 0
#endif
#ifndef RABITQ_CPU_PARITY_REALTIME_MERGE
// 1: merge exact stream into exact-topk every iteration before candidate stage.
// This is required when exact cutoff is used for candidate admission.
// For output-only exact pool, periodic/final merge is usually enough.
#if RABITQ_CPU_PARITY_USE_EXACT_CUTOFF
#define RABITQ_CPU_PARITY_REALTIME_MERGE 1
#else
#define RABITQ_CPU_PARITY_REALTIME_MERGE 0
#endif
#endif
#ifndef RABITQ_REFLOW_FIXED
// 1: lock RabitQ to the reflow baseline path so experiments focus on mechanism
// implementation quality instead of switch interactions.
#define RABITQ_REFLOW_FIXED 0
#endif
#if RABITQ_REFLOW_FIXED
#ifndef REFLOW_TOPN
#define REFLOW_TOPN 20
#endif
#ifndef REFLOW_POP_BATCH
#define REFLOW_POP_BATCH 7
#endif
#ifndef REFLOW_ANCHOR_EXACT
#define REFLOW_ANCHOR_EXACT 32
#endif
#ifndef REFLOW_SCORE_WORDS
#define REFLOW_SCORE_WORDS 4
#endif
#ifndef REFLOW_SKIP_ATOMIC_POLICY
// 0: force SKIP_ATOMIC_WHEN_POOL_FULL=0
// 1: force SKIP_ATOMIC_WHEN_POOL_FULL=1
// 2: auto by beam (>=400 -> 0, else 1)
#define REFLOW_SKIP_ATOMIC_POLICY 0
#endif
#ifndef REFLOW_FASTSCAN_ENABLE
#define REFLOW_FASTSCAN_ENABLE 1
#endif
#ifndef REFLOW_PACKED_CODE_ENABLE
#define REFLOW_PACKED_CODE_ENABLE 1
#endif
#ifndef REFLOW_TOPN_EARLY
#define REFLOW_TOPN_EARLY REFLOW_TOPN
#endif
#ifndef REFLOW_TOPN_EARLY_ITERS
#define REFLOW_TOPN_EARLY_ITERS 0
#endif
#ifndef REFLOW_USE_FACTORS
#define REFLOW_USE_FACTORS 0
#endif
#ifndef REFLOW_FACTOR_LIGHT_PRESELECT
#define REFLOW_FACTOR_LIGHT_PRESELECT 0
#endif
#ifndef REFLOW_FACTOR_PRESELECT_EXTRA
#define REFLOW_FACTOR_PRESELECT_EXTRA 0
#endif
#ifndef REFLOW_FACTOR_PACKED_DQVQ
#define REFLOW_FACTOR_PACKED_DQVQ 1
#endif
#ifndef REFLOW_FACTOR_FP16_DQVQ
#define REFLOW_FACTOR_FP16_DQVQ 1
#endif
#ifndef REFLOW_QUERY_PROJ_ENABLE
#define REFLOW_QUERY_PROJ_ENABLE 0
#endif
#ifndef REFLOW_GPU_DUALPOOL_ENGINE
#define REFLOW_GPU_DUALPOOL_ENGINE 1
#endif
#undef RABITQ_TOPN
#define RABITQ_TOPN REFLOW_TOPN
#undef RABITQ_TOPN_EARLY
#define RABITQ_TOPN_EARLY REFLOW_TOPN_EARLY
#undef RABITQ_TOPN_EARLY_ITERS
#define RABITQ_TOPN_EARLY_ITERS REFLOW_TOPN_EARLY_ITERS
#undef RABITQ_ADAPTIVE_TOPN_BOOST
#define RABITQ_ADAPTIVE_TOPN_BOOST 0
#undef RABITQ_ADAPTIVE_TOPN_STALL_ITERS
#define RABITQ_ADAPTIVE_TOPN_STALL_ITERS 0
#undef RABITQ_ADAPTIVE_TOPN_STALL_ITERS_LOW_BEAM
#define RABITQ_ADAPTIVE_TOPN_STALL_ITERS_LOW_BEAM 0
#undef RABITQ_ADAPTIVE_TOPN_LOW_BEAM_MAX
#define RABITQ_ADAPTIVE_TOPN_LOW_BEAM_MAX 0
#undef RABITQ_DISABLE_ON_STALL_ITERS
#define RABITQ_DISABLE_ON_STALL_ITERS 0
#undef RABITQ_DISABLE_ON_STALL_ITERS_LOW_BEAM
#define RABITQ_DISABLE_ON_STALL_ITERS_LOW_BEAM 0
#undef RABITQ_DISABLE_ON_STALL_LOW_BEAM_MAX
#define RABITQ_DISABLE_ON_STALL_LOW_BEAM_MAX 0
#undef RABITQ_DISABLE_IF_NO_PRUNE_ITERS
#define RABITQ_DISABLE_IF_NO_PRUNE_ITERS 0
#undef RABITQ_REQUIRE_STALL
#define RABITQ_REQUIRE_STALL 0
#undef RABITQ_REQUIRE_FRONTIER_FULL
#define RABITQ_REQUIRE_FRONTIER_FULL 0
#undef RABITQ_DEFER_PRUNE_UNTIL_FRONTIER_FULL
#define RABITQ_DEFER_PRUNE_UNTIL_FRONTIER_FULL 1
#undef RABITQ_WARMUP_ITERS
#define RABITQ_WARMUP_ITERS 0
#undef RABITQ_COOL_RATIO
#define RABITQ_COOL_RATIO 0.0f
#undef RABITQ_ACTIVE_ITERS
#define RABITQ_ACTIVE_ITERS 0
#undef RABITQ_SCORE_MODE
#define RABITQ_SCORE_MODE 0
#undef RABITQ_SCORE_WORDS
#define RABITQ_SCORE_WORDS REFLOW_SCORE_WORDS
#undef RABITQ_MAX_WORDS
#define RABITQ_MAX_WORDS REFLOW_SCORE_WORDS
#undef RABITQ_USE_FACTORS
#define RABITQ_USE_FACTORS REFLOW_USE_FACTORS
#undef RABITQ_FACTOR_LIGHT_PRESELECT
#define RABITQ_FACTOR_LIGHT_PRESELECT REFLOW_FACTOR_LIGHT_PRESELECT
#undef RABITQ_FACTOR_PRESELECT_EXTRA
#define RABITQ_FACTOR_PRESELECT_EXTRA REFLOW_FACTOR_PRESELECT_EXTRA
#undef RABITQ_USE_PACKED_DQVQ
#define RABITQ_USE_PACKED_DQVQ \
    ((REFLOW_USE_FACTORS) ? (REFLOW_FACTOR_PACKED_DQVQ) : 0)
#undef RABITQ_USE_FP16_DQVQ
#define RABITQ_USE_FP16_DQVQ \
    ((REFLOW_USE_FACTORS) ? (REFLOW_FACTOR_FP16_DQVQ) : 0)
#undef RABITQ_QUERY_PROJ_ENABLE
#define RABITQ_QUERY_PROJ_ENABLE REFLOW_QUERY_PROJ_ENABLE
#undef RABITQ_PACKED_CODE_ENABLE
#define RABITQ_PACKED_CODE_ENABLE REFLOW_PACKED_CODE_ENABLE
#undef RABITQ_FASTSCAN_ENABLE
#define RABITQ_FASTSCAN_ENABLE REFLOW_FASTSCAN_ENABLE
#undef RABITQ_SCORE1_PACKED_FASTSCAN_ENABLE
#define RABITQ_SCORE1_PACKED_FASTSCAN_ENABLE REFLOW_FASTSCAN_ENABLE
#undef RABITQ_TWO_PHASE_EXACT
#define RABITQ_TWO_PHASE_EXACT 0
#undef RABITQ_GLOBAL_APPROX_PRESELECT
#define RABITQ_GLOBAL_APPROX_PRESELECT 0
#undef RABITQ_SELECT_MODE
#define RABITQ_SELECT_MODE 0
#undef RABITQ_SELECT_TIE_CAP_ENABLE
#define RABITQ_SELECT_TIE_CAP_ENABLE 0
#undef RABITQ_SELECT_TIE_CAP_EXTRA
#define RABITQ_SELECT_TIE_CAP_EXTRA 0
#undef RABITQ_RAW_KEEP
#define RABITQ_RAW_KEEP 0
#undef RABITQ_RAW_KEEP_LOW
#define RABITQ_RAW_KEEP_LOW 0
#undef RABITQ_RAW_KEEP_HIGH
#define RABITQ_RAW_KEEP_HIGH 0
#undef RABITQ_RAW_KEEP_SPLIT_BEAM
#define RABITQ_RAW_KEEP_SPLIT_BEAM 0
#undef RABITQ_INT_KEY_FASTPATH
#define RABITQ_INT_KEY_FASTPATH 1
#undef RABITQ_INT_KEY_ALLOW_FASTSCAN
#define RABITQ_INT_KEY_ALLOW_FASTSCAN 1
#undef HASH_WARP_DEDUP
#define HASH_WARP_DEDUP 0
#undef SKIP_ATOMIC_WHEN_POOL_FULL
#if REFLOW_SKIP_ATOMIC_POLICY == 0
#define SKIP_ATOMIC_WHEN_POOL_FULL 0
#elif REFLOW_SKIP_ATOMIC_POLICY == 1
#define SKIP_ATOMIC_WHEN_POOL_FULL 1
#else
#if BEAM_SIZE >= 400
#define SKIP_ATOMIC_WHEN_POOL_FULL 0
#else
#define SKIP_ATOMIC_WHEN_POOL_FULL 1
#endif
#endif
#undef RABITQ_GPU_DUALPOOL_ENGINE
#define RABITQ_GPU_DUALPOOL_ENGINE REFLOW_GPU_DUALPOOL_ENGINE
#undef RABITQ_GPU_DUALPOOL_POP_BATCH
#define RABITQ_GPU_DUALPOOL_POP_BATCH REFLOW_POP_BATCH
#undef RABITQ_GPU_DUALPOOL_ANCHOR_EXACT
#define RABITQ_GPU_DUALPOOL_ANCHOR_EXACT REFLOW_ANCHOR_EXACT
#undef RABITQ_CPU_PARITY_EXACT_POOL
#define RABITQ_CPU_PARITY_EXACT_POOL 0
#undef RABITQ_CPU_ALIGN_EXACT_EXPANDER_SQRY
#define RABITQ_CPU_ALIGN_EXACT_EXPANDER_SQRY 0
#endif
#if RABITQ_GPU_DUALPOOL_ENGINE
// Unified dual-pool mode: lock to CPU-like semantics.
#undef RABITQ_CPU_ALIGN_FLOW
#define RABITQ_CPU_ALIGN_FLOW 1
#undef RABITQ_CPU_PARITY_MODE
#define RABITQ_CPU_PARITY_MODE 1
#undef RABITQ_CPU_PARITY_POP_BATCH
#define RABITQ_CPU_PARITY_POP_BATCH \
    ((RABITQ_GPU_DUALPOOL_POP_BATCH > 0) ? (RABITQ_GPU_DUALPOOL_POP_BATCH) : 1)
#undef RABITQ_CPU_PARITY_FORCE_EXPAND_WIDTH
#define RABITQ_CPU_PARITY_FORCE_EXPAND_WIDTH 1
#undef RABITQ_CPU_PARITY_USE_EXACT_CUTOFF
#define RABITQ_CPU_PARITY_USE_EXACT_CUTOFF 0
#undef RABITQ_CPU_PARITY_REALTIME_MERGE
#define RABITQ_CPU_PARITY_REALTIME_MERGE 0
#undef RABITQ_CPU_PARITY_EXACT_POOL
#define RABITQ_CPU_PARITY_EXACT_POOL 0
#undef RABITQ_CPU_ALIGN_EXACT_EXPANDER_SQRY
#define RABITQ_CPU_ALIGN_EXACT_EXPANDER_SQRY 0
#undef RABITQ_TWO_PHASE_EXACT
#define RABITQ_TWO_PHASE_EXACT 0
#undef RABITQ_GLOBAL_APPROX_PRESELECT
#define RABITQ_GLOBAL_APPROX_PRESELECT 0
#undef RABITQ_DISABLE_IF_NO_PRUNE_ITERS
#define RABITQ_DISABLE_IF_NO_PRUNE_ITERS 0
#undef RABITQ_REQUIRE_STALL
#define RABITQ_REQUIRE_STALL 0
#undef RABITQ_DISABLE_ON_STALL_ITERS
#define RABITQ_DISABLE_ON_STALL_ITERS 0
#undef RABITQ_DISABLE_ON_STALL_ITERS_LOW_BEAM
#define RABITQ_DISABLE_ON_STALL_ITERS_LOW_BEAM 0
#undef RABITQ_DISABLE_ON_STALL_LOW_BEAM_MAX
#define RABITQ_DISABLE_ON_STALL_LOW_BEAM_MAX 0
#endif
#ifndef RABITQ_APPROX_CUTOFF_RATIO
// <=0: disabled; >0: when frontier full, prune candidate if approx > cutoff * ratio.
#define RABITQ_APPROX_CUTOFF_RATIO 0.0f
#endif
#ifndef RABITQ_SCORE_MODE
// 0: popcount-based sign score (legacy)
// 1: quantized-query LUT weighted sign score (CPU RaBitQ-inspired)
#define RABITQ_SCORE_MODE 0
#endif
#ifndef RABITQ_FASTSCAN_ENABLE
// 0: disabled
// 1: enable LUT-accumulate fastscan-style scoring on mode0/no-factor path.
#define RABITQ_FASTSCAN_ENABLE 0
#endif
#ifndef RABITQ_SCORE1_PACKED_FASTSCAN_ENABLE
// 1: allow packed-code fastscan path in score_mode=1 (with factors).
// This is used to align GPU path with CPU batch-32 packed accumulation.
#define RABITQ_SCORE1_PACKED_FASTSCAN_ENABLE 0
#endif
#ifndef RABITQ_PACKED_CODE_ENABLE
// 1: use prepacked RaBitQ code blocks loaded from *.rabitq.bin (block0) for
// score_mode=0 fastscan scoring path (dim=128, degree=32 specialized path).
// 0: keep sign-word based fastscan path.
#define RABITQ_PACKED_CODE_ENABLE 0
#endif
#ifndef RABITQ_PACKED_PAYLOAD_INDEX
// Which [n,dim]+u32 payload block to use in *.rabitq.bin for packed fastscan.
// 0: first payload (legacy); 1: second payload when present.
#define RABITQ_PACKED_PAYLOAD_INDEX 1
#endif
#ifndef RABITQ_QUERY_PROJ_ENABLE
// 1: apply query-side projection from *.rabitq.proj.bin before q6 LUT build
// on fastscan paths (keeps exact-distance query unchanged).
#define RABITQ_QUERY_PROJ_ENABLE 0
#endif
#ifndef RABITQ_QUERY_PROJ_STEPS
// Number of sign-diagonal + FWHT steps; 3 matches current proj sidecar.
#define RABITQ_QUERY_PROJ_STEPS 3
#endif
#ifndef RABITQ_QUERY_PROJ_MODE
// 0: repeated D(step)*FWHT for each step
// 1: single diagonal only (row0), no FWHT
// 2: single diagonal + one FWHT (row0)
#define RABITQ_QUERY_PROJ_MODE 0
#endif
#ifndef RABITQ_SCORE_ALPHA
#define RABITQ_SCORE_ALPHA 1.0f
#endif
#ifndef RABITQ_SCORE_WORDS
// 0: use all words; >0: use only first N 32-bit words for RabitQ score (cheaper but coarser).
#define RABITQ_SCORE_WORDS 0
#endif
#ifndef RABITQ_SCORE_WORDS_FALLBACK
// Fallback when RABITQ_SCORE_WORDS is left as 0 (auto).
// Keep a stable/lightweight default; explicit RABITQ_SCORE_WORDS still overrides.
#define RABITQ_SCORE_WORDS_FALLBACK 1
#endif
#ifndef RABITQ_QWORDS_SHARED_WORDS
// Shared-memory footprint control for query sign words used by RabitQ scoring.
// Keep mode1 unchanged (needs full words), and shrink mode0 to configured score words.
#if RABITQ_SCORE_MODE == 1
#define RABITQ_QWORDS_SHARED_WORDS RABITQ_MAX_WORDS
#elif RABITQ_USE_FACTORS
#define RABITQ_QWORDS_SHARED_WORDS RABITQ_MAX_WORDS
#elif RABITQ_SCORE_WORDS > 0
#define RABITQ_QWORDS_SHARED_WORDS RABITQ_SCORE_WORDS
#else
#define RABITQ_QWORDS_SHARED_WORDS RABITQ_SCORE_WORDS_FALLBACK
#endif
#endif
#if RABITQ_QWORDS_SHARED_WORDS < 1
#undef RABITQ_QWORDS_SHARED_WORDS
#define RABITQ_QWORDS_SHARED_WORDS 1
#endif
#if RABITQ_QWORDS_SHARED_WORDS > RABITQ_MAX_WORDS
#undef RABITQ_QWORDS_SHARED_WORDS
#define RABITQ_QWORDS_SHARED_WORDS RABITQ_MAX_WORDS
#endif
#ifndef RABITQ_Q6_LUT_WORDS
// Shared-memory footprint control for q6 LUT.
// For fastscan mode0 we only need score_words, not full RABITQ_MAX_WORDS.
#if RABITQ_SCORE_MODE == 1
#define RABITQ_Q6_LUT_WORDS RABITQ_MAX_WORDS
#elif RABITQ_FASTSCAN_ENABLE
#if RABITQ_SCORE_WORDS > 0
#define RABITQ_Q6_LUT_WORDS RABITQ_SCORE_WORDS
#else
#define RABITQ_Q6_LUT_WORDS RABITQ_SCORE_WORDS_FALLBACK
#endif
#else
#define RABITQ_Q6_LUT_WORDS 1
#endif
#endif
#if RABITQ_Q6_LUT_WORDS < 1
#undef RABITQ_Q6_LUT_WORDS
#define RABITQ_Q6_LUT_WORDS 1
#endif
#if RABITQ_Q6_LUT_WORDS > RABITQ_MAX_WORDS
#undef RABITQ_Q6_LUT_WORDS
#define RABITQ_Q6_LUT_WORDS RABITQ_MAX_WORDS
#endif
#ifndef RABITQ_PACK_LUT_ACTIVE
#if RABITQ_PACKED_CODE_ENABLE && \
    (((RABITQ_FASTSCAN_ENABLE != 0)) || \
     ((RABITQ_SCORE_MODE == 1) && (RABITQ_SCORE1_PACKED_FASTSCAN_ENABLE != 0)))
#define RABITQ_PACK_LUT_ACTIVE 1
#else
#define RABITQ_PACK_LUT_ACTIVE 0
#endif
#endif
#ifndef RABITQ_EDGE_TILE_CACHE_WORDS
// 0: disabled
// >0: cache first N sign words for current 32-neighbor tile into shared memory.
#define RABITQ_EDGE_TILE_CACHE_WORDS 0
#endif
#if RABITQ_EDGE_TILE_CACHE_WORDS < 0
#undef RABITQ_EDGE_TILE_CACHE_WORDS
#define RABITQ_EDGE_TILE_CACHE_WORDS 0
#endif
#if RABITQ_EDGE_TILE_CACHE_WORDS > RABITQ_MAX_WORDS
#undef RABITQ_EDGE_TILE_CACHE_WORDS
#define RABITQ_EDGE_TILE_CACHE_WORDS RABITQ_MAX_WORDS
#endif
#ifndef RABITQ_REQUIRE_STALL
// 0: always allow RabitQ in active window; 1: only allow when best-distance stalls.
#define RABITQ_REQUIRE_STALL 0
#endif
#ifndef RABITQ_STALL_TRIGGER
#define RABITQ_STALL_TRIGGER 2
#endif
#ifndef RABITQ_STALL_IMPROVE_EPS_ABS
#define RABITQ_STALL_IMPROVE_EPS_ABS 1e-4f
#endif
#ifndef RABITQ_STALL_IMPROVE_EPS_REL
#define RABITQ_STALL_IMPROVE_EPS_REL 1e-4f
#endif
#ifndef RABITQ_EARLY_STOP_STALL_ITERS
#define RABITQ_EARLY_STOP_STALL_ITERS 64
#endif
#ifndef RABITQ_EARLY_STOP_WARMUP_ITERS
#define RABITQ_EARLY_STOP_WARMUP_ITERS 8
#endif
#ifndef RABITQ_EARLY_STOP_REQUIRE_FULL
// 1: only allow RabitQ early-stop after frontier is full.
// This avoids stopping too early when top-k frontier is still forming.
#define RABITQ_EARLY_STOP_REQUIRE_FULL 1
#endif
#ifndef RABITQ_STALL_TRACK_MODE
// 0: stall is tracked by best distance frontier[0] (legacy).
// 1: stall is tracked by top-k boundary distance frontier[min(K, nf)-1],
//    which is more aligned with Recall@K behavior.
#define RABITQ_STALL_TRACK_MODE 1
#endif
#ifndef RABITQ_MAX_ITER_CAP
// Hard cap for RabitQ query loop to avoid pathological long-tail blocks.
#define RABITQ_MAX_ITER_CAP 256
#endif
#ifndef RABITQ_SELECT_MODE
// 0: exact warp top-k
// 1: fast threshold select (mode0/no-factor path only), fallback to exact when too few kept
#define RABITQ_SELECT_MODE 0
#endif
#ifndef RABITQ_MATCH_SLACK
// In fast mode, keep candidates with score >= (best_score - slack).
#define RABITQ_MATCH_SLACK 1
#endif
#ifndef RABITQ_FAST_EXTRA_KEEP
// In fast mode, allow up to topn + extra before downsampling.
#define RABITQ_FAST_EXTRA_KEEP 6
#endif
#ifndef RABITQ_FAST_DOWNSAMPLE_MODE
// 0: stride/mod downsample (legacy)
// 1: keep first target_keep lanes by fast-mask prefix rank (cheaper)
#define RABITQ_FAST_DOWNSAMPLE_MODE 0
#endif
#ifndef RABITQ_SELECT_TIE_CAP_ENABLE
// 1: after RabitQ top-k selection, cap excessive tie-overflow by light downsample.
// 0: keep legacy mask behavior.
#define RABITQ_SELECT_TIE_CAP_ENABLE 0
#endif
#ifndef RABITQ_SELECT_TIE_CAP_EXTRA
// cap_keep = topn + extra when tie-cap is enabled.
#define RABITQ_SELECT_TIE_CAP_EXTRA RABITQ_FAST_EXTRA_KEEP
#endif
#ifndef RABITQ_SELECT_EARLY_STOP_ON_FULL
// 1: in RabitQ selector, stop picking lower-score levels once selected >= topk.
// Keeps high-score tie groups while avoiding extra lower-level iterations.
#define RABITQ_SELECT_EARLY_STOP_ON_FULL 0
#endif
#ifndef RABITQ_REQUIRE_FRONTIER_FULL
// 1: enable RabitQ filter only when frontier is full (late-stage gating).
#define RABITQ_REQUIRE_FRONTIER_FULL 0
#endif
#ifndef RABITQ_DEFER_PRUNE_UNTIL_FRONTIER_FULL
// 1: defer RabitQ tile pruning until frontier is full.
// This keeps early exploration exact-driven and reduces recall loss from early
// approximation errors.
#define RABITQ_DEFER_PRUNE_UNTIL_FRONTIER_FULL 1
#endif
#ifndef RABITQ_INT_KEY_FASTPATH
// 1: when mode0/no-factor and no cutoff/recheck, rank by integer score key without float approx.
// 0: keep legacy float-approx key path.
#define RABITQ_INT_KEY_FASTPATH 0
#endif
#ifndef RABITQ_INT_KEY_ALLOW_FASTSCAN
// 1: allow integer-key fastpath even when FASTSCAN scoring path is enabled.
// This is rank-equivalent for selector ordering when cutoff is disabled.
#define RABITQ_INT_KEY_ALLOW_FASTSCAN 0
#endif
#ifndef RABITQ_WARP_L1_DEDUP_ENABLE
// 1: enable per-warp local hash pre-dedup in RabitQ path.
// 0: disable RabitQ local pre-dedup to save shared memory / improve occupancy.
#define RABITQ_WARP_L1_DEDUP_ENABLE 0
#endif
#ifndef RABITQ_INT_KEY_TIEBREAK
// 1: add lane-id tiebreak to integer key (more deterministic, more aggressive).
// 0: keep tie groups (closer to legacy float-key behavior).
#define RABITQ_INT_KEY_TIEBREAK 0
#endif
#ifndef RABITQ_INT_KEY_RESPECT_POS_CLAMP
// 1: mirror legacy approx path guard: if approx <= 0, treat as worst key.
#define RABITQ_INT_KEY_RESPECT_POS_CLAMP 1
#endif
#ifndef RABITQ_USE_FACTORS
// 1: use precomputed factor tables (triple/factor_dq/factor_vq)
// 0: use sign-dot-only score (no factor table dependency)
#define RABITQ_USE_FACTORS 0
#endif
#ifndef RABITQ_USE_PACKED_DQVQ
// 1: use packed float2 table for (factor_dq, factor_vq) to reduce factor loads.
// 0: use separate factor_dq/factor_vq arrays.
#define RABITQ_USE_PACKED_DQVQ 0
#endif
#ifndef RABITQ_USE_FP16_DQVQ
// 1: with packed mode, store (dq,vq) as half2 to reduce bandwidth.
// 0: packed mode uses float2.
#define RABITQ_USE_FP16_DQVQ 0
#endif
#ifndef RABITQ_FACTOR_PRESELECT_EXTRA
// In factor mode (score_mode=0), preselect top-(topn+extra) by cheap sign score,
// then compute factor-based approx only for this shortlist.
#define RABITQ_FACTOR_PRESELECT_EXTRA 8
#endif
#ifndef RABITQ_FACTOR_LIGHT_PRESELECT
// 1: in factor mode, run a cheap sign-popcount preselect first, and execute
// heavy factor/q6 scoring only for shortlisted lanes.
#define RABITQ_FACTOR_LIGHT_PRESELECT 0
#endif
#ifndef RABITQ_FACTOR_LIGHT_PRESELECT_WORDS
// Number of sign words used by lightweight factor preselect.
#define RABITQ_FACTOR_LIGHT_PRESELECT_WORDS 2
#endif
#ifndef RABITQ_WARP_SELECT_SORT
// 1: use warp bitonic-sort based top-k selector in RabitQ paths.
// 0: use iterative reduce-based selector.
#define RABITQ_WARP_SELECT_SORT 1
#endif
#ifndef RABITQ_WARP_SELECT_SORT_MIN
// Minimum topk to enable sort-based selector.
#define RABITQ_WARP_SELECT_SORT_MIN 8
#endif
#ifndef RABITQ_SORT_TIE_INCLUSIVE
// 1: keep all lanes with key >= k-th key after sort (tie-inclusive)
// 0: keep exactly top-k lanes after sort
#define RABITQ_SORT_TIE_INCLUSIVE 1
#endif
#ifndef RABITQ_WORD1_FASTPATH
// 1: specialized popcount path when score uses exactly one sign-word.
#define RABITQ_WORD1_FASTPATH 0
#endif
#ifndef RABITQ_STRICT_TOPK_TIEBREAK
// 1: in RabitQ selector, break score ties strictly to keep exactly topk lanes.
// 0: keep legacy "select all equal-max lanes" behavior.
#define RABITQ_STRICT_TOPK_TIEBREAK 0
#endif
#ifndef RABITQ_SKIP_NEAR_FULL_REMAIN
// If remaining neighbor slots are below this threshold, skip RabitQ
// score/select work and fall back to normal keep path.
// Set <=0 to disable this implementation shortcut.
#define RABITQ_SKIP_NEAR_FULL_REMAIN 0
#endif
#ifndef RABITQ_COARSE_PREFILTER
// 1: in nofactor sign-score mode, prefilter by cheap coarse words and
// compute full sign score only for a shortlist.
#define RABITQ_COARSE_PREFILTER 0
#endif
#ifndef RABITQ_COARSE_PREFILTER_MODE
// 1: top-k coarse shortlist (more precise, higher selector overhead)
// 2: threshold shortlist by best coarse score (lower selector overhead)
#define RABITQ_COARSE_PREFILTER_MODE 1
#endif
#ifndef RABITQ_COARSE_WORDS
// Number of 32-bit words used in coarse prefilter stage.
#define RABITQ_COARSE_WORDS 2
#endif
#ifndef RABITQ_COARSE_EXTRA_KEEP
// coarse shortlist size = topn + extra_keep
#define RABITQ_COARSE_EXTRA_KEEP 8
#endif
#ifndef RABITQ_COARSE_MATCH_SLACK0
// threshold mode: keep if coarse_score >= best_coarse_score - slack0
#define RABITQ_COARSE_MATCH_SLACK0 8
#endif
#if ENABLE_RABITQ && (BEAM_SIZE >= RABITQ_MIN_BEAM) && \
    ((RABITQ_MAX_BEAM <= 0) || (BEAM_SIZE <= RABITQ_MAX_BEAM))
#define RABITQ_KERNEL_ACTIVE 1
#else
#define RABITQ_KERNEL_ACTIVE 0
#endif
#define WARP_L1_DEDUP_FEATURE_ACTIVE \
    (ENABLE_DGS || SQG_KERNEL_ACTIVE || \
     (RABITQ_KERNEL_ACTIVE && RABITQ_WARP_L1_DEDUP_ENABLE))
#if RABITQ_TWO_PHASE_EXACT || RABITQ_GLOBAL_APPROX_PRESELECT || RABITQ_GPU_DUALPOOL_ENGINE
#define RABITQ_NEIGHBOR_APPROX_BUFFER 1
#else
#define RABITQ_NEIGHBOR_APPROX_BUFFER 0
#endif
#ifndef RABITQ_REUSE_CAND_TMP_AS_APPROX
// 1: reuse strict-mode candidate tmp distance buffer as RabitQ neighbor approx
// storage to reduce per-block shared memory pressure on large beams.
#if RABITQ_REFLOW_FIXED
#define RABITQ_REUSE_CAND_TMP_AS_APPROX 1
#else
#define RABITQ_REUSE_CAND_TMP_AS_APPROX 0
#endif
#endif
#ifndef RABITQ_EST_ITERS
#define RABITQ_EST_ITERS (BEAM_SIZE / SEARCH_WIDTH)
#endif
#ifndef ENABLE_SQG
#define ENABLE_SQG 0
#endif
#ifndef SQG_TOPN
#define SQG_TOPN 8
#endif
#ifndef SQG_SAMPLE_DIMS
// Number of sampled dimensions for SQG approximate neighbor scoring.
#define SQG_SAMPLE_DIMS 128
#endif
#ifndef SQG_USE_FHT
// 0: use original vector space for SQG sign coding
// 1: apply FWHT (dim=128 path) before SQG sign coding
#define SQG_USE_FHT 0
#endif
#ifndef SQG_SIGNBIT_MODE
// 0: raw sign-bit (x >= 0)
// 1: mean-centered sign-bit (x >= mean_dim)
#define SQG_SIGNBIT_MODE 1
#endif
#ifndef SQG_MAX_WORDS
#define SQG_MAX_WORDS 32
#endif
#ifndef SQG_HAMMING_FAST4
// 1: specialized hamming path for <=4 words (common for SQG_SAMPLE_DIMS<=128)
#define SQG_HAMMING_FAST4 1
#endif
#ifndef SQG_RAW_KEEP
// Keep additional uniformly-sampled raw neighbors (by lane index) to reduce false negatives.
#define SQG_RAW_KEEP 0
#endif
#ifndef SQG_TOPN_EARLY
#define SQG_TOPN_EARLY SQG_TOPN
#endif
#ifndef SQG_TOPN_EARLY_ITERS
#define SQG_TOPN_EARLY_ITERS 0
#endif
#ifndef SQG_FULLSCAN_STRIDE
// 0 disables periodic full-scan fallback; >0 disables SQG filter every N iterations.
#define SQG_FULLSCAN_STRIDE 0
#endif
#ifndef SQG_SCORE_MODE
// 0: sign-bit popcount score (default)
// 1: SQ8 sampled-L2 score (dim=128 only, requires ENABLE_SQ8)
// 2: sign-bit prefilter + SQ8 sampled-L2 refine
#define SQG_SCORE_MODE 0
#endif
#ifndef SQG_SQ8_STRIDE
// For SQ8 sampled score: use dims 0, stride, 2*stride, ...
#define SQG_SQ8_STRIDE 4
#endif
#ifndef SQG_REFINE_TOPM
// In score mode 2, first keep top-M by sign score, then refine by SQ8.
#define SQG_REFINE_TOPM 12
#endif
#ifndef SQG_SELECT_MODE
// 0: exact warp top-k (baseline)
// 1: fast threshold-select with bounded keep count
// 2: pure-fast threshold-select (no exact top-k fallback)
#define SQG_SELECT_MODE 0
#endif
#ifndef SQG_MATCH_SLACK
// In fast mode, keep entries with (best_match - match) <= slack.
#define SQG_MATCH_SLACK 1
#endif
#ifndef SQG_FAST_EXTRA_KEEP
// In fast mode, allow up to topn + extra candidates before falling back to exact top-k.
#define SQG_FAST_EXTRA_KEEP 6
#endif
#ifndef SQG_SELECT_OVERFLOW_DOWNSAMPLE
// In fast select mode, when approx keep count is too large, downsample first to avoid exact top-k fallback.
#define SQG_SELECT_OVERFLOW_DOWNSAMPLE 1
#endif
#ifndef SQG_COARSE_PREFILTER
// 0: disabled (baseline)
// 1: use a cheap coarse sign filter (word0) to reduce full hamming work.
#define SQG_COARSE_PREFILTER 0
#endif
#ifndef SQG_COARSE_PREFILTER_MODE
// 1: top-k coarse keep (higher filter precision, higher overhead)
// 2: threshold coarse keep by best word0 match (lower overhead)
#define SQG_COARSE_PREFILTER_MODE 1
#endif
#ifndef SQG_COARSE_EXTRA_KEEP
// coarse_keep = min(deg, topn + SQG_COARSE_EXTRA_KEEP)
#define SQG_COARSE_EXTRA_KEEP 12
#endif
#ifndef SQG_COARSE_MIN_WORDS
// Only enable coarse filter when sign words are large enough.
#define SQG_COARSE_MIN_WORDS 3
#endif
#ifndef SQG_COARSE_MATCH_SLACK0
// coarse threshold mode: keep if word0_match >= best_word0_match - slack0
#define SQG_COARSE_MATCH_SLACK0 8
#endif
#ifndef SQG_COARSE_WORDS
// Number of leading sign words used by coarse prefilter (clamped by sqg_words).
#define SQG_COARSE_WORDS 1
#endif
#ifndef SQG_WARMUP_ITERS
#define SQG_WARMUP_ITERS 0
#endif
#ifndef SQG_ADAPT_EXPAND_ENABLE
// 1: when SQG gate is active, shrink expand width to reduce expansion workload.
#define SQG_ADAPT_EXPAND_ENABLE 0
#endif
#ifndef SQG_ADAPT_EXPAND_WIDTH
#define SQG_ADAPT_EXPAND_WIDTH SEARCH_WIDTH
#endif
#ifndef SQG_ADAPT_EXPAND_MIN_FRONTIER
// Apply adaptive expand only when frontier size >= this threshold (0 disables thresholding).
#define SQG_ADAPT_EXPAND_MIN_FRONTIER 0
#endif
#ifndef SQG_NEIGHBOR_BUDGET_ENABLE
// 1: when SQG gate is active, cap staged neighbor count per iteration.
#define SQG_NEIGHBOR_BUDGET_ENABLE 0
#endif
#ifndef SQG_NEIGHBOR_BUDGET
// Effective cap is min(NEIGHBOR_BUFFER_SIZE, SQG_NEIGHBOR_BUDGET).
#define SQG_NEIGHBOR_BUDGET NEIGHBOR_BUFFER_SIZE
#endif
#ifndef SQG_NEIGHBOR_BUDGET_PRESAMPLE
// 1: pre-downsample keep-mask by remaining budget before hashtable_insert.
#define SQG_NEIGHBOR_BUDGET_PRESAMPLE 1
#endif
#ifndef SQG_COOL_RATIO
#define SQG_COOL_RATIO 0.0f
#endif
#ifndef SQG_ACTIVE_ITERS
// 0: use SQG_COOL_RATIO to derive active window
// >0: directly set SQG active window iterations
#define SQG_ACTIVE_ITERS 0
#endif
#ifndef SQG_EARLY_STOP_ENABLE
// 1: allow SQG path to stop query iterations early when best distance stalls.
#define SQG_EARLY_STOP_ENABLE 0
#endif
#ifndef SQG_EARLY_STOP_STALL_ITERS
#define SQG_EARLY_STOP_STALL_ITERS 4
#endif
#ifndef SQG_EARLY_STOP_WARMUP_ITERS
#define SQG_EARLY_STOP_WARMUP_ITERS 8
#endif
#ifndef SQG_EARLY_STOP_REQUIRE_FULL
// 1: early-stop only when frontier is already full.
#define SQG_EARLY_STOP_REQUIRE_FULL 1
#endif
#ifndef SQG_STALL_IMPROVE_EPS_ABS
#define SQG_STALL_IMPROVE_EPS_ABS 1e-6f
#endif
#ifndef SQG_STALL_IMPROVE_EPS_REL
#define SQG_STALL_IMPROVE_EPS_REL 1e-4f
#endif
#ifndef SQG_MIN_BEAM
// Disable SQG automatically when compile-time beam is too small.
#define SQG_MIN_BEAM 0
#endif
#if ENABLE_SQG && (BEAM_SIZE >= SQG_MIN_BEAM)
#define SQG_KERNEL_ACTIVE 1
#else
#define SQG_KERNEL_ACTIVE 0
#endif
#ifndef SQG_EST_ITERS
#define SQG_EST_ITERS (BEAM_SIZE / SEARCH_WIDTH)
#endif
#ifndef DGS_BIND_WIDE_ACTIVE
#define DGS_BIND_WIDE_ACTIVE 1
#endif
#ifndef DGS_STALL_FALLBACK_ENABLE
#define DGS_STALL_FALLBACK_ENABLE 1
#endif
#ifndef DGS_STALL_TRIGGER
#define DGS_STALL_TRIGGER 4
#endif
#ifndef DGS_STALL_FALLBACK_ITERS
#define DGS_STALL_FALLBACK_ITERS 2
#endif
#ifndef DGS_STALL_IMPROVE_EPS_ABS
#define DGS_STALL_IMPROVE_EPS_ABS 1e-6f
#endif
#ifndef DGS_STALL_IMPROVE_EPS_REL
#define DGS_STALL_IMPROVE_EPS_REL 1e-4f
#endif
#ifndef ENABLE_FRONTIER_DEDUP
#define ENABLE_FRONTIER_DEDUP 0
#endif
#ifndef FRONTIER_DEDUP_BITLEN
#define FRONTIER_DEDUP_BITLEN 10
#endif
#ifndef FRONTIER_DEDUP_HASH_SIZE
#define FRONTIER_DEDUP_HASH_SIZE (1u << (FRONTIER_DEDUP_BITLEN))
#endif
#ifndef FRONTIER_DEDUP_INTERVAL
#define FRONTIER_DEDUP_INTERVAL 1
#endif
#ifndef FRONTIER_DEDUP_METHOD
#define FRONTIER_DEDUP_METHOD 1
#endif
#ifndef FRONTIER_DEDUP_ADAPTIVE
#define FRONTIER_DEDUP_ADAPTIVE 1
#endif
#ifndef FRONTIER_DEDUP_ADAPTIVE_MIN_SIZE
#define FRONTIER_DEDUP_ADAPTIVE_MIN_SIZE 96
#endif
#ifndef FRONTIER_DEDUP_ADAPTIVE_MIN_ADJ_DUP
#define FRONTIER_DEDUP_ADAPTIVE_MIN_ADJ_DUP 2
#endif
#ifndef FRONTIER_DEDUP_PARALLEL
#define FRONTIER_DEDUP_PARALLEL 1
#endif
#ifndef OUTPUT_TOPK_UNIQUE
#define OUTPUT_TOPK_UNIQUE 0
#endif
static __device__ __forceinline__ void warp_bitonic_sort_u32_asc(uint32_t& key, uint32_t& val) {
    const unsigned lane = threadIdx.x & 31;
    #pragma unroll
    for (uint32_t k = 2; k <= 32; k <<= 1) {
        #pragma unroll
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            const uint32_t other_key = __shfl_xor_sync(FULL_MASK, key, j);
            const uint32_t other_val = __shfl_xor_sync(FULL_MASK, val, j);
            const bool up = ((lane & k) == 0);       // sort direction for this stage
            const bool low = ((lane & j) == 0);      // lower lane in the pair
            if (up) {
                if (low) {
                    if (key > other_key) { key = other_key; val = other_val; }
                } else {
                    if (key < other_key) { key = other_key; val = other_val; }
                }
            } else {
                if (low) {
                    if (key < other_key) { key = other_key; val = other_val; }
                } else {
                    if (key > other_key) { key = other_key; val = other_val; }
                }
            }
        }
    }
}

static __device__ __forceinline__ void warp_min_reduce_f32_u32_lane(
    float& key,
    uint32_t& val,
    int& lane) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_key = __shfl_down_sync(FULL_MASK, key, offset);
        const uint32_t other_val = __shfl_down_sync(FULL_MASK, val, offset);
        const int other_lane = __shfl_down_sync(FULL_MASK, lane, offset);
        const bool better = (other_key < key) || ((other_key == key) && (other_lane < lane));
        if (better) {
            key = other_key;
            val = other_val;
            lane = other_lane;
        }
    }
}

static __device__ __forceinline__ void warp_min_reduce_f32_u32_lane_masked(
    float& key,
    uint32_t& val,
    int& lane,
    const uint32_t mask,
    const int width) {
    for (int offset = (width >> 1); offset > 0; offset >>= 1) {
        const float other_key = __shfl_down_sync(mask, key, offset);
        const uint32_t other_val = __shfl_down_sync(mask, val, offset);
        const int other_lane = __shfl_down_sync(mask, lane, offset);
        const bool better = (other_key < key) || ((other_key == key) && (other_lane < lane));
        if (better) {
            key = other_key;
            val = other_val;
            lane = other_lane;
        }
    }
}

static __device__ __forceinline__ void block_select_topk_from_candidate_tmp_exact(
    INDEX_T* candidate_tmp_idx,
    DISTANCE_T* candidate_tmp_dist,
    int valid_count,
    int num_warps,
    INDEX_T* candidate_idx,
    DISTANCE_T* candidate_dist,
    int* out_count) {
    const int tid = threadIdx.x;
    const int lane_id = tid & (WARP_SIZE - 1);
    const int warp_id = tid / WARP_SIZE;
    const int limit = min(valid_count, 256);

    const int n = warp_id * WARP_SIZE + lane_id;
    const uint32_t inf_key = __float_as_uint((float)FLT_MAX);
    uint32_t key = inf_key;
    uint32_t val = (uint32_t)MAX_INDEX;
    if (n < limit) {
        const INDEX_T idx = candidate_tmp_idx[n];
        if (idx != MAX_INDEX) {
            key = __float_as_uint((float)candidate_tmp_dist[n]);
            val = (uint32_t)idx;
        }
    }
    warp_bitonic_sort_u32_asc(key, val);
    if (n < 256) {
        candidate_tmp_idx[n] = (INDEX_T)val;
        candidate_tmp_dist[n] = (DISTANCE_T)__uint_as_float(key);
    }
    __syncthreads();

    if (warp_id == 0) {
        int merge_width = 1;
        while (merge_width < num_warps && merge_width < WARP_SIZE) merge_width <<= 1;
        const uint32_t merge_mask =
            (merge_width >= WARP_SIZE) ? FULL_MASK : ((1u << merge_width) - 1u);

        int my_pos = 0;
        int my_len = 0;
        if (lane_id < num_warps) {
            int remain = limit - lane_id * WARP_SIZE;
            if (remain > 0) {
                if (remain > WARP_SIZE) remain = WARP_SIZE;
                my_len = remain;
            }
        }

        if (lane_id < merge_width) {
            int nc = 0;
            for (; nc < CANDIDATE_BUFFER_SIZE; nc++) {
                float local_key = FLT_MAX;
                uint32_t local_idx = (uint32_t)MAX_INDEX;
                if (lane_id < num_warps && my_pos < my_len) {
                    const int src = lane_id * WARP_SIZE + my_pos;
                    local_idx = (uint32_t)candidate_tmp_idx[src];
                    if (local_idx != (uint32_t)MAX_INDEX) {
                        local_key = (float)candidate_tmp_dist[src];
                    }
                }

                int local_lane = lane_id;
                warp_min_reduce_f32_u32_lane_masked(
                    local_key, local_idx, local_lane, merge_mask, merge_width);

                const float best_key = __shfl_sync(merge_mask, local_key, 0);
                const uint32_t best_idx = __shfl_sync(merge_mask, local_idx, 0);
                const int best_lane = __shfl_sync(merge_mask, local_lane, 0);
                const bool done =
                    (best_key == FLT_MAX) ||
                    (best_idx == (uint32_t)MAX_INDEX) ||
                    (best_lane >= num_warps);
                if (done) break;

                if (lane_id == 0) {
                    candidate_idx[nc] = (INDEX_T)best_idx;
                    candidate_dist[nc] = (DISTANCE_T)best_key;
                }
                if (lane_id == best_lane) {
                    my_pos++;
                }
            }

            if (lane_id == 0) {
                *out_count = nc;
                for (int i = nc; i < CANDIDATE_BUFFER_SIZE; i++) {
                    candidate_idx[i] = MAX_INDEX;
                    candidate_dist[i] = FLT_MAX;
                }
            }
        }
    }

    __syncthreads();
}

template <int CAP>
static __device__ __forceinline__ void rabitq_parity_merge_exact_stream(
    INDEX_T* topk_idx,
    DISTANCE_T* topk_dist,
    int* topk_size_ptr,
    INDEX_T* stream_idx,
    DISTANCE_T* stream_dist,
    int* stream_size_ptr,
    int exact_cap,
    INDEX_T kIndexMsbMask) {
    int topk_size = *topk_size_ptr;
    int stream_size = *stream_size_ptr;
    if (exact_cap <= 0 || stream_size <= 0) {
        *stream_size_ptr = 0;
        return;
    }
    if (topk_size > exact_cap) topk_size = exact_cap;
    for (int si = 0; si < stream_size; ++si) {
        const INDEX_T sid = stream_idx[si] & ~kIndexMsbMask;
        if (sid == MAX_INDEX) continue;
        const DISTANCE_T sd = stream_dist[si];

        int found = -1;
        for (int i = 0; i < topk_size; ++i) {
            if ((topk_idx[i] & ~kIndexMsbMask) == sid) {
                found = i;
                break;
            }
        }

        if (found >= 0) {
            if (sd < topk_dist[found]) {
                topk_dist[found] = sd;
                INDEX_T key_i = topk_idx[found];
                DISTANCE_T key_d = topk_dist[found];
                int j = found - 1;
                while (j >= 0 && topk_dist[j] > key_d) {
                    topk_dist[j + 1] = topk_dist[j];
                    topk_idx[j + 1] = topk_idx[j];
                    --j;
                }
                topk_dist[j + 1] = key_d;
                topk_idx[j + 1] = key_i;
            }
            continue;
        }

        if (topk_size < exact_cap) {
            int pos = topk_size++;
            topk_idx[pos] = sid;
            topk_dist[pos] = sd;
            while (pos > 0 && topk_dist[pos] < topk_dist[pos - 1]) {
                const DISTANCE_T td = topk_dist[pos - 1];
                topk_dist[pos - 1] = topk_dist[pos];
                topk_dist[pos] = td;
                const INDEX_T ti = topk_idx[pos - 1];
                topk_idx[pos - 1] = topk_idx[pos];
                topk_idx[pos] = ti;
                --pos;
            }
        } else if (sd < topk_dist[exact_cap - 1]) {
            int pos = exact_cap - 1;
            topk_idx[pos] = sid;
            topk_dist[pos] = sd;
            while (pos > 0 && topk_dist[pos] < topk_dist[pos - 1]) {
                const DISTANCE_T td = topk_dist[pos - 1];
                topk_dist[pos - 1] = topk_dist[pos];
                topk_dist[pos] = td;
                const INDEX_T ti = topk_idx[pos - 1];
                topk_idx[pos - 1] = topk_idx[pos];
                topk_idx[pos] = ti;
                --pos;
            }
        }
    }

    *topk_size_ptr = topk_size;
    *stream_size_ptr = 0;
}

static __device__ __forceinline__ uint32_t warp_reduce_max_u32(uint32_t v) {
    v = max(v, __shfl_down_sync(FULL_MASK, v, 16));
    v = max(v, __shfl_down_sync(FULL_MASK, v, 8));
    v = max(v, __shfl_down_sync(FULL_MASK, v, 4));
    v = max(v, __shfl_down_sync(FULL_MASK, v, 2));
    v = max(v, __shfl_down_sync(FULL_MASK, v, 1));
    return __shfl_sync(FULL_MASK, v, 0);
}

static __device__ __forceinline__ uint32_t warp_reduce_min_u32(uint32_t v) {
    v = min(v, __shfl_down_sync(FULL_MASK, v, 16));
    v = min(v, __shfl_down_sync(FULL_MASK, v, 8));
    v = min(v, __shfl_down_sync(FULL_MASK, v, 4));
    v = min(v, __shfl_down_sync(FULL_MASK, v, 2));
    v = min(v, __shfl_down_sync(FULL_MASK, v, 1));
    return __shfl_sync(FULL_MASK, v, 0);
}

static __device__ __forceinline__ int warp_read_volatile_shared_i32(
    volatile int* ptr) {
    int v = 0;
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
        v = *ptr;
    }
    return __shfl_sync(FULL_MASK, v, 0);
}

template <typename T>
static __device__ __forceinline__ DISTANCE_T l2_distance_scalar_device(
    int dim, const T* __restrict__ a, const T* __restrict__ b) {
    DISTANCE_T sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const DISTANCE_T d = (DISTANCE_T)a[i] - (DISTANCE_T)b[i];
        sum = fmaf(d, d, sum);
    }
    return sum;
}

static __device__ __forceinline__ uint32_t warp_select_topk_mask_desc_u32(
    uint32_t key,
    bool valid,
    int topk) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    uint32_t remain = __ballot_sync(FULL_MASK, valid);
    if (topk <= 0 || remain == 0u) return 0u;

    const int pick = min(topk, __popc(remain));
    uint32_t selected = 0u;

    #pragma unroll
    for (int s = 0; s < 32; s++) {
        if (s >= pick) break;
        const bool in_pool = ((remain >> lane) & 1u) != 0u;
        const uint32_t local_key = in_pool ? key : 0u;
        const uint32_t best_key = warp_reduce_max_u32(local_key);
        const uint32_t winner = __ballot_sync(FULL_MASK, in_pool && (key == best_key));
        if (winner == 0u) break;
        selected |= winner;
        remain &= ~winner;
        if (remain == 0u) break;
    }

    return selected;
}

static __device__ __forceinline__ uint32_t warp_select_topk_mask_desc_u32_strict(
    uint32_t key,
    bool valid,
    int topk) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    uint32_t remain = __ballot_sync(FULL_MASK, valid);
    if (topk <= 0 || remain == 0u) return 0u;

    const int pick = min(topk, __popc(remain));
    uint32_t selected = 0u;

    #pragma unroll
    for (int s = 0; s < 32; s++) {
        if (s >= pick) break;
        const bool in_pool = ((remain >> lane) & 1u) != 0u;
        const uint32_t local_key = in_pool ? key : 0u;
        const uint32_t best_key = warp_reduce_max_u32(local_key);
        const uint32_t winner_all = __ballot_sync(FULL_MASK, in_pool && (key == best_key));
        if (winner_all == 0u) break;
        // Keep exactly one winner lane per step to avoid >topk on tie-heavy keys.
        const uint32_t winner_one = winner_all & (~winner_all + 1u);
        selected |= winner_one;
        remain &= ~winner_one;
        if (remain == 0u) break;
    }

    return selected;
}

static __device__ __forceinline__ uint32_t warp_select_topk_mask_desc_u32_sort(
    uint32_t key,
    bool valid,
    int topk) {
    const uint32_t valid_mask = __ballot_sync(FULL_MASK, valid);
    if (topk <= 0 || valid_mask == 0u) return 0u;

    const int valid_cnt = __popc(valid_mask);
    const int pick = min(max(topk, 0), valid_cnt);
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    uint32_t sort_key = valid ? key : 0u;
    uint32_t lane_id_key = (uint32_t)lane;
    warp_bitonic_sort_u32_asc(sort_key, lane_id_key);
#if RABITQ_SORT_TIE_INCLUSIVE
    // Keep all lanes whose key is at least the k-th key (tie-inclusive),
    // matching legacy selector behavior under tie-heavy score distributions.
    const int kth_lane = WARP_SIZE - pick;
    const uint32_t kth_key = __shfl_sync(FULL_MASK, sort_key, kth_lane);
    return __ballot_sync(FULL_MASK, valid && (key >= kth_key));
#else
    const bool keep_exact = (lane >= (WARP_SIZE - pick)) && valid;
    return __ballot_sync(FULL_MASK, keep_exact);
#endif
}

#ifndef RABITQ_WARP_SELECT_DROP_MIN_ENABLE
// 1: for high topk, use drop-min selector (remove only few worst lanes) instead
// of full 32-lane bitonic sort.
#define RABITQ_WARP_SELECT_DROP_MIN_ENABLE 0
#endif
#ifndef RABITQ_WARP_SELECT_DROP_MIN_TOPK
// Activate drop-min when topk is large enough that dropped count is small.
#define RABITQ_WARP_SELECT_DROP_MIN_TOPK 24
#endif

static __device__ __forceinline__ uint32_t warp_select_topk_mask_desc_u32_dropmin(
    uint32_t key,
    bool valid,
    int topk) {
    const uint32_t valid_mask = __ballot_sync(FULL_MASK, valid);
    if (topk <= 0 || valid_mask == 0u) return 0u;

    const int valid_cnt = __popc(valid_mask);
    const int pick = min(max(topk, 0), valid_cnt);
    if (pick >= valid_cnt) return valid_mask;

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int drop = valid_cnt - pick;
    uint32_t remain = valid_mask;

    #pragma unroll
    for (int s = 0; s < 32; ++s) {
        if (s >= drop) break;
        const bool in_pool = ((remain >> lane) & 1u) != 0u;
        const uint32_t local_key = in_pool ? key : 0xFFFFFFFFu;
        const uint32_t min_key = warp_reduce_min_u32(local_key);
        const uint32_t loser_all = __ballot_sync(FULL_MASK, in_pool && (key == min_key));
        if (loser_all == 0u) break;
        // Drop exactly one lane each step to preserve deterministic rank count.
        const uint32_t loser_one = loser_all & (~loser_all + 1u);
        remain &= ~loser_one;
        if (__popc(remain) <= pick) break;
    }

#if RABITQ_SORT_TIE_INCLUSIVE
    // Map back to tie-inclusive semantics: keep all lanes whose key is at least
    // the k-th key threshold.
    const uint32_t local_keep_key =
        (((remain >> lane) & 1u) != 0u) ? key : 0xFFFFFFFFu;
    const uint32_t kth_key = warp_reduce_min_u32(local_keep_key);
    return __ballot_sync(FULL_MASK, valid && (key >= kth_key));
#else
    return remain;
#endif
}

static __device__ __forceinline__ uint32_t warp_select_topk_mask_desc_u32_rabitq(
    uint32_t key,
    bool valid,
    int topk) {
#if RABITQ_WARP_SELECT_SORT
    if (topk >= (int)RABITQ_WARP_SELECT_SORT_MIN) {
#if RABITQ_WARP_SELECT_DROP_MIN_ENABLE
        if (topk >= (int)RABITQ_WARP_SELECT_DROP_MIN_TOPK) {
            return warp_select_topk_mask_desc_u32_dropmin(key, valid, topk);
        }
#endif
        return warp_select_topk_mask_desc_u32_sort(key, valid, topk);
    }
#endif
#if RABITQ_SELECT_EARLY_STOP_ON_FULL
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    uint32_t remain = __ballot_sync(FULL_MASK, valid);
    if (topk <= 0 || remain == 0u) return 0u;

    const int pick = min(topk, __popc(remain));
    uint32_t selected = 0u;

    #pragma unroll
    for (int s = 0; s < 32; s++) {
        if (s >= pick) break;
        const bool in_pool = ((remain >> lane) & 1u) != 0u;
        const uint32_t local_key = in_pool ? key : 0u;
        const uint32_t best_key = warp_reduce_max_u32(local_key);
        const uint32_t winner = __ballot_sync(FULL_MASK, in_pool && (key == best_key));
        if (winner == 0u) break;
        selected |= winner;
        remain &= ~winner;
        if (remain == 0u) break;
        if (__popc(selected) >= pick) break;
    }
    return selected;
#endif
#if RABITQ_STRICT_TOPK_TIEBREAK
    return warp_select_topk_mask_desc_u32_strict(key, valid, topk);
#else
    return warp_select_topk_mask_desc_u32(key, valid, topk);
#endif
}

#if SQG_KERNEL_ACTIVE
static __device__ __forceinline__ int sqg_hamming_diff_words(
    const uint32_t* __restrict__ v_bits,
    const uint32_t* __restrict__ q_bits,
    int sqg_words,
    int sqg_tail_bits,
    uint32_t sqg_tail_mask) {
    if (sqg_words <= 0) return 0;

    // Hot path for SIFT-like settings (SQG_SAMPLE_DIMS=128 -> 4 words).
    if (sqg_tail_bits == 0) {
        switch (sqg_words) {
            case 1:
                return __popc(v_bits[0] ^ q_bits[0]);
            case 2:
                return __popc(v_bits[0] ^ q_bits[0]) +
                       __popc(v_bits[1] ^ q_bits[1]);
            case 3:
                return __popc(v_bits[0] ^ q_bits[0]) +
                       __popc(v_bits[1] ^ q_bits[1]) +
                       __popc(v_bits[2] ^ q_bits[2]);
            case 4:
                return __popc(v_bits[0] ^ q_bits[0]) +
                       __popc(v_bits[1] ^ q_bits[1]) +
                       __popc(v_bits[2] ^ q_bits[2]) +
                       __popc(v_bits[3] ^ q_bits[3]);
            default:
                break;
        }
    }

    int diff_cnt = 0;
    for (int w = 0; w < sqg_words; ++w) {
        if ((w == sqg_words - 1) && (sqg_tail_bits != 0)) {
            const uint32_t vb = v_bits[w] & sqg_tail_mask;
            const uint32_t qb = q_bits[w] & sqg_tail_mask;
            diff_cnt += __popc(vb ^ qb);
        } else {
            diff_cnt += __popc(v_bits[w] ^ q_bits[w]);
        }
    }
    return diff_cnt;
}
#endif

#ifndef ENABLE_SQ8
#define ENABLE_SQ8 0
#endif
#ifndef SQ8_PRUNE_SLACK
#define SQ8_PRUNE_SLACK 1.0f
#endif
#ifndef SQ8_DIST_PRUNE_ENABLE
// 1: enable SQ8-based distance-stage pruning when frontier is full (legacy behavior)
// 0: disable distance-stage SQ8 pruning; SQ8 can still be used by SQG scoring paths.
#define SQ8_DIST_PRUNE_ENABLE 1
#endif
#if ENABLE_SQ8
// Per-dimension scale and scale^2 for dim=128 (copied from host).
// Keep static to avoid cross-TU linkage issues when included in multiple .cu files.
static __device__ __constant__ float g_sq8_scale[128];
static __device__ __constant__ float g_sq8_scale2[128];

static __device__ __forceinline__ float warp_l2_distance_sq8_128(
    const int8_t* __restrict__ q8,
    const int8_t* __restrict__ x8) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int base = lane << 2;  // lane * 4 dims
    const char4 qv = reinterpret_cast<const char4*>(q8)[lane];
    const char4 xv = reinterpret_cast<const char4*>(x8)[lane];
    const int dq0 = (int)qv.x - (int)xv.x;
    const int dq1 = (int)qv.y - (int)xv.y;
    const int dq2 = (int)qv.z - (int)xv.z;
    const int dq3 = (int)qv.w - (int)xv.w;
    float sum = 0.0f;
    sum = fmaf((float)(dq0 * dq0), g_sq8_scale2[base + 0], sum);
    sum = fmaf((float)(dq1 * dq1), g_sq8_scale2[base + 1], sum);
    sum = fmaf((float)(dq2 * dq2), g_sq8_scale2[base + 2], sum);
    sum = fmaf((float)(dq3 * dq3), g_sq8_scale2[base + 3], sum);
    sum += __shfl_down_sync(FULL_MASK, sum, 16);
    sum += __shfl_down_sync(FULL_MASK, sum, 8);
    sum += __shfl_down_sync(FULL_MASK, sum, 4);
    sum += __shfl_down_sync(FULL_MASK, sum, 2);
    sum += __shfl_down_sync(FULL_MASK, sum, 1);
    return __shfl_sync(FULL_MASK, sum, 0);
}
#endif
#ifndef ENABLE_PARTIAL_L2_128
#define ENABLE_PARTIAL_L2_128 0
#endif
#ifndef PARTIAL_L2_128_LANES
#define PARTIAL_L2_128_LANES 24  // 24 lanes * 4 = 96 dims
#endif
template <int VALID_LANES>
static __device__ __forceinline__ float warp_l2_distance_128_prefix(const float* __restrict__ a,
                                                                    const float* __restrict__ b) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    float sum = 0.0f;
    if (lane < VALID_LANES) {
        const float4 av = reinterpret_cast<const float4*>(a)[lane];
        const float4 bv = reinterpret_cast<const float4*>(b)[lane];
        const float dx0 = av.x - bv.x;
        const float dx1 = av.y - bv.y;
        const float dx2 = av.z - bv.z;
        const float dx3 = av.w - bv.w;
        sum = fmaf(dx0, dx0, fmaf(dx1, dx1, fmaf(dx2, dx2, dx3 * dx3)));
    }
    sum += __shfl_down_sync(FULL_MASK, sum, 16);
    sum += __shfl_down_sync(FULL_MASK, sum, 8);
    sum += __shfl_down_sync(FULL_MASK, sum, 4);
    sum += __shfl_down_sync(FULL_MASK, sum, 2);
    sum += __shfl_down_sync(FULL_MASK, sum, 1);
    return __shfl_sync(FULL_MASK, sum, 0);
}

template <int START_LANE>
static __device__ __forceinline__ float warp_l2_distance_128_suffix(const float* __restrict__ a,
                                                                    const float* __restrict__ b) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    float sum = 0.0f;
    if (lane >= START_LANE) {
        const float4 av = reinterpret_cast<const float4*>(a)[lane];
        const float4 bv = reinterpret_cast<const float4*>(b)[lane];
        const float dx0 = av.x - bv.x;
        const float dx1 = av.y - bv.y;
        const float dx2 = av.z - bv.z;
        const float dx3 = av.w - bv.w;
        sum = fmaf(dx0, dx0, fmaf(dx1, dx1, fmaf(dx2, dx2, dx3 * dx3)));
    }
    sum += __shfl_down_sync(FULL_MASK, sum, 16);
    sum += __shfl_down_sync(FULL_MASK, sum, 8);
    sum += __shfl_down_sync(FULL_MASK, sum, 4);
    sum += __shfl_down_sync(FULL_MASK, sum, 2);
    sum += __shfl_down_sync(FULL_MASK, sum, 1);
    return __shfl_sync(FULL_MASK, sum, 0);
}


template <bool FRONTIER_FULL>
static __device__ __forceinline__ DISTANCE_T rabitq_fixed_exact_distance_128(
    const float* __restrict__ query,
    const float* __restrict__ nvec,
    DISTANCE_T pass_cutoff,
    bool* __restrict__ computed_full) {
#if ENABLE_PARTIAL_L2_128
    if constexpr (FRONTIER_FULL) {
        const float partial =
            warp_l2_distance_128_prefix<PARTIAL_L2_128_LANES>(query, nvec);
        if (partial < pass_cutoff) {
            const float tail =
                warp_l2_distance_128_suffix<PARTIAL_L2_128_LANES>(query, nvec);
            return partial + tail;
        }
        *computed_full = false;
        return FLT_MAX;
    }
#endif
    (void)pass_cutoff;
    return warp_l2_distance_128(query, nvec);
}

static __host__ inline uint32_t calculate_shared_mem_size(int dim, int topk, int max_deg, 
                                                           int search_width, int bitlen) {
    return 0;
}

static __global__ void init_identity_map(size_t n, vid_t* map) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) map[idx] = idx;
}

static inline std::vector<uint32_t> load_sign_bit(const char* filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    std::streamoff bytes = f.tellg();
    if (bytes <= 0) return {};
    if ((bytes % static_cast<std::streamoff>(sizeof(uint32_t))) != 0) return {};
    f.seekg(0, std::ios::beg);
    std::vector<uint32_t> buf(static_cast<size_t>(bytes / sizeof(uint32_t)));
    f.read(reinterpret_cast<char*>(buf.data()), bytes);
    if (!f) return {};
    return buf;
}

static inline std::vector<uint32_t> load_sign_bit_read_bin(const char* filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return {};

    int32_t n = 0;
    int32_t dim = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    if (!f || n <= 0 || dim <= 0) return {};

    const size_t words = static_cast<size_t>(n) * static_cast<size_t>(dim);
    std::vector<uint32_t> buf(words);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(words * sizeof(uint32_t)));
    if (!f) return {};
    // Some datasets append extra payload after the leading [n,dim]+u32 block.
    // We only need the leading sign-bit block here, so trailing bytes are allowed.
    return buf;
}

static inline bool save_sign_bit(const char* filename, const uint32_t* data, size_t words) {
    std::ofstream f(filename, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(words * sizeof(uint32_t)));
    return static_cast<bool>(f);
}

static __device__ __forceinline__ uint32_t hashtable_insert_noatomic(
    INDEX_T* const table, const unsigned bitlen, const INDEX_T key) {
    const uint32_t size = 1u << bitlen;
    const uint32_t bit_mask = size - 1;
    uint32_t h = static_cast<uint32_t>(key);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    uint32_t index = h & bit_mask;
    uint32_t stride = ((h >> bitlen) | 1u);
    #pragma unroll 1
    for (unsigned i = 0; i < size; i++) {
        const INDEX_T cur = table[index];
        if (cur == key) return 0;
        if (cur == MAX_INDEX) {
            table[index] = key;
            return 1;
        }
        index = (index + stride) & bit_mask;
    }
    // Local table saturated: fall back to global atomic hash path
    // instead of dropping potentially-new candidates.
    return 2;
}

static __device__ __forceinline__ uint32_t hashtable_insert_adaptive(
    INDEX_T* const table, const unsigned bitlen, const INDEX_T key, const int staged_neighbors) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
    if (staged_neighbors >= (int)NEIGHBOR_BUFFER_SIZE) {
        return 0u;
    }
#endif
#if HASH_INSERT_MAX_PROBES_HIGH_FILL > 0
    if (staged_neighbors >= (int)HASH_INSERT_HIGH_FILL_START) {
        return hashtable_insert_limited(
            table, bitlen, key, (unsigned)HASH_INSERT_MAX_PROBES_HIGH_FILL);
    }
#endif
#if HASH_INSERT_MAX_PROBES > 0
    if (staged_neighbors >= (int)HASH_INSERT_LIMIT_START_FILL) {
        return hashtable_insert_limited(table, bitlen, key, (unsigned)HASH_INSERT_MAX_PROBES);
    }
#endif
    return hashtable_insert(table, bitlen, key);
}

#if RABITQ_KERNEL_ACTIVE && ((RABITQ_SCORE_MODE == 1) || (RABITQ_FASTSCAN_ENABLE != 0))
static __device__ __noinline__ void rabitq_prepare_q6_lut(
    const float* __restrict__ query,
#if RABITQ_QUERY_PROJ_ENABLE
    const float* __restrict__ query_proj_precomputed,
    const float* __restrict__ d_rabitq_proj,
#endif
    int rabitq_q6_dims,
    int rabitq_q6_words,
#if RABITQ_QUERY_PROJ_ENABLE
    float* __restrict__ qproj_buf,
#endif
    uint8_t* __restrict__ q6_lut_flat,
#if RABITQ_PACK_LUT_ACTIVE
    uint8_t* __restrict__ pack_lut_flat,
#endif
    int* __restrict__ q6_sum_out,
    float* __restrict__ q6_lower_out,
    float* __restrict__ q6_width_out) {
    const float* q_lut_src = query;
#if RABITQ_QUERY_PROJ_ENABLE
    if ((query_proj_precomputed != nullptr) && (rabitq_q6_dims == 128)) {
        q_lut_src = query_proj_precomputed;
    } else if ((d_rabitq_proj != nullptr) && (rabitq_q6_dims == 128)) {
        for (int i = 0; i < 128; ++i) {
            qproj_buf[i] = query[i];
        }
#if RABITQ_QUERY_PROJ_MODE == 1
        const float* proj_row = d_rabitq_proj;
        for (int i = 0; i < 128; ++i) {
            qproj_buf[i] *= proj_row[i];
        }
#elif RABITQ_QUERY_PROJ_MODE == 2
        const float* proj_row = d_rabitq_proj;
        for (int i = 0; i < 128; ++i) {
            qproj_buf[i] *= proj_row[i];
        }
        for (int h = 1; h < 128; h <<= 1) {
            for (int i = 0; i < 128; i += (h << 1)) {
                for (int j = 0; j < h; ++j) {
                    const float a = qproj_buf[i + j];
                    const float b = qproj_buf[i + j + h];
                    qproj_buf[i + j] = a + b;
                    qproj_buf[i + j + h] = a - b;
                }
            }
        }
#else
        #pragma unroll
        for (int st = 0; st < (int)RABITQ_QUERY_PROJ_STEPS; ++st) {
            const float* proj_row = d_rabitq_proj + ((size_t)st * 128u);
            for (int i = 0; i < 128; ++i) {
                qproj_buf[i] *= proj_row[i];
            }
            for (int h = 1; h < 128; h <<= 1) {
                for (int i = 0; i < 128; i += (h << 1)) {
                    for (int j = 0; j < h; ++j) {
                        const float a = qproj_buf[i + j];
                        const float b = qproj_buf[i + j + h];
                        qproj_buf[i + j] = a + b;
                        qproj_buf[i + j + h] = a - b;
                    }
                }
            }
        }
#endif
        q_lut_src = qproj_buf;
    }
#endif
    float qlo = FLT_MAX;
    float qhi = -FLT_MAX;
    for (int i = 0; i < rabitq_q6_dims; ++i) {
        const float qv = q_lut_src[i];
        if (qv < qlo) qlo = qv;
        if (qv > qhi) qhi = qv;
    }
    float qwidth = (qhi - qlo) / 63.0f;
    if (!(qwidth > 0.0f) || qwidth != qwidth) qwidth = 1.0f;
    const float inv_w = 1.0f / qwidth;
    int qsum = 0;
    for (int w = 0; w < rabitq_q6_words; ++w) {
        for (int n4 = 0; n4 < 8; ++n4) {
            int qv0 = 0, qv1 = 0, qv2 = 0, qv3 = 0;
            const int i0 = (w << 5) + (n4 << 2) + 0;
            const int i1 = i0 + 1;
            const int i2 = i0 + 2;
            const int i3 = i0 + 3;
            if (i0 < rabitq_q6_dims) {
                int qq = (int)lrintf(((q_lut_src[i0] - qlo) * inv_w) + 0.5f);
                if (qq < 0) qq = 0;
                else if (qq > 63) qq = 63;
                qv0 = qq;
                qsum += qq;
            }
            if (i1 < rabitq_q6_dims) {
                int qq = (int)lrintf(((q_lut_src[i1] - qlo) * inv_w) + 0.5f);
                if (qq < 0) qq = 0;
                else if (qq > 63) qq = 63;
                qv1 = qq;
                qsum += qq;
            }
            if (i2 < rabitq_q6_dims) {
                int qq = (int)lrintf(((q_lut_src[i2] - qlo) * inv_w) + 0.5f);
                if (qq < 0) qq = 0;
                else if (qq > 63) qq = 63;
                qv2 = qq;
                qsum += qq;
            }
            if (i3 < rabitq_q6_dims) {
                int qq = (int)lrintf(((q_lut_src[i3] - qlo) * inv_w) + 0.5f);
                if (qq < 0) qq = 0;
                else if (qq > 63) qq = 63;
                qv3 = qq;
                qsum += qq;
            }
            const int base = (((w << 3) + n4) << 4);
            for (int code = 0; code < 16; ++code) {
                int s = 0;
                if (code & 1) s += qv0;
                if (code & 2) s += qv1;
                if (code & 4) s += qv2;
                if (code & 8) s += qv3;
                q6_lut_flat[base + code] = (uint8_t)s;
            }
#if RABITQ_PACK_LUT_ACTIVE
            pack_lut_flat[base + 0] = 0;
            #pragma unroll
            for (int code = 1; code < 16; ++code) {
                const int lowbit = code & -code;
                int qv = 0;
                if (lowbit == 1) qv = qv3;
                else if (lowbit == 2) qv = qv2;
                else if (lowbit == 4) qv = qv1;
                else qv = qv0;
                pack_lut_flat[base + code] =
                    (uint8_t)(pack_lut_flat[base + code - lowbit] + qv);
            }
#endif
        }
    }
    *q6_sum_out = qsum;
    *q6_lower_out = qlo;
    *q6_width_out = qwidth;
}
#endif

template <typename T>
static __global__ void BeamSearch(
    int K, int beam_size, int nq, int dim, size_t npoints,
    const T* __restrict__ d_queries,
    const T* __restrict__ d_data,
    const int8_t* __restrict__ d_data_sq8,
    vid_t* __restrict__ d_results,
    gpu_long_t* d_total_count_dc,
    gpu_long_t* d_total_hops,
    vid_t* d_starting_points,
    vid_t* d_seed_map,
    uint32_t* d_sign_bit,
    int sign_words_per_edge,
    const uint8_t* d_rabitq_packed_codes,
    int rabitq_packed_code_bytes_per_edge,
    const float* d_rabitq_proj,
    const float* d_rabitq_queries_proj,
    const float* d_rabitq_triple_x,
    const float* d_rabitq_factor_dq,
    const float* d_rabitq_factor_vq,
    const float2* d_rabitq_factor_dqvq,
    const half2* d_rabitq_factor_dqvq_h2,
    const uint32_t* d_sqg_sign_bit,
    const float* d_sqg_center,
    GraphGPU<vid_t> gg
) {
    const int qid = blockIdx.x;
    if (qid >= nq) return;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;  // 256/32 = 8 warps
    const int max_degree = gg.get_max_degree();
#if SQG_KERNEL_ACTIVE
    const int sqg_dims = min(dim, (int)SQG_SAMPLE_DIMS);
    const int sqg_words = (sqg_dims + 31) >> 5;
    const int sqg_tail_bits = (sqg_dims & 31);
    const uint32_t sqg_tail_mask = (sqg_tail_bits == 0) ? 0xFFFFFFFFu : ((1u << sqg_tail_bits) - 1u);
#endif
#if RABITQ_KERNEL_ACTIVE
    const int rabitq_dims = dim;
    const int rabitq_words = (rabitq_dims + 31) >> 5;
    const int rabitq_tail_bits = (rabitq_dims & 31);
    const uint32_t rabitq_tail_mask =
        (rabitq_tail_bits == 0) ? 0xFFFFFFFFu : ((1u << rabitq_tail_bits) - 1u);
#if RABITQ_USE_FACTORS
    // Factor mode follows full-dimension RabitQ scoring.
    const int rabitq_score_words = rabitq_words;
#else
    const int rabitq_score_words =
        (RABITQ_SCORE_WORDS > 0)
            ? min((int)RABITQ_SCORE_WORDS, rabitq_words)
            : min(max((int)RABITQ_SCORE_WORDS_FALLBACK, 1), rabitq_words);
#endif
    const int rabitq_score_dims =
        (rabitq_score_words == rabitq_words) ? rabitq_dims : (rabitq_score_words << 5);
    const int rabitq_score_tail_bits =
        (rabitq_score_words == rabitq_words) ? rabitq_tail_bits : 0;
    const uint32_t rabitq_score_tail_mask =
        (rabitq_score_tail_bits == 0) ? 0xFFFFFFFFu : ((1u << rabitq_score_tail_bits) - 1u);
#if RABITQ_SCORE_MODE == 1
    const int rabitq_q6_words = rabitq_words;
    const int rabitq_q6_dims = rabitq_dims;
#elif RABITQ_FASTSCAN_ENABLE
    const int rabitq_q6_words = rabitq_score_words;
    const int rabitq_q6_dims = rabitq_score_dims;
#endif
#if RABITQ_SCORE_MODE == 1
    const int rabitq_sign_words_required = rabitq_words;
#elif RABITQ_USE_FACTORS
    const int rabitq_sign_words_required = rabitq_words;
#else
    const int rabitq_sign_words_required = rabitq_score_words;
#endif
#endif
    const int sign_word_stride = max(0, sign_words_per_edge);
    if (beam_size <= 0 || beam_size > BEAM_SIZE) return;
    constexpr INDEX_T kIndexMsbMask = 0x80000000;
    __shared__ INDEX_T s_frontier_idx[BEAM_SIZE];
    __shared__ DISTANCE_T s_frontier_dist[BEAM_SIZE];
    __shared__ int s_frontier_size;
#if RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_PARITY_MODE && \
    RABITQ_CPU_PARITY_EXACT_POOL
    __shared__ INDEX_T s_exact_topk_idx[BEAM_SIZE];
    __shared__ DISTANCE_T s_exact_topk_dist[BEAM_SIZE];
    __shared__ int s_exact_topk_size;
    __shared__ INDEX_T s_exact_stream_idx[RABITQ_CPU_PARITY_STREAM_CAP];
    __shared__ DISTANCE_T s_exact_stream_dist[RABITQ_CPU_PARITY_STREAM_CAP];
    __shared__ int s_exact_stream_size;
#endif
    __shared__ int s_dc_count;
    __shared__ int s_iter_count;
    __shared__ int s_no_progress_iters;
    __shared__ int s_break_now;
    __shared__ INDEX_T s_hash[HASH_SIZE];
    __shared__ INDEX_T s_neighbor_idx[NEIGHBOR_BUFFER_SIZE];
    __shared__ DISTANCE_T s_neighbor_dist[NEIGHBOR_BUFFER_SIZE];
#if RABITQ_KERNEL_ACTIVE && RABITQ_NEIGHBOR_APPROX_BUFFER
#if !(RABITQ_REUSE_CAND_TMP_AS_APPROX && (CANDIDATE_SELECT_STRICT == 2))
    __shared__ DISTANCE_T s_neighbor_approx[NEIGHBOR_BUFFER_SIZE];
#endif
#if RABITQ_GLOBAL_APPROX_PRESELECT
    __shared__ INDEX_T s_rabitq_sort_idx[NEIGHBOR_BUFFER_SIZE];
    __shared__ DISTANCE_T s_rabitq_sort_approx[NEIGHBOR_BUFFER_SIZE];
    __shared__ int s_rabitq_global_pool_count;
    __shared__ int s_rabitq_global_pool_base;
#endif
#endif
#if RABITQ_KERNEL_ACTIVE && RABITQ_TWO_PHASE_EXACT
    __shared__ DISTANCE_T s_rabitq_exact_pool[BEAM_SIZE];
    __shared__ int s_rabitq_exact_pool_size;
    __shared__ DISTANCE_T s_rabitq_exact_cutoff;
    __shared__ uint32_t s_rabitq_batch_anchor_mask;
    __shared__ DISTANCE_T s_rabitq_batch_gate;
#elif RABITQ_KERNEL_ACTIVE && RABITQ_GPU_DUALPOOL_ENGINE
    __shared__ int s_rabitq_dualpool_anchor_budget;
#endif
    __shared__ int s_num_neighbors;
    __shared__ INDEX_T s_candidate_idx[CANDIDATE_BUFFER_SIZE];
    __shared__ DISTANCE_T s_candidate_dist[CANDIDATE_BUFFER_SIZE];
    __shared__ int s_candidate_count;
    __shared__ int s_candidate_sorted;
#if CANDIDATE_SELECT_STRICT == 2
    __shared__ INDEX_T s_candidate_tmp_idx[NEIGHBOR_BUFFER_SIZE];
    __shared__ DISTANCE_T s_candidate_tmp_dist[NEIGHBOR_BUFFER_SIZE];
    __shared__ int s_candidate_tmp_count;
#if CANDIDATE_SELECT_STRICT_FAST && STRICT_FAST_DEBUG_PRINT
    __shared__ int s_strict_fast_active_iters;
    __shared__ int s_strict_total_active_iters;
    __shared__ int s_strict_fast_low_iters;
    __shared__ int s_strict_fast_high_iters;
    __shared__ int s_strict_fast_valid_sum;
#endif
#endif
#if RABITQ_KERNEL_ACTIVE && RABITQ_NEIGHBOR_APPROX_BUFFER && \
    RABITQ_REUSE_CAND_TMP_AS_APPROX && (CANDIDATE_SELECT_STRICT == 2)
#define s_neighbor_approx s_candidate_tmp_dist
#endif
    __shared__ INDEX_T s_expand_pos[EXPAND_WIDTH_MAX];
    __shared__ int s_num_expanders;
#if ENABLE_FRONTIER_DEDUP
#if FRONTIER_DEDUP_METHOD == 2
    __shared__ INDEX_T s_frontier_dedup_hash[FRONTIER_DEDUP_HASH_SIZE];
#endif
#if FRONTIER_DEDUP_PARALLEL || FRONTIER_DEDUP_METHOD == 2
    __shared__ int s_frontier_dedup_keep[BEAM_SIZE];
    __shared__ int s_frontier_dedup_pos[BEAM_SIZE];
#endif
#endif
    __shared__ int s_frontier_dedup_apply;
    __shared__ int s_frontier_adj_dup_cnt;
#if WARP_L1_DEDUP_ENABLE && WARP_L1_DEDUP_FEATURE_ACTIVE
    __shared__ INDEX_T s_warp_l1_hash[WARPS_PER_BLOCK][1 << WARP_L1_HASH_BITLEN];
#endif
#if ENABLE_DGS
    // Per-warp cached query-direction words (avoid repeated shfl broadcasts).
    __shared__ uint32_t s_dgs_qwords[8][DGS_MAX_WORDS];
    __shared__ DISTANCE_T s_dgs_best_dist;
    __shared__ int s_dgs_stall_iters;
    __shared__ int s_dgs_fallback_left;
    __shared__ int s_dgs_tree_hits;
#endif
#if RABITQ_KERNEL_ACTIVE
    // Query sign words are block-invariant; one shared copy avoids redundant per-warp setup.
    __shared__ uint32_t s_rabitq_qwords[RABITQ_QWORDS_SHARED_WORDS];
    __shared__ DISTANCE_T s_rabitq_best_dist;
    __shared__ int s_rabitq_stall_iters;
    __shared__ int s_rabitq_break_now;
    __shared__ int s_rabitq_frontier_full_iter;
#if RABITQ_DISABLE_IF_NO_PRUNE_ITERS > 0
    __shared__ int s_rabitq_force_off;
    __shared__ int s_rabitq_no_prune_iters;
    __shared__ int s_rabitq_iter_active_tiles;
    __shared__ int s_rabitq_iter_pruned_tiles;
#endif
#if RABITQ_EDGE_TILE_CACHE_WORDS > 0
    __shared__ uint32_t s_rabitq_tile_words[32][RABITQ_EDGE_TILE_CACHE_WORDS];
#endif
#if (RABITQ_SCORE_MODE == 1) || (RABITQ_FASTSCAN_ENABLE != 0)
    __shared__ uint8_t s_rabitq_q6_lut[RABITQ_Q6_LUT_WORDS][8][16];
#if RABITQ_PACK_LUT_ACTIVE
    __shared__ uint8_t s_rabitq_pack_lut[RABITQ_Q6_LUT_WORDS * 8][16];
#endif
    __shared__ int s_rabitq_q6_sum;
    __shared__ float s_rabitq_q6_lower;
    __shared__ float s_rabitq_q6_width;
    __shared__ int s_rabitq_q6_ready;
#if RABITQ_QUERY_PROJ_ENABLE
    __shared__ float s_rabitq_qproj[128];
#endif
#endif
#if RABITQ_USE_FACTORS
    __shared__ float s_rabitq_qsign_sum;
#endif
#endif
#if SQG_KERNEL_ACTIVE
    __shared__ uint32_t s_sqg_qwords[8][SQG_MAX_WORDS];
#if SQG_USE_FHT
    __shared__ float s_sqg_query_fht[128];
#endif
#if SQG_EARLY_STOP_ENABLE
    __shared__ DISTANCE_T s_sqg_best_dist;
    __shared__ int s_sqg_stall_iters;
    __shared__ int s_sqg_break_now;
#endif
#endif
#if ENABLE_SQ8
    __shared__ __align__(4) int8_t s_query_sq8[128];
#endif
    __shared__ int s_pick_counts[WARPS_PER_BLOCK];
    __shared__ int s_pick_num;
    __shared__ int s_pick_round_base;
    const T* query = d_queries + qid * dim;
#if RABITQ_QUERY_PROJ_ENABLE
    const float* query_proj_precomputed = nullptr;
    if ((d_rabitq_queries_proj != nullptr) && (dim == 128)) {
        query_proj_precomputed = d_rabitq_queries_proj + ((size_t)qid * 128u);
    }
#endif
#if ENABLE_DGS && DGS_DEBUG_PRINT
    if (qid == 0 && tid == 0) {
        printf("DGS kernel: ENABLE_DGS=1, DGS_TOPN=%d, DGS_COOL_RATIO=%f, d_sign_bit=%p\n",
               (int)DGS_TOPN, (double)DGS_COOL_RATIO, (void*)d_sign_bit);
    }
#endif
    hashtable_init(s_hash, BITLEN, 0);

#if ENABLE_SQ8
    if (dim == 128 && d_data_sq8 != nullptr) {
        for (int i = tid; i < 128; i += blockDim.x) {
            const float scale = g_sq8_scale[i];
            const float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
            int q = __float2int_rn(((const float*)query)[i] * inv);
            q = max(-127, min(127, q));
            s_query_sq8[i] = (int8_t)q;
        }
    }
#endif
#if RABITQ_KERNEL_ACTIVE
    // Q6 LUT / packed LUT are lazily initialized when RabitQ gate first opens.
    if ((rabitq_words > 0) && (tid < (int)RABITQ_QWORDS_SHARED_WORDS)) {
        const int w = tid;
        uint32_t word = 0u;
        #pragma unroll
        for (int b = 0; b < 32; b++) {
            const int i = (w << 5) + b;
            if (i >= rabitq_dims) break;
            const float qv = (float)query[i];
            word |= ((qv >= 0.0f) ? 1u : 0u) << b;
        }
        s_rabitq_qwords[w] = word;
    }
    if (tid == 0) {
#if RABITQ_USE_FACTORS
        float qsign_sum = 0.0f;
        for (int i = 0; i < rabitq_dims; i++) {
            qsign_sum += (((float)query[i] >= 0.0f) ? 1.0f : -1.0f);
        }
        s_rabitq_qsign_sum = qsign_sum;
#endif
    }
#endif
#if SQG_KERNEL_ACTIVE
#if SQG_USE_FHT
    if (dim == 128 && sqg_dims == 128) {
        for (int i = tid; i < 128; i += blockDim.x) {
            s_sqg_query_fht[i] = (float)query[i];
        }
        __syncthreads();
        for (int step = 1; step < 128; step <<= 1) {
            for (int i = tid; i < 128; i += blockDim.x) {
                if ((i & step) == 0) {
                    const float a = s_sqg_query_fht[i];
                    const float b = s_sqg_query_fht[i + step];
                    s_sqg_query_fht[i] = a + b;
                    s_sqg_query_fht[i + step] = a - b;
                }
            }
            __syncthreads();
        }
    }
#endif
    if ((sqg_words > 0) && (sqg_words <= (int)SQG_MAX_WORDS) && (lane_id < sqg_words)) {
        const int w = lane_id;
        uint32_t word = 0;
        #pragma unroll
        for (int b = 0; b < 32; b++) {
            const int i = (w << 5) + b;
            if (i >= sqg_dims) break;
            float thr = 0.0f;
#if SQG_SIGNBIT_MODE == 1
            if (d_sqg_center != nullptr) thr = d_sqg_center[i];
#endif
            float qv = (float)query[i];
#if SQG_USE_FHT
            if (dim == 128 && sqg_dims == 128) {
                qv = s_sqg_query_fht[i];
            }
#endif
            word |= ((qv >= thr) ? 1u : 0u) << b;
        }
        s_sqg_qwords[warp_id][w] = word;
    }
#endif
    
    for (int i = tid; i < BEAM_SIZE; i += blockDim.x) {
        s_frontier_idx[i] = MAX_INDEX;
        s_frontier_dist[i] = FLT_MAX;
#if RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_PARITY_MODE && \
    RABITQ_CPU_PARITY_EXACT_POOL
        s_exact_topk_idx[i] = MAX_INDEX;
        s_exact_topk_dist[i] = FLT_MAX;
#endif
    }
#if RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_PARITY_MODE && \
    RABITQ_CPU_PARITY_EXACT_POOL
    for (int i = tid; i < (int)RABITQ_CPU_PARITY_STREAM_CAP; i += blockDim.x) {
        s_exact_stream_idx[i] = MAX_INDEX;
        s_exact_stream_dist[i] = FLT_MAX;
    }
#endif
    for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
        s_candidate_idx[i] = MAX_INDEX;
        s_candidate_dist[i] = FLT_MAX;
    }

    if (tid == 0) {
        s_frontier_size = 0;
#if RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_PARITY_MODE && \
    RABITQ_CPU_PARITY_EXACT_POOL
        s_exact_topk_size = 0;
        s_exact_stream_size = 0;
#endif
        s_dc_count = 0;
        s_iter_count = 0;
        s_no_progress_iters = 0;
        s_break_now = 0;
        s_num_neighbors = 0;
        s_candidate_count = 0;
#if ENABLE_DGS
        s_dgs_best_dist = FLT_MAX;
        s_dgs_stall_iters = 0;
        s_dgs_fallback_left = 0;
        s_dgs_tree_hits = 0;
#endif
#if RABITQ_KERNEL_ACTIVE
        s_rabitq_best_dist = FLT_MAX;
        s_rabitq_stall_iters = 0;
        s_rabitq_break_now = 0;
        s_rabitq_frontier_full_iter = -1;
#if RABITQ_DISABLE_IF_NO_PRUNE_ITERS > 0
        s_rabitq_force_off = 0;
        s_rabitq_no_prune_iters = 0;
        s_rabitq_iter_active_tiles = 0;
        s_rabitq_iter_pruned_tiles = 0;
#endif
#if (RABITQ_SCORE_MODE == 1) || (RABITQ_FASTSCAN_ENABLE != 0)
        s_rabitq_q6_ready = 0;
#endif
#endif
#if SQG_KERNEL_ACTIVE && SQG_EARLY_STOP_ENABLE
        s_sqg_best_dist = FLT_MAX;
        s_sqg_stall_iters = 0;
        s_sqg_break_now = 0;
#endif
#if CANDIDATE_SELECT_STRICT == 2 && CANDIDATE_SELECT_STRICT_FAST && STRICT_FAST_DEBUG_PRINT
        s_strict_fast_active_iters = 0;
        s_strict_total_active_iters = 0;
        s_strict_fast_low_iters = 0;
        s_strict_fast_high_iters = 0;
        s_strict_fast_valid_sum = 0;
#endif
    }
    __syncthreads();
    {
        int num_starts = min(K, beam_size);
#if RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_PARITY_MODE
        // Decoupled parity: keep default multi-start unless explicitly capped.
#if RABITQ_CPU_PARITY_START_CAP > 0
        num_starts = min(num_starts, (int)RABITQ_CPU_PARITY_START_CAP);
#endif
#endif
        
        for (int s = warp_id; s < num_starts; s += num_warps) {
#if ENABLE_GHOST_STAGE
            vid_t start_vid = d_starting_points[(size_t)qid * (size_t)beam_size + (size_t)(s % beam_size)];
#else
            vid_t start_vid = d_starting_points[s % beam_size];
#endif
            if (start_vid >= npoints) continue;
            uint32_t is_new = 0;
            if (lane_id == 0) {
                is_new = hashtable_insert(s_hash, BITLEN, start_vid);
            }
            is_new = __shfl_sync(FULL_MASK, is_new, 0);
            if (!is_new) continue;
#if RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_PARITY_MODE
            const DISTANCE_T dist = FLT_MAX;
#else
            const T* svec = d_data + start_vid * dim;
            DISTANCE_T dist = (dim == 128) ? warp_l2_distance_128(query, svec)
                                           : warp_l2_distance(dim, query, svec);
#endif
            if (lane_id == 0) {
                int pos = atomicAdd(&s_frontier_size, 1);
                if (pos < beam_size) {
                    s_frontier_idx[pos] = start_vid;
                    s_frontier_dist[pos] = dist;
#if !(RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_PARITY_MODE)
                    atomicAdd(&s_dc_count, 1);
#endif
                }
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        int n = min(s_frontier_size, beam_size);
        for (int i = 1; i < n; i++) {
            DISTANCE_T key_d = s_frontier_dist[i];
            INDEX_T key_i = s_frontier_idx[i];
            int j = i - 1;
            while (j >= 0 && s_frontier_dist[j] > key_d) {
                s_frontier_dist[j + 1] = s_frontier_dist[j];
                s_frontier_idx[j + 1] = s_frontier_idx[j];
                j--;
            }
            s_frontier_dist[j + 1] = key_d;
            s_frontier_idx[j + 1] = key_i;
        }
    }
    
    __syncthreads();

#if ENABLE_DGS
    if (tid == 0) {
        s_dgs_best_dist = (s_frontier_size > 0) ? s_frontier_dist[0] : FLT_MAX;
    }
    __syncthreads();
#endif
#if RABITQ_KERNEL_ACTIVE
    if (tid == 0) {
        const int nf0 = min(s_frontier_size, beam_size);
#if RABITQ_STALL_TRACK_MODE == 1
        if (nf0 > 0) {
            const int kpos0 = min(max(K, 1), nf0) - 1;
            s_rabitq_best_dist = s_frontier_dist[kpos0];
        } else {
            s_rabitq_best_dist = FLT_MAX;
        }
#else
        s_rabitq_best_dist = (nf0 > 0) ? s_frontier_dist[0] : FLT_MAX;
#endif
    }
    __syncthreads();
#endif

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        if (tid == 0) {
            s_break_now = 0;
#if RABITQ_KERNEL_ACTIVE && (RABITQ_DISABLE_IF_NO_PRUNE_ITERS > 0)
            s_rabitq_iter_active_tiles = 0;
            s_rabitq_iter_pruned_tiles = 0;
#endif
        }
        __syncthreads();
#if RABITQ_KERNEL_ACTIVE
        if (iter >= (int)RABITQ_MAX_ITER_CAP) {
            break;
        }
#endif
        if (iter > 0 && (iter % SMALL_HASH_RESET_INTERVAL) == 0) {
            hashtable_init(s_hash, BITLEN, 0);
            __syncthreads();
            hashtable_restore(s_hash, BITLEN, s_frontier_idx, min(s_frontier_size, beam_size), 0);
            __syncthreads();
        }
#if ENABLE_DGS
        if (tid == 0 && DGS_STALL_FALLBACK_ENABLE && iter > 0) {
            const DISTANCE_T curr_best = (s_frontier_size > 0) ? s_frontier_dist[0] : FLT_MAX;
            const DISTANCE_T prev_best = s_dgs_best_dist;
            DISTANCE_T improve_need = (DISTANCE_T)DGS_STALL_IMPROVE_EPS_ABS;
            const DISTANCE_T rel_need = (DISTANCE_T)(fabsf((float)prev_best) * (float)DGS_STALL_IMPROVE_EPS_REL);
            if (rel_need > improve_need) improve_need = rel_need;

            const DISTANCE_T improve = prev_best - curr_best;
            if (improve > improve_need) {
                s_dgs_best_dist = curr_best;
                s_dgs_stall_iters = 0;
            } else {
                s_dgs_stall_iters++;
            }

            if (s_dgs_fallback_left > 0) {
                s_dgs_fallback_left--;
            }
            if (s_dgs_fallback_left == 0 && s_dgs_stall_iters >= (int)DGS_STALL_TRIGGER) {
                s_dgs_fallback_left = max(1, (int)DGS_STALL_FALLBACK_ITERS);
                s_dgs_stall_iters = 0;
            }
        }
        __syncthreads();

        const int cool_start = (int)((1.0f - (float)DGS_COOL_RATIO) * (float)DGS_EST_ITERS);
        const int dgs_word_count = (dim + 31) >> 5;
        const bool dgs_static_ready =
            (d_sign_bit != nullptr) &&
            (sign_word_stride >= dgs_word_count) &&
            (max_degree >= 32) &&
            (DGS_TOPN > 0) &&
            (DGS_TOPN < 32) &&
            (dgs_word_count > 0) &&
            (dgs_word_count <= (int)DGS_MAX_WORDS);
        const bool dgs_fallback_active =
            DGS_STALL_FALLBACK_ENABLE && (s_dgs_fallback_left > 0);
        bool dgs_filter_gate_base =
            dgs_static_ready &&
            !dgs_fallback_active &&
            (iter >= (int)DGS_WARMUP_ITERS) &&
            (iter < cool_start) &&
            (DGS_TOPN > 0);
#if !DGS_BIND_WIDE_ACTIVE
        dgs_filter_gate_base =
            (d_sign_bit != nullptr) &&
            (sign_word_stride >= dgs_word_count) &&
            !dgs_fallback_active &&
            (iter >= (int)DGS_WARMUP_ITERS) &&
            (iter < cool_start) &&
            (DGS_TOPN > 0);
#endif
        bool dgs_filter_gate = dgs_filter_gate_base;
        bool use_wide_expand = dgs_filter_gate_base;
        bool dgs_tree_relax_topn_active = false;
#if DGS_TREE_OFF_MODE != 0
        bool dgs_tree_off = false;
        if (dgs_filter_gate_base) {
            const int active_span = cool_start - (int)DGS_WARMUP_ITERS;
            const int t = iter - (int)DGS_WARMUP_ITERS;
            if (active_span >= (int)DGS_TREE_MIN_ACTIVE_ITERS && t >= (int)DGS_TREE_START_ITERS) {
                const int u = t - (int)DGS_TREE_START_ITERS;
#if DGS_TREE_OFF_PATTERN == 1
                dgs_tree_off = ((u & (u + 1)) == 0);
#else
                dgs_tree_off = (((u + 2) & (u + 1)) == 0);
#endif
            }
        }
        bool dgs_tree_apply = dgs_tree_off;
        if (dgs_tree_apply) {
#if DGS_TREE_REQUIRE_STALL
            dgs_tree_apply = (s_dgs_stall_iters >= (int)DGS_TREE_STALL_THRESHOLD);
#endif
#if DGS_TREE_MAX_HITS > 0
            dgs_tree_apply = dgs_tree_apply && (s_dgs_tree_hits < (int)DGS_TREE_MAX_HITS);
#endif
        }
        if (tid == 0 && dgs_tree_apply) {
            s_dgs_tree_hits++;
        }
        __syncthreads();
#if DGS_TREE_OFF_MODE == 1
        if (dgs_tree_apply) {
            dgs_filter_gate = false;
            use_wide_expand = false;
        }
#elif DGS_TREE_OFF_MODE == 2
        if (dgs_tree_apply) {
#if DGS_TREE_RELAX_TOPN > 0
            dgs_tree_relax_topn_active = true;
#else
            dgs_filter_gate = false;
#endif
        }
#endif
#endif
#if DGS_TREE_OFF_MODE == 2
#if DGS_TREE_MODE2_EARLY_WARMUP_ITERS > 0
        if (iter < (int)DGS_TREE_MODE2_EARLY_WARMUP_ITERS) {
            // Absorb warmup's early-stage advantage: keep mode2 but delay wide expansion.
            dgs_filter_gate = false;
            dgs_tree_relax_topn_active = false;
            use_wide_expand = false;
        }
#endif
#endif
#if DGS_FORCE_WIDE_EXPAND >= 0
        use_wide_expand = (DGS_FORCE_WIDE_EXPAND != 0);
#endif
        int expand_width = SEARCH_WIDTH;
        if (use_wide_expand) {
#if DGS_DECOUPLE_EXPAND_WIDTH
            expand_width = (int)DGS_DECOUPLED_EXPAND_WIDTH;
#else
            expand_width = (int)DGS_SEARCH_WIDTH;
#endif
        }
#if DGS_CAP_EXPAND_WIDTH_BY_CANDIDATE
        if (use_wide_expand) {
            constexpr int kTopN = (int)DGS_TOPN;
            constexpr int kCand = (int)CANDIDATE_BUFFER_SIZE;
            constexpr int kSafe = (kTopN > 0) ? (kCand / kTopN) : 1;
            const int safe_width = (kSafe > 0) ? kSafe : 1;
            expand_width = min(expand_width, safe_width);
        }
#endif
#if DGS_TREE_OFF_MODE == 3
#if DGS_TREE_WIDE_CAP > 0
        if (dgs_tree_apply && use_wide_expand) {
            expand_width = min(expand_width, (int)DGS_TREE_WIDE_CAP);
        }
#endif
#endif
#if DGS_FORCE_EXPAND_WIDTH > 0
        expand_width = (int)DGS_FORCE_EXPAND_WIDTH;
#endif
#if RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_PARITY_MODE
        // Decoupled parity: do not force narrow width by default.
#if RABITQ_CPU_PARITY_FORCE_EXPAND_WIDTH
        expand_width = max(1, (int)RABITQ_CPU_PARITY_POP_BATCH);
#else
        expand_width = max(expand_width, max(1, (int)RABITQ_CPU_PARITY_POP_BATCH));
#endif
#endif
        if (expand_width < 1) expand_width = 1;
#else
        int expand_width = SEARCH_WIDTH;
#if SQG_KERNEL_ACTIVE && SQG_ADAPT_EXPAND_ENABLE
        int sqg_cool_start_expand =
            (int)((1.0f - (float)SQG_COOL_RATIO) * (float)SQG_EST_ITERS);
#if SQG_ACTIVE_ITERS > 0
        sqg_cool_start_expand = min((int)SQG_ACTIVE_ITERS, (int)SQG_EST_ITERS);
#endif
        bool sqg_expand_gate =
            (d_sqg_sign_bit != nullptr) &&
            (SQG_TOPN > 0) &&
            (SQG_TOPN < 32) &&
            (sqg_dims > 0) &&
            (sqg_words > 0) &&
            (sqg_words <= (int)SQG_MAX_WORDS) &&
            (iter >= (int)SQG_WARMUP_ITERS) &&
            (iter < sqg_cool_start_expand);
#if SQG_FULLSCAN_STRIDE > 0
        if (iter > 0 && (iter % (int)SQG_FULLSCAN_STRIDE) == 0) {
            sqg_expand_gate = false;
        }
#endif
#if SQG_ADAPT_EXPAND_MIN_FRONTIER > 0
        sqg_expand_gate = sqg_expand_gate &&
                          (min(s_frontier_size, beam_size) >= (int)SQG_ADAPT_EXPAND_MIN_FRONTIER);
#endif
        if (sqg_expand_gate) {
            expand_width = min(expand_width, (int)SQG_ADAPT_EXPAND_WIDTH);
        }
#endif
#if RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_PARITY_MODE
        // Decoupled parity: do not force narrow width by default.
#if RABITQ_CPU_PARITY_FORCE_EXPAND_WIDTH
        expand_width = max(1, (int)RABITQ_CPU_PARITY_POP_BATCH);
#else
        expand_width = max(expand_width, max(1, (int)RABITQ_CPU_PARITY_POP_BATCH));
#endif
#endif
        if (expand_width < 1) expand_width = 1;
#endif
#if RABITQ_KERNEL_ACTIVE && (RABITQ_MAIN_POP_BATCH > 0)
        bool rabitq_pop_cap_on = true;
#if RABITQ_MAIN_POP_BATCH_START_ITER > 0
        rabitq_pop_cap_on = rabitq_pop_cap_on && (iter >= (int)RABITQ_MAIN_POP_BATCH_START_ITER);
#endif
#if RABITQ_MAIN_POP_BATCH_REQUIRE_FRONTIER_FULL
        rabitq_pop_cap_on = rabitq_pop_cap_on && (s_frontier_size >= beam_size);
#endif
#if RABITQ_MAIN_POP_BATCH_REQUIRE_STALL
        rabitq_pop_cap_on =
            rabitq_pop_cap_on && (s_rabitq_stall_iters >= (int)RABITQ_MAIN_POP_BATCH_STALL_ITERS);
#endif
        if (rabitq_pop_cap_on) {
            expand_width = min(expand_width, (int)RABITQ_MAIN_POP_BATCH);
        }
#endif
#if PICK_EXPANDERS_BLOCK_WIDE
        if (tid == 0) s_pick_num = 0;
        __syncthreads();
        const int M = min(s_frontier_size, beam_size);
        int itopk_max = M;
        if (itopk_max % blockDim.x) itopk_max += blockDim.x - (itopk_max % blockDim.x);
        for (int base = 0; base < itopk_max; base += blockDim.x) {
            const int j = base + tid;
            INDEX_T index = 0;
            bool new_parent = false;
            if (j < M) {
                index = s_frontier_idx[j];
                new_parent = ((index & kIndexMsbMask) == 0);
            }
            const uint32_t ballot = __ballot_sync(FULL_MASK, new_parent);
            const int warp_count = __popc(ballot);
            if (lane_id == 0) s_pick_counts[warp_id] = warp_count;
            __syncthreads();
            if (tid == 0) {
                int total = 0;
                #pragma unroll
                for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                    if (w < num_warps) total += s_pick_counts[w];
                }
                s_pick_round_base = s_pick_num;
                s_pick_num += total;
            }
            __syncthreads();
            int warp_prefix = 0;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                if (w < warp_id) warp_prefix += s_pick_counts[w];
            }
            const uint32_t lane_mask = (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
            const int lane_prefix = __popc(ballot & lane_mask);
            const int pos = s_pick_round_base + warp_prefix + lane_prefix;
            if (new_parent && pos < expand_width) {
                s_expand_pos[pos] = j;
                // mark as expanded
                s_frontier_idx[j] |= kIndexMsbMask;
            }
            __syncthreads();
        }
        if (tid == 0) {
            s_num_expanders = min(s_pick_num, expand_width);
        }
        __syncthreads();
#else
        if (warp_id == 0) {
            int num_exp = pick_expanders(expand_width, s_expand_pos, min(s_frontier_size, beam_size), s_frontier_idx);
            if (lane_id == 0) s_num_expanders = num_exp;
        }
        __syncthreads();
#endif
        if (s_num_expanders == 0) break;
        if (tid == 0) {
            s_iter_count++;
        }
        if (tid == 0) s_num_neighbors = 0;
        __syncthreads();
#if WARP_L1_DEDUP_ENABLE && WARP_L1_DEDUP_FEATURE_ACTIVE
        for (int i = lane_id; i < (1 << WARP_L1_HASH_BITLEN); i += WARP_SIZE) {
            s_warp_l1_hash[warp_id][i] = MAX_INDEX;
        }
        __syncwarp();
#endif
#if ENABLE_DGS
        if (!use_wide_expand) {
            for (int e = warp_id; e < s_num_expanders; e += num_warps) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
                if (*((volatile int*)&s_num_neighbors) >= (int)NEIGHBOR_BUFFER_SIZE) break;
#endif
                INDEX_T expand_pos = s_expand_pos[e];
                if (expand_pos >= beam_size) continue;
                vid_t current = s_frontier_idx[expand_pos] & ~kIndexMsbMask;
                if (current >= npoints) continue;
                vid_t* neighbors = gg.N(current);
                for (int base = 0; base < max_degree; base += 32) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
                    if (*((volatile int*)&s_num_neighbors) >= (int)NEIGHBOR_BUFFER_SIZE) break;
#endif
                    int n = base + lane_id;
                    vid_t neighbor = (n < max_degree) ? neighbors[n] : UINT_MAX;
                    bool valid = (n < max_degree) && (neighbor < npoints);
#if SKIP_ATOMIC_WHEN_POOL_FULL
                    const uint32_t valid_mask0 = __ballot_sync(FULL_MASK, valid);
                    if (valid_mask0 != 0u) {
                        const int remain_slots =
                            (int)NEIGHBOR_BUFFER_SIZE - (*((volatile int*)&s_num_neighbors));
                        if (remain_slots <= 0) {
                            valid = false;
                        } else {
                            const int valid_cnt = __popc(valid_mask0);
                            if (valid_cnt > remain_slots) {
                                const uint32_t lane_mask =
                                    (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
                                const int rank = __popc(valid_mask0 & lane_mask);
                                valid = (((valid_mask0 >> lane_id) & 1u) != 0u) && (rank < remain_slots);
                            }
                        }
                    }
#endif
                    bool valid_local = valid;
#if WARP_L1_DEDUP_ENABLE && WARP_L1_DEDUP_FEATURE_ACTIVE
                    if (valid_local) {
                        valid_local = (hashtable_insert_noatomic(
                            s_warp_l1_hash[warp_id], WARP_L1_HASH_BITLEN, neighbor) != 0u);
                    }
#endif
                    uint32_t is_new = 0;
#if HASH_WARP_DEDUP
                    const uint32_t active = __ballot_sync(FULL_MASK, valid_local);
                    if (active != 0u) {
                        const uint32_t same = __match_any_sync(active, (uint32_t)neighbor);
                        const int leader = __ffs(same) - 1;
                        if (valid_local && lane_id == leader) {
                                is_new = hashtable_insert_adaptive(
                                s_hash, BITLEN, neighbor,
                                *((volatile int*)&s_num_neighbors));
                        }
                    }
#else
                    if (valid_local) {
                        is_new = hashtable_insert_adaptive(
                            s_hash, BITLEN, neighbor,
                            *((volatile int*)&s_num_neighbors));
                    }
#endif
                    uint32_t ballot = __ballot_sync(FULL_MASK, is_new);
                    int warp_count = __popc(ballot);
                    int base_offset = 0;
                    if (lane_id == 0 && warp_count > 0) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
                        if (*((volatile int*)&s_num_neighbors) < (int)NEIGHBOR_BUFFER_SIZE) {
                            base_offset = atomicAdd(&s_num_neighbors, warp_count);
                        } else {
                            base_offset = (int)NEIGHBOR_BUFFER_SIZE;
                        }
#else
                        base_offset = atomicAdd(&s_num_neighbors, warp_count);
#endif
                    }
                    base_offset = __shfl_sync(FULL_MASK, base_offset, 0);
                    if (is_new) {
                        int my_offset = __popc(ballot & ((1u << lane_id) - 1));
                        int slot = base_offset + my_offset;
                        if (slot < (int)NEIGHBOR_BUFFER_SIZE) {
                            s_neighbor_idx[slot] = neighbor;
                        }
                    }
                }
            }
        } else {
            for (int e = warp_id; e < s_num_expanders && e < expand_width; e += num_warps) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
                if (*((volatile int*)&s_num_neighbors) >= (int)NEIGHBOR_BUFFER_SIZE) break;
#endif
                INDEX_T expand_pos = s_expand_pos[e];
                if (expand_pos < beam_size) {
                    vid_t current = s_frontier_idx[expand_pos] & ~kIndexMsbMask;
                    if (current < npoints) {
                        vid_t* neighbors = gg.N(current);
                        const int deg = min(max_degree, 32);  // graph-d32 fast path

                        const int W = (dim + 31) >> 5;
                        const bool dgs_active = dgs_filter_gate &&
                                                ((int)DGS_TOPN < deg) &&
                                                (W > 0) && (W <= (int)DGS_MAX_WORDS) &&
                                                (deg == 32);
                        if (dgs_active && lane_id < W) {
                            const T* uvec = d_data + ((size_t)current * (size_t)dim);
                            const int w = lane_id;
                            uint32_t word = 0;
                            #pragma unroll
                            for (int b = 0; b < 32; b++) {
                                const int i = (w << 5) + b;
                                if (i >= dim) break;
                                const T diff = query[i] - uvec[i];
                                word |= ((diff >= (T)0) ? 1u : 0u) << b;
                            }
                            s_dgs_qwords[warp_id][w] = word;
                        }
                        if (dgs_active) __syncwarp();
                        const vid_t nb = (lane_id < deg) ? neighbors[lane_id] : UINT_MAX;
                        const uint32_t raw_nb = (uint32_t)nb;
                        uint32_t val = (uint32_t)nb;
                        const bool nb_valid = (lane_id < deg) && (nb < npoints);
                        bool keep_sign = nb_valid;
                        bool keep_raw = false;

                        if (dgs_active) {
                            uint32_t key0 = 0;  // larger is better
                            if (nb < npoints) {
                                const uint32_t* uv_bits =
                                    d_sign_bit + (((size_t)current * (size_t)max_degree + (size_t)lane_id) * (size_t)sign_word_stride);
                                int diff_cnt = 0;
                                if (W == 4) {
                                    const uint32_t qw0 = s_dgs_qwords[warp_id][0];
                                    const uint32_t qw1 = s_dgs_qwords[warp_id][1];
                                    const uint32_t qw2 = s_dgs_qwords[warp_id][2];
                                    const uint32_t qw3 = s_dgs_qwords[warp_id][3];
                                    const uint4 bits = *reinterpret_cast<const uint4*>(uv_bits);
                                    diff_cnt = __popc(bits.x ^ qw0) +
                                               __popc(bits.y ^ qw1) +
                                               __popc(bits.z ^ qw2) +
                                               __popc(bits.w ^ qw3);
                                } else if (W == 3) {
                                    const uint32_t qw0 = s_dgs_qwords[warp_id][0];
                                    const uint32_t qw1 = s_dgs_qwords[warp_id][1];
                                    const uint32_t qw2 = s_dgs_qwords[warp_id][2];
                                    diff_cnt = __popc(uv_bits[0] ^ qw0) +
                                               __popc(uv_bits[1] ^ qw1) +
                                               __popc(uv_bits[2] ^ qw2);
                                } else if (W == 2) {
                                    const uint32_t qw0 = s_dgs_qwords[warp_id][0];
                                    const uint32_t qw1 = s_dgs_qwords[warp_id][1];
                                    diff_cnt = __popc(uv_bits[0] ^ qw0) +
                                               __popc(uv_bits[1] ^ qw1);
                                } else if (W == 1) {
                                    const uint32_t qw0 = s_dgs_qwords[warp_id][0];
                                    diff_cnt = __popc(uv_bits[0] ^ qw0);
                                } else {
                                    #pragma unroll
                                    for (int w = 0; w < DGS_MAX_WORDS; w++) {
                                        if (w >= W) break;
                                        const uint32_t qw = s_dgs_qwords[warp_id][w];
                                        diff_cnt += __popc(uv_bits[w] ^ qw);
                                    }
                                }
                                const int match_cnt = dim - diff_cnt;
                                const int score_cnt = (DGS_SIGNBIT_INVERT != 0) ? diff_cnt : match_cnt;
                                key0 = (static_cast<uint32_t>(score_cnt) << 16) | (0xFFFFu - (uint32_t)lane_id);
                            }
                            int topn_cfg = (int)DGS_TOPN;
                            if ((int)DGS_TOPN_EARLY_ITERS > 0 && iter < (int)DGS_TOPN_EARLY_ITERS) {
                                topn_cfg = (int)DGS_TOPN_EARLY;
                            }
                            if (dgs_tree_relax_topn_active) {
                                topn_cfg = max(topn_cfg, (int)DGS_TREE_RELAX_TOPN);
                            }
                            const int topn = min(max(topn_cfg, 1), deg);
                            const int diversity_keep = min((int)DGS_DIVERSITY_KEEP, topn);
                            const int strict_topn = max(0, topn - diversity_keep);
                            const bool use_fast_topk_select =
                                (DGS_TOPK_SELECT_FAST > 0) &&
                                (diversity_keep == 0) &&
                                (strict_topn > 0) &&
                                (strict_topn <= (int)DGS_TOPK_SELECT_FAST_MAX);

                            if (use_fast_topk_select) {
                                const uint32_t selected_mask =
                                    warp_select_topk_mask_desc_u32(key0, nb_valid, strict_topn);
                                keep_sign = (((selected_mask >> lane_id) & 1u) != 0u);
                            } else {
                                uint32_t key = ~key0;
                                warp_bitonic_sort_u32_asc(key, val);
                                keep_sign = (val < npoints) && (lane_id < strict_topn);
                                if (!keep_sign && (diversity_keep > 0) && (val < npoints)) {
                                    const int rem = max(0, deg - strict_topn);
                                    if (rem > 0) {
                                        for (int s = 0; s < diversity_keep; s++) {
                                            int pick = strict_topn + (((s + 1) * rem) / (diversity_keep + 1));
                                            if (pick >= deg) pick = deg - 1;
                                            if (pick < strict_topn) pick = strict_topn;
                                            if (lane_id == pick) {
                                                keep_sign = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
#if DGS_RAW_KEEP > 0
                            if ((lane_id < deg) && (raw_nb < npoints)) {
                                const int raw_keep_n = min((int)DGS_RAW_KEEP, deg);
                                for (int s = 0; s < raw_keep_n; s++) {
                                    int pick = (((s + 1) * deg) / (raw_keep_n + 1));
                                    if (pick >= deg) pick = deg - 1;
                                    if (lane_id == pick) {
                                        keep_raw = true;
                                        break;
                                    }
                                }
                            }
#endif
                        }
#if SKIP_ATOMIC_WHEN_POOL_FULL
                        const uint32_t keep_sign_mask0 = __ballot_sync(FULL_MASK, keep_sign);
                        if (keep_sign_mask0 != 0u) {
                            const int remain_slots =
                                (int)NEIGHBOR_BUFFER_SIZE - (*((volatile int*)&s_num_neighbors));
                            if (remain_slots <= 0) {
                                keep_sign = false;
                            } else {
                                const int keep_cnt = __popc(keep_sign_mask0);
                                if (keep_cnt > remain_slots) {
                                    const uint32_t lane_mask =
                                        (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
                                    const int rank = __popc(keep_sign_mask0 & lane_mask);
                                    keep_sign = (((keep_sign_mask0 >> lane_id) & 1u) != 0u) &&
                                                (rank < remain_slots);
                                }
                            }
                        }
                        const uint32_t keep_raw_mask0 = __ballot_sync(FULL_MASK, keep_raw);
                        if (keep_raw_mask0 != 0u) {
                            const int remain_slots_raw =
                                (int)NEIGHBOR_BUFFER_SIZE - (*((volatile int*)&s_num_neighbors));
                            if (remain_slots_raw <= 0) {
                                keep_raw = false;
                            } else {
                                const int keep_cnt_raw = __popc(keep_raw_mask0);
                                if (keep_cnt_raw > remain_slots_raw) {
                                    const uint32_t lane_mask =
                                        (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
                                    const int rank_raw = __popc(keep_raw_mask0 & lane_mask);
                                    keep_raw = (((keep_raw_mask0 >> lane_id) & 1u) != 0u) &&
                                               (rank_raw < remain_slots_raw);
                                }
                            }
                        }
#endif

                        bool keep_sign_local = keep_sign;
                        bool keep_raw_local = keep_raw;
#if WARP_L1_DEDUP_ENABLE && WARP_L1_DEDUP_FEATURE_ACTIVE
                        if (keep_sign_local) {
                            keep_sign_local = (hashtable_insert_noatomic(
                                s_warp_l1_hash[warp_id], WARP_L1_HASH_BITLEN, val) != 0u);
                        }
                        if (keep_raw_local) {
                            keep_raw_local = (hashtable_insert_noatomic(
                                s_warp_l1_hash[warp_id], WARP_L1_HASH_BITLEN, raw_nb) != 0u);
                        }
#endif

                        uint32_t is_new = 0;
#if HASH_WARP_DEDUP
                        const uint32_t active = __ballot_sync(FULL_MASK, keep_sign_local);
                        if (active != 0u) {
                            const uint32_t same = __match_any_sync(active, val);
                            const int leader = __ffs(same) - 1;
                            if (keep_sign_local && lane_id == leader) {
                                is_new = hashtable_insert_adaptive(
                                    s_hash, BITLEN, val,
                                    *((volatile int*)&s_num_neighbors));
                            }
                        }
#else
                        if (keep_sign_local) {
                            is_new = hashtable_insert_adaptive(
                                s_hash, BITLEN, val,
                                *((volatile int*)&s_num_neighbors));
                        }
#endif
                        uint32_t ballot = __ballot_sync(FULL_MASK, is_new);
                        int warp_count = __popc(ballot);
                        int base_offset = 0;
                        if (lane_id == 0 && warp_count > 0) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
                            if (*((volatile int*)&s_num_neighbors) < (int)NEIGHBOR_BUFFER_SIZE) {
                                base_offset = atomicAdd(&s_num_neighbors, warp_count);
                            } else {
                                base_offset = (int)NEIGHBOR_BUFFER_SIZE;
                            }
#else
                            base_offset = atomicAdd(&s_num_neighbors, warp_count);
#endif
                        }
                        base_offset = __shfl_sync(FULL_MASK, base_offset, 0);
                        if (is_new) {
                            int my_offset = __popc(ballot & ((1u << lane_id) - 1));
                            int slot = base_offset + my_offset;
                            if (slot < (int)NEIGHBOR_BUFFER_SIZE) {
                                s_neighbor_idx[slot] = val;
#if RABITQ_KERNEL_ACTIVE && RABITQ_NEIGHBOR_APPROX_BUFFER
                                s_neighbor_approx[slot] = FLT_MAX;
#endif
                            }
                        }
#if DGS_RAW_KEEP > 0
                        if (dgs_active) {
                            uint32_t is_new_raw = 0;
#if HASH_WARP_DEDUP
                            const uint32_t active_raw = __ballot_sync(FULL_MASK, keep_raw_local);
                            if (active_raw != 0u) {
                                const uint32_t same_raw = __match_any_sync(active_raw, raw_nb);
                                const int leader_raw = __ffs(same_raw) - 1;
                                if (keep_raw_local && lane_id == leader_raw) {
                                    is_new_raw = hashtable_insert_adaptive(
                                        s_hash, BITLEN, raw_nb,
                                        *((volatile int*)&s_num_neighbors));
                                }
                            }
#else
                            if (keep_raw_local) {
                                is_new_raw = hashtable_insert_adaptive(
                                    s_hash, BITLEN, raw_nb,
                                    *((volatile int*)&s_num_neighbors));
                            }
#endif
                            const uint32_t ballot_raw = __ballot_sync(FULL_MASK, is_new_raw);
                            const int warp_count_raw = __popc(ballot_raw);
                            int base_offset_raw = 0;
                            if (lane_id == 0 && warp_count_raw > 0) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
                                if (*((volatile int*)&s_num_neighbors) < (int)NEIGHBOR_BUFFER_SIZE) {
                                    base_offset_raw = atomicAdd(&s_num_neighbors, warp_count_raw);
                                } else {
                                    base_offset_raw = (int)NEIGHBOR_BUFFER_SIZE;
                                }
#else
                                base_offset_raw = atomicAdd(&s_num_neighbors, warp_count_raw);
#endif
                            }
                            base_offset_raw = __shfl_sync(FULL_MASK, base_offset_raw, 0);
                            if (is_new_raw) {
                                int my_offset_raw = __popc(ballot_raw & ((1u << lane_id) - 1));
                                int slot_raw = base_offset_raw + my_offset_raw;
                                if (slot_raw < (int)NEIGHBOR_BUFFER_SIZE) {
                                    s_neighbor_idx[slot_raw] = raw_nb;
#if RABITQ_KERNEL_ACTIVE && RABITQ_NEIGHBOR_APPROX_BUFFER
                                    s_neighbor_approx[slot_raw] = FLT_MAX;
#endif
                                }
                            }
                        }
#endif
                    }
                }
            }
        }
#else
#if RABITQ_KERNEL_ACTIVE
        if (tid == 0 && s_rabitq_frontier_full_iter < 0 && s_frontier_size >= beam_size) {
            s_rabitq_frontier_full_iter = iter;
        }
        __syncthreads();
        if (tid == 0 && iter > 0) {
            const int nf = min(s_frontier_size, beam_size);
#if RABITQ_STALL_TRACK_MODE == 1
            DISTANCE_T curr_best = FLT_MAX;
            if (nf > 0) {
                const int kpos = min(max(K, 1), nf) - 1;
                curr_best = s_frontier_dist[kpos];
            }
#else
            const DISTANCE_T curr_best = (nf > 0) ? s_frontier_dist[0] : FLT_MAX;
#endif
            const DISTANCE_T prev_best = s_rabitq_best_dist;
            DISTANCE_T improve_need = (DISTANCE_T)RABITQ_STALL_IMPROVE_EPS_ABS;
            const DISTANCE_T rel_need =
                (DISTANCE_T)(fabsf((float)prev_best) * (float)RABITQ_STALL_IMPROVE_EPS_REL);
            if (rel_need > improve_need) improve_need = rel_need;
            const DISTANCE_T improve = prev_best - curr_best;
            if (improve > improve_need) {
                s_rabitq_best_dist = curr_best;
                s_rabitq_stall_iters = 0;
            } else {
                s_rabitq_stall_iters++;
            }
        }
        __syncthreads();
        if (tid == 0) {
            const bool can_stop = (iter >= (int)RABITQ_EARLY_STOP_WARMUP_ITERS);
            bool stop_ok = can_stop;
#if RABITQ_EARLY_STOP_REQUIRE_FULL
            stop_ok = stop_ok && (s_frontier_size >= beam_size);
#endif
            s_rabitq_break_now =
                stop_ok && (s_rabitq_stall_iters >= (int)RABITQ_EARLY_STOP_STALL_ITERS);
        }
        __syncthreads();
        if (s_rabitq_break_now) {
            break;
        }
        int rabitq_cool_start =
            (int)((1.0f - (float)RABITQ_COOL_RATIO) * (float)RABITQ_EST_ITERS);
#if RABITQ_ACTIVE_ITERS > 0
        rabitq_cool_start = min((int)RABITQ_ACTIVE_ITERS, (int)RABITQ_EST_ITERS);
#endif
#if RABITQ_GPU_DUALPOOL_ENGINE
        // Keep unified dual-pool semantics active through the full search loop.
        rabitq_cool_start = (int)RABITQ_MAX_ITER_CAP;
#endif
        const bool rabitq_static_ready =
            (d_sign_bit != nullptr) &&
            (sign_word_stride >= rabitq_sign_words_required) &&
#if RABITQ_USE_FACTORS
            (d_rabitq_triple_x != nullptr) &&
#if RABITQ_USE_PACKED_DQVQ
#if RABITQ_USE_FP16_DQVQ
            (d_rabitq_factor_dqvq_h2 != nullptr) &&
#else
            (d_rabitq_factor_dqvq != nullptr) &&
#endif
#else
            (d_rabitq_factor_dq != nullptr) &&
            (d_rabitq_factor_vq != nullptr) &&
#endif
#endif
            (RABITQ_TOPN > 0) &&
            (RABITQ_TOPN < 32) &&
            (rabitq_sign_words_required > 0) &&
            (rabitq_sign_words_required <= (int)RABITQ_MAX_WORDS) &&
            (rabitq_score_words > 0);
        bool rabitq_filter_gate =
            rabitq_static_ready &&
            (BEAM_SIZE >= (int)RABITQ_MIN_BEAM) &&
            (iter >= (int)RABITQ_WARMUP_ITERS) &&
            (iter < rabitq_cool_start);
#if RABITQ_DISABLE_IF_NO_PRUNE_ITERS > 0
        rabitq_filter_gate = rabitq_filter_gate && (s_rabitq_force_off == 0);
#endif
#if (RABITQ_DISABLE_ON_STALL_ITERS > 0) || \
    ((RABITQ_DISABLE_ON_STALL_ITERS_LOW_BEAM > 0) && (RABITQ_DISABLE_ON_STALL_LOW_BEAM_MAX > 0))
        int rabitq_disable_stall_iters = (int)RABITQ_DISABLE_ON_STALL_ITERS;
#if (RABITQ_DISABLE_ON_STALL_ITERS_LOW_BEAM > 0) && (RABITQ_DISABLE_ON_STALL_LOW_BEAM_MAX > 0)
        if (BEAM_SIZE <= (int)RABITQ_DISABLE_ON_STALL_LOW_BEAM_MAX) {
            rabitq_disable_stall_iters = max(
                rabitq_disable_stall_iters,
                (int)RABITQ_DISABLE_ON_STALL_ITERS_LOW_BEAM);
        }
#endif
        if ((rabitq_disable_stall_iters > 0) &&
            (s_rabitq_stall_iters >= rabitq_disable_stall_iters)) {
            rabitq_filter_gate = false;
        }
#endif
#if RABITQ_REQUIRE_STALL
        rabitq_filter_gate =
            rabitq_filter_gate && (s_rabitq_stall_iters >= (int)RABITQ_STALL_TRIGGER);
#endif
        int rabitq_topn_iter = (int)RABITQ_TOPN;
        int rabitq_early_ref_iter = iter;
        if ((int)RABITQ_DEFER_PRUNE_UNTIL_FRONTIER_FULL != 0) {
            rabitq_early_ref_iter =
                (s_rabitq_frontier_full_iter >= 0) ? (iter - s_rabitq_frontier_full_iter) : -1;
        }
        if ((int)RABITQ_TOPN_EARLY_ITERS > 0 &&
            rabitq_early_ref_iter >= 0 &&
            rabitq_early_ref_iter < (int)RABITQ_TOPN_EARLY_ITERS) {
            rabitq_topn_iter = max(rabitq_topn_iter, (int)RABITQ_TOPN_EARLY);
        }
#if RABITQ_NEIGHBOR_BUDGET > 0
        int rabitq_neighbor_budget_cfg = (int)RABITQ_NEIGHBOR_BUDGET;
#if (RABITQ_NEIGHBOR_BUDGET_LOW > 0) && (RABITQ_NEIGHBOR_BUDGET_HIGH > 0) && \
    (RABITQ_NEIGHBOR_BUDGET_SPLIT_BEAM > 0)
        rabitq_neighbor_budget_cfg =
            (BEAM_SIZE <= (int)RABITQ_NEIGHBOR_BUDGET_SPLIT_BEAM)
                ? (int)RABITQ_NEIGHBOR_BUDGET_LOW
                : (int)RABITQ_NEIGHBOR_BUDGET_HIGH;
#endif
        const int rabitq_neighbor_budget =
            min((int)NEIGHBOR_BUFFER_SIZE, max(rabitq_neighbor_budget_cfg, 1));
#endif
#if SQG_HAMMING_FAST4
        const bool rabitq_fast4 = (rabitq_score_words > 0) && (rabitq_score_words <= 4);
        uint32_t rabitq_q0 = 0u, rabitq_q1 = 0u, rabitq_q2 = 0u, rabitq_q3 = 0u;
        if (rabitq_fast4) {
            const uint32_t* q_bits = s_rabitq_qwords;
            rabitq_q0 = q_bits[0];
            if (rabitq_score_words > 1) rabitq_q1 = q_bits[1];
            if (rabitq_score_words > 2) rabitq_q2 = q_bits[2];
            if (rabitq_score_words > 3) rabitq_q3 = q_bits[3];
        }
#endif
#endif
#if SQG_KERNEL_ACTIVE
        int sqg_cool_start = (int)((1.0f - (float)SQG_COOL_RATIO) * (float)SQG_EST_ITERS);
#if SQG_ACTIVE_ITERS > 0
        sqg_cool_start = min((int)SQG_ACTIVE_ITERS, (int)SQG_EST_ITERS);
#endif
        const bool sqg_static_ready =
            (d_sqg_sign_bit != nullptr) &&
            (SQG_TOPN > 0) &&
            (SQG_TOPN < 32) &&
            (sqg_dims > 0) &&
            (sqg_words > 0) &&
            (sqg_words <= (int)SQG_MAX_WORDS);
        const bool sqg_filter_gate =
            sqg_static_ready &&
            (BEAM_SIZE >= (int)SQG_MIN_BEAM) &&
            (iter >= (int)SQG_WARMUP_ITERS) &&
            (iter < sqg_cool_start);
        bool sqg_filter_gate_runtime = sqg_filter_gate;
#if SQG_FULLSCAN_STRIDE > 0
        if (iter > 0 && (iter % (int)SQG_FULLSCAN_STRIDE) == 0) {
            sqg_filter_gate_runtime = false;
        }
#endif
        int sqg_topn_iter = (int)SQG_TOPN;
        if ((int)SQG_TOPN_EARLY_ITERS > 0 && iter < (int)SQG_TOPN_EARLY_ITERS) {
            sqg_topn_iter = max(sqg_topn_iter, (int)SQG_TOPN_EARLY);
        }
#if SQG_NEIGHBOR_BUDGET_ENABLE
        const int sqg_neighbor_budget =
            min((int)NEIGHBOR_BUFFER_SIZE, max((int)SQG_NEIGHBOR_BUDGET, 1));
#endif
#if SQG_HAMMING_FAST4
        const bool sqg_fast4 = (sqg_words > 0) && (sqg_words <= 4);
        uint32_t sqg_q0 = 0u, sqg_q1 = 0u, sqg_q2 = 0u, sqg_q3 = 0u;
        if (sqg_fast4) {
            const uint32_t* q_bits = s_sqg_qwords[warp_id];
            sqg_q0 = q_bits[0];
            if (sqg_words > 1) sqg_q1 = q_bits[1];
            if (sqg_words > 2) sqg_q2 = q_bits[2];
            if (sqg_words > 3) sqg_q3 = q_bits[3];
        }
#endif
#endif
        const bool frontier_full_now = (s_frontier_size >= beam_size);
        const int frontier_size_now = min(s_frontier_size, beam_size);
#if RABITQ_KERNEL_ACTIVE && (RABITQ_NEIGHBOR_BUDGET > 0)
        const bool rabitq_budget_gate =
            rabitq_filter_gate &&
            (((int)RABITQ_NEIGHBOR_BUDGET_REQUIRE_FRONTIER_FULL == 0) || frontier_full_now);
#endif
        for (int e = warp_id; e < s_num_expanders; e += num_warps) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
            if (warp_read_volatile_shared_i32((volatile int*)&s_num_neighbors) >=
                (int)NEIGHBOR_BUFFER_SIZE) {
                break;
            }
#endif
            INDEX_T expand_pos = s_expand_pos[e];
            if (expand_pos >= beam_size) continue;
            vid_t current = s_frontier_idx[expand_pos] & ~kIndexMsbMask;
            if (current >= npoints) continue;
            DISTANCE_T rabitq_expand_sqr_y = s_frontier_dist[expand_pos];
#if RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_PARITY_MODE && \
    RABITQ_CPU_PARITY_EXACT_POOL
            {
                if (rabitq_filter_gate) {
                    // CPU parity: exact distance is computed at pop time.
                    // Keep parity stream aligned to current expansion state.
                    const T* cur_vec = d_data + current * dim;
                    rabitq_expand_sqr_y =
                        (dim == 128)
                            ? warp_l2_distance_128((const float*)query, (const float*)cur_vec)
                            : warp_l2_distance(dim, query, cur_vec);
                    if (lane_id == 0) {
                        atomicAdd(&s_dc_count, 1);
                        const int slot = atomicAdd(&s_exact_stream_size, 1);
                        if (slot < (int)RABITQ_CPU_PARITY_STREAM_CAP) {
                            s_exact_stream_idx[slot] = current;
                            s_exact_stream_dist[slot] = rabitq_expand_sqr_y;
                        }
                    }
                }
            }
#else
#if RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW
            // CPU-aligned semantics: sqr_y in RabitQ formula must be exact L2(q, current).
            if (rabitq_filter_gate &&
                ((int)RABITQ_CPU_ALIGN_EXACT_EXPANDER_SQRY != 0)) {
                const T* cur_vec = d_data + current * dim;
                rabitq_expand_sqr_y =
                    (dim == 128)
                        ? warp_l2_distance_128((const float*)query, (const float*)cur_vec)
                        : warp_l2_distance(dim, query, cur_vec);
                if (lane_id == 0) {
                    atomicAdd(&s_dc_count, 1);
                }
            }
#endif
#endif
            const vid_t* neighbors = gg.N(current);
            for (int base = 0; base < max_degree; base += 32) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
                if (warp_read_volatile_shared_i32((volatile int*)&s_num_neighbors) >=
                    (int)NEIGHBOR_BUFFER_SIZE) {
                    break;
                }
#endif
                const int deg = min(32, max_degree - base);
#if RABITQ_KERNEL_ACTIVE
                const int rabitq_topn_deg = min(max(rabitq_topn_iter, 1), deg);
                const bool rabitq_dualpool_engine = ((int)RABITQ_GPU_DUALPOOL_ENGINE != 0);
                const bool rabitq_cpu_align_fullscan =
                    ((int)RABITQ_CPU_ALIGN_FLOW != 0) || rabitq_dualpool_engine;
                const bool rabitq_prune_stage_ready =
                    (rabitq_dualpool_engine ||
                     ((int)RABITQ_DEFER_PRUNE_UNTIL_FRONTIER_FULL == 0) ||
                     rabitq_cpu_align_fullscan || frontier_full_now);
                const bool rabitq_active_base =
                    rabitq_filter_gate &&
                    rabitq_prune_stage_ready &&
                    (rabitq_dualpool_engine ||
                     (((int)RABITQ_REQUIRE_FRONTIER_FULL == 0) || frontier_full_now)) &&
                    (((int)RABITQ_MIN_EXPANDERS <= 0) ||
                     (s_num_expanders >= (int)RABITQ_MIN_EXPANDERS)) &&
                    (deg > 1) &&
                    (rabitq_cpu_align_fullscan || (rabitq_topn_deg < deg));
#endif
                const int n = base + lane_id;
                vid_t neighbor = (lane_id < deg) ? neighbors[n] : UINT_MAX;
                const bool valid = (lane_id < deg) && (neighbor < npoints);
                bool keep = valid;
                bool keep_decided = false;
                DISTANCE_T rabitq_neighbor_approx = FLT_MAX;
#if RABITQ_KERNEL_ACTIVE && RABITQ_REFLOW_FIXED && RABITQ_GPU_DUALPOOL_ENGINE && \
    (RABITQ_GPU_DUALPOOL_ANCHOR_EXACT >= 32)
                // Current fixed p7 path on degree-32 graphs exact-evaluates the
                // whole staged tile, so skip the RabitQ score/select frontend.
                if (rabitq_filter_gate && (max_degree == 32)) {
                    keep = valid;
                    keep_decided = true;
                } else
#endif
#if RABITQ_KERNEL_ACTIVE
                if (rabitq_filter_gate) {
                bool rabitq_active = rabitq_active_base;
                int rabitq_staged_now = 0;
#if (RABITQ_NEIGHBOR_BUDGET > 0) || (RABITQ_SKIP_NEAR_FULL_REMAIN > 0) || \
    SKIP_ATOMIC_WHEN_POOL_FULL
                if (rabitq_active) {
                    rabitq_staged_now =
                        warp_read_volatile_shared_i32((volatile int*)&s_num_neighbors);
                }
#endif
#if RABITQ_NEIGHBOR_BUDGET > 0
                // Implementation optimization: skip RabitQ scoring/select path early
                // when neighbor budget gate is already full.
                if (rabitq_active && rabitq_budget_gate &&
                    (rabitq_staged_now >= rabitq_neighbor_budget)) {
                    rabitq_active = false;
                }
#endif
#if RABITQ_SKIP_NEAR_FULL_REMAIN > 0
                if (rabitq_active &&
                    ((float)RABITQ_APPROX_CUTOFF_RATIO <= 0.0f)) {
                    const int near_full_threshold =
                        max((int)NEIGHBOR_BUFFER_SIZE - (int)RABITQ_SKIP_NEAR_FULL_REMAIN, 0);
                    if (rabitq_staged_now >= near_full_threshold) {
                        rabitq_active = false;
                    }
                }
#endif
                if (rabitq_active) {
                    int topn = rabitq_topn_deg;
#if RABITQ_GPU_DUALPOOL_ENGINE
                    // CPU-aligned dual-pool: keep all tile neighbors, rely on
                    // approximate pool merge/truncation for global gating.
                    topn = deg;
#if RABITQ_GPU_DUALPOOL_ANCHOR_EXACT >= 32
                    // Performance fast path:
                    // when per-pop anchor budget already covers the whole degree-32 tile,
                    // this iteration will exact-evaluate all staged neighbors.
                    // Skip expensive RabitQ score/select and keep all valid lanes directly.
                    keep = valid;
                    keep_decided = true;
                    rabitq_neighbor_approx = FLT_MAX;
                    rabitq_active = false;
#endif
#endif
#if RABITQ_ADAPTIVE_TOPN_BOOST > 0
                    int rabitq_adaptive_stall_iters = (int)RABITQ_ADAPTIVE_TOPN_STALL_ITERS;
#if (RABITQ_ADAPTIVE_TOPN_STALL_ITERS_LOW_BEAM > 0) && \
    (RABITQ_ADAPTIVE_TOPN_LOW_BEAM_MAX > 0)
                    if (BEAM_SIZE <= (int)RABITQ_ADAPTIVE_TOPN_LOW_BEAM_MAX) {
                        rabitq_adaptive_stall_iters = max(
                            rabitq_adaptive_stall_iters,
                            (int)RABITQ_ADAPTIVE_TOPN_STALL_ITERS_LOW_BEAM);
                    }
#endif
                    if (s_rabitq_stall_iters < rabitq_adaptive_stall_iters) {
                        topn = min(deg, topn + (int)RABITQ_ADAPTIVE_TOPN_BOOST);
                    }
#endif
#if RABITQ_CPU_ALIGN_FLOW
                    // Optional CPU-align compatibility mode: force tile fullscan.
                    // Disabled by default so topn can actually prune at tile stage.
                    if (((int)RABITQ_CPU_ALIGN_TILE_FULLSCAN != 0) &&
                        ((int)RABITQ_CPU_ALIGN_FORCE_EXACT_DIST == 0)) {
                        topn = deg;
                    }
#endif
#if SKIP_ATOMIC_WHEN_POOL_FULL
                    // When the staging buffer is almost full, clip RabitQ keep-count to
                    // remaining slots so we do less shortlist/hash work and avoid late
                    // lane-order truncation.
                    const int remain_slots =
                        max((int)NEIGHBOR_BUFFER_SIZE - rabitq_staged_now, 0);
                    if (remain_slots <= 0) {
                        rabitq_active = false;
                    } else {
                        topn = min(topn, remain_slots);
                    }
#endif
#if (RABITQ_GLOBAL_APPROX_PRESELECT == 0) && (RABITQ_TWO_PHASE_EXACT == 0)
                    if (rabitq_active && (topn >= deg) &&
                        ((int)RABITQ_GPU_DUALPOOL_ENGINE == 0)) {
                        // No effective tile pruning: skip heavy RabitQ scoring/select path.
                        // Keep semantics identical to full-keep while avoiding wasted compute.
                        keep = valid;
                        keep_decided = true;
                        rabitq_active = false;
                    }
#endif
#if RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_ALIGN_SKIP_TRIVIAL_SCORE
                    if (rabitq_active &&
                        (topn >= deg)) {
                        // No tile-level RabitQ pruning leverage: keep all valid neighbors,
                        // but still provide a cheap approx signal for later exact-pool shortlist.
                        if (valid) {
                            const size_t trivial_edge_pos =
                                (size_t)current * (size_t)max_degree + (size_t)n;
                            const uint32_t* v_bits_trivial =
                                d_sign_bit + trivial_edge_pos * (size_t)sign_word_stride;
                            const uint32_t* q_bits_trivial = s_rabitq_qwords;
                            const int trivial_words =
                                min(max((int)RABITQ_CPU_ALIGN_TRIVIAL_APPROX_WORDS, 1),
                                    rabitq_score_words);
                            int diff_cnt = 0;
                            for (int w = 0; w < trivial_words; ++w) {
                                uint32_t vb = v_bits_trivial[w];
                                uint32_t qb = q_bits_trivial[w];
                                if ((w == (trivial_words - 1)) &&
                                    (trivial_words == rabitq_score_words) &&
                                    (rabitq_score_tail_bits != 0)) {
                                    vb &= rabitq_score_tail_mask;
                                    qb &= rabitq_score_tail_mask;
                                }
                                diff_cnt += __popc(vb ^ qb);
                            }
                            const int trivial_dims =
                                (trivial_words == rabitq_score_words)
                                    ? rabitq_score_dims
                                    : (trivial_words << 5);
                            const int trivial_score_i = trivial_dims - (diff_cnt << 1);
                            float approx =
                                (float)rabitq_expand_sqr_y -
                                ((float)RABITQ_SCORE_ALPHA * (float)trivial_score_i);
                            if (!(approx > 0.0f) || (approx != approx) || (approx >= FLT_MAX)) {
                                approx = FLT_MAX * 0.5f;
                            }
                            rabitq_neighbor_approx = (DISTANCE_T)approx;
                        }
                        keep = valid;
                        keep_decided = true;
                        rabitq_active = false;
                    }
#endif
#if (RABITQ_SCORE_MODE == 1) || (RABITQ_FASTSCAN_ENABLE != 0)
                    if (rabitq_active && (s_rabitq_q6_ready != 1)) {
                        if (lane_id == 0) {
                            const int q6_prev = atomicCAS(&s_rabitq_q6_ready, 0, 2);
                            if (q6_prev == 0) {
                                rabitq_prepare_q6_lut(
                                    (const float*)query,
#if RABITQ_QUERY_PROJ_ENABLE
                                    query_proj_precomputed,
                                    d_rabitq_proj,
#endif
                                    rabitq_q6_dims,
                                    rabitq_q6_words,
#if RABITQ_QUERY_PROJ_ENABLE
                                    &s_rabitq_qproj[0],
#endif
                                    &s_rabitq_q6_lut[0][0][0],
#if RABITQ_PACK_LUT_ACTIVE
                                    &s_rabitq_pack_lut[0][0],
#endif
                                    &s_rabitq_q6_sum,
                                    &s_rabitq_q6_lower,
                                    &s_rabitq_q6_width);
                                __threadfence_block();
                                s_rabitq_q6_ready = 1;
                            }
                        }
                        __syncwarp();
                        while (*((volatile int*)&s_rabitq_q6_ready) != 1) {
                        }
                        __syncwarp();
                    }
#endif
                if (rabitq_active) {
#if RABITQ_REFLOW_FIXED
                    const size_t edge_pos =
                        (size_t)current * (size_t)max_degree + (size_t)n;
#if RABITQ_PACKED_CODE_ENABLE && \
    (((RABITQ_FASTSCAN_ENABLE != 0)) || \
     ((RABITQ_SCORE_MODE == 1) && (RABITQ_SCORE1_PACKED_FASTSCAN_ENABLE != 0)))
                    const bool rabitq_packed_fastscan =
                        (d_rabitq_packed_codes != nullptr) &&
                        (rabitq_packed_code_bytes_per_edge == 16) &&
                        (max_degree == 32) &&
                        (deg == 32) &&
                        (base == 0) &&
                        (rabitq_q6_dims == 128);
#else
                    const bool rabitq_packed_fastscan = false;
#endif
                    const uint32_t rabitq_valid_mask =
                        __ballot_sync(FULL_MASK, valid);
                    (void)rabitq_valid_mask;
                    const uint8_t* rabitq_code_block = nullptr;
                    if (rabitq_packed_fastscan) {
                        rabitq_code_block =
                            d_rabitq_packed_codes +
                            ((size_t)current * (size_t)max_degree + (size_t)base) *
                                (size_t)rabitq_packed_code_bytes_per_edge;
                    }

                    if (valid) {
                        const uint32_t* v_bits =
                            d_sign_bit + edge_pos * (size_t)sign_word_stride;
                        int rabitq_score_i = -0x3fffffff;

#if (RABITQ_FASTSCAN_ENABLE != 0)
                        int set_sum = 0;
#if RABITQ_PACK_LUT_ACTIVE
                        if (rabitq_packed_fastscan) {
                            const int lane_pair = lane_id & 15;
                            const int idx = ((lane_pair & 7) << 1) | (lane_pair >> 3);
                            const bool upper = ((lane_id & 16) != 0);
                            #pragma unroll
                            for (int p = 0; p < 16; ++p) {
                                const int off = p << 5;
                                const uint8_t b0 =
                                    __ldg(rabitq_code_block + off + idx);
                                const uint8_t b1 =
                                    __ldg(rabitq_code_block + off + 16 + idx);
                                const uint8_t nib0 =
                                    upper ? (b0 >> 4) : (b0 & 0xFu);
                                const uint8_t nib1 =
                                    upper ? (b1 >> 4) : (b1 & 0xFu);
                                set_sum +=
                                    (int)s_rabitq_pack_lut[(p << 1) + 0][nib0];
                                set_sum +=
                                    (int)s_rabitq_pack_lut[(p << 1) + 1][nib1];
                            }
                        } else
#endif
#if RABITQ_SCORE_WORDS == 1
                        {
                            uint32_t vw = v_bits[0];
                            if (rabitq_score_tail_bits != 0) {
                                vw &= rabitq_score_tail_mask;
                            }
                            set_sum += (int)s_rabitq_q6_lut[0][0][(vw >> 0) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][1][(vw >> 4) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][2][(vw >> 8) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][3][(vw >> 12) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][4][(vw >> 16) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][5][(vw >> 20) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][6][(vw >> 24) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][7][(vw >> 28) & 0xFu];
                        }
#elif RABITQ_SCORE_WORDS == 4
                        {
                            const uint32_t vw0 = v_bits[0];
                            const uint32_t vw1 = v_bits[1];
                            const uint32_t vw2 = v_bits[2];
                            uint32_t vw3 = v_bits[3];
                            if (rabitq_score_tail_bits != 0) {
                                vw3 &= rabitq_score_tail_mask;
                            }
                            set_sum += (int)s_rabitq_q6_lut[0][0][(vw0 >> 0) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][1][(vw0 >> 4) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][2][(vw0 >> 8) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][3][(vw0 >> 12) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][4][(vw0 >> 16) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][5][(vw0 >> 20) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][6][(vw0 >> 24) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][7][(vw0 >> 28) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[1][0][(vw1 >> 0) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[1][1][(vw1 >> 4) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[1][2][(vw1 >> 8) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[1][3][(vw1 >> 12) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[1][4][(vw1 >> 16) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[1][5][(vw1 >> 20) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[1][6][(vw1 >> 24) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[1][7][(vw1 >> 28) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[2][0][(vw2 >> 0) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[2][1][(vw2 >> 4) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[2][2][(vw2 >> 8) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[2][3][(vw2 >> 12) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[2][4][(vw2 >> 16) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[2][5][(vw2 >> 20) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[2][6][(vw2 >> 24) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[2][7][(vw2 >> 28) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[3][0][(vw3 >> 0) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[3][1][(vw3 >> 4) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[3][2][(vw3 >> 8) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[3][3][(vw3 >> 12) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[3][4][(vw3 >> 16) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[3][5][(vw3 >> 20) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[3][6][(vw3 >> 24) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[3][7][(vw3 >> 28) & 0xFu];
                        }
#else
                        {
                            for (int w = 0; w < rabitq_score_words; ++w) {
                                uint32_t vw = v_bits[w];
                                if ((w == rabitq_score_words - 1) &&
                                    (rabitq_score_tail_bits != 0)) {
                                    vw &= rabitq_score_tail_mask;
                                }
                                set_sum +=
                                    (int)s_rabitq_q6_lut[w][0][(vw >> 0) & 0xFu];
                                set_sum +=
                                    (int)s_rabitq_q6_lut[w][1][(vw >> 4) & 0xFu];
                                set_sum +=
                                    (int)s_rabitq_q6_lut[w][2][(vw >> 8) & 0xFu];
                                set_sum +=
                                    (int)s_rabitq_q6_lut[w][3][(vw >> 12) & 0xFu];
                                set_sum +=
                                    (int)s_rabitq_q6_lut[w][4][(vw >> 16) & 0xFu];
                                set_sum +=
                                    (int)s_rabitq_q6_lut[w][5][(vw >> 20) & 0xFu];
                                set_sum +=
                                    (int)s_rabitq_q6_lut[w][6][(vw >> 24) & 0xFu];
                                set_sum +=
                                    (int)s_rabitq_q6_lut[w][7][(vw >> 28) & 0xFu];
                            }
                        }
#endif
                        rabitq_score_i = (set_sum << 1) - s_rabitq_q6_sum;
#else
                        int diff_cnt = 0;
#if RABITQ_HAMMING_FAST4
                        if (rabitq_fast4) {
#if RABITQ_SCORE_WORDS == 1
                            uint32_t x0 = v_bits[0] ^ rabitq_q0;
                            if (rabitq_score_tail_bits != 0) {
                                x0 &= rabitq_score_tail_mask;
                            }
                            diff_cnt = __popc(x0);
#elif RABITQ_SCORE_WORDS == 2
                            uint32_t x0 = v_bits[0] ^ rabitq_q0;
                            uint32_t x1 = v_bits[1] ^ rabitq_q1;
                            if (rabitq_score_tail_bits != 0) {
                                x1 &= rabitq_score_tail_mask;
                            }
                            diff_cnt = __popc(x0) + __popc(x1);
#elif RABITQ_SCORE_WORDS == 3
                            uint32_t x0 = v_bits[0] ^ rabitq_q0;
                            uint32_t x1 = v_bits[1] ^ rabitq_q1;
                            uint32_t x2 = v_bits[2] ^ rabitq_q2;
                            if (rabitq_score_tail_bits != 0) {
                                x2 &= rabitq_score_tail_mask;
                            }
                            diff_cnt = __popc(x0) + __popc(x1) + __popc(x2);
#elif RABITQ_SCORE_WORDS == 4
                            const uint32_t x0 = v_bits[0] ^ rabitq_q0;
                            const uint32_t x1 = v_bits[1] ^ rabitq_q1;
                            const uint32_t x2 = v_bits[2] ^ rabitq_q2;
                            uint32_t x3 = v_bits[3] ^ rabitq_q3;
                            if (rabitq_score_tail_bits != 0) {
                                x3 &= rabitq_score_tail_mask;
                            }
                            diff_cnt =
                                __popc(x0) + __popc(x1) + __popc(x2) + __popc(x3);
#else
                            const uint32_t* q_bits = s_rabitq_qwords;
                            for (int w = 0; w < rabitq_score_words; ++w) {
                                if ((w == rabitq_score_words - 1) &&
                                    (rabitq_score_tail_bits != 0)) {
                                    const uint32_t vb =
                                        v_bits[w] & rabitq_score_tail_mask;
                                    const uint32_t qb =
                                        q_bits[w] & rabitq_score_tail_mask;
                                    diff_cnt += __popc(vb ^ qb);
                                } else {
                                    diff_cnt += __popc(v_bits[w] ^ q_bits[w]);
                                }
                            }
#endif
                        } else
#endif
                        {
                            const uint32_t* q_bits = s_rabitq_qwords;
                            for (int w = 0; w < rabitq_score_words; ++w) {
                                if ((w == rabitq_score_words - 1) &&
                                    (rabitq_score_tail_bits != 0)) {
                                    const uint32_t vb =
                                        v_bits[w] & rabitq_score_tail_mask;
                                    const uint32_t qb =
                                        q_bits[w] & rabitq_score_tail_mask;
                                    diff_cnt += __popc(vb ^ qb);
                                } else {
                                    diff_cnt += __popc(v_bits[w] ^ q_bits[w]);
                                }
                            }
                        }
                        rabitq_score_i = rabitq_score_dims - (diff_cnt << 1);
#endif

                        float approx = (float)rabitq_expand_sqr_y -
                                       ((float)RABITQ_SCORE_ALPHA *
                                        (float)rabitq_score_i);
                        if (!(approx > 0.0f) || (approx != approx) ||
                            (approx >= FLT_MAX)) {
                            approx = FLT_MAX * 0.5f;
                        }
                        rabitq_neighbor_approx = (DISTANCE_T)approx;

#if RABITQ_GPU_DUALPOOL_ENGINE
                        // Fixed dual-pool keeps full tile; cutoff is applied in
                        // candidate merge after exact/approx staging.
                        keep = true;
#else
                        const int score_bias = rabitq_score_i + 4096;
                        uint32_t key_hi = (uint32_t)score_bias;
#if RABITQ_INT_KEY_RESPECT_POS_CLAMP
                        bool key_pos_ok = true;
                        if ((float)RABITQ_SCORE_ALPHA == 1.0f) {
                            const int sqr_y_ceil =
                                __float2int_ru((float)rabitq_expand_sqr_y);
                            key_pos_ok = (rabitq_score_i < sqr_y_ceil);
                        } else {
                            key_pos_ok =
                                ((float)rabitq_expand_sqr_y -
                                 ((float)RABITQ_SCORE_ALPHA *
                                  (float)rabitq_score_i)) > 0.0f;
                        }
                        if (!key_pos_ok) {
                            key_hi = 0u;
                        }
#endif
#if RABITQ_INT_KEY_TIEBREAK
                        const uint32_t key_rabitq =
                            (key_hi << 16) | (0xFFFFu - (uint32_t)lane_id);
#else
                        const uint32_t key_rabitq = (key_hi << 16);
#endif
                        const uint32_t selected_mask =
                            warp_select_topk_mask_desc_u32_rabitq(
                                key_rabitq, valid, topn);
                        keep = (((selected_mask >> lane_id) & 1u) != 0u);
#endif
                    } else {
                        keep = false;
                    }
                    keep_decided = true;
#else
                    const size_t edge_pos = (size_t)current * (size_t)max_degree + (size_t)n;
#if RABITQ_PACKED_CODE_ENABLE && \
    (((RABITQ_FASTSCAN_ENABLE != 0)) || \
     ((RABITQ_SCORE_MODE == 1) && (RABITQ_SCORE1_PACKED_FASTSCAN_ENABLE != 0)))
                    const bool rabitq_packed_fastscan =
                        (d_rabitq_packed_codes != nullptr) &&
                        (rabitq_packed_code_bytes_per_edge == 16) &&
                        (max_degree == 32) &&
                        (deg == 32) &&
                        (base == 0) &&
                        (rabitq_q6_dims == 128);
#else
                    const bool rabitq_packed_fastscan = false;
#endif
                    const uint32_t rabitq_valid_mask =
                        __ballot_sync(FULL_MASK, valid);
                    const bool rabitq_packed_pair_shuffle =
                        rabitq_packed_fastscan && (rabitq_valid_mask == FULL_MASK);
                    const uint8_t* rabitq_code_block = nullptr;
                    if (rabitq_packed_fastscan) {
                        rabitq_code_block =
                            d_rabitq_packed_codes +
                            ((size_t)current * (size_t)max_degree + (size_t)base) *
                                (size_t)rabitq_packed_code_bytes_per_edge;
                    }
#if RABITQ_EDGE_TILE_CACHE_WORDS > 0
                    const bool rabitq_tile_cache_on =
                        (sign_word_stride >= (int)RABITQ_EDGE_TILE_CACHE_WORDS);
                    if (rabitq_tile_cache_on && (lane_id < deg)) {
                        if (valid) {
                            const uint32_t* v_src =
                                d_sign_bit + edge_pos * (size_t)sign_word_stride;
                            #pragma unroll
                            for (int w = 0; w < (int)RABITQ_EDGE_TILE_CACHE_WORDS; ++w) {
                                s_rabitq_tile_words[lane_id][w] = v_src[w];
                            }
                        } else {
                            #pragma unroll
                            for (int w = 0; w < (int)RABITQ_EDGE_TILE_CACHE_WORDS; ++w) {
                                s_rabitq_tile_words[lane_id][w] = 0u;
                            }
                        }
                    }
                    __syncwarp();
#endif
                    uint32_t key_rabitq = 0u;
                    int rabitq_score_i = -0x3fffffff;
                    bool need_approx_score = false;
                    bool in_factor_pre = false;
                    bool rabitq_preonly_keep = false;
                    bool in_coarse_pre = valid;
#if RABITQ_USE_FACTORS
                    bool factor_pre_ready = false;
                    int factor_pre_keep = 0;
                    uint32_t factor_key_pre = 0u;
#endif
#if (RABITQ_SCORE_MODE == 0) && !RABITQ_USE_FACTORS
                    int rabitq_pre_diff_cnt = 0;
                    int rabitq_pre_words = 0;
#endif
                    if (valid) {
#if RABITQ_EDGE_TILE_CACHE_WORDS > 0
                        const uint32_t* v_bits =
                            (rabitq_tile_cache_on &&
                             ((int)RABITQ_EDGE_TILE_CACHE_WORDS >= rabitq_score_words))
                                ? &s_rabitq_tile_words[lane_id][0]
                                : (d_sign_bit + edge_pos * (size_t)sign_word_stride);
#else
                        const uint32_t* v_bits =
                            d_sign_bit + edge_pos * (size_t)sign_word_stride;
#endif
#if RABITQ_USE_FACTORS && RABITQ_FACTOR_LIGHT_PRESELECT
                        if ((topn < deg) && (rabitq_score_words > 0)) {
                            const int light_words =
                                min(max((int)RABITQ_FACTOR_LIGHT_PRESELECT_WORDS, 1),
                                    rabitq_score_words);
                            int light_diff_cnt = 0;
                            const uint32_t* q_bits0 = s_rabitq_qwords;
                            for (int w = 0; w < light_words; ++w) {
                                uint32_t vb = v_bits[w];
                                uint32_t qb = q_bits0[w];
                                if ((w == (light_words - 1)) &&
                                    (light_words == rabitq_score_words) &&
                                    (rabitq_score_tail_bits != 0)) {
                                    vb &= rabitq_score_tail_mask;
                                    qb &= rabitq_score_tail_mask;
                                }
                                light_diff_cnt += __popc(vb ^ qb);
                            }
                            const int light_dims =
                                (light_words == rabitq_score_words)
                                    ? rabitq_score_dims
                                    : (light_words << 5);
                            const int light_score_i = light_dims - (light_diff_cnt << 1);
                            factor_pre_keep =
                                min(deg, topn + max((int)RABITQ_FACTOR_PRESELECT_EXTRA, 0));
                            const int score_bias = light_score_i + 4096;
                            factor_key_pre =
                                ((uint32_t)score_bias << 16) |
                                (0xFFFFu - (uint32_t)lane_id);
                            const uint32_t pre_mask =
                                warp_select_topk_mask_desc_u32_rabitq(
                                    factor_key_pre, valid, factor_pre_keep);
                            in_factor_pre =
                                valid && (((pre_mask >> lane_id) & 1u) != 0u);
                            factor_pre_ready = true;
                        }
#endif
#if RABITQ_COARSE_PREFILTER && (RABITQ_SCORE_MODE == 0) && !RABITQ_USE_FACTORS
                        if (!rabitq_packed_fastscan) {
                        const int coarse_words =
                            min(max((int)RABITQ_COARSE_WORDS, 1), rabitq_score_words);
                        const int coarse_keep =
                            min(deg, topn + max((int)RABITQ_COARSE_EXTRA_KEEP, 0));
                        if ((topn < deg) &&
                            (coarse_keep < deg) &&
                            (coarse_words < rabitq_score_words)) {
                            int coarse_diff_cnt = 0;
                            const uint32_t* q_bits0 = s_rabitq_qwords;
                            for (int w = 0; w < coarse_words; ++w) {
                                uint32_t vb = v_bits[w];
                                uint32_t qb = q_bits0[w];
                                if ((w == rabitq_score_words - 1) &&
                                    (rabitq_score_tail_bits != 0)) {
                                    vb &= rabitq_score_tail_mask;
                                    qb &= rabitq_score_tail_mask;
                                }
                                coarse_diff_cnt += __popc(vb ^ qb);
                            }
                            // Reuse coarse prefix popcount in full RabitQ sign score.
                            rabitq_pre_diff_cnt = coarse_diff_cnt;
                            rabitq_pre_words = coarse_words;
                            const int coarse_dims =
                                (coarse_words == rabitq_score_words)
                                    ? rabitq_score_dims
                                    : (coarse_words << 5);
                            const int coarse_score_i =
                                coarse_dims - (coarse_diff_cnt << 1);
#if RABITQ_COARSE_PREFILTER_MODE == 2
                            const int coarse_bias = coarse_score_i + 4096;
                            const uint32_t local_coarse =
                                valid ? (uint32_t)coarse_bias : 0u;
                            const uint32_t best_coarse =
                                warp_reduce_max_u32(local_coarse);
                            const int coarse_th =
                                max((int)best_coarse -
                                        (int)RABITQ_COARSE_MATCH_SLACK0,
                                    0);
                            const bool keep_coarse =
                                valid && (coarse_bias >= coarse_th);
                            uint32_t coarse_mask =
                                __ballot_sync(FULL_MASK, keep_coarse);
                            if (__popc(coarse_mask) < topn) {
                                coarse_mask = __ballot_sync(FULL_MASK, valid);
                            }
#else
                            const int score_bias = coarse_score_i + 4096;
                            const uint32_t key_coarse =
                                ((uint32_t)score_bias << 16) |
                                (0xFFFFu - (uint32_t)lane_id);
                            const uint32_t coarse_mask =
                                warp_select_topk_mask_desc_u32_rabitq(
                                    key_coarse, valid, coarse_keep);
#endif
                            in_coarse_pre =
                                (((coarse_mask >> lane_id) & 1u) != 0u);
                        }
                        }
 #endif
#if RABITQ_USE_FACTORS && RABITQ_FACTOR_LIGHT_PRESELECT
                        if (factor_pre_ready) {
                            // Stage-1 factor preselect: lanes outside shortlist skip
                            // heavy factor/q6 scoring in this tile.
                            in_coarse_pre = in_coarse_pre && in_factor_pre;
                        }
#endif
                        if (in_coarse_pre) {
                        float rabitq_score = 0.0f;
#if RABITQ_SCORE_MODE == 1
                        int set_sum = 0;
#if RABITQ_PACK_LUT_ACTIVE
                        if (rabitq_packed_fastscan) {
                            const int lane_pair = lane_id & 15;
                            const int idx = ((lane_pair & 7) << 1) | (lane_pair >> 3);
                            const bool upper = ((lane_id & 16) != 0);
                            if (rabitq_packed_pair_shuffle) {
                                #pragma unroll
                                for (int p = 0; p < 16; ++p) {
                                    const int off = p << 5;
                                    unsigned int b01 = 0u;
                                    if (lane_id < 16) {
                                        const uint8_t b0 =
                                            __ldg(rabitq_code_block + off + idx);
                                        const uint8_t b1 =
                                            __ldg(rabitq_code_block + off + 16 + idx);
                                        b01 = (unsigned int)b0 | ((unsigned int)b1 << 8);
                                    }
                                    b01 = (unsigned int)__shfl_sync(FULL_MASK, (int)b01, lane_pair);
                                    const uint8_t b0 = (uint8_t)(b01 & 0xFFu);
                                    const uint8_t b1 = (uint8_t)((b01 >> 8) & 0xFFu);
                                    const uint8_t nib0 = upper ? (b0 >> 4) : (b0 & 0xFu);
                                    const uint8_t nib1 = upper ? (b1 >> 4) : (b1 & 0xFu);
                                    set_sum += (int)s_rabitq_pack_lut[(p << 1) + 0][nib0];
                                    set_sum += (int)s_rabitq_pack_lut[(p << 1) + 1][nib1];
                                }
                            } else {
                                #pragma unroll
                                for (int p = 0; p < 16; ++p) {
                                    const int off = p << 5;
                                    const uint8_t b0 = __ldg(rabitq_code_block + off + idx);
                                    const uint8_t b1 = __ldg(rabitq_code_block + off + 16 + idx);
                                    const uint8_t nib0 = upper ? (b0 >> 4) : (b0 & 0xFu);
                                    const uint8_t nib1 = upper ? (b1 >> 4) : (b1 & 0xFu);
                                    set_sum += (int)s_rabitq_pack_lut[(p << 1) + 0][nib0];
                                    set_sum += (int)s_rabitq_pack_lut[(p << 1) + 1][nib1];
                                }
                            }
                        } else
#endif
                        {
                            for (int w = 0; w < rabitq_words; ++w) {
                                const uint32_t vw = v_bits[w];
                                set_sum += (int)s_rabitq_q6_lut[w][0][(vw >> 0) & 0xFu];
                                set_sum += (int)s_rabitq_q6_lut[w][1][(vw >> 4) & 0xFu];
                                set_sum += (int)s_rabitq_q6_lut[w][2][(vw >> 8) & 0xFu];
                                set_sum += (int)s_rabitq_q6_lut[w][3][(vw >> 12) & 0xFu];
                                set_sum += (int)s_rabitq_q6_lut[w][4][(vw >> 16) & 0xFu];
                                set_sum += (int)s_rabitq_q6_lut[w][5][(vw >> 20) & 0xFu];
                                set_sum += (int)s_rabitq_q6_lut[w][6][(vw >> 24) & 0xFu];
                                set_sum += (int)s_rabitq_q6_lut[w][7][(vw >> 28) & 0xFu];
                            }
                        }
                        const int signed_sum = (set_sum << 1) - s_rabitq_q6_sum;
                        rabitq_score = (float)signed_sum;
#elif (RABITQ_FASTSCAN_ENABLE != 0) && !RABITQ_USE_FACTORS
                        int set_sum = 0;
#if RABITQ_PACK_LUT_ACTIVE
                        if (rabitq_packed_fastscan) {
                            const int lane_pair = lane_id & 15;
                            const int idx = ((lane_pair & 7) << 1) | (lane_pair >> 3);
                            const bool upper = ((lane_id & 16) != 0);
                            if (rabitq_packed_pair_shuffle) {
                                #pragma unroll
                                for (int p = 0; p < 16; ++p) {
                                    const int off = p << 5;
                                    unsigned int b01 = 0u;
                                    if (lane_id < 16) {
                                        const uint8_t b0 =
                                            __ldg(rabitq_code_block + off + idx);
                                        const uint8_t b1 =
                                            __ldg(rabitq_code_block + off + 16 + idx);
                                        b01 = (unsigned int)b0 | ((unsigned int)b1 << 8);
                                    }
                                    b01 = (unsigned int)__shfl_sync(FULL_MASK, (int)b01, lane_pair);
                                    const uint8_t b0 = (uint8_t)(b01 & 0xFFu);
                                    const uint8_t b1 = (uint8_t)((b01 >> 8) & 0xFFu);
                                    const uint8_t nib0 = upper ? (b0 >> 4) : (b0 & 0xFu);
                                    const uint8_t nib1 = upper ? (b1 >> 4) : (b1 & 0xFu);
                                    set_sum += (int)s_rabitq_pack_lut[(p << 1) + 0][nib0];
                                    set_sum += (int)s_rabitq_pack_lut[(p << 1) + 1][nib1];
                                }
                            } else {
                                #pragma unroll
                                for (int p = 0; p < 16; ++p) {
                                    const int off = p << 5;
                                    const uint8_t b0 = __ldg(rabitq_code_block + off + idx);
                                    const uint8_t b1 = __ldg(rabitq_code_block + off + 16 + idx);
                                    const uint8_t nib0 = upper ? (b0 >> 4) : (b0 & 0xFu);
                                    const uint8_t nib1 = upper ? (b1 >> 4) : (b1 & 0xFu);
                                    set_sum += (int)s_rabitq_pack_lut[(p << 1) + 0][nib0];
                                    set_sum += (int)s_rabitq_pack_lut[(p << 1) + 1][nib1];
                                }
                            }
                        } else
#endif
                        if (rabitq_score_words == 1) {
                            uint32_t vw = v_bits[0];
                            if (rabitq_score_tail_bits != 0) vw &= rabitq_score_tail_mask;
                            set_sum += (int)s_rabitq_q6_lut[0][0][(vw >> 0) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][1][(vw >> 4) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][2][(vw >> 8) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][3][(vw >> 12) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][4][(vw >> 16) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][5][(vw >> 20) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][6][(vw >> 24) & 0xFu];
                            set_sum += (int)s_rabitq_q6_lut[0][7][(vw >> 28) & 0xFu];
                        } else {
                            for (int w = 0; w < rabitq_score_words; ++w) {
                                uint32_t vw = v_bits[w];
                                if ((w == rabitq_score_words - 1) &&
                                    (rabitq_score_tail_bits != 0)) {
                                    vw &= rabitq_score_tail_mask;
                                }
                                set_sum += (int)s_rabitq_q6_lut[w][0][(vw >> 0) & 0xFu];
                                set_sum += (int)s_rabitq_q6_lut[w][1][(vw >> 4) & 0xFu];
                                set_sum += (int)s_rabitq_q6_lut[w][2][(vw >> 8) & 0xFu];
                                set_sum += (int)s_rabitq_q6_lut[w][3][(vw >> 12) & 0xFu];
                                set_sum += (int)s_rabitq_q6_lut[w][4][(vw >> 16) & 0xFu];
                                set_sum += (int)s_rabitq_q6_lut[w][5][(vw >> 20) & 0xFu];
                                set_sum += (int)s_rabitq_q6_lut[w][6][(vw >> 24) & 0xFu];
                                set_sum += (int)s_rabitq_q6_lut[w][7][(vw >> 28) & 0xFu];
                            }
                        }
                        const int signed_sum = (set_sum << 1) - s_rabitq_q6_sum;
                        rabitq_score = (float)signed_sum;
#else
                        int diff_cnt = 0;
                        int diff_start_word = 0;
#if RABITQ_COARSE_PREFILTER && (RABITQ_SCORE_MODE == 0) && !RABITQ_USE_FACTORS
                        diff_cnt = rabitq_pre_diff_cnt;
                        diff_start_word = rabitq_pre_words;
#endif
#if RABITQ_WORD1_FASTPATH && (RABITQ_SCORE_MODE == 0) && !RABITQ_USE_FACTORS && \
    (RABITQ_MAX_WORDS == 1) && (RABITQ_SCORE_WORDS == 1)
                        if (diff_start_word == 0) {
                            uint32_t x0 = v_bits[0] ^ rabitq_q0;
                            if (rabitq_score_tail_bits != 0) x0 &= rabitq_score_tail_mask;
                            diff_cnt = __popc(x0);
                        }
#else
#if SQG_HAMMING_FAST4
                        if (rabitq_fast4 && diff_start_word == 0) {
                            if (rabitq_score_words == 1) {
                                uint32_t x0 = v_bits[0] ^ rabitq_q0;
                                if (rabitq_score_tail_bits != 0) x0 &= rabitq_score_tail_mask;
                                diff_cnt = __popc(x0);
                            } else if (rabitq_score_words == 2) {
                                uint32_t x0 = v_bits[0] ^ rabitq_q0;
                                uint32_t x1 = v_bits[1] ^ rabitq_q1;
                                if (rabitq_score_tail_bits != 0) x1 &= rabitq_score_tail_mask;
                                diff_cnt = __popc(x0) + __popc(x1);
                            } else if (rabitq_score_words == 3) {
                                uint32_t x0 = v_bits[0] ^ rabitq_q0;
                                uint32_t x1 = v_bits[1] ^ rabitq_q1;
                                uint32_t x2 = v_bits[2] ^ rabitq_q2;
                                if (rabitq_score_tail_bits != 0) x2 &= rabitq_score_tail_mask;
                                diff_cnt = __popc(x0) + __popc(x1) + __popc(x2);
                            } else {
                                uint32_t x0, x1, x2, x3;
                                const uintptr_t addr = (uintptr_t)v_bits;
                                if ((addr & 0xFu) == 0u) {
                                    const uint4 vb = *reinterpret_cast<const uint4*>(v_bits);
                                    x0 = vb.x ^ rabitq_q0;
                                    x1 = vb.y ^ rabitq_q1;
                                    x2 = vb.z ^ rabitq_q2;
                                    x3 = vb.w ^ rabitq_q3;
                                } else {
                                    x0 = v_bits[0] ^ rabitq_q0;
                                    x1 = v_bits[1] ^ rabitq_q1;
                                    x2 = v_bits[2] ^ rabitq_q2;
                                    x3 = v_bits[3] ^ rabitq_q3;
                                }
                                if (rabitq_score_tail_bits != 0) x3 &= rabitq_score_tail_mask;
                                diff_cnt = __popc(x0) + __popc(x1) + __popc(x2) + __popc(x3);
                            }
                        } else
#endif
                        {
                            const uint32_t* q_bits = s_rabitq_qwords;
                            for (int w = diff_start_word; w < rabitq_score_words; ++w) {
                                if ((w == rabitq_score_words - 1) && (rabitq_score_tail_bits != 0)) {
                                    const uint32_t vb = v_bits[w] & rabitq_score_tail_mask;
                                    const uint32_t qb = q_bits[w] & rabitq_score_tail_mask;
                                    diff_cnt += __popc(vb ^ qb);
                                } else {
                                    diff_cnt += __popc(v_bits[w] ^ q_bits[w]);
                                }
                            }
                        }
#endif

                        rabitq_score = (float)(rabitq_score_dims - (diff_cnt << 1));
#endif
#if (RABITQ_SCORE_MODE == 0) && !RABITQ_USE_FACTORS
                        rabitq_score_i = (int)rabitq_score;
#endif
                        // score_mode=0 has an integer score key for cheap shortlist preselect.
#if RABITQ_USE_FACTORS
                        rabitq_score_i = (int)rabitq_score;
#endif
                        need_approx_score = true;
#if (RABITQ_SCORE_MODE == 0) && !RABITQ_USE_FACTORS
                        // Fast path: when neither cutoff nor recheck uses approx values,
                        // rank directly by integer sign score and skip per-neighbor float approx.
                        // Safety gate: on richer RabitQ configs (raw keep / coarse prefilter /
                        // neighbor budget), pure integer key fast path can destabilize convergence.
                        const bool rabitq_fastpath_safe_cfg =
                            ((int)RABITQ_RAW_KEEP == 0) &&
                            ((int)RABITQ_NEIGHBOR_BUDGET == 0) &&
                            ((int)RABITQ_COARSE_PREFILTER == 0);
                        if (((int)RABITQ_INT_KEY_FASTPATH != 0) &&
                            (((int)RABITQ_FASTSCAN_ENABLE == 0) ||
                             ((int)RABITQ_INT_KEY_ALLOW_FASTSCAN != 0)) &&
                            rabitq_fastpath_safe_cfg) {
                            need_approx_score =
                                ((float)RABITQ_APPROX_CUTOFF_RATIO > 0.0f);
                            if (!need_approx_score) {
                                const int score_bias = rabitq_score_i + 4096;
                                uint32_t key_hi = (uint32_t)score_bias;
#if RABITQ_INT_KEY_RESPECT_POS_CLAMP
                                bool key_pos_ok = true;
                                if ((float)RABITQ_SCORE_ALPHA == 1.0f) {
                                    // Fast equivalent check for:
                                    // (sqr_y - score_i) > 0  <=>  score_i < ceil(sqr_y)
                                    const int sqr_y_ceil =
                                        __float2int_ru((float)rabitq_expand_sqr_y);
                                    key_pos_ok = (rabitq_score_i < sqr_y_ceil);
                                } else {
                                    const float approx_pos_check =
                                        (float)rabitq_expand_sqr_y -
                                        ((float)RABITQ_SCORE_ALPHA * (float)rabitq_score_i);
                                    key_pos_ok = (approx_pos_check > 0.0f);
                                }
                                if (!key_pos_ok) key_hi = 0u;
#endif
#if RABITQ_INT_KEY_TIEBREAK
                                key_rabitq =
                                    (key_hi << 16) | (0xFFFFu - (uint32_t)lane_id);
#else
                                key_rabitq = (key_hi << 16);
#endif
                            }
                        }
#endif
#if RABITQ_USE_FACTORS
                        const int pre_keep = factor_pre_ready
                            ? factor_pre_keep
                            : min(deg, topn + max((int)RABITQ_FACTOR_PRESELECT_EXTRA, 0));
                        uint32_t key_pre = factor_key_pre;
                        bool in_pre = in_factor_pre;
                        if (!factor_pre_ready) {
                            const int score_bias = rabitq_score_i + 4096;
                            key_pre =
                                ((uint32_t)score_bias << 16) |
                                (0xFFFFu - (uint32_t)lane_id);
                            const uint32_t pre_mask =
                                warp_select_topk_mask_desc_u32_rabitq(
                                    key_pre, valid, pre_keep);
                            in_pre = valid && (((pre_mask >> lane_id) & 1u) != 0u);
                        }
                        in_factor_pre = in_pre;
                        // Fast path: preselect already chooses exactly the final keep-set.
                        const bool can_skip_approx_select =
                            ((int)RABITQ_GPU_DUALPOOL_ENGINE == 0) &&
                            (pre_keep <= topn) &&
                            ((float)RABITQ_APPROX_CUTOFF_RATIO <= 0.0f);
                        rabitq_preonly_keep = can_skip_approx_select;
                        need_approx_score = in_pre && !can_skip_approx_select;
                        if (!need_approx_score) {
                            key_rabitq = key_pre;
                        }
#endif
                        if (need_approx_score) {
                        const float sqr_y = (float)rabitq_expand_sqr_y;
#if RABITQ_USE_FACTORS
                        const float triple = d_rabitq_triple_x[edge_pos];
#if RABITQ_USE_PACKED_DQVQ
#if RABITQ_USE_FP16_DQVQ
                        const float2 dqvq = __half22float2(d_rabitq_factor_dqvq_h2[edge_pos]);
#else
                        const float2 dqvq = d_rabitq_factor_dqvq[edge_pos];
#endif
                        const float fdq = dqvq.x;
                        const float fvq = dqvq.y;
#else
                        const float fdq = d_rabitq_factor_dq[edge_pos];
                        const float fvq = d_rabitq_factor_vq[edge_pos];
#endif
                        float approx = 0.0f;
#if RABITQ_SCORE_MODE == 1
                        const float q_term = (float)RABITQ_SCORE_ALPHA *
                                             (s_rabitq_q6_width * rabitq_score);
                        approx = sqr_y + triple + (fdq * q_term) + (fvq * s_rabitq_q6_lower);
#else
                        approx = sqr_y + triple +
                                 ((float)RABITQ_SCORE_ALPHA * fdq * rabitq_score) +
                                 (fvq * s_rabitq_qsign_sum);
#endif
#else
                        float approx = 0.0f;
#if RABITQ_SCORE_MODE == 1
                        approx = sqr_y - ((float)RABITQ_SCORE_ALPHA *
                                          (s_rabitq_q6_width * rabitq_score));
#elif (RABITQ_FASTSCAN_ENABLE != 0) && !RABITQ_USE_FACTORS
                        approx = sqr_y - ((float)RABITQ_SCORE_ALPHA *
                                          (s_rabitq_q6_width * rabitq_score));
#else
                        approx = sqr_y - ((float)RABITQ_SCORE_ALPHA * rabitq_score);
#endif
#endif
                        if (!(approx > 0.0f) || (approx != approx) || (approx >= FLT_MAX)) {
                            approx = FLT_MAX * 0.5f;
                        }
                        rabitq_neighbor_approx = (DISTANCE_T)approx;
                        key_rabitq = 0xFFFFFFFFu - __float_as_uint(approx);
                        }
                        if (valid && !(rabitq_neighbor_approx < FLT_MAX) &&
                            (rabitq_score_i > -0x3fffffff)) {
                            float approx_f =
                                (float)rabitq_expand_sqr_y -
                                ((float)RABITQ_SCORE_ALPHA * (float)rabitq_score_i);
                            if (!(approx_f > 0.0f) || (approx_f != approx_f) ||
                                (approx_f >= FLT_MAX)) {
                                approx_f = FLT_MAX * 0.5f;
                            }
                            rabitq_neighbor_approx = (DISTANCE_T)approx_f;
                        }
                        }
                    }
                    bool valid_for_select = valid;
#if (RABITQ_SCORE_MODE == 0) && !RABITQ_USE_FACTORS && RABITQ_COARSE_PREFILTER
                    if ((int)RABITQ_GLOBAL_APPROX_PRESELECT == 0) {
                        valid_for_select = in_coarse_pre;
                    }
#endif
#if RABITQ_USE_FACTORS
                    if ((int)RABITQ_GLOBAL_APPROX_PRESELECT == 0) {
                        valid_for_select = in_factor_pre;
                    }
#endif
                    const uint32_t valid_select_mask =
                        __ballot_sync(FULL_MASK, valid_for_select);
                    uint32_t selected_mask = 0u;
#if RABITQ_REFLOW_FIXED
                    // Fixed-path selector: always use one warp-topk primitive.
                    // This removes mode/tie/raw branching from the RabitQ hot tile path.
                    selected_mask =
                        warp_select_topk_mask_desc_u32_rabitq(
                            key_rabitq, valid_for_select, topn);
#else
                    if ((int)RABITQ_GLOBAL_APPROX_PRESELECT != 0) {
                        selected_mask = valid_select_mask;
                    } else if (rabitq_preonly_keep) {
                        selected_mask = valid_select_mask;
                    } else {
#if (RABITQ_SELECT_MODE == 1) && (RABITQ_SCORE_MODE == 0) && !RABITQ_USE_FACTORS
                    int best_score = valid ? rabitq_score_i : -0x3fffffff;
                    #pragma unroll
                    for (int ofs = 16; ofs > 0; ofs >>= 1) {
                        const int v = __shfl_down_sync(FULL_MASK, best_score, ofs);
                        if (v > best_score) best_score = v;
                    }
                    best_score = __shfl_sync(FULL_MASK, best_score, 0);
                    const int score_th = best_score - (int)RABITQ_MATCH_SLACK;
                    const bool keep_fast = valid && (rabitq_score_i >= score_th);
                    const uint32_t fast_mask = __ballot_sync(FULL_MASK, keep_fast);
                    const int fast_cnt = __popc(fast_mask);
                    if (fast_cnt < topn) {
                        selected_mask = warp_select_topk_mask_desc_u32_rabitq(
                            key_rabitq, valid, topn);
                    } else if (fast_cnt > (topn + (int)RABITQ_FAST_EXTRA_KEEP)) {
                        const int target_keep = min(deg, topn + (int)RABITQ_FAST_EXTRA_KEEP);
                        const uint32_t lane_mask =
                            (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
                        const int rank = __popc(fast_mask & lane_mask);
#if RABITQ_FAST_DOWNSAMPLE_MODE == 1
                        const bool keep_ds =
                            (((fast_mask >> lane_id) & 1u) != 0u) && (rank < target_keep);
#else
                        const int stride = max(1, (fast_cnt + target_keep - 1) / target_keep);
                        const bool keep_ds =
                            (((fast_mask >> lane_id) & 1u) != 0u) && ((rank % stride) == 0);
#endif
                        selected_mask = __ballot_sync(FULL_MASK, keep_ds);
                        if (__popc(selected_mask) < topn) {
                            selected_mask = fast_mask;
                        }
                    } else {
                        selected_mask = fast_mask;
                    }
#else
                    selected_mask =
                        warp_select_topk_mask_desc_u32_rabitq(
                            key_rabitq, valid_for_select, topn);
#endif
                    }
#if RABITQ_SELECT_TIE_CAP_ENABLE
                    if (!rabitq_preonly_keep &&
                        ((int)RABITQ_GLOBAL_APPROX_PRESELECT == 0)) {
                        const int cap_keep =
                            min(deg, topn + max((int)RABITQ_SELECT_TIE_CAP_EXTRA, 0));
                        const int sel_cnt = __popc(selected_mask);
                        if ((cap_keep > 0) && (sel_cnt > cap_keep)) {
                            const uint32_t lane_mask =
                                (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
                            const int rank = __popc(selected_mask & lane_mask);
                            const int stride = max(1, (sel_cnt + cap_keep - 1) / cap_keep);
                            const bool keep_ds =
                                (((selected_mask >> lane_id) & 1u) != 0u) &&
                                ((rank % stride) == 0);
                            const uint32_t ds_mask = __ballot_sync(FULL_MASK, keep_ds);
                            if (__popc(ds_mask) >= topn) {
                                selected_mask = ds_mask;
                            }
                        }
                    }
#endif
#endif
#if RABITQ_DISABLE_IF_NO_PRUNE_ITERS > 0
                    if (lane_id == 0) {
                        atomicAdd(&s_rabitq_iter_active_tiles, 1);
                        const int valid_cnt = __popc(valid_select_mask);
                        const int keep_cnt = __popc(selected_mask);
                        if (keep_cnt < valid_cnt) {
                            atomicAdd(&s_rabitq_iter_pruned_tiles, 1);
                        }
                    }
#endif
#if RABITQ_REFLOW_FIXED
                    keep = (((selected_mask >> lane_id) & 1u) != 0u);
#else
                    bool keep_rabitq = (((selected_mask >> lane_id) & 1u) != 0u);
                    bool keep_raw = false;
#if (RABITQ_RAW_KEEP > 0) || (RABITQ_RAW_KEEP_LOW > 0) || (RABITQ_RAW_KEEP_HIGH > 0)
                    if (valid) {
                        int raw_keep_n_cfg = (int)RABITQ_RAW_KEEP;
#if (RABITQ_RAW_KEEP_LOW > 0) && (RABITQ_RAW_KEEP_HIGH > 0) && \
    (RABITQ_RAW_KEEP_SPLIT_BEAM > 0)
                        raw_keep_n_cfg =
                            (BEAM_SIZE <= (int)RABITQ_RAW_KEEP_SPLIT_BEAM)
                                ? (int)RABITQ_RAW_KEEP_LOW
                                : (int)RABITQ_RAW_KEEP_HIGH;
#endif
                        const int raw_keep_n = min(raw_keep_n_cfg, deg);
                        if (raw_keep_n > 0) {
                            for (int s = 0; s < raw_keep_n; s++) {
                                int pick = (((s + 1) * deg) / (raw_keep_n + 1));
                                if (pick >= deg) pick = deg - 1;
                                if (lane_id == pick) {
                                    keep_raw = true;
                                    break;
                                }
                            }
                        }
                    }
#endif
                    keep = keep_rabitq || keep_raw;
#endif
                    keep_decided = true;
#endif
                }
                }
                }
#endif
#if SQG_KERNEL_ACTIVE
                if (!keep_decided) {
                const int sqg_topn_deg = min(max(sqg_topn_iter, 1), deg);
                const bool sqg_active = sqg_filter_gate_runtime && (deg > 1) && (sqg_topn_deg < deg);
                if (sqg_active) {
                    const int topn = sqg_topn_deg;
                    bool valid_for_select = valid;
#if SQG_COARSE_PREFILTER && (SQG_SCORE_MODE == 0)
                    if ((sqg_words >= (int)SQG_COARSE_MIN_WORDS) && (topn < deg)) {
                        int coarse_match = -1;
                        uint32_t key_coarse = 0u;
                        if (valid) {
                            const uint32_t* v_bits0 =
                                d_sqg_sign_bit + (size_t)neighbor * (size_t)sqg_words;
                            const uint32_t* q_bits0 = s_sqg_qwords[warp_id];
                            const int coarse_words = min(max((int)SQG_COARSE_WORDS, 1), sqg_words);
                            int coarse_sum = 0;
                            #pragma unroll
                            for (int wi = 0; wi < 8; wi++) {
                                if (wi >= coarse_words) break;
                                uint32_t x = v_bits0[wi] ^ q_bits0[wi];
                                if ((wi == sqg_words - 1) && (sqg_tail_bits != 0)) {
                                    x &= sqg_tail_mask;
                                }
                                coarse_sum += 32 - __popc(x);
                            }
                            coarse_match = coarse_sum;
                            key_coarse = (static_cast<uint32_t>(coarse_match) << 16) |
                                         (0xFFFFu - (uint32_t)lane_id);
                        }
#if SQG_COARSE_PREFILTER_MODE == 1
                        const int coarse_keep = min(deg, topn + (int)SQG_COARSE_EXTRA_KEEP);
                        const uint32_t coarse_mask =
                            warp_select_topk_mask_desc_u32(key_coarse, valid, coarse_keep);
                        valid_for_select = valid && (((coarse_mask >> lane_id) & 1u) != 0u);
#elif SQG_COARSE_PREFILTER_MODE == 2
                        const uint32_t local_coarse = valid ? (uint32_t)coarse_match : 0u;
                        const uint32_t best_coarse = warp_reduce_max_u32(local_coarse);
                        const int coarse_th =
                            max((int)best_coarse - (int)SQG_COARSE_MATCH_SLACK0, 0);
                        valid_for_select = valid && (coarse_match >= coarse_th);
#endif
                    }
#endif
                    uint32_t key_sign = 0;  // larger is better
                    if (valid_for_select) {
                        const uint32_t* v_bits = d_sqg_sign_bit + (size_t)neighbor * (size_t)sqg_words;
                        int diff_cnt = 0;
#if SQG_HAMMING_FAST4
                        if (sqg_fast4) {
                            if (sqg_words == 1) {
                                uint32_t x0 = v_bits[0] ^ sqg_q0;
                                if (sqg_tail_bits != 0) x0 &= sqg_tail_mask;
                                diff_cnt = __popc(x0);
                            } else if (sqg_words == 2) {
                                uint32_t x0 = v_bits[0] ^ sqg_q0;
                                uint32_t x1 = v_bits[1] ^ sqg_q1;
                                if (sqg_tail_bits != 0) x1 &= sqg_tail_mask;
                                diff_cnt = __popc(x0) + __popc(x1);
                            } else if (sqg_words == 3) {
                                uint32_t x0 = v_bits[0] ^ sqg_q0;
                                uint32_t x1 = v_bits[1] ^ sqg_q1;
                                uint32_t x2 = v_bits[2] ^ sqg_q2;
                                if (sqg_tail_bits != 0) x2 &= sqg_tail_mask;
                                diff_cnt = __popc(x0) + __popc(x1) + __popc(x2);
                            } else {  // sqg_words == 4
                                uint32_t x0, x1, x2, x3;
                                const uintptr_t addr = (uintptr_t)v_bits;
                                if ((addr & 0xFu) == 0u) {
                                    const uint4 vb = *reinterpret_cast<const uint4*>(v_bits);
                                    x0 = vb.x ^ sqg_q0;
                                    x1 = vb.y ^ sqg_q1;
                                    x2 = vb.z ^ sqg_q2;
                                    x3 = vb.w ^ sqg_q3;
                                } else {
                                    x0 = v_bits[0] ^ sqg_q0;
                                    x1 = v_bits[1] ^ sqg_q1;
                                    x2 = v_bits[2] ^ sqg_q2;
                                    x3 = v_bits[3] ^ sqg_q3;
                                }
                                if (sqg_tail_bits != 0) x3 &= sqg_tail_mask;
                                diff_cnt = __popc(x0) + __popc(x1) + __popc(x2) + __popc(x3);
                            }
                        } else
#endif
                        {
                            const uint32_t* q_bits = s_sqg_qwords[warp_id];
                            diff_cnt = sqg_hamming_diff_words(
                                v_bits, q_bits, sqg_words, sqg_tail_bits, sqg_tail_mask);
                        }
                        const int match_cnt = sqg_dims - diff_cnt;
                        key_sign = (static_cast<uint32_t>(match_cnt) << 16) | (0xFFFFu - (uint32_t)lane_id);
                    }
                    uint32_t selected_mask = 0u;
#if SQG_SCORE_MODE == 1 && ENABLE_SQ8
                    if (dim == 128 && d_data_sq8 != nullptr) {
                        uint32_t key_sq8 = 0;
                        if (valid_for_select) {
                            const int8_t* x8 = d_data_sq8 + (size_t)neighbor * (size_t)128;
                            int approx = 0;
                            #pragma unroll
                            for (int i = 0; i < 128; i += SQG_SQ8_STRIDE) {
                                const int dq = (int)s_query_sq8[i] - (int)x8[i];
                                approx += dq * dq;
                            }
                            key_sq8 = 0xFFFFFFFFu - (uint32_t)approx;
                        }
                        selected_mask = warp_select_topk_mask_desc_u32(key_sq8, valid_for_select, topn);
                    } else {
                        selected_mask = warp_select_topk_mask_desc_u32(key_sign, valid_for_select, topn);
                    }
#elif SQG_SCORE_MODE == 2 && ENABLE_SQ8
                    if (dim == 128 && d_data_sq8 != nullptr) {
                        const int refine_m_cfg = max((int)SQG_REFINE_TOPM, topn);
                        const int refine_m = min(max(refine_m_cfg, topn), deg);
                        const uint32_t pre_mask =
                            warp_select_topk_mask_desc_u32(key_sign, valid_for_select, refine_m);
                        const bool in_pre =
                            valid_for_select && (((pre_mask >> lane_id) & 1u) != 0u);
                        uint32_t key_refine = 0;
                        if (in_pre) {
                            const int8_t* x8 = d_data_sq8 + (size_t)neighbor * (size_t)128;
                            int approx = 0;
                            #pragma unroll
                            for (int i = 0; i < 128; i += SQG_SQ8_STRIDE) {
                                const int dq = (int)s_query_sq8[i] - (int)x8[i];
                                approx += dq * dq;
                            }
                            key_refine = 0xFFFFFFFFu - (uint32_t)approx;
                        }
                        selected_mask = warp_select_topk_mask_desc_u32(key_refine, in_pre, topn);
                    } else {
                        selected_mask = warp_select_topk_mask_desc_u32(key_sign, valid_for_select, topn);
                    }
#else
#if SQG_SELECT_MODE == 1
                    // Fast approximate selector for sign-score mode:
                    // 1) threshold by best-match slack
                    // 2) downsample overflow keep-set
                    // 3) only fill the missing quota by exact top-k (avoid full exact fallback).
                    const uint32_t local_key = valid_for_select ? key_sign : 0u;
                    const uint32_t best_key = warp_reduce_max_u32(local_key);
                    const uint32_t slack_key = ((uint32_t)max((int)SQG_MATCH_SLACK, 0)) << 16;
                    const uint32_t th_key = (best_key > slack_key) ? (best_key - slack_key) : 0u;
                    const uint32_t approx_mask =
                        __ballot_sync(FULL_MASK, valid_for_select && (key_sign >= th_key));
                    const int approx_cnt = __popc(approx_mask);
                    if (approx_cnt >= topn && approx_cnt <= (topn + (int)SQG_FAST_EXTRA_KEEP)) {
                        selected_mask = approx_mask;
#if SQG_SELECT_OVERFLOW_DOWNSAMPLE
                    } else if (approx_cnt > (topn + (int)SQG_FAST_EXTRA_KEEP)) {
                        const int target_keep = min(deg, topn + (int)SQG_FAST_EXTRA_KEEP);
                        const int rank = __popc(approx_mask & ((1u << lane_id) - 1));
                        const int stride = max(1, (approx_cnt + target_keep - 1) / target_keep);
                        const bool keep_ds = (((approx_mask >> lane_id) & 1u) != 0u) && ((rank % stride) == 0);
                        selected_mask = __ballot_sync(FULL_MASK, keep_ds);
                        const int selected_cnt = __popc(selected_mask);
                        if (selected_cnt < topn) {
                            const bool valid_fill =
                                valid_for_select && (((selected_mask >> lane_id) & 1u) == 0u);
                            const uint32_t fill_mask = warp_select_topk_mask_desc_u32(
                                key_sign, valid_fill, topn - selected_cnt);
                            selected_mask |= fill_mask;
                        }
#endif
                    } else {
                        // approx_cnt < topn: keep approximate candidates, then fill only missing slots.
                        selected_mask = approx_mask;
                        if (approx_cnt < topn) {
                            const bool valid_fill =
                                valid_for_select && (((approx_mask >> lane_id) & 1u) == 0u);
                            const uint32_t fill_mask = warp_select_topk_mask_desc_u32(
                                key_sign, valid_fill, topn - approx_cnt);
                            selected_mask |= fill_mask;
                        }
                    }
#elif SQG_SELECT_MODE == 2
                    // Pure-fast selector: avoid exact top-k in the hot loop.
                    const uint32_t local_key = valid_for_select ? key_sign : 0u;
                    const uint32_t best_key = warp_reduce_max_u32(local_key);
                    const uint32_t slack_key = ((uint32_t)max((int)SQG_MATCH_SLACK, 0)) << 16;
                    const uint32_t th_key = (best_key > slack_key) ? (best_key - slack_key) : 0u;
                    const uint32_t approx_mask =
                        __ballot_sync(FULL_MASK, valid_for_select && (key_sign >= th_key));
                    const int approx_cnt = __popc(approx_mask);
                    if (approx_cnt == 0) {
                        selected_mask = warp_select_topk_mask_desc_u32(key_sign, valid_for_select, 1);
                    } else {
                        const int target_keep = min(deg, topn + (int)SQG_FAST_EXTRA_KEEP);
                        if (approx_cnt > target_keep) {
                            const int rank = __popc(approx_mask & ((1u << lane_id) - 1));
                            const int stride = max(1, (approx_cnt + target_keep - 1) / target_keep);
                            const bool keep_ds = (((approx_mask >> lane_id) & 1u) != 0u) && ((rank % stride) == 0);
                            selected_mask = __ballot_sync(FULL_MASK, keep_ds);
                            if (__popc(selected_mask) < topn) {
                                selected_mask = approx_mask;
                            }
                        } else {
                            selected_mask = approx_mask;
                        }
                    }
#else
                    selected_mask = warp_select_topk_mask_desc_u32(key_sign, valid_for_select, topn);
#endif
#endif
                    const bool keep_sqg = (((selected_mask >> lane_id) & 1u) != 0u);
                    bool keep_raw = false;
#if SQG_RAW_KEEP > 0
                    if (valid) {
                        const int raw_keep_n = min((int)SQG_RAW_KEEP, deg);
                        for (int s = 0; s < raw_keep_n; s++) {
                            int pick = (((s + 1) * deg) / (raw_keep_n + 1));
                            if (pick >= deg) pick = deg - 1;
                            if (lane_id == pick) {
                                keep_raw = true;
                                break;
                            }
                        }
                    }
#endif
                    keep = keep_sqg || keep_raw;
                }
                }
#endif
#if SQG_KERNEL_ACTIVE && SQG_NEIGHBOR_BUDGET_ENABLE
                if (sqg_filter_gate_runtime && s_num_neighbors >= sqg_neighbor_budget) {
                    keep = false;
                }
#if SQG_NEIGHBOR_BUDGET_PRESAMPLE
                if (sqg_filter_gate_runtime) {
                    const uint32_t keep_mask0 = __ballot_sync(FULL_MASK, keep);
                    if (keep_mask0 != 0u) {
                        const int remain_budget = sqg_neighbor_budget - s_num_neighbors;
                        if (remain_budget <= 0) {
                            keep = false;
                        } else {
                            const int keep_cnt0 = __popc(keep_mask0);
                            if (keep_cnt0 > remain_budget) {
                                const uint32_t lane_mask =
                                    (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
                                const int rank = __popc(keep_mask0 & lane_mask);
                                keep = (((keep_mask0 >> lane_id) & 1u) != 0u) &&
                                       (rank < remain_budget);
                            }
                        }
                    }
                }
#endif
#endif
#if RABITQ_KERNEL_ACTIVE && (RABITQ_NEIGHBOR_BUDGET > 0)
                if (rabitq_budget_gate && s_num_neighbors >= rabitq_neighbor_budget) {
                    keep = false;
                }
#if RABITQ_NEIGHBOR_BUDGET_PRESAMPLE
                if (rabitq_budget_gate) {
                    const uint32_t keep_mask_rb = __ballot_sync(FULL_MASK, keep);
                    if (keep_mask_rb != 0u) {
                        const int remain_budget = rabitq_neighbor_budget - s_num_neighbors;
                        if (remain_budget <= 0) {
                            keep = false;
                        } else {
                            const int keep_cnt_rb = __popc(keep_mask_rb);
                            if (keep_cnt_rb > remain_budget) {
                                const uint32_t lane_mask_rb =
                                    (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
                                const int rank_rb = __popc(keep_mask_rb & lane_mask_rb);
                                keep = (((keep_mask_rb >> lane_id) & 1u) != 0u) &&
                                       (rank_rb < remain_budget);
                            }
                        }
                    }
                }
#endif
#endif
                uint32_t is_new = 0;
                bool keep_local = keep;
#if WARP_L1_DEDUP_ENABLE && WARP_L1_DEDUP_FEATURE_ACTIVE
                if (keep_local) {
                    keep_local = (hashtable_insert_noatomic(
                        s_warp_l1_hash[warp_id], WARP_L1_HASH_BITLEN, neighbor) != 0u);
                }
#endif
#if SKIP_ATOMIC_WHEN_POOL_FULL
                const uint32_t keep_mask0 = __ballot_sync(FULL_MASK, keep_local);
                if (keep_mask0 != 0u) {
                    const int remain_slots =
                        (int)NEIGHBOR_BUFFER_SIZE -
                        warp_read_volatile_shared_i32((volatile int*)&s_num_neighbors);
                    if (remain_slots <= 0) {
                        keep_local = false;
                    } else {
                        const int keep_cnt = __popc(keep_mask0);
                        if (keep_cnt > remain_slots) {
                            const uint32_t lane_mask =
                                (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
                            const int rank = __popc(keep_mask0 & lane_mask);
                            keep_local = (((keep_mask0 >> lane_id) & 1u) != 0u) && (rank < remain_slots);
                        }
                    }
                }
#endif
                const uint32_t active = __ballot_sync(FULL_MASK, keep_local);
                if (active != 0u) {
                    const int staged_neighbors_base =
                        warp_read_volatile_shared_i32((volatile int*)&s_num_neighbors);
#if HASH_WARP_DEDUP
                    const uint32_t same = __match_any_sync(active, (uint32_t)neighbor);
                    const int leader = __ffs(same) - 1;
                    if (keep_local && lane_id == leader) {
                        is_new = hashtable_insert_adaptive(
                            s_hash, BITLEN, neighbor, staged_neighbors_base);
                    }
#else
                    if (keep_local) {
                        is_new = hashtable_insert_adaptive(
                            s_hash, BITLEN, neighbor, staged_neighbors_base);
                    }
#endif
                }
                uint32_t ballot = __ballot_sync(FULL_MASK, is_new);
                int warp_count = __popc(ballot);
                int base_offset = 0;
                if (lane_id == 0 && warp_count > 0) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
                    if (*((volatile int*)&s_num_neighbors) < (int)NEIGHBOR_BUFFER_SIZE) {
                        base_offset = atomicAdd(&s_num_neighbors, warp_count);
                    } else {
                        base_offset = (int)NEIGHBOR_BUFFER_SIZE;
                    }
#else
                    base_offset = atomicAdd(&s_num_neighbors, warp_count);
#endif
                }
                base_offset = __shfl_sync(FULL_MASK, base_offset, 0);
                if (is_new) {
                    int my_offset = __popc(ballot & ((1u << lane_id) - 1));
                    int slot = base_offset + my_offset;
                    if (slot < (int)NEIGHBOR_BUFFER_SIZE
#if SQG_KERNEL_ACTIVE && SQG_NEIGHBOR_BUDGET_ENABLE
                        && (!sqg_filter_gate_runtime || slot < sqg_neighbor_budget)
#endif
                    ) {
                        s_neighbor_idx[slot] = neighbor;
#if RABITQ_KERNEL_ACTIVE && RABITQ_NEIGHBOR_APPROX_BUFFER
                        s_neighbor_approx[slot] = rabitq_neighbor_approx;
#endif
                    }
                }
            }
        }
#endif
        __syncthreads();
#if RABITQ_KERNEL_ACTIVE && (RABITQ_DISABLE_IF_NO_PRUNE_ITERS > 0)
        if (tid == 0 && rabitq_filter_gate) {
            if (s_rabitq_iter_active_tiles > 0) {
                if (s_rabitq_iter_pruned_tiles == 0) {
                    s_rabitq_no_prune_iters++;
                } else {
                    s_rabitq_no_prune_iters = 0;
                }
                if (s_rabitq_no_prune_iters >= (int)RABITQ_DISABLE_IF_NO_PRUNE_ITERS) {
                    s_rabitq_force_off = 1;
                }
            } else {
                s_rabitq_no_prune_iters = 0;
            }
        }
        __syncthreads();
#endif
        int num_neighbors = min(s_num_neighbors, (int)NEIGHBOR_BUFFER_SIZE);
#if SQG_KERNEL_ACTIVE && SQG_NEIGHBOR_BUDGET_ENABLE
        if (sqg_filter_gate_runtime) {
            num_neighbors = min(num_neighbors, sqg_neighbor_budget);
        }
#endif
#if RABITQ_KERNEL_ACTIVE && (RABITQ_NEIGHBOR_BUDGET > 0)
        if (rabitq_budget_gate) {
            num_neighbors = min(num_neighbors, rabitq_neighbor_budget);
        }
#endif
#if RABITQ_KERNEL_ACTIVE && (RABITQ_STAGE_CAP > 0)
        if (rabitq_filter_gate) {
            num_neighbors = min(num_neighbors, (int)RABITQ_STAGE_CAP);
        }
#endif
        if (tid == 0) {
            if (num_neighbors == 0) {
                s_no_progress_iters++;
                if (s_no_progress_iters >= (int)NO_PROGRESS_BREAK_ITERS) {
                    s_break_now = 1;
                }
            } else {
                s_no_progress_iters = 0;
            }
        }
        __syncthreads();
        if (s_break_now) break;
        if (num_neighbors == 0) continue;
        const bool frontier_full = (s_frontier_size >= beam_size);
        const DISTANCE_T cutoff = frontier_full ? s_frontier_dist[beam_size - 1] : FLT_MAX;
#if RABITQ_KERNEL_ACTIVE && RABITQ_GLOBAL_APPROX_PRESELECT && RABITQ_NEIGHBOR_APPROX_BUFFER
        {
            bool do_global_preselect = rabitq_filter_gate;
#if RABITQ_GLOBAL_APPROX_PRESELECT_REQUIRE_FRONTIER_FULL
            do_global_preselect = do_global_preselect && frontier_full;
#endif
            if (do_global_preselect && num_neighbors > 1) {
                const int per_expand_keep = min(max(rabitq_topn_iter, 1), 32);
                int keep_n =
                    s_num_expanders * per_expand_keep +
                    max((int)RABITQ_GLOBAL_APPROX_PRESELECT_EXTRA, 0);
                keep_n = max(1, min(keep_n, num_neighbors));
                // Current block-level bitonic preselect path is bounded to 256 items.
                // With the current search-width settings this still covers staged neighbors.
                const int global_sort_cap = min((int)NEIGHBOR_BUFFER_SIZE, 256);
                if (keep_n < num_neighbors) {
                    if (num_neighbors <= global_sort_cap) {
                        int sort_size = 1;
                        while (sort_size < num_neighbors) sort_size <<= 1;
                        if (sort_size < 2) sort_size = 2;

                        for (int i = tid; i < sort_size; i += blockDim.x) {
                            if (i < num_neighbors) {
                                s_rabitq_sort_idx[i] = s_neighbor_idx[i];
                                DISTANCE_T approx = s_neighbor_approx[i];
                                if (!(approx < FLT_MAX) || (approx != approx)) {
                                    approx = FLT_MAX;
                                }
                                s_rabitq_sort_approx[i] = approx;
                            } else {
                                s_rabitq_sort_idx[i] = MAX_INDEX;
                                s_rabitq_sort_approx[i] = FLT_MAX;
                            }
                        }
                        __syncthreads();

                        for (int k = 2; k <= sort_size; k <<= 1) {
                            for (int j = (k >> 1); j > 0; j >>= 1) {
                                for (int i = tid; i < sort_size; i += blockDim.x) {
                                    const int ixj = i ^ j;
                                    if (ixj > i && ixj < sort_size) {
                                        const bool up = ((i & k) == 0);
                                        const DISTANCE_T di = s_rabitq_sort_approx[i];
                                        const DISTANCE_T dj = s_rabitq_sort_approx[ixj];
                                        bool do_swap = false;
                                        if (up) {
                                            do_swap = (di > dj);
                                        } else {
                                            do_swap = (di < dj);
                                        }
                                        if (do_swap) {
                                            s_rabitq_sort_approx[i] = dj;
                                            s_rabitq_sort_approx[ixj] = di;
                                            const INDEX_T vi = s_rabitq_sort_idx[i];
                                            s_rabitq_sort_idx[i] = s_rabitq_sort_idx[ixj];
                                            s_rabitq_sort_idx[ixj] = vi;
                                        }
                                    }
                                }
                                __syncthreads();
                            }
                        }
                    } else {
                        // Large staged-neighbor path: two-stage global shortlist.
                        // 1) Sort each <=256 chunk and keep a small chunk-local top.
                        // 2) Merge chunk-local pool and sort once to global top-k.
                        const int num_chunks =
                            (num_neighbors + global_sort_cap - 1) / global_sort_cap;
                        const int keep_each_base =
                            (keep_n + num_chunks - 1) / num_chunks;
                        const int keep_each_cap = max(1, global_sort_cap / num_chunks);
                        const int keep_each =
                            min(global_sort_cap,
                                min(max(1, keep_each_base + 2), keep_each_cap));

                        if (tid == 0) {
                            s_rabitq_global_pool_count = 0;
                            s_rabitq_global_pool_base = 0;
                        }
                        __syncthreads();

                        for (int chunk = 0; chunk < num_chunks; ++chunk) {
                            const int chunk_begin = chunk * global_sort_cap;
                            const int chunk_len =
                                min(global_sort_cap, num_neighbors - chunk_begin);
                            int sort_size = 1;
                            while (sort_size < chunk_len) sort_size <<= 1;
                            if (sort_size < 2) sort_size = 2;

                            for (int i = tid; i < sort_size; i += blockDim.x) {
                                if (i < chunk_len) {
                                    s_rabitq_sort_idx[i] =
                                        s_neighbor_idx[chunk_begin + i];
                                    DISTANCE_T approx =
                                        s_neighbor_approx[chunk_begin + i];
                                    if (!(approx < FLT_MAX) || (approx != approx)) {
                                        approx = FLT_MAX;
                                    }
                                    s_rabitq_sort_approx[i] = approx;
                                } else {
                                    s_rabitq_sort_idx[i] = MAX_INDEX;
                                    s_rabitq_sort_approx[i] = FLT_MAX;
                                }
                            }
                            __syncthreads();

                            for (int k = 2; k <= sort_size; k <<= 1) {
                                for (int j = (k >> 1); j > 0; j >>= 1) {
                                    for (int i = tid; i < sort_size; i += blockDim.x) {
                                        const int ixj = i ^ j;
                                        if (ixj > i && ixj < sort_size) {
                                            const bool up = ((i & k) == 0);
                                            const DISTANCE_T di =
                                                s_rabitq_sort_approx[i];
                                            const DISTANCE_T dj =
                                                s_rabitq_sort_approx[ixj];
                                            bool do_swap = false;
                                            if (up) {
                                                do_swap = (di > dj);
                                            } else {
                                                do_swap = (di < dj);
                                            }
                                            if (do_swap) {
                                                s_rabitq_sort_approx[i] = dj;
                                                s_rabitq_sort_approx[ixj] = di;
                                                const INDEX_T vi =
                                                    s_rabitq_sort_idx[i];
                                                s_rabitq_sort_idx[i] =
                                                    s_rabitq_sort_idx[ixj];
                                                s_rabitq_sort_idx[ixj] = vi;
                                            }
                                        }
                                    }
                                    __syncthreads();
                                }
                            }

                            const int chunk_keep = min(keep_each, chunk_len);
                            if (tid == 0) {
                                s_rabitq_global_pool_base = s_rabitq_global_pool_count;
                                int next_pool = s_rabitq_global_pool_count + chunk_keep;
                                if (next_pool > (int)NEIGHBOR_BUFFER_SIZE) {
                                    next_pool = (int)NEIGHBOR_BUFFER_SIZE;
                                }
                                s_rabitq_global_pool_count = next_pool;
                            }
                            __syncthreads();

                            const int pool_base = s_rabitq_global_pool_base;
                            const int pool_keep =
                                max(0, min(chunk_keep,
                                           s_rabitq_global_pool_count - pool_base));
                            for (int i = tid; i < pool_keep; i += blockDim.x) {
                                s_neighbor_idx[pool_base + i] = s_rabitq_sort_idx[i];
                                s_neighbor_approx[pool_base + i] =
                                    s_rabitq_sort_approx[i];
                            }
                            __syncthreads();
                        }

                        const int pool_count =
                            min(s_rabitq_global_pool_count,
                                (int)NEIGHBOR_BUFFER_SIZE);
                        int sort_size = 1;
                        while (sort_size < pool_count) sort_size <<= 1;
                        if (sort_size < 2) sort_size = 2;

                        for (int i = tid; i < sort_size; i += blockDim.x) {
                            if (i < pool_count) {
                                s_rabitq_sort_idx[i] = s_neighbor_idx[i];
                                DISTANCE_T approx = s_neighbor_approx[i];
                                if (!(approx < FLT_MAX) || (approx != approx)) {
                                    approx = FLT_MAX;
                                }
                                s_rabitq_sort_approx[i] = approx;
                            } else {
                                s_rabitq_sort_idx[i] = MAX_INDEX;
                                s_rabitq_sort_approx[i] = FLT_MAX;
                            }
                        }
                        __syncthreads();

                        for (int k = 2; k <= sort_size; k <<= 1) {
                            for (int j = (k >> 1); j > 0; j >>= 1) {
                                for (int i = tid; i < sort_size; i += blockDim.x) {
                                    const int ixj = i ^ j;
                                    if (ixj > i && ixj < sort_size) {
                                        const bool up = ((i & k) == 0);
                                        const DISTANCE_T di = s_rabitq_sort_approx[i];
                                        const DISTANCE_T dj = s_rabitq_sort_approx[ixj];
                                        bool do_swap = false;
                                        if (up) {
                                            do_swap = (di > dj);
                                        } else {
                                            do_swap = (di < dj);
                                        }
                                        if (do_swap) {
                                            s_rabitq_sort_approx[i] = dj;
                                            s_rabitq_sort_approx[ixj] = di;
                                            const INDEX_T vi = s_rabitq_sort_idx[i];
                                            s_rabitq_sort_idx[i] = s_rabitq_sort_idx[ixj];
                                            s_rabitq_sort_idx[ixj] = vi;
                                        }
                                    }
                                }
                                __syncthreads();
                            }
                        }
                    }

                    for (int i = tid; i < keep_n; i += blockDim.x) {
                        s_neighbor_idx[i] = s_rabitq_sort_idx[i];
                        s_neighbor_approx[i] = s_rabitq_sort_approx[i];
                    }
                    if (tid == 0) {
                        s_num_neighbors = keep_n;
                    }
                    __syncthreads();
                    num_neighbors = min(s_num_neighbors, (int)NEIGHBOR_BUFFER_SIZE);
                }
            }
        }
#endif
#if RABITQ_KERNEL_ACTIVE && RABITQ_TWO_PHASE_EXACT
        const bool rabitq_two_phase_runtime =
            rabitq_filter_gate && frontier_full &&
            (num_neighbors >= (int)RABITQ_TWO_PHASE_MIN_NEIGHBORS);
#endif
        int dc_warp = 0;
#if USE_SUBWARP_L2_128
        if (dim == 128) {
            int dc_local = 0;
            for (int base = warp_id * 4; base < num_neighbors; base += num_warps * 4) {
                const int group = lane_id >> 3;      // 0..3 (which distance within the warp)
                const int sublane = lane_id & 7;     // 0..7 within the 8-lane group
                const int n = base + group;
                const bool valid = (n < num_neighbors);
                const vid_t neighbor = valid ? (vid_t)s_neighbor_idx[n] : (vid_t)0;
                const float* bptr = valid ? (const float*)(d_data + (size_t)neighbor * (size_t)128)
                                          : (const float*)query;  // safe: yields 0 distance
                const float sum = subwarp8_l2_distance_128_leader((const float*)query, bptr);
                if (sublane == 0 && valid) {
                    s_neighbor_dist[n] = sum;
                }
                if (lane_id == 0) {
                    int remain = num_neighbors - base;
                    int cnt = (remain >= 4) ? 4 : ((remain > 0) ? remain : 0);
                    dc_local += cnt;
                }
            }
            if (lane_id == 0) {
                dc_warp = dc_local;
            }
        } else {
            int dc_local = 0;
            for (int n = warp_id; n < num_neighbors; n += num_warps) {
                vid_t neighbor = s_neighbor_idx[n];
                const T* nvec = d_data + neighbor * dim;
                DISTANCE_T dist = warp_l2_distance(dim, query, nvec);
                if (lane_id == 0) {
                    s_neighbor_dist[n] = dist;
                    dc_local++;
                }
            }
            if (lane_id == 0) {
                dc_warp = dc_local;
            }
        }
#else
        int dc_local = 0;
#if !(RABITQ_KERNEL_ACTIVE && RABITQ_REFLOW_FIXED && RABITQ_GPU_DUALPOOL_ENGINE && \
      !RABITQ_TWO_PHASE_EXACT)
        const int rabitq_exact_batch = max(1, (int)RABITQ_TWO_PHASE_BATCH);
#if RABITQ_KERNEL_ACTIVE && RABITQ_TWO_PHASE_EXACT
        const bool rabitq_use_two_phase_exact = rabitq_two_phase_runtime;
        const DISTANCE_T rabitq_exact_ratio = (DISTANCE_T)RABITQ_TWO_PHASE_APPROX_RATIO;
#else
        constexpr bool rabitq_use_two_phase_exact = false;
        constexpr DISTANCE_T rabitq_exact_ratio = (DISTANCE_T)0;
#endif
#endif
#if RABITQ_KERNEL_ACTIVE && RABITQ_GPU_DUALPOOL_ENGINE
        if (tid == 0) {
            s_rabitq_dualpool_anchor_budget = 0;
        }
        __syncthreads();
        if (tid == 0 &&
            rabitq_filter_gate &&
            ((int)RABITQ_GPU_DUALPOOL_ANCHOR_EXACT > 0) &&
            (num_neighbors > 0)) {
            // Scale anchor budget by popped expanders so pop-batch>1 remains calibrated.
            const int per_pop = max((int)RABITQ_GPU_DUALPOOL_ANCHOR_EXACT, 0);
            int budget = per_pop * max(s_num_expanders, 1);
            if (budget < 1) budget = 1;
            if (budget > num_neighbors) budget = num_neighbors;
            s_rabitq_dualpool_anchor_budget = budget;
        }
        __syncthreads();
#endif
 #if RABITQ_KERNEL_ACTIVE && RABITQ_REFLOW_FIXED && RABITQ_GPU_DUALPOOL_ENGINE && \
     !RABITQ_TWO_PHASE_EXACT
        int exact_limit = num_neighbors;
        if (rabitq_filter_gate) {
            exact_limit =
                min(num_neighbors, max(s_rabitq_dualpool_anchor_budget, 0));
        }
        const DISTANCE_T pass_cutoff = cutoff;

#if !ENABLE_SQ8
        if (dim == 128) {
#if ENABLE_PARTIAL_L2_128
            if (frontier_full) {
                for (int n = warp_id; n < exact_limit; n += num_warps) {
                    vid_t neighbor = s_neighbor_idx[n];
                    const float* nvec128 = (const float*)(d_data + (size_t)neighbor * (size_t)dim);
                    bool computed_full = true;
                    const DISTANCE_T dist =
                        rabitq_fixed_exact_distance_128<true>(
                            (const float*)query, nvec128, pass_cutoff, &computed_full);
                    if (lane_id == 0) {
                        s_neighbor_dist[n] = dist;
                        if (computed_full) {
                            dc_local++;
                        }
                    }
                }
            } else
#endif
            {
                for (int n = warp_id; n < exact_limit; n += num_warps) {
                    vid_t neighbor = s_neighbor_idx[n];
                    const float* nvec128 = (const float*)(d_data + (size_t)neighbor * (size_t)dim);
                    bool computed_full = true;
                    const DISTANCE_T dist =
                        rabitq_fixed_exact_distance_128<false>(
                            (const float*)query, nvec128, pass_cutoff, &computed_full);
                    if (lane_id == 0) {
                        s_neighbor_dist[n] = dist;
                        dc_local++;
                    }
                }
            }
        } else {
            for (int n = warp_id; n < exact_limit; n += num_warps) {
                vid_t neighbor = s_neighbor_idx[n];
                const T* nvec = d_data + neighbor * dim;
                const DISTANCE_T dist = warp_l2_distance(dim, query, nvec);
                if (lane_id == 0) {
                    s_neighbor_dist[n] = dist;
                    dc_local++;
                }
            }
        }
#else
        // Phase-1: exact anchors (dense warp work, no per-item do_exact branch).
        for (int n = warp_id; n < exact_limit; n += num_warps) {
            vid_t neighbor = s_neighbor_idx[n];
            const T* nvec = d_data + neighbor * dim;
            DISTANCE_T dist;
            bool computed_full = true;

            if (dim == 128) {
#if SQ8_DIST_PRUNE_ENABLE
                if (frontier_full && d_data_sq8 != nullptr) {
                    const int8_t* x8 =
                        d_data_sq8 + (size_t)neighbor * (size_t)128;
                    const float approx = warp_l2_distance_sq8_128(s_query_sq8, x8);
                    if (approx < (float)pass_cutoff * (float)SQ8_PRUNE_SLACK) {
                        dist = warp_l2_distance_128(
                            (const float*)query, (const float*)nvec);
                    } else {
                        dist = FLT_MAX;
                        computed_full = false;
                    }
                } else {
#if ENABLE_PARTIAL_L2_128
                    if (frontier_full) {
                        const float partial =
                            warp_l2_distance_128_prefix<PARTIAL_L2_128_LANES>(
                                (const float*)query, (const float*)nvec);
                        if (partial < pass_cutoff) {
                            const float tail =
                                warp_l2_distance_128_suffix<PARTIAL_L2_128_LANES>(
                                    (const float*)query, (const float*)nvec);
                            dist = partial + tail;
                        } else {
                            dist = FLT_MAX;
                            computed_full = false;
                        }
                    } else {
                        dist = warp_l2_distance_128(
                            (const float*)query, (const float*)nvec);
                    }
#else
                    dist = warp_l2_distance_128(
                        (const float*)query, (const float*)nvec);
#endif
                }
#else
#if ENABLE_PARTIAL_L2_128
                if (frontier_full) {
                    const float partial =
                        warp_l2_distance_128_prefix<PARTIAL_L2_128_LANES>(
                            (const float*)query, (const float*)nvec);
                    if (partial < pass_cutoff) {
                        const float tail =
                            warp_l2_distance_128_suffix<PARTIAL_L2_128_LANES>(
                                (const float*)query, (const float*)nvec);
                        dist = partial + tail;
                    } else {
                        dist = FLT_MAX;
                        computed_full = false;
                    }
                } else {
                    dist = warp_l2_distance_128(
                        (const float*)query, (const float*)nvec);
                }
#else
                dist = warp_l2_distance_128(
                    (const float*)query, (const float*)nvec);
#endif
#endif
            } else {
                dist = warp_l2_distance(dim, query, nvec);
            }

            if (lane_id == 0) {
                s_neighbor_dist[n] = dist;
                if (computed_full) {
                    dc_local++;
                }
            }
        }
#endif

        // Phase-2: approximate tail. Fixed-path scoring already clamps invalid
        // values at production time, so tail copy can stay branch-free here.
        for (int n = exact_limit + warp_id; n < num_neighbors; n += num_warps) {
            const DISTANCE_T dist = s_neighbor_approx[n];
            if (lane_id == 0) {
                s_neighbor_dist[n] = dist;
            }
        }
        __syncthreads();
#else
        for (int b = 0; b < num_neighbors;
             b += (rabitq_use_two_phase_exact ? rabitq_exact_batch : num_neighbors)) {
            const int b_end =
                rabitq_use_two_phase_exact ? min(num_neighbors, b + rabitq_exact_batch) : num_neighbors;

#if RABITQ_KERNEL_ACTIVE && RABITQ_TWO_PHASE_EXACT
            if (rabitq_use_two_phase_exact) {
                if (warp_id == 0) {
                    const int n = b + lane_id;
                    const bool valid_lane = (n < b_end);
                    const DISTANCE_T approx = valid_lane ? s_neighbor_approx[n] : FLT_MAX;
                    const bool approx_ok = valid_lane && (approx < FLT_MAX);
                    const uint32_t key_local =
                        approx_ok ? (0xFFFFFFFFu - __float_as_uint((float)approx)) : 0u;
                    const int keep_anchor = min((int)RABITQ_TWO_PHASE_ANCHORS, b_end - b);
                    const uint32_t anchor_mask =
                        warp_select_topk_mask_desc_u32_rabitq(
                            key_local, approx_ok, max(1, keep_anchor));
                    if (lane_id == 0) {
                        s_rabitq_batch_anchor_mask = anchor_mask;
                        s_rabitq_batch_gate = cutoff;
                    }
                }
                __syncthreads();
            }
#endif

            const int pass_count = rabitq_use_two_phase_exact ? 2 : 1;
            for (int pass = 0; pass < pass_count; ++pass) {
                DISTANCE_T pass_cutoff = cutoff;
#if RABITQ_KERNEL_ACTIVE && RABITQ_TWO_PHASE_EXACT
                if (rabitq_use_two_phase_exact && pass == 1) {
                    pass_cutoff = s_rabitq_batch_gate;
                }
#endif

                for (int n = b + warp_id; n < b_end; n += num_warps) {
                    vid_t neighbor = s_neighbor_idx[n];
                    const T* nvec = d_data + neighbor * dim;
                    DISTANCE_T dist;
                    bool computed_full = true;
                    bool do_exact = true;
#if RABITQ_KERNEL_ACTIVE && RABITQ_GPU_DUALPOOL_ENGINE
                    if (rabitq_filter_gate) {
                        // Unified dual-pool semantics:
                        // exact only on popped node; neighbors stay approximate here.
                        bool dualpool_anchor_exact = false;
                        if (((int)RABITQ_GPU_DUALPOOL_ANCHOR_EXACT > 0)) {
                            dualpool_anchor_exact =
                                (n < s_rabitq_dualpool_anchor_budget);
                        }
                        do_exact = dualpool_anchor_exact;
                    }
#endif

#if RABITQ_KERNEL_ACTIVE && RABITQ_TWO_PHASE_EXACT
                    bool is_anchor = false;
                    if (rabitq_use_two_phase_exact) {
                        const int local_n = n - b;
                        is_anchor = (((s_rabitq_batch_anchor_mask >> local_n) & 1u) != 0u);
                        if (pass == 0) {
                            do_exact = is_anchor;
                        } else {
                            if (is_anchor) {
                                do_exact = false;
                            } else if (rabitq_exact_ratio > (DISTANCE_T)0.0f) {
                                const DISTANCE_T approx_d = s_neighbor_approx[n];
                                const DISTANCE_T gate = pass_cutoff * rabitq_exact_ratio;
                                if ((approx_d < FLT_MAX) && !(approx_d <= gate)) {
                                    do_exact = false;
                                }
                            }
                        }
                    }
#endif

                    if (do_exact) {
                        if (dim == 128) {
#if ENABLE_SQ8
#if SQ8_DIST_PRUNE_ENABLE
                            if (frontier_full && d_data_sq8 != nullptr) {
                                const int8_t* x8 = d_data_sq8 + (size_t)neighbor * (size_t)128;
                                const float approx = warp_l2_distance_sq8_128(s_query_sq8, x8);
                                if (approx < (float)pass_cutoff * (float)SQ8_PRUNE_SLACK) {
                                    dist = warp_l2_distance_128((const float*)query, (const float*)nvec);
                                } else {
                                    dist = FLT_MAX;
                                    computed_full = false;
                                }
                            } else {
#if ENABLE_PARTIAL_L2_128
                                if (frontier_full) {
                                    const float partial =
                                        warp_l2_distance_128_prefix<PARTIAL_L2_128_LANES>(
                                            (const float*)query, (const float*)nvec);
                                    if (partial < pass_cutoff) {
                                        const float tail =
                                            warp_l2_distance_128_suffix<PARTIAL_L2_128_LANES>(
                                                (const float*)query, (const float*)nvec);
                                        dist = partial + tail;
                                    } else {
                                        dist = FLT_MAX;
                                        computed_full = false;
                                    }
                                } else {
                                    dist = warp_l2_distance_128((const float*)query, (const float*)nvec);
                                }
#else
                                dist = warp_l2_distance_128((const float*)query, (const float*)nvec);
#endif
                            }
#else
#if ENABLE_PARTIAL_L2_128
                            if (frontier_full) {
                                const float partial =
                                    warp_l2_distance_128_prefix<PARTIAL_L2_128_LANES>(
                                        (const float*)query, (const float*)nvec);
                                if (partial < pass_cutoff) {
                                    const float tail =
                                        warp_l2_distance_128_suffix<PARTIAL_L2_128_LANES>(
                                            (const float*)query, (const float*)nvec);
                                    dist = partial + tail;
                                } else {
                                    dist = FLT_MAX;
                                    computed_full = false;
                                }
                            } else {
                                dist = warp_l2_distance_128((const float*)query, (const float*)nvec);
                            }
#else
                            dist = warp_l2_distance_128((const float*)query, (const float*)nvec);
#endif
#endif
#else
#if ENABLE_PARTIAL_L2_128
                            if (frontier_full) {
                                const float partial =
                                    warp_l2_distance_128_prefix<PARTIAL_L2_128_LANES>(
                                        (const float*)query, (const float*)nvec);
                                if (partial < pass_cutoff) {
                                    const float tail =
                                        warp_l2_distance_128_suffix<PARTIAL_L2_128_LANES>(
                                            (const float*)query, (const float*)nvec);
                                    dist = partial + tail;
                                } else {
                                    dist = FLT_MAX;
                                    computed_full = false;
                                }
                            } else {
                                dist = warp_l2_distance_128((const float*)query, (const float*)nvec);
                            }
#else
                            dist = warp_l2_distance_128((const float*)query, (const float*)nvec);
#endif
#endif
                        } else {
                            dist = warp_l2_distance(dim, query, nvec);
                        }
                    } else {
 #if RABITQ_KERNEL_ACTIVE && RABITQ_GPU_DUALPOOL_ENGINE
                        dist = s_neighbor_approx[n];
                        if (!(dist < FLT_MAX) || (dist != dist)) {
                            dist = FLT_MAX;
                        }
 #else
                        dist = FLT_MAX;
 #endif
                        computed_full = false;
                    }
                    if (lane_id == 0) {
                        if (!(rabitq_use_two_phase_exact && pass == 1 &&
#if RABITQ_KERNEL_ACTIVE && RABITQ_TWO_PHASE_EXACT
                              (((s_rabitq_batch_anchor_mask >> (n - b)) & 1u) != 0u)
#else
                              false
#endif
                              )) {
                            s_neighbor_dist[n] = dist;
                            if (computed_full) {
                                dc_local++;
                            }
                        }
                    }
                }

                __syncthreads();
#if RABITQ_KERNEL_ACTIVE && RABITQ_TWO_PHASE_EXACT
                if (rabitq_use_two_phase_exact && pass == 0) {
                    if (tid == 0) {
                        DISTANCE_T diff_min = FLT_MAX;
                        bool have_anchor = false;
                        int anchor_hits = 0;
                        for (int n = b; n < b_end; ++n) {
                            const int local_n = n - b;
                            if (((s_rabitq_batch_anchor_mask >> local_n) & 1u) == 0u) continue;
                            const DISTANCE_T d = s_neighbor_dist[n];
                            const DISTANCE_T a = s_neighbor_approx[n];
                            if (!(d < FLT_MAX) || !(a < FLT_MAX)) continue;
                            const DISTANCE_T diff = d - a;
                            if (!have_anchor || diff < diff_min) diff_min = diff;
                            if (d <= cutoff) anchor_hits++;
                            have_anchor = true;
                        }
                        DISTANCE_T gate = cutoff;
                        if (have_anchor && (diff_min < FLT_MAX)) {
                            gate = cutoff - diff_min;
                            if (!(gate > (DISTANCE_T)0.0f) || (gate >= FLT_MAX)) {
                                gate = cutoff;
                            }
                        }
                        // Recall guard:
                        // if anchor calibration misses all in-cutoff points, disable pass1
                        // approx gate for this batch to avoid false-prune cascades.
                        if (!have_anchor || anchor_hits == 0) {
                            gate = FLT_MAX;
                        } else if (anchor_hits == 1 && gate < cutoff) {
                            gate = cutoff;
                        }
                        s_rabitq_batch_gate = gate;
                    }
                    __syncthreads();
                }
#endif
            }
        }
#endif
        if (lane_id == 0) {
            dc_warp = dc_local;
        }
#endif
        if (lane_id == 0) {
            s_pick_counts[warp_id] = dc_warp;
        }
        __syncthreads();
        if (tid == 0) {
            int dc_sum = 0;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                if (w < num_warps) dc_sum += s_pick_counts[w];
            }
            s_dc_count += dc_sum;
        }
        __syncthreads();
#if RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_PARITY_MODE && \
    RABITQ_CPU_PARITY_EXACT_POOL
#if RABITQ_CPU_PARITY_REALTIME_MERGE
        if (tid == 0 && s_exact_stream_size > 0) {
            // Real-time parity feedback: exact pop distances are merged before
            // candidate cutoff/admission in this same iteration.
            int exact_cap = max(1, min(K, beam_size));
#if RABITQ_CPU_PARITY_FEEDBACK_CAP > 0
            exact_cap = min(exact_cap, (int)RABITQ_CPU_PARITY_FEEDBACK_CAP);
            exact_cap = max(exact_cap, 1);
#endif
            rabitq_parity_merge_exact_stream<RABITQ_CPU_PARITY_STREAM_CAP>(
                s_exact_topk_idx,
                s_exact_topk_dist,
                &s_exact_topk_size,
                s_exact_stream_idx,
                s_exact_stream_dist,
                &s_exact_stream_size,
                exact_cap,
                kIndexMsbMask);
        }
        __syncthreads();
#endif
#endif
        DISTANCE_T candidate_cutoff = cutoff;
#if RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_PARITY_MODE
#if RABITQ_CPU_PARITY_USE_EXACT_CUTOFF
        if (frontier_full) {
            int exact_cap = max(1, min(K, beam_size));
#if RABITQ_CPU_PARITY_FEEDBACK_CAP > 0
            exact_cap = min(exact_cap, (int)RABITQ_CPU_PARITY_FEEDBACK_CAP);
            exact_cap = max(exact_cap, 1);
#endif
            if (s_exact_topk_size >= exact_cap) {
                const DISTANCE_T exact_cutoff = s_exact_topk_dist[exact_cap - 1];
                if (exact_cutoff < candidate_cutoff) {
                    candidate_cutoff = exact_cutoff;
                }
            }
        }
#endif
#endif
        for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
            s_candidate_idx[i] = MAX_INDEX;
            s_candidate_dist[i] = FLT_MAX;
        }
        if (tid == 0) s_candidate_count = 0;
        if (tid == 0) s_candidate_sorted = 0;
        __syncthreads();
        if (num_neighbors == 1) {
            if (tid == 0) {
                const DISTANCE_T dist = s_neighbor_dist[0];
                const bool keep_one =
                    (dist < candidate_cutoff || s_frontier_size < beam_size);
                if (keep_one) {
                    s_candidate_idx[0] = s_neighbor_idx[0];
                    s_candidate_dist[0] = dist;
                    s_candidate_count = 1;
                } else {
                    s_candidate_idx[0] = MAX_INDEX;
                    s_candidate_dist[0] = FLT_MAX;
                    s_candidate_count = 0;
                }
                s_candidate_sorted = 1;
            }
        } else {
 #if CANDIDATE_SELECT_STRICT == 1
        const bool strict_active =
            (CANDIDATE_SELECT_STRICT_ITERS <= 0) || (iter < (int)CANDIDATE_SELECT_STRICT_ITERS);
        if (strict_active) {
            if (tid == 0) {
                int nc = 0;
                for (int n = 0; n < num_neighbors; n++) {
                    const DISTANCE_T dist = s_neighbor_dist[n];
                    if (!((dist < candidate_cutoff ||
                           s_frontier_size < beam_size))) {
                        continue;
                    }
                    const INDEX_T idx = s_neighbor_idx[n];
                    if (nc < CANDIDATE_BUFFER_SIZE) {
                        s_candidate_idx[nc] = idx;
                        s_candidate_dist[nc] = dist;
                        nc++;
                    } else {
                        int worst_pos = 0;
                        DISTANCE_T worst_dist = s_candidate_dist[0];
                        #pragma unroll
                        for (int i = 1; i < CANDIDATE_BUFFER_SIZE; i++) {
                            if (s_candidate_dist[i] > worst_dist) {
                                worst_dist = s_candidate_dist[i];
                                worst_pos = i;
                            }
                        }
                        if (dist < worst_dist) {
                            s_candidate_idx[worst_pos] = idx;
                            s_candidate_dist[worst_pos] = dist;
                        }
                    }
                }
                s_candidate_count = nc;
            }
        } else {
            for (int base = warp_id * 32; base < num_neighbors; base += num_warps * 32) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
                if (*((volatile int*)&s_candidate_count) >= CANDIDATE_BUFFER_SIZE) break;
#endif
                int n = base + lane_id;
                bool valid = (n < num_neighbors);
                DISTANCE_T dist = valid ? s_neighbor_dist[n] : FLT_MAX;
                bool keep =
                    valid && (dist < candidate_cutoff || s_frontier_size < beam_size);
#if SKIP_ATOMIC_WHEN_POOL_FULL
                keep = keep && (*((volatile int*)&s_candidate_count) < CANDIDATE_BUFFER_SIZE);
#endif
                uint32_t ballot = __ballot_sync(FULL_MASK, keep);
                int warp_count = __popc(ballot);
                int base_offset = 0;
                if (lane_id == 0 && warp_count > 0) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
                    if (*((volatile int*)&s_candidate_count) < CANDIDATE_BUFFER_SIZE) {
                        base_offset = atomicAdd(&s_candidate_count, warp_count);
                    } else {
                        base_offset = CANDIDATE_BUFFER_SIZE;
                    }
#else
                    base_offset = atomicAdd(&s_candidate_count, warp_count);
#endif
                }
                base_offset = __shfl_sync(FULL_MASK, base_offset, 0);
                if (keep) {
                    int my_offset = __popc(ballot & ((1u << lane_id) - 1));
                    int slot = base_offset + my_offset;
                    if (slot < CANDIDATE_BUFFER_SIZE) {
                        s_candidate_idx[slot] = s_neighbor_idx[n];
                        s_candidate_dist[slot] = dist;
                    }
                }
            }
        }
#elif CANDIDATE_SELECT_STRICT == 2
        {
            const bool strict_active =
                (CANDIDATE_SELECT_STRICT_ITERS <= 0) || (iter < (int)CANDIDATE_SELECT_STRICT_ITERS);
            if (strict_active) {
#if CANDIDATE_SELECT_STRICT_FAST
#if CANDIDATE_SELECT_STRICT_FAST_DGS_ONLY
#if ENABLE_DGS
                const bool strict_fast_runtime = use_wide_expand;
#else
                constexpr bool strict_fast_runtime = false;
#endif
#else
                constexpr bool strict_fast_runtime = true;
#endif
#if STRICT_FAST_DEBUG_PRINT
                if (tid == 0) {
                    s_strict_total_active_iters++;
                    if (strict_fast_runtime) s_strict_fast_active_iters++;
                }
#endif
                if (strict_fast_runtime) {
                constexpr int kNeighborCap = (int)NEIGHBOR_BUFFER_SIZE;
                const int n_local = warp_id * 32 + lane_id;
                const bool in_range_local = (n_local < num_neighbors);
                const DISTANCE_T dist_local = in_range_local ? s_neighbor_dist[n_local] : FLT_MAX;
                const bool keep_local =
                    in_range_local && (dist_local < candidate_cutoff || s_frontier_size < beam_size);
                const uint32_t keep_ballot = __ballot_sync(FULL_MASK, keep_local);
                if (lane_id == 0) {
                    s_pick_counts[warp_id] = __popc(keep_ballot);
                }
                __syncthreads();

                if (tid == 0) {
                    int vc = 0;
                    #pragma unroll
                    for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                        if (w < num_warps) vc += s_pick_counts[w];
                    }
                    s_candidate_tmp_count = min(vc, kNeighborCap);
                }
                __syncthreads();

                int valid_count = min(s_candidate_tmp_count, kNeighborCap);
                bool use_strict_select = true;
#if CANDIDATE_SELECT_STRICT_ADAPTIVE
                {
                    const int min_valid = max(1, (int)CANDIDATE_SELECT_STRICT_ADAPTIVE_MIN_VALID);
                    use_strict_select = (valid_count >= min_valid);
                }
#endif

                if (use_strict_select) {
                    constexpr int kWarpTop = CANDIDATE_BUFFER_SIZE / WARPS_PER_BLOCK;
                    const int warp_base = warp_id * kWarpTop;
#if STRICT_FAST_DEBUG_PRINT
                    if (tid == 0) {
                        s_strict_fast_valid_sum += valid_count;
                        s_strict_fast_low_iters++;
                    }
#endif
#if STRICT_FAST_SECONDARY_SORT_MODE == 0
                    if (lane_id < kWarpTop) {
                        s_candidate_idx[warp_base + lane_id] = MAX_INDEX;
                        s_candidate_dist[warp_base + lane_id] = FLT_MAX;
                    }
                    __syncwarp();
#endif

                    {
#if STRICT_FAST_WARP_TOPK_MODE == 0
                        const uint32_t kInfKey = __float_as_uint((float)FLT_MAX);
                        uint32_t key = keep_local ? __float_as_uint((float)dist_local) : kInfKey;
                        uint32_t val = keep_local ? (uint32_t)s_neighbor_idx[n_local] : (uint32_t)MAX_INDEX;
                        warp_bitonic_sort_u32_asc(key, val);
                        if ((lane_id < kWarpTop) && (val != (uint32_t)MAX_INDEX)) {
                            s_candidate_idx[warp_base + lane_id] = (INDEX_T)val;
                            s_candidate_dist[warp_base + lane_id] = (DISTANCE_T)__uint_as_float(key);
                        }
#else
                        const int warp_keep = s_pick_counts[warp_id];
                        if (warp_keep > 0) {
                            float work_key = keep_local ? (float)dist_local : FLT_MAX;
                            uint32_t work_val = keep_local ? (uint32_t)s_neighbor_idx[n_local] : (uint32_t)MAX_INDEX;
                            const int take = min(kWarpTop, warp_keep);
                            #pragma unroll
                            for (int t = 0; t < kWarpTop; t++) {
                                if (t >= take) break;
                                float min_key = work_key;
                                uint32_t min_val = work_val;
                                int min_lane = lane_id;
                                warp_min_reduce_f32_u32_lane(min_key, min_val, min_lane);
                                const float best_key = __shfl_sync(FULL_MASK, min_key, 0);
                                const uint32_t best_val = __shfl_sync(FULL_MASK, min_val, 0);
                                const int best_lane = __shfl_sync(FULL_MASK, min_lane, 0);
                                if ((best_val == (uint32_t)MAX_INDEX) || (best_key == FLT_MAX)) {
                                    break;
                                }
                                if (lane_id == 0) {
                                    s_candidate_idx[warp_base + t] = (INDEX_T)best_val;
                                    s_candidate_dist[warp_base + t] = (DISTANCE_T)best_key;
                                }
                                if (lane_id == best_lane) {
                                    work_key = FLT_MAX;
                                    work_val = (uint32_t)MAX_INDEX;
                                }
                            }
                        }
#endif
                    }
                    __syncwarp();
                    __syncthreads();

#if STRICT_FAST_SECONDARY_SORT_MODE == 0
                    if (warp_id == 0) {
                        candidate_by_bitonic_sort<2, 4>(s_candidate_idx, s_candidate_dist, CANDIDATE_BUFFER_SIZE);
                    }
                    __syncthreads();

                    if (tid == 0) {
                        int nc = 0;
                        while (nc < CANDIDATE_BUFFER_SIZE && s_candidate_idx[nc] != MAX_INDEX) {
                            nc++;
                        }
                        s_candidate_count = nc;
                        s_candidate_sorted = 1;
                    }
#else
#if STRICT_FAST_SECONDARY_MERGE_MODE == 0
                    if (tid == 0) {
                        int pos[WARPS_PER_BLOCK];
                        int len[WARPS_PER_BLOCK];

                        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                            if (w < num_warps) {
                                const int c = s_pick_counts[w];
                                len[w] = (c < kWarpTop) ? c : kWarpTop;
                            } else {
                                len[w] = 0;
                            }
                            pos[w] = 0;
                        }

                        int nc = 0;
                        for (; nc < CANDIDATE_BUFFER_SIZE; nc++) {
                            int best_w = -1;
                            DISTANCE_T best_d = FLT_MAX;
                            for (int w = 0; w < num_warps; w++) {
                                if (pos[w] >= len[w]) continue;
                                const int src = w * kWarpTop + pos[w];
                                const DISTANCE_T d = s_candidate_dist[src];
                                if (d < best_d) {
                                    best_d = d;
                                    best_w = w;
                                }
                            }
                            if (best_w < 0) break;

                            const int src = best_w * kWarpTop + pos[best_w];
                            s_candidate_tmp_idx[nc] = s_candidate_idx[src];
                            s_candidate_tmp_dist[nc] = s_candidate_dist[src];
                            pos[best_w]++;
                        }

                        s_candidate_count = nc;
                        s_candidate_sorted = 1;
                    }
#else
                    if (warp_id == 0) {
                        int my_len = 0;
                        int my_pos = 0;
                        int merge_width = 1;
                        while (merge_width < num_warps && merge_width < 32) merge_width <<= 1;
                        const uint32_t merge_mask =
                            (merge_width >= 32) ? FULL_MASK : ((1u << merge_width) - 1u);
                        if (lane_id < num_warps) {
                            const int c = s_pick_counts[lane_id];
                            my_len = (c < kWarpTop) ? c : kWarpTop;
                        }

                        if (lane_id < merge_width) {
                            int nc = 0;
                            for (; nc < CANDIDATE_BUFFER_SIZE; nc++) {
                                float local_key = FLT_MAX;
                                uint32_t local_idx = (uint32_t)MAX_INDEX;
                                if (lane_id < num_warps && my_pos < my_len) {
                                    const int src = lane_id * kWarpTop + my_pos;
                                    local_key = (float)s_candidate_dist[src];
                                    local_idx = (uint32_t)s_candidate_idx[src];
                                }

                                int local_lane = lane_id;
                                warp_min_reduce_f32_u32_lane_masked(
                                    local_key, local_idx, local_lane, merge_mask, merge_width);

                                const float best_key = __shfl_sync(merge_mask, local_key, 0);
                                const uint32_t best_idx = __shfl_sync(merge_mask, local_idx, 0);
                                const int best_lane = __shfl_sync(merge_mask, local_lane, 0);
                                const bool done =
                                    (best_key == FLT_MAX) ||
                                    (best_idx == (uint32_t)MAX_INDEX) ||
                                    (best_lane >= num_warps);

                                if (done) break;

                                if (lane_id == 0) {
                                    s_candidate_tmp_idx[nc] = (INDEX_T)best_idx;
                                    s_candidate_tmp_dist[nc] = (DISTANCE_T)best_key;
                                }
                                if (lane_id == best_lane) {
                                    my_pos++;
                                }
                            }

                            if (lane_id == 0) {
                                s_candidate_count = nc;
                                s_candidate_sorted = 1;
                            }
                        }
                    }
#endif

                    __syncthreads();

                    const int nc_shared = s_candidate_count;
                    for (int i = tid; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
                        if (i < nc_shared) {
                            s_candidate_idx[i] = s_candidate_tmp_idx[i];
                            s_candidate_dist[i] = s_candidate_tmp_dist[i];
                        } else {
                            s_candidate_idx[i] = MAX_INDEX;
                            s_candidate_dist[i] = FLT_MAX;
                        }
                    }
                    __syncthreads();
#endif
                } else {
                    if (tid == 0) s_pick_num = 0;
                    __syncthreads();

                    int padded_neighbors = num_neighbors;
                    if (padded_neighbors % blockDim.x) {
                        padded_neighbors += blockDim.x - (padded_neighbors % blockDim.x);
                    }

                    for (int base = 0; base < padded_neighbors; base += blockDim.x) {
                        const int n = base + tid;
                        const bool valid = (n < num_neighbors);
                        const DISTANCE_T dist = valid ? s_neighbor_dist[n] : FLT_MAX;
                        const bool keep =
                            valid && (dist < candidate_cutoff || s_frontier_size < beam_size);

                        const uint32_t ballot = __ballot_sync(FULL_MASK, keep);
                        const int warp_count = __popc(ballot);
                        int round_base = 0;
                        if (lane_id == 0 && warp_count > 0) {
                            round_base = atomicAdd(&s_pick_num, warp_count);
                        }
                        round_base = __shfl_sync(FULL_MASK, round_base, 0);
                        const uint32_t lane_mask = (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
                        const int lane_prefix = __popc(ballot & lane_mask);
                        const int slot = round_base + lane_prefix;

                        if (keep && slot < kNeighborCap) {
                            s_candidate_tmp_idx[slot] = s_neighbor_idx[n];
                            s_candidate_tmp_dist[slot] = dist;
                        }
                    }

                    __syncthreads();
                    if (tid == 0) {
                        s_candidate_tmp_count = min(s_pick_num, kNeighborCap);
                    }
                    __syncthreads();

                    const int fallback_valid_count = s_candidate_tmp_count;
                    if (tid == 0) {
                        int nc = 0;
                        for (int i = 0; i < fallback_valid_count && nc < CANDIDATE_BUFFER_SIZE; i++) {
                            if (s_candidate_tmp_idx[i] != MAX_INDEX) {
                                s_candidate_idx[nc] = s_candidate_tmp_idx[i];
                                s_candidate_dist[nc] = s_candidate_tmp_dist[i];
                                nc++;
                            }
                        }
                        s_candidate_count = nc;
                        s_candidate_sorted = 0;
                        for (int i = nc; i < CANDIDATE_BUFFER_SIZE; i++) {
                            s_candidate_idx[i] = MAX_INDEX;
                            s_candidate_dist[i] = FLT_MAX;
                        }
                    }
                }
                } else {
                    constexpr int kNeighborCap = (int)NEIGHBOR_BUFFER_SIZE;
                    if (tid == 0) s_pick_num = 0;
                    __syncthreads();

                    int padded_neighbors = num_neighbors;
                    if (padded_neighbors % blockDim.x) {
                        padded_neighbors += blockDim.x - (padded_neighbors % blockDim.x);
                    }

                    for (int base = 0; base < padded_neighbors; base += blockDim.x) {
                        const int n = base + tid;
                        const bool valid = (n < num_neighbors);
                        const DISTANCE_T dist = valid ? s_neighbor_dist[n] : FLT_MAX;
                        const bool keep =
                            valid && (dist < candidate_cutoff || s_frontier_size < beam_size);

                        const uint32_t ballot = __ballot_sync(FULL_MASK, keep);
                        const int warp_count = __popc(ballot);
                        int round_base = 0;
                        if (lane_id == 0 && warp_count > 0) {
                            round_base = atomicAdd(&s_pick_num, warp_count);
                        }
                        round_base = __shfl_sync(FULL_MASK, round_base, 0);
                        const uint32_t lane_mask = (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
                        const int lane_prefix = __popc(ballot & lane_mask);
                        const int slot = round_base + lane_prefix;

                        if (keep && slot < kNeighborCap) {
                            s_candidate_tmp_idx[slot] = s_neighbor_idx[n];
                            s_candidate_tmp_dist[slot] = dist;
                        }
                    }

                    __syncthreads();
                    if (tid == 0) {
                        s_candidate_tmp_count = min(s_pick_num, kNeighborCap);
                    }
                    __syncthreads();

                    int valid_count = s_candidate_tmp_count;
                    bool use_strict_sort = true;
#if CANDIDATE_SELECT_STRICT_ADAPTIVE
                    {
                        const int min_valid = max(1, (int)CANDIDATE_SELECT_STRICT_ADAPTIVE_MIN_VALID);
                        use_strict_sort = (valid_count >= min_valid);
                    }
#endif
                    if (use_strict_sort) {
#if STRICT_TOPK_PARTIAL_SELECT_MODE == 0
                        int sort_size = 1;
                        while (sort_size < valid_count) sort_size <<= 1;
                        if (sort_size < 2) sort_size = 2;

                        for (int i = tid; i < sort_size; i += blockDim.x) {
                            if (i >= valid_count) {
                                s_candidate_tmp_idx[i] = MAX_INDEX;
                                s_candidate_tmp_dist[i] = FLT_MAX;
                            }
                        }
                        __syncthreads();

                        for (int k = 2; k <= sort_size; k <<= 1) {
                            for (int j = (k >> 1); j > 0; j >>= 1) {
                                for (int i = tid; i < sort_size; i += blockDim.x) {
                                    const int ixj = i ^ j;
                                    if (ixj > i && ixj < sort_size) {
                                        const bool up = ((i & k) == 0);
                                        const DISTANCE_T di = s_candidate_tmp_dist[i];
                                        const DISTANCE_T dj = s_candidate_tmp_dist[ixj];
                                        bool do_swap = false;
                                        if (up) {
                                            do_swap = (di > dj);
                                        } else {
                                            do_swap = (di < dj);
                                        }
                                        if (do_swap) {
                                            s_candidate_tmp_dist[i] = dj;
                                            s_candidate_tmp_dist[ixj] = di;
                                            const INDEX_T vi = s_candidate_tmp_idx[i];
                                            s_candidate_tmp_idx[i] = s_candidate_tmp_idx[ixj];
                                            s_candidate_tmp_idx[ixj] = vi;
                                        }
                                    }
                                }
                                __syncthreads();
                            }
                        }

                        if (tid == 0) {
                            int nc = 0;
                            for (int i = 0; i < valid_count && nc < CANDIDATE_BUFFER_SIZE; i++) {
                                if (s_candidate_tmp_idx[i] != MAX_INDEX) {
                                    s_candidate_idx[nc] = s_candidate_tmp_idx[i];
                                    s_candidate_dist[nc] = s_candidate_tmp_dist[i];
                                    nc++;
                                }
                            }
                            s_candidate_count = nc;
                            s_candidate_sorted = 1;
                            for (int i = nc; i < CANDIDATE_BUFFER_SIZE; i++) {
                                s_candidate_idx[i] = MAX_INDEX;
                                s_candidate_dist[i] = FLT_MAX;
                            }
                        }
#else
                        block_select_topk_from_candidate_tmp_exact(
                            s_candidate_tmp_idx,
                            s_candidate_tmp_dist,
                            valid_count,
                            num_warps,
                            s_candidate_idx,
                            s_candidate_dist,
                            &s_candidate_count);
                        if (tid == 0) {
                            s_candidate_sorted = 1;
                        }
#endif
                    } else {
                        if (tid == 0) {
                            int nc = 0;
                            for (int i = 0; i < valid_count && nc < CANDIDATE_BUFFER_SIZE; i++) {
                                if (s_candidate_tmp_idx[i] != MAX_INDEX) {
                                    s_candidate_idx[nc] = s_candidate_tmp_idx[i];
                                    s_candidate_dist[nc] = s_candidate_tmp_dist[i];
                                    nc++;
                                }
                            }
                            s_candidate_count = nc;
                            s_candidate_sorted = 0;
                            for (int i = nc; i < CANDIDATE_BUFFER_SIZE; i++) {
                                s_candidate_idx[i] = MAX_INDEX;
                                s_candidate_dist[i] = FLT_MAX;
                            }
                        }
                    }
                }
#else
                constexpr int kNeighborCap = (int)NEIGHBOR_BUFFER_SIZE;
                if (tid == 0) s_pick_num = 0;
                __syncthreads();

                int padded_neighbors = num_neighbors;
                if (padded_neighbors % blockDim.x) {
                    padded_neighbors += blockDim.x - (padded_neighbors % blockDim.x);
                }

                for (int base = 0; base < padded_neighbors; base += blockDim.x) {
                    const int n = base + tid;
                    const bool valid = (n < num_neighbors);
                    const DISTANCE_T dist = valid ? s_neighbor_dist[n] : FLT_MAX;
                    const bool keep =
                        valid && (dist < candidate_cutoff || s_frontier_size < beam_size);

                    const uint32_t ballot = __ballot_sync(FULL_MASK, keep);
                    const int warp_count = __popc(ballot);
                    int round_base = 0;
                    if (lane_id == 0 && warp_count > 0) {
                        round_base = atomicAdd(&s_pick_num, warp_count);
                    }
                    round_base = __shfl_sync(FULL_MASK, round_base, 0);
                    const uint32_t lane_mask = (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
                    const int lane_prefix = __popc(ballot & lane_mask);
                    const int slot = round_base + lane_prefix;

                    if (keep && slot < kNeighborCap) {
                        s_candidate_tmp_idx[slot] = s_neighbor_idx[n];
                        s_candidate_tmp_dist[slot] = dist;
                    }
                }

                __syncthreads();
                if (tid == 0) {
                    s_candidate_tmp_count = min(s_pick_num, kNeighborCap);
                }
                __syncthreads();

                int valid_count = s_candidate_tmp_count;
                bool use_strict_sort = true;
#if CANDIDATE_SELECT_STRICT_ADAPTIVE
                {
                    const int min_valid = max(1, (int)CANDIDATE_SELECT_STRICT_ADAPTIVE_MIN_VALID);
                    use_strict_sort = (valid_count >= min_valid);
                }
#endif
                if (use_strict_sort) {
#if STRICT_TOPK_PARTIAL_SELECT_MODE == 0
                    int sort_size = 1;
                    while (sort_size < valid_count) sort_size <<= 1;
                    if (sort_size < 2) sort_size = 2;

                    for (int i = tid; i < sort_size; i += blockDim.x) {
                        if (i >= valid_count) {
                            s_candidate_tmp_idx[i] = MAX_INDEX;
                            s_candidate_tmp_dist[i] = FLT_MAX;
                        }
                    }
                    __syncthreads();

                    for (int k = 2; k <= sort_size; k <<= 1) {
                        for (int j = (k >> 1); j > 0; j >>= 1) {
                            for (int i = tid; i < sort_size; i += blockDim.x) {
                                const int ixj = i ^ j;
                                if (ixj > i && ixj < sort_size) {
                                    const bool up = ((i & k) == 0);
                                    const DISTANCE_T di = s_candidate_tmp_dist[i];
                                    const DISTANCE_T dj = s_candidate_tmp_dist[ixj];
                                    bool do_swap = false;
                                    if (up) {
                                        do_swap = (di > dj);
                                    } else {
                                        do_swap = (di < dj);
                                    }
                                    if (do_swap) {
                                        s_candidate_tmp_dist[i] = dj;
                                        s_candidate_tmp_dist[ixj] = di;
                                        const INDEX_T vi = s_candidate_tmp_idx[i];
                                        s_candidate_tmp_idx[i] = s_candidate_tmp_idx[ixj];
                                        s_candidate_tmp_idx[ixj] = vi;
                                    }
                                }
                            }
                            __syncthreads();
                        }
                    }

                    if (tid == 0) {
                        int nc = 0;
                        for (int i = 0; i < valid_count && nc < CANDIDATE_BUFFER_SIZE; i++) {
                            if (s_candidate_tmp_idx[i] != MAX_INDEX) {
                                s_candidate_idx[nc] = s_candidate_tmp_idx[i];
                                s_candidate_dist[nc] = s_candidate_tmp_dist[i];
                                nc++;
                            }
                        }
                        s_candidate_count = nc;
                        s_candidate_sorted = 1;
                        for (int i = nc; i < CANDIDATE_BUFFER_SIZE; i++) {
                            s_candidate_idx[i] = MAX_INDEX;
                            s_candidate_dist[i] = FLT_MAX;
                        }
                    }
#else
                    block_select_topk_from_candidate_tmp_exact(
                        s_candidate_tmp_idx,
                        s_candidate_tmp_dist,
                        valid_count,
                        num_warps,
                        s_candidate_idx,
                        s_candidate_dist,
                        &s_candidate_count);
                    if (tid == 0) {
                        s_candidate_sorted = 1;
                    }
#endif
                } else {
                    if (tid == 0) {
                        int nc = 0;
                        for (int i = 0; i < valid_count && nc < CANDIDATE_BUFFER_SIZE; i++) {
                            if (s_candidate_tmp_idx[i] != MAX_INDEX) {
                                s_candidate_idx[nc] = s_candidate_tmp_idx[i];
                                s_candidate_dist[nc] = s_candidate_tmp_dist[i];
                                nc++;
                            }
                        }
                        s_candidate_count = nc;
                        s_candidate_sorted = 0;
                        for (int i = nc; i < CANDIDATE_BUFFER_SIZE; i++) {
                            s_candidate_idx[i] = MAX_INDEX;
                            s_candidate_dist[i] = FLT_MAX;
                        }
                    }
                }
#endif
            } else {
                for (int base = warp_id * 32; base < num_neighbors; base += num_warps * 32) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
                    if (*((volatile int*)&s_candidate_count) >= CANDIDATE_BUFFER_SIZE) break;
#endif
                    int n = base + lane_id;
                    bool valid = (n < num_neighbors);
                    DISTANCE_T dist = valid ? s_neighbor_dist[n] : FLT_MAX;
                    bool keep =
                        valid && (dist < candidate_cutoff || s_frontier_size < beam_size);
#if SKIP_ATOMIC_WHEN_POOL_FULL
                    keep = keep && (*((volatile int*)&s_candidate_count) < CANDIDATE_BUFFER_SIZE);
#endif
                    uint32_t ballot = __ballot_sync(FULL_MASK, keep);
                    int warp_count = __popc(ballot);
                    int base_offset = 0;
                    if (lane_id == 0 && warp_count > 0) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
                        if (*((volatile int*)&s_candidate_count) < CANDIDATE_BUFFER_SIZE) {
                            base_offset = atomicAdd(&s_candidate_count, warp_count);
                        } else {
                            base_offset = CANDIDATE_BUFFER_SIZE;
                        }
#else
                        base_offset = atomicAdd(&s_candidate_count, warp_count);
#endif
                    }
                    base_offset = __shfl_sync(FULL_MASK, base_offset, 0);
                    if (keep) {
                        int my_offset = __popc(ballot & ((1u << lane_id) - 1));
                        int slot = base_offset + my_offset;
                        if (slot < CANDIDATE_BUFFER_SIZE) {
                            s_candidate_idx[slot] = s_neighbor_idx[n];
                            s_candidate_dist[slot] = dist;
                        }
                    }
                }
            }
        }
 #elif CANDIDATE_SELECT_STRICT == 0
        for (int base = warp_id * 32; base < num_neighbors; base += num_warps * 32) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
            if (*((volatile int*)&s_candidate_count) >= CANDIDATE_BUFFER_SIZE) break;
#endif
            int n = base + lane_id;
            bool valid = (n < num_neighbors);
            DISTANCE_T dist = valid ? s_neighbor_dist[n] : FLT_MAX;
            bool keep =
                valid && (dist < candidate_cutoff || s_frontier_size < beam_size);
#if SKIP_ATOMIC_WHEN_POOL_FULL
            keep = keep && (*((volatile int*)&s_candidate_count) < CANDIDATE_BUFFER_SIZE);
#endif
            uint32_t ballot = __ballot_sync(FULL_MASK, keep);
            int warp_count = __popc(ballot);
            int base_offset = 0;
            if (lane_id == 0 && warp_count > 0) {
#if SKIP_ATOMIC_WHEN_POOL_FULL
                if (*((volatile int*)&s_candidate_count) < CANDIDATE_BUFFER_SIZE) {
                    base_offset = atomicAdd(&s_candidate_count, warp_count);
                } else {
                    base_offset = CANDIDATE_BUFFER_SIZE;
                }
#else
                base_offset = atomicAdd(&s_candidate_count, warp_count);
#endif
            }
            base_offset = __shfl_sync(FULL_MASK, base_offset, 0);
            if (keep) {
                int my_offset = __popc(ballot & ((1u << lane_id) - 1));
                int slot = base_offset + my_offset;
                if (slot < CANDIDATE_BUFFER_SIZE) {
                    s_candidate_idx[slot] = s_neighbor_idx[n];
                    s_candidate_dist[slot] = dist;
                }
            }
        }
 #else
 #error "Unsupported CANDIDATE_SELECT_STRICT"
 #endif
        }
        __syncthreads();
        if (tid == 0) {
            if (s_candidate_count == 0) {
                s_no_progress_iters++;
                if (s_no_progress_iters >= (int)NO_PROGRESS_BREAK_ITERS) {
                    s_break_now = 1;
                }
            } else {
                s_no_progress_iters = 0;
            }
        }
        __syncthreads();
#if RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_PARITY_MODE && \
    RABITQ_CPU_PARITY_EXACT_POOL
        if (tid == 0) {
            int stream_n = s_exact_stream_size;
            if (stream_n > 0) {
                int exact_cap = max(1, min(K, beam_size));
#if RABITQ_CPU_PARITY_FEEDBACK_CAP > 0
                exact_cap = min(exact_cap, (int)RABITQ_CPU_PARITY_FEEDBACK_CAP);
                exact_cap = max(exact_cap, 1);
#endif
                const bool force_flush =
                    (stream_n >= (int)RABITQ_CPU_PARITY_STREAM_CAP) || s_break_now;
                const int flush_interval = (int)RABITQ_CPU_PARITY_FLUSH_INTERVAL;
                const bool interval_hit =
                    (flush_interval > 0) && (((iter + 1) % flush_interval) == 0);
                if (force_flush || interval_hit) {
                    rabitq_parity_merge_exact_stream<RABITQ_CPU_PARITY_STREAM_CAP>(
                        s_exact_topk_idx,
                        s_exact_topk_dist,
                        &s_exact_topk_size,
                        s_exact_stream_idx,
                        s_exact_stream_dist,
                        &s_exact_stream_size,
                        exact_cap,
                        kIndexMsbMask);
                }
            }
        }
        __syncthreads();
#endif
        if (s_break_now) break;
        if (s_candidate_count == 0) continue;
#if 1
        if (warp_id == 0 && !s_candidate_sorted && s_candidate_count > 1) {
            candidate_by_bitonic_sort<2, 4>(s_candidate_idx, s_candidate_dist, CANDIDATE_BUFFER_SIZE);
        }
#endif
#if RABITQ_FEEDBACK_MERGE_BATCH > 0
        if (tid == 0 && s_candidate_sorted && s_frontier_size >= beam_size && s_candidate_count > 0) {
            const int nf_fb = min(s_frontier_size, beam_size);
            const int nc_fb = min(s_candidate_count, CANDIDATE_BUFFER_SIZE);
            DISTANCE_T fb_cutoff = s_frontier_dist[nf_fb - 1];
            int keep_n = 0;
            int since_feedback = 0;
            const int feedback_batch = max(1, (int)RABITQ_FEEDBACK_MERGE_BATCH);
            for (int i = 0; i < nc_fb; ++i) {
                const DISTANCE_T cand_d = s_candidate_dist[i];
                if (cand_d > fb_cutoff) break;
                keep_n++;
                since_feedback++;
                if (since_feedback >= feedback_batch) {
                    // Lightweight exact-cutoff feedback:
                    // after taking keep_n candidates from sorted candidate list,
                    // the merged tail is bounded by max(frontier[beam-1-keep_n], candidate[keep_n-1]).
                    const int repl = min(keep_n, beam_size - 1);
                    const DISTANCE_T front_ref = s_frontier_dist[beam_size - 1 - repl];
                    const DISTANCE_T cand_ref = s_candidate_dist[keep_n - 1];
                    fb_cutoff = max(front_ref, cand_ref);
                    since_feedback = 0;
                }
            }
            s_candidate_count = keep_n;
        }
#endif
        __syncthreads();
        {
            int nf = min(s_frontier_size, beam_size);
            int nc = min(s_candidate_count, CANDIDATE_BUFFER_SIZE);
            int total = nf + nc;
            int output_size = min(total, beam_size);
            for (int out_pos = tid; out_pos < output_size; out_pos += blockDim.x) {
                const int k = out_pos;
                int lo = max(0, k - nf);
                int hi = min(k, nc);

                while (lo < hi) {
                    const int cand_take = (lo + hi + 1) >> 1;
                    const int front_take = k - cand_take;
                    const DISTANCE_T cand_left = (cand_take > 0) ? s_candidate_dist[cand_take - 1] : -FLT_MAX;
                    const DISTANCE_T front_right = (front_take < nf) ? s_frontier_dist[front_take] : FLT_MAX;
                    if (cand_left <= front_right) {
                        lo = cand_take;
                    } else {
                        hi = cand_take - 1;
                    }
                }

                const int cand_take = lo;
                const int front_take = k - cand_take;
                const bool cand_valid = (cand_take < nc);
                const bool front_valid = (front_take < nf);
                const DISTANCE_T cand_dist = cand_valid ? s_candidate_dist[cand_take] : FLT_MAX;
                const DISTANCE_T front_dist = front_valid ? s_frontier_dist[front_take] : FLT_MAX;

                if (cand_valid && cand_dist <= front_dist) {
                    s_neighbor_dist[out_pos] = cand_dist;
                    s_neighbor_idx[out_pos] = s_candidate_idx[cand_take];
                } else if (front_valid) {
                    s_neighbor_dist[out_pos] = front_dist;
                    s_neighbor_idx[out_pos] = s_frontier_idx[front_take];
                } else {
                    s_neighbor_dist[out_pos] = FLT_MAX;
                    s_neighbor_idx[out_pos] = MAX_INDEX;
                }
            }
            __syncthreads();
#if ENABLE_FRONTIER_DEDUP
            if (tid == 0) {
                s_frontier_adj_dup_cnt = 0;
            }
            int local_adj_dup = 0;
            for (int i = tid + 1; i < output_size; i += blockDim.x) {
                const INDEX_T prev_idx = s_neighbor_idx[i - 1];
                const INDEX_T cur_idx = s_neighbor_idx[i];
                if (prev_idx == MAX_INDEX || cur_idx == MAX_INDEX) continue;
                const INDEX_T prev_key = prev_idx & ~kIndexMsbMask;
                const INDEX_T cur_key = cur_idx & ~kIndexMsbMask;
                if (prev_key == cur_key) {
                    local_adj_dup++;
                }
            }
            if (local_adj_dup > 0) {
                atomicAdd(&s_frontier_adj_dup_cnt, local_adj_dup);
            }
            __syncthreads();

            if (tid == 0) {
                const bool interval_trigger =
                    (FRONTIER_DEDUP_INTERVAL <= 1) || ((iter % FRONTIER_DEDUP_INTERVAL) == 0);
#if FRONTIER_DEDUP_ADAPTIVE
                const int min_size = (int)max(1, (int)FRONTIER_DEDUP_ADAPTIVE_MIN_SIZE);
                const int min_adj_dup = (int)max(1, (int)FRONTIER_DEDUP_ADAPTIVE_MIN_ADJ_DUP);
                s_frontier_dedup_apply =
                    interval_trigger && (output_size >= min_size) && (s_frontier_adj_dup_cnt >= min_adj_dup);
#else
                s_frontier_dedup_apply = interval_trigger;
#endif
            }
            __syncthreads();

            if (s_frontier_dedup_apply) {
#if FRONTIER_DEDUP_METHOD == 1
#if FRONTIER_DEDUP_PARALLEL
                if (tid < output_size) {
                    const INDEX_T idx = s_neighbor_idx[tid];
                    int keep = 0;
                    if (idx != MAX_INDEX) {
                        const INDEX_T key = idx & ~kIndexMsbMask;
                        if (tid == 0) {
                            keep = 1;
                        } else {
                            const INDEX_T prev_idx = s_neighbor_idx[tid - 1];
                            if (prev_idx == MAX_INDEX) {
                                keep = 1;
                            } else {
                                const INDEX_T prev_key = prev_idx & ~kIndexMsbMask;
                                keep = (key != prev_key);
                            }
                        }
                    }
                    s_frontier_dedup_keep[tid] = keep;
                }
                __syncthreads();
                if (tid == 0) {
                    int unique_size = 0;
                    for (int i = 0; i < output_size; i++) {
                        if (s_frontier_dedup_keep[i] && unique_size < beam_size) {
                            s_frontier_dedup_pos[i] = unique_size;
                            unique_size++;
                        } else {
                            s_frontier_dedup_pos[i] = -1;
                        }
                    }
                    s_frontier_size = unique_size;
                }
                __syncthreads();
                for (int i = tid; i < output_size; i += blockDim.x) {
                    const int pos = s_frontier_dedup_pos[i];
                    if (pos >= 0 && pos < beam_size) {
                        s_frontier_idx[pos] = s_neighbor_idx[i];
                        s_frontier_dist[pos] = s_neighbor_dist[i];
                    }
                }
#else
                if (tid == 0) {
                    int unique_size = 0;
                    INDEX_T prev_key = MAX_INDEX;
                    bool has_prev = false;
                    for (int i = 0; i < output_size && unique_size < beam_size; i++) {
                        const INDEX_T idx = s_neighbor_idx[i];
                        if (idx == MAX_INDEX) continue;
                        const INDEX_T key = idx & ~kIndexMsbMask;
                        if (!has_prev || key != prev_key) {
                            s_frontier_idx[unique_size] = idx;
                            s_frontier_dist[unique_size] = s_neighbor_dist[i];
                            prev_key = key;
                            has_prev = true;
                            unique_size++;
                        }
                    }
                    s_frontier_size = unique_size;
                }
                __syncthreads();
#endif
#elif FRONTIER_DEDUP_METHOD == 2
                hashtable_init(s_frontier_dedup_hash, FRONTIER_DEDUP_BITLEN, 0);
                __syncthreads();
#if FRONTIER_DEDUP_PARALLEL
                if (tid < output_size) {
                    const INDEX_T idx = s_neighbor_idx[tid];
                    int keep = 0;
                    if (idx != MAX_INDEX) {
                        const INDEX_T key = idx & ~kIndexMsbMask;
                        keep = hashtable_insert(s_frontier_dedup_hash, FRONTIER_DEDUP_BITLEN, key) ? 1 : 0;
                    }
                    s_frontier_dedup_keep[tid] = keep;
                }
                __syncthreads();
                if (tid == 0) {
                    int unique_size = 0;
                    for (int i = 0; i < output_size; i++) {
                        if (s_frontier_dedup_keep[i] && unique_size < beam_size) {
                            s_frontier_dedup_pos[i] = unique_size;
                            unique_size++;
                        } else {
                            s_frontier_dedup_pos[i] = -1;
                        }
                    }
                    s_frontier_size = unique_size;
                }
                __syncthreads();
#else
                if (tid == 0) {
                    int unique_size = 0;
                    for (int i = 0; i < output_size; i++) {
                        const INDEX_T idx = s_neighbor_idx[i];
                        if (idx == MAX_INDEX) {
                            s_frontier_dedup_pos[i] = -1;
                            continue;
                        }
                        const INDEX_T key = idx & ~kIndexMsbMask;
                        if (hashtable_insert_noatomic(s_frontier_dedup_hash, FRONTIER_DEDUP_BITLEN, key)
                            && unique_size < beam_size) {
                            s_frontier_dedup_pos[i] = unique_size;
                            unique_size++;
                        } else {
                            s_frontier_dedup_pos[i] = -1;
                        }
                    }
                    s_frontier_size = unique_size;
                }
                __syncthreads();
#endif
                for (int i = tid; i < output_size; i += blockDim.x) {
                    const int pos = s_frontier_dedup_pos[i];
                    if (pos >= 0 && pos < beam_size) {
                        s_frontier_idx[pos] = s_neighbor_idx[i];
                        s_frontier_dist[pos] = s_neighbor_dist[i];
                    }
                }
#else
#error "Unsupported FRONTIER_DEDUP_METHOD"
#endif
            } else {
                for (int i = tid; i < output_size; i += blockDim.x) {
                    s_frontier_idx[i] = s_neighbor_idx[i];
                    s_frontier_dist[i] = s_neighbor_dist[i];
                }
                if (tid == 0) {
                    s_frontier_size = output_size;
                }
            }
#else
            for (int i = tid; i < output_size; i += blockDim.x) {
                s_frontier_idx[i] = s_neighbor_idx[i];
                s_frontier_dist[i] = s_neighbor_dist[i];
            }
            if (tid == 0) {
                s_frontier_size = output_size;
            }
#endif
        }
        __syncthreads();
#if SQG_KERNEL_ACTIVE && SQG_EARLY_STOP_ENABLE
        if (tid == 0) {
            const DISTANCE_T cur_best =
                (s_frontier_size > 0) ? s_frontier_dist[0] : FLT_MAX;
            bool improved = false;
            if (cur_best + (DISTANCE_T)SQG_STALL_IMPROVE_EPS_ABS < s_sqg_best_dist) {
                improved = true;
            } else if (s_sqg_best_dist < FLT_MAX) {
                const DISTANCE_T delta = s_sqg_best_dist - cur_best;
                improved = delta > fabsf((float)s_sqg_best_dist) * (DISTANCE_T)SQG_STALL_IMPROVE_EPS_REL;
            }
            if (improved) {
                s_sqg_best_dist = cur_best;
                s_sqg_stall_iters = 0;
            } else {
                s_sqg_stall_iters++;
            }
            bool can_stop =
                sqg_filter_gate_runtime && (iter >= (int)SQG_EARLY_STOP_WARMUP_ITERS);
#if SQG_EARLY_STOP_REQUIRE_FULL
            can_stop = can_stop && (s_frontier_size >= beam_size);
#endif
            s_sqg_break_now =
                can_stop && (s_sqg_stall_iters >= (int)SQG_EARLY_STOP_STALL_ITERS);
        }
        __syncthreads();
        if (s_sqg_break_now) {
            break;
        }
#endif
    }
#if RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_PARITY_MODE && \
    RABITQ_CPU_PARITY_EXACT_POOL
    if (tid == 0 && s_exact_stream_size > 0) {
        int exact_cap = max(1, min(K, beam_size));
#if RABITQ_CPU_PARITY_FEEDBACK_CAP > 0
        exact_cap = min(exact_cap, (int)RABITQ_CPU_PARITY_FEEDBACK_CAP);
        exact_cap = max(exact_cap, 1);
#endif
        rabitq_parity_merge_exact_stream<RABITQ_CPU_PARITY_STREAM_CAP>(
            s_exact_topk_idx,
            s_exact_topk_dist,
            &s_exact_topk_size,
            s_exact_stream_idx,
            s_exact_stream_dist,
            &s_exact_stream_size,
            exact_cap,
            kIndexMsbMask);
    }
    __syncthreads();
#if RABITQ_CPU_PARITY_FINAL_FRONTIER_EXACT
    {
        const int nf = min(s_frontier_size, beam_size);
        for (int i = warp_id; i < nf; i += num_warps) {
            const INDEX_T fidx = s_frontier_idx[i] & ~kIndexMsbMask;
            if (fidx >= npoints) continue;
            const T* fvec = d_data + (size_t)fidx * (size_t)dim;
            DISTANCE_T fdist =
                (dim == 128)
                    ? warp_l2_distance_128((const float*)query, (const float*)fvec)
                    : warp_l2_distance(dim, query, fvec);
            if (lane_id == 0) {
                atomicAdd(&s_dc_count, 1);
                const int slot = atomicAdd(&s_exact_stream_size, 1);
                if (slot < (int)RABITQ_CPU_PARITY_STREAM_CAP) {
                    s_exact_stream_idx[slot] = fidx;
                    s_exact_stream_dist[slot] = fdist;
                }
            }
        }
    }
    __syncthreads();
    if (tid == 0 && s_exact_stream_size > 0) {
        int exact_cap = max(1, min(K, beam_size));
#if RABITQ_CPU_PARITY_FEEDBACK_CAP > 0
        exact_cap = min(exact_cap, (int)RABITQ_CPU_PARITY_FEEDBACK_CAP);
        exact_cap = max(exact_cap, 1);
#endif
        rabitq_parity_merge_exact_stream<RABITQ_CPU_PARITY_STREAM_CAP>(
            s_exact_topk_idx,
            s_exact_topk_dist,
            &s_exact_topk_size,
            s_exact_stream_idx,
            s_exact_stream_dist,
            &s_exact_stream_size,
            exact_cap,
            kIndexMsbMask);
    }
    __syncthreads();
#endif
#endif
#if CANDIDATE_SELECT_STRICT == 2 && CANDIDATE_SELECT_STRICT_FAST && STRICT_FAST_DEBUG_PRINT
    if (qid == 0 && tid == 0) {
        const int total_iters = s_strict_total_active_iters;
        const int fast_iters = s_strict_fast_active_iters;
        const int low_iters = s_strict_fast_low_iters;
        const int high_iters = s_strict_fast_high_iters;
        const int valid_sum = s_strict_fast_valid_sum;
        const float avg_valid = (fast_iters > 0) ? ((float)valid_sum / (float)fast_iters) : 0.0f;
        const float ratio = (total_iters > 0) ? ((float)fast_iters / (float)total_iters) : 0.0f;
        printf("strict_fast_debug: total_active_iters=%d, fast_active_iters=%d, low_iters=%d, high_iters=%d, avg_valid=%f, ratio=%f\n",
               total_iters, fast_iters, low_iters, high_iters, avg_valid, ratio);
    }
#endif
#if RABITQ_KERNEL_ACTIVE && RABITQ_CPU_ALIGN_FLOW && RABITQ_CPU_PARITY_MODE && \
    RABITQ_CPU_PARITY_EXACT_POOL
#if RABITQ_CPU_PARITY_UPDATE_RESULTS
    if (tid == 0) {
        int exact_cap = max(1, min(K, beam_size));
#if RABITQ_CPU_PARITY_FEEDBACK_CAP > 0
        exact_cap = min(exact_cap, (int)RABITQ_CPU_PARITY_FEEDBACK_CAP);
        exact_cap = max(exact_cap, 1);
#endif
        if (s_exact_topk_size < exact_cap) {
            const int seed_n = min(s_exact_topk_size, exact_cap);
            for (int si = 0; si < seed_n && s_exact_topk_size < exact_cap; ++si) {
                const INDEX_T sid = s_exact_topk_idx[si] & ~kIndexMsbMask;
                if (sid >= npoints) continue;
                const vid_t* nbs = gg.N((vid_t)sid);
                for (int j = 0; j < max_degree && s_exact_topk_size < exact_cap; ++j) {
                    const INDEX_T nb = (INDEX_T)nbs[j];
                    if (nb >= npoints) continue;
                    if (!hashtable_insert_noatomic(s_hash, BITLEN, nb)) continue;
                    const T* nb_vec = d_data + (size_t)nb * (size_t)dim;
                    const DISTANCE_T nb_dist = l2_distance_scalar_device(dim, query, nb_vec);
                    s_dc_count += 1;

                    int found = -1;
                    for (int k = 0; k < s_exact_topk_size; ++k) {
                        if ((s_exact_topk_idx[k] & ~kIndexMsbMask) == nb) {
                            found = k;
                            break;
                        }
                    }
                    if (found >= 0) {
                        if (nb_dist < s_exact_topk_dist[found]) {
                            s_exact_topk_dist[found] = nb_dist;
                            INDEX_T key_i = s_exact_topk_idx[found];
                            DISTANCE_T key_d = s_exact_topk_dist[found];
                            int p = found - 1;
                            while (p >= 0 && s_exact_topk_dist[p] > key_d) {
                                s_exact_topk_dist[p + 1] = s_exact_topk_dist[p];
                                s_exact_topk_idx[p + 1] = s_exact_topk_idx[p];
                                --p;
                            }
                            s_exact_topk_dist[p + 1] = key_d;
                            s_exact_topk_idx[p + 1] = key_i;
                        }
                        continue;
                    }

                    if (s_exact_topk_size < exact_cap) {
                        int pos = s_exact_topk_size++;
                        s_exact_topk_idx[pos] = nb;
                        s_exact_topk_dist[pos] = nb_dist;
                        while (pos > 0 && s_exact_topk_dist[pos] < s_exact_topk_dist[pos - 1]) {
                            const DISTANCE_T td = s_exact_topk_dist[pos - 1];
                            s_exact_topk_dist[pos - 1] = s_exact_topk_dist[pos];
                            s_exact_topk_dist[pos] = td;
                            const INDEX_T ti = s_exact_topk_idx[pos - 1];
                            s_exact_topk_idx[pos - 1] = s_exact_topk_idx[pos];
                            s_exact_topk_idx[pos] = ti;
                            --pos;
                        }
                    } else if (nb_dist < s_exact_topk_dist[exact_cap - 1]) {
                        int pos = exact_cap - 1;
                        s_exact_topk_idx[pos] = nb;
                        s_exact_topk_dist[pos] = nb_dist;
                        while (pos > 0 && s_exact_topk_dist[pos] < s_exact_topk_dist[pos - 1]) {
                            const DISTANCE_T td = s_exact_topk_dist[pos - 1];
                            s_exact_topk_dist[pos - 1] = s_exact_topk_dist[pos];
                            s_exact_topk_dist[pos] = td;
                            const INDEX_T ti = s_exact_topk_idx[pos - 1];
                            s_exact_topk_idx[pos - 1] = s_exact_topk_idx[pos];
                            s_exact_topk_idx[pos] = ti;
                            --pos;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
#endif
    if (tid == 0) {
        int out = 0;
        const int exact_n = min(s_exact_topk_size, min(K, beam_size));
        for (int i = 0; i < exact_n && out < K; ++i) {
            d_results[qid * K + out] = s_exact_topk_idx[i] & ~kIndexMsbMask;
            out++;
        }
        const int nf = min(s_frontier_size, beam_size);
        for (int i = 0; i < nf && out < K; ++i) {
            const INDEX_T key = s_frontier_idx[i] & ~kIndexMsbMask;
            bool seen = false;
            for (int j = 0; j < out; ++j) {
                if (d_results[qid * K + j] == key) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                d_results[qid * K + out] = key;
                out++;
            }
        }
        while (out < K) {
            d_results[qid * K + out] = 0;
            out++;
        }
    }
#elif OUTPUT_TOPK_UNIQUE
    if (tid == 0) {
        const int nf = min(s_frontier_size, beam_size);
        int out = 0;
        for (int i = 0; i < nf && out < K; i++) {
            const INDEX_T key = s_frontier_idx[i] & ~kIndexMsbMask;
            bool seen = false;
            for (int j = 0; j < out; j++) {
                if (d_results[qid * K + j] == key) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                d_results[qid * K + out] = key;
                out++;
            }
        }
        while (out < K) {
            d_results[qid * K + out] = 0;
            out++;
        }
    }
#else
    for (int i = tid; i < K; i += blockDim.x) {
        if (i < s_frontier_size) {
            d_results[qid * K + i] = s_frontier_idx[i] & ~kIndexMsbMask;
        } else {
            d_results[qid * K + i] = 0;
        }
    }
#endif
    if (tid == 0) {
        atomicAdd(d_total_count_dc, (gpu_long_t)s_dc_count);
        if (d_total_hops != nullptr) {
            atomicAdd(d_total_hops, (gpu_long_t)s_iter_count);
        }
    }
#if RABITQ_KERNEL_ACTIVE && RABITQ_NEIGHBOR_APPROX_BUFFER && \
    RABITQ_REUSE_CAND_TMP_AS_APPROX && (CANDIDATE_SELECT_STRICT == 2)
#undef s_neighbor_approx
#endif
}
