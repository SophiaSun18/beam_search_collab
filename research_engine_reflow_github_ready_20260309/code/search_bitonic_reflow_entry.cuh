#pragma once

// Reflow minimal control surface (new folder only).
// Keep these as the only tunable compile-time knobs for RabitQ flow studies.
#ifndef RABITQ_REFLOW_FIXED
#define RABITQ_REFLOW_FIXED 1
#endif
#ifndef REFLOW_TOPN
#define REFLOW_TOPN 20
#endif
#ifndef REFLOW_POP_BATCH
#define REFLOW_POP_BATCH 7
#endif
#ifndef REFLOW_ANCHOR_EXACT
#define REFLOW_ANCHOR_EXACT 32
#endif

// Map minimal reflow knobs to legacy macro names before pulling kernel body.
#if RABITQ_REFLOW_FIXED
#ifndef RABITQ_TOPN
#define RABITQ_TOPN REFLOW_TOPN
#endif
#ifndef RABITQ_TOPN_EARLY
#define RABITQ_TOPN_EARLY REFLOW_TOPN
#endif
#ifndef RABITQ_GPU_DUALPOOL_POP_BATCH
#define RABITQ_GPU_DUALPOOL_POP_BATCH REFLOW_POP_BATCH
#endif
#ifndef RABITQ_GPU_DUALPOOL_ANCHOR_EXACT
#define RABITQ_GPU_DUALPOOL_ANCHOR_EXACT REFLOW_ANCHOR_EXACT
#endif
#endif

#include "search_bitonic.cuh"
