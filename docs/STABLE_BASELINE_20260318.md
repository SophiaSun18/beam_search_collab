# Stable Baseline Snapshot

Folder:

- `research_engine_reflow_stable_mainline_20260318`

Source of truth for this extraction:

- `research_engine_reflow_split_mainlines_20260316/config_mainline`

Validated remote baseline:

- result file: `results/revert_baseline_check_sift10m_b300_20260318_030339.summary.txt`
- dataset: `sift10m`
- beam: `300`

Metrics:

```text
OFF:    qps=84362.2 recall=0.97681 avg_dc=7715 avg_hops=46 kernel_ms=11.735
CONFIG: qps=88613.2 recall=0.97689 avg_dc=5312 avg_hops=47 kernel_ms=11.134
DELTA:  qps_pct=+5.039 recall=+0.000080 dc=-2403
```

Stable compile-time path:

- `REFLOW_TOPN=20`
- `REFLOW_POP_BATCH=7`
- `REFLOW_ANCHOR_EXACT=32`
- `partial-L2 enabled`
- `score words = 4`

Current override surface:

- stable runner now overrides only `BEAM_SIZE`, `BITLEN`, and `SEARCH_WIDTH`
- `REFLOW_TOPN`, `REFLOW_POP_BATCH`, `REFLOW_ANCHOR_EXACT`, and `PARTIAL_L2_128_LANES` are fixed in source

Notes:

- This extraction intentionally keeps only the stable CONFIG line.
- Failed `anchor<32>` Route-A experiments are not part of this directory.
- Use this folder as the clean base for future readability refactors or stable-line engineering work.
- Latest readability-cleanup validation: `results/stable_sift10m_b300_20260320_135427.summary.txt`
  - `CONFIG: qps=88589.8 recall=0.97684 avg_dc=5311 avg_hops=47 kernel_ms=11.228`
