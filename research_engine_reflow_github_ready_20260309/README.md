# Reflow Stable GitHub Snapshot

This folder contains the last stable CUDA Reflow / RaBitQ snapshot that is still suitable for delivery and remote A/B validation.

## Contents
- `code/`: minimal source files required to build the search binary
- `run_fixed_reflow_ab_telerun.sh`: single fixed remote OFF-vs-RB runner
- `.gitignore`: ignores generated binaries and benchmark outputs

## Fixed configuration
The runner is intentionally frozen to the stable delivery path.

- `BEAM=300`
- `K=100`
- `RUNS=1`
- `WARMUP=1`
- `TOPN=20`
- `ANCHOR_EXACT=32`
- `SCORE_WORDS=4`
- `PARTIAL_L2_128_LANES=24`
- `POP_BATCH=6` on `sift1m`
- `POP_BATCH=7` on `sift10m`
- CUDA target: `sm_89`

## Datasets
Supported datasets:
- `sift10m`
- `sift1m`

Default remote dataset paths are embedded in the runner:
- `sift10m`: `/home/sunzeai/proj/.data/sift10m/...`
- `sift1m`: `/home/sunzeai/proj/.data/sift/...`

You can override them with `BASE`, `QUERY`, `GT`, `GRAPH`, and `SIGNBIT_FILE`.

## Usage
```bash
bash run_fixed_reflow_ab_telerun.sh
DATASET=sift1m bash run_fixed_reflow_ab_telerun.sh
MAX_QUERIES=1000 bash run_fixed_reflow_ab_telerun.sh
TS=my_run DATASET=sift10m BEAM=300 MAX_QUERIES=10000 bash run_fixed_reflow_ab_telerun.sh
```

## What the runner does
`run_fixed_reflow_ab_telerun.sh` is the only tracked runner, but at runtime it will generate a few temporary helper scripts under `results/.runner_<TS>/` to keep the submission path robust:
- single-flight `telerun` wrapper
- remote binary check / install helper
- remote persistent benchmark wrapper

These temporary helpers are generated automatically and are not part of the repo API.

## Outputs
Generated files go under `results/`:
- `*.compile.log`
- `*.remote_install.log`
- `*.telerun.log`
- `*.metrics.txt`
- `*.summary.txt`

## Remote validation
This single runner was smoke-tested and fixed-point validated on 2026-03-09.

Current one-run verification with this exact folder:
- `sift10m`, `beam=300`, `q10000`: OFF `84144.1 qps / 0.97681 recall`, RB `84433.2 qps / 0.97692 recall`, delta `+0.344% qps`, `-2400 avg_dc`
- `sift1m`, `beam=300`, `q10000`: OFF `115902 qps / 0.993577 recall`, RB `118806 qps / 0.993586 recall`, delta `+2.506% qps`, `-2165 avg_dc`

Reference validation from the already verified stable bundle built from the same stable branch:
- `sift10m`, `beam=300`, `q10000`: `+5.437% qps`
- `sift1m`, `beam=300`, `q10000`: `+5.950% qps`

## Notes
- This folder is intended for the stable delivery path, not for the newer relaxed-searchq experiments.
- `results/` and generated binaries are ignored by `.gitignore`.
