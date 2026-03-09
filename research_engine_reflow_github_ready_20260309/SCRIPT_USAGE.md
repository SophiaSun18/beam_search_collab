# Script Usage

This folder is driven by a single script:
- `run_fixed_reflow_ab_telerun.sh`

## Default run
Run the fixed SIFT-10M benchmark point:
```bash
bash run_fixed_reflow_ab_telerun.sh
```

## Run SIFT-1M
```bash
DATASET=sift1m bash run_fixed_reflow_ab_telerun.sh
```

## Change the query count
Use a shorter smoke test:
```bash
MAX_QUERIES=1000 bash run_fixed_reflow_ab_telerun.sh
```

## Set a custom run tag
```bash
TS=my_run DATASET=sift10m BEAM=300 MAX_QUERIES=10000 bash run_fixed_reflow_ab_telerun.sh
```

## Optional environment variables
- `DATASET`: `sift10m` or `sift1m`
- `BEAM`: beam size, default `300`
- `K`: top-k, default `100`
- `RUNS`: benchmark runs, default `1`
- `WARMUP`: warmup runs, default `1`
- `MAX_QUERIES`: number of queries, default `10000`
- `TS`: custom output tag
- `BASE`, `QUERY`, `GT`, `GRAPH`, `SIGNBIT_FILE`: override remote dataset paths if needed

## Output files
The script writes logs and summaries under `results/`.
