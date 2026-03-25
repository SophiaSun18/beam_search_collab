#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

ensure_nvcc() {
  if command -v nvcc >/dev/null 2>&1; then
    return 0
  fi
  for cuda_bin in \
    /usr/local/cuda/bin \
    /usr/local/cuda-12/bin \
    /usr/local/cuda-12.6/bin
  do
    if [[ -x "${cuda_bin}/nvcc" ]]; then
      export PATH="${cuda_bin}:${PATH}"
      return 0
    fi
  done
  echo "ERROR: nvcc not found; checked PATH and common CUDA install dirs" >&2
  exit 127
}

ensure_nvcc

DATASET="${DATASET:-sift10m}"
BEAM="${BEAM:-300}"
K="${K:-100}"
RUNS="${RUNS:-1}"
WARMUP="${WARMUP:-1}"
MAX_QUERIES="${MAX_QUERIES:-10000}"

case "$DATASET" in
  sift1m|sift10m)
    ;;
  *) echo "ERROR: unsupported DATASET=$DATASET" >&2; exit 2 ;;
esac

BITLEN=12
SEARCH_WIDTH=7

NVFLAGS="--use_fast_math -std=c++17 -O3 -gencode arch=compute_89,code=sm_89"
NVFLAGS+=" -DBEAM_SIZE=${BEAM}"
NVFLAGS+=" -DBITLEN=${BITLEN}"
NVFLAGS+=" -DSEARCH_WIDTH=${SEARCH_WIDTH}"

RESULTS_DIR="$ROOT/results" CODE_DIR="$ROOT/code" BUILD_DIR="$ROOT/build" MODE=routea NVFLAGS="$NVFLAGS" \
  DATASET="$DATASET" BEAM="$BEAM" K="$K" RUNS="$RUNS" WARMUP="$WARMUP" MAX_QUERIES="$MAX_QUERIES" \
  REFLOW_APPROX_MODE=routea \
  "$ROOT/scripts/run_single_mode_telerun.sh"
