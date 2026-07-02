#!/usr/bin/env bash
# telerun_distance_cost_worker.sh
# Runs INSIDE telerun's RTX 4000 Ada (sm_89) host. It:
#   1. extracts the standalone distance_cost_bench source tree from a tar.gz,
#   2. runs a fuser-based GPU occupancy precheck (AGENTS.md method),
#   3. builds Part-1 distance cost benchmarks: CPU (host CPU) + GPU (sm_89),
#   4. runs both and emits structured banners for the parser.
#
# Call shape:
#   telerun ... env <env vars> <this script> <source-tar.gz>
set -euo pipefail

SRC_TAR="${1:?usage: $0 <source-tar.gz>}"

DATA_ROOT="${DATA_ROOT:-/home/sunzeai/proj/.data}"
DATASETS="${DATASETS:-sift,gist,deep1M}"
DEGREE="${DEGREE:-32}"
REPS="${REPS:-2000000}"        # CPU ops per measurement
NODES="${NODES:-256}"          # CPU distinct nodes cycled
GPU_SCALE="${GPU_SCALE:-4}"    # GPU blocks per SM
SOURCE_DIR_NAME="${SOURCE_DIR_NAME:-distance_cost_bench}"
STRICT_PRECHECK="${STRICT_PRECHECK:-1}"
PERSIST_ROOT="${PERSIST_ROOT:-${DATA_ROOT}/.codex_logs}"
PERSIST_TAG="${PERSIST_TAG:-distance_cost_$(date +%Y%m%d_%H%M%S)}"
PERSIST_LOG="${PERSIST_ROOT}/${PERSIST_TAG}.log"

split_csv() { local c="$1"; c="${c//,/ }"; printf '%s\n' ${c}; }

# dataset -> raw dim (DIM) and padded dim (PADDED = next pow2 >= dim)
dataset_dims() {
  case "$1" in
    sift|sift1m) DIM=128;  PADDED=128 ;;
    deep1M)      DIM=256;  PADDED=256 ;;
    gist)        DIM=960;  PADDED=1024 ;;
    *) echo "ERROR: unknown dataset dims for $1" >&2; return 2 ;;
  esac
}

dataset_files() {
  BASE="${DATA_ROOT}/$1/$1_base.fvecs"
  QUERY="${DATA_ROOT}/$1/$1_query.fvecs"
  QG="${DATA_ROOT}/$1/symphonyqg${DEGREE}.noNan.fastscan.index"
}

choose_nvcc() {
  if command -v nvcc >/dev/null 2>&1; then command -v nvcc; return 0; fi
  for p in /usr/local/cuda/bin/nvcc /usr/local/cuda-13.1/bin/nvcc \
           /usr/local/cuda-12.8/bin/nvcc /usr/local/cuda-12.2/bin/nvcc \
           /usr/local/cuda-12.1/bin/nvcc; do
    [ -x "$p" ] && { printf '%s\n' "$p"; return 0; }
  done
  return 1
}

run_gpu_occupancy_precheck() {
  echo "--- gpu occupancy precheck (fuser /dev/nvidia*) ---"
  if ! command -v fuser >/dev/null 2>&1; then
    echo "__GPU_OCCUPANCY_METHOD__=fuser_missing"
    [ "${STRICT_PRECHECK}" = "1" ] && { echo "ERROR: fuser missing, STRICT_PRECHECK=1" >&2; return 127; }
    return 0
  fi
  local out; set +e
  out="$(fuser -v /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm 2>&1)"; set -e
  printf '%s\n' "${out}"
  local busy
  busy="$(printf '%s\n' "${out}" | awk '
    /COMMAND/ { next } NF == 0 { next }
    { cmd=$NF; if (cmd!="Xorg"&&cmd!="gnome-shell"&&cmd!="gdm"&&cmd!="nvidia-persiste"&&cmd!="nvidia-persistenced") print }')"
  if [ -n "${busy}" ]; then
    echo "__GPU_OCCUPANCY_BUSY__=1"; printf '%s\n' "${busy}"
    [ "${STRICT_PRECHECK}" = "1" ] && { echo "ERROR: non-baseline GPU clients" >&2; return 75; }
  else
    echo "__GPU_OCCUPANCY_BUSY__=0"
  fi
}

main() {
  mkdir -p "${PERSIST_ROOT}"
  echo "=== telerun Part-1 distance cost benchmark ==="
  echo "HOST=$(hostname)"
  echo "SRC_TAR=${SRC_TAR}"
  echo "DATASETS=${DATASETS}  DEGREE=${DEGREE}  REPS=${REPS}  NODES=${NODES}  GPU_SCALE=${GPU_SCALE}"
  echo "PERSIST_LOG=${PERSIST_LOG}"

  run_gpu_occupancy_precheck

  local WORK="${PWD}/distance_cost_work"
  rm -rf "${WORK}"; mkdir -p "${WORK}"
  tar -xzf "${SRC_TAR}" -C "${WORK}"
  local TREE
  TREE="$(find "${WORK}" -maxdepth 1 -type d -name "${SOURCE_DIR_NAME}" | head -1)"
  [ -n "${TREE}" ] || { echo "ERROR: source dir ${SOURCE_DIR_NAME} not found" >&2; find "${WORK}" -maxdepth 2 -type d | sort; return 2; }
  cd "${TREE}"; echo "PWD=$(pwd)"

  echo "=== compilers ==="
  g++ --version | head -1
  local NVCC; NVCC="$(choose_nvcc)"; echo "NVCC=${NVCC}"
  "${NVCC}" --version | tail -1

  # CPU benchmark is dimension-agnostic (reads dim from codebook); build once.
  echo "=== build CPU distance benchmark (host CPU) ==="
  make distance_cpu

  for ds in $(split_csv "${DATASETS}"); do
    dataset_dims "${ds}"
    dataset_files "${ds}"
    echo "============================================================"
    echo "=== DATASET ${ds} (DIM=${DIM} PADDED=${PADDED}) ==="
    local ok=1
    for f in "${BASE}" "${QUERY}" "${QG}"; do
      if [ ! -f "$f" ]; then echo "WARN: missing required file: $f (skip ${ds})" >&2; ok=0; break; fi
      ls -lh "$f"
    done
    [ "${ok}" = "1" ] || { echo "__MB_SKIP__=${ds}"; continue; }

    echo "=== CPU distance benchmark ${ds} (ns/op) ==="
    echo "__MB_CPU_BEGIN__ ${ds}"
    bin/distance_cpu -base "${BASE}" -query "${QUERY}" -qg "${QG}" \
      -degree "${DEGREE}" -reps "${REPS}" -nodes "${NODES}"
    echo "__MB_CPU_END__ ${ds}"

    # GPU benchmark dims are compile-time; rebuild per dataset.
    echo "=== build GPU distance benchmark ${ds} (sm_89, DIM=${DIM} PADDED=${PADDED}) ==="
    rm -f bin/distance_gpu
    make distance_gpu NVCC="${NVCC}" \
      EXTRA_NVFLAGS="-DMB_DIM=${DIM} -DMB_PADDED=${PADDED}"
    sha256sum bin/distance_gpu

    run_gpu_occupancy_precheck
    echo "=== GPU distance benchmark ${ds} (ns/op, kernel-only) ==="
    echo "__MB_GPU_BEGIN__ ${ds}"
    bin/distance_gpu "${GPU_SCALE}"
    echo "__MB_GPU_END__ ${ds}"
  done

  echo "__DISTANCE_COST_DONE__=1"
}

{
  set +e
  ( set -e; main "$@" )
  rc=$?
  set -e
  echo "__EXIT_CODE__=${rc}"
  exit "${rc}"
} 2>&1 | tee "${PERSIST_LOG}"
