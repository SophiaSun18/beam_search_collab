#!/usr/bin/env bash
# run_distance_cost_telerun.sh
# Local runner: tars this standalone distance_cost_bench tree, submits ONE telerun job to the
# RTX 4000 Ada (sm_89) executor that builds + runs the Part-1 distance
# cost benchmark (CPU on host CPU, GPU sm_89), and tees the raw log locally.
#
# Usage: DATASETS=sift REPS=2000000 ./telerun/run_distance_cost_telerun.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKER="${WORKER:-${ROOT}/telerun/telerun_distance_cost_worker.sh}"
TREE_DIR="${TREE_DIR:-${ROOT}}"
TREE_LABEL="${TREE_LABEL:-distance_cost_bench}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

DATA_ROOT="${DATA_ROOT:-/home/sunzeai/proj/.data}"
DATASETS="${DATASETS:-sift,gist,deep1M}"
DEGREE="${DEGREE:-32}"
REPS="${REPS:-2000000}"
NODES="${NODES:-256}"
GPU_SCALE="${GPU_SCALE:-4}"
STRICT_PRECHECK="${STRICT_PRECHECK:-1}"
TELERUN_TIMEOUT="${TELERUN_TIMEOUT:-3600}"
TELERUN_OVERRIDE_PENDING="${TELERUN_OVERRIDE_PENDING:-0}"

command -v telerun >/dev/null 2>&1 || { echo "ERROR: telerun not found" >&2; exit 127; }
[ -f "${WORKER}" ] || { echo "ERROR: worker not found: ${WORKER}" >&2; exit 2; }
[ -d "${TREE_DIR}" ] || { echo "ERROR: tree not found: ${TREE_DIR}" >&2; exit 2; }

mkdir -p "${ROOT}/experiments/artifacts/src" "${ROOT}/experiments/results/raw"
SRC_TAR="${ROOT}/experiments/artifacts/src/${TREE_LABEL}_${STAMP}.tar.gz"
RAW_LOG="${ROOT}/experiments/results/raw/distance_cost_${STAMP}.log"

tree_parent="$(dirname "${TREE_DIR}")"
tree_base="$(basename "${TREE_DIR}")"
tar --exclude='.git' --exclude='bin' --exclude='*.o' --exclude='*.tar.gz' \
    --exclude='*.log' --exclude='result' \
    -C "${tree_parent}" -czf "${SRC_TAR}" "${tree_base}"
sha256sum "${SRC_TAR}" > "${SRC_TAR}.sha256"

args=(--bypass-last-job)
[ "${TELERUN_OVERRIDE_PENDING}" = "1" ] && args+=(--override-pending)
PERSIST_TAG="distance_cost_${TREE_LABEL}_${STAMP}"

{
  echo "=== local distance-cost telerun submission ==="
  echo "SRC_TAR=${SRC_TAR}"
  echo "DATASETS=${DATASETS}  DEGREE=${DEGREE}  REPS=${REPS}  NODES=${NODES}  GPU_SCALE=${GPU_SCALE}"
  echo "PERSIST_TAG=${PERSIST_TAG}"
} | tee "${RAW_LOG}"

set +e
timeout "${TELERUN_TIMEOUT}" \
  telerun "${args[@]}" env \
    DATA_ROOT="${DATA_ROOT}" \
    DATASETS="${DATASETS}" \
    DEGREE="${DEGREE}" \
    REPS="${REPS}" \
    NODES="${NODES}" \
    GPU_SCALE="${GPU_SCALE}" \
    SOURCE_DIR_NAME="${tree_base}" \
    STRICT_PRECHECK="${STRICT_PRECHECK}" \
    PERSIST_TAG="${PERSIST_TAG}" \
    "${WORKER}" "${SRC_TAR}" 2>&1 | tee -a "${RAW_LOG}"
rc=${PIPESTATUS[0]}
set -e
echo "__TELERUN_RC__=${rc}" | tee -a "${RAW_LOG}"

if grep -q '__DISTANCE_COST_DONE__=1' "${RAW_LOG}"; then
  echo "__LOCAL_DISTANCE_COST_LOG__=${RAW_LOG}"
  exit 0
fi
exit "${rc:-1}"
