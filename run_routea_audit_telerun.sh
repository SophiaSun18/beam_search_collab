#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

ensure_nvcc() {
  if command -v nvcc >/dev/null 2>&1; then
    return 0
  fi
  for cuda_bin in /usr/local/cuda/bin /usr/local/cuda-12/bin /usr/local/cuda-12.6/bin; do
    if [[ -x "${cuda_bin}/nvcc" ]]; then
      export PATH="${cuda_bin}:${PATH}"
      return 0
    fi
  done
  echo "ERROR: nvcc not found" >&2
  exit 127
}

ensure_tools() {
  ensure_nvcc
  if ! command -v telerun >/dev/null 2>&1; then
    echo "ERROR: telerun command not found" >&2
    exit 127
  fi
}

ensure_tools

DATASET="${DATASET:-sift10m}"
QUERY_SAMPLES="${QUERY_SAMPLES:-8}"
EDGE_SAMPLES="${EDGE_SAMPLES:-128}"
SOURCE_SAMPLES="${SOURCE_SAMPLES:-128}"
VERBOSE_EDGES="${VERBOSE_EDGES:-8}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build_routea_audit_runner}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
TS="${TS:-routea_audit_${DATASET}_$(date +%Y%m%d_%H%M%S)}"
TMP_DIR="$RESULTS_DIR/.runner_${TS}"
mkdir -p "$RESULTS_DIR" "$TMP_DIR" "$BUILD_DIR"

PERSIST_ROOT="${PERSIST_ROOT:-/home/sunzeai/proj/.data/.codex_logs}"
REMOTE_BIN_ROOT="${REMOTE_BIN_ROOT:-/home/sunzeai/proj/.data/.codex_bins}"
TELERUN_TIMEOUT="${TELERUN_TIMEOUT:-7200}"
REMOTE_INSTALL_TIMEOUT="${REMOTE_INSTALL_TIMEOUT:-180}"

case "$DATASET" in
  sift1m)
    BASE="${BASE:-/home/sunzeai/proj/.data/sift/sift_base.fvecs}"
    QUERY="${QUERY:-/home/sunzeai/proj/.data/sift/sift_query.fvecs}"
    GRAPH="${GRAPH:-/home/sunzeai/proj/.data/sift/graph-d32}"
    ;;
  sift10m)
    BASE="${BASE:-/home/sunzeai/proj/.data/sift10m/sift10m_base.fvecs}"
    QUERY="${QUERY:-/home/sunzeai/proj/.data/sift10m/sift10m_query.fvecs}"
    GRAPH="${GRAPH:-/home/sunzeai/proj/.data/sift10m/graph-d32}"
    ;;
  *)
    echo "ERROR: unsupported DATASET=$DATASET" >&2
    exit 2
    ;;
esac

LOCAL_BIN="$BUILD_DIR/beam_search_gpu"
REMOTE_BIN="$REMOTE_BIN_ROOT/beam_search_gpu_${TS}"
COMPILE_LOG="$RESULTS_DIR/${TS}.compile.log"
INSTALL_LOG="$RESULTS_DIR/${TS}.remote_install.log"
TELERUN_LOG="$RESULTS_DIR/${TS}.telerun.log"
SUMMARY_TXT="$RESULTS_DIR/${TS}.summary.txt"

cat > "$TMP_DIR/remote_install_file.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
SRC_PATH="${1:-}"
DST_PATH="${2:-}"
if [[ -z "${SRC_PATH}" || -z "${DST_PATH}" ]]; then
  exit 2
fi
if [[ ! -f "${SRC_PATH}" ]]; then
  base="$(basename "${SRC_PATH}")"
  if [[ -f "${base}" ]]; then
    SRC_PATH="${base}"
  fi
fi
if [[ ! -f "${SRC_PATH}" ]]; then
  echo "missing src: ${SRC_PATH}" >&2
  exit 3
fi
mkdir -p "$(dirname "${DST_PATH}")"
cp -f "${SRC_PATH}" "${DST_PATH}"
chmod +x "${DST_PATH}" 2>/dev/null || true
echo "__REMOTE_INSTALL_OK__=1"
EOS

cat > "$TMP_DIR/run_routea_audit_remote.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
BIN="${1:?bin required}"
BASE="${2:?base required}"
QUERY="${3:?query required}"
GRAPH="${4:?graph required}"
QUERY_SAMPLES="${5:-8}"
EDGE_SAMPLES="${6:-128}"
SOURCE_SAMPLES="${7:-128}"
VERBOSE_EDGES="${8:-8}"
PERSIST_ROOT="${9:-/home/sunzeai/proj/.data/.codex_logs}"
PERSIST_TAG="${10:-routea_audit_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PERSIST_ROOT}"
LOG_FILE="${PERSIST_ROOT}/${PERSIST_TAG}.log"
export GRAPH_INDEX_FORMAT="${GRAPH_INDEX_FORMAT:-read_bin}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
{
  echo "PERSIST_TAG=${PERSIST_TAG}"
  echo "PERSIST_LOG=${LOG_FILE}"
  echo "BIN=${BIN}"
  echo "BASE=${BASE}"
  echo "QUERY=${QUERY}"
  echo "GRAPH=${GRAPH}"
  echo "QUERY_SAMPLES=${QUERY_SAMPLES}"
  echo "EDGE_SAMPLES=${EDGE_SAMPLES}"
  echo "SOURCE_SAMPLES=${SOURCE_SAMPLES}"
  echo "VERBOSE_EDGES=${VERBOSE_EDGES}"
  "${BIN}" "${BASE}" "${QUERY}" "${GRAPH}" \
    --routea-audit \
    --audit-queries "${QUERY_SAMPLES}" \
    --audit-edges "${EDGE_SAMPLES}" \
    --audit-sources "${SOURCE_SAMPLES}" \
    --audit-verbose "${VERBOSE_EDGES}"
} 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
echo "__EXIT_CODE__=${status}" | tee -a "${LOG_FILE}"
exit "${status}"
EOS

chmod +x "$TMP_DIR/remote_install_file.sh" "$TMP_DIR/run_routea_audit_remote.sh"

NVFLAGS="--use_fast_math -std=c++17 -O3 -gencode arch=compute_89,code=sm_89"
NVFLAGS+=" -DBEAM_SIZE=300 -DBITLEN=12 -DSEARCH_WIDTH=7"
make -B -C "$ROOT/code" sm89 BUILD_DIR="$BUILD_DIR" NVFLAGS="$NVFLAGS" >"$COMPILE_LOG" 2>&1

timeout "$REMOTE_INSTALL_TIMEOUT" telerun env "$TMP_DIR/remote_install_file.sh" \
  "$LOCAL_BIN" "$REMOTE_BIN" >"$INSTALL_LOG" 2>&1
if ! grep -q "__REMOTE_INSTALL_OK__=1" "$INSTALL_LOG"; then
  echo "ERROR: remote install failed" >&2
  cat "$INSTALL_LOG" >&2 || true
  exit 1
fi

timeout "$TELERUN_TIMEOUT" \
  telerun env "$TMP_DIR/run_routea_audit_remote.sh" \
  "$REMOTE_BIN" "$BASE" "$QUERY" "$GRAPH" \
  "$QUERY_SAMPLES" "$EDGE_SAMPLES" "$SOURCE_SAMPLES" "$VERBOSE_EDGES" \
  "$PERSIST_ROOT" "$TS" \
  >"$TELERUN_LOG" 2>&1

{
  echo "DATASET=$DATASET"
  echo "BASE=$BASE"
  echo "QUERY=$QUERY"
  echo "GRAPH=$GRAPH"
  echo "QUERY_SAMPLES=$QUERY_SAMPLES"
  echo "EDGE_SAMPLES=$EDGE_SAMPLES"
  echo "SOURCE_SAMPLES=$SOURCE_SAMPLES"
  echo "VERBOSE_EDGES=$VERBOSE_EDGES"
  echo "REMOTE_BIN=$REMOTE_BIN"
  echo "TELERUN_LOG=$TELERUN_LOG"
  echo "PERSIST_LOG_HINT=${PERSIST_ROOT}/${TS}.log"
} >"$SUMMARY_TXT"

cat "$SUMMARY_TXT"
