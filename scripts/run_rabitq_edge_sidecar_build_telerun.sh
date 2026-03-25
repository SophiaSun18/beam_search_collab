#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

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

if ! command -v telerun >/dev/null 2>&1; then
  echo "ERROR: telerun command not found" >&2
  exit 127
fi

DATASET="${DATASET:-sift1m}"
STAMP="$(date +%Y%m%d_%H%M%S)"
TS="${TS:-rabitq_edge_sidecar_build_${DATASET}_${STAMP}}"
RESULTS_DIR="$ROOT/results"
TMP_RUN_DIR="$RESULTS_DIR/.runner_${TS}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
LOCAL_BIN_DIR="${LOCAL_BIN_DIR:-$BUILD_DIR/telerun_bins}"
mkdir -p "$RESULTS_DIR" "$TMP_RUN_DIR" "$BUILD_DIR" "$LOCAL_BIN_DIR"

PERSIST_ROOT="${PERSIST_ROOT:-/home/sunzeai/proj/.data/.codex_logs}"
REMOTE_BIN_ROOT="${REMOTE_BIN_ROOT:-/home/sunzeai/proj/.data/.codex_bins}"
REMOTE_BIN_CACHE="${REMOTE_BIN_CACHE:-1}"
REMOTE_BIN_REFRESH="${REMOTE_BIN_REFRESH:-0}"
REMOTE_INSTALL_TIMEOUT="${REMOTE_INSTALL_TIMEOUT:-180}"
TELERUN_SUBMIT_TIMEOUT="${TELERUN_SUBMIT_TIMEOUT:-180}"
TELERUN_PULL_TIMEOUT="${TELERUN_PULL_TIMEOUT:-60}"
POLL_ROUNDS="${POLL_ROUNDS:-360}"
POLL_SLEEP_SEC="${POLL_SLEEP_SEC:-10}"
POLL_TAIL_LINES="${POLL_TAIL_LINES:-400}"

case "$DATASET" in
  sift1m)
    BASE="${BASE:-/home/sunzeai/proj/.data/sift/sift_base.fvecs}"
    GRAPH="${GRAPH:-/home/sunzeai/proj/.data/sift/graph-d32}"
    ;;
  sift10m)
    BASE="${BASE:-/home/sunzeai/proj/.data/sift10m/sift10m_base.fvecs}"
    GRAPH="${GRAPH:-/home/sunzeai/proj/.data/sift10m/graph-d32}"
    ;;
  *)
    echo "ERROR: unsupported DATASET=$DATASET (use sift1m or sift10m)" >&2
    exit 2
    ;;
esac

OUT_PREFIX="${OUT_PREFIX:-${GRAPH}.routea_rebuild_${STAMP}}"
RABITQ_SIGNBIT_OUT="${RABITQ_SIGNBIT_OUT:-${OUT_PREFIX}.signbit.bin}"
RABITQ_TRIPLE_OUT="${RABITQ_TRIPLE_OUT:-${OUT_PREFIX}.rabitq.triple.f32.bin}"
RABITQ_FACTOR_DQ_OUT="${RABITQ_FACTOR_DQ_OUT:-${OUT_PREFIX}.rabitq.factor_dq.f32.bin}"
RABITQ_FACTOR_VQ_OUT="${RABITQ_FACTOR_VQ_OUT:-${OUT_PREFIX}.rabitq.factor_vq.f32.bin}"

write_helpers() {
  cat > "$TMP_RUN_DIR/telerun_submit_single.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
LOCK_FILE="${TELERUN_LOCK_FILE:-/tmp/codex_telerun_single.lock}"
mkdir -p "$(dirname "${LOCK_FILE}")"
exec 9>"${LOCK_FILE}"
flock 9
args=()
if [ "${TELERUN_OVERRIDE_PENDING:-1}" = "1" ]; then
  args+=(--override-pending)
fi
if [ "${TELERUN_USE_UTILS:-0}" = "1" ]; then
  args+=(--utils)
fi
if [ "${TELERUN_BYPASS_LAST_JOB:-0}" = "1" ]; then
  args+=(--bypass-last-job)
fi
telerun "${args[@]}" "$@"
EOS

  cat > "$TMP_RUN_DIR/remote_check_file.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
FILE_PATH="${1:-}"
if [ -z "${FILE_PATH}" ]; then
  exit 2
fi
if [ -f "${FILE_PATH}" ]; then
  ls -lh "${FILE_PATH}" || true
  echo "__REMOTE_FILE_OK__=1"
else
  echo "MISSING: ${FILE_PATH}"
  echo "__REMOTE_FILE_OK__=0"
fi
EOS

  cat > "$TMP_RUN_DIR/remote_install_file.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
SRC_PATH="${1:-}"
DST_PATH="${2:-}"
if [ -z "${SRC_PATH}" ] || [ -z "${DST_PATH}" ]; then
  exit 2
fi
if [ ! -f "${SRC_PATH}" ]; then
  src_base="$(basename "${SRC_PATH}")"
  if [ -f "${src_base}" ]; then
    SRC_PATH="${src_base}"
  fi
fi
if [ ! -f "${SRC_PATH}" ]; then
  exit 3
fi
mkdir -p "$(dirname "${DST_PATH}")"
cp -f "${SRC_PATH}" "${DST_PATH}"
chmod +x "${DST_PATH}" 2>/dev/null || true
ls -lh "${DST_PATH}" || true
echo "__REMOTE_INSTALL_OK__=1"
EOS

  cat > "$TMP_RUN_DIR/run_rabitq_edge_sidecar_build_persist_telerun.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
PERSIST_ROOT="${PERSIST_ROOT:-/home/sunzeai/proj/.data/.codex_logs}"
PERSIST_TAG="${PERSIST_TAG:-rabitq_edge_sidecar_build_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PERSIST_ROOT}"
LOG_FILE="${PERSIST_ROOT}/${PERSIST_TAG}.log"
run_main() {
  BIN="${1:-./beam_search_gpu}"
  if [ ! -f "${BIN}" ]; then
    bin_base="$(basename "${BIN}")"
    if [ -f "${bin_base}" ]; then
      BIN="${bin_base}"
    fi
  fi
  if [ ! -f "${BASE}" ] || [ ! -e "${GRAPH}" ]; then
    echo "ERROR: dataset path check failed."
    echo "BASE=${BASE}"
    echo "GRAPH=${GRAPH}"
    return 2
  fi
  export GRAPH_INDEX_FORMAT="${GRAPH_INDEX_FORMAT:-read_bin}"
  echo "=== Launch RaBitQ Edge Sidecar Builder ==="
  echo "BIN=${BIN}"
  echo "GRAPH_INDEX_FORMAT=${GRAPH_INDEX_FORMAT}"
  echo "BASE=${BASE}"
  echo "GRAPH=${GRAPH}"
  echo "RABITQ_SIGNBIT_OUT=${RABITQ_SIGNBIT_OUT}"
  echo "RABITQ_TRIPLE_OUT=${RABITQ_TRIPLE_OUT}"
  echo "RABITQ_FACTOR_DQ_OUT=${RABITQ_FACTOR_DQ_OUT}"
  echo "RABITQ_FACTOR_VQ_OUT=${RABITQ_FACTOR_VQ_OUT}"
  "${BIN}" "${BASE}" "${GRAPH}" \
    --build-rabitq-edge-sidecars \
    --rabitq-signbit-out "${RABITQ_SIGNBIT_OUT}" \
    --rabitq-triple-out "${RABITQ_TRIPLE_OUT}" \
    --rabitq-factor-dq-out "${RABITQ_FACTOR_DQ_OUT}" \
    --rabitq-factor-vq-out "${RABITQ_FACTOR_VQ_OUT}"
}
{
  echo "PERSIST_TAG=${PERSIST_TAG}"
  echo "PERSIST_LOG=${LOG_FILE}"
  run_main "$@"
} 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
echo "__EXIT_CODE__=${status}" | tee -a "${LOG_FILE}"
exit "${status}"
EOS

  chmod +x "$TMP_RUN_DIR"/*.sh
}

local_bin_path() {
  printf '%s/beam_search_gpu_edge_sidecar_build_%s\n' "$LOCAL_BIN_DIR" "$TS"
}

remote_bin_path() {
  printf '%s/%s\n' "$REMOTE_BIN_ROOT" "$(basename "$(local_bin_path)")"
}

compile_bin() {
  local log="$RESULTS_DIR/${TS}.compile.log"
  make -B -C code sm89 BUILD_DIR="$BUILD_DIR" > "$log" 2>&1
  cp "$BUILD_DIR/beam_search_gpu" "$(local_bin_path)"
}

install_remote_bin() {
  local local_bin remote_bin check_out install_out
  local_bin="$(local_bin_path)"
  remote_bin="$(remote_bin_path)"
  if [ "$REMOTE_BIN_CACHE" = "1" ] && [ "$REMOTE_BIN_REFRESH" != "1" ]; then
    check_out="$({
      timeout "$REMOTE_INSTALL_TIMEOUT" env TELERUN_BYPASS_LAST_JOB=1 \
        "$TMP_RUN_DIR/telerun_submit_single.sh" env "$TMP_RUN_DIR/remote_check_file.sh" "$remote_bin"
      echo "__CRC__=$?"
    } 2>&1 || true)"
    if printf '%s\n' "$check_out" | grep -Eq '__REMOTE_FILE_OK__=1'; then
      printf '%s\n' "$remote_bin" > "$RESULTS_DIR/${TS}.binpath"
      return 0
    fi
  fi
  install_out="$({
    timeout "$REMOTE_INSTALL_TIMEOUT" env TELERUN_BYPASS_LAST_JOB=1 \
      "$TMP_RUN_DIR/telerun_submit_single.sh" env "$TMP_RUN_DIR/remote_install_file.sh" "$local_bin" "$remote_bin"
    echo "__IRC__=$?"
  } 2>&1 || true)"
  printf '%s\n' "$install_out" > "$RESULTS_DIR/${TS}.install.log"
  if ! printf '%s\n' "$install_out" | grep -Eq '__REMOTE_INSTALL_OK__=1'; then
    echo "ERROR: failed to install remote binary" >&2
    return 1
  fi
  printf '%s\n' "$remote_bin" > "$RESULTS_DIR/${TS}.binpath"
}

submit_build() {
  local remote_bin log
  remote_bin="$(cat "$RESULTS_DIR/${TS}.binpath")"
  log="$RESULTS_DIR/${TS}.submit.log"
  timeout "$TELERUN_SUBMIT_TIMEOUT" env TELERUN_BYPASS_LAST_JOB=1 \
    "$TMP_RUN_DIR/telerun_submit_single.sh" env \
    GRAPH_INDEX_FORMAT=read_bin \
    BASE="$BASE" GRAPH="$GRAPH" \
    RABITQ_SIGNBIT_OUT="$RABITQ_SIGNBIT_OUT" \
    RABITQ_TRIPLE_OUT="$RABITQ_TRIPLE_OUT" \
    RABITQ_FACTOR_DQ_OUT="$RABITQ_FACTOR_DQ_OUT" \
    RABITQ_FACTOR_VQ_OUT="$RABITQ_FACTOR_VQ_OUT" \
    PERSIST_ROOT="$PERSIST_ROOT" PERSIST_TAG="$TS" \
    "$TMP_RUN_DIR/run_rabitq_edge_sidecar_build_persist_telerun.sh" "$remote_bin" \
    > "$log" 2>&1
}

poll_build() {
  local remote_log log out
  remote_log="$PERSIST_ROOT/${TS}.log"
  log="$RESULTS_DIR/${TS}.telerun.log"
  : > "$log"
  for p in $(seq 1 "$POLL_ROUNDS"); do
    out="$({
      timeout "$TELERUN_PULL_TIMEOUT" env TELERUN_OVERRIDE_PENDING=0 TELERUN_BYPASS_LAST_JOB=1 \
        "$TMP_RUN_DIR/telerun_submit_single.sh" env tail -n "$POLL_TAIL_LINES" "$remote_log"
      echo "__PRC__=$?"
    } 2>&1 || true)"
    printf '\n[pull %s/%s]\n%s\n' "$p" "$POLL_ROUNDS" "$out" >> "$log"
    if printf '%s\n' "$out" | grep -Eq 'save_signbit: OK' \
      && printf '%s\n' "$out" | grep -Eq 'save_triple: OK' \
      && printf '%s\n' "$out" | grep -Eq 'save_factor_dq: OK' \
      && printf '%s\n' "$out" | grep -Eq 'save_factor_vq: OK' \
      && printf '%s\n' "$out" | grep -Eq '__EXIT_CODE__=0'; then
      echo "$out" > "$RESULTS_DIR/${TS}.build.log"
      return 0
    fi
    sleep "$POLL_SLEEP_SEC"
  done
  echo "ERROR: edge sidecar build log did not reach terminal success state" >&2
  return 1
}

write_helpers

echo "=== RaBitQ Edge Sidecar Build Telerun ==="
echo "DATASET=$DATASET"
echo "GRAPH=$GRAPH"
echo "OUT_PREFIX=$OUT_PREFIX"
echo "RABITQ_SIGNBIT_OUT=$RABITQ_SIGNBIT_OUT"
echo "RABITQ_TRIPLE_OUT=$RABITQ_TRIPLE_OUT"
echo "RABITQ_FACTOR_DQ_OUT=$RABITQ_FACTOR_DQ_OUT"
echo "RABITQ_FACTOR_VQ_OUT=$RABITQ_FACTOR_VQ_OUT"
compile_bin
install_remote_bin
submit_build
poll_build

echo "BUILD_RESULT_LOG=$RESULTS_DIR/${TS}.build.log"
echo "BUILD_POLL_LOG=$RESULTS_DIR/${TS}.telerun.log"
echo "BUILD_SUBMIT_LOG=$RESULTS_DIR/${TS}.submit.log"
