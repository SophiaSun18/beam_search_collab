#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v telerun >/dev/null 2>&1; then
  echo "ERROR: telerun command not found" >&2
  exit 127
fi

: "${MODE:?MODE is required}"
: "${CODE_DIR:?CODE_DIR is required}"
: "${RESULTS_DIR:?RESULTS_DIR is required}"
: "${NVFLAGS:?NVFLAGS is required}"

BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
LOCAL_BIN_DIR="${LOCAL_BIN_DIR:-$BUILD_DIR/telerun_bins}"

DATASET="${DATASET:-sift10m}"
BEAM="${BEAM:-300}"
K="${K:-100}"
RUNS="${RUNS:-1}"
WARMUP="${WARMUP:-1}"
MAX_QUERIES="${MAX_QUERIES:-10000}"
TS="${TS:-${MODE}_${DATASET}_b${BEAM}_$(date +%Y%m%d_%H%M%S)}"
TMP_RUN_DIR="$RESULTS_DIR/.runner_${TS}"
mkdir -p "$RESULTS_DIR" "$TMP_RUN_DIR" "$BUILD_DIR" "$LOCAL_BIN_DIR"

PERSIST_ROOT="${PERSIST_ROOT:-/home/sunzeai/proj/.data/.codex_logs}"
REMOTE_BIN_ROOT="${REMOTE_BIN_ROOT:-/home/sunzeai/proj/.data/.codex_bins}"
REMOTE_BIN_CACHE="${REMOTE_BIN_CACHE:-1}"
REMOTE_BIN_REFRESH="${REMOTE_BIN_REFRESH:-0}"
RETRY="${RETRY:-3}"
SLEEP_SEC="${SLEEP_SEC:-5}"
POLL_ROUNDS="${POLL_ROUNDS:-24}"
POLL_SLEEP_SEC="${POLL_SLEEP_SEC:-5}"
POLL_TAIL_LINES="${POLL_TAIL_LINES:-600}"
TELERUN_SUBMIT_TIMEOUT="${TELERUN_SUBMIT_TIMEOUT:-180}"
TELERUN_PULL_TIMEOUT="${TELERUN_PULL_TIMEOUT:-60}"
SUBMIT_NET_RETRIES="${SUBMIT_NET_RETRIES:-3}"
SUBMIT_NET_SLEEP_SEC="${SUBMIT_NET_SLEEP_SEC:-3}"
REMOTE_INSTALL_TRIES="${REMOTE_INSTALL_TRIES:-2}"
REMOTE_INSTALL_TIMEOUT="${REMOTE_INSTALL_TIMEOUT:-120}"
REMOTE_INSTALL_SLEEP_SEC="${REMOTE_INSTALL_SLEEP_SEC:-2}"
REMOTE_CHECK_NET_RETRIES="${REMOTE_CHECK_NET_RETRIES:-2}"

case "$DATASET" in
  sift1m)
    BASE="${BASE:-/home/sunzeai/proj/.data/sift/sift_base.fvecs}"
    QUERY="${QUERY:-/home/sunzeai/proj/.data/sift/sift_query.fvecs}"
    GT="${GT:-/home/sunzeai/proj/.data/sift/sift_groundtruth.ivecs}"
    GRAPH="${GRAPH:-/home/sunzeai/proj/.data/sift/graph-d32}"
    SIGNBIT_FILE="${SIGNBIT_FILE:-/home/sunzeai/proj/.data/sift/graph-d32.signbit.bin}"
    ;;
  sift10m)
    BASE="${BASE:-/home/sunzeai/proj/.data/sift10m/sift10m_base.fvecs}"
    QUERY="${QUERY:-/home/sunzeai/proj/.data/sift10m/sift10m_query.fvecs}"
    GT="${GT:-/home/sunzeai/proj/.data/sift10m/sift10m_groundtruth.ivecs}"
    GRAPH="${GRAPH:-/home/sunzeai/proj/.data/sift10m/graph-d32}"
    SIGNBIT_FILE="${SIGNBIT_FILE:-/home/sunzeai/proj/.data/sift10m/graph-d32.signbit.bin}"
    ;;
  *)
    echo "ERROR: unsupported DATASET=$DATASET" >&2
    exit 2
    ;;
esac

write_helpers() {
  cat > "$TMP_RUN_DIR/telerun_submit_single.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
LOCK_FILE="${TELERUN_LOCK_FILE:-/tmp/codex_telerun_single.lock}"
mkdir -p "$(dirname "${LOCK_FILE}")"
exec 9>"${LOCK_FILE}"
flock 9
if ! command -v telerun >/dev/null 2>&1; then
  echo "ERROR: telerun not found" >&2
  exit 127
fi
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
NET_RETRIES="${TELERUN_NET_RETRIES:-4}"
NET_SLEEP_SEC="${TELERUN_NET_SLEEP_SEC:-2}"
is_transient_net_error() {
  local s="$1"
  printf "%s\n" "$s" | grep -Eqi "UNEXPECTED_EOF_WHILE_READING|SSLEOFError|RemoteDisconnected|Temporary failure in name resolution|Name or service not known|Connection reset by peer|timed out|TLS|EOF occurred in violation of protocol"
}
attempt=1
while [ "$attempt" -le "$NET_RETRIES" ]; do
  tmp_out="$(mktemp)"
  set +e
  telerun "${args[@]}" "$@" >"${tmp_out}" 2>&1
  rc=$?
  set -e
  out="$(cat "${tmp_out}")"
  rm -f "${tmp_out}"
  printf "%s\n" "${out}"
  if [ "$rc" -eq 0 ]; then
    exit 0
  fi
  if [ "$attempt" -lt "$NET_RETRIES" ] && is_transient_net_error "${out}"; then
    sleep_sec=$(( NET_SLEEP_SEC * attempt ))
    sleep "${sleep_sec}"
    attempt=$((attempt + 1))
    continue
  fi
  exit "$rc"
done
exit 1
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
sha256sum "${DST_PATH}" || true
echo "__REMOTE_INSTALL_OK__=1"
EOS

  cat > "$TMP_RUN_DIR/run_clean_gpu_then_sift_persist_telerun.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
PERSIST_ROOT="${PERSIST_ROOT:-/home/sunzeai/proj/.data/.codex_logs}"
PERSIST_TAG="${PERSIST_TAG:-run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PERSIST_ROOT}"
LOG_FILE="${PERSIST_ROOT}/${PERSIST_TAG}.log"
run_main() {
  if [ "${CLEAN_GPU_BEFORE_RUN:-1}" = "1" ]; then
    nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader || true
    if [ "${CLEAN_GPU_KILL_ALL_COMPUTE:-0}" = "1" ]; then
      gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sed '/^$/d' | sort -u || true)"
      if [ -n "${gpu_pids}" ]; then
        kill -9 ${gpu_pids} 2>/dev/null || true
        sleep 1
      fi
    fi
    nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader || true
  fi
  BIN="$1"
  K="${2:-100}"
  BEAM="${3:-500}"
  WARMUP="${4:-0}"
  RUNS="${5:-1}"
  export GRAPH_INDEX_FORMAT="${GRAPH_INDEX_FORMAT:-read_bin}"
  export SIGNBIT_FILE="${SIGNBIT_FILE:-}"
  export RABITQ_TRIPLE_FILE="${RABITQ_TRIPLE_FILE:-}"
  export RABITQ_FACTOR_DQ_FILE="${RABITQ_FACTOR_DQ_FILE:-}"
  export RABITQ_FACTOR_VQ_FILE="${RABITQ_FACTOR_VQ_FILE:-}"
  export MAX_QUERIES="${MAX_QUERIES:-10000}"
  "${BIN}" --warmup "${WARMUP}" --runs "${RUNS}" "${BASE}" "${QUERY}" "${GT}" "${GRAPH}" "${K}" "${BEAM}"
}
{
  echo "PERSIST_TAG=${PERSIST_TAG}"
  echo "PERSIST_LOG=${LOG_FILE}"
  echo "ARGS=$*"
  run_main "$@"
} 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
echo "__EXIT_CODE__=${status}" | tee -a "${LOG_FILE}"
exit "${status}"
EOS

  chmod +x "$TMP_RUN_DIR"/*.sh
}

local_bin_path() {
  printf '%s/beam_search_gpu_%s_%s\n' "$LOCAL_BIN_DIR" "$MODE" "$TS"
}

remote_bin_path() {
  printf '%s/%s\n' "$REMOTE_BIN_ROOT" "$(basename "$(local_bin_path)")"
}

compile_bin() {
  local log="$RESULTS_DIR/${TS}.compile.log"
  make -B -C "$CODE_DIR" sm89 BUILD_DIR="$BUILD_DIR" NVFLAGS="$NVFLAGS" > "$log" 2>&1
  cp "$BUILD_DIR/beam_search_gpu" "$(local_bin_path)"
}

install_remote_bin_if_needed() {
  local local_bin remote_bin install_log check_out install_out install_ok=0 attempt
  local_bin="$(local_bin_path)"
  remote_bin="$(remote_bin_path)"
  install_log="$RESULTS_DIR/${TS}.remote_install.log"
  : > "$install_log"
  if [ "$REMOTE_BIN_CACHE" = "1" ] && [ "$REMOTE_BIN_REFRESH" != "1" ]; then
    check_out="$({
      timeout "$REMOTE_INSTALL_TIMEOUT" env TELERUN_BYPASS_LAST_JOB=1 TELERUN_NET_RETRIES="$REMOTE_CHECK_NET_RETRIES" \
        "$TMP_RUN_DIR/telerun_submit_single.sh" env "$TMP_RUN_DIR/remote_check_file.sh" "$remote_bin"
      echo "__CRC__=$?"
    } 2>&1 || true)"
    printf '%s\n' "$check_out" > "$install_log"
    if printf '%s\n' "$check_out" | grep -Eq '__REMOTE_FILE_OK__=1'; then
      printf '%s\n' "$remote_bin" > "$RESULTS_DIR/${TS}.binpath"
      return 0
    fi
  fi
  for attempt in $(seq 1 "$REMOTE_INSTALL_TRIES"); do
    install_out="$({
      timeout "$REMOTE_INSTALL_TIMEOUT" env TELERUN_BYPASS_LAST_JOB=1 \
        "$TMP_RUN_DIR/telerun_submit_single.sh" env "$TMP_RUN_DIR/remote_install_file.sh" "$local_bin" "$remote_bin"
      echo "__IRC__=$?"
    } 2>&1 || true)"
    printf '%s\n' "$install_out" >> "$install_log"
    if printf '%s\n' "$install_out" | grep -Eq '__REMOTE_INSTALL_OK__=1'; then
      install_ok=1
      break
    fi
    if [ "$attempt" -lt "$REMOTE_INSTALL_TRIES" ]; then
      sleep "$REMOTE_INSTALL_SLEEP_SEC"
    fi
  done
  if [ "$install_ok" = "1" ]; then
    printf '%s\n' "$remote_bin" > "$RESULTS_DIR/${TS}.binpath"
    return 0
  fi
  return 1
}

extract_metric() {
  local log="$1"
  local pattern="$2"
  python3 -c 'import re, sys; t=open(sys.argv[1], encoding="utf-8", errors="ignore").read(); m=re.findall(sys.argv[2], t); print(m[-1] if m else "NA")' "$log" "$pattern"
}

run_remote() {
  local bin log out ok=0 persist_tag remote_log pull_out i sr p
  local invalid_runtime_pattern='out of memory|cudaMalloc\(d_[^)]*\) failed|error 2: Cuda error|cudaError|Segmentation fault|Job failed\.|ERROR: dataset path check failed'
  bin="$(cat "$RESULTS_DIR/${TS}.binpath")"
  log="$RESULTS_DIR/${TS}.telerun.log"
  : > "$log"
  for i in $(seq 1 "$RETRY"); do
    persist_tag="${TS}_try${i}"
    remote_log="${PERSIST_ROOT}/${persist_tag}.log"
    out=""
    for sr in $(seq 1 "$SUBMIT_NET_RETRIES"); do
      out="$({
        timeout "$TELERUN_SUBMIT_TIMEOUT" env TELERUN_BYPASS_LAST_JOB=1 \
          "$TMP_RUN_DIR/telerun_submit_single.sh" env \
          MAX_QUERIES="$MAX_QUERIES" \
          CLEAN_GPU_BEFORE_RUN=1 \
          CLEAN_GPU_KILL_ALL_COMPUTE=1 \
          GRAPH_INDEX_FORMAT=read_bin \
          REFLOW_APPROX_MODE="${REFLOW_APPROX_MODE:-routea}" \
          SIGNBIT_FILE="$SIGNBIT_FILE" \
          RABITQ_TRIPLE_FILE="${RABITQ_TRIPLE_FILE:-}" \
          RABITQ_FACTOR_DQ_FILE="${RABITQ_FACTOR_DQ_FILE:-}" \
          RABITQ_FACTOR_VQ_FILE="${RABITQ_FACTOR_VQ_FILE:-}" \
          BASE="$BASE" QUERY="$QUERY" GT="$GT" GRAPH="$GRAPH" \
          PERSIST_ROOT="$PERSIST_ROOT" \
          PERSIST_TAG="$persist_tag" \
          "$TMP_RUN_DIR/run_clean_gpu_then_sift_persist_telerun.sh" "$bin" "$K" "$BEAM" "$WARMUP" "$RUNS"
        echo "__RC__=$?"
      } 2>&1 || true)"
      printf '%s\n' "$out" >> "$log"
      if printf '%s\n' "$out" | grep -Eq 'Submitted job'; then
        break
      fi
      if printf '%s\n' "$out" | grep -Eq 'Job completed successfully\.' && printf '%s\n' "$out" | grep -Eq 'Throughput:' && printf '%s\n' "$out" | grep -Eq 'Recall@100 \(unique hit'; then
        ok=1
        break 2
      fi
      if printf '%s\n' "$out" | grep -Eq 'RemoteDisconnected|UNEXPECTED_EOF_WHILE_READING|SSLEOFError|Temporary failure in name resolution|__RC__=124'; then
        sleep "$SUBMIT_NET_SLEEP_SEC"
        continue
      fi
      break
    done
    if [ "$ok" -eq 1 ]; then
      break
    fi
    if ! printf '%s\n' "$out" | grep -Eq 'Submitted job'; then
      sleep "$SLEEP_SEC"
      continue
    fi
    for p in $(seq 1 "$POLL_ROUNDS"); do
      pull_out="$({
        timeout "$TELERUN_PULL_TIMEOUT" env TELERUN_OVERRIDE_PENDING=0 TELERUN_BYPASS_LAST_JOB=1 \
          "$TMP_RUN_DIR/telerun_submit_single.sh" env tail -n "$POLL_TAIL_LINES" "$remote_log"
        echo "__PRC__=$?"
      } 2>&1 || true)"
      printf '%s\n' "$pull_out" >> "$log"
      if printf '%s\n' "$pull_out" | grep -Eq 'Throughput:' && printf '%s\n' "$pull_out" | grep -Eq 'Recall@100 \(unique hit' && ! printf '%s\n' "$pull_out" | grep -Eqi "$invalid_runtime_pattern"; then
        ok=1
        break 2
      fi
      if printf '%s\n' "$pull_out" | grep -Eqi "$invalid_runtime_pattern|ERROR:"; then
        break
      fi
      sleep "$POLL_SLEEP_SEC"
    done
    sleep "$SLEEP_SEC"
  done
  if [ "$ok" -ne 1 ]; then
    return 1
  fi
  local qps recall dc hops kernel
  qps="$(extract_metric "$log" 'Throughput:\s*([\-\+]?\d+(?:\.\d+)?(?:[eE][\-\+]?\d+)?)')"
  recall="$(extract_metric "$log" 'Recall@100 \(unique hit, on \d+ queries\):\s*([\-\+]?\d+(?:\.\d+)?(?:[eE][\-\+]?\d+)?)')"
  dc="$(extract_metric "$log" 'average # distance computation:\s*([\-\+]?\d+(?:\.\d+)?(?:[eE][\-\+]?\d+)?)')"
  hops="$(extract_metric "$log" 'average # hops:\s*([\-\+]?\d+(?:\.\d+)?(?:[eE][\-\+]?\d+)?)')"
  kernel="$(extract_metric "$log" 'GPU kernel time \(BeamSearch \+ sync\):\s*([\-\+]?\d+(?:\.\d+)?(?:[eE][\-\+]?\d+)?) ms')"
  printf 'mode=%s\nqps=%s\nrecall=%s\navg_dc=%s\navg_hops=%s\nkernel_ms=%s\n' "$MODE" "$qps" "$recall" "$dc" "$hops" "$kernel" > "$RESULTS_DIR/${TS}.metrics.txt"
  printf '%s,%s,%s,%s,%s,%s\n' "$MODE" "$qps" "$recall" "$dc" "$hops" "$kernel" > "$RESULTS_DIR/${TS}.metrics.csv"
  cat > "$RESULTS_DIR/${TS}.summary.txt" <<EOS
MODE=$MODE
DATASET=$DATASET
BEAM=$BEAM
K=$K
RUNS=$RUNS
WARMUP=$WARMUP
MAX_QUERIES=$MAX_QUERIES
QPS=$qps
RECALL=$recall
AVG_DC=$dc
AVG_HOPS=$hops
KERNEL_MS=$kernel
EOS
  cat "$RESULTS_DIR/${TS}.summary.txt"
}

write_helpers
compile_bin
install_remote_bin_if_needed
run_remote
