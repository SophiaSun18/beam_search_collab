#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v telerun >/dev/null 2>&1; then
  echo "ERROR: telerun command not found" >&2
  exit 127
fi

DATASET="${DATASET:-sift10m}"
BEAM="${BEAM:-300}"
K="${K:-100}"
RUNS="${RUNS:-1}"
WARMUP="${WARMUP:-1}"
MAX_QUERIES="${MAX_QUERIES:-10000}"
TS="${TS:-fixed_reflow_${DATASET}_b${BEAM}_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="$ROOT/results"
TMP_RUN_DIR="$RESULTS_DIR/.runner_${TS}"
mkdir -p "$RESULTS_DIR" "$TMP_RUN_DIR"

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

TOPN=20
ANCHOR_EXACT=32
SCORE_WORDS=4
PARTIAL_L2_LANES=24
SEARCH_WIDTH=7
BITLEN=12

case "$DATASET" in
  sift1m)
    POP_BATCH=6
    BASE="${BASE:-/home/sunzeai/proj/.data/sift/sift_base.fvecs}"
    QUERY="${QUERY:-/home/sunzeai/proj/.data/sift/sift_query.fvecs}"
    GT="${GT:-/home/sunzeai/proj/.data/sift/sift_groundtruth.ivecs}"
    GRAPH="${GRAPH:-/home/sunzeai/proj/.data/sift/graph-d32}"
    SIGNBIT_FILE="${SIGNBIT_FILE:-/home/sunzeai/proj/.data/sift/graph-d32.ref.signbit.bin}"
    ;;
  sift10m)
    POP_BATCH=7
    BASE="${BASE:-/home/sunzeai/proj/.data/sift10m/sift10m_base.fvecs}"
    QUERY="${QUERY:-/home/sunzeai/proj/.data/sift10m/sift10m_query.fvecs}"
    GT="${GT:-/home/sunzeai/proj/.data/sift10m/sift10m_groundtruth.ivecs}"
    GRAPH="${GRAPH:-/home/sunzeai/proj/.data/sift10m/graph-d32}"
    SIGNBIT_FILE="${SIGNBIT_FILE:-/home/sunzeai/proj/.data/sift10m/graph-d32.ref.signbit.bin}"
    ;;
  *)
    echo "ERROR: unsupported DATASET=$DATASET (use sift1m or sift10m)" >&2
    exit 2
    ;;
esac

COMMON_FLAGS="--use_fast_math -std=c++17 -O3 -gencode arch=compute_89,code=sm_89"
COMMON_FLAGS+=" -DBEAM_SIZE=${BEAM} -DBITLEN=${BITLEN} -DSEARCH_WIDTH=${SEARCH_WIDTH}"
COMMON_FLAGS+=" -DENABLE_GHOST_STAGE=0 -DENABLE_SQ8=0 -DSQ8_DIST_PRUNE_ENABLE=0 -DENABLE_DGS=0 -DENABLE_SQG=0"
COMMON_FLAGS+=" -DCANDIDATE_SELECT_STRICT=2 -DSTRICT_TOPK_PARTIAL_SELECT_MODE=0"
COMMON_FLAGS+=" -DSKIP_ATOMIC_WHEN_POOL_FULL=1 -DHASH_INSERT_MAX_PROBES=24"
COMMON_FLAGS+=" -DHASH_INSERT_LIMIT_START_FILL=160 -DHASH_INSERT_MAX_PROBES_HIGH_FILL=0 -DHASH_INSERT_HIGH_FILL_START=224"
COMMON_FLAGS+=" -DWARP_L1_DEDUP_ENABLE=1 -DWARP_L1_HASH_BITLEN=7"
COMMON_FLAGS+=" -DPARTIAL_L2_128_LANES=${PARTIAL_L2_LANES} -DUSE_SUBWARP_L2_128=0"

OFF_FLAGS="${COMMON_FLAGS} -DENABLE_PARTIAL_L2_128=0 -DENABLE_RABITQ=0"
RB_FLAGS="${COMMON_FLAGS}"
RB_FLAGS+=" -DCANDIDATE_SELECT_STRICT_FAST=0 -DCANDIDATE_SELECT_STRICT_FAST_DGS_ONLY=0 -DCANDIDATE_SELECT_STRICT_ADAPTIVE=0"
RB_FLAGS+=" -DENABLE_PARTIAL_L2_128=1 -DPARTIAL_L2_128_LANES=${PARTIAL_L2_LANES}"
RB_FLAGS+=" -DENABLE_RABITQ=1 -DRABITQ_REFLOW_FIXED=1"
RB_FLAGS+=" -DRABITQ_WARP_SELECT_SORT=0 -DRABITQ_WARP_SELECT_SORT_MIN=12"
RB_FLAGS+=" -DREFLOW_TOPN=${TOPN} -DREFLOW_TOPN_EARLY=${TOPN} -DREFLOW_TOPN_EARLY_ITERS=0"
RB_FLAGS+=" -DREFLOW_POP_BATCH=${POP_BATCH} -DREFLOW_ANCHOR_EXACT=${ANCHOR_EXACT}"
RB_FLAGS+=" -DREFLOW_SCORE_WORDS=${SCORE_WORDS} -DREFLOW_SKIP_ATOMIC_POLICY=0"
RB_FLAGS+=" -DREFLOW_FASTSCAN_ENABLE=1 -DREFLOW_PACKED_CODE_ENABLE=1"
RB_FLAGS+=" -DREFLOW_USE_FACTORS=0 -DREFLOW_FACTOR_LIGHT_PRESELECT=0 -DREFLOW_FACTOR_PRESELECT_EXTRA=0"
RB_FLAGS+=" -DREFLOW_FACTOR_PACKED_DQVQ=1 -DREFLOW_FACTOR_FP16_DQVQ=1"
RB_FLAGS+=" -DREFLOW_QUERY_PROJ_ENABLE=0 -DREFLOW_GPU_DUALPOOL_ENGINE=1"

write_helpers() {
  cat > "$TMP_RUN_DIR/telerun_submit_single.sh" <<'EOF'
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
  printf "%s\n" "$s" | rg -qi "UNEXPECTED_EOF_WHILE_READING|SSLEOFError|RemoteDisconnected|Temporary failure in name resolution|Name or service not known|Connection reset by peer|timed out|TLS|EOF occurred in violation of protocol"
}
echo "[singleflight] lock=${LOCK_FILE}"
echo "[singleflight] submit: telerun ${args[*]} $*"
attempt=1
while [ "${attempt}" -le "${NET_RETRIES}" ]; do
  tmp_out="$(mktemp)"
  set +e
  telerun "${args[@]}" "$@" >"${tmp_out}" 2>&1
  rc=$?
  set -e
  out="$(cat "${tmp_out}")"
  rm -f "${tmp_out}"
  printf "%s\n" "${out}"
  if [ "${rc}" -eq 0 ]; then
    echo "[singleflight] exit=${rc}"
    exit 0
  fi
  if [ "${attempt}" -lt "${NET_RETRIES}" ] && is_transient_net_error "${out}"; then
    sleep_sec=$(( NET_SLEEP_SEC * attempt ))
    echo "[singleflight] transient network error, retry ${attempt}/${NET_RETRIES} after ${sleep_sec}s"
    sleep "${sleep_sec}"
    attempt=$((attempt + 1))
    continue
  fi
  echo "[singleflight] exit=${rc}"
  exit "${rc}"
done
echo "[singleflight] exit=1"
exit 1
EOF

  cat > "$TMP_RUN_DIR/remote_check_file.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
FILE_PATH="${1:-}"
if [ -z "${FILE_PATH}" ]; then
  echo "ERROR: missing file path" >&2
  exit 2
fi
if [ -f "${FILE_PATH}" ]; then
  ls -lh "${FILE_PATH}" || true
  echo "__REMOTE_FILE_OK__=1"
else
  echo "MISSING: ${FILE_PATH}"
  echo "__REMOTE_FILE_OK__=0"
fi
EOF

  cat > "$TMP_RUN_DIR/remote_install_file.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SRC_PATH="${1:-}"
DST_PATH="${2:-}"
if [ -z "${SRC_PATH}" ] || [ -z "${DST_PATH}" ]; then
  echo "ERROR: usage remote_install_file.sh <src> <dst>" >&2
  exit 2
fi
if [ ! -f "${SRC_PATH}" ]; then
  src_base="$(basename "${SRC_PATH}")"
  if [ -f "${src_base}" ]; then
    SRC_PATH="${src_base}"
  fi
fi
if [ ! -f "${SRC_PATH}" ]; then
  echo "ERROR: source file not found: ${SRC_PATH}" >&2
  exit 3
fi
mkdir -p "$(dirname "${DST_PATH}")"
cp -f "${SRC_PATH}" "${DST_PATH}"
chmod +x "${DST_PATH}" 2>/dev/null || true
ls -lh "${DST_PATH}" || true
sha256sum "${DST_PATH}" || true
echo "__REMOTE_INSTALL_OK__=1"
EOF

  cat > "$TMP_RUN_DIR/run_clean_gpu_then_sift_persist_telerun.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
PERSIST_ROOT="${PERSIST_ROOT:-/home/sunzeai/proj/.data/.codex_logs}"
PERSIST_TAG="${PERSIST_TAG:-run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PERSIST_ROOT}"
LOG_FILE="${PERSIST_ROOT}/${PERSIST_TAG}.log"
run_main() {
  if [ "${CLEAN_GPU_BEFORE_RUN:-1}" = "1" ]; then
    echo "=== GPU Preflight ==="
    nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader || true
    if [ "${CLEAN_GPU_KILL_ALL_COMPUTE:-0}" = "1" ]; then
      gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sed '/^$/d' | sort -u || true)"
      if [ -n "${gpu_pids}" ]; then
        echo "Aggressive cleanup: killing all GPU compute pids: ${gpu_pids}"
        kill -9 ${gpu_pids} 2>/dev/null || true
        sleep 1
      fi
    fi
    nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader || true
  fi
  BIN="./beam_search_gpu"
  if [ "${1:-}" != "" ]; then
    if [ ! "${1//[0-9]/}" = "" ] && [ -f "$1" ]; then
      chmod +x "$1" 2>/dev/null || true
      BIN="$1"
      shift
    fi
  fi
  K="${1:-100}"
  BEAM="${2:-500}"
  WARMUP="${3:-0}"
  RUNS="${4:-1}"
  if [ ! -f "${BASE}" ] || [ ! -f "${QUERY}" ] || [ ! -f "${GT}" ] || [ ! -e "${GRAPH}" ]; then
    echo "ERROR: dataset path check failed."
    echo "BASE=${BASE}"
    echo "QUERY=${QUERY}"
    echo "GT=${GT}"
    echo "GRAPH=${GRAPH}"
    return 2
  fi
  export GRAPH_INDEX_FORMAT="${GRAPH_INDEX_FORMAT:-read_bin}"
  export SIGNBIT_FILE="${SIGNBIT_FILE}"
  export MAX_QUERIES="${MAX_QUERIES:-10000}"
  echo "=== Launch Search Binary ==="
  echo "BIN=${BIN}"
  echo "GRAPH_INDEX_FORMAT=${GRAPH_INDEX_FORMAT}"
  echo "SIGNBIT_FILE=${SIGNBIT_FILE}"
  echo "BASE=${BASE}"
  echo "QUERY=${QUERY}"
  echo "GT=${GT}"
  echo "GRAPH=${GRAPH}"
  "${BIN}" --warmup "${WARMUP}" --runs "${RUNS}" "${BASE}" "${QUERY}" "${GT}" "${GRAPH}" "${K}" "${BEAM}"
}
{
  echo "=== Persist Wrapper Start ==="
  echo "DATE=$(date -Iseconds)"
  echo "HOST=$(hostname)"
  echo "PWD=$(pwd)"
  echo "PERSIST_TAG=${PERSIST_TAG}"
  echo "PERSIST_LOG=${LOG_FILE}"
  echo "ARGS=$*"
  run_main "$@"
} 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
echo "__EXIT_CODE__=${status}" | tee -a "${LOG_FILE}"
exit "${status}"
EOF

  chmod +x "$TMP_RUN_DIR"/*.sh
}

compile_case() {
  local mode="$1"
  local flags="$2"
  local log="$RESULTS_DIR/${TS}.${mode}.compile.log"
  make -B -C code sm89 NVFLAGS="$flags" > "$log" 2>&1
  cp code/beam_search_gpu "code/beam_search_gpu_${mode}_${TS}"
}

local_bin_path() {
  local mode="$1"
  printf '%s/code/beam_search_gpu_%s_%s\n' "$ROOT" "$mode" "$TS"
}

remote_bin_path() {
  local mode="$1"
  printf '%s/%s\n' "$REMOTE_BIN_ROOT" "$(basename "$(local_bin_path "$mode")")"
}

install_remote_bin_if_needed() {
  local mode="$1"
  local install_log local_bin remote_bin check_out install_out install_ok=0 attempt
  local_bin="$(local_bin_path "$mode")"
  remote_bin="$(remote_bin_path "$mode")"
  install_log="$RESULTS_DIR/${TS}.${mode}.remote_install.log"
  : > "$install_log"

  if [ "$REMOTE_BIN_CACHE" = "1" ] && [ "$REMOTE_BIN_REFRESH" != "1" ]; then
    check_out="$({
      timeout "$REMOTE_INSTALL_TIMEOUT" env TELERUN_BYPASS_LAST_JOB=1 TELERUN_NET_RETRIES="$REMOTE_CHECK_NET_RETRIES" \
        "$TMP_RUN_DIR/telerun_submit_single.sh" env "$TMP_RUN_DIR/remote_check_file.sh" "$remote_bin"
      echo "__CRC__=$?"
    } 2>&1 || true)"
    printf '[remote-check %s]\n%s\n' "$mode" "$check_out" > "$install_log"
    if printf '%s\n' "$check_out" | rg -q '__REMOTE_FILE_OK__=1'; then
      printf '%s\n' "$remote_bin" > "$RESULTS_DIR/${TS}.${mode}.binpath"
      return 0
    fi
  fi

  for attempt in $(seq 1 "$REMOTE_INSTALL_TRIES"); do
    install_out="$({
      timeout "$REMOTE_INSTALL_TIMEOUT" env TELERUN_BYPASS_LAST_JOB=1 \
        "$TMP_RUN_DIR/telerun_submit_single.sh" env "$TMP_RUN_DIR/remote_install_file.sh" "$local_bin" "$remote_bin"
      echo "__IRC__=$?"
    } 2>&1 || true)"
    printf '\n[remote-install %s attempt %s/%s]\n%s\n' "$mode" "$attempt" "$REMOTE_INSTALL_TRIES" "$install_out" >> "$install_log"
    if printf '%s\n' "$install_out" | rg -q '__REMOTE_INSTALL_OK__=1'; then
      install_ok=1
      break
    fi
    if [ "$attempt" -lt "$REMOTE_INSTALL_TRIES" ]; then
      sleep "$REMOTE_INSTALL_SLEEP_SEC"
    fi
  done

  if [ "$install_ok" = "1" ]; then
    printf '%s\n' "$remote_bin" > "$RESULTS_DIR/${TS}.${mode}.binpath"
    return 0
  fi

  echo "ERROR: remote binary install failed for mode=$mode" | tee -a "$install_log" >&2
  return 1
}

extract_metric() {
  local log="$1"
  local pattern="$2"
  python3 -c 'import re, sys; t=open(sys.argv[1], encoding="utf-8", errors="ignore").read(); m=re.findall(sys.argv[2], t); print(m[-1] if m else "NA")' "$log" "$pattern"
}

run_case() {
  local mode="$1"
  local bin log out ok=0 persist_tag remote_log pull_out i sr p
  local invalid_runtime_pattern='out of memory|cudaMalloc\(d_[^)]*\) failed|error 2: Cuda error|cudaError|Segmentation fault|Job failed\.|ERROR: dataset path check failed'
  bin="$(cat "$RESULTS_DIR/${TS}.${mode}.binpath")"
  log="$RESULTS_DIR/${TS}.${mode}.telerun.log"
  : > "$log"

  for i in $(seq 1 "$RETRY"); do
    persist_tag="${TS}_${mode}_try${i}"
    remote_log="${PERSIST_ROOT}/${persist_tag}.log"
    echo "[$mode] try $i/$RETRY" | tee -a "$log"
    out=""

    for sr in $(seq 1 "$SUBMIT_NET_RETRIES"); do
      out="$({
        timeout "$TELERUN_SUBMIT_TIMEOUT" env TELERUN_BYPASS_LAST_JOB=1 \
          "$TMP_RUN_DIR/telerun_submit_single.sh" env \
          MAX_QUERIES="$MAX_QUERIES" \
          CLEAN_GPU_BEFORE_RUN=1 \
          CLEAN_GPU_KILL_ALL_COMPUTE=1 \
          GRAPH_INDEX_FORMAT=read_bin \
          SIGNBIT_FILE="$SIGNBIT_FILE" \
          BASE="$BASE" QUERY="$QUERY" GT="$GT" GRAPH="$GRAPH" \
          PERSIST_ROOT="$PERSIST_ROOT" \
          PERSIST_TAG="$persist_tag" \
          "$TMP_RUN_DIR/run_clean_gpu_then_sift_persist_telerun.sh" "$bin" "$K" "$BEAM" "$WARMUP" "$RUNS"
        echo "__RC__=$?"
      } 2>&1 || true)"
      printf '[%s] submit-net %s/%s\n%s\n' "$mode" "$sr" "$SUBMIT_NET_RETRIES" "$out" >> "$log"

      if printf '%s\n' "$out" | rg -q 'Submitted job'; then
        break
      fi
      if printf '%s\n' "$out" | rg -q 'Job completed successfully\.' \
        && printf '%s\n' "$out" | rg -q 'Throughput:' \
        && printf '%s\n' "$out" | rg -q 'Recall@100 \(unique hit'; then
        ok=1
        break 2
      fi
      if printf '%s\n' "$out" | rg -q 'RemoteDisconnected|UNEXPECTED_EOF_WHILE_READING|SSLEOFError|Temporary failure in name resolution|__RC__=124'; then
        echo "[$mode] transient submit error; net-retry sleep ${SUBMIT_NET_SLEEP_SEC}s" | tee -a "$log"
        sleep "$SUBMIT_NET_SLEEP_SEC"
        continue
      fi
      break
    done

    if [ "$ok" -eq 1 ]; then
      break
    fi

    if ! printf '%s\n' "$out" | rg -q 'Submitted job'; then
      echo "[$mode] submit not acknowledged, retry sleep ${SLEEP_SEC}s" | tee -a "$log"
      sleep "$SLEEP_SEC"
      continue
    fi

    for p in $(seq 1 "$POLL_ROUNDS"); do
      pull_out="$({
        timeout "$TELERUN_PULL_TIMEOUT" env TELERUN_OVERRIDE_PENDING=0 TELERUN_BYPASS_LAST_JOB=1 \
          "$TMP_RUN_DIR/telerun_submit_single.sh" env tail -n "$POLL_TAIL_LINES" "$remote_log"
        echo "__PRC__=$?"
      } 2>&1 || true)"
      printf '\n[%s] pull %s/%s\n%s\n' "$mode" "$p" "$POLL_ROUNDS" "$pull_out" >> "$log"

      if printf '%s\n' "$pull_out" | rg -q 'Throughput:' \
        && printf '%s\n' "$pull_out" | rg -q 'Recall@100 \(unique hit' \
        && ! printf '%s\n' "$pull_out" | rg -qi "$invalid_runtime_pattern"; then
        ok=1
        break 2
      fi
      if printf '%s\n' "$pull_out" | rg -qi "$invalid_runtime_pattern|ERROR:"; then
        echo "[$mode] detected runtime failure signal on persisted log" | tee -a "$log"
        break
      fi
      sleep "$POLL_SLEEP_SEC"
    done

    echo "[$mode] retry sleep ${SLEEP_SEC}s" | tee -a "$log"
    sleep "$SLEEP_SEC"
  done

  if [ "$ok" -ne 1 ]; then
    echo "ERROR: remote run failed for mode=$mode" | tee -a "$log" >&2
    return 1
  fi

  local qps recall dc hops kernel
  qps="$(extract_metric "$log" 'Throughput:\s*([-+]?\d+(?:\.\d+)?)')"
  recall="$(extract_metric "$log" 'Recall@100 \(unique hit, on \d+ queries\):\s*([-+]?\d+(?:\.\d+)?)')"
  dc="$(extract_metric "$log" 'average # distance computation:\s*([-+]?\d+(?:\.\d+)?)')"
  hops="$(extract_metric "$log" 'average # hops:\s*([-+]?\d+(?:\.\d+)?)')"
  kernel="$(extract_metric "$log" 'GPU kernel time \(BeamSearch \+ sync\):\s*([-+]?\d+(?:\.\d+)?) ms')"
  printf '%s,%s,%s,%s,%s,%s\n' "$mode" "$qps" "$recall" "$dc" "$hops" "$kernel" > "$RESULTS_DIR/${TS}.${mode}.metrics.csv"
  printf 'mode=%s\nqps=%s\nrecall=%s\navg_dc=%s\navg_hops=%s\nkernel_ms=%s\n' "$mode" "$qps" "$recall" "$dc" "$hops" "$kernel" > "$RESULTS_DIR/${TS}.${mode}.metrics.txt"
}

write_helpers

echo "=== Fixed Reflow A/B ==="
echo "DATASET=$DATASET BEAM=$BEAM K=$K RUNS=$RUNS WARMUP=$WARMUP MAX_QUERIES=$MAX_QUERIES"
echo "TOPN=$TOPN POP_BATCH=$POP_BATCH ANCHOR_EXACT=$ANCHOR_EXACT SCORE_WORDS=$SCORE_WORDS PARTIAL_L2_LANES=$PARTIAL_L2_LANES"

echo "[off] compiling..."
compile_case off "$OFF_FLAGS"
echo "[rb] compiling..."
compile_case rb "$RB_FLAGS"

echo "[off] installing remote binary..."
install_remote_bin_if_needed off
echo "[rb] installing remote binary..."
install_remote_bin_if_needed rb

echo "[off] running..."
run_case off
echo "[rb] running..."
run_case rb

off_qps="$(sed -n 's/^qps=//p' "$RESULTS_DIR/${TS}.off.metrics.txt")"
off_recall="$(sed -n 's/^recall=//p' "$RESULTS_DIR/${TS}.off.metrics.txt")"
off_dc="$(sed -n 's/^avg_dc=//p' "$RESULTS_DIR/${TS}.off.metrics.txt")"
off_hops="$(sed -n 's/^avg_hops=//p' "$RESULTS_DIR/${TS}.off.metrics.txt")"
off_kernel="$(sed -n 's/^kernel_ms=//p' "$RESULTS_DIR/${TS}.off.metrics.txt")"
rb_qps="$(sed -n 's/^qps=//p' "$RESULTS_DIR/${TS}.rb.metrics.txt")"
rb_recall="$(sed -n 's/^recall=//p' "$RESULTS_DIR/${TS}.rb.metrics.txt")"
rb_dc="$(sed -n 's/^avg_dc=//p' "$RESULTS_DIR/${TS}.rb.metrics.txt")"
rb_hops="$(sed -n 's/^avg_hops=//p' "$RESULTS_DIR/${TS}.rb.metrics.txt")"
rb_kernel="$(sed -n 's/^kernel_ms=//p' "$RESULTS_DIR/${TS}.rb.metrics.txt")"

delta_qps="$(awk -v o="$off_qps" -v r="$rb_qps" 'BEGIN{printf "%+.3f", (r-o)/o*100.0}')"
delta_recall="$(awk -v o="$off_recall" -v r="$rb_recall" 'BEGIN{printf "%+.6f", r-o}')"
delta_dc="$(awk -v o="$off_dc" -v r="$rb_dc" 'BEGIN{printf "%+.0f", r-o}')"

cat > "$RESULTS_DIR/${TS}.summary.txt" <<SUMMARY
DATASET=$DATASET
BEAM=$BEAM
K=$K
RUNS=$RUNS
WARMUP=$WARMUP
MAX_QUERIES=$MAX_QUERIES
TOPN=$TOPN
POP_BATCH=$POP_BATCH
ANCHOR_EXACT=$ANCHOR_EXACT
SCORE_WORDS=$SCORE_WORDS
PARTIAL_L2_LANES=$PARTIAL_L2_LANES

OFF: qps=$off_qps recall=$off_recall avg_dc=$off_dc avg_hops=$off_hops kernel_ms=$off_kernel
RB:  qps=$rb_qps recall=$rb_recall avg_dc=$rb_dc avg_hops=$rb_hops kernel_ms=$rb_kernel
DELTA: qps_pct=$delta_qps recall=$delta_recall dc=$delta_dc
SUMMARY

cat "$RESULTS_DIR/${TS}.summary.txt"
echo "RESULT_DIR=$RESULTS_DIR"
