#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROUTEA_ROOT="${ROUTEA_ROOT:-$ROOT}"
BASELINE_ROOT="${BASELINE_ROOT:-/workspace/project/research_engine_reflow_split_mainlines_20260316/off_mainline}"
BASELINE_RUNNER="${BASELINE_RUNNER:-./run_off_telerun.sh}"

DATASET="${DATASET:-sift10m}"
K="${K:-100}"
RUNS="${RUNS:-1}"
WARMUP="${WARMUP:-1}"
MAX_QUERIES="${MAX_QUERIES:-10000}"
BEAMS="${BEAMS:-100 200 300 400 500 600 700 800 900 1000 1100 1200}"
STAMP="${STAMP:-beam_sweep_compare_${DATASET}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROUTEA_ROOT/results/${STAMP}}"

mkdir -p "$OUT_DIR"

SUMMARY_CSV="$OUT_DIR/beam_sweep_summary.csv"
printf 'mode,beam,qps,recall,avg_dc,avg_iter,kernel_ms,summary_path,source_root\n' > "$SUMMARY_CSV"

run_one() {
  local mode="$1"
  local beam="$2"
  local workdir=""
  local runner=""
  local ts=""

  case "$mode" in
    routea)
      workdir="$ROUTEA_ROOT"
      runner="./run_routea_telerun.sh"
      ts="routea_${DATASET}_b${beam}_${STAMP}"
      ;;
    stable)
      workdir="$BASELINE_ROOT"
      runner="$BASELINE_RUNNER"
      ts="off_${DATASET}_b${beam}_${STAMP}"
      ;;
    *)
      echo "unknown mode: $mode" >&2
      return 2
      ;;
  esac

  echo "[run] mode=$mode beam=$beam ts=$ts"
  (
    cd "$workdir"
    TS="$ts" DATASET="$DATASET" BEAM="$beam" K="$K" RUNS="$RUNS" WARMUP="$WARMUP" MAX_QUERIES="$MAX_QUERIES" \
      "$runner"
  )

  local summary="$workdir/results/${ts}.summary.txt"
  if [[ ! -f "$summary" ]]; then
    echo "missing summary: $summary" >&2
    return 3
  fi

  local qps recall avg_dc avg_iter kernel_ms
  qps="$(awk -F= '/^QPS=/{print $2}' "$summary")"
  recall="$(awk -F= '/^RECALL=/{print $2}' "$summary")"
  avg_dc="$(awk -F= '/^AVG_DC=/{print $2}' "$summary")"
  avg_iter="$(awk -F= '/^AVG_HOPS=/{print $2}' "$summary")"
  kernel_ms="$(awk -F= '/^KERNEL_MS=/{print $2}' "$summary")"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$mode" "$beam" "$qps" "$recall" "$avg_dc" "$avg_iter" "$kernel_ms" "$summary" "$workdir" \
    >> "$SUMMARY_CSV"
}

for beam in $BEAMS; do
  run_one stable "$beam"
  run_one routea "$beam"
done

python3 "$ROOT/scripts/plot_beam_sweep_compare.py" "$SUMMARY_CSV" "$OUT_DIR"

echo "summary_csv=$SUMMARY_CSV"
echo "out_dir=$OUT_DIR"
