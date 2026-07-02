#!/usr/bin/env bash
# telerun_distance_cost_worker.sh
# Runs INSIDE telerun's RTX 4000 Ada (sm_89) host. It:
#   1. extracts the standalone distance_cost_bench source tree from a tar.gz,
#   2. runs a fuser-based GPU occupancy precheck (AGENTS.md method),
#   3. builds Part-1 distance cost benchmarks: CPU (host CPU) + GPU (sm_89),
#   4. optionally exports real access traces from search and replays them,
#   5. emits structured banners for the parser.
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
RUN_PRIMITIVE="${RUN_PRIMITIVE:-1}"
RUN_TRACE_REPLAY="${RUN_TRACE_REPLAY:-1}"
TOPK="${TOPK:-100}"
TRACE_BEAM="${TRACE_BEAM:-128}"
TRACE_QUERY_LIMIT="${TRACE_QUERY_LIMIT:-100}"
TRACE_CPU_REPEAT="${TRACE_CPU_REPEAT:-3}"
TRACE_GPU_REPEAT="${TRACE_GPU_REPEAT:-10}"
TRACE_WARMUP="${TRACE_WARMUP:-1}"
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
  GT="${DATA_ROOT}/$1/$1_groundtruth.ivecs"
  GRAPH="${DATA_ROOT}/$1/SymQG.$1.graph.noNan.fastscan.bin"
  QG_CPU="${DATA_ROOT}/$1/symphonyqg${DEGREE}.noNan.fastscan.index"
  QG_GPU="${DATA_ROOT}/$1/symphonyqg${DEGREE}.noNan.sequential.LUT.index"
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

emit_file() {
  local tag="$1" f="$2"
  echo "__FILE_BEGIN__ ${tag}"
  cat "$f" 2>/dev/null || echo "(missing $f)"
  echo "__FILE_END__ ${tag}"
}

run_gpu_cmd() {
  local label="$1"; shift
  set +e
  "$@"
  local rc=$?
  set -e
  if [ "${rc}" != "0" ]; then
    echo "__GPU_RUN_FAIL__=${label} rc=${rc}"
  fi
  return "${rc}"
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
  echo "RUN_PRIMITIVE=${RUN_PRIMITIVE}  RUN_TRACE_REPLAY=${RUN_TRACE_REPLAY}  TOPK=${TOPK}  TRACE_BEAM=${TRACE_BEAM}  TRACE_QUERY_LIMIT=${TRACE_QUERY_LIMIT}"
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

  local OUT="${PWD}/distance_cost_out"
  mkdir -p "${OUT}"

  if [ "${RUN_PRIMITIVE}" = "1" ]; then
    # CPU benchmark is dimension-agnostic (reads dim from codebook); build once.
    echo "=== build CPU distance benchmark (host CPU) ==="
    make distance_cpu
  fi

  if [ "${RUN_TRACE_REPLAY}" = "1" ]; then
    echo "=== build CPU access trace exporters + replay ==="
    make trace_cpu replay_cpu
    ls -lh bin/trace_cpu_base bin/trace_cpu_rabitq bin/replay_cpu

    echo "=== build GPU access trace exporter + replay ==="
    make trace_gpu replay_gpu NVCC="${NVCC}" PTXAS_FLAGS="" \
      EXTRA_NVFLAGS="-DINTERNAL_TOPK=${TRACE_BEAM} -cudart static"
    ls -lh bin/trace_gpu bin/replay_gpu
  fi

  for ds in $(split_csv "${DATASETS}"); do
    dataset_dims "${ds}"
    dataset_files "${ds}"
    echo "============================================================"
    echo "=== DATASET ${ds} (DIM=${DIM} PADDED=${PADDED}) ==="
    local ok=1
    for f in "${BASE}" "${QUERY}" "${QG_CPU}"; do
      if [ ! -f "$f" ]; then echo "WARN: missing required file: $f (skip ${ds})" >&2; ok=0; break; fi
      ls -lh "$f"
    done
    [ "${ok}" = "1" ] || { echo "__MB_SKIP__=${ds}"; continue; }

    if [ "${RUN_PRIMITIVE}" = "1" ]; then
      echo "=== CPU distance benchmark ${ds} (ns/op) ==="
      echo "__MB_CPU_BEGIN__ ${ds}"
      bin/distance_cpu -base "${BASE}" -query "${QUERY}" -qg "${QG_CPU}" \
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
    fi

    if [ "${RUN_TRACE_REPLAY}" = "1" ]; then
      local trace_ok=1
      for f in "${GT}" "${GRAPH}" "${QG_CPU}" "${QG_GPU}"; do
        if [ ! -f "$f" ]; then echo "WARN: missing trace/replay file: $f (skip access trace for ${ds})" >&2; trace_ok=0; break; fi
      done
      [ "${trace_ok}" = "1" ] || { echo "__TRACE_SKIP__=${ds}"; continue; }

      local limit_args=()
      if [ "${TRACE_QUERY_LIMIT}" -gt 0 ]; then
        limit_args=(-query_limit "${TRACE_QUERY_LIMIT}")
      fi

      local cb_access="${OUT}/cpu_base_${ds}_b${TRACE_BEAM}.access.csv"
      local cr_access="${OUT}/cpu_rabitq_${ds}_b${TRACE_BEAM}.access.csv"
      local gb_access="${OUT}/gpu_base_${ds}_b${TRACE_BEAM}.access.csv"
      local gr_access="${OUT}/gpu_rabitq_${ds}_b${TRACE_BEAM}.access.csv"
      local cb_replay="${OUT}/cpu_base_${ds}_b${TRACE_BEAM}.replay.txt"
      local cr_replay="${OUT}/cpu_rabitq_${ds}_b${TRACE_BEAM}.replay.txt"
      local gb_replay="${OUT}/gpu_base_${ds}_b${TRACE_BEAM}.replay.txt"
      local gr_replay="${OUT}/gpu_rabitq_${ds}_b${TRACE_BEAM}.replay.txt"

      echo "=== CPU access trace export + replay ${ds} beam=${TRACE_BEAM} ==="
      echo "__TRACE_CPU_BEGIN__ ${ds}"
      bin/trace_cpu_base -base "${BASE}" -query "${QUERY}" -gt "${GT}" -graph "${GRAPH}" \
        -k "${TOPK}" -beam "${TRACE_BEAM}" -access_trace_file "${cb_access}" "${limit_args[@]}"
      bin/replay_cpu -base "${BASE}" -query "${QUERY}" -graph "${GRAPH}" \
        -trace "${cb_access}" -repeat "${TRACE_CPU_REPEAT}" -warmup "${TRACE_WARMUP}" > "${cb_replay}"
      bin/trace_cpu_rabitq -base "${BASE}" -query "${QUERY}" -gt "${GT}" -degree "${DEGREE}" \
        -mode codebook -qg "${QG_CPU}" -k "${TOPK}" -beam "${TRACE_BEAM}" \
        -access_trace_file "${cr_access}" "${limit_args[@]}"
      bin/replay_cpu -base "${BASE}" -query "${QUERY}" -degree "${DEGREE}" \
        -mode codebook -qg "${QG_CPU}" -trace "${cr_access}" \
        -repeat "${TRACE_CPU_REPEAT}" -warmup "${TRACE_WARMUP}" > "${cr_replay}"
      emit_file "cpu_base_${ds}_b${TRACE_BEAM}_replay" "${cb_replay}"
      emit_file "cpu_rabitq_${ds}_b${TRACE_BEAM}_replay" "${cr_replay}"
      echo "__TRACE_CPU_END__ ${ds}"

      run_gpu_occupancy_precheck
      echo "=== GPU access trace export + replay ${ds} beam=${TRACE_BEAM} ==="
      echo "__TRACE_GPU_BEGIN__ ${ds}"
      run_gpu_cmd "gpu_base_${ds}_access" \
        bin/trace_gpu gpu_base "${BASE}" "${QUERY}" "${GT}" "${GRAPH}" "${TOPK}" "${TRACE_BEAM}" \
          -access_trace_file "${gb_access}" "${limit_args[@]}"
      run_gpu_cmd "gpu_base_${ds}_replay" \
        bin/replay_gpu -base "${BASE}" -query "${QUERY}" -graph "${GRAPH}" \
          -trace "${gb_access}" -degree "${DEGREE}" -repeat "${TRACE_GPU_REPEAT}" -warmup "${TRACE_WARMUP}" > "${gb_replay}"
      run_gpu_occupancy_precheck
      run_gpu_cmd "gpu_rabitq_${ds}_access" \
        bin/trace_gpu gpu_rabitq "${BASE}" "${QUERY}" "${GT}" "${QG_GPU}" "${TOPK}" "${TRACE_BEAM}" "${DEGREE}" \
          -access_trace_file "${gr_access}" "${limit_args[@]}"
      run_gpu_cmd "gpu_rabitq_${ds}_replay" \
        bin/replay_gpu -base "${BASE}" -query "${QUERY}" -qg "${QG_GPU}" \
          -trace "${gr_access}" -degree "${DEGREE}" -repeat "${TRACE_GPU_REPEAT}" -warmup "${TRACE_WARMUP}" > "${gr_replay}"
      emit_file "gpu_base_${ds}_b${TRACE_BEAM}_replay" "${gb_replay}"
      emit_file "gpu_rabitq_${ds}_b${TRACE_BEAM}_replay" "${gr_replay}"
      echo "__TRACE_GPU_END__ ${ds}"
    fi
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
