#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [target] [debug on|off for CPU targets]"
    echo
    echo "Targets:"
    echo "  cpu_base cpu_pathw cpu_rabitq cpu_rabitq_pruned preprocess_rabitq"
    echo "  gpu_base gpu_pathw gpu_rabitq gpu_rabitq_pruned"
    echo "  gpu_rabitq_pruned_spec gpu_rabitq_pruned_spec_adaptive all"
    echo
    echo "Most knobs can be adjusted at the top of this file or overridden with *_OVERRIDE env vars."
}

# --------------------------------------------
# Example commands
# --------------------------------------------
# AP, one dataset/degree, one beam, reuse existing binary:
#   DATASETS_OVERRIDE=sift,32 TOPK_OVERRIDE=100 BEAM_SIZES_OVERRIDE=128 SKIP_BUILD_OVERRIDE=1 bash collect_data.sh gpu_rabitq_pruned_spec_adaptive
#
# AP full build, one lane per neighbor. This loads the fastscan index:
#   GPU_RABITQ_FASTSCAN_SUBWARP_LANES_OVERRIDE=1 DATASETS_OVERRIDE=sift,32 TOPK_OVERRIDE=100 BEAM_SIZES_OVERRIDE=128 bash collect_data.sh gpu_rabitq_pruned_spec_adaptive
#
# AP full build, multiple lanes per neighbor. This loads the sequential.LUT index:
#   GPU_RABITQ_FASTSCAN_SUBWARP_LANES_OVERRIDE=8 DATASETS_OVERRIDE=sift,32 TOPK_OVERRIDE=100 BEAM_SIZES_OVERRIDE=128 bash collect_data.sh gpu_rabitq_pruned_spec_adaptive
#
# Regular pruned-spec sweep with explicit pruning and iteration ratios:
#   SEARCH_WIDTH_OVERRIDE=16 PRUNE_RATIOS_OVERRIDE=0.875 RABITQ_PRUNED_ITER_RATIOS_OVERRIDE=0.7 bash collect_data.sh gpu_rabitq_pruned_spec

# --------------------------------------------
# Editable run configuration
# --------------------------------------------
# These defaults are intentionally near the top so the script can be used by
# editing this block and running `bash collect_data.sh`.
DEFAULT_RUN_TARGET="all"
DEFAULT_DEBUG_MODE="off"
DEFAULT_DATASETS=("sift" 32)
DEFAULT_TOPK="100"
DEFAULT_DEGREE=""
DEFAULT_GRAPH_DEGREE=""
DEFAULT_BEAM_SIZES=(128 192 256 320 384 448 512 576 640 704 768 832 896 960 1024)
DEFAULT_TRIALS="1"

# Index/artifact defaults. GPU RaBitQ-family artifact mode is derived from
# GPU_RABITQ_FASTSCAN_SUBWARP_LANES below unless GPU_RABITQ_MODE_OVERRIDE is set.
# CPU RaBitQ still defaults to fastscan.
DEFAULT_MODE="sequential.LUT"
DEFAULT_CPU_RABITQ_MODE="fastscan"
DEFAULT_RABITQ_LOAD_MODE="codebook"  # codebook or bins for CPU RaBitQ

# Optional direct artifact overrides. Leave blank to auto-resolve from dataset,
# degree, mode, and graph degree.
BASE_FILE_OVERRIDE="${BASE_FILE_OVERRIDE:-}"
QUERY_FILE_OVERRIDE="${QUERY_FILE_OVERRIDE:-}"
GT_FILE_OVERRIDE="${GT_FILE_OVERRIDE:-}"
GRAPH_FILE_OVERRIDE="${GRAPH_FILE_OVERRIDE:-}"
SIGNBIT_FILE_OVERRIDE="${SIGNBIT_FILE_OVERRIDE:-}"
CPU_QG_CODEBOOK_OVERRIDE="${CPU_QG_CODEBOOK_OVERRIDE:-}"
GPU_QG_CODEBOOK_OVERRIDE="${GPU_QG_CODEBOOK_OVERRIDE:-}"
RABITQ_TABLE_OVERRIDE="${RABITQ_TABLE_OVERRIDE:-}"
RABITQ_SIGN_OVERRIDE="${RABITQ_SIGN_OVERRIDE:-}"

# Compile-time implementation knobs.
SEARCH_WIDTH="${SEARCH_WIDTH_OVERRIDE:-1}"
GPU_RABITQ_FASTSCAN_SUBWARP_LANES="${GPU_RABITQ_FASTSCAN_SUBWARP_LANES_OVERRIDE:-8}"
case "$GPU_RABITQ_FASTSCAN_SUBWARP_LANES" in
    1)
        DEFAULT_GPU_RABITQ_MODE="fastscan"
        ;;
    2|4|8|16|32)
        DEFAULT_GPU_RABITQ_MODE="$DEFAULT_MODE"
        ;;
    *)
        echo "Error: GPU_RABITQ_FASTSCAN_SUBWARP_LANES must be one of 1, 2, 4, 8, 16, or 32"
        exit 1
        ;;
esac
GPU_RABITQ_PRUNED_LOCAL_GATE_MIN_WIDTH="${GPU_RABITQ_PRUNED_LOCAL_GATE_MIN_WIDTH_OVERRIDE:-1}"
GPU_RABITQ_PRUNED_LOCAL_GATE_USE_LUT="${GPU_RABITQ_PRUNED_LOCAL_GATE_USE_LUT_OVERRIDE:-1}"

# --------------------------------------------
# Parse which binary to run
# --------------------------------------------
RUN_TARGET=${1:-${RUN_TARGET_OVERRIDE:-$DEFAULT_RUN_TARGET}}
DEBUG_MODE=${2:-${DEBUG_MODE_OVERRIDE:-$DEFAULT_DEBUG_MODE}}
if [[ $# -ge 3 ]]; then
    DEBUG_MODE=${3:-off}
fi
if [[ "$RUN_TARGET" == "gpu_rabitq_pruned_spec_adaptive" ]]; then
    SEARCH_WIDTH=16
fi
SKIP_BUILD=${SKIP_BUILD_OVERRIDE:-0}
SKIP_GPU_BUILD=${SKIP_GPU_BUILD_OVERRIDE:-0}
echo "Selected target: $RUN_TARGET"
echo "Debug mode: $DEBUG_MODE"
echo "Search width: $SEARCH_WIDTH"
echo "Skip build: $SKIP_BUILD"
echo "Skip GPU build: $SKIP_GPU_BUILD"

case "$DEBUG_MODE" in
    on|off|1|0|true|false)
        ;;
    *)
        echo "Unknown debug mode: $DEBUG_MODE"
        usage
        exit 1
        ;;
esac

# --------------------------------------------
# Map run target to make target
# --------------------------------------------
case "$RUN_TARGET" in
    cpu_base)
        MAKE_TARGET="cpu_base"
        ;;
    cpu_pathw)
        MAKE_TARGET="cpu_pathw"
        ;;
    cpu_rabitq)
        MAKE_TARGET="cpu_rabitq"
        ;;
    cpu_rabitq_pruned)
        MAKE_TARGET="cpu_rabitq_pruned"
        ;;
    preprocess_rabitq)
        MAKE_TARGET="preprocess_rabitq"
        ;;
    gpu_base)
        MAKE_TARGET="gpu_base"
        ;;
    gpu_pathw)
        MAKE_TARGET="gpu_pathw"
        ;;
    gpu_rabitq)
        MAKE_TARGET="gpu_rabitq"
        ;;
    gpu_rabitq_pruned)
        MAKE_TARGET="gpu_rabitq_pruned"
        ;;
    gpu_rabitq_pruned_spec|gpu_rabitq_pruned_spec_adaptive)
        MAKE_TARGET="gpu_rabitq_pruned"
        ;;
    all)
        MAKE_TARGET="all"
        ;;
    *)
        echo "Unknown target: $RUN_TARGET"
        usage
        exit 1
        ;;
esac

# --------------------------------------------
# Build only the selected binary
# --------------------------------------------
if [[ "$SKIP_BUILD" == "1" || "$SKIP_BUILD" == "true" || "$SKIP_BUILD" == "on" ]]; then
    echo "Skipping build step due to SKIP_BUILD=$SKIP_BUILD"
elif [[ "$RUN_TARGET" == "all" ]]; then
    echo "Cleaning and building cpu ..."
    make clean
    make cpu EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS_OVERRIDE:-}" EXTRA_NVFLAGS="${EXTRA_NVFLAGS_OVERRIDE:-}"
    echo "CPU build done."
elif [[ "$RUN_TARGET" != gpu_* ]]; then
    echo "Cleaning and building $MAKE_TARGET ..."
    make clean
    make "$MAKE_TARGET" EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS_OVERRIDE:-}" EXTRA_NVFLAGS="${EXTRA_NVFLAGS_OVERRIDE:-}"
    echo "Build done."
fi

# --------------------------------------------
# Configurations
# --------------------------------------------
if [ -n "${THREADS_OVERRIDE:-}" ]; then
    IFS=',' read -r -a THREADS <<< "$THREADS_OVERRIDE"
else
    THREADS=(16)
fi
TOPK="${TOPK_OVERRIDE:-$DEFAULT_TOPK}"
DEGREE_CONFIG="${DEGREE_OVERRIDE:-$DEFAULT_DEGREE}"
GRAPH_DEGREE="${GRAPH_DEGREE_OVERRIDE:-$DEFAULT_GRAPH_DEGREE}"
# BEAM_SIZES=(100 150 200 250 300 400 500 600 700 800 1000)
if [ -n "${BEAM_SIZES_OVERRIDE:-}" ]; then
    IFS=',' read -r -a BEAM_SIZES <<< "$BEAM_SIZES_OVERRIDE"
else
    BEAM_SIZES=("${DEFAULT_BEAM_SIZES[@]}")
    # BEAM_SIZES=(32 64 96 128)
fi
# PRUNE_RATIOS and APPROX_RATIOS are the generic knobs for base/RaBitQ.
# PathW uses separate knobs below because CPU PathW takes a prune fraction
# while GPU PathW's older -p_ratio takes a keep fraction.
if [ -n "${PRUNE_RATIOS_OVERRIDE:-}" ]; then
    IFS=',' read -r -a PRUNE_RATIOS <<< "$PRUNE_RATIOS_OVERRIDE"
else
    PRUNE_RATIOS=(0)
fi
if [ -n "${APPROX_RATIOS_OVERRIDE:-}" ]; then
    IFS=',' read -r -a APPROX_RATIOS <<< "$APPROX_RATIOS_OVERRIDE"
else
    APPROX_RATIOS=(0)
fi
if [ -n "${PATHW_PRUNE_RATIOS_OVERRIDE:-}" ]; then
    IFS=',' read -r -a PATHW_PRUNE_RATIOS <<< "$PATHW_PRUNE_RATIOS_OVERRIDE"
else
    PATHW_PRUNE_RATIOS=(0.875)
fi
if [ -n "${CPU_PATHW_APPROX_RATIOS_OVERRIDE:-}" ]; then
    IFS=',' read -r -a CPU_PATHW_APPROX_RATIOS <<< "$CPU_PATHW_APPROX_RATIOS_OVERRIDE"
else
    CPU_PATHW_APPROX_RATIOS=(0)
fi
if [ -n "${GPU_PATHW_ITER_RATIOS_OVERRIDE:-}" ]; then
    IFS=',' read -r -a GPU_PATHW_ITER_RATIOS <<< "$GPU_PATHW_ITER_RATIOS_OVERRIDE"
else
    GPU_PATHW_ITER_RATIOS=(0.7)
fi
if [ -n "${RABITQ_PRUNED_PRUNE_RATIOS_OVERRIDE:-}" ]; then
    IFS=',' read -r -a RABITQ_PRUNED_PRUNE_RATIOS <<< "$RABITQ_PRUNED_PRUNE_RATIOS_OVERRIDE"
else
    RABITQ_PRUNED_PRUNE_RATIOS=("${PRUNE_RATIOS[@]}")
fi
if [ -n "${RABITQ_PRUNED_ITER_RATIOS_OVERRIDE:-}" ]; then
    IFS=',' read -r -a RABITQ_PRUNED_ITER_RATIOS <<< "$RABITQ_PRUNED_ITER_RATIOS_OVERRIDE"
else
    RABITQ_PRUNED_ITER_RATIOS=(0.7)
fi
GPU_PATHW_WARMUP_RUNS="${GPU_PATHW_WARMUP_RUNS_OVERRIDE:-0}"
if [ -n "${L2_START_ITERS_OVERRIDE:-}" ]; then
    IFS=',' read -r -a L2_START_ITERS <<< "$L2_START_ITERS_OVERRIDE"
else
    L2_START_ITERS=(0)
fi
L2_END_ITER="${L2_END_ITER_OVERRIDE:-0}"
L2_PERIOD="${L2_PERIOD_OVERRIDE:-0}"
TRIALS=${TRIALS_OVERRIDE:-$DEFAULT_TRIALS}
MODE="${MODE_OVERRIDE:-$DEFAULT_MODE}"               # default GPU/SymQG artifact mode
GRAPH_MODE="${GRAPH_MODE_OVERRIDE:-$MODE}"
CPU_RABITQ_MODE="${CPU_RABITQ_MODE_OVERRIDE:-$DEFAULT_CPU_RABITQ_MODE}"
GPU_RABITQ_MODE="${GPU_RABITQ_MODE_OVERRIDE:-$DEFAULT_GPU_RABITQ_MODE}"
if [[ -z "${GPU_RABITQ_MODE_OVERRIDE:-}" ]]; then
    echo "GPU RaBitQ artifact mode auto-selected from lanes=$GPU_RABITQ_FASTSCAN_SUBWARP_LANES: $GPU_RABITQ_MODE"
else
    echo "GPU RaBitQ artifact mode overridden: $GPU_RABITQ_MODE"
fi
CSV_TAG="${CSV_TAG_OVERRIDE:-$GPU_RABITQ_MODE}"
RABITQ_LOAD_MODE="${RABITQ_LOAD_MODE_OVERRIDE:-$DEFAULT_RABITQ_LOAD_MODE}"  # "bins" = -graph + -rabitq + -sign, "codebook" = -qg
ESTIMATE_STATS="${ESTIMATE_STATS_OVERRIDE:-off}"           # "on" = write estimate reuse stats, "off" = disable
STEP_EFFICIENCY="${STEP_EFFICIENCY_OVERRIDE:-off}"         # "on" = write per-step exact-distance reduction stats
ACC32="${ACC32_OVERRIDE:-}"

BIN_DIR="bin"
RUN_DATE="${RUN_DATE_OVERRIDE:-$(date +%Y%m%d)}"
RESULT_ROOT="${RESULT_ROOT_OVERRIDE:-result/${RUN_DATE}_collect_${RUN_TARGET}}"

# --------------------------------------------
# List of dataset prefixes and K values
# --------------------------------------------
if [ -n "${DATASETS_OVERRIDE:-}" ]; then
    IFS=',' read -r -a DATASETS <<< "$DATASETS_OVERRIDE"
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
    # DATASETS=("sift" 32 "gist" 32 "deep1M" 32 "deep10m" 32 "openai1m" 32 "fanns5000" 32 "msturing1m" 32 "text2image1m" 32)
fi
# DATASETS=("sift10m" 32 "gist" 32 "deep10m" 32 "deep1M" 32)

build_gpu_for_runtime_beams() {
    if [[ "$SKIP_BUILD" == "1" || "$SKIP_BUILD" == "true" || "$SKIP_BUILD" == "on" ]]; then
        echo "Skipping GPU build due to SKIP_BUILD=$SKIP_BUILD"
        return 0
    fi
    if [[ "$SKIP_GPU_BUILD" == "1" || "$SKIP_GPU_BUILD" == "true" || "$SKIP_GPU_BUILD" == "on" ]]; then
        echo "Skipping GPU build due to SKIP_GPU_BUILD=$SKIP_GPU_BUILD"
        return 0
    fi
    local gpu_make_target="$MAKE_TARGET"
    if [[ "$RUN_TARGET" == "all" ]]; then
        gpu_make_target="gpu"
    fi
    local max_beam=0
    local beam
    for beam in "${BEAM_SIZES[@]}"; do
        if [ "$beam" -gt "$max_beam" ]; then
            max_beam="$beam"
        fi
    done
    local extra_nvflags="${EXTRA_NVFLAGS_OVERRIDE:-}"
    local internal_topk="${INTERNAL_TOPK_OVERRIDE:-$max_beam}"
    local generated_nvflags="-DINTERNAL_TOPK=$internal_topk -DSEARCH_WIDTH=$SEARCH_WIDTH -DGPU_RABITQ_FASTSCAN_SUBWARP_LANES=$GPU_RABITQ_FASTSCAN_SUBWARP_LANES -DPATHW_WARMUP_RUNS=$GPU_PATHW_WARMUP_RUNS"
    case "$RUN_TARGET" in
        gpu_rabitq_pruned_spec)
            generated_nvflags+=" -DGPU_RABITQ_PRUNED_SPEC=1 -DGPU_RABITQ_PRUNED_LOCAL_GATE_MIN_WIDTH=$GPU_RABITQ_PRUNED_LOCAL_GATE_MIN_WIDTH -DGPU_RABITQ_PRUNED_LOCAL_GATE_USE_LUT=$GPU_RABITQ_PRUNED_LOCAL_GATE_USE_LUT"
            ;;
        gpu_rabitq_pruned_spec_adaptive)
            generated_nvflags+=" -DGPU_RABITQ_PRUNED_SPEC_ADAPTIVE=1 -DGPU_RABITQ_PRUNED_LOCAL_GATE_MIN_WIDTH=$GPU_RABITQ_PRUNED_LOCAL_GATE_MIN_WIDTH -DGPU_RABITQ_PRUNED_LOCAL_GATE_USE_LUT=$GPU_RABITQ_PRUNED_LOCAL_GATE_USE_LUT"
            ;;
    esac
    if [ -n "$extra_nvflags" ]; then
        extra_nvflags="$extra_nvflags $generated_nvflags"
    else
        extra_nvflags="$generated_nvflags"
    fi
    echo "Building $gpu_make_target with: $generated_nvflags"
    make "$gpu_make_target" EXTRA_NVFLAGS="$extra_nvflags"
    echo "Build done for INTERNAL_TOPK=$internal_topk."
}

pathw_keep_ratio_from_prune() {
    awk -v prune="$1" 'BEGIN {
        keep = 1.0 - prune;
        if (keep < 0.0) keep = 0.0;
        if (keep > 1.0) keep = 1.0;
        printf "%.6g", keep;
    }'
}

# ============================================================
# Function: Preprocess RaBitQ
# ============================================================
preprocess_rabitq() {

    RABITQ_TABLE="${GRAPH_FILE}.Sophia.rabitq.bin"
    RABITQ_SIGN="${GRAPH_FILE}.Sophia.rabitq.sign.bin"

    # Remove existing files if they exist
    if [ -f "$RABITQ_TABLE" ] || [ -f "$RABITQ_SIGN" ]; then
        echo "Existing RaBitQ files found. Removing..."
        rm -f "$RABITQ_TABLE" "$RABITQ_SIGN"
    fi

    echo "Preprocessing RaBitQ table..."
    echo "Running: $BIN_DIR/preprocess_rabitq -base $BASE_FILE -graph $GRAPH_FILE -out $RABITQ_TABLE -sign $RABITQ_SIGN"
    "$BIN_DIR/preprocess_rabitq" \
        -base "$BASE_FILE" \
        -graph "$GRAPH_FILE" \
        -out "$RABITQ_TABLE" \
        -sign "$RABITQ_SIGN"

    echo "RaBitQ preprocessing done."
}

resolve_existing_path() {
    local candidate
    for candidate in "$@"; do
        if [[ -n "$candidate" && -f "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

resolve_qg_codebook_for_mode() {
    local qg_mode="$1"
    local resolved
    resolved=$(resolve_existing_path \
        "$DATA_PATH/symphonyqg${K}.noNan.${qg_mode}.${ACC32}index" \
        "$DATA_PATH/symphonyqg${K}.${qg_mode}.${ACC32}index" \
        "$DATA_PATH/symphonyqg${K}.noNan.${qg_mode}.index" \
        "$DATA_PATH/symphonyqg${K}.${qg_mode}.index" || true)
    if [[ -n "$resolved" ]]; then
        echo "$resolved"
        return 0
    fi
    if [[ "$qg_mode" == "fastscan" ]]; then
        resolve_existing_path \
            "$DATA_PATH/symphonyqg${K}.noNan.fastscan.index" \
            "$DATA_PATH/symphonyqg${K}.fastscan.index" \
            "$DATA_PATH/symphonyqg_${K}.index" || true
    fi
}

ensure_pathw_signbit() {
    if [[ -f "$SIGNBIT_FILE" ]]; then
        return 0
    fi
    echo "Missing PathW signbit file; generating it with prepare_pathw_signbit."
    echo "  output=$SIGNBIT_FILE"
    if [[ ! -x "$BIN_DIR/prepare_pathw_signbit" ]]; then
        echo "Building prepare_pathw_signbit ..."
        make prepare_pathw_signbit EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS_OVERRIDE:-}" EXTRA_NVFLAGS="${EXTRA_NVFLAGS_OVERRIDE:-}"
    fi
    "$BIN_DIR/prepare_pathw_signbit" \
        -base "$BASE_FILE" \
        -graph "$GRAPH_FILE" \
        -out "$SIGNBIT_FILE"
}

require_graph_file() {
    if [[ ! -f "$GRAPH_FILE" ]]; then
        echo "Missing graph file for $RUN_TARGET under $DATA_PATH"
        exit 1
    fi
}

require_cpu_rabitq_files() {
    case "$RABITQ_LOAD_MODE" in
        codebook)
            if [[ ! -f "$CPU_QG_CODEBOOK" ]]; then
                echo "Missing CPU QG codebook for $RUN_TARGET under $DATA_PATH"
                exit 1
            fi
            ;;
        bins)
            if [[ ! -f "$GRAPH_FILE" || ! -f "$RABITQ_TABLE" || ! -f "$RABITQ_SIGN" ]]; then
                echo "Missing CPU RaBitQ bins-mode files for $RUN_TARGET under $DATA_PATH"
                exit 1
            fi
            ;;
        *)
            echo "Error: unsupported RABITQ_LOAD_MODE=$RABITQ_LOAD_MODE"
            echo "Expected one of: codebook, bins"
            exit 1
            ;;
    esac
}

require_gpu_qg_file() {
    if [[ ! -f "$GPU_QG_CODEBOOK" ]]; then
        echo "Missing GPU QG codebook for $RUN_TARGET under $DATA_PATH"
        exit 1
    fi
}

# ============================================================
# Function: CPU BASE (with approx ratios)
# ============================================================
run_cpu_base() {
    BINARY="$BIN_DIR/cpu_base"
    BIN_NAME="cpu_base"
    CSV_FILE="${LOG_ROOT}/${PREFIX}_deg${K}_top${TOPK}_base.csv"
    rm -f "$CSV_FILE"

    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t

        for beam in "${BEAM_SIZES[@]}"; do
            for pr in "${PRUNE_RATIOS[@]}"; do
                for ar in "${APPROX_RATIOS[@]}"; do

                    LOG_FILE="${LOG_ROOT_2}/${BIN_NAME}_beam${beam}_threads${t}_pr${pr}_a${ar}_base.log"
                    > "$LOG_FILE"

                    for ((trial=1; trial<=TRIALS; trial++)); do
                        echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"

                        echo "$BINARY" \
                            -base "$BASE_FILE" \
                            -query "$QUERY_FILE" \
                            -gt "$GT_FILE" \
                            -k "$TOPK" \
                            -graph "$GRAPH_FILE" \
                            -beam $beam \
                            -p_ratio $pr \
                            -a_ratio $ar \
                            -debug "$DEBUG_MODE" \
                            -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                        "$BINARY" \
                            -base "$BASE_FILE" \
                            -query "$QUERY_FILE" \
                            -gt "$GT_FILE" \
                            -k "$TOPK" \
                            -graph "$GRAPH_FILE" \
                            -beam $beam \
                            -p_ratio $pr \
                            -a_ratio $ar \
                            -debug "$DEBUG_MODE" \
                            -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                        echo >> "$LOG_FILE"
                    done
                done
            done
        done
    done
}

# ============================================================
# Function: CPU PATHW (with approx ratios)
# ============================================================
run_cpu_pathw() {
    BINARY="$BIN_DIR/cpu_pathw"
    BIN_NAME="cpu_pathw"
    CSV_FILE="${LOG_ROOT}/${PREFIX}_deg${K}_top${TOPK}_${BIN_NAME}.csv"
    rm -f "$CSV_FILE"

    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t

        for beam in "${BEAM_SIZES[@]}"; do
            for pr in "${PATHW_PRUNE_RATIOS[@]}"; do
                for ar in "${CPU_PATHW_APPROX_RATIOS[@]}"; do

                    LOG_FILE="${LOG_ROOT_2}/${BIN_NAME}_beam${beam}_threads${t}_pr${pr}_a${ar}_signbit.log"
                    > "$LOG_FILE"

                    for ((trial=1; trial<=TRIALS; trial++)); do
                        echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"

                        echo "$BINARY" \
                            -base "$BASE_FILE" \
                            -query "$QUERY_FILE" \
                            -gt "$GT_FILE" \
                            -k "$TOPK" \
                            -graph "$GRAPH_FILE" \
                            -signbit "$SIGNBIT_FILE" \
                            -beam $beam \
                            -p_ratio $pr \
                            -a_ratio $ar \
                            -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                        "$BINARY" \
                            -base "$BASE_FILE" \
                            -query "$QUERY_FILE" \
                            -gt "$GT_FILE" \
                            -k "$TOPK" \
                            -graph "$GRAPH_FILE" \
                            -signbit "$SIGNBIT_FILE" \
                            -beam $beam \
                            -p_ratio $pr \
                            -a_ratio $ar \
                            -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                        echo >> "$LOG_FILE"
                    done
                done
            done
        done
    done
}


# ============================================================
# Function: CPU RABITQ (with approx ratios)
# ============================================================
run_cpu_rabitq() {
    BINARY="$BIN_DIR/cpu_rabitq"
    BIN_NAME="cpu_rabitq"
    LOAD_MODE_SUFFIX="$RABITQ_LOAD_MODE"
    local -a MODE_ARGS

    case "$RABITQ_LOAD_MODE" in
        codebook)
            if [ -z "$CPU_QG_CODEBOOK" ]; then
                echo "Error: RABITQ_LOAD_MODE=codebook requires CPU_QG_CODEBOOK"
                exit 1
            fi
            MODE_ARGS=(
                -mode codebook
                -qg "$CPU_QG_CODEBOOK"
            )
            ;;
        bins)
            if [ -z "$GRAPH_FILE" ] || [ -z "$RABITQ_TABLE" ] || [ -z "$RABITQ_SIGN" ]; then
                echo "Error: RABITQ_LOAD_MODE=bins requires GRAPH_FILE, RABITQ_TABLE, and RABITQ_SIGN"
                exit 1
            fi
            MODE_ARGS=(
                -mode bins
                -graph "$GRAPH_FILE"
                -rabitq "$RABITQ_TABLE"
                -sign "$RABITQ_SIGN"
            )
            ;;
        *)
            echo "Error: unsupported RABITQ_LOAD_MODE=$RABITQ_LOAD_MODE"
            echo "Expected one of: codebook, bins"
            exit 1
            ;;
    esac

    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t
        CSV_FILE="${LOG_ROOT}/${PREFIX}_deg${K}_top${TOPK}_threads${t}_${BIN_NAME}_${LOAD_MODE_SUFFIX}_${CPU_RABITQ_MODE}.csv"
        rm -f "$CSV_FILE"

        for beam in "${BEAM_SIZES[@]}"; do
            for pr in "${PRUNE_RATIOS[@]}"; do
                for ar in "${APPROX_RATIOS[@]}"; do
                    for l2_iter in "${L2_START_ITERS[@]}"; do
                        LOG_FILE="${LOG_ROOT_2}/${BIN_NAME}_${LOAD_MODE_SUFFIX}_beam${beam}_threads${t}_pr${pr}_a${ar}_l2i${l2_iter}_l2e${L2_END_ITER}_l2p${L2_PERIOD}.log"
                        ESTIMATE_STATS_FILE="${LOG_ROOT_2}/${BIN_NAME}_${LOAD_MODE_SUFFIX}_beam${beam}_threads${t}_pr${pr}_a${ar}_l2i${l2_iter}_l2e${L2_END_ITER}_l2p${L2_PERIOD}.estimate_stats.txt"
                        STEP_EFFICIENCY_FILE="${LOG_ROOT_2}/${BIN_NAME}_${LOAD_MODE_SUFFIX}_beam${beam}_threads${t}_pr${pr}_a${ar}_l2i${l2_iter}_l2e${L2_END_ITER}_l2p${L2_PERIOD}.step_efficiency.txt"
                        ESTIMATE_STATS_ARGS=(
                            -estimate_stats "$ESTIMATE_STATS"
                        )
                        if [[ "$ESTIMATE_STATS" == "on" || "$ESTIMATE_STATS" == "1" || "$ESTIMATE_STATS" == "true" ]]; then
                            ESTIMATE_STATS_ARGS+=(
                                -estimate_stats_file "$ESTIMATE_STATS_FILE"
                            )
                        fi
                        STEP_EFFICIENCY_ARGS=(
                            -step_efficiency "$STEP_EFFICIENCY"
                        )
                        if [[ "$STEP_EFFICIENCY" == "on" || "$STEP_EFFICIENCY" == "1" || "$STEP_EFFICIENCY" == "true" ]]; then
                            STEP_EFFICIENCY_ARGS+=(
                                -step_efficiency_file "$STEP_EFFICIENCY_FILE"
                            )
                        fi
                        > "$LOG_FILE"

                        for ((trial=1; trial<=TRIALS; trial++)); do
                            echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"

                            echo "$BINARY" \
                                -base "$BASE_FILE" \
                                -query "$QUERY_FILE" \
                                -gt "$GT_FILE" \
                                -k "$TOPK" \
                                -degree "$DEGREE" \
                                "${MODE_ARGS[@]}" \
                                -beam $beam \
                                -p_ratio $pr \
                                -a_ratio $ar \
                                -l2_iter $l2_iter \
                                -l2_end_iter $L2_END_ITER \
                                -l2_period $L2_PERIOD \
                                -debug "$DEBUG_MODE" \
                                "${ESTIMATE_STATS_ARGS[@]}" \
                                "${STEP_EFFICIENCY_ARGS[@]}" \
                                -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                            "$BINARY" \
                                -base "$BASE_FILE" \
                                -query "$QUERY_FILE" \
                                -gt "$GT_FILE" \
                                -k "$TOPK" \
                                -degree "$DEGREE" \
                                "${MODE_ARGS[@]}" \
                                -beam $beam \
                                -p_ratio $pr \
                                -a_ratio $ar \
                                -l2_iter $l2_iter \
                                -l2_end_iter $L2_END_ITER \
                                -l2_period $L2_PERIOD \
                                -debug "$DEBUG_MODE" \
                                "${ESTIMATE_STATS_ARGS[@]}" \
                                "${STEP_EFFICIENCY_ARGS[@]}" \
                                -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                            echo >> "$LOG_FILE"
                        done
                    done
                done
            done
        done
    done
}

# ============================================================
# Function: CPU RABITQ PRUNED
# ============================================================
run_cpu_rabitq_pruned() {
    BINARY="$BIN_DIR/cpu_rabitq_pruned"
    BIN_NAME="cpu_rabitq_pruned"
    LOAD_MODE_SUFFIX="$RABITQ_LOAD_MODE"
    local -a MODE_ARGS

    case "$RABITQ_LOAD_MODE" in
        codebook)
            if [ -z "$CPU_QG_CODEBOOK" ]; then
                echo "Error: RABITQ_LOAD_MODE=codebook requires CPU_QG_CODEBOOK"
                exit 1
            fi
            MODE_ARGS=(
                -mode codebook
                -qg "$CPU_QG_CODEBOOK"
            )
            ;;
        bins)
            if [ -z "$GRAPH_FILE" ] || [ -z "$RABITQ_TABLE" ] || [ -z "$RABITQ_SIGN" ]; then
                echo "Error: RABITQ_LOAD_MODE=bins requires GRAPH_FILE, RABITQ_TABLE, and RABITQ_SIGN"
                exit 1
            fi
            MODE_ARGS=(
                -mode bins
                -graph "$GRAPH_FILE"
                -rabitq "$RABITQ_TABLE"
                -sign "$RABITQ_SIGN"
            )
            ;;
        *)
            echo "Error: unsupported RABITQ_LOAD_MODE=$RABITQ_LOAD_MODE"
            echo "Expected one of: codebook, bins"
            exit 1
            ;;
    esac

    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t
        CSV_FILE="${LOG_ROOT}/${PREFIX}_deg${K}_top${TOPK}_threads${t}_${BIN_NAME}_${LOAD_MODE_SUFFIX}_${CPU_RABITQ_MODE}.csv"
        rm -f "$CSV_FILE"

        for beam in "${BEAM_SIZES[@]}"; do
            for pr in "${RABITQ_PRUNED_PRUNE_RATIOS[@]}"; do
                for ar in "${RABITQ_PRUNED_ITER_RATIOS[@]}"; do
                    LOG_FILE="${LOG_ROOT_2}/${BIN_NAME}_${LOAD_MODE_SUFFIX}_beam${beam}_threads${t}_pr${pr}_a${ar}.log"
                    > "$LOG_FILE"

                    for ((trial=1; trial<=TRIALS; trial++)); do
                        echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"

                        echo "$BINARY" \
                            -base "$BASE_FILE" \
                            -query "$QUERY_FILE" \
                            -gt "$GT_FILE" \
                            -k "$TOPK" \
                            -degree "$DEGREE" \
                            "${MODE_ARGS[@]}" \
                            -beam $beam \
                            -p_ratio $pr \
                            -a_ratio $ar \
                            -debug "$DEBUG_MODE" \
                            -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                        "$BINARY" \
                            -base "$BASE_FILE" \
                            -query "$QUERY_FILE" \
                            -gt "$GT_FILE" \
                            -k "$TOPK" \
                            -degree "$DEGREE" \
                            "${MODE_ARGS[@]}" \
                            -beam $beam \
                            -p_ratio $pr \
                            -a_ratio $ar \
                            -debug "$DEBUG_MODE" \
                            -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                        echo >> "$LOG_FILE"
                    done
                done
            done
        done
    done
}

# --------------------------------------------
# Function to run GPU base
# --------------------------------------------
run_gpu_base() {
    BINARY="$BIN_DIR/gpu_base"
    BIN_NAME="gpu_base"
    CSV_FILE="${LOG_ROOT}/${PREFIX}_deg${K}_top${TOPK}_${BIN_NAME}.csv"
    rm -f "$CSV_FILE"

    for beam in "${BEAM_SIZES[@]}"; do
        LOG_FILE="${LOG_ROOT_2}/${BIN_NAME}_beam${beam}_topk${TOPK}.log"
        > "$LOG_FILE"

        for ((trial=1; trial<=TRIALS; trial++)); do
            echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"
            echo "Running $BIN_NAME, dataset=$PREFIX, topK=$TOPK, beam=$beam, trial=$trial" >> "$LOG_FILE"

            echo "$BINARY" \
                "$BASE_FILE" \
                "$QUERY_FILE" \
                "$GT_FILE" \
                "$GRAPH_FILE" \
                "$TOPK" \
                "$beam" \
                -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

            "$BINARY" \
                "$BASE_FILE" \
                "$QUERY_FILE" \
                "$GT_FILE" \
                "$GRAPH_FILE" \
                "$TOPK" \
                "$beam" \
                -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

            echo >> "$LOG_FILE"
        done
    done
}

# --------------------------------------------
# Function to run GPU PathW
# --------------------------------------------
run_gpu_pathw() {
    BINARY="$BIN_DIR/gpu_pathw"
    BIN_NAME="gpu_pathw"
    PATHW_INTERNAL_TOPK="${PATHW_INTERNAL_TOPK_OVERRIDE:-128}"
    PATHW_USE_BEAM_LIST_AS_INTERNAL_TOPK="${PATHW_USE_BEAM_LIST_AS_INTERNAL_TOPK_OVERRIDE:-1}"
    CSV_FILE="${LOG_ROOT}/${PREFIX}_deg${DATASET_GRAPH_DEGREE}_top${TOPK}_${BIN_NAME}.csv"
    rm -f "$CSV_FILE"

    local pathw_beams=()
    case "$PATHW_USE_BEAM_LIST_AS_INTERNAL_TOPK" in
        1|true|on|yes)
            pathw_beams=("${BEAM_SIZES[@]}")
            ;;
        *)
            pathw_beams=("$PATHW_INTERNAL_TOPK")
            ;;
    esac

    for pathw_beam in "${pathw_beams[@]}"; do
        for pr in "${PATHW_PRUNE_RATIOS[@]}"; do
            local keep_ratio
            keep_ratio=$(pathw_keep_ratio_from_prune "$pr")
            for ar in "${GPU_PATHW_ITER_RATIOS[@]}"; do
                LOG_FILE="${LOG_ROOT_2}/${BIN_NAME}_beam${pathw_beam}_topk${TOPK}_prune${pr}_keep${keep_ratio}_iter${ar}.log"
                > "$LOG_FILE"

                for ((trial=1; trial<=TRIALS; trial++)); do
                    echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"
                    echo "Running $BIN_NAME, dataset=$PREFIX, topK=$TOPK, beam=$pathw_beam, iteration_limit=beam, prune=$pr, keep=$keep_ratio, iter_ratio=$ar, trial=$trial" >> "$LOG_FILE"

                    echo "$BINARY" \
                        "$BASE_FILE" \
                        "$QUERY_FILE" \
                        "$GT_FILE" \
                        "$GRAPH_FILE" \
                        "$SIGNBIT_FILE" \
                        "$TOPK" \
                        "$pathw_beam" \
                        -p_ratio "$keep_ratio" \
                        -a_ratio "$ar" \
                        -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                    "$BINARY" \
                        "$BASE_FILE" \
                        "$QUERY_FILE" \
                        "$GT_FILE" \
                        "$GRAPH_FILE" \
                        "$SIGNBIT_FILE" \
                        "$TOPK" \
                        "$pathw_beam" \
                        -p_ratio "$keep_ratio" \
                        -a_ratio "$ar" \
                        -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                    echo >> "$LOG_FILE"
                done
            done
        done
    done
}

# --------------------------------------------
# Function to run GPU RabitQ
# --------------------------------------------
run_gpu_rabitq() {
    BINARY="$BIN_DIR/gpu_rabitq"
    BIN_NAME="gpu_rabitq"
    CSV_FILE="${LOG_ROOT}/${PREFIX}_deg${K}_top${TOPK}_w${SEARCH_WIDTH}_${BIN_NAME}_${CSV_TAG}.csv"
    rm -f "$CSV_FILE"

    for beam in "${BEAM_SIZES[@]}"; do
        LOG_FILE="${LOG_ROOT_2}/${BIN_NAME}_beam${beam}_topk${TOPK}_degree${DEGREE}.log"
        > "$LOG_FILE"

        for ((trial=1; trial<=TRIALS; trial++)); do
            echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"
            echo "Running $BIN_NAME, dataset=$PREFIX, topK=$TOPK, beam=$beam, degree=$DEGREE, trial=$trial" >> "$LOG_FILE"

            echo "$BINARY" \
                "$BASE_FILE" \
                "$QUERY_FILE" \
                "$GT_FILE" \
                "$GPU_QG_CODEBOOK" \
                "$TOPK" \
                "$beam" \
                "$DEGREE" \
                -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

            "$BINARY" \
                "$BASE_FILE" \
                "$QUERY_FILE" \
                "$GT_FILE" \
                "$GPU_QG_CODEBOOK" \
                "$TOPK" \
                "$beam" \
                "$DEGREE" \
                -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

            echo >> "$LOG_FILE"
        done
    done
}

# --------------------------------------------
# Function to run GPU RabitQ pruned
# --------------------------------------------
run_gpu_rabitq_pruned() {
    BINARY="$BIN_DIR/gpu_rabitq_pruned"
    BIN_NAME="$RUN_TARGET"
    CSV_FILE="${LOG_ROOT}/${PREFIX}_deg${K}_top${TOPK}_w${SEARCH_WIDTH}_${BIN_NAME}_${CSV_TAG}.csv"
    rm -f "$CSV_FILE"

    for beam in "${BEAM_SIZES[@]}"; do
        for pr in "${RABITQ_PRUNED_PRUNE_RATIOS[@]}"; do
            for ar in "${RABITQ_PRUNED_ITER_RATIOS[@]}"; do
                LOG_FILE="${LOG_ROOT_2}/${BIN_NAME}_beam${beam}_topk${TOPK}_degree${DEGREE}_pr${pr}_a${ar}.log"
                > "$LOG_FILE"

                for ((trial=1; trial<=TRIALS; trial++)); do
                    echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"
                    echo "Running $BIN_NAME, dataset=$PREFIX, topK=$TOPK, beam=$beam, degree=$DEGREE, prune=$pr, iter_ratio=$ar, trial=$trial" >> "$LOG_FILE"

                    echo "$BINARY" \
                        "$BASE_FILE" \
                        "$QUERY_FILE" \
                        "$GT_FILE" \
                        "$GPU_QG_CODEBOOK" \
                        "$TOPK" \
                        "$beam" \
                        "$DEGREE" \
                        -p_ratio "$pr" \
                        -a_ratio "$ar" \
                        -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                    "$BINARY" \
                        "$BASE_FILE" \
                        "$QUERY_FILE" \
                        "$GT_FILE" \
                        "$GPU_QG_CODEBOOK" \
                        "$TOPK" \
                        "$beam" \
                        "$DEGREE" \
                        -p_ratio "$pr" \
                        -a_ratio "$ar" \
                        -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                    echo >> "$LOG_FILE"
                done
            done
        done
    done
}

# --------------------------------------------
# Build GPU binary once for all dataset runs
# --------------------------------------------
case "$RUN_TARGET" in
    gpu_base|gpu_pathw|gpu_rabitq|gpu_rabitq_pruned|gpu_rabitq_pruned_spec|gpu_rabitq_pruned_spec_adaptive|all)
        build_gpu_for_runtime_beams
        ;;
esac

# --------------------------------------------
# Run based on the selected target
# --------------------------------------------
i=0
while [ $i -lt ${#DATASETS[@]} ]; do
    PREFIX="${DATASETS[$i]}"
    DATASET_DEGREE="${DATASETS[$((i+1))]}"
    DEGREE="${DEGREE_CONFIG:-$DATASET_DEGREE}"
    K="$DEGREE"
    echo "Running experiments for dataset $PREFIX with degree=$K"
    DATASET_GRAPH_DEGREE="${GRAPH_DEGREE:-$DEGREE}"

    # set file paths
    DATA_PATH="../.data/$PREFIX"
    BASE_FILE="${BASE_FILE_OVERRIDE:-$DATA_PATH/${PREFIX}_base.fvecs}"
    QUERY_FILE="${QUERY_FILE_OVERRIDE:-$DATA_PATH/${PREFIX}_query.fvecs}"
    GT_FILE="${GT_FILE_OVERRIDE:-$DATA_PATH/${PREFIX}_groundtruth.ivecs}"
    # GRAPH_FILE="$DATA_PATH/graph-d${K}"
    # GRAPH_FILE="$DATA_PATH/symphonyqg_32.index"
    DEFAULT_SIGNBIT_FILE="$DATA_PATH/graph-d${DATASET_GRAPH_DEGREE}.ref.signbit.bin"

    # RABITQ_TABLE="${GRAPH_FILE}.Sophia.rabitq.bin"
    # RABITQ_SIGN="${GRAPH_FILE}.Sophia.rabitq.sign.bin"

    GRAPH_FILE="${GRAPH_FILE_OVERRIDE:-$(resolve_existing_path \
        "$DATA_PATH/graph-d${DATASET_GRAPH_DEGREE}" \
        "$DATA_PATH/graph-d${K}" \
        "$DATA_PATH/SymQG.${PREFIX}.graph.noNan.${GRAPH_MODE}.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.graph.${GRAPH_MODE}.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.graph.noNan.fastscan.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.graph.fastscan.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.graph.noNan.sequential.LUT.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.graph.sequential.LUT.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.graph.noNan.sequential.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.graph.sequential.bin" \
        "$DATA_PATH/symphonyqg${K}.noNan.${GRAPH_MODE}.index.Sophia.graph.bin" \
        "$DATA_PATH/symphonyqg${K}.${GRAPH_MODE}.index.Sophia.graph.bin" \
        "$DATA_PATH/symphonyqg${K}.noNan.fastscan.index.Sophia.graph.bin" \
        "$DATA_PATH/symphonyqg${K}.fastscan.index.Sophia.graph.bin" || true)}"
    RABITQ_TABLE="${RABITQ_TABLE_OVERRIDE:-$(resolve_existing_path \
        "$DATA_PATH/SymQG.${PREFIX}.rabitq.noNan.${GRAPH_MODE}.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.rabitq.${GRAPH_MODE}.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.rabitq.noNan.fastscan.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.rabitq.fastscan.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.rabitq.noNan.sequential.LUT.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.rabitq.sequential.LUT.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.rabitq.noNan.sequential.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.rabitq.sequential.bin" \
        "$DATA_PATH/symphonyqg${K}.noNan.${GRAPH_MODE}.index.Sophia.rabitq.bin" \
        "$DATA_PATH/symphonyqg${K}.${GRAPH_MODE}.index.Sophia.rabitq.bin" \
        "$DATA_PATH/symphonyqg${K}.noNan.fastscan.index.Sophia.rabitq.bin" \
        "$DATA_PATH/symphonyqg${K}.fastscan.index.Sophia.rabitq.bin" || true)}"
    RABITQ_SIGN="${RABITQ_SIGN_OVERRIDE:-$(resolve_existing_path \
        "$DATA_PATH/SymQG.${PREFIX}.signs.noNan.${GRAPH_MODE}.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.signs.${GRAPH_MODE}.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.signs.noNan.fastscan.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.signs.fastscan.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.signs.noNan.sequential.LUT.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.signs.sequential.LUT.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.signs.noNan.sequential.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.signs.sequential.bin" \
        "$DATA_PATH/symphonyqg${K}.noNan.${GRAPH_MODE}.index.Sophia.rabitq.sign.bin" \
        "$DATA_PATH/symphonyqg${K}.${GRAPH_MODE}.index.Sophia.rabitq.sign.bin" \
        "$DATA_PATH/symphonyqg${K}.noNan.fastscan.index.Sophia.rabitq.sign.bin" \
        "$DATA_PATH/symphonyqg${K}.fastscan.index.Sophia.rabitq.sign.bin" || true)}"
    CPU_QG_CODEBOOK="${CPU_QG_CODEBOOK_OVERRIDE:-$(resolve_qg_codebook_for_mode "$CPU_RABITQ_MODE")}"
    GPU_QG_CODEBOOK="${GPU_QG_CODEBOOK_OVERRIDE:-$(resolve_qg_codebook_for_mode "$GPU_RABITQ_MODE")}"
    SIGNBIT_FILE="${SIGNBIT_FILE_OVERRIDE:-$(resolve_existing_path \
        "${GRAPH_FILE}.pathw.signbit.bin" \
        "${GRAPH_FILE}.ref.signbit.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.graph.noNan.${GRAPH_MODE}.signbit.bin" \
        "$DATA_PATH/SymQG.${PREFIX}.graph.${GRAPH_MODE}.signbit.bin" \
        "$DEFAULT_SIGNBIT_FILE" || true)}"
    if [[ -z "$SIGNBIT_FILE" ]]; then
        SIGNBIT_FILE="${GRAPH_FILE}.pathw.signbit.bin"
    fi

    if [[ ! -f "$BASE_FILE" || ! -f "$QUERY_FILE" || ! -f "$GT_FILE" ]]; then
        echo "Missing dataset vec/gt files under $DATA_PATH"
        exit 1
    fi

    case "$RUN_TARGET" in
        cpu_base|cpu_pathw|gpu_base|gpu_pathw)
            require_graph_file
            ;;
        cpu_rabitq|cpu_rabitq_pruned)
            require_cpu_rabitq_files
            ;;
        gpu_rabitq|gpu_rabitq_pruned|gpu_rabitq_pruned_spec|gpu_rabitq_pruned_spec_adaptive)
            require_gpu_qg_file
            ;;
        preprocess_rabitq)
            require_graph_file
            ;;
    esac
    if [[ "$RUN_TARGET" == "all" ]]; then
        require_graph_file
        require_cpu_rabitq_files
        require_gpu_qg_file
    fi
    case "$RUN_TARGET" in
        cpu_pathw|gpu_pathw|all)
            ensure_pathw_signbit
            ;;
    esac

    echo "Resolved dataset files:"
    echo "  BASE_FILE=$BASE_FILE"
    echo "  QUERY_FILE=$QUERY_FILE"
    echo "  GT_FILE=$GT_FILE"
    echo "  GRAPH_FILE=$GRAPH_FILE"
    echo "  SIGNBIT_FILE=$SIGNBIT_FILE"
    echo "  RABITQ_TABLE=$RABITQ_TABLE"
    echo "  RABITQ_SIGN=$RABITQ_SIGN"
    echo "  CPU_QG_CODEBOOK=$CPU_QG_CODEBOOK"
    echo "  GPU_QG_CODEBOOK=$GPU_QG_CODEBOOK"

    # create a separate log folder for this dataset/K
    LOG_ROOT="${RESULT_ROOT}/${PREFIX}_K${K}"
    LOG_ROOT_2="${LOG_ROOT}/log"
    mkdir -p "$LOG_ROOT"
    mkdir -p "$LOG_ROOT_2"

    # run target
    case "$RUN_TARGET" in
        cpu_base) run_cpu_base ;;
        cpu_pathw) run_cpu_pathw ;;
        preprocess_rabitq) preprocess_rabitq ;;
        cpu_rabitq) run_cpu_rabitq ;;
        cpu_rabitq_pruned) run_cpu_rabitq_pruned ;;
        gpu_base) run_gpu_base ;;
        gpu_pathw) run_gpu_pathw ;;
        gpu_rabitq) run_gpu_rabitq ;;
        gpu_rabitq_pruned|gpu_rabitq_pruned_spec|gpu_rabitq_pruned_spec_adaptive) run_gpu_rabitq_pruned ;;
        all)
            run_cpu_base
            run_cpu_pathw
            run_cpu_rabitq
            run_cpu_rabitq_pruned
            run_gpu_base
            run_gpu_pathw
            run_gpu_rabitq
            run_gpu_rabitq_pruned
            ;;
    esac

    i=$((i+2))
done

echo "All runs complete. Logs stored in $LOG_ROOT/"
