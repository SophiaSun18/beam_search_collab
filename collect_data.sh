#!/usr/bin/env bash
set -e

# --------------------------------------------
# Parse which binary to run
# --------------------------------------------
RUN_TARGET=${1:-all}   # default to "all" if not specified
TIMER_MODE=${2:-off}
DEBUG_MODE=${3:-off}
echo "Selected target: $RUN_TARGET"
echo "Timer mode: $TIMER_MODE"
echo "Debug mode: $DEBUG_MODE"

case "$TIMER_MODE" in
    on|off|1|0|true|false)
        ;;
    *)
        echo "Unknown timer mode: $TIMER_MODE"
        echo "Usage: $0 [cpu_base|cpu_pathw|cpu_rabitq|preprocess_rabitq|gpu_base|gpu_rabitq|all] [timer on|off] [debug on|off]"
        exit 1
        ;;
esac

case "$DEBUG_MODE" in
    on|off|1|0|true|false)
        ;;
    *)
        echo "Unknown debug mode: $DEBUG_MODE"
        echo "Usage: $0 [cpu_base|cpu_pathw|cpu_rabitq|preprocess_rabitq|gpu_base|gpu_rabitq|all] [timer on|off] [debug on|off]"
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
    preprocess_rabitq)
        MAKE_TARGET="preprocess_rabitq"
        ;;
    gpu_base)
        MAKE_TARGET="gpu"
        ;;
    gpu_rabitq)
        MAKE_TARGET="gpu"
        ;;
    all)
        MAKE_TARGET="all"
        ;;
    *)
        echo "Unknown target: $RUN_TARGET"
        echo "Usage: $0 [cpu_base|cpu_pathw|cpu_rabitq|preprocess_rabitq|gpu_base|gpu_rabitq|all] [timer on|off] [debug on|off]"
        exit 1
        ;;
esac

# --------------------------------------------
# Build only the selected binary
# --------------------------------------------
if [[ "$RUN_TARGET" == "all" ]]; then
    echo "Cleaning and building cpu ..."
    make clean
    make cpu EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS_OVERRIDE:-}" EXTRA_NVFLAGS="${EXTRA_NVFLAGS_OVERRIDE:-}"
    echo "CPU build done."
elif [[ "$RUN_TARGET" != "gpu_base" && "$RUN_TARGET" != "gpu_rabitq" ]]; then
    echo "Cleaning and building $MAKE_TARGET ..."
    make clean
    make "$MAKE_TARGET" EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS_OVERRIDE:-}" EXTRA_NVFLAGS="${EXTRA_NVFLAGS_OVERRIDE:-}"
    echo "Build done."
fi

# --------------------------------------------
# Configurations
# --------------------------------------------
THREADS=(16)
TOPK="${TOPK_OVERRIDE:-100}"
DEGREE="${DEGREE_OVERRIDE:-32}"
# BEAM_SIZES=(100 150 200 250 300 400 500 600 700 800 1000)
if [ -n "${BEAM_SIZES_OVERRIDE:-}" ]; then
    IFS=',' read -r -a BEAM_SIZES <<< "$BEAM_SIZES_OVERRIDE"
else
    BEAM_SIZES=(128 256 384 512 640 768 896 1024)
fi
# PRUNE_RATIOS=(0 0.125 0.25 0.375 0.5 0.625 0.75 0.875)
PRUNE_RATIOS=(0)
# APPROX_RATIOS=(0.1 0.2 0.3 0.4 0.5)
APPROX_RATIOS=(0)
# L2_START_ITERS=(0 10 20 30 40 50 60 70 80 90 100)
L2_START_ITERS=0
L2_END_ITER=0
L2_PERIOD=0
RERANK_TOP_M=1
# TRIALS=${TRIALS_OVERRIDE:-1}
TRIALS=1
MODE="${MODE_OVERRIDE:-fastscan}"               # "fastscan" = enabled, "sequential" = fastscan disabled
RABITQ_LOAD_MODE="codebook"  # "bins" = -graph + -rabitq + -sign, "codebook" = -qg
ESTIMATE_STATS="off"          # "on" = write estimate reuse stats, "off" = disable
STEP_EFFICIENCY="off"         # "on" = write per-step exact-distance reduction stats
ACC32=""

BIN_DIR="bin"
RESULT_ROOT="${RESULT_ROOT_OVERRIDE:-result/4.22GpuRabitqTESTuseSeqCodebook}"
# RESULT_ROOT="${RESULT_ROOT_OVERRIDE:-result/4.16GPUTensorCore_2}"

# --------------------------------------------
# List of dataset prefixes and K values
# --------------------------------------------
DATASETS=("sift" 32)
# DATASETS=("sift" 32 "gist" 32 "deep10m" 32 "deep1M" 32)
# DATASETS=("sift10m" 32 "gist" 32 "deep10m" 32 "deep1M" 32)

build_gpu_for_beam() {
    local beam="$1"
    local extra_nvflags="${EXTRA_NVFLAGS_OVERRIDE:-}"
    if [ -n "$extra_nvflags" ]; then
        extra_nvflags="$extra_nvflags -DINTERNAL_TOPK=$beam"
    else
        extra_nvflags="-DINTERNAL_TOPK=$beam"
    fi
    echo "Cleaning and building gpu with INTERNAL_TOPK=$beam ..."
    make clean
    make gpu EXTRA_NVFLAGS="$extra_nvflags"
    echo "Build done for INTERNAL_TOPK=$beam."
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
                            -timer "$TIMER_MODE" \
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
                            -timer "$TIMER_MODE" \
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

    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t

        for beam in "${BEAM_SIZES[@]}"; do
            for pr in "${PRUNE_RATIOS[@]}"; do
                for ar in "${APPROX_RATIOS[@]}"; do

                    CSV_FILE="${LOG_ROOT}/${PREFIX}_deg${K}_top${TOPK}_signbit.csv"
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

    case "$RABITQ_LOAD_MODE" in
        codebook)
            if [ -z "$QG_CODEBOOK" ]; then
                echo "Error: RABITQ_LOAD_MODE=codebook requires QG_CODEBOOK"
                exit 1
            fi
            MODE_ARGS=(
                -mode codebook
                -qg "$QG_CODEBOOK"
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

        for beam in "${BEAM_SIZES[@]}"; do
            for pr in "${PRUNE_RATIOS[@]}"; do
                for ar in "${APPROX_RATIOS[@]}"; do
                    for l2_iter in "${L2_START_ITERS[@]}"; do
                        CSV_FILE="${LOG_ROOT}/${PREFIX}_deg${K}_top${TOPK}_threads${t}_rabitq_${LOAD_MODE_SUFFIX}_${MODE}.csv"
                        if [[ $trial -eq 1 && "$beam" == "${BEAM_SIZES[0]}" && "$pr" == "${PRUNE_RATIOS[0]}" && "$ar" == "${APPROX_RATIOS[0]}" && "$l2_iter" == "${L2_START_ITERS[0]}" ]]; then
                            rm -f "$CSV_FILE"
                        fi
                        LOG_FILE="${LOG_ROOT_2}/${BIN_NAME}_${LOAD_MODE_SUFFIX}_beam${beam}_threads${t}_pr${pr}_a${ar}_l2i${l2_iter}_l2e${L2_END_ITER}_l2p${L2_PERIOD}_rtm${RERANK_TOP_M}.log"
                        ESTIMATE_STATS_FILE="${LOG_ROOT_2}/${BIN_NAME}_${LOAD_MODE_SUFFIX}_beam${beam}_threads${t}_pr${pr}_a${ar}_l2i${l2_iter}_l2e${L2_END_ITER}_l2p${L2_PERIOD}_rtm${RERANK_TOP_M}.estimate_stats.txt"
                        STEP_EFFICIENCY_FILE="${LOG_ROOT_2}/${BIN_NAME}_${LOAD_MODE_SUFFIX}_beam${beam}_threads${t}_pr${pr}_a${ar}_l2i${l2_iter}_l2e${L2_END_ITER}_l2p${L2_PERIOD}_rtm${RERANK_TOP_M}.step_efficiency.txt"
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
                                -rerank_top_m $RERANK_TOP_M \
                                -timer "$TIMER_MODE" \
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
                                -rerank_top_m $RERANK_TOP_M \
                                -timer "$TIMER_MODE" \
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

# --------------------------------------------
# Function to run GPU base
# --------------------------------------------
run_gpu_base() {
    BINARY="$BIN_DIR/gpu_search"
    BIN_NAME="gpu_base"
    CSV_FILE="${LOG_ROOT}/${PREFIX}_deg${K}_top${TOPK}_${BIN_NAME}.csv"
    rm -f "$CSV_FILE"

    for beam in "${BEAM_SIZES[@]}"; do
        build_gpu_for_beam "$beam"
        LOG_FILE="${LOG_ROOT_2}/${BIN_NAME}_beam${beam}_topk${TOPK}.log"
        > "$LOG_FILE"

        for ((trial=1; trial<=TRIALS; trial++)); do
            echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"
            echo "Running $BIN_NAME, dataset=$PREFIX, topK=$TOPK, beam=$beam, trial=$trial" >> "$LOG_FILE"

            echo "$BINARY" \
                gpu_base \
                "$BASE_FILE" \
                "$QUERY_FILE" \
                "$GT_FILE" \
                "$GRAPH_FILE" \
                "$TOPK" \
                "$beam" \
                -debug "$DEBUG_MODE" \
                -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

            "$BINARY" \
                gpu_base \
                "$BASE_FILE" \
                "$QUERY_FILE" \
                "$GT_FILE" \
                "$GRAPH_FILE" \
                "$TOPK" \
                "$beam" \
                -debug "$DEBUG_MODE" \
                -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

            echo >> "$LOG_FILE"
        done
    done
}

# --------------------------------------------
# Function to run GPU RabitQ
# --------------------------------------------
run_gpu_rabitq() {
    BINARY="$BIN_DIR/gpu_search"
    BIN_NAME="gpu_rabitq"
    CSV_FILE="${LOG_ROOT}/${PREFIX}_deg${K}_top${TOPK}_${BIN_NAME}_${MODE}.csv"
    rm -f "$CSV_FILE"

    for beam in "${BEAM_SIZES[@]}"; do
        build_gpu_for_beam "$beam"
        LOG_FILE="${LOG_ROOT_2}/${BIN_NAME}_beam${beam}_topk${TOPK}_degree${DEGREE}.log"
        > "$LOG_FILE"

        for ((trial=1; trial<=TRIALS; trial++)); do
            echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"
            echo "Running $BIN_NAME, dataset=$PREFIX, topK=$TOPK, beam=$beam, degree=$DEGREE, trial=$trial" >> "$LOG_FILE"

            echo "$BINARY" \
                gpu_rabitq \
                "$BASE_FILE" \
                "$QUERY_FILE" \
                "$GT_FILE" \
                "$QG_CODEBOOK" \
                "$TOPK" \
                "$beam" \
                "$DEGREE" \
                -debug "$DEBUG_MODE" \
                -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

            "$BINARY" \
                gpu_rabitq \
                "$BASE_FILE" \
                "$QUERY_FILE" \
                "$GT_FILE" \
                "$QG_CODEBOOK" \
                "$TOPK" \
                "$beam" \
                "$DEGREE" \
                -debug "$DEBUG_MODE" \
                -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

            echo >> "$LOG_FILE"
        done
    done
}

# --------------------------------------------
# Run based on the selected target
# --------------------------------------------
i=0
while [ $i -lt ${#DATASETS[@]} ]; do
    PREFIX="${DATASETS[$i]}"
    K="${DATASETS[$((i+1))]}"
    echo "Running experiments for dataset $PREFIX with K=$K"

    # set file paths
    DATA_PATH="../.data/$PREFIX"
    BASE_FILE="$DATA_PATH/${PREFIX}_base.fvecs"
    QUERY_FILE="$DATA_PATH/${PREFIX}_query.fvecs"
    GT_FILE="$DATA_PATH/${PREFIX}_groundtruth.ivecs"
    # GRAPH_FILE="$DATA_PATH/graph-d${K}"
    # GRAPH_FILE="$DATA_PATH/symphonyqg_32.index"
    SIGNBIT_FILE="$DATA_PATH/graph-d${K}.ref.signbit.bin"

    # RABITQ_TABLE="${GRAPH_FILE}.Sophia.rabitq.bin"
    # RABITQ_SIGN="${GRAPH_FILE}.Sophia.rabitq.sign.bin"

    GRAPH_FILE="$DATA_PATH/SymQG.${PREFIX}.graph.noNan.${MODE}.bin"
    RABITQ_TABLE="$DATA_PATH/SymQG.${PREFIX}.rabitq.noNan.${MODE}.bin"
    RABITQ_SIGN="$DATA_PATH/SymQG.${PREFIX}.signs.noNan.${MODE}.bin"
    QG_CODEBOOK="$DATA_PATH/symphonyqg${K}.noNan.${MODE}.${ACC32}index"

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
        gpu_base) run_gpu_base ;;
        gpu_rabitq) run_gpu_rabitq ;;
        all)
            run_cpu_base
            run_cpu_pathw
            run_cpu_rabitq
            run_gpu_base
            run_gpu_rabitq
            ;;
    esac

    i=$((i+2))
done

echo "All runs complete. Logs stored in $LOG_ROOT/"
