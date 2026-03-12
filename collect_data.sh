#!/usr/bin/env bash
set -e

# --------------------------------------------
# Parse which binary to run
# --------------------------------------------
RUN_TARGET=${1:-all}   # default to "all" if not specified
echo "Selected target: $RUN_TARGET"

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
    all)
        MAKE_TARGET="all"
        ;;
    *)
        echo "Unknown target: $RUN_TARGET"
        echo "Usage: $0 [cpu_base|cpu_pathw|gpu_base|all]"
        exit 1
        ;;
esac

# --------------------------------------------
# Build only the selected binary
# --------------------------------------------
echo "Cleaning and building $MAKE_TARGET ..."
make clean
make "$MAKE_TARGET"
echo "Build done."

# --------------------------------------------
# Configurations
# --------------------------------------------
THREADS=(16)
TOPK=100
BEAM_SIZES=(100 150 200 250 300 400 500 600 700 800 1000)
# PRUNE_RATIOS=(0 0.125 0.25 0.375 0.5 0.625 0.75 0.875)
PRUNE_RATIOS=(0)
# APPROX_RATIOS=(0 0.1 0.2 0.3 0.4 0.5)
APPROX_RATIOS=(0)
TRIALS=1

BIN_DIR="bin"
RESULT_ROOT="result/3.11"

# --------------------------------------------
# List of dataset prefixes and K values
# --------------------------------------------
# DATASETS=("gist" 32 "gist" 64 "sift" 32 "sift" 64 "sift10m" 32 "deep10m" 32 "deep10m" 64)
# DATASETS=("sift" 32 "sift" 64 "gist" 32 "deep1M" 32 "deep10m" 64 "sift10m" 32)
DATASETS=("sift" 32)

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

    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t

        for beam in "${BEAM_SIZES[@]}"; do
            for pr in "${PRUNE_RATIOS[@]}"; do
                for ar in "${APPROX_RATIOS[@]}"; do

                    CSV_FILE="${LOG_ROOT}/${PREFIX}_deg${K}_top${TOPK}_base.csv"
                    LOG_FILE="${LOG_ROOT_2}/${BIN_NAME}_beam${beam}_threads${t}_pr${pr}_a${ar}_base.log"
                    > "$LOG_FILE"

                    for ((trial=1; trial<=TRIALS; trial++)); do
                        echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"

                        echo "$BINARY" \
                            -base "$BASE_FILE" \
                            -query "$QUERY_FILE" \
                            -gt "$GT_FILE" \
                            -graph "$GRAPH_FILE" \
                            -beam $beam \
                            -p_ratio $pr \
                            -a_ratio $ar \
                            -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                        "$BINARY" \
                            -base "$BASE_FILE" \
                            -query "$QUERY_FILE" \
                            -gt "$GT_FILE" \
                            -graph "$GRAPH_FILE" \
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

    CSV_FILE="${LOG_ROOT}/${PREFIX}_deg${K}_top${TOPK}_rabitq.csv"
    if [ -f "$CSV_FILE" ]; then
        echo "Existing CSV file found. Removing..."
        rm -f "$CSV_FILE"
    fi

    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t

        for beam in "${BEAM_SIZES[@]}"; do
            for pr in "${PRUNE_RATIOS[@]}"; do
                for ar in "${APPROX_RATIOS[@]}"; do

                    LOG_FILE="${LOG_ROOT_2}/${BIN_NAME}_beam${beam}_threads${t}_pr${pr}_a${ar}_rabitq.log"
                    > "$LOG_FILE"

                    for ((trial=1; trial<=TRIALS; trial++)); do
                        echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"

                        echo "$BINARY" \
                            -base "$BASE_FILE" \
                            -query "$QUERY_FILE" \
                            -gt "$GT_FILE" \
                            -graph "$GRAPH_FILE" \
                            -rabitq "$RABITQ_TABLE" \
                            -sign "$RABITQ_SIGN" \
                            -beam $beam \
                            -p_ratio $pr \
                            -a_ratio $ar \
                            -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                        "$BINARY" \
                            -base "$BASE_FILE" \
                            -query "$QUERY_FILE" \
                            -gt "$GT_FILE" \
                            -graph "$GRAPH_FILE" \
                            -rabitq "$RABITQ_TABLE" \
                            -sign "$RABITQ_SIGN" \
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

# --------------------------------------------
# Function to run GPU base
# --------------------------------------------
run_gpu_base() {
    BINARY="$BIN_DIR/gpu_base"
    BIN_NAME=$(basename "$BINARY")
    BEAM=128

    for topk in "${TOPK[@]}"; do

        LOG_FILE="${LOG_DIR}/${BIN_NAME}_beam${BEAM}_topk${topk}.log"
        > "$LOG_FILE"

        for ((trial=1; trial<=TRIALS; trial++)); do
            echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"
            echo "Running $BIN_NAME, dataset=$PREFIX, topK=$topk, beam=$BEAM, trial=$trial" >> "$LOG_FILE"

            "$BINARY" \
                "$BASE_FILE" \
                "$QUERY_FILE" \
                "$GT_FILE" \
                "$GRAPH_FILE" \
                $topk \
                $BEAM >> "$LOG_FILE" 2>&1

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
    GRAPH_FILE="$DATA_PATH/graph-d${K}"
    SIGNBIT_FILE="$DATA_PATH/graph-d${K}.ref.signbit.bin"

    RABITQ_TABLE="${GRAPH_FILE}.Sophia.rabitq.bin"
    RABITQ_SIGN="${GRAPH_FILE}.Sophia.rabitq.sign.bin"

    # create a separate log folder for this dataset/K
    LOG_ROOT="${RESULT_ROOT}/${PREFIX}_K${K}"
    LOG_ROOT_2="${LOG_ROOT}/log"
    mkdir -p "$LOG_ROOT"
    mkdir -p "$LOG_ROOT_2"

    # run target
    case "$RUN_TARGET" in
        cpu_base) run_cpu_base ;;
        cpu_pathw) run_cpu_pathw ;;
        cpu_rabitq) 
            preprocess_rabitq
            run_cpu_rabitq ;;
        gpu_base) run_gpu_base ;;
        all)
            run_cpu_base
            run_cpu_pathw
            run_cpu_rabitq
            run_gpu_base
            ;;
    esac

    i=$((i+2))
done

echo "All runs complete. Logs stored in $LOG_ROOT/"
