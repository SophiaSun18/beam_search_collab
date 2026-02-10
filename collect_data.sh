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
BEAM_SIZES=(128)                   
PRUNE_RATIOS=(0.125)               
TRIALS=3
DATASET=1                           

LOG_ROOT="result/all_test"
mkdir -p "$LOG_ROOT"

# --------------------------------------------
# Set up file paths
# --------------------------------------------
PREFIX="sift"
DATA_PATH="../data/$PREFIX"
BASE_FILE="$DATA_PATH/${PREFIX}_base.fvecs"
QUERY_FILE="$DATA_PATH/${PREFIX}_query.fvecs"
GT_FILE="$DATA_PATH/${PREFIX}_groundtruth.ivecs"
GRAPH_FILE="$DATA_PATH/graph-d32"
SIGNBIT_FILE="$DATA_PATH/graph-d32.ref.signbit.bin"

BIN_DIR="bin"

# --------------------------------------------
# Function to run CPU base
# --------------------------------------------
run_cpu_base() {
    BINARY="$BIN_DIR/cpu_base"
    BIN_NAME=$(basename "$BINARY")
    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t
        for beam in "${BEAM_SIZES[@]}"; do
            LOG_FILE="${LOG_ROOT}/${BIN_NAME}_beam${beam}_threads${t}.log"
            > "$LOG_FILE"
            for ((trial=1; trial<=TRIALS; trial++)); do
                echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"
                echo "Running $BIN_NAME, beam=$beam, threads=$t, trial=$trial" >> "$LOG_FILE"
                "$BINARY" \
                    -base "$BASE_FILE" \
                    -query "$QUERY_FILE" \
                    -gt "$GT_FILE" \
                    -graph "$GRAPH_FILE" \
                    -beam $beam >> "$LOG_FILE" 2>&1
                echo >> "$LOG_FILE"
            done
        done
    done
}

# --------------------------------------------
# Function to run CPU pathw
# --------------------------------------------
run_cpu_pathw() {
    BINARY="$BIN_DIR/cpu_pathw"
    BIN_NAME=$(basename "$BINARY")
    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t
        for beam in "${BEAM_SIZES[@]}"; do
            for pr in "${PRUNE_RATIOS[@]}"; do
                LOG_FILE="${LOG_ROOT}/${BIN_NAME}_beam${beam}_threads${t}_pr${pr}.log"
                > "$LOG_FILE"
                for ((trial=1; trial<=TRIALS; trial++)); do
                    echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"
                    echo "Running $BIN_NAME, beam=$beam, threads=$t, prune_ratio=$pr, trial=$trial" >> "$LOG_FILE"
                    "$BINARY" \
                        -base "$BASE_FILE" \
                        -query "$QUERY_FILE" \
                        -gt "$GT_FILE" \
                        -graph "$GRAPH_FILE" \
                        -signbit "$SIGNBIT_FILE" \
                        -beam $beam \
                        -p_ratio $pr >> "$LOG_FILE" 2>&1
                    echo >> "$LOG_FILE"
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
    K=100
    BEAM=128
    LOG_FILE="${LOG_ROOT}/${BIN_NAME}_beam${BEAM}.log"
    > "$LOG_FILE"
    for ((trial=1; trial<=TRIALS; trial++)); do
        echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"
        echo "Running $BIN_NAME, beam=$BEAM, trial=$trial" >> "$LOG_FILE"
        "$BINARY" \
            "$BASE_FILE" \
            "$QUERY_FILE" \
            "$GT_FILE" \
            "$GRAPH_FILE" \
            $K \
            $BEAM >> "$LOG_FILE" 2>&1
        echo >> "$LOG_FILE"
    done
}

# --------------------------------------------
# Run based on the selected target
# --------------------------------------------
case "$RUN_TARGET" in
    cpu_base)
        run_cpu_base
        ;;
    cpu_pathw)
        run_cpu_pathw
        ;;
    gpu_base)
        run_gpu_base
        ;;
    all)
        run_cpu_base
        run_cpu_pathw
        run_gpu_base
        ;;
esac

echo "All runs complete. Logs stored in $LOG_ROOT/"
