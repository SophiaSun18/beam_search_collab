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
TOPK=(100)                    
# BEAM_SIZES=(32 64 96 128 160 192 224 256)  
BEAM_SIZES=(128 192 256 320 384 448 512)                 
PRUNE_RATIOS=(0.125 0.25 0.375 0.5 0.625 0.75)    
# PRUNE_RATIOS=(0.125)       
WARM_RATIOS=(0)
# WARM_RATIOS=(0.1 0.2 0.3 0.4 0.5)
COOL_RATIOS=(0)
# COOL_RATIOS=(0.3 0.4 0.5 0.6 0.7)
MAX_ITERS=(0)
# MAX_ITERS=(64 128 192)    
TRIALS=3                    

LOG_ROOT="result/gpu_base_pathW_graph_beam_sizes"
mkdir -p "$LOG_ROOT"

BIN_DIR="bin"

# --------------------------------------------
# List of dataset prefixes and K values
# --------------------------------------------
# DATASETS=("gist" 32 "gist" 64 "sift" 32 "sift" 64 "sift10m" 32 "deep10m" 32 "deep10m" 64)
# DATASETS=("sift" 32 "sift" 64 "gist" 32 "deep1M" 32 "deep10m" 64 "sift10m" 32)
DATASETS=("sift" 64)

# --------------------------------------------
# Function to run CPU base
# --------------------------------------------
run_cpu_base() {
    BINARY="$BIN_DIR/cpu_base"
    BIN_NAME=$(basename "$BINARY")

    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t

        for topk in "${TOPK[@]}"; do
            for beam in "${BEAM_SIZES[@]}"; do

                # Optional safety check (recommended)
                if [ "$beam" -le "$topk" ]; then
                    echo "Skipping beam=$beam for topK=$topk (beam must be > topK)"
                    continue
                fi

                LOG_FILE="${LOG_DIR}/${BIN_NAME}_beam${beam}_topk${topk}_threads${t}.log"
                > "$LOG_FILE"

                CSV_FILE="${LOG_DIR}/summary.csv"

                for ((trial=1; trial<=TRIALS; trial++)); do
                    echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"
                    echo "Running $BIN_NAME, dataset=$PREFIX, topK=$topk, beam=$beam, threads=$t, trial=$trial" >> "$LOG_FILE"

                    "$BINARY" \
                        -base "$BASE_FILE" \
                        -query "$QUERY_FILE" \
                        -gt "$GT_FILE" \
                        -graph "$GRAPH_FILE" \
                        -topK $topk \
                        -beam $beam \
                        -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                    echo >> "$LOG_FILE"
                done
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
            for topk in "${TOPK[@]}"; do
                # Skip invalid combinations
                if [ "$beam" -le "$topk" ]; then
                    echo "Skipping beam=$beam for topK=$topk (beam must be > topK)"
                    continue
                fi

                for pr in "${PRUNE_RATIOS[@]}"; do
                    for warm in "${WARM_RATIOS[@]}"; do
                        for cool in "${COOL_RATIOS[@]}"; do
                            for max_iter in "${MAX_ITERS[@]}"; do

                                LOG_FILE="${LOG_DIR}/${BIN_NAME}_beam${beam}_topk${topk}_threads${t}_pr${pr}_warm${warm}_cool${cool}_iter${max_iter}.log"
                                > "$LOG_FILE"

                                CSV_FILE="${LOG_DIR}/summary.csv"

                                for ((trial=1; trial<=TRIALS; trial++)); do
                                    echo "=== Trial $trial/$TRIALS ===" >> "$LOG_FILE"
                                    echo "Running $BIN_NAME, dataset=$PREFIX, topK=$topk, beam=$beam, threads=$t, prune_ratio=$pr, warm_ratio=$warm, cool_ratio=$cool, max_iter=$max_iter, trial=$trial" >> "$LOG_FILE"

                                    "$BINARY" \
                                        -base "$BASE_FILE" \
                                        -query "$QUERY_FILE" \
                                        -gt "$GT_FILE" \
                                        -graph "$GRAPH_FILE" \
                                        -signbit "$SIGNBIT_FILE" \
                                        -topK $topk \
                                        -beam $beam \
                                        -p_ratio $pr \
                                        -warm_ratio $warm \
                                        -cool_ratio $cool \
                                        -max_iter $max_iter \
                                        -csv "$CSV_FILE" >> "$LOG_FILE" 2>&1

                                    echo >> "$LOG_FILE"
                                done

                            done
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
    GRAPH_FILE="$DATA_PATH/graph-d${K}-pathW.graph"
    SIGNBIT_FILE="$DATA_PATH/graph-d${K}.ref.signbit.bin"

    # create a separate log folder for this dataset/K
    LOG_DIR="${LOG_ROOT}/${PREFIX}_K${K}"
    mkdir -p "$LOG_DIR"

    # run target
    case "$RUN_TARGET" in
        cpu_base) run_cpu_base ;;
        cpu_pathw) run_cpu_pathw ;;
        gpu_base) run_gpu_base ;;
        all)
            run_cpu_base
            run_cpu_pathw
            run_gpu_base
            ;;
    esac

    i=$((i+2))
done

echo "All runs complete. Logs stored in $LOG_ROOT/"
