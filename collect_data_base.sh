#!/usr/bin/env bash
# Base GPU beam search (gpu_base) collection only: build one binary per search width (speculation, 1 = plain Base),
# then run every beam as its own process, TRIALS times, and summarize QPS (median of trials) and recall.
# usage: bash collect_data_base.sh <dataset> <graph file> [widths]
#   <dataset>     prefix of $DATA/<dataset>/<dataset>_{base,query}.fvecs and <dataset>_groundtruth.ivecs
#   <graph file>  fixed-degree graph: uint32 n, uint32 degree, then n x degree uint32 neighbor ids
#   [widths]      search widths, e.g. "1 4" (default "1")
# env: DATA=../.data  TOPK=100  BEAMS="128 256 512 768 1024"  TRIALS=3  NVCC=/usr/bin/nvcc  ARCH=sm_89
#      OUT=result/<date>_base_<dataset>_<graph name>  BASE / QUERY / GT override the dataset files
# CUDA 13.x nvcc fails to compile beam_gpu_base.cu (CCCL / libstdc++ data() ambiguity); use nvcc 12.x.
set -euo pipefail
cd "$(dirname "$0")"

[[ $# -ge 2 ]] || { sed -n 2,10p "$0"; exit 1; }
DS=$1 GRAPH=$2 WIDTHS=${3:-1}
DATA=${DATA:-../.data} TOPK=${TOPK:-100} BEAMS=${BEAMS:-"128 256 512 768 1024"} TRIALS=${TRIALS:-3}
NVCC=${NVCC:-/usr/bin/nvcc} ARCH=${ARCH:-sm_89}
BASE=${BASE:-$DATA/$DS/${DS}_base.fvecs} QUERY=${QUERY:-$DATA/$DS/${DS}_query.fvecs}
GT=${GT:-$DATA/$DS/${DS}_groundtruth.ivecs}
OUT=${OUT:-result/$(date +%Y%m%d)_base_${DS}_$(basename "$GRAPH")}

check() {
    local f
    for f in "$BASE" "$QUERY" "$GT" "$GRAPH"; do
        [[ -f $f ]] || { echo "missing file: $f"; exit 1; }
    done
    read -r n degree < <(od -A n -t u4 -N 8 "$GRAPH")
    echo "dataset $DS: base $BASE, query $QUERY, gt $GT"
    echo "graph $GRAPH: n $n, degree $degree"
    echo "nvcc $NVCC: $("$NVCC" --version | tail -1)"
}

build() {
    local width=$1 maxbeam
    maxbeam=$(printf '%s\n' $BEAMS | sort -n | tail -1)
    mkdir -p bin "$OUT/log"
    echo "build bin/gpu_base_w$width (SEARCH_WIDTH $width, INTERNAL_TOPK $maxbeam)"
    "$NVCC" -std=c++20 -O3 -arch="$ARCH" -I. -DINTERNAL_TOPK="$maxbeam" -DSEARCH_WIDTH="$width" -DGPU_SEARCH_MODE=1 \
        -o "bin/gpu_base_w$width" main.cu beam_gpu_base.cu > "$OUT/log/build_w$width.log" 2>&1 ||
        { echo "build failed, see $OUT/log/build_w$width.log"; exit 1; }
}

run() {
    local width=$1 csv="$OUT/w$1.csv" beam trial log
    rm -f "$csv"
    for trial in $(seq 1 "$TRIALS"); do
        for beam in $BEAMS; do
            log="$OUT/log/w${width}_b${beam}_t$trial.log"
            "bin/gpu_base_w$width" "$BASE" "$QUERY" "$GT" "$GRAPH" "$TOPK" "$beam" -csv "$csv" > "$log" 2>&1 ||
                { echo "run failed, see $log"; exit 1; }
            echo "width $width trial $trial beam $beam: $(tail -1 "$csv" | cut -d, -f10,11 | sed 's/,/ QPS, recall /')"
        done
    done
}

summary() {
    local width
    {
        echo "width beam qps_median recall"
        for width in $WIDTHS; do
            tail -n +2 "$OUT/w$width.csv" | sort -t, -k2,2n -k10,10g | awk -F, -v w="$width" -v t="$TRIALS" '
                { k = ++c[$2]; qps[$2, k] = $10; rec[$2, k] = $11; if (k == 1) order[++m] = $2 }
                END { for (i = 1; i <= m; i++) { b = order[i]; j = int((c[b] + 1) / 2); print w, b, qps[b, j], rec[b, j] } }'
        done
    } | column -t | tee "$OUT/summary.txt"
}

check
for width in $WIDTHS; do
    build "$width"
    run "$width"
done
summary
echo "results in $OUT"
