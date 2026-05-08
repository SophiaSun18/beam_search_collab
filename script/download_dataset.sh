#!/usr/bin/env bash
set -euo pipefail

USAGE="Usage: $0 <dataset-name> [options]

Download an ANN-benchmarks dataset and build graph + codebook.

Arguments:
  dataset-name    Name on ann-benchmarks.com, e.g. sift-128-euclidean,
                  glove-100-angular, fashion-mnist-784-euclidean, etc.

Options:
  --degree <int>       Graph out-degree (default: 32)
  --beam-width <int>   Graph build beam width (default: 128)
  --alpha <float>      Vamana pruning alpha (default: 2.0)
  --seed <int>         Random seed (default: 1234)
  --replicate <int>    Tile queries/gt N times (default: 10)
  --skip-graph         Skip graph building
  --skip-codebook      Skip codebook building
  --help               Show this help

Example:
  $0 sift-128-euclidean
  $0 fashion-mnist-784-euclidean --degree 64
"

# ── Locate project root (parent of script/) ──────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEGREE=32
BEAM_WIDTH=128
ALPHA=2.0
SEED=1234
REPLICATE=10
SKIP_GRAPH=false
SKIP_CODEBOOK=false

# ── Parse arguments ───────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "$USAGE" >&2
    exit 1
fi

DATASET="$1"; shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --degree)      DEGREE="$2";     shift 2 ;;
        --beam-width)  BEAM_WIDTH="$2"; shift 2 ;;
        --alpha)       ALPHA="$2";      shift 2 ;;
        --seed)        SEED="$2";       shift 2 ;;
        --replicate)   REPLICATE="$2"; shift 2 ;;
        --skip-graph)  SKIP_GRAPH=true; shift ;;
        --skip-codebook) SKIP_CODEBOOK=true; shift ;;
        --help)        echo "$USAGE"; exit 0 ;;
        *)             echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Derive a short name (strip metric suffix) for the output directory ────────
# e.g. "sift-128-euclidean" -> "sift", "fashion-mnist-784-euclidean" -> "fashion-mnist"
SHORT_NAME="$(echo "$DATASET" | sed -E 's/-[0-9]+-(euclidean|angular|hamming|jaccard)$//')"

DATA_DIR="$ROOT/data/$SHORT_NAME"
HDF5_FILE="$ROOT/data/$DATASET.hdf5"

mkdir -p "$DATA_DIR"

# ── Step 1: Download HDF5 ────────────────────────────────────────────────────
if [[ -f "$HDF5_FILE" ]]; then
    echo "HDF5 already exists: $HDF5_FILE"
else
    URL="https://ann-benchmarks.com/$DATASET.hdf5"
    echo "Downloading $URL ..."
    curl -fSL --progress-bar -o "$HDF5_FILE" "$URL"
    echo "Saved to $HDF5_FILE"
fi

# ── Step 2: Unpack to fvecs/ivecs ────────────────────────────────────────────
UNPACK_ARGS=("$HDF5_FILE" "$DATA_DIR")
if [[ "$DATASET" == *-angular || "$DATASET" == *-dot ]]; then
    UNPACK_ARGS+=(--normalize)
fi

if [[ -f "$DATA_DIR/base.fvecs" && -f "$DATA_DIR/query.fvecs" && -f "$DATA_DIR/groundtruth.ivecs" ]]; then
    echo "Vectors already unpacked in $DATA_DIR"
else
    echo "Unpacking HDF5 -> $DATA_DIR ..."
    uv run "$ROOT/script/unpack.py" "${UNPACK_ARGS[@]}"
fi

# ── Step 3: Build graph ──────────────────────────────────────────────────────
GRAPH_FILE="$DATA_DIR/graph.bin"
if [[ "$SKIP_GRAPH" == true ]]; then
    echo "Skipping graph build (--skip-graph)"
elif [[ -f "$GRAPH_FILE" ]]; then
    echo "Graph already exists: $GRAPH_FILE"
else
    echo "Building graph (degree=$DEGREE, beam_width=$BEAM_WIDTH, alpha=$ALPHA) ..."
    "$ROOT/build/build_graph" \
        --base "$DATA_DIR/base.fvecs" \
        --graph "$GRAPH_FILE" \
        --degree "$DEGREE" \
        --beam-width "$BEAM_WIDTH" \
        --alpha "$ALPHA" \
        --seed "$SEED"
    echo "Graph saved to $GRAPH_FILE"
fi

# ── Step 4: Build RAbitQ codebook ─────────────────────────────────────────────
CODEBOOK_FILE="$DATA_DIR/codebook.bin"
if [[ "$SKIP_CODEBOOK" == true ]]; then
    echo "Skipping codebook build (--skip-codebook)"
elif [[ -f "$CODEBOOK_FILE" ]]; then
    echo "Codebook already exists: $CODEBOOK_FILE"
elif [[ ! -f "$GRAPH_FILE" ]]; then
    echo "Skipping codebook build (no graph.bin found)"
else
    echo "Building RAbitQ codebook ..."
    "$ROOT/build/build_rabitq" \
        -base "$DATA_DIR/base.fvecs" \
        -graph "$GRAPH_FILE" \
        -out "$CODEBOOK_FILE" \
        -seed "$SEED"
    echo "Codebook saved to $CODEBOOK_FILE"
fi

# ── Step 5: Replicate queries/groundtruth ────────────────────────────────────
if [[ "$REPLICATE" -gt 0 ]]; then
    QUERY_REP="$DATA_DIR/query_${REPLICATE}x.fvecs"
    GT_REP="$DATA_DIR/groundtruth_${REPLICATE}x.ivecs"
    if [[ -f "$QUERY_REP" && -f "$GT_REP" ]]; then
        echo "Replicated files already exist: ${REPLICATE}x"
    else
        echo "Replicating queries & groundtruth ${REPLICATE}x ..."
        "$ROOT/build/replicate_queries" "$DATA_DIR/query.fvecs"        "$QUERY_REP" "$REPLICATE" fvecs
        "$ROOT/build/replicate_queries" "$DATA_DIR/groundtruth.ivecs"  "$GT_REP"    "$REPLICATE" ivecs
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "Dataset ready: $DATA_DIR"
ls -lh "$DATA_DIR"
