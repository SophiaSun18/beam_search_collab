#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH -J warpann-bench
#SBATCH -o log/bench.o%j
#SBATCH -e log/bench.e%j

set -eo pipefail

module --force purge
module load modtree/gpu
module load cuda/12.8.0
module load gcc/11.2.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

mkdir -p result log

for cfg in config/*.json; do
    name=$(basename "$cfg" .json)
    echo "========================================"
    echo "Config: $name"
    echo "========================================"
    ./build/gpu_cli < "$cfg" 2>&1 | tee "result/${name}.txt"
    echo ""
done

echo "==> All configs done. Results in result/"
