#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH -J warpann-bench
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"

module --force purge
module load modtree/gpu
module load cuda/12.8.0
module load gcc/11.2.0

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
