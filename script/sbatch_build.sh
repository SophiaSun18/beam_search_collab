#!/bin/bash
#SBATCH -p gpu-debug
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH -J warpann-build
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"

module --force purge
module load modtree/gpu
module load cuda/12.8.0
module load gcc/11.2.0

mkdir -p build log
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="80;90"
make -C build -j"$(nproc)"

echo "==> Build complete."
