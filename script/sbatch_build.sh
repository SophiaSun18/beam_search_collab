#!/bin/bash
#SBATCH -p gpu-debug
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH -J warpann-build
#SBATCH -o log/build.o%j
#SBATCH -e log/build.e%j

set -euo pipefail
source ~/.bashrc

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

mkdir -p build log
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="80;90"
make -C build -j"$(nproc)"

echo "==> Build complete."
