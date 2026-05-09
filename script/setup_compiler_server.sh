#!/bin/bash
# Set up compiler environment on Anvil using modules.
#
# Usage:
#   source script/setup_compiler_server.sh

module --force purge
module load modtree/gpu
module load gcc/11.2.0
module load cuda/12.8.0

echo "Compilers:"
nvcc --version
g++ --version | head -1
