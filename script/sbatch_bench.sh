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
module load gcc/11.2.0
module load cuda/12.8.0

mkdir -p result

for data_cfg in config/dataset/*.json; do
    data_name=$(basename "$data_cfg" .json)
    for search_cfg in config/search/*.json; do
        search_name=$(basename "$search_cfg" .json)
        result_file="result/${data_name}__${search_name}.txt"
        echo "========================================"
        echo "Data: $data_name  Search: $search_name"
        echo "========================================"
        jq -s '{data: .[0], search: .[1]}' "$data_cfg" "$search_cfg" \
            | ./build/gpu_cli 2>&1 | tee "$result_file"
        echo ""
    done
done

echo "==> All configs done. Results in result/"
