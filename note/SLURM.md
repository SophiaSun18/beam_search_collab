# SLURM on Purdue Anvil

## Filesystem

All nodes (login + compute) share a single Lustre parallel filesystem.
Home (`/home/x-tyang19/`) and scratch (`/anvil/scratch/x-tyang19/`) are visible everywhere — no staging or copying needed.

Software stacks (CUDA, compilers, MPI) are installed but not on `PATH` by default.
They are exposed via `module load`. Alternatively, a local toolchain in `~/.local/` (installed by `script/setup_compiler.sh`) works without the module system.

## GPU Partitions

| Partition   | Max Time | GPU Limit                          | Notes                     |
|-------------|----------|------------------------------------|---------------------------|
| `gpu`       | 48h      | 12 GPUs/user, 32/allocation        | Shared nodes              |
| `gpu-debug` | 30min    | 2 GPUs/job, 1 job at a time        | Quick testing             |

Hardware: 16 nodes x 4 A100, 21 nodes x 4 H100. Use `--constraint=A100` or `--constraint=H100` to select.

## Submitting Jobs

```bash
sbatch job.sh
```

Required flags:
- `-A <allocation>` — run `mybalance` to find yours
- `-p gpu` or `-p gpu-debug`
- `--gpus-per-node=N` — must use this form, not `--gres`

Minimal sbatch script:
```bash
#!/bin/sh
#SBATCH -A myallocation
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00

source ~/.bashrc
cd /home/x-tyang19/warpann
./build/gpu_cli < config/sift.json
```

## Useful Commands

- `sbatch job.sh` — submit
- `squeue -u $USER` — check running jobs
- `scancel <jobid>` — cancel
- `mybalance` — check allocation SUs
