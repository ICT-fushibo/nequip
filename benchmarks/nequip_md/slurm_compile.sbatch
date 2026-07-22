#!/usr/bin/env bash
#SBATCH --job-name=nequip-compile
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

: "${BENCH_SCRIPT_DIR:?BENCH_SCRIPT_DIR must be exported by submit_compile_slurm.sh}"
SCRIPT_DIR="${BENCH_SCRIPT_DIR}"

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

# Slurm has already restricted CUDA_VISIBLE_DEVICES to the allocated GPU.
unset GPU_ID || true
bash "${SCRIPT_DIR}/compile_model.sh"
