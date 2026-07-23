#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

: "${BENCHMARK_JOB_ID:?BENCHMARK_JOB_ID must identify the performance array}"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmark_results/nequip_e0_e1_b0_b1}"
export NEQUIP_CONDA_ENV="${NEQUIP_CONDA_ENV:-nequip_opt}"
export BENCH_SCRIPT_DIR="${SCRIPT_DIR}"

mkdir -p /share/home/fushibo/MD_opt/nequip/log

sbatch_args=(
    --parsable
    --dependency="afterany:${BENCHMARK_JOB_ID}"
    --export=ALL
)
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
    sbatch_args+=(--account="${SLURM_ACCOUNT}")
fi

echo "Submitting summary after performance array ${BENCHMARK_JOB_ID}" >&2
sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/slurm_summary.sbatch"
