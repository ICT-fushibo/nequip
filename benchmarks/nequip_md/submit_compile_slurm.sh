#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PARTITION="${PARTITION:-H100}"
GRES="${GRES:-gpu:h100:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${REPO_ROOT}/benchmark_results/slurm_logs}"

export MODEL_PACKAGE="${MODEL_PACKAGE:-/share/home/fushibo/NequIP-OAM-L-0.1.nequip.zip}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/benchmark_artifacts}"
export WITH_CONSTANT_FOLD="${WITH_CONSTANT_FOLD:-0}"
export CONDA_ENV="${CONDA_ENV:-nequip_opt}"
export ENV_SETUP="${ENV_SETUP:-}"
export BENCH_SCRIPT_DIR="${SCRIPT_DIR}"

mkdir -p "${SLURM_LOG_DIR}"

sbatch_args=(
    --partition="${PARTITION}"
    --gres="${GRES}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEMORY}"
    --time="${TIME_LIMIT}"
    --output="${SLURM_LOG_DIR}/compile-%j.out"
    --error="${SLURM_LOG_DIR}/compile-%j.err"
    --export=ALL
)
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
    sbatch_args+=(--account="${SLURM_ACCOUNT}")
fi

sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/slurm_compile.sbatch"
