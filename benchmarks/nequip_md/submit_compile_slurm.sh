#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SLURM_LOG_DIR="/share/home/fushibo/MD_opt/nequip/log"

export ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/benchmark_artifacts}"
export MODEL_SOURCE="${MODEL_SOURCE:-nequip.net:mir-group/NequIP-OAM-L:0.1}"
export MODEL_PACKAGE="${MODEL_PACKAGE:-${ARTIFACT_DIR}/models/NequIP-OAM-L-0.1.nequip.zip}"
export WITH_CONSTANT_FOLD="${WITH_CONSTANT_FOLD:-0}"
export BENCH_SCRIPT_DIR="${SCRIPT_DIR}"

mkdir -p "${SLURM_LOG_DIR}"

sbatch_args=(
    --parsable
    --export=ALL
)
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
    sbatch_args+=(--account="${SLURM_ACCOUNT}")
fi

echo "Submitting AOTI+OpenEquivariance compilation" >&2
sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/slurm_compile.sbatch"
