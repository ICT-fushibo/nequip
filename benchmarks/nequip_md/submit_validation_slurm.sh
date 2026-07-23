#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export SYSTEMS_FILE="${SYSTEMS_FILE:-${SCRIPT_DIR}/systems.tsv}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/benchmark_artifacts}"
export MODEL_PACKAGE="${MODEL_PACKAGE:-${ARTIFACT_DIR}/models/NequIP-OAM-L-0.1.nequip.zip}"
export COMPILED_MODEL="${COMPILED_MODEL:-${ARTIFACT_DIR}/NequIP-OAM-L-0.1-ase-oeq-no-cg.nequip.pt2}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmark_results/nequip_e0_e1_b0_b1}"
export VALIDATION_DIR="${VALIDATION_DIR:-${OUTPUT_DIR}/validation}"
export TIMESTEP_FS="${TIMESTEP_FS:-1.0}"
export TEMPERATURE_K="${TEMPERATURE_K:-300.0}"
export VELOCITY_MODE="${VELOCITY_MODE:-maxwell}"
export SEED="${SEED:-20260722}"
export NEQUIP_CONDA_ENV="${NEQUIP_CONDA_ENV:-nequip_opt}"
export BENCH_SCRIPT_DIR="${SCRIPT_DIR}"

MAX_CONCURRENT="${MAX_VALIDATION_CONCURRENT:-8}"

for required_path in "${SYSTEMS_FILE}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 2
    fi
done
if [[ ! -e "${MODEL_PACKAGE}" && -z "${COMPILE_JOB_ID:-}" ]]; then
    echo "Official model package not found: ${MODEL_PACKAGE}" >&2
    echo "Run compilation first, or provide COMPILE_JOB_ID for an afterok dependency." >&2
    exit 2
fi
if [[ -z "${COMPILE_JOB_ID:-}" && ! -e "${COMPILED_MODEL}" ]]; then
    echo "Compiled model not found: ${COMPILED_MODEL}" >&2
    echo "Compile it first, or provide COMPILE_JOB_ID for an afterok dependency." >&2
    exit 2
fi

system_count="$(awk -F '\t' 'NF >= 2 && $1 !~ /^[[:space:]]*#/ && $1 !~ /^[[:space:]]*$/ {count++} END {print count+0}' "${SYSTEMS_FILE}")"
if (( system_count == 0 )); then
    echo "No systems found in ${SYSTEMS_FILE}" >&2
    exit 2
fi
array_end="$((system_count - 1))"
mkdir -p "${VALIDATION_DIR}" /share/home/fushibo/MD_opt/nequip/log

sbatch_args=(
    --parsable
    --array="0-${array_end}%${MAX_CONCURRENT}"
    --export=ALL
)
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
    sbatch_args+=(--account="${SLURM_ACCOUNT}")
fi
if [[ -n "${COMPILE_JOB_ID:-}" ]]; then
    sbatch_args+=(--dependency="afterok:${COMPILE_JOB_ID}")
fi

echo "Submitting ${system_count} trajectory-validation jobs" >&2
sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/slurm_validation.sbatch"
