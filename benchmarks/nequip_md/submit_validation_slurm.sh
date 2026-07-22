#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export SYSTEMS_FILE="${SYSTEMS_FILE:-${SCRIPT_DIR}/systems.tsv}"
export MODEL_PACKAGE="${MODEL_PACKAGE:-/share/home/fushibo/NequIP-OAM-L-0.1.nequip.zip}"
export COMPILED_MODEL="${COMPILED_MODEL:-${REPO_ROOT}/benchmark_artifacts/NequIP-OAM-L-0.1-ase-oeq-no-cg.nequip.pt2}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmark_results/nequip_e0_e1_b0_b1}"
export VALIDATION_DIR="${VALIDATION_DIR:-${OUTPUT_DIR}/validation}"
export TIMESTEP_FS="${TIMESTEP_FS:-1.0}"
export TEMPERATURE_K="${TEMPERATURE_K:-300.0}"
export VELOCITY_MODE="${VELOCITY_MODE:-maxwell}"
export SEED="${SEED:-20260722}"
export CONDA_ENV="${CONDA_ENV:-nequip_opt}"
export ENV_SETUP="${ENV_SETUP:-}"
export BENCH_SCRIPT_DIR="${SCRIPT_DIR}"

PARTITION="${PARTITION:-H100}"
GRES="${GRES:-gpu:h100:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${VALIDATION_TIME_LIMIT:-08:00:00}"
MAX_CONCURRENT="${MAX_VALIDATION_CONCURRENT:-6}"

for required_path in "${SYSTEMS_FILE}" "${MODEL_PACKAGE}" "${COMPILED_MODEL}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 2
    fi
done

system_count="$(awk -F '\t' 'NF >= 2 && $1 !~ /^[[:space:]]*#/ && $1 !~ /^[[:space:]]*$/ {count++} END {print count+0}' "${SYSTEMS_FILE}")"
if (( system_count == 0 )); then
    echo "No systems found in ${SYSTEMS_FILE}" >&2
    exit 2
fi
array_end="$((system_count - 1))"
mkdir -p "${VALIDATION_DIR}" "${OUTPUT_DIR}/slurm_validation"

sbatch_args=(
    --parsable
    --partition="${PARTITION}"
    --gres="${GRES}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEMORY}"
    --time="${TIME_LIMIT}"
    --array="0-${array_end}%${MAX_CONCURRENT}"
    --output="${OUTPUT_DIR}/slurm_validation/slurm-%A_%a.out"
    --error="${OUTPUT_DIR}/slurm_validation/slurm-%A_%a.err"
    --export=ALL
)
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
    sbatch_args+=(--account="${SLURM_ACCOUNT}")
fi

echo "Submitting ${system_count} trajectory-validation jobs" >&2
sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/slurm_validation.sbatch"
