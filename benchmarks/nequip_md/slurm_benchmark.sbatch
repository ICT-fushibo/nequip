#!/usr/bin/env bash
#SBATCH --job-name=nequip-md
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

: "${BENCH_SCRIPT_DIR:?BENCH_SCRIPT_DIR must be exported by submit_benchmarks_slurm.sh}"
SCRIPT_DIR="${BENCH_SCRIPT_DIR}"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

: "${SYSTEMS_FILE:?SYSTEMS_FILE must be exported by submit_benchmarks_slurm.sh}"
: "${MODEL_PACKAGE:?MODEL_PACKAGE must be set}"
: "${COMPILED_MODEL:?COMPILED_MODEL must be set}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
: "${VALIDATION_DIR:?VALIDATION_DIR must be set}"

PYTHON_BIN="${PYTHON_BIN:-python}"
STEPS="${STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
TIMESTEP_FS="${TIMESTEP_FS:-1.0}"
TEMPERATURE_K="${TEMPERATURE_K:-300.0}"
VELOCITY_MODE="${VELOCITY_MODE:-maxwell}"
SEED="${SEED:-20260722}"

if [[ -n "${ENV_SETUP:-}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_SETUP}"
elif [[ -n "${CONDA_ENV:-}" ]]; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export PYTHONUNBUFFERED=1

mapfile -t SYSTEMS < <(
    awk -F '\t' 'NF >= 2 && $1 !~ /^[[:space:]]*#/ && $1 !~ /^[[:space:]]*$/ {print $1 "\t" $2}' "${SYSTEMS_FILE}"
)
if (( ${#SYSTEMS[@]} == 0 )); then
    echo "No systems found in ${SYSTEMS_FILE}" >&2
    exit 2
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
system_count="${#SYSTEMS[@]}"
repeat="$((task_id / system_count))"
system_index="$((task_id % system_count))"
IFS=$'\t' read -r label structure <<< "${SYSTEMS[system_index]}"

if [[ ! -f "${structure}" ]]; then
    echo "Structure not found: ${structure}" >&2
    exit 2
fi
validation_path="${VALIDATION_DIR}/${label}.json"
if [[ "${REQUIRE_VALIDATION:-1}" == "1" && ! -f "${validation_path}" ]]; then
    echo "Validation result not found: ${validation_path}" >&2
    exit 2
fi

case "$(((repeat + system_index) % 4))" in
    0) modes=(E0 E1 B0 B1) ;;
    1) modes=(E1 B0 B1 E0) ;;
    2) modes=(B0 B1 E0 E1) ;;
    3) modes=(B1 E0 E1 B0) ;;
esac

mkdir -p "${OUTPUT_DIR}/json" "${OUTPUT_DIR}/logs"
cd "${REPO_ROOT}"

echo "Job ID: ${SLURM_JOB_ID:-none}; array task: ${task_id}"
echo "Node: $(hostname); CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "System: ${label}; repeat: ${repeat}; mode order: ${modes[*]}"

# The four modes are deliberately separate processes on the same allocated GPU.
for mode in "${modes[@]}"; do
    output_path="${OUTPUT_DIR}/json/${label}.${mode}.repeat${repeat}.json"
    log_path="${OUTPUT_DIR}/logs/${label}.${mode}.repeat${repeat}.job${SLURM_JOB_ID:-none}.log"
    echo "Running ${label} repeat=${repeat} mode=${mode}"
    if [[ "${mode}" == "E0" || "${mode}" == "E1" ]]; then
        selected_hash="${MODEL_PACKAGE_SHA256:-}"
    else
        selected_hash="${COMPILED_MODEL_SHA256:-}"
    fi
    benchmark_command=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_md.py"
        --mode "${mode}"
        --structure "${structure}"
        --system-label "${label}"
        --model-package "${MODEL_PACKAGE}"
        --compiled-model "${COMPILED_MODEL}"
        --steps "${STEPS}"
        --warmup-steps "${WARMUP_STEPS}"
        --timestep-fs "${TIMESTEP_FS}"
        --temperature-k "${TEMPERATURE_K}"
        --velocity-mode "${VELOCITY_MODE}"
        --seed "${SEED}"
        --repeat "${repeat}"
        --output "${output_path}"
    )
    if [[ -f "${validation_path}" ]]; then
        benchmark_command+=(--validation-result "${validation_path}")
    fi
    if [[ -n "${selected_hash}" ]]; then
        benchmark_command+=(--model-sha256 "${selected_hash}")
    else
        benchmark_command+=(--skip-model-hash)
    fi
    "${benchmark_command[@]}" >"${log_path}" 2>&1
done

echo "Completed ${label} repeat=${repeat}"
