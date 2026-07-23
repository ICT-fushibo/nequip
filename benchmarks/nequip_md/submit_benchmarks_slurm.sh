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
export REQUIRE_VALIDATION="${REQUIRE_VALIDATION:-1}"
export REPEATS="${REPEATS:-5}"
export STEPS="${STEPS:-1000}"
export WARMUP_STEPS="${WARMUP_STEPS:-3}"
export TIMESTEP_FS="${TIMESTEP_FS:-1.0}"
export TEMPERATURE_K="${TEMPERATURE_K:-300.0}"
export VELOCITY_MODE="${VELOCITY_MODE:-maxwell}"
export SEED="${SEED:-20260722}"
export NEQUIP_CONDA_ENV="${NEQUIP_CONDA_ENV:-nequip_opt}"
export BENCH_SCRIPT_DIR="${SCRIPT_DIR}"

artifact_hash() {
    local artifact="$1"
    if [[ -f "${artifact}.sha256" ]]; then
        awk 'NR == 1 {print $1}' "${artifact}.sha256"
    elif command -v sha256sum >/dev/null 2>&1 && [[ -f "${artifact}" ]]; then
        sha256sum "${artifact}" | awk '{print $1}'
    else
        echo ""
    fi
}

MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

for required_path in "${SYSTEMS_FILE}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 2
    fi
done
if [[ ! -e "${MODEL_PACKAGE}" && -z "${VALIDATION_JOB_ID:-}" ]]; then
    echo "Official model package not found: ${MODEL_PACKAGE}" >&2
    echo "Run compilation first, or provide VALIDATION_JOB_ID for an afterok dependency." >&2
    exit 2
fi

# In the complete pipeline this script runs immediately after submitting the
# compile and validation jobs. The AOTI artifact does not exist on the submit
# host yet, but the afterok dependency guarantees that the benchmark cannot
# start until validation (and therefore compilation) has completed.
if [[ ! -e "${COMPILED_MODEL}" ]]; then
    if [[ -z "${VALIDATION_JOB_ID:-}" ]]; then
        echo "Compiled model not found: ${COMPILED_MODEL}" >&2
        echo "Compile it first, or provide VALIDATION_JOB_ID for an afterok dependency." >&2
        exit 2
    fi
    echo "Compiled model is pending dependency job ${VALIDATION_JOB_ID}: ${COMPILED_MODEL}" >&2
fi

if [[ "${REQUIRE_VALIDATION}" == "1" && -z "${VALIDATION_JOB_ID:-}" ]]; then
    while IFS=$'\t' read -r label structure _; do
        [[ -z "${label}" || "${label}" =~ ^[[:space:]]*# ]] && continue
        if [[ ! -f "${VALIDATION_DIR}/${label}.json" ]]; then
            echo "Validation result missing: ${VALIDATION_DIR}/${label}.json" >&2
            echo "Run/submit numerical validation first, or provide VALIDATION_JOB_ID." >&2
            exit 2
        fi
    done < "${SYSTEMS_FILE}"
fi

# Hash artifacts already present on the submit host. If the compiled artifact
# is still being produced, slurm_benchmark.sbatch reads its .sha256 sidecar
# after the dependency resolves and before starting any timed work.
export MODEL_PACKAGE_SHA256="${MODEL_PACKAGE_SHA256:-$(artifact_hash "${MODEL_PACKAGE}")}"
export COMPILED_MODEL_SHA256="${COMPILED_MODEL_SHA256:-$(artifact_hash "${COMPILED_MODEL}")}"

system_count="$(awk -F '\t' 'NF >= 2 && $1 !~ /^[[:space:]]*#/ && $1 !~ /^[[:space:]]*$/ {count++} END {print count+0}' "${SYSTEMS_FILE}")"
if (( system_count == 0 )); then
    echo "No systems found in ${SYSTEMS_FILE}" >&2
    exit 2
fi
task_count="${system_count}"
array_end="$((task_count - 1))"

mkdir -p "${OUTPUT_DIR}/json" "${OUTPUT_DIR}/logs" /share/home/fushibo/MD_opt/nequip/log

sbatch_args=(
    --parsable
    --array="0-${array_end}%${MAX_CONCURRENT}"
    --export=ALL
)
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
    sbatch_args+=(--account="${SLURM_ACCOUNT}")
fi
if [[ -n "${VALIDATION_JOB_ID:-}" ]]; then
    sbatch_args+=(--dependency="afterok:${VALIDATION_JOB_ID}")
fi

echo "Submitting ${task_count} system-grouped array tasks; each runs ${REPEATS} repeats" >&2
echo "At most ${MAX_CONCURRENT} H100 jobs will run concurrently" >&2
sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/slurm_benchmark.sbatch"
