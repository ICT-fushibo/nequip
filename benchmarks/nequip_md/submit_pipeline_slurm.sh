#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Fixed production inputs for every CIF system except demo.cif.
export SYSTEMS_FILE="${SYSTEMS_FILE:-${SCRIPT_DIR}/systems.slurm.tsv}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/benchmark_artifacts}"
export MODEL_SOURCE="${MODEL_SOURCE:-nequip.net:mir-group/NequIP-OAM-L:0.1}"
export MODEL_PACKAGE="${MODEL_PACKAGE:-${ARTIFACT_DIR}/models/NequIP-OAM-L-0.1.nequip.zip}"
export COMPILED_MODEL="${COMPILED_MODEL:-${ARTIFACT_DIR}/NequIP-OAM-L-0.1-ase-oeq-no-cg.nequip.pt2}"
export OUTPUT_MODEL="${OUTPUT_MODEL:-${COMPILED_MODEL}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmark_results/nequip_e0_e1_b0_b1/${RUN_ID}}"
export VALIDATION_DIR="${VALIDATION_DIR:-${OUTPUT_DIR}/validation}"
export REPEATS="${REPEATS:-5}"
export STEPS="${STEPS:-1000}"
export WARMUP_STEPS="${WARMUP_STEPS:-3}"
export TIMESTEP_FS="${TIMESTEP_FS:-1.0}"
export TEMPERATURE_K="${TEMPERATURE_K:-300.0}"
export VELOCITY_MODE="${VELOCITY_MODE:-maxwell}"
export SEED="${SEED:-20260722}"
export NEQUIP_CONDA_ENV="${NEQUIP_CONDA_ENV:-nequip_opt}"

for required_path in "${SYSTEMS_FILE}" "${MODEL_PACKAGE}" "${MODEL_PACKAGE}.sha256" "${MODEL_PACKAGE}.source"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 2
    fi
done

system_count=0
while IFS=$'\t' read -r label structure _; do
    [[ -z "${label}" || "${label}" =~ ^[[:space:]]*# ]] && continue
    if [[ ! -f "${structure}" ]]; then
        echo "Structure for ${label} not found: ${structure}" >&2
        exit 2
    fi
    system_count="$((system_count + 1))"
done < "${SYSTEMS_FILE}"

EXPECTED_SYSTEMS="${EXPECTED_SYSTEMS:-13}"
if (( system_count != EXPECTED_SYSTEMS )); then
    echo "Expected ${EXPECTED_SYSTEMS} systems, found ${system_count} in ${SYSTEMS_FILE}" >&2
    exit 2
fi

compile_submission="$(bash "${SCRIPT_DIR}/submit_compile_slurm.sh")"
compile_job_id="${compile_submission%%;*}"
if ! [[ "${compile_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse compile job ID from: ${compile_submission}" >&2
    exit 2
fi

validation_submission="$(COMPILE_JOB_ID="${compile_job_id}" bash "${SCRIPT_DIR}/submit_validation_slurm.sh")"
validation_job_id="${validation_submission%%;*}"
if ! [[ "${validation_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse validation job ID from: ${validation_submission}" >&2
    exit 2
fi

benchmark_submission="$(VALIDATION_JOB_ID="${validation_job_id}" bash "${SCRIPT_DIR}/submit_benchmarks_slurm.sh")"
benchmark_job_id="${benchmark_submission%%;*}"
if ! [[ "${benchmark_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse benchmark job ID from: ${benchmark_submission}" >&2
    exit 2
fi

summary_submission="$(BENCHMARK_JOB_ID="${benchmark_job_id}" bash "${SCRIPT_DIR}/submit_summary_slurm.sh")"
summary_job_id="${summary_submission%%;*}"
if ! [[ "${summary_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse summary job ID from: ${summary_submission}" >&2
    exit 2
fi

echo "Compile job:          ${compile_job_id}"
echo "Validation array job: ${validation_job_id}"
echo "Performance array job: ${benchmark_job_id}"
echo "Summary job:          ${summary_job_id}"
echo "Pipeline: ${compile_job_id} -> ${validation_job_id} -> ${benchmark_job_id} -> ${summary_job_id}"
echo "Results directory: ${OUTPUT_DIR}"
