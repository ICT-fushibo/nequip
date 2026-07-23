#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SYSTEMS_FILE="${SYSTEMS_FILE:-${SCRIPT_DIR}/systems.tsv}"
MODEL_PACKAGE="${MODEL_PACKAGE:-${REPO_ROOT}/benchmark_artifacts/models/NequIP-OAM-L-0.1.nequip.zip}"
COMPILED_MODEL="${COMPILED_MODEL:-${REPO_ROOT}/benchmark_artifacts/NequIP-OAM-L-0.1-ase-oeq-no-cg.nequip.pt2}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmark_results/nequip_e0_e1_b0_b1}"
VALIDATION_DIR="${VALIDATION_DIR:-${OUTPUT_DIR}/validation}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
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

for required_path in "${SYSTEMS_FILE}" "${MODEL_PACKAGE}" "${COMPILED_MODEL}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 2
    fi
done

mkdir -p "${VALIDATION_DIR}"
while IFS=$'\t' read -r label structure _; do
    [[ -z "${label}" || "${label}" =~ ^[[:space:]]*# ]] && continue
    echo "Validating ${label}: ${structure}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_modes.py" \
        --structure "${structure}" \
        --system-label "${label}" \
        --model-package "${MODEL_PACKAGE}" \
        --compiled-model "${COMPILED_MODEL}" \
        --timestep-fs "${TIMESTEP_FS}" \
        --temperature-k "${TEMPERATURE_K}" \
        --velocity-mode "${VELOCITY_MODE}" \
        --seed "${SEED}" \
        --output "${VALIDATION_DIR}/${label}.json"
done < "${SYSTEMS_FILE}"
