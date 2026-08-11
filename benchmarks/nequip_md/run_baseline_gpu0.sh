#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# All benchmark inputs and outputs are repository-relative by default.
STRUCTURE_DIR="${STRUCTURE_DIR:-benchmark_structures}"
SYSTEMS_FILE="${SYSTEMS_FILE:-${STRUCTURE_DIR}/systems.tsv}"
ARTIFACT_DIR="${ARTIFACT_DIR:-benchmark_artifacts}"
MODEL_SOURCE="${MODEL_SOURCE:-nequip.net:mir-group/NequIP-OAM-L:0.1}"
MODEL_PACKAGE="${MODEL_PACKAGE:-${ARTIFACT_DIR}/models/NequIP-OAM-L-0.1.nequip.zip}"
COMPILED_MODEL="${COMPILED_MODEL:-${ARTIFACT_DIR}/NequIP-OAM-L-0.1-ase-oeq-no-cg.nequip.pt2}"
RUN_ID="${RUN_ID:-baseline_gpu0_seed42_$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results/${RUN_ID}}"
VALIDATION_DIR="${VALIDATION_DIR:-${OUTPUT_DIR}/validation}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
STRUCTURE_SEED="${STRUCTURE_SEED:-42}"
REPEATS="${REPEATS:-5}"
STEPS="${STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
TIMESTEP_FS="${TIMESTEP_FS:-1.0}"
TEMPERATURE_K="${TEMPERATURE_K:-300.0}"
VELOCITY_MODE="${VELOCITY_MODE:-maxwell}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"

echo "Generating all benchmark structures with seed ${STRUCTURE_SEED}"
python "${SCRIPT_DIR}/generate_structures.py" \
    --output-dir "${STRUCTURE_DIR}" \
    --manifest "${SYSTEMS_FILE}" \
    --seed "${STRUCTURE_SEED}"

if [[ "${DOWNLOAD_MODEL}" == "1" ]]; then
    echo "Downloading the official model"
    python "${SCRIPT_DIR}/fetch_official_model.py" \
        --source "${MODEL_SOURCE}" \
        --output "${MODEL_PACKAGE}"
else
    echo "Verifying the local official model (offline mode)"
    python "${SCRIPT_DIR}/fetch_official_model.py" \
        --source "${MODEL_SOURCE}" \
        --output "${MODEL_PACKAGE}" \
        --verify-only
fi

echo "Checking the B1 GPU neighbor-list dependency"
python "${SCRIPT_DIR}/check_b1_dependencies.py"

echo "Compiling the shared B0/B1 AOTInductor + OpenEquivariance model on GPU ${GPU_ID}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
ARTIFACT_DIR="${ARTIFACT_DIR}" \
MODEL_SOURCE="${MODEL_SOURCE}" \
MODEL_PACKAGE="${MODEL_PACKAGE}" \
OUTPUT_MODEL="${COMPILED_MODEL}" \
WITH_CONSTANT_FOLD=0 \
bash "${SCRIPT_DIR}/compile_model.sh"

echo "Validating E0/E1/B0/B1 trajectories through step 1000"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
GPU_ID=0 \
SYSTEMS_FILE="${SYSTEMS_FILE}" \
MODEL_PACKAGE="${MODEL_PACKAGE}" \
COMPILED_MODEL="${COMPILED_MODEL}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
VALIDATION_DIR="${VALIDATION_DIR}" \
TIMESTEP_FS="${TIMESTEP_FS}" \
TEMPERATURE_K="${TEMPERATURE_K}" \
VELOCITY_MODE="${VELOCITY_MODE}" \
SEED="${SEED}" \
bash "${SCRIPT_DIR}/validate_all_modes.sh"

echo "Running the complete single-GPU baseline benchmark"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
NGPUS=1 \
SYSTEMS_FILE="${SYSTEMS_FILE}" \
MODEL_PACKAGE="${MODEL_PACKAGE}" \
COMPILED_MODEL="${COMPILED_MODEL}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
VALIDATION_DIR="${VALIDATION_DIR}" \
REQUIRE_VALIDATION=1 \
REPEATS="${REPEATS}" \
STEPS="${STEPS}" \
WARMUP_STEPS="${WARMUP_STEPS}" \
TIMESTEP_FS="${TIMESTEP_FS}" \
TEMPERATURE_K="${TEMPERATURE_K}" \
VELOCITY_MODE="${VELOCITY_MODE}" \
SEED="${SEED}" \
bash "${SCRIPT_DIR}/run_8xh100.sh"

echo "Summarizing benchmark results"
python "${SCRIPT_DIR}/summarize_results.py" "${OUTPUT_DIR}"

echo "Complete baseline finished: ${OUTPUT_DIR}"
