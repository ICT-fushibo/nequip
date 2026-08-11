#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PACKAGE="${MODEL_PACKAGE:-NequIP-OAM-L-0.1.nequip.zip}"
COMPILED_MODEL="${COMPILED_MODEL:-benchmark_artifacts/NequIP-OAM-L-0.1-ase-oeq-no-cg.nequip.pt2}"
STRUCTURE_DIR="${STRUCTURE_DIR:-../MatRIS-09bk/example/cif_file}"
RUN_ID="${RUN_ID:-nequip_nvt_baselines_$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_results/${RUN_ID}}"

GPU_ID="${GPU_ID:-1}"
REPEATS="${REPEATS:-3}"
STEPS="${STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
TIMESTEP_FS="${TIMESTEP_FS:-1.0}"
TAUT_FS="${TAUT_FS:-100.0}"
SEED="${SEED:-42}"
TEMPERATURES="${TEMPERATURES:-300 800}"
SYSTEMS="${SYSTEMS:-Cu16 Cu32 Cu64 Cu192 Cu512 Cu1024 H2O16 H2O32 H2O60 H2O64 H2O192 H2O512 H2O1024}"

if [[ "${SEED}" != "42" ]]; then
    echo "The numerical-reference seed is fixed at 42; got ${SEED}" >&2
    exit 2
fi
for required_model in "${MODEL_PACKAGE}" "${COMPILED_MODEL}"; do
    if [[ ! -f "${required_model}" ]]; then
        echo "Required model not found: ${required_model}" >&2
        exit 2
    fi
done
python "${SCRIPT_DIR}/check_b1_dependencies.py"

read -r -a system_names <<< "${SYSTEMS}"
read -r -a temperatures <<< "${TEMPERATURES}"
if (( ${#system_names[@]} == 0 || ${#temperatures[@]} == 0 )); then
    echo "SYSTEMS and TEMPERATURES must not be empty" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
SYSTEMS_FILE="${OUTPUT_ROOT}/systems.tsv"
printf '# label\tstructure\n' > "${SYSTEMS_FILE}"
for system in "${system_names[@]}"; do
    structure="${STRUCTURE_DIR}/${system}.cif"
    if [[ ! -f "${structure}" ]]; then
        echo "Structure not found: ${structure}" >&2
        exit 2
    fi
    printf '%s\t%s\n' "${system}" "${structure}" >> "${SYSTEMS_FILE}"
done

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"

{
    echo "run_id=${RUN_ID}"
    echo "repository=${REPO_ROOT}"
    echo "model_package=${MODEL_PACKAGE}"
    echo "compiled_model=${COMPILED_MODEL}"
    echo "structure_dir=${STRUCTURE_DIR}"
    echo "physical_gpu=${GPU_ID}"
    echo "systems=${SYSTEMS}"
    echo "temperatures_k=${TEMPERATURES}"
    echo "ensemble=nvt"
    echo "thermostat=berendsen"
    echo "timestep_fs=${TIMESTEP_FS}"
    echo "taut_fs=${TAUT_FS}"
    echo "steps=${STEPS}"
    echo "warmup_steps=${WARMUP_STEPS}"
    echo "repeats=${REPEATS}"
    echo "seed=${SEED}"
    git rev-parse HEAD
    python -c 'import ase, nequip, torch; print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"ase={ase.__version__}"); print(f"nequip={nequip.__version__}")'
    nvidia-smi -i "${GPU_ID}" --query-gpu=index,name,uuid,driver_version --format=csv,noheader
} > "${OUTPUT_ROOT}/run_metadata.txt"

for temperature in "${temperatures[@]}"; do
    TEMPERATURE_OUTPUT="${OUTPUT_ROOT}/${temperature}K"
    VALIDATION_DIR="${TEMPERATURE_OUTPUT}/validation"

    echo "Validating all systems at ${temperature} K on physical GPU ${GPU_ID}"
    GPU_ID="${GPU_ID}" \
    SYSTEMS_FILE="${SYSTEMS_FILE}" \
    MODEL_PACKAGE="${MODEL_PACKAGE}" \
    COMPILED_MODEL="${COMPILED_MODEL}" \
    OUTPUT_DIR="${TEMPERATURE_OUTPUT}" \
    VALIDATION_DIR="${VALIDATION_DIR}" \
    TIMESTEP_FS="${TIMESTEP_FS}" \
    TEMPERATURE_K="${temperature}" \
    ENSEMBLE=nvt \
    THERMOSTAT=berendsen \
    TAUT_FS="${TAUT_FS}" \
    VELOCITY_MODE=maxwell \
    SEED="${SEED}" \
    bash "${SCRIPT_DIR}/validate_all_modes.sh"

    echo "Benchmarking all systems at ${temperature} K on physical GPU ${GPU_ID}"
    GPU_ID="${GPU_ID}" \
    NGPUS=1 \
    SYSTEMS_FILE="${SYSTEMS_FILE}" \
    MODEL_PACKAGE="${MODEL_PACKAGE}" \
    COMPILED_MODEL="${COMPILED_MODEL}" \
    OUTPUT_DIR="${TEMPERATURE_OUTPUT}" \
    VALIDATION_DIR="${VALIDATION_DIR}" \
    REQUIRE_VALIDATION=1 \
    REPEATS="${REPEATS}" \
    STEPS="${STEPS}" \
    WARMUP_STEPS="${WARMUP_STEPS}" \
    TIMESTEP_FS="${TIMESTEP_FS}" \
    TEMPERATURE_K="${temperature}" \
    ENSEMBLE=nvt \
    THERMOSTAT=berendsen \
    TAUT_FS="${TAUT_FS}" \
    VELOCITY_MODE=maxwell \
    SEED="${SEED}" \
    bash "${SCRIPT_DIR}/run_8xh100.sh"

    python "${SCRIPT_DIR}/summarize_results.py" "${TEMPERATURE_OUTPUT}"
done

echo "All NVT baselines completed: ${OUTPUT_ROOT}"
