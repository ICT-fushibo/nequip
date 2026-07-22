#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PACKAGE="${MODEL_PACKAGE:-/share/home/fushibo/NequIP-OAM-L-0.1.nequip.zip}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/benchmark_artifacts}"
WITH_CONSTANT_FOLD="${WITH_CONSTANT_FOLD:-0}"

if [[ "${WITH_CONSTANT_FOLD}" == "1" ]]; then
    OUTPUT_MODEL="${OUTPUT_MODEL:-${ARTIFACT_DIR}/NequIP-OAM-L-0.1-ase-oeq-cf-no-cg.nequip.pt2}"
else
    OUTPUT_MODEL="${OUTPUT_MODEL:-${ARTIFACT_DIR}/NequIP-OAM-L-0.1-ase-oeq-no-cg.nequip.pt2}"
fi

if [[ -n "${ENV_SETUP:-}" ]]; then
    # ENV_SETUP may load modules and/or activate a virtual environment.
    # shellcheck disable=SC1090
    source "${ENV_SETUP}"
elif [[ -n "${CONDA_ENV:-}" ]]; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

if [[ ! -f "${MODEL_PACKAGE}" ]]; then
    echo "Model package not found: ${MODEL_PACKAGE}" >&2
    exit 2
fi
if [[ -e "${OUTPUT_MODEL}" ]]; then
    echo "Compiled model already exists; reusing: ${OUTPUT_MODEL}"
    exit 0
fi

mkdir -p "$(dirname -- "${OUTPUT_MODEL}")"
cd "${REPO_ROOT}"

compile_command=(
    nequip-compile
    "${MODEL_PACKAGE}"
    "${OUTPUT_MODEL}"
    --mode aotinductor
    --device cuda
    --target ase
    --no-tf32
    --modifiers enable_OpenEquivariance
    --inductor-configs triton.cudagraphs=False
)
if [[ "${WITH_CONSTANT_FOLD}" == "1" ]]; then
    compile_command+=(--constant-fold)
fi

echo "Repository: ${REPO_ROOT}"
echo "Input model: ${MODEL_PACKAGE}"
echo "Output model: ${OUTPUT_MODEL}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<managed by runtime>}"
printf 'Command:'
printf ' %q' "${compile_command[@]}"
printf '\n'

"${compile_command[@]}"

if command -v sha256sum >/dev/null 2>&1 && [[ -f "${OUTPUT_MODEL}" ]]; then
    sha256sum "${OUTPUT_MODEL}" >"${OUTPUT_MODEL}.sha256"
fi
echo "Compilation completed: ${OUTPUT_MODEL}"
