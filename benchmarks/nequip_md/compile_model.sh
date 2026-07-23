#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/benchmark_artifacts}"
MODEL_SOURCE="${MODEL_SOURCE:-nequip.net:mir-group/NequIP-OAM-L:0.1}"
MODEL_PACKAGE="${MODEL_PACKAGE:-${ARTIFACT_DIR}/models/NequIP-OAM-L-0.1.nequip.zip}"
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

mkdir -p "$(dirname -- "${OUTPUT_MODEL}")" "$(dirname -- "${MODEL_PACKAGE}")"
cd "${REPO_ROOT}"

echo "Verifying pre-downloaded official model (offline)"
python "${SCRIPT_DIR}/fetch_official_model.py" \
    --source "${MODEL_SOURCE}" \
    --output "${MODEL_PACKAGE}" \
    --verify-only

if ! command -v sha256sum >/dev/null 2>&1; then
    echo "sha256sum is required to validate benchmark artifacts" >&2
    exit 2
fi
MODEL_PACKAGE_SHA256="$(sha256sum "${MODEL_PACKAGE}" | awk '{print $1}')"
MODEL_PROVENANCE="${OUTPUT_MODEL}.model.sha256"

if [[ -f "${OUTPUT_MODEL}" && -f "${OUTPUT_MODEL}.sha256" && -f "${MODEL_PROVENANCE}" ]] \
    && sha256sum --check --status "${OUTPUT_MODEL}.sha256" \
    && [[ "$(awk 'NR == 1 {print $1}' "${MODEL_PROVENANCE}")" == "${MODEL_PACKAGE_SHA256}" ]]; then
    echo "Verified compiled model already exists; reusing: ${OUTPUT_MODEL}"
    exit 0
fi

# nequip-compile writes the package before running its numerical sanity check.
# Compile to a temporary .nequip.pt2 name so a failed check can never leave a
# formal artifact that a later pipeline run mistakes for valid.
OUTPUT_BASENAME="$(basename -- "${OUTPUT_MODEL}" .nequip.pt2)"
PARTIAL_MODEL="$(dirname -- "${OUTPUT_MODEL}")/${OUTPUT_BASENAME}.partial.${SLURM_JOB_ID:-$$}.nequip.pt2"
rm -f "${PARTIAL_MODEL}"
cleanup_partial() {
    rm -f "${PARTIAL_MODEL}"
}
trap cleanup_partial EXIT

compile_command=(
    nequip-compile
    "${MODEL_PACKAGE}"
    "${PARTIAL_MODEL}"
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
echo "Official model ID: ${MODEL_SOURCE}"
echo "Input model: ${MODEL_PACKAGE}"
echo "Output model: ${OUTPUT_MODEL}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<managed by runtime>}"
printf 'Command:'
printf ' %q' "${compile_command[@]}"
printf '\n'

"${compile_command[@]}"

mv -f "${PARTIAL_MODEL}" "${OUTPUT_MODEL}"
trap - EXIT
sha256sum "${OUTPUT_MODEL}" >"${OUTPUT_MODEL}.sha256"
printf '%s  %s\n' "${MODEL_PACKAGE_SHA256}" "${MODEL_PACKAGE}" >"${MODEL_PROVENANCE}"
echo "Compilation completed: ${OUTPUT_MODEL}"
