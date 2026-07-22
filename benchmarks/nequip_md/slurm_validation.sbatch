#!/usr/bin/env bash
#SBATCH --job-name=nequip-validate
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/share/home/fushibo/MD_opt/nequip/log/validation-%A_%a.log

# Fixed cluster environment.
source /etc/profile.d/modules.sh
module load cuda/12.8
source /share/home/fushibo/software/miniconda3/bin/activate nequip_opt

set -euo pipefail

: "${BENCH_SCRIPT_DIR:?BENCH_SCRIPT_DIR must be exported}"
SCRIPT_DIR="${BENCH_SCRIPT_DIR}"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

: "${SYSTEMS_FILE:?SYSTEMS_FILE must be set}"
: "${MODEL_PACKAGE:?MODEL_PACKAGE must be set}"
: "${COMPILED_MODEL:?COMPILED_MODEL must be set}"
: "${VALIDATION_DIR:?VALIDATION_DIR must be set}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTEP_FS="${TIMESTEP_FS:-1.0}"
TEMPERATURE_K="${TEMPERATURE_K:-300.0}"
VELOCITY_MODE="${VELOCITY_MODE:-maxwell}"
SEED="${SEED:-20260722}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export PYTHONUNBUFFERED=1

mapfile -t SYSTEMS < <(
    awk -F '\t' 'NF >= 2 && $1 !~ /^[[:space:]]*#/ && $1 !~ /^[[:space:]]*$/ {print $1 "\t" $2}' "${SYSTEMS_FILE}"
)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_id >= ${#SYSTEMS[@]} )); then
    echo "Array index ${task_id} exceeds system count ${#SYSTEMS[@]}" >&2
    exit 2
fi
IFS=$'\t' read -r label structure <<< "${SYSTEMS[task_id]}"

mkdir -p "${VALIDATION_DIR}"
cd "${REPO_ROOT}"

echo "Validating ${label} through 1000 MD steps on $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_modes.py" \
    --structure "${structure}" \
    --system-label "${label}" \
    --model-package "${MODEL_PACKAGE}" \
    --compiled-model "${COMPILED_MODEL}" \
    --timestep-fs "${TIMESTEP_FS}" \
    --temperature-k "${TEMPERATURE_K}" \
    --velocity-mode "${VELOCITY_MODE}" \
    --seed "${SEED}" \
    --output "${VALIDATION_DIR}/${label}.json"
