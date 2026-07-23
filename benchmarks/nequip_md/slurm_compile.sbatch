#!/usr/bin/env bash
#SBATCH --job-name=compile
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/share/home/fushibo/MD_opt/nequip/log/compile.log

: "${BENCH_SCRIPT_DIR:?BENCH_SCRIPT_DIR must be exported by submit_compile_slurm.sh}"
SCRIPT_DIR="${BENCH_SCRIPT_DIR}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_slurm_env.sh"

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

# Slurm has already restricted CUDA_VISIBLE_DEVICES to the allocated GPU.
unset GPU_ID || true
unset CONDA_ENV ENV_SETUP || true

echo "Precompiling the shared OpenEquivariance extension"
python - <<'PY'
from openequivariance._torch.extlib import DeviceProp

print("OpenEquivariance CUDA device:", DeviceProp(0))
PY

bash "${SCRIPT_DIR}/compile_model.sh"
