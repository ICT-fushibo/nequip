#!/usr/bin/env bash
# Shared Slurm environment for compile, validation, benchmark, and summary jobs.
# The submit shell may already have another CUDA module loaded. Purging and
# rebuilding CUDA_HOME from the selected nvcc prevents colon-joined paths such
# as cuda-12.8:cuda-12.4 from reaching torch.utils.cpp_extension.

source /etc/profile.d/modules.sh || exit $?
module purge || exit $?
unset CUDA_HOME CUDA_PATH CUDACXX
module load cuda/12.8 || exit $?
source /share/home/fushibo/software/miniconda3/bin/activate nequip_opt || exit $?

# Match the cluster template: enable strict mode only after modules and Conda
# are initialized, because their activation scripts are not guaranteed to be
# nounset-safe.
set -euo pipefail

if ! NVCC_PATH="$(command -v nvcc)"; then
    echo "nvcc is unavailable after loading cuda/12.8" >&2
    exit 2
fi
NVCC_PATH="$(readlink -f "${NVCC_PATH}")"
export CUDA_HOME="$(dirname -- "$(dirname -- "${NVCC_PATH}")")"
export CUDA_PATH="${CUDA_HOME}"
export CUDACXX="${NVCC_PATH}"

if [[ "${CUDA_HOME}" == *:* ]]; then
    echo "Invalid CUDA_HOME contains more than one path: ${CUDA_HOME}" >&2
    exit 2
fi
if [[ ! -f "${CUDA_HOME}/include/cuda_runtime_api.h" ]]; then
    echo "CUDA header not found: ${CUDA_HOME}/include/cuda_runtime_api.h" >&2
    exit 2
fi

# Compile OpenEquivariance once in the dependency job and reuse the resulting
# extension in validation/benchmark jobs. The versioned directory also avoids
# the stale failed build left under the default ~/.cache/torch_extensions.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/share/home/fushibo/.cache/torch_extensions/nequip_py310_torch29_cu126_cuda128}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-32}}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

echo "Slurm CUDA environment:"
echo "  CUDA_HOME=${CUDA_HOME}"
echo "  CUDACXX=${CUDACXX}"
echo "  TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
module list 2>&1
"${CUDACXX}" --version | tail -n 1
python - <<'PY'
import torch

print(f"  torch={torch.__version__}")
print(f"  torch.version.cuda={torch.version.cuda}")
print(f"  gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}")
PY
