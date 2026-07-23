#!/usr/bin/env bash
# Shared Slurm environment for compile, validation, benchmark, and summary jobs.
# The submit shell may already have another CUDA module loaded. Purging and
# rebuilding CUDA_HOME from the selected nvcc prevents colon-joined paths such
# as cuda-12.8:cuda-12.4 from reaching torch.utils.cpp_extension.

source /etc/profile.d/modules.sh || exit $?
module purge || exit $?
unset CUDA_HOME CUDA_PATH CUDACXX
module load cuda/12.8 || exit $?
NEQUIP_CONDA_ENV="${NEQUIP_CONDA_ENV:-nequip_opt}"
source /share/home/fushibo/software/miniconda3/bin/activate "${NEQUIP_CONDA_ENV}" || exit $?
export NEQUIP_CONDA_ENV

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
# extension in validation/benchmark jobs. Include the active Python and Torch
# versions in the directory because PyTorch extensions are interpreter- and
# ABI-specific.
EXTENSION_ABI_TAG="$(
    python - <<'PY'
import re
import sys
import torch

raw = (
    f"py{sys.version_info.major}{sys.version_info.minor}_"
    f"torch{torch.__version__}_"
    f"torchcuda{torch.version.cuda}"
)
print(re.sub(r"[^A-Za-z0-9_.-]", "_", raw))
PY
)"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/share/home/fushibo/.cache/torch_extensions/nequip_${EXTENSION_ABI_TAG}_cuda128}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-32}}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

# AOTInductor, Triton, and the CUDA driver all maintain generated-code caches.
# Keep these job-local so a failed compile on a reused node cannot contaminate
# a later correctness check. OpenEquivariance's C++ extension cache above stays
# shared because the compile dependency builds and verifies it before arrays run.
JOB_CACHE_KEY="job${SLURM_JOB_ID:-manual}-task${SLURM_ARRAY_TASK_ID:-0}"
JOB_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/nequip_runtime_cache/${JOB_CACHE_KEY}"
export TORCHINDUCTOR_CACHE_DIR="${JOB_CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${JOB_CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="${JOB_CACHE_ROOT}/cuda"
mkdir -p \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}"

echo "Slurm CUDA environment:"
echo "  NEQUIP_CONDA_ENV=${NEQUIP_CONDA_ENV}"
echo "  EXTENSION_ABI_TAG=${EXTENSION_ABI_TAG}"
echo "  CUDA_HOME=${CUDA_HOME}"
echo "  CUDACXX=${CUDACXX}"
echo "  TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "  TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
echo "  TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "  CUDA_CACHE_PATH=${CUDA_CACHE_PATH}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
module list 2>&1
"${CUDACXX}" --version | tail -n 1
echo "gcc=$(gcc -dumpfullversion -dumpversion)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python - <<'PY'
import importlib.metadata
import sys
import torch

print(f"  python={sys.version.split()[0]}")
print(f"  python.executable={sys.executable}")
print(f"  torch={torch.__version__}")
print(f"  torch.version.git_version={torch.version.git_version}")
print(f"  torch.version.cuda={torch.version.cuda}")
for package in (
    "nequip",
    "e3nn",
    "openequivariance",
    "nvalchemi-toolkit-ops",
    "warp-lang",
    "triton",
):
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed"
    print(f"  {package}={version}")
print(f"  gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}")
PY
