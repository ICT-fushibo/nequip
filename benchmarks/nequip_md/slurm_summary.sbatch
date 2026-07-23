#!/usr/bin/env bash
#SBATCH --job-name=nequip-summary
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/share/home/fushibo/MD_opt/nequip/log/summary-%j.log

: "${BENCH_SCRIPT_DIR:?BENCH_SCRIPT_DIR must be exported}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be exported}"

SCRIPT_DIR="${BENCH_SCRIPT_DIR}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_slurm_env.sh"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python "${SCRIPT_DIR}/summarize_results.py" "${OUTPUT_DIR}"
