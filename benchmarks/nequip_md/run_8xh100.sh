#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SYSTEMS_FILE="${SYSTEMS_FILE:-${SCRIPT_DIR}/systems.tsv}"
MODEL_PACKAGE="${MODEL_PACKAGE:-/share/home/fushibo/NequIP-OAM-L-0.1.nequip.zip}"
COMPILED_MODEL="${COMPILED_MODEL:-${REPO_ROOT}/benchmark_artifacts/NequIP-OAM-L-0.1-ase-oeq-no-cg.nequip.pt2}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmark_results/nequip_e0_e1_b0_b1}"
VALIDATION_DIR="${VALIDATION_DIR:-${OUTPUT_DIR}/validation}"
REQUIRE_VALIDATION="${REQUIRE_VALIDATION:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NGPUS="${NGPUS:-8}"
REPEATS="${REPEATS:-5}"
STEPS="${STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
TIMESTEP_FS="${TIMESTEP_FS:-1.0}"
TEMPERATURE_K="${TEMPERATURE_K:-300.0}"
VELOCITY_MODE="${VELOCITY_MODE:-maxwell}"
SEED="${SEED:-20260722}"

artifact_hash() {
    local artifact="$1"
    if [[ -f "${artifact}.sha256" ]]; then
        awk 'NR == 1 {print $1}' "${artifact}.sha256"
    elif command -v sha256sum >/dev/null 2>&1 && [[ -f "${artifact}" ]]; then
        sha256sum "${artifact}" | awk '{print $1}'
    else
        echo ""
    fi
}

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
if ! [[ "${NGPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NGPUS must be a positive integer, got: ${NGPUS}" >&2
    exit 2
fi
if ! [[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPEATS must be a positive integer, got: ${REPEATS}" >&2
    exit 2
fi

mapfile -t SYSTEMS < <(
    awk -F '\t' 'NF >= 2 && $1 !~ /^[[:space:]]*#/ && $1 !~ /^[[:space:]]*$/ {print $1 "\t" $2}' "${SYSTEMS_FILE}"
)
if (( ${#SYSTEMS[@]} == 0 )); then
    echo "No systems found in ${SYSTEMS_FILE}" >&2
    exit 2
fi
for entry in "${SYSTEMS[@]}"; do
    IFS=$'\t' read -r label structure <<< "${entry}"
    if [[ ! -f "${structure}" ]]; then
        echo "Structure for ${label} not found: ${structure}" >&2
        exit 2
    fi
    if [[ "${REQUIRE_VALIDATION}" == "1" && ! -f "${VALIDATION_DIR}/${label}.json" ]]; then
        echo "Validation result missing for ${label}: ${VALIDATION_DIR}/${label}.json" >&2
        echo "Run validate_all_modes.sh before the performance benchmark." >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_DIR}/json" "${OUTPUT_DIR}/logs"
cd "${REPO_ROOT}"

# Hash once before workers start. Hashing every 78 MB artifact in every process
# would create shared-filesystem traffic while other GPUs are being timed.
MODEL_PACKAGE_SHA256="${MODEL_PACKAGE_SHA256:-$(artifact_hash "${MODEL_PACKAGE}")}"
COMPILED_MODEL_SHA256="${COMPILED_MODEL_SHA256:-$(artifact_hash "${COMPILED_MODEL}")}"

# A group is one (system, repeat) pair.  All four modes in a group run
# sequentially on the same physical GPU.  Different groups use different GPUs.
TASK_GROUPS=()
for ((repeat = 0; repeat < REPEATS; repeat++)); do
    for system_index in "${!SYSTEMS[@]}"; do
        IFS=$'\t' read -r label structure <<< "${SYSTEMS[system_index]}"
        TASK_GROUPS+=("${system_index}"$'\t'"${repeat}"$'\t'"${label}"$'\t'"${structure}")
    done
done

mode_order() {
    case "$(( $1 % 4 ))" in
        0) echo "E0 E1 B0 B1" ;;
        1) echo "E1 B0 B1 E0" ;;
        2) echo "B0 B1 E0 E1" ;;
        3) echo "B1 E0 E1 B0" ;;
    esac
}

run_worker() {
    local gpu="$1"
    local group_index system_index repeat label structure order mode
    local output_path log_path validation_path cpuset_var cpuset selected_hash
    local -a bench_command

    cpuset_var="CPUSET_${gpu}"
    cpuset="${!cpuset_var:-}"
    if [[ -n "${cpuset}" ]] && ! command -v taskset >/dev/null 2>&1; then
        echo "${cpuset_var} is set, but taskset is unavailable" >&2
        return 2
    fi

    for ((group_index = gpu; group_index < ${#TASK_GROUPS[@]}; group_index += NGPUS)); do
        IFS=$'\t' read -r system_index repeat label structure <<< "${TASK_GROUPS[group_index]}"
        order="$(mode_order "$((repeat + system_index))")"
        for mode in ${order}; do
            output_path="${OUTPUT_DIR}/json/${label}.${mode}.repeat${repeat}.json"
            log_path="${OUTPUT_DIR}/logs/${label}.${mode}.repeat${repeat}.gpu${gpu}.log"
            validation_path="${VALIDATION_DIR}/${label}.json"
            if [[ "${mode}" == "E0" || "${mode}" == "E1" ]]; then
                selected_hash="${MODEL_PACKAGE_SHA256}"
            else
                selected_hash="${COMPILED_MODEL_SHA256}"
            fi
            bench_command=(
                "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_md.py"
                --mode "${mode}"
                --structure "${structure}"
                --system-label "${label}"
                --model-package "${MODEL_PACKAGE}"
                --compiled-model "${COMPILED_MODEL}"
                --steps "${STEPS}"
                --warmup-steps "${WARMUP_STEPS}"
                --timestep-fs "${TIMESTEP_FS}"
                --temperature-k "${TEMPERATURE_K}"
                --velocity-mode "${VELOCITY_MODE}"
                --seed "${SEED}"
                --repeat "${repeat}"
                --output "${output_path}"
            )
            if [[ -f "${validation_path}" ]]; then
                bench_command+=(--validation-result "${validation_path}")
            fi
            if [[ -n "${selected_hash}" ]]; then
                bench_command+=(--model-sha256 "${selected_hash}")
            else
                bench_command+=(--skip-model-hash)
            fi
            echo "GPU ${gpu}: ${label} repeat=${repeat} mode=${mode}"
            if [[ -n "${cpuset}" ]]; then
                CUDA_VISIBLE_DEVICES="${gpu}" taskset -c "${cpuset}" "${bench_command[@]}" >"${log_path}" 2>&1
            else
                CUDA_VISIBLE_DEVICES="${gpu}" "${bench_command[@]}" >"${log_path}" 2>&1
            fi
        done
    done
}

worker_pids=()
for ((gpu = 0; gpu < NGPUS; gpu++)); do
    run_worker "${gpu}" &
    worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if (( status != 0 )); then
    echo "At least one GPU worker failed; inspect ${OUTPUT_DIR}/logs" >&2
    exit "${status}"
fi
echo "All benchmarks completed: ${OUTPUT_DIR}"
