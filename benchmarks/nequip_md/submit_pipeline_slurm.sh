#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

validation_submission="$(bash "${SCRIPT_DIR}/submit_validation_slurm.sh")"
# --parsable commonly returns JOBID or JOBID;cluster.
validation_job_id="${validation_submission%%;*}"
if ! [[ "${validation_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse validation job ID from: ${validation_submission}" >&2
    exit 2
fi

echo "Validation array job: ${validation_job_id}"
echo "Submitting performance array with afterok dependency"
VALIDATION_JOB_ID="${validation_job_id}" \
    bash "${SCRIPT_DIR}/submit_benchmarks_slurm.sh"
