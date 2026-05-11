#!/usr/bin/env bash
set -euo pipefail

cd /iris/u/tiangao/projects/robomimic
PROJECT_ROOT="$(pwd)"

JOBS="${JOBS:-4}"
COMMAND_LIST="${COMMAND_LIST:-/tmp/square_knn_commands_all.txt}"
LOG_DIR="${LOG_DIR:-logs/square_knn_parallel_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${PROJECT_ROOT}/${LOG_DIR}"

source /iris/u/tiangao/miniconda3/etc/profile.d/conda.sh
conda activate robomimic_py310

mkdir -p vis/knn_json/square "${LOG_DIR}"

{
    rg --no-heading --no-line-number '^python ' \
        robomimic/scripts/square_knn_commands_best_validation.txt
    rg --no-heading --no-line-number '^python ' \
        robomimic/scripts/square_knn_commands.txt
} > "${COMMAND_LIST}"

echo "[$(date)] env=${CONDA_DEFAULT_ENV}"
echo "[$(date)] commands=$(wc -l < "${COMMAND_LIST}") jobs=${JOBS}"
echo "[$(date)] logs=${LOG_DIR}"

run_one() {
    local index="$1"
    local command="$2"
    local output_path
    local log_path

    cd "${PROJECT_ROOT}"
    output_path="$(awk '{
        for (i = 1; i <= NF; i++) {
            if ($i == "--output") {
                print $(i + 1)
                exit
            }
        }
    }' <<< "${command}")"
    log_path="${LOG_DIR}/$(printf "%03d" "${index}").log"

    if [[ -n "${output_path}" && -s "${output_path}" ]]; then
        echo "[$(date)] skip ${index}: ${output_path}" | tee "${log_path}"
        return 0
    fi

    echo "[$(date)] start ${index}: ${command}" > "${log_path}"
    if /bin/bash -c "${command}" >> "${log_path}" 2>&1; then
        echo "[$(date)] done ${index}" >> "${log_path}"
        return 0
    fi

    echo "[$(date)] failed ${index}" >> "${log_path}"
    return 1
}

export LOG_DIR
export PROJECT_ROOT
export -f run_one

seq "$(wc -l < "${COMMAND_LIST}")" \
    | xargs -I {} -P "${JOBS}" /bin/bash -c '
        index="$1"
        command="$(sed -n "${index}p" "$2")"
        run_one "${index}" "${command}"
    ' _ {} "${COMMAND_LIST}"

echo "[$(date)] all done"
