#!/usr/bin/env bash
set -euo pipefail

cd /iris/u/tiangao/projects/robomimic

source /iris/u/tiangao/miniconda3/etc/profile.d/conda.sh
conda activate robomimic_py310

echo "[$(date)] env=${CONDA_DEFAULT_ENV}"
echo "[$(date)] running square_knn_commands_best_validation.txt"
bash robomimic/scripts/square_knn_commands_best_validation.txt

echo "[$(date)] running square_knn_commands.txt"
bash robomimic/scripts/square_knn_commands.txt

echo "[$(date)] done"
