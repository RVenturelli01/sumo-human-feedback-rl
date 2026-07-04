#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${CONDA_ENV_NAME:-sumo-rlhf}"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}" || conda activate "${HOME}/miniconda3/envs/${ENV_NAME}"
fi

python scripts/train_guided_cost_learning.py "$@"
