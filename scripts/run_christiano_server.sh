#!/bin/bash

set -euo pipefail

# ── configurazione ─────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="sumo-rlhf"
OUTPUT_DIR="/storage/fis3/christiano-exp-various_seg_lens"

SEG_LENS=(1 2 5 10 20 30 40 50 100)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$REPO_ROOT"

mkdir -p "$OUTPUT_DIR"

for SEG_LEN in "${SEG_LENS[@]}"; do
    RUN_NAME="seg_${SEG_LEN}"

    echo "===================================================="
    echo "Running experiment: $RUN_NAME"
    echo "fragment_length=$SEG_LEN"
    echo "===================================================="

    echo "y" | taskset -c 36-47 python scripts/train.py \
        algo=christiano \
        algo/agent=PPO \
        env.kwargs.ego=continuous \
        run.name="$RUN_NAME" \
        run.seed=0 \
        run.output_dir="$REPO_ROOT/outputs" \
        wandb.enabled=true \
        algo.kwargs.n_ensembles_rew=3 \
        algo.kwargs.lr_rew=3e-4 \
        algo.kwargs.batch_size_rew=64 \
        algo.kwargs.n_ephochs_rew=1 \
        algo.kwargs.n_iterations=50 \
        algo.kwargs.train_comparison_frac=0.8 \
        algo.kwargs.fragment_length="$SEG_LEN" \
        algo.kwargs.transition_oversampling=10.0 \
        algo.kwargs.initial_comparison_frac=0.1 \
        algo.kwargs.initial_epoch_multiplier=1.0 \
        algo.kwargs.query_schedule=constant \
        algo.train.kwargs.total_timesteps=1000000 \
        algo.train.kwargs.total_comparisons=2500
done