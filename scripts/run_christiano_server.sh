#!/bin/bash

set -euo pipefail

# ── configurazione ─────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="sumo-rlhf"
OUTPUT_DIR="/storage/fis3/christiano-exp"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$REPO_ROOT"

mkdir -p "$OUTPUT_DIR"

taskset -c 36-47 python scripts/train.py \
    algo=christiano \
    algo/agent=PPO \
    env.kwargs.ego=continuous \
    run.name=seg_1 \
    run.seed=0 \
    run.output_dir="$REPO_ROOT/outputs" \
    wandb.enabled=true \
    algo.kwargs.n_ensembles_rew=2 \
    algo.kwargs.lr_rew=0.0005930355149468689 \
    algo.kwargs.batch_size_rew=128 \
    algo.kwargs.n_ephochs_rew=3 \
    algo.kwargs.n_iterations=120 \
    algo.kwargs.train_comparison_frac=0.2 \
    algo.kwargs.fragment_length=1 \
    algo.kwargs.transition_oversampling=1.0 \
    algo.kwargs.initial_comparison_frac=0.1 \
    algo.kwargs.initial_epoch_multiplier=2.0 \
    algo.kwargs.query_schedule=constant \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.total_comparisons=20000

taskset -c 36-47 python scripts/train.py \
    algo=christiano \
    algo/agent=PPO \
    env.kwargs.ego=continuous \
    run.name=seg_10 \
    run.seed=0 \
    run.output_dir="$REPO_ROOT/outputs" \
    wandb.enabled=true \
    algo.kwargs.n_ensembles_rew=2 \
    algo.kwargs.lr_rew=0.0005930355149468689 \
    algo.kwargs.batch_size_rew=128 \
    algo.kwargs.n_ephochs_rew=3 \
    algo.kwargs.n_iterations=120 \
    algo.kwargs.train_comparison_frac=0.2 \
    algo.kwargs.fragment_length=10 \
    algo.kwargs.transition_oversampling=1.0 \
    algo.kwargs.initial_comparison_frac=0.1 \
    algo.kwargs.initial_epoch_multiplier=2.0 \
    algo.kwargs.query_schedule=constant \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.total_comparisons=20000

taskset -c 36-47 python scripts/train.py \
    algo=christiano \
    algo/agent=PPO \
    env.kwargs.ego=continuous \
    run.name=seg_20 \
    run.seed=0 \
    run.output_dir="$REPO_ROOT/outputs" \
    wandb.enabled=true \
    algo.kwargs.n_ensembles_rew=2 \
    algo.kwargs.lr_rew=0.0005930355149468689 \
    algo.kwargs.batch_size_rew=128 \
    algo.kwargs.n_ephochs_rew=3 \
    algo.kwargs.n_iterations=120 \
    algo.kwargs.train_comparison_frac=0.2 \
    algo.kwargs.fragment_length=20 \
    algo.kwargs.transition_oversampling=1.0 \
    algo.kwargs.initial_comparison_frac=0.1 \
    algo.kwargs.initial_epoch_multiplier=2.0 \
    algo.kwargs.query_schedule=constant \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.total_comparisons=20000

taskset -c 36-47 python scripts/train.py \
    algo=christiano \
    algo/agent=PPO \
    env.kwargs.ego=continuous \
    run.name=seg_30 \
    run.seed=0 \
    run.output_dir="$REPO_ROOT/outputs" \
    wandb.enabled=true \
    algo.kwargs.n_ensembles_rew=2 \
    algo.kwargs.lr_rew=0.0005930355149468689 \
    algo.kwargs.batch_size_rew=128 \
    algo.kwargs.n_ephochs_rew=3 \
    algo.kwargs.n_iterations=120 \
    algo.kwargs.train_comparison_frac=0.2 \
    algo.kwargs.fragment_length=30 \
    algo.kwargs.transition_oversampling=1.0 \
    algo.kwargs.initial_comparison_frac=0.1 \
    algo.kwargs.initial_epoch_multiplier=2.0 \
    algo.kwargs.query_schedule=constant \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.total_comparisons=20000

taskset -c 36-47 python scripts/train.py \
    algo=christiano \
    algo/agent=PPO \
    env.kwargs.ego=continuous \
    run.name=seg_40 \
    run.seed=0 \
    run.output_dir="$REPO_ROOT/outputs" \
    wandb.enabled=true \
    algo.kwargs.n_ensembles_rew=2 \
    algo.kwargs.lr_rew=0.0005930355149468689 \
    algo.kwargs.batch_size_rew=128 \
    algo.kwargs.n_ephochs_rew=3 \
    algo.kwargs.n_iterations=120 \
    algo.kwargs.train_comparison_frac=0.2 \
    algo.kwargs.fragment_length=40 \
    algo.kwargs.transition_oversampling=1.0 \
    algo.kwargs.initial_comparison_frac=0.1 \
    algo.kwargs.initial_epoch_multiplier=2.0 \
    algo.kwargs.query_schedule=constant \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.total_comparisons=20000

taskset -c 36-47 python scripts/train.py \
    algo=christiano \
    algo/agent=PPO \
    env.kwargs.ego=continuous \
    run.name=seg_50 \
    run.seed=0 \
    run.output_dir="$REPO_ROOT/outputs" \
    wandb.enabled=true \
    algo.kwargs.n_ensembles_rew=2 \
    algo.kwargs.lr_rew=0.0005930355149468689 \
    algo.kwargs.batch_size_rew=128 \
    algo.kwargs.n_ephochs_rew=3 \
    algo.kwargs.n_iterations=120 \
    algo.kwargs.train_comparison_frac=0.2 \
    algo.kwargs.fragment_length=50 \
    algo.kwargs.transition_oversampling=1.0 \
    algo.kwargs.initial_comparison_frac=0.1 \
    algo.kwargs.initial_epoch_multiplier=2.0 \
    algo.kwargs.query_schedule=constant \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.total_comparisons=20000