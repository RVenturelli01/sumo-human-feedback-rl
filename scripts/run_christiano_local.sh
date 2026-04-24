#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

python scripts/train.py \
    algo=christiano \
    algo/agent=PPO \
    env.kwargs.ego=continuous \
    run.name=christiano_local \
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