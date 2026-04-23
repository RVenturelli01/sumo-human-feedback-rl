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
    wandb.enabled=true