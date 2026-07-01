#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Runtime and logging
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED=0
OUTPUT_DIR="outputs/dagger_debug"
WANDB_ENTITY="andrea02polimi-politecnico-di-milano"
WANDB_PROJECT="dagger-debug"
WANDB_TAGS=null

# Environment
N_ENVS=1
EGO=continuous
REWARD=fast

# Expert (rule-based: FastPolicy-v0 | models: sac-fast, dqn-fast, ppo-fast)
EXPERT_ID=ppo-fast

# BCPolicy
AGENT_ARCH="[64,64]"

# DAgger / behaviour cloning
BC_EPOCHS=50
BC_BATCH_SIZE=256
BC_LR=1e-3
N_EVAL_EPISODES=50
N_EXPERT_ROLLOUT_EPISODES=50
BETA_DECAY=0.7

# Training
N_ROUNDS=80
NUM_EPISODES=50

cmd=(
  "$PYTHON_BIN" scripts/test_dagger.py
  run.seed="$SEED"
  run.output_dir="$OUTPUT_DIR"
  wandb.entity="$WANDB_ENTITY"
  wandb.project="$WANDB_PROJECT"
  wandb.tags="$WANDB_TAGS"
  env.n_envs="$N_ENVS"
  env.kwargs.ego="$EGO"
  env.kwargs.reward="$REWARD"
  expert.id="$EXPERT_ID"
  "agent.kwargs.policy_kwargs.net_arch=$AGENT_ARCH"
  algo.kwargs.bc_epochs="$BC_EPOCHS"
  algo.kwargs.bc_batch_size="$BC_BATCH_SIZE"
  algo.kwargs.bc_lr="$BC_LR"
  algo.kwargs.n_eval_episodes="$N_EVAL_EPISODES"
  algo.kwargs.n_expert_rollout_episodes="$N_EXPERT_ROLLOUT_EPISODES"
  algo.kwargs.beta_decay="$BETA_DECAY"
  train.kwargs.n_rounds="$N_ROUNDS"
  train.kwargs.num_episodes="$NUM_EPISODES"
)

printf 'Running:'
printf ' %q' "${cmd[@]}" "$@"
printf '\n'
exec "${cmd[@]}" "$@"
