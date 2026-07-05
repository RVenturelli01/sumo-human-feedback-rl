#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Runtime and logging
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED=0
OUTPUT_DIR="outputs/chri_ppo"
# Set baseline=true to train plain PPO on the true reward (preference baseline).
BASELINE=false
WANDB_ENTITY=null
WANDB_PROJECT="chri-ppo"
WANDB_TAGS=null

# Environment
N_ENVS=4

# PPO
AGENT_LR=0.0003
N_STEPS=1000
N_EPOCHS=10
AGENT_BATCH_SIZE=64
ENT_COEF=0
GAE_LAMBDA=0.95
GAMMA=0.997
AGENT_ARCH="[64,64]"
DEVICE=cpu

# Preference / reward learning
REWARD_LR=0.0003
REWARD_GRADIENT_STEPS=300
REWARD_BATCH_SIZE=500
REWARD_L2=0.0001
FRAGMENT_LENGTH=1
INITIAL_QUERIES=200
EXPLORATION_FRAC=0.0
EXPLORATION_EPS=0.5
QUERY_SCHEDULE=constant
COMPARISON_QUEUE_SIZE=1000000
TRAIN_COMPARISON_FRAC=0.8
TEMPERATURE=20.0
FRAGMENTER_TYPE=random
LABELS_TYPE=soft
N_ENSEMBLES=3
REWARD_ARCH="[128,128]"
REWARD_ACTIVATION=tanh
REWARD_ALPHA=1

# Training
TOTAL_TIMESTEPS=2000000
TOTAL_QUERIES=10000
TIMESTEPS_PER_ITERATION=20000
CHECKPOINT_INTERVAL=10

cmd=(
  "$PYTHON_BIN" scripts/test_chri_PPO.py
  run.seed="$SEED"
  run.output_dir="$OUTPUT_DIR"
  run.baseline="$BASELINE"
  wandb.entity="$WANDB_ENTITY"
  wandb.project="$WANDB_PROJECT"
  wandb.tags="$WANDB_TAGS"
  env.n_envs="$N_ENVS"
  agent.kwargs.learning_rate="$AGENT_LR"
  agent.kwargs.n_steps="$N_STEPS"
  agent.kwargs.n_epochs="$N_EPOCHS"
  agent.kwargs.batch_size="$AGENT_BATCH_SIZE"
  agent.kwargs.ent_coef="$ENT_COEF"
  agent.kwargs.gae_lambda="$GAE_LAMBDA"
  agent.kwargs.gamma="$GAMMA"
  "agent.kwargs.policy_kwargs.net_arch=$AGENT_ARCH"
  agent.kwargs.device="$DEVICE"
  algo.kwargs.lr_rew="$REWARD_LR"
  algo.kwargs.gradient_steps_rew="$REWARD_GRADIENT_STEPS"
  algo.kwargs.batch_size_rew="$REWARD_BATCH_SIZE"
  algo.kwargs.l2_rew="$REWARD_L2"
  "algo.kwargs.fragment_length=$FRAGMENT_LENGTH"
  algo.kwargs.initial_queries="$INITIAL_QUERIES"
  algo.kwargs.exploration_frac="$EXPLORATION_FRAC"
  algo.kwargs.exploration_eps="$EXPLORATION_EPS"
  algo.kwargs.query_schedule="$QUERY_SCHEDULE"
  algo.kwargs.comparison_queue_size="$COMPARISON_QUEUE_SIZE"
  algo.kwargs.train_comparison_frac="$TRAIN_COMPARISON_FRAC"
  algo.kwargs.temperature="$TEMPERATURE"
  algo.kwargs.fragmenter_type="$FRAGMENTER_TYPE"
  algo.kwargs.labels_type="$LABELS_TYPE"
  algo.kwargs.reward_model_kwargs.n_ensembles="$N_ENSEMBLES"
  "algo.kwargs.reward_model_kwargs.net_arch=$REWARD_ARCH"
  algo.kwargs.reward_model_kwargs.activation_fn="$REWARD_ACTIVATION"
  algo.kwargs.reward_model_kwargs.alpha="$REWARD_ALPHA"
  train.kwargs.total_timesteps="$TOTAL_TIMESTEPS"
  train.kwargs.total_queries="$TOTAL_QUERIES"
  train.kwargs.timesteps_per_iteration="$TIMESTEPS_PER_ITERATION"
  train.kwargs.checkpoint_interval="$CHECKPOINT_INTERVAL"
)

printf 'Running:'
printf ' %q' "${cmd[@]}" "$@"
printf '\n'
exec "${cmd[@]}" "$@"
