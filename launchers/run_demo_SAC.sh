#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Runtime and logging
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED=0
OUTPUT_DIR="outputs/demo_sac_debug"
WANDB_ENTITY="andrea02polimi-politecnico-di-milano"
WANDB_PROJECT="demo-sac"
WANDB_TAGS=null

# Environment
N_ENVS=1

# SAC
AGENT_LR=0.0001242983309370202
BUFFER_SIZE=300000
LEARNING_STARTS=2000
AGENT_BATCH_SIZE=256
GAMMA=0.999
TAU=0.0031102923983872435
ENT_COEF=auto
TRAIN_FREQ=8
AGENT_GRADIENT_STEPS=8
AGENT_ARCH="[64,64]"
DEVICE=cpu

# Reward learning
# Historical: maxent, maxent_2, demo. Corrected: maxent_corrected, demo_corrected.
LOSS_TYPE=maxent_2
RELABEL_REWARDS=true
NORMALIZE_AGENT_REWARD=true
REWARD_LR=0.0001
REWARD_GRADIENT_STEPS=200
EXPERT_BATCH_SIZE=64
MODEL_BATCH_SIZE=64
REWARD_L2=0.0
TEMPERATURE=1.0
INITIAL_AGENT_TIMESTEPS=20000
EXPLORATION_FRAC=0.0
EXPLORATION_EPS=0.5
N_ENSEMBLES=3
REWARD_ARCH="[64,64]"
REWARD_ACTIVATION=tanh

# Training
TOTAL_TIMESTEPS=2000000
TIMESTEPS_PER_ITERATION=20000
LOG_INTERVAL=100
CHECKPOINT_INTERVAL=10

cmd=(
  "$PYTHON_BIN" scripts/test_demo_SAC.py
  run.seed="$SEED"
  run.output_dir="$OUTPUT_DIR"
  wandb.entity="$WANDB_ENTITY"
  wandb.project="$WANDB_PROJECT"
  wandb.tags="$WANDB_TAGS"
  env.n_envs="$N_ENVS"
  agent.kwargs.learning_rate="$AGENT_LR"
  agent.kwargs.buffer_size="$BUFFER_SIZE"
  agent.kwargs.learning_starts="$LEARNING_STARTS"
  agent.kwargs.batch_size="$AGENT_BATCH_SIZE"
  agent.kwargs.gamma="$GAMMA"
  agent.kwargs.tau="$TAU"
  agent.kwargs.ent_coef="$ENT_COEF"
  agent.kwargs.train_freq="$TRAIN_FREQ"
  agent.kwargs.gradient_steps="$AGENT_GRADIENT_STEPS"
  "agent.kwargs.policy_kwargs.net_arch=$AGENT_ARCH"
  agent.kwargs.device="$DEVICE"
  algo.kwargs.loss_type="$LOSS_TYPE"
  algo.kwargs.relabel_rewards="$RELABEL_REWARDS"
  algo.kwargs.normalize_agent_reward="$NORMALIZE_AGENT_REWARD"
  algo.kwargs.lr_rew="$REWARD_LR"
  algo.kwargs.gradient_steps_rew="$REWARD_GRADIENT_STEPS"
  algo.kwargs.batch_size_expert="$EXPERT_BATCH_SIZE"
  algo.kwargs.batch_size_model="$MODEL_BATCH_SIZE"
  algo.kwargs.l2_rew="$REWARD_L2"
  algo.kwargs.temperature="$TEMPERATURE"
  algo.kwargs.initial_agent_timesteps="$INITIAL_AGENT_TIMESTEPS"
  algo.kwargs.exploration_frac="$EXPLORATION_FRAC"
  algo.kwargs.exploration_eps="$EXPLORATION_EPS"
  algo.kwargs.reward_model_kwargs.n_ensembles="$N_ENSEMBLES"
  "algo.kwargs.reward_model_kwargs.net_arch=$REWARD_ARCH"
  algo.kwargs.reward_model_kwargs.activation_fn="$REWARD_ACTIVATION"
  train.kwargs.total_timesteps="$TOTAL_TIMESTEPS"
  train.kwargs.timesteps_per_iteration="$TIMESTEPS_PER_ITERATION"
  train.kwargs.log_interval="$LOG_INTERVAL"
  train.kwargs.checkpoint_interval="$CHECKPOINT_INTERVAL"
)

printf 'Running:'
printf ' %q' "${cmd[@]}" "$@"
printf '\n'
exec "${cmd[@]}" "$@"
