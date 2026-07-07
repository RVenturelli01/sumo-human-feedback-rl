#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Runtime and logging
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-[0,1,2]}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sac_baseline}"
WANDB_ENTITY="${WANDB_ENTITY:-andrea02polimi-politecnico-di-milano}"
WANDB_PROJECT="${WANDB_PROJECT:-preference+demonstration}"
WANDB_TAGS="${WANDB_TAGS:-[sac_baseline,true_reward]}"

# Environment
N_ENVS="${N_ENVS:-1}"
EGO="${EGO:-continuous}"
REWARD="${REWARD:-fast}"

# SAC baseline
AGENT_LR="${AGENT_LR:-0.0001242983309370202}"
BUFFER_SIZE="${BUFFER_SIZE:-300000}"
LEARNING_STARTS="${LEARNING_STARTS:-2000}"
AGENT_BATCH_SIZE="${AGENT_BATCH_SIZE:-256}"
GAMMA="${GAMMA:-0.997}"
TAU="${TAU:-0.00311}"
ENT_COEF="${ENT_COEF:-auto}"
TRAIN_FREQ="${TRAIN_FREQ:-8}"
AGENT_GRADIENT_STEPS="${AGENT_GRADIENT_STEPS:-8}"
AGENT_ARCH="${AGENT_ARCH:-[64,64]}"
DEVICE="${DEVICE:-cpu}"

# Training and evaluation
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
AGENT_LOG_TIMESTEP_INTERVAL="${AGENT_LOG_TIMESTEP_INTERVAL:-null}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_SEED="${EVAL_SEED:-1000}"

cmd=(
  "$PYTHON_BIN" scripts/train_sac_baseline.py
  "run.seeds=$SEEDS"
  run.output_dir="$OUTPUT_DIR"
  wandb.entity="$WANDB_ENTITY"
  wandb.project="$WANDB_PROJECT"
  "wandb.tags=$WANDB_TAGS"
  env.n_envs="$N_ENVS"
  env.kwargs.ego="$EGO"
  env.kwargs.reward="$REWARD"
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
  train.kwargs.total_timesteps="$TOTAL_TIMESTEPS"
  train.kwargs.log_interval="$LOG_INTERVAL"
  train.agent_log_timestep_interval="$AGENT_LOG_TIMESTEP_INTERVAL"
  eval.n_episodes="$EVAL_EPISODES"
  eval.seed="$EVAL_SEED"
)

printf 'Running:'
printf ' %q' "${cmd[@]}" "$@"
printf '\n'
exec "${cmd[@]}" "$@"
