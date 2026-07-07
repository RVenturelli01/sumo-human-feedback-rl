#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Runtime and logging
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-[0,1,2]}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ppo_baseline}"
WANDB_ENTITY="${WANDB_ENTITY:-andrea02polimi-politecnico-di-milano}"
WANDB_PROJECT="${WANDB_PROJECT:-preference+demonstration}"
WANDB_TAGS="${WANDB_TAGS:-[ppo_baseline,true_reward]}"

# Environment
N_ENVS="${N_ENVS:-4}"
EGO="${EGO:-continuous}"
REWARD="${REWARD:-fast}"

# PPO baseline
AGENT_LR="${AGENT_LR:-0.0003}"
N_STEPS="${N_STEPS:-1000}"
N_EPOCHS="${N_EPOCHS:-10}"
AGENT_BATCH_SIZE="${AGENT_BATCH_SIZE:-64}"
ENT_COEF="${ENT_COEF:-0}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
GAMMA="${GAMMA:-0.997}"
AGENT_ARCH="${AGENT_ARCH:-[64,64]}"
DEVICE="${DEVICE:-cpu}"

# Training and evaluation
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_SEED="${EVAL_SEED:-1000}"

cmd=(
  "$PYTHON_BIN" scripts/train_ppo_baseline.py
  "run.seeds=$SEEDS"
  run.output_dir="$OUTPUT_DIR"
  wandb.entity="$WANDB_ENTITY"
  wandb.project="$WANDB_PROJECT"
  "wandb.tags=$WANDB_TAGS"
  env.n_envs="$N_ENVS"
  env.kwargs.ego="$EGO"
  env.kwargs.reward="$REWARD"
  agent.kwargs.learning_rate="$AGENT_LR"
  agent.kwargs.n_steps="$N_STEPS"
  agent.kwargs.n_epochs="$N_EPOCHS"
  agent.kwargs.batch_size="$AGENT_BATCH_SIZE"
  agent.kwargs.ent_coef="$ENT_COEF"
  agent.kwargs.gae_lambda="$GAE_LAMBDA"
  agent.kwargs.gamma="$GAMMA"
  "agent.kwargs.policy_kwargs.net_arch=$AGENT_ARCH"
  agent.kwargs.device="$DEVICE"
  train.kwargs.total_timesteps="$TOTAL_TIMESTEPS"
  train.kwargs.log_interval="$LOG_INTERVAL"
  eval.n_episodes="$EVAL_EPISODES"
  eval.seed="$EVAL_SEED"
)

printf 'Running:'
printf ' %q' "${cmd[@]}" "$@"
printf '\n'
exec "${cmd[@]}" "$@"
