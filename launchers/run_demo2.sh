#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Runtime and logging
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED=0
OUTPUT_DIR="outputs/demo2_weighted_bc"
WANDB_ENTITY="andrea02polimi-politecnico-di-milano"
WANDB_PROJECT="demo2-weighted-bc"
WANDB_TAGS=null

# Environment
N_ENVS=4

# Standalone policy (tanh-squashed Gaussian, trained by weighted BC)
POLICY_ARCH="[64,64]"
POLICY_ACTIVATION=tanh
POLICY_LOG_STD_INIT=-0.5
POLICY_LOG_STD_BOUNDS="[-0.9,2.0]"
DEVICE=cpu

# Reward learning (batch-mixed partition: maxent_2 or maxent)
LOSS_TYPE=maxent_2
REWARD_LR=0.0003
REWARD_GRADIENT_STEPS=30
EXPERT_BATCH_SIZE=96
MODEL_BATCH_SIZE=32
REWARD_L2=0.03
TEMPERATURE=1.0
INITIAL_REWARD_TIMESTEPS=20000
N_ENSEMBLES=3
REWARD_ARCH="[64,64]"
REWARD_ACTIVATION=tanh

# Weighted behavior cloning (section 13.2/13.3)
POLICY_LR=0.001
POLICY_GRADIENT_STEPS=20
WEIGHT_TEMPERATURE=1.0
STANDARDIZE_WEIGHTS=true
ENT_COEF=0.03

# Training
TOTAL_TIMESTEPS=4000000
TIMESTEPS_PER_ITERATION=20000
LOG_INTERVAL=100
CHECKPOINT_INTERVAL=10
IMITATION_DIAGNOSTICS_INTERVAL=5

cmd=(
  "$PYTHON_BIN" scripts/test_demo2.py
  run.seed="$SEED"
  run.output_dir="$OUTPUT_DIR"
  wandb.entity="$WANDB_ENTITY"
  wandb.project="$WANDB_PROJECT"
  wandb.tags="$WANDB_TAGS"
  env.n_envs="$N_ENVS"
  algo.kwargs.loss_type="$LOSS_TYPE"
  algo.kwargs.lr_rew="$REWARD_LR"
  algo.kwargs.gradient_steps_rew="$REWARD_GRADIENT_STEPS"
  algo.kwargs.batch_size_expert="$EXPERT_BATCH_SIZE"
  algo.kwargs.batch_size_model="$MODEL_BATCH_SIZE"
  algo.kwargs.l2_rew="$REWARD_L2"
  algo.kwargs.temperature="$TEMPERATURE"
  algo.kwargs.lr_policy="$POLICY_LR"
  algo.kwargs.gradient_steps_policy="$POLICY_GRADIENT_STEPS"
  algo.kwargs.weight_temperature="$WEIGHT_TEMPERATURE"
  algo.kwargs.standardize_weights="$STANDARDIZE_WEIGHTS"
  algo.kwargs.ent_coef="$ENT_COEF"
  "algo.kwargs.policy_kwargs.net_arch=$POLICY_ARCH"
  algo.kwargs.policy_kwargs.activation_fn="$POLICY_ACTIVATION"
  algo.kwargs.policy_kwargs.log_std_init="$POLICY_LOG_STD_INIT"
  "algo.kwargs.policy_kwargs.log_std_bounds=$POLICY_LOG_STD_BOUNDS"
  algo.kwargs.initial_reward_timesteps="$INITIAL_REWARD_TIMESTEPS"
  algo.kwargs.reward_model_kwargs.n_ensembles="$N_ENSEMBLES"
  "algo.kwargs.reward_model_kwargs.net_arch=$REWARD_ARCH"
  algo.kwargs.reward_model_kwargs.activation_fn="$REWARD_ACTIVATION"
  algo.kwargs.device="$DEVICE"
  train.kwargs.total_timesteps="$TOTAL_TIMESTEPS"
  train.kwargs.timesteps_per_iteration="$TIMESTEPS_PER_ITERATION"
  train.kwargs.log_interval="$LOG_INTERVAL"
  train.kwargs.checkpoint_interval="$CHECKPOINT_INTERVAL"
  train.kwargs.imitation_diagnostics_interval="$IMITATION_DIAGNOSTICS_INTERVAL"
)

printf 'Running:'
printf ' %q' "${cmd[@]}" "$@"
printf '\n'
exec "${cmd[@]}" "$@"
