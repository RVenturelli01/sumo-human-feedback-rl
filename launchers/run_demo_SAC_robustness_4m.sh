#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Runtime and logging
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/demo_sac_robustness_4m_alberto_no_collision}"
WANDB_ENTITY="andrea02polimi-politecnico-di-milano"
WANDB_PROJECT="${WANDB_PROJECT:-demo-sac-robustness-4m}"
WANDB_TAGS=null

# Environment
N_ENVS=4

# SAC
AGENT_LR=0.0001242983309370202
BUFFER_SIZE=300000 # transitions: 300k / 20k = 15 iterations 
LEARNING_STARTS=2000
AGENT_BATCH_SIZE=256
GAMMA=0.995
TAU=0.005
ENT_COEF=auto # auto 
TRAIN_FREQ=8
AGENT_GRADIENT_STEPS=64 # 16 * n_envs
AGENT_ARCH="[64,64]"
DEVICE=cpu

# Reward learning
# Historical: maxent, maxent_2, demo. Corrected: maxent_corrected, demo_corrected.
LOSS_TYPE=demo
RELABEL_REWARDS=true
NORMALIZE_AGENT_REWARD=false
REWARD_LR=0.0003
REWARD_GRADIENT_STEPS=20
EXPERT_BATCH_SIZE=64
MODEL_BATCH_SIZE=64
REWARD_L2=0.05
TEMPERATURE=1.0
# FRAGMENT_LENGTH=null
INITIAL_AGENT_TIMESTEPS=20000
EXPLORATION_FRAC=0.0
EXPLORATION_EPS=0.5
N_ENSEMBLES=3
REWARD_ARCH="[64,64]"
REWARD_ACTIVATION=tanh

# Training
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-4000000}"
TIMESTEPS_PER_ITERATION="${TIMESTEPS_PER_ITERATION:-20000}"
LOG_INTERVAL=100 # SAC log per episode, PPO per rollout.
CHECKPOINT_INTERVAL=10
IMITATION_DIAGNOSTICS_INTERVAL=5

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
  # algo.kwargs.fragment_length="$FRAGMENT_LENGTH"
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
  train.kwargs.imitation_diagnostics_interval="$IMITATION_DIAGNOSTICS_INTERVAL"
)

printf 'Running:'
printf ' %q' "${cmd[@]}" "$@"
printf '\n'
exec "${cmd[@]}" "$@"
