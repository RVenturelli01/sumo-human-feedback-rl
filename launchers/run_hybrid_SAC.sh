#!/usr/bin/env bash
set -euo pipefail

# Hybrid reward learning: expert demonstrations (GCL) + human preferences (BT).
#
# Select the experimental mode with the MODE env var (default: hybrid):
#   MODE=pref_only ./launchers/run_hybrid_SAC.sh
#   MODE=demo_only ./launchers/run_hybrid_SAC.sh
#   MODE=hybrid    ./launchers/run_hybrid_SAC.sh
# Any extra CLI args are forwarded to the Hydra command as overrides.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-hybrid}"

# Mode -> loss weights.
case "$MODE" in
  pref_only) LAMBDA_DEMO=0.0; LAMBDA_PREF=1.0 ;;
  demo_only) LAMBDA_DEMO=1.0; LAMBDA_PREF=0.0 ;;
  hybrid)    LAMBDA_DEMO=1.0; LAMBDA_PREF=1.0 ;;
  *) echo "Unknown MODE=$MODE (use pref_only | demo_only | hybrid)" >&2; exit 1 ;;
esac

# Runtime and logging
SEED=0
OUTPUT_DIR="outputs/hybrid_sac_${MODE}"
WANDB_ENTITY=null
WANDB_PROJECT="hybrid-sac"
WANDB_TAGS=null

# Environment
N_ENVS=1

# SAC
AGENT_LR=0.0003
BUFFER_SIZE=100000
LEARNING_STARTS=2000
AGENT_BATCH_SIZE=256
GAMMA=0.997
TAU=0.005
ENT_COEF=auto
TRAIN_FREQ=1
AGENT_GRADIENT_STEPS=1
AGENT_ARCH="[64,64]"
DEVICE=cpu

# Reward learning (shared model)
LOSS_TYPE=maxent_2
RELABEL_REWARDS=true
NORMALIZE_AGENT_REWARD=true
REWARD_LR=0.001
REWARD_GRADIENT_STEPS=100
EXPERT_BATCH_SIZE=32
MODEL_BATCH_SIZE=64
# 0.0001 (not 0.01): with preferences active the BT gradient is weaker than a
# 0.01 weight decay, which collapses the reward net to a constant. Matches the
# working Christiano recipe.
REWARD_L2=0.0001
# Equalize GCL vs BT gradient norms on the shared reward net.
BALANCE_REWARD_GRADS=true
MAX_GRAD_BALANCE=1e5
TEMPERATURE=1.0
INITIAL_AGENT_TIMESTEPS=10000
EXPLORATION_FRAC=0.0
EXPLORATION_EPS=0.5
N_ENSEMBLES=3
REWARD_ARCH="[32,32]"
REWARD_ACTIVATION=tanh

# Preference branch
PREF_FRAGMENTER=random
PREF_LABELS=bernoulli
PREF_FRAGMENT_LENGTH=1
PREF_TEMPERATURE=20.0
PREF_BATCH_SIZE=32
QUERIES_PER_ITERATION=200
PREF_TRAIN_FRAC=0.8

# Training
TOTAL_TIMESTEPS=2000000
TIMESTEPS_PER_ITERATION=10000
LOG_INTERVAL=100
CHECKPOINT_INTERVAL=10
IMITATION_DIAGNOSTICS_INTERVAL=10

cmd=(
  "$PYTHON_BIN" scripts/test_hybrid_SAC.py
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
  algo.kwargs.lambda_demo="$LAMBDA_DEMO"
  algo.kwargs.lambda_pref="$LAMBDA_PREF"
  algo.kwargs.balance_reward_grads="$BALANCE_REWARD_GRADS"
  algo.kwargs.max_grad_balance="$MAX_GRAD_BALANCE"
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
  algo.kwargs.pref_fragmenter_type="$PREF_FRAGMENTER"
  algo.kwargs.pref_labels_type="$PREF_LABELS"
  algo.kwargs.pref_fragment_length="$PREF_FRAGMENT_LENGTH"
  algo.kwargs.pref_temperature="$PREF_TEMPERATURE"
  algo.kwargs.pref_batch_size="$PREF_BATCH_SIZE"
  algo.kwargs.queries_per_iteration="$QUERIES_PER_ITERATION"
  algo.kwargs.pref_train_frac="$PREF_TRAIN_FRAC"
  algo.kwargs.reward_model_kwargs.n_ensembles="$N_ENSEMBLES"
  "algo.kwargs.reward_model_kwargs.net_arch=$REWARD_ARCH"
  algo.kwargs.reward_model_kwargs.activation_fn="$REWARD_ACTIVATION"
  train.kwargs.total_timesteps="$TOTAL_TIMESTEPS"
  train.kwargs.timesteps_per_iteration="$TIMESTEPS_PER_ITERATION"
  train.kwargs.log_interval="$LOG_INTERVAL"
  train.kwargs.checkpoint_interval="$CHECKPOINT_INTERVAL"
  train.kwargs.imitation_diagnostics_interval="$IMITATION_DIAGNOSTICS_INTERVAL"
)

printf 'Running (MODE=%s):' "$MODE"
printf ' %q' "${cmd[@]}" "$@"
printf '\n'
exec "${cmd[@]}" "$@"
