#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Experiment arm. Sets demo_mode/demo_weight/queries presets:
#   pref_only | demo_only | hybrid_v0 | demos_as_pref
MODE="${MODE:-hybrid_v0}"

# Runtime and logging
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hybrid_ppo/$MODE}"
WANDB_ENTITY="${WANDB_ENTITY:-andrea02polimi-politecnico-di-milano}"
WANDB_PROJECT="${WANDB_PROJECT:-preference+demonstration}"
WANDB_TAGS="${WANDB_TAGS:-[$MODE]}"

# Environment
N_ENVS="${N_ENVS:-4}"

# PPO
AGENT_LR="${AGENT_LR:-0.0003}"
N_STEPS="${N_STEPS:-1000}"
N_EPOCHS="${N_EPOCHS:-10}"
AGENT_BATCH_SIZE="${AGENT_BATCH_SIZE:-64}"
ENT_COEF="${ENT_COEF:-0}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
GAMMA="${GAMMA:-0.997}"
AGENT_ARCH="${AGENT_ARCH:-[64,64]}"
DEVICE="${DEVICE:-cpu}"

# Hybrid reward learning
# Demo side: maxent, maxent_2, demo, demo_corrected, maxent_corrected.
LOSS_TYPE="${LOSS_TYPE:-maxent_2}"
DEMO_MODE="${DEMO_MODE:-gcl}"
RELABEL_REWARDS="${RELABEL_REWARDS:-false}"
NORMALIZE_AGENT_REWARD="${NORMALIZE_AGENT_REWARD:-true}"
REWARD_LR="${REWARD_LR:-0.0003}"
REWARD_GRADIENT_STEPS="${REWARD_GRADIENT_STEPS:-300}"
EXPERT_BATCH_SIZE="${EXPERT_BATCH_SIZE:-64}"
MODEL_BATCH_SIZE="${MODEL_BATCH_SIZE:-64}"
PREF_BATCH_SIZE="${PREF_BATCH_SIZE:-500}"
REWARD_L2="${REWARD_L2:-0.0001}"
TEMPERATURE="${TEMPERATURE:-1.0}"
PREF_TEMPERATURE="${PREF_TEMPERATURE:-20.0}"
FRAGMENT_LENGTH="${FRAGMENT_LENGTH:-null}"
PREFERENCE_FRAGMENT_LENGTH="${PREFERENCE_FRAGMENT_LENGTH:-1}"
FRAGMENTER_TYPE="${FRAGMENTER_TYPE:-active}"
LABELS_TYPE="${LABELS_TYPE:-soft}"
COMPARISON_QUEUE_SIZE="${COMPARISON_QUEUE_SIZE:-1000000}"
TRAIN_COMPARISON_FRAC="${TRAIN_COMPARISON_FRAC:-0.8}"
TOTAL_QUERIES="${TOTAL_QUERIES:-10000}"
INITIAL_QUERIES="${INITIAL_QUERIES:-200}"
QUERY_SCHEDULE="${QUERY_SCHEDULE:-constant}"
DEMO_WEIGHT="${DEMO_WEIGHT:-1.0}"
MAX_BALANCE_SCALE="${MAX_BALANCE_SCALE:-100.0}"
BALANCE_EPS="${BALANCE_EPS:-1e-8}"
DEMO_PREF_PAIRS_PER_ITERATION="${DEMO_PREF_PAIRS_PER_ITERATION:-64}"
DEMO_PREF_BATCH_FRACTION="${DEMO_PREF_BATCH_FRACTION:-0.5}"

# ---- MODE presets (override the defaults above) -----------------------------
case "$MODE" in
  pref_only)
    DEMO_WEIGHT=0.0; DEMO_MODE=gcl ;;
  demo_only)
    TOTAL_QUERIES=0; INITIAL_QUERIES=0; DEMO_MODE=gcl ;;
  hybrid_v0)
    DEMO_MODE=gcl ;;
  demos_as_pref)
    DEMO_MODE=preferences ;;
  *) echo "unknown MODE: $MODE" >&2; exit 1 ;;
esac
INITIAL_AGENT_TIMESTEPS="${INITIAL_AGENT_TIMESTEPS:-20000}"
EXPLORATION_FRAC="${EXPLORATION_FRAC:-0.0}"
EXPLORATION_EPS="${EXPLORATION_EPS:-0.5}"
AGENT_LOG_TIMESTEP_INTERVAL="${AGENT_LOG_TIMESTEP_INTERVAL:-null}"
N_ENSEMBLES="${N_ENSEMBLES:-3}"
REWARD_ARCH="${REWARD_ARCH:-[128,128]}"
REWARD_ACTIVATION="${REWARD_ACTIVATION:-tanh}"
REWARD_ALPHA="${REWARD_ALPHA:-1}"

# Training
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
TIMESTEPS_PER_ITERATION="${TIMESTEPS_PER_ITERATION:-20000}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
SCATTER_INTERVAL="${SCATTER_INTERVAL:-null}"

cmd=(
  "$PYTHON_BIN" scripts/test_hybrid_PPO.py
  run.seed="$SEED"
  run.output_dir="$OUTPUT_DIR"
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
  algo.kwargs.loss_type="$LOSS_TYPE"
  algo.kwargs.demo_mode="$DEMO_MODE"
  algo.kwargs.pref_temperature="$PREF_TEMPERATURE"
  algo.kwargs.demo_pref_pairs_per_iteration="$DEMO_PREF_PAIRS_PER_ITERATION"
  algo.kwargs.demo_pref_batch_fraction="$DEMO_PREF_BATCH_FRACTION"
  algo.kwargs.relabel_rewards="$RELABEL_REWARDS"
  algo.kwargs.normalize_agent_reward="$NORMALIZE_AGENT_REWARD"
  algo.kwargs.lr_rew="$REWARD_LR"
  algo.kwargs.gradient_steps_rew="$REWARD_GRADIENT_STEPS"
  algo.kwargs.batch_size_expert="$EXPERT_BATCH_SIZE"
  algo.kwargs.batch_size_model="$MODEL_BATCH_SIZE"
  algo.kwargs.batch_size_pref="$PREF_BATCH_SIZE"
  algo.kwargs.l2_rew="$REWARD_L2"
  algo.kwargs.temperature="$TEMPERATURE"
  algo.kwargs.fragment_length="$FRAGMENT_LENGTH"
  algo.kwargs.preference_fragment_length="$PREFERENCE_FRAGMENT_LENGTH"
  algo.kwargs.fragmenter_type="$FRAGMENTER_TYPE"
  algo.kwargs.labels_type="$LABELS_TYPE"
  algo.kwargs.comparison_queue_size="$COMPARISON_QUEUE_SIZE"
  algo.kwargs.train_comparison_frac="$TRAIN_COMPARISON_FRAC"
  algo.kwargs.total_queries="$TOTAL_QUERIES"
  algo.kwargs.initial_queries="$INITIAL_QUERIES"
  algo.kwargs.query_schedule="$QUERY_SCHEDULE"
  algo.kwargs.demo_weight="$DEMO_WEIGHT"
  algo.kwargs.max_balance_scale="$MAX_BALANCE_SCALE"
  algo.kwargs.balance_eps="$BALANCE_EPS"
  algo.kwargs.initial_agent_timesteps="$INITIAL_AGENT_TIMESTEPS"
  algo.kwargs.exploration_frac="$EXPLORATION_FRAC"
  algo.kwargs.exploration_eps="$EXPLORATION_EPS"
  algo.kwargs.agent_log_timestep_interval="$AGENT_LOG_TIMESTEP_INTERVAL"
  algo.kwargs.reward_model_kwargs.n_ensembles="$N_ENSEMBLES"
  "algo.kwargs.reward_model_kwargs.net_arch=$REWARD_ARCH"
  algo.kwargs.reward_model_kwargs.activation_fn="$REWARD_ACTIVATION"
  algo.kwargs.reward_model_kwargs.alpha="$REWARD_ALPHA"
  train.kwargs.total_timesteps="$TOTAL_TIMESTEPS"
  train.kwargs.timesteps_per_iteration="$TIMESTEPS_PER_ITERATION"
  train.kwargs.log_interval="$LOG_INTERVAL"
  train.kwargs.checkpoint_interval="$CHECKPOINT_INTERVAL"
  train.kwargs.scatter_interval="$SCATTER_INTERVAL"
  train.kwargs.total_queries="$TOTAL_QUERIES"
)

printf 'Running:'
printf ' %q' "${cmd[@]}" "$@"
printf '\n'
exec "${cmd[@]}" "$@"
