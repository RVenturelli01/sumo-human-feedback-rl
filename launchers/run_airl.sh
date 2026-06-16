#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── WandB ──────────────────────────────────────────────────────────────────
ENTITY="andrea02polimi-politecnico-di-milano"
PROJECT="PPO_AIRL"

# ── Paths ──────────────────────────────────────────────────────────────────
OUTPUT_DIR="$REPO_ROOT/outputs"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

# ── run.* ──────────────────────────────────────────────────────────────────
SEED=0

# ── env.* ──────────────────────────────────────────────────────────────────
ENV_ID="HighwayEgo-v0"
N_ENVS=4
ENV_EGO="continuous"
ENV_REWARD="fast"

# ── agent.kwargs.* ─────────────────────────────────────────────────────────
POLICY="MlpPolicy"
N_STEPS=1000
N_EPOCHS=10
LR=0.0003
BATCH_SIZE=64
ENT_COEF=0.01
GAE_LAMBDA=0.95
GAMMA=0.997
NET_ARCH_AGENT="[64,64]"
DEVICE="cpu"

# ── algo.kwargs.* ──────────────────────────────────────────────────────────
LR_REW=0.0003
GRADIENT_STEPS_REW=200
BATCH_SIZE_EXPERT=64
BATCH_SIZE_MODEL=64
L2_REW=0.01
FRAGMENT_LENGTH=1
TEMPERATURE=1.0
INITIAL_QUERIES=100
EXPLORATION_FRAC=0.0
EXPLORATION_EPS=0.0
QUERY_SCHEDULE="constant"
N_ENSEMBLES=3
NET_ARCH_REW="[64,64]"
ACTIVATION_FN="tanh"

# ── train.kwargs.* ─────────────────────────────────────────────────────────
TOTAL_TIMESTEPS=2000000
TOTAL_QUERIES=10000
TIMESTEPS_PER_ITERATION=20000
CHECKPOINT_INTERVAL=10

# ── Setup ──────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

echo "============================================================"
echo " PPO AIRL run"
echo " seed=$SEED  entity=$ENTITY  project=$PROJECT"
echo " output=$OUTPUT_DIR"
echo " Started at: $(date)"
echo "============================================================"
echo ""

"$PYTHON_BIN" scripts/test_airl_PPO.py \
    --config-path ../configs \
    --config-name test_airl_PPO \
    run.seed="$SEED" \
    run.output_dir="$OUTPUT_DIR" \
    wandb.entity="$ENTITY" \
    wandb.project="$PROJECT" \
    wandb.tags=null \
    env.id="$ENV_ID" \
    env.n_envs="$N_ENVS" \
    env.kwargs.ego="$ENV_EGO" \
    env.kwargs.reward="$ENV_REWARD" \
    agent.kwargs.policy="$POLICY" \
    agent.kwargs.n_steps="$N_STEPS" \
    agent.kwargs.n_epochs="$N_EPOCHS" \
    agent.kwargs.learning_rate="$LR" \
    agent.kwargs.batch_size="$BATCH_SIZE" \
    agent.kwargs.ent_coef="$ENT_COEF" \
    agent.kwargs.gae_lambda="$GAE_LAMBDA" \
    agent.kwargs.gamma="$GAMMA" \
    "agent.kwargs.policy_kwargs.net_arch=$NET_ARCH_AGENT" \
    agent.kwargs.device="$DEVICE" \
    algo.kwargs.lr_rew="$LR_REW" \
    algo.kwargs.gradient_steps_rew="$GRADIENT_STEPS_REW" \
    algo.kwargs.batch_size_expert="$BATCH_SIZE_EXPERT" \
    algo.kwargs.batch_size_model="$BATCH_SIZE_MODEL" \
    algo.kwargs.l2_rew="$L2_REW" \
    algo.kwargs.fragment_length="$FRAGMENT_LENGTH" \
    algo.kwargs.temperature="$TEMPERATURE" \
    algo.kwargs.initial_queries="$INITIAL_QUERIES" \
    algo.kwargs.exploration_frac="$EXPLORATION_FRAC" \
    algo.kwargs.exploration_eps="$EXPLORATION_EPS" \
    algo.kwargs.query_schedule="$QUERY_SCHEDULE" \
    algo.kwargs.grad_clip_rew=null \
    algo.kwargs.reward_model_kwargs.n_ensembles="$N_ENSEMBLES" \
    "algo.kwargs.reward_model_kwargs.net_arch=$NET_ARCH_REW" \
    algo.kwargs.reward_model_kwargs.activation_fn="$ACTIVATION_FN" \
    algo.kwargs.reward_model_kwargs.gamma="$GAMMA" \
    train.kwargs.total_timesteps="$TOTAL_TIMESTEPS" \
    train.kwargs.total_queries="$TOTAL_QUERIES" \
    train.kwargs.timesteps_per_iteration="$TIMESTEPS_PER_ITERATION" \
    train.kwargs.checkpoint_interval="$CHECKPOINT_INTERVAL"
