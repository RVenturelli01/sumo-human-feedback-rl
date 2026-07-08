#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Christiano preference PPO, labels_type=soft, on 3 seeds.
#
# Hyperparameters are taken from launchers/run_chri_PPO.sh (the canonical chri
# PPO soft recipe: fragment_length=1, fragmenter=random, gradient_steps_rew=300,
# batch_size_rew=500), NOT the "best-from-sweep" config.
#
# 1 config x 3 seeds = 3 runs, logged to a NEW W&B project, all in the same W&B
# *group* (test_chri_PPO.py sets group = "ppo_chri_soft seg=1") so a grouped plot
# with x-axis `agent/time/total_timesteps` shows the mean +/- std band.
#
# Pins each run to cores in [24, 47] and activates the `sumo-rlhf` conda env.
# With 3 runs on 24 cores the default is 8 cores/run (good for env n_envs=4).
#
# Run it detached so it survives disconnects, e.g.:
#     tmux new -s chrisoft 'bash launchers/run_chri_soft_3seeds.sh'
#   or
#     nohup bash launchers/run_chri_soft_3seeds.sh > chri_soft.out 2>&1 &
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---- conda -----------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-sumo-rlhf}"
for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
  if [[ -f "$c/etc/profile.d/conda.sh" ]]; then source "$c/etc/profile.d/conda.sh"; break; fi
done
conda activate "$CONDA_ENV"
echo "conda env: $(which python)"

# ---- experiment knobs ------------------------------------------------------
WANDB_ENTITY="${WANDB_ENTITY:-andrea02polimi-politecnico-di-milano}"
WANDB_PROJECT="${WANDB_PROJECT:-chri_soft_3seeds}"
SEEDS=(${SEEDS:-0 1 2})
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/chri_soft_3seeds}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

# ---- core pool -------------------------------------------------------------
CORE_START="${CORE_START:-24}"
CORE_END="${CORE_END:-47}"
CORES_PER_JOB="${CORES_PER_JOB:-8}"          # 3 runs x 8 cores = 24 (all parallel)
NCORES=$(( CORE_END - CORE_START + 1 ))
NSLOTS=$(( NCORES / CORES_PER_JOB ))
[[ $NSLOTS -lt 1 ]] && { echo "CORES_PER_JOB too large for the pool"; exit 1; }

export OMP_NUM_THREADS="$CORES_PER_JOB"
export MKL_NUM_THREADS="$CORES_PER_JOB"
export OPENBLAS_NUM_THREADS="$CORES_PER_JOB"
export NUMEXPR_NUM_THREADS="$CORES_PER_JOB"

# ---- chri PPO soft hyperparameters (from run_chri_PPO.sh) -------------------
BASELINE=false
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

OVERRIDES=(
  run.baseline="$BASELINE"
  wandb.entity="$WANDB_ENTITY"
  wandb.project="$WANDB_PROJECT"
  "wandb.tags=[chri,soft,3seeds]"
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

launch() {
  local seed="$1" slot="$2"
  local c0=$(( CORE_START + slot * CORES_PER_JOB ))
  local c1=$(( c0 + CORES_PER_JOB - 1 ))
  local log="$LOG_DIR/chri_soft_seed${seed}.log"
  echo "-> chri_soft seed=$seed cores ${c0}-${c1}  (log: $log)"
  taskset -c "${c0}-${c1}" python scripts/test_chri_PPO.py \
    "${OVERRIDES[@]}" \
    run.seed="$seed" \
    "run.output_dir=$OUTPUT_ROOT" \
    > "$log" 2>&1 &
}

total=${#SEEDS[@]}
echo "launching $total chri_soft runs on cores ${CORE_START}-${CORE_END} (${NSLOTS} slots, ${CORES_PER_JOB} core/job)"

# ---- run in waves of NSLOTS ------------------------------------------------
i=0
fail=0
while [ $i -lt $total ]; do
  pids=()
  seeds_wave=()
  for (( slot=0; slot<NSLOTS && i<total; slot++, i++ )); do
    launch "${SEEDS[$i]}" "$slot"
    pids+=($!)
    seeds_wave+=("${SEEDS[$i]}")
  done
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "!! FAILED: seed=${seeds_wave[$idx]} (see log)"; fail=$(( fail + 1 ))
    fi
  done
done

echo "done. $((total - fail))/$total succeeded. logs in $LOG_DIR"
[[ $fail -eq 0 ]]
