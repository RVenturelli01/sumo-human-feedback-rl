#!/usr/bin/env bash
# Launch focused Christiano-style preference-learning runs for reward-model plots.
#
# Presets:
#   pilot  - one seed, small matrix to verify the pipeline and notebook plots.
#   paper  - three seeds and query-budget sweeps for the thesis comparison.
#
# Optional CPU pinning:
#   CORE_RANGE=36-47 ./scripts/launch_reward_label_experiments.sh
#
# Checkpoints are saved by scripts/test_chri_PPO.py under each run directory as:
#   checkpoint_XXXX/reward_model.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PRESET="${PRESET:-paper}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/reward_label_experiments/$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/_logs}"

WANDB_ENTITY="${WANDB_ENTITY:-andrea02polimi-politecnico-di-milano}"
WANDB_PROJECT="${WANDB_PROJECT:-reward_label_experiments}"
WANDB_MODE="${WANDB_MODE:-online}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
TIMESTEPS_PER_ITERATION="${TIMESTEPS_PER_ITERATION:-20000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"

LR_REW="${LR_REW:-3e-4}"
GRADIENT_STEPS_REW="${GRADIENT_STEPS_REW:-100}"
BATCH_SIZE_REW="${BATCH_SIZE_REW:-128}"
FRAGMENT_LENGTH="${FRAGMENT_LENGTH:-1}"
FRAGMENTER_TYPE="${FRAGMENTER_TYPE:-active}"
NET_ARCH="${NET_ARCH:-[32,32]}"
N_ENSEMBLES="${N_ENSEMBLES:-3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CORE_RANGE="${CORE_RANGE:-}"

declare -a SEEDS
declare -a LABEL_QUERY_TEMPERATURE

case "$PRESET" in
  pilot)
    SEEDS=(0)
    LABEL_QUERY_TEMPERATURE=(
      "binary|10000|20|deterministic_binary"
      "binary_bernoulli|10000|20|bernoulli_10k"
      "binary_bernoulli|50000|20|bernoulli_50k"
      "soft|2000|20|soft_2k_T20"
      "soft|10000|20|soft_10k_T20"
      "soft|10000|5|soft_10k_T5"
      "soft|10000|50|soft_10k_T50"
    )
    ;;
  paper)
    SEEDS=(0 1 2)
    LABEL_QUERY_TEMPERATURE=(
      "binary|10000|20|deterministic_binary_10k"
      "binary|50000|20|deterministic_binary_50k"
      "binary_bernoulli|10000|20|bernoulli_10k"
      "binary_bernoulli|50000|20|bernoulli_50k"
      "binary_bernoulli|100000|20|bernoulli_100k"
      "soft|2000|20|soft_2k_T20"
      "soft|5000|20|soft_5k_T20"
      "soft|10000|20|soft_10k_T20"
      "soft|10000|5|soft_10k_T5"
      "soft|10000|50|soft_10k_T50"
    )
    ;;
  *)
    echo "Unknown PRESET=$PRESET. Use PRESET=pilot or PRESET=paper." >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

export WANDB_MODE

submitted_jobs=0
total_jobs=$(( ${#SEEDS[@]} * ${#LABEL_QUERY_TEMPERATURE[@]} ))
failures=0

echo "Reward-label experiments"
echo "Preset:      $PRESET"
echo "Runs:        $total_jobs"
echo "Output dir:  $OUTPUT_DIR"
echo "Logs:        $LOG_DIR"
echo "WandB:       $WANDB_ENTITY / $WANDB_PROJECT ($WANDB_MODE)"
echo "Parallel:    disabled (runs execute sequentially)"
if [[ -n "$CORE_RANGE" ]]; then
  echo "CPU cores:   $CORE_RANGE"
else
  echo "CPU cores:   all available"
fi
echo ""

launch_one() {
  local job_id="$1"
  local seed="$2"
  local labels_type="$3"
  local total_queries="$4"
  local temperature="$5"
  local name_suffix="$6"
  local initial_queries
  local log_file
  local -a cmd

  initial_queries=$(( total_queries / 50 ))
  if (( initial_queries < 1 )); then
    initial_queries=1
  fi

  log_file="$LOG_DIR/$(printf '%03d' "$job_id")_${name_suffix}_seed${seed}.log"

  echo "[$(date '+%H:%M:%S')] start $job_id/$total_jobs: label=$labels_type queries=$total_queries temp=$temperature seed=$seed"

  cmd=(
    "$PYTHON_BIN" scripts/test_chri_PPO.py
    --config-path ../configs \
    --config-name test_chri_PPO \
    run.seed="$seed" \
    run.output_dir="$OUTPUT_DIR" \
    run.baseline=false \
    "+run.name_suffix=$name_suffix" \
    wandb.entity="$WANDB_ENTITY" \
    wandb.project="$WANDB_PROJECT" \
    "wandb.tags=[$PRESET,$labels_type,$name_suffix]" \
    algo.kwargs.lr_rew="$LR_REW" \
    algo.kwargs.gradient_steps_rew="$GRADIENT_STEPS_REW" \
    algo.kwargs.batch_size_rew="$BATCH_SIZE_REW" \
    algo.kwargs.fragment_length="$FRAGMENT_LENGTH" \
    algo.kwargs.fragmenter_type="$FRAGMENTER_TYPE" \
    algo.kwargs.labels_type="$labels_type" \
    algo.kwargs.temperature="$temperature" \
    algo.kwargs.reward_model_kwargs.n_ensembles="$N_ENSEMBLES" \
    "algo.kwargs.reward_model_kwargs.net_arch=$NET_ARCH" \
    algo.kwargs.initial_queries="$initial_queries" \
    train.kwargs.total_timesteps="$TOTAL_TIMESTEPS" \
    train.kwargs.total_queries="$total_queries" \
    train.kwargs.timesteps_per_iteration="$TIMESTEPS_PER_ITERATION" \
    train.kwargs.checkpoint_interval="$CHECKPOINT_INTERVAL"
  )

  if [[ -n "$CORE_RANGE" ]]; then
    if command -v taskset >/dev/null 2>&1; then
      taskset -c "$CORE_RANGE" "${cmd[@]}" > "$log_file" 2>&1
    else
      {
        echo "Warning: CORE_RANGE=$CORE_RANGE requested, but taskset is not available. Running without CPU pinning."
        "${cmd[@]}"
      } > "$log_file" 2>&1
    fi
  else
    "${cmd[@]}" > "$log_file" 2>&1
  fi
}

for seed in "${SEEDS[@]}"; do
  for spec in "${LABEL_QUERY_TEMPERATURE[@]}"; do
    IFS='|' read -r labels_type total_queries temperature name_suffix <<< "$spec"
    submitted_jobs=$((submitted_jobs + 1))
    if launch_one "$submitted_jobs" "$seed" "$labels_type" "$total_queries" "$temperature" "$name_suffix"; then
      echo "[$(date '+%H:%M:%S')] done  $submitted_jobs/$total_jobs: $name_suffix seed=$seed"
    else
      rc=$?
      failures=$((failures + 1))
      echo "[$(date '+%H:%M:%S')] fail  $submitted_jobs/$total_jobs: $name_suffix seed=$seed rc=$rc"
    fi
  done
done

echo ""
if (( failures > 0 )); then
  echo "Finished with $failures failed run(s). Check logs in $LOG_DIR."
  exit 1
else
  echo "All experiments finished."
fi
echo "Checkpoints are under: $OUTPUT_DIR"
