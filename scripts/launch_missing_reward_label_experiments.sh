#!/usr/bin/env bash
# Launch missing reward-label experiments on the server.
#
# Defaults are conservative for server use:
#   - sequential execution, no background job pool
#   - CPU pinning to cores 27-47
#
# Presets:
#   priority - minimum missing runs for a stronger thesis story:
#              low-query soft labels, more Bernoulli queries, and one binary sanity point.
#   extended - extra temperature sweeps for soft and Bernoulli.
#
# Example:
#   MISSING_PRESET=priority ./scripts/launch_missing_reward_label_experiments.sh
#   MISSING_PRESET=extended OUTPUT_DIR=/work/fis3/... CORE_RANGE=27-47 ./scripts/launch_missing_reward_label_experiments.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MISSING_PRESET="${MISSING_PRESET:-extended}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/reward_label_experiments/missing_$(date +%Y%m%d_%H%M%S)}"
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
CORE_RANGE="${CORE_RANGE:-27-47}"
BERNOULLI_LABEL_TYPE="${BERNOULLI_LABEL_TYPE:-binary_bernoulli}"

declare -a EXPERIMENTS=()

add_run() {
  local seed="$1"
  local labels_type="$2"
  local total_queries="$3"
  local temperature="$4"
  local name_suffix="$5"
  EXPERIMENTS+=("${seed}|${labels_type}|${total_queries}|${temperature}|${name_suffix}")
}

case "$MISSING_PRESET" in
  priority)
    # Soft labels: prove sample efficiency below 2k queries.
    for seed in 0 1 2 3 4; do
      add_run "$seed" "soft" 500  20 "soft_500_T20"
      add_run "$seed" "soft" 1000 20 "soft_1k_T20"
    done

    # Complete the existing soft 2k/10k T20 groups from 3 to 5 seeds.
    for seed in 3 4; do
      add_run "$seed" "soft" 2000  20 "soft_2k_T20"
      add_run "$seed" "soft" 10000 20 "soft_10k_T20"
    done

    # Bernoulli: test whether it improves only with many more queries.
    for seed in 3 4; do
      add_run "$seed" "$BERNOULLI_LABEL_TYPE" 100000 20 "bernoulli_100k"
    done
    for seed in 0 1 2 3 4; do
      add_run "$seed" "$BERNOULLI_LABEL_TYPE" 200000 20 "bernoulli_200k"
      add_run "$seed" "$BERNOULLI_LABEL_TYPE" 500000 20 "bernoulli_500k"
    done

    # Binary sanity point: not essential beyond this, but useful to show that
    # more deterministic binary labels still do not recover reward scale.
    for seed in 0 1 2; do
      add_run "$seed" "binary" 100000 20 "deterministic_binary_100k"
    done
    ;;

  extended)
    # Soft temperature sweep around the current T=20 choice.
    for seed in 0 1 2; do
      for temp in 5 10 15 30 50; do
        add_run "$seed" "soft" 2000 "$temp" "soft_2k_T${temp}"
      done
      for temp in 10 15 30; do
        add_run "$seed" "soft" 10000 "$temp" "soft_10k_T${temp}"
      done
    done

    # Bernoulli temperature sweep at the largest existing query budget.
    for seed in 0 1 2; do
      for temp in 5 10 50; do
        add_run "$seed" "$BERNOULLI_LABEL_TYPE" 100000 "$temp" "bernoulli_100k_T${temp}"
      done
    done
    ;;

  *)
    echo "Unknown MISSING_PRESET=$MISSING_PRESET. Use priority or extended." >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

export WANDB_MODE

total_jobs=${#EXPERIMENTS[@]}
failures=0

echo "Missing reward-label experiments"
echo "Preset:      $MISSING_PRESET"
echo "Runs:        $total_jobs"
echo "Output dir:  $OUTPUT_DIR"
echo "Logs:        $LOG_DIR"
echo "WandB:       $WANDB_ENTITY / $WANDB_PROJECT ($WANDB_MODE)"
echo "Parallel:    disabled (runs execute sequentially)"
echo "CPU cores:   $CORE_RANGE"
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
    --config-path ../configs
    --config-name test_chri_PPO
    run.seed="$seed"
    run.output_dir="$OUTPUT_DIR"
    run.baseline=false
    "+run.name_suffix=$name_suffix"
    wandb.entity="$WANDB_ENTITY"
    wandb.project="$WANDB_PROJECT"
    "wandb.tags=[missing,$MISSING_PRESET,$labels_type,$name_suffix]"
    algo.kwargs.lr_rew="$LR_REW"
    algo.kwargs.gradient_steps_rew="$GRADIENT_STEPS_REW"
    algo.kwargs.batch_size_rew="$BATCH_SIZE_REW"
    algo.kwargs.fragment_length="$FRAGMENT_LENGTH"
    algo.kwargs.fragmenter_type="$FRAGMENTER_TYPE"
    algo.kwargs.labels_type="$labels_type"
    algo.kwargs.temperature="$temperature"
    algo.kwargs.reward_model_kwargs.n_ensembles="$N_ENSEMBLES"
    "algo.kwargs.reward_model_kwargs.net_arch=$NET_ARCH"
    algo.kwargs.initial_queries="$initial_queries"
    train.kwargs.total_timesteps="$TOTAL_TIMESTEPS"
    train.kwargs.total_queries="$total_queries"
    train.kwargs.timesteps_per_iteration="$TIMESTEPS_PER_ITERATION"
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

job_id=0
for spec in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r seed labels_type total_queries temperature name_suffix <<< "$spec"
  job_id=$((job_id + 1))
  if launch_one "$job_id" "$seed" "$labels_type" "$total_queries" "$temperature" "$name_suffix"; then
    echo "[$(date '+%H:%M:%S')] done  $job_id/$total_jobs: $name_suffix seed=$seed"
  else
    rc=$?
    failures=$((failures + 1))
    echo "[$(date '+%H:%M:%S')] fail  $job_id/$total_jobs: $name_suffix seed=$seed rc=$rc"
  fi
done

echo ""
if (( failures > 0 )); then
  echo "Finished with $failures failed run(s). Check logs in $LOG_DIR."
  exit 1
else
  echo "All missing experiments finished."
fi
echo "Checkpoints are under: $OUTPUT_DIR"
