#!/usr/bin/env bash
# Launch the remaining reward-label experiments on the server.
#
# Default preset "all" includes:
#   - soft q=500,1000 at T=20 for sample efficiency
#   - soft temperature sweeps
#   - Bernoulli temperature sweeps
#   - Bernoulli q=200k,500k at T=20
#   - binary q=100k
#   - PPO true-reward baseline
#
# CPU pinning:
#   CORE_RANGE=21-47 and CORES_PER_RUN=9 gives three parallel slots:
#     21-29, 30-38, 39-47
#
# Example:
#   ./scripts/launch_missing_reward_label_experiments.sh
#   CORE_RANGE=21-47 CORES_PER_RUN=9 MISSING_PRESET=all ./scripts/launch_missing_reward_label_experiments.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MISSING_PRESET="${MISSING_PRESET:-all}"
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

CORE_RANGE="${CORE_RANGE:-21-47}"
CORES_PER_RUN="${CORES_PER_RUN:-9}"
SEED_LIST="${SEED_LIST:-0 1 2}"
BERNOULLI_LABEL_TYPE="${BERNOULLI_LABEL_TYPE:-binary_bernoulli}"

declare -a EXPERIMENTS=()
declare -a SLOT_RANGES=()
declare -a SLOT_PIDS=()
declare -a SLOT_LABELS=()

add_reward_run() {
  local seed="$1"
  local labels_type="$2"
  local total_queries="$3"
  local temperature="$4"
  local name_suffix="$5"
  EXPERIMENTS+=("reward|${seed}|${labels_type}|${total_queries}|${temperature}|${name_suffix}")
}

add_baseline_run() {
  local seed="$1"
  local name_suffix="$2"
  EXPERIMENTS+=("baseline|${seed}|baseline|0|20|${name_suffix}")
}

build_all_experiments() {
  local seed temp

  for seed in $SEED_LIST; do
    # Soft sample efficiency.
    add_reward_run "$seed" "soft" 500 20 "soft_500_T20"
    add_reward_run "$seed" "soft" 1000 20 "soft_1k_T20"

    # Soft temperature sweep at low-query and final-query regimes.
    for temp in 5 10 15 30 50; do
      add_reward_run "$seed" "soft" 2000 "$temp" "soft_2k_T${temp}"
    done
    for temp in 10 15 30; do
      add_reward_run "$seed" "soft" 10000 "$temp" "soft_10k_T${temp}"
    done

    # Bernoulli temperature sweep.
    for temp in 5 10 50; do
      add_reward_run "$seed" "$BERNOULLI_LABEL_TYPE" 100000 "$temp" "bernoulli_100k_T${temp}"
    done

    # Bernoulli many-query regime.
    add_reward_run "$seed" "$BERNOULLI_LABEL_TYPE" 200000 20 "bernoulli_200k_T20"
    add_reward_run "$seed" "$BERNOULLI_LABEL_TYPE" 500000 20 "bernoulli_500k_T20"

    # Binary sanity check.
    add_reward_run "$seed" "binary" 100000 20 "deterministic_binary_100k"

    # PPO trained directly on the true environment reward.
    add_baseline_run "$seed" "ppo_true_reward_baseline"
  done
}

build_priority_experiments() {
  local seed
  for seed in $SEED_LIST; do
    add_reward_run "$seed" "soft" 500 20 "soft_500_T20"
    add_reward_run "$seed" "soft" 1000 20 "soft_1k_T20"
    add_reward_run "$seed" "$BERNOULLI_LABEL_TYPE" 200000 20 "bernoulli_200k_T20"
    add_reward_run "$seed" "$BERNOULLI_LABEL_TYPE" 500000 20 "bernoulli_500k_T20"
    add_reward_run "$seed" "binary" 100000 20 "deterministic_binary_100k"
    add_baseline_run "$seed" "ppo_true_reward_baseline"
  done
}

build_extended_experiments() {
  local seed temp
  for seed in $SEED_LIST; do
    for temp in 5 10 15 30 50; do
      add_reward_run "$seed" "soft" 2000 "$temp" "soft_2k_T${temp}"
    done
    for temp in 10 15 30; do
      add_reward_run "$seed" "soft" 10000 "$temp" "soft_10k_T${temp}"
    done
    for temp in 5 10 50; do
      add_reward_run "$seed" "$BERNOULLI_LABEL_TYPE" 100000 "$temp" "bernoulli_100k_T${temp}"
    done
  done
}

case "$MISSING_PRESET" in
  all)
    build_all_experiments
    ;;
  priority)
    build_priority_experiments
    ;;
  extended)
    build_extended_experiments
    ;;
  *)
    echo "Unknown MISSING_PRESET=$MISSING_PRESET. Use all, priority, or extended." >&2
    exit 2
    ;;
esac

build_slot_ranges() {
  local range_start range_end n_cores n_slots slot start end
  if [[ ! "$CORE_RANGE" =~ ^[0-9]+-[0-9]+$ ]]; then
    echo "CORE_RANGE must have form START-END, got: $CORE_RANGE" >&2
    exit 2
  fi

  range_start="${CORE_RANGE%-*}"
  range_end="${CORE_RANGE#*-}"
  if (( range_end < range_start )); then
    echo "Invalid CORE_RANGE=$CORE_RANGE" >&2
    exit 2
  fi

  n_cores=$(( range_end - range_start + 1 ))
  n_slots=$(( n_cores / CORES_PER_RUN ))
  if (( n_slots < 1 )); then
    echo "CORE_RANGE=$CORE_RANGE has only $n_cores cores, fewer than CORES_PER_RUN=$CORES_PER_RUN" >&2
    exit 2
  fi

  SLOT_RANGES=()
  for ((slot = 0; slot < n_slots; slot++)); do
    start=$(( range_start + slot * CORES_PER_RUN ))
    end=$(( start + CORES_PER_RUN - 1 ))
    SLOT_RANGES+=("${start}-${end}")
    SLOT_PIDS+=(0)
    SLOT_LABELS+=("")
  done
}

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"
build_slot_ranges

export WANDB_MODE

total_jobs=${#EXPERIMENTS[@]}
failures=0
completed=0
submitted=0

echo "Missing reward-label experiments"
echo "Preset:      $MISSING_PRESET"
echo "Runs:        $total_jobs"
echo "Seeds:       $SEED_LIST"
echo "Output dir:  $OUTPUT_DIR"
echo "Logs:        $LOG_DIR"
echo "WandB:       $WANDB_ENTITY / $WANDB_PROJECT ($WANDB_MODE)"
echo "Parallel:    ${#SLOT_RANGES[@]} slot(s) x $CORES_PER_RUN cores"
echo "CPU slots:   ${SLOT_RANGES[*]}"
echo ""

make_command() {
  local kind="$1"
  local seed="$2"
  local labels_type="$3"
  local total_queries="$4"
  local temperature="$5"
  local name_suffix="$6"
  local initial_queries

  if [[ "$kind" == "baseline" ]]; then
    printf '%q ' \
      "$PYTHON_BIN" scripts/test_chri_PPO.py \
      --config-path ../configs \
      --config-name test_chri_PPO \
      run.seed="$seed" \
      run.output_dir="$OUTPUT_DIR" \
      run.baseline=true \
      "+run.name_suffix=$name_suffix" \
      wandb.entity="$WANDB_ENTITY" \
      wandb.project="$WANDB_PROJECT" \
      "wandb.tags=[missing,$MISSING_PRESET,baseline,$name_suffix]" \
      train.kwargs.total_timesteps="$TOTAL_TIMESTEPS" \
      train.kwargs.timesteps_per_iteration="$TIMESTEPS_PER_ITERATION"
  else
    initial_queries=$(( total_queries / 50 ))
    if (( initial_queries < 1 )); then
      initial_queries=1
    fi

    printf '%q ' \
      "$PYTHON_BIN" scripts/test_chri_PPO.py \
      --config-path ../configs \
      --config-name test_chri_PPO \
      run.seed="$seed" \
      run.output_dir="$OUTPUT_DIR" \
      run.baseline=false \
      "+run.name_suffix=$name_suffix" \
      wandb.entity="$WANDB_ENTITY" \
      wandb.project="$WANDB_PROJECT" \
      "wandb.tags=[missing,$MISSING_PRESET,$labels_type,$name_suffix]" \
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
  fi
}

run_experiment() {
  local slot="$1"
  local job_id="$2"
  local spec="$3"
  local core_range="${SLOT_RANGES[$slot]}"
  local kind seed labels_type total_queries temperature name_suffix
  local log_file cmd

  IFS='|' read -r kind seed labels_type total_queries temperature name_suffix <<< "$spec"
  log_file="$LOG_DIR/$(printf '%03d' "$job_id")_${name_suffix}_seed${seed}.log"

  echo "[$(date '+%H:%M:%S')] start job=$job_id/$total_jobs slot=$slot cores=$core_range kind=$kind label=$labels_type q=$total_queries T=$temperature seed=$seed"

  cmd="$(make_command "$kind" "$seed" "$labels_type" "$total_queries" "$temperature" "$name_suffix")"
  if command -v taskset >/dev/null 2>&1; then
    bash -lc "taskset -c '$core_range' $cmd" > "$log_file" 2>&1
  else
    {
      echo "Warning: taskset is not available. Running without CPU pinning."
      bash -lc "$cmd"
    } > "$log_file" 2>&1
  fi
}

reap_slot_if_done() {
  local slot="$1"
  local pid="${SLOT_PIDS[$slot]}"
  local rc

  if (( pid == 0 )); then
    return 0
  fi

  if kill -0 "$pid" 2>/dev/null; then
    return 1
  fi

  if wait "$pid"; then
    rc=0
  else
    rc=$?
  fi

  completed=$((completed + 1))
  if (( rc == 0 )); then
    echo "[$(date '+%H:%M:%S')] done  ${SLOT_LABELS[$slot]} completed=$completed/$total_jobs"
  else
    failures=$((failures + 1))
    echo "[$(date '+%H:%M:%S')] fail  ${SLOT_LABELS[$slot]} rc=$rc completed=$completed/$total_jobs"
  fi

  SLOT_PIDS[$slot]=0
  SLOT_LABELS[$slot]=""
  return 0
}

while (( completed < total_jobs )); do
  for slot in "${!SLOT_RANGES[@]}"; do
    reap_slot_if_done "$slot" || true

    if (( SLOT_PIDS[$slot] == 0 && submitted < total_jobs )); then
      submitted=$((submitted + 1))
      spec="${EXPERIMENTS[$((submitted - 1))]}"
      SLOT_LABELS[$slot]="job=$submitted ${spec}"
      run_experiment "$slot" "$submitted" "$spec" &
      SLOT_PIDS[$slot]=$!
    fi
  done

  if (( completed < total_jobs )); then
    sleep 5
  fi
done

echo ""
if (( failures > 0 )); then
  echo "Finished with $failures failed run(s). Check logs in $LOG_DIR."
  exit 1
else
  echo "All requested experiments finished."
fi
echo "Checkpoints are under: $OUTPUT_DIR"
