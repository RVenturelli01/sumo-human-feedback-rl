#!/usr/bin/env bash
# 2D sweep over preference-fragment (segment) length x label family for
# Christiano-style preference learning, holding the rest of the config fixed.
#
# Motivation:
#   The reward_label_checkpoint_analysis notebook only has runs at
#   fragment_length=1. This script extends the study to longer segments (and the
#   whole-episode case) AND crosses it with the label families, so the notebook
#   can separate two effects that interact along the length axis:
#     - binary            : T-independent -> isolates the pure segment-length
#                           (temporal credit-assignment) effect.
#     - binary_bernoulli  : realistic noisy human (BTL sampled hard labels) ->
#                           reveals the collapse to coin-flip labels as segments
#                           grow and sigma((r1-r2)/T) -> 0.5.
#     - soft              : idealized oracle (continuous probability target) ->
#                           smooth upper bound, degrades without label-flip noise.
#
# Fixed base configuration (chosen to match the existing length=1 runs):
#   total_queries = 10000, temperature = 20, seed = 0
#   fragmenter_type = active (same as the other launchers)
#
# Segment lengths swept (override via SEGMENT_LENGTHS):
#   1 5 10 25 50 episode
#   where "episode" maps to Hydra `algo.kwargs.fragment_length=null`, which the
#   fragmenter (common/fragmenters.py:_sample_fragments) interprets as
#   "use the whole trajectory as one fragment". fragment_avg_reward divides by
#   the true fragment length, so any length (incl. full episode) is valid.
#
# Parallelism / CPU pinning (same model as launch_missing_reward_label_experiments.sh):
#   CORE_RANGE=START-END and CORES_PER_RUN define the number of parallel slots:
#     n_slots = (END - START + 1) / CORES_PER_RUN
#   Default CORE_RANGE=0-8, CORES_PER_RUN=9 -> a single slot (sequential).
#   On machines without `taskset` (e.g. macOS) runs execute without pinning.
#
# Examples:
#   ./scripts/launch_segment_length_sweep.sh                                     # 3 families x 6 lengths x seed 0 = 18 runs
#   LABEL_TYPES="binary soft" ./scripts/launch_segment_length_sweep.sh           # drop bernoulli
#   SEGMENT_LENGTHS="1 10 50 episode" ./scripts/launch_segment_length_sweep.sh
#   CORE_RANGE=21-47 CORES_PER_RUN=9 ./scripts/launch_segment_length_sweep.sh    # 3 parallel slots
#   SEED_LIST="0 1 2" ./scripts/launch_segment_length_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/reward_label_experiments/seglen_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/_logs}"

WANDB_ENTITY="${WANDB_ENTITY:-andrea02polimi-politecnico-di-milano}"
WANDB_PROJECT="${WANDB_PROJECT:-reward_label_experiments}"
WANDB_MODE="${WANDB_MODE:-online}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
TIMESTEPS_PER_ITERATION="${TIMESTEPS_PER_ITERATION:-20000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"

# ---- Fixed base configuration (held constant across the sweep) -------------
LR_REW="${LR_REW:-3e-4}"
GRADIENT_STEPS_REW="${GRADIENT_STEPS_REW:-100}"
BATCH_SIZE_REW="${BATCH_SIZE_REW:-128}"
FRAGMENTER_TYPE="${FRAGMENTER_TYPE:-active}"
NET_ARCH="${NET_ARCH:-[32,32]}"
N_ENSEMBLES="${N_ENSEMBLES:-3}"
LABEL_TYPES="${LABEL_TYPES:-binary binary_bernoulli soft}"
TOTAL_QUERIES="${TOTAL_QUERIES:-10000}"
TEMPERATURE="${TEMPERATURE:-20}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---- Sweep axis -----------------------------------------------------------
SEGMENT_LENGTHS="${SEGMENT_LENGTHS:-1 5 10 25 50 episode}"
SEED_LIST="${SEED_LIST:-0}"

# ---- Parallel slots -------------------------------------------------------
CORE_RANGE="${CORE_RANGE:-0-8}"
CORES_PER_RUN="${CORES_PER_RUN:-9}"

declare -a EXPERIMENTS=()
declare -a SLOT_RANGES=()
declare -a SLOT_PIDS=()
declare -a SLOT_LABELS=()

build_experiments() {
  local seed labels_type seglen frag_value seglen_tag name_suffix
  for seed in $SEED_LIST; do
    for labels_type in $LABEL_TYPES; do
      for seglen in $SEGMENT_LENGTHS; do
        if [[ "$seglen" == "episode" || "$seglen" == "null" || "$seglen" == "full" ]]; then
          frag_value="null"
          seglen_tag="episode"
        elif [[ "$seglen" =~ ^[0-9]+$ ]]; then
          frag_value="$seglen"
          seglen_tag="$seglen"
        else
          echo "Invalid segment length '$seglen' (use a positive integer or 'episode')." >&2
          exit 2
        fi
        name_suffix="${labels_type}_seglen_${seglen_tag}"
        # spec: seed | labels_type | fragment_length_override | name_suffix
        EXPERIMENTS+=("${seed}|${labels_type}|${frag_value}|${name_suffix}")
      done
    done
  done
}

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

make_command() {
  local seed="$1"
  local labels_type="$2"
  local frag_value="$3"
  local name_suffix="$4"
  local initial_queries

  initial_queries=$(( TOTAL_QUERIES / 50 ))
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
    "wandb.tags=[seglen_sweep,$labels_type,seglen_${frag_value},$name_suffix]" \
    algo.kwargs.lr_rew="$LR_REW" \
    algo.kwargs.gradient_steps_rew="$GRADIENT_STEPS_REW" \
    algo.kwargs.batch_size_rew="$BATCH_SIZE_REW" \
    algo.kwargs.fragment_length="$frag_value" \
    algo.kwargs.fragmenter_type="$FRAGMENTER_TYPE" \
    algo.kwargs.labels_type="$labels_type" \
    algo.kwargs.temperature="$TEMPERATURE" \
    algo.kwargs.reward_model_kwargs.n_ensembles="$N_ENSEMBLES" \
    "algo.kwargs.reward_model_kwargs.net_arch=$NET_ARCH" \
    algo.kwargs.initial_queries="$initial_queries" \
    train.kwargs.total_timesteps="$TOTAL_TIMESTEPS" \
    train.kwargs.total_queries="$TOTAL_QUERIES" \
    train.kwargs.timesteps_per_iteration="$TIMESTEPS_PER_ITERATION" \
    train.kwargs.checkpoint_interval="$CHECKPOINT_INTERVAL"
}

run_experiment() {
  local slot="$1"
  local job_id="$2"
  local spec="$3"
  local core_range="${SLOT_RANGES[$slot]}"
  local seed labels_type frag_value name_suffix log_file cmd

  IFS='|' read -r seed labels_type frag_value name_suffix <<< "$spec"
  log_file="$LOG_DIR/$(printf '%03d' "$job_id")_${name_suffix}_seed${seed}.log"

  echo "[$(date '+%H:%M:%S')] start job=$job_id/$total_jobs slot=$slot cores=$core_range label=$labels_type fragment_length=$frag_value seed=$seed"

  cmd="$(make_command "$seed" "$labels_type" "$frag_value" "$name_suffix")"
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

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"
build_experiments
build_slot_ranges

export WANDB_MODE

total_jobs=${#EXPERIMENTS[@]}
failures=0
completed=0
submitted=0

echo "Segment-length x label-family sweep (Christiano preference learning)"
echo "Fixed base:  queries=$TOTAL_QUERIES  T=$TEMPERATURE  fragmenter=$FRAGMENTER_TYPE"
echo "Labels:      $LABEL_TYPES"
echo "Lengths:     $SEGMENT_LENGTHS"
echo "Seeds:       $SEED_LIST"
echo "Runs:        $total_jobs"
echo "Output dir:  $OUTPUT_DIR"
echo "Logs:        $LOG_DIR"
echo "WandB:       $WANDB_ENTITY / $WANDB_PROJECT ($WANDB_MODE)"
echo "Parallel:    ${#SLOT_RANGES[@]} slot(s) x $CORES_PER_RUN cores"
echo "CPU slots:   ${SLOT_RANGES[*]}"
echo ""

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
  echo "All segment-length runs finished."
fi
echo "Checkpoints are under: $OUTPUT_DIR"
