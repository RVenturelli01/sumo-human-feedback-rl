#!/usr/bin/env bash
# Sample-budget curves for one baseline arm with its best Optuna config.
#
# A plain 1D grid (no optimizer needed): budget levels x seeds at the tuned
# configuration. Pref arms sweep total_queries; demo arms sweep
# n_expert_trajectories. Minimum viable budget = the smallest level whose
# across-seed mean of sweep/mean_fast_return AND sweep/success_rate stays
# >= 90% of the largest level, with the next level up also passing.
#
# Usage:
#   ./launchers/run_budget_curves.sh <arm>
#
# Env knobs: LEVELS (space-separated budgets; defaults below), SEEDS="1 2 3",
#   TOTAL_TIMESTEPS=2000000, WANDB_PROJECT=tuning-thesis, FIRST_CORE=33,
#   MAX_PARALLEL=5, STORAGE=outputs/optuna/journal.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ARM="${1:?usage: run_budget_curves.sh <arm>}"

case "$ARM" in
  pref_soft)      AXIS="pref";  DEFAULT_LEVELS="10000 5000 2000 1000 500" ;;
  # Bernoulli labels are sampled (noisy): the viable range sits ~1-2 orders of
  # magnitude above soft (validated in reward_label_experiments, rescaled to 1M).
  pref_bernoulli) AXIS="pref";  DEFAULT_LEVELS="250000 100000 50000 25000 10000" ;;
  demo_1|demo_2)  AXIS="demo";  DEFAULT_LEVELS="2723 1000 500 200 100 50" ;;
  *) echo "budget curves are defined for the 4 baseline arms, got: $ARM" >&2; exit 1 ;;
esac

LEVELS="${LEVELS:-$DEFAULT_LEVELS}"
SEEDS="${SEEDS:-1 2 3}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
TIMESTEPS_PER_ITERATION="${TIMESTEPS_PER_ITERATION:-20000}"
N_ENVS="${N_ENVS:-2}"
WANDB_ENTITY="${WANDB_ENTITY:-andrea02polimi-politecnico-di-milano}"
WANDB_PROJECT="${WANDB_PROJECT:-tuning-thesis}"
FIRST_CORE="${FIRST_CORE:-33}"
CORES_PER_RUN="${CORES_PER_RUN:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
STORAGE="${STORAGE:-outputs/optuna/journal.log}"
STUDY_SUFFIX="${STUDY_SUFFIX:-}"
CORE_SLOTS="${CORE_SLOTS:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Explicit slot list ("33-35 39-41") wins over the FIRST_CORE arithmetic; used
# by the orchestrator to avoid slots still busy with tuning workers.
if [[ -n "$CORE_SLOTS" ]]; then
    read -r -a SLOTS <<< "$CORE_SLOTS"
    MAX_PARALLEL=${#SLOTS[@]}
else
    SLOTS=()
    for ((i = 0; i < MAX_PARALLEL; i++)); do
        lo=$((FIRST_CORE + i * CORES_PER_RUN))
        SLOTS+=("${lo}-$((lo + CORES_PER_RUN - 1))")
    done
fi

mkdir -p logs

# Best config at the tuning budgets; per-level budget overrides are appended
# afterwards, so they win (later Hydra overrides take precedence).
OVERRIDES="$(cd scripts && "$PYTHON_BIN" export_best_config.py \
    --arm "$ARM" --study-suffix "$STUDY_SUFFIX" --format full --storage-path "../$STORAGE")"
BEST_INITIAL_QUERIES="$(cd scripts && "$PYTHON_BIN" export_best_config.py \
    --arm "$ARM" --study-suffix "$STUDY_SUFFIX" --format params --storage-path "../$STORAGE" \
    | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("initial_queries", 500))')"

slot=0
for LEVEL in $LEVELS; do
  for SEED in $SEEDS; do
    GROUP="budget_${ARM}_${LEVEL}"
    EXTRA=()
    if [[ "$AXIS" == "pref" ]]; then
      IQ=$(( BEST_INITIAL_QUERIES < LEVEL / 5 ? BEST_INITIAL_QUERIES : LEVEL / 5 ))
      EXTRA+=(
        "algo.kwargs.total_queries=$LEVEL"
        "train.kwargs.total_queries=$LEVEL"
        "algo.kwargs.initial_queries=$IQ"
      )
    else
      EXTRA+=( "run.n_expert_trajectories=$LEVEL" )
    fi
    range="${SLOTS[$slot]}"
    # shellcheck disable=SC2086
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        nohup taskset -c "$range" "$PYTHON_BIN" scripts/test_hybrid_SAC.py \
            $OVERRIDES \
            "${EXTRA[@]}" \
            run.seed="$SEED" \
            run.output_dir="outputs/budget_curves/${GROUP}" \
            run.name="${GROUP}-seed${SEED}" \
            run.group="$GROUP" \
            wandb.entity="$WANDB_ENTITY" \
            wandb.project="$WANDB_PROJECT" \
            wandb.tags="[budget_curve,${ARM}]" \
            env.n_envs="$N_ENVS" \
            train.kwargs.total_timesteps="$TOTAL_TIMESTEPS" \
            train.kwargs.timesteps_per_iteration="$TIMESTEPS_PER_ITERATION" \
            > "logs/budget_${ARM}_${LEVEL}_seed${SEED}.log" 2>&1 &
    echo "  $ARM budget=$LEVEL seed=$SEED on cores ${range} (pid $!)"
    slot=$(( (slot + 1) % MAX_PARALLEL ))
    if [[ "$slot" -eq 0 ]]; then wait; fi
  done
done
wait
echo "Budget curve finished for $ARM."
