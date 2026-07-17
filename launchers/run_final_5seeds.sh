#!/usr/bin/env bash
# Final multi-seed runs of one arm with its best Optuna config (thesis runs).
#
# Reads the best trial from the Optuna journal (export_best_config.py --format
# full: fixed + arm + best params) and launches one run per seed, pinned to 3
# cores each, grouped on W&B so the project shows one 5-seed group per arm.
#
# Usage:
#   ./launchers/run_final_5seeds.sh <arm> [group_suffix]
#
# Hybrid budget strategies (X=pref budget, Y=demo budget of the baselines):
#   strategy A (half budgets):  PREF_BUDGET=2500 DEMO_BUDGET=250 ./launchers/run_final_5seeds.sh hybrid_demo_1 _A
#   strategy B (full budgets):  PREF_BUDGET=5000 DEMO_BUDGET=500 ./launchers/run_final_5seeds.sh hybrid_demo_1 _B
#
# Env knobs: SEEDS="1 2 3 4 5", PREF_BUDGET=5000, DEMO_BUDGET=500,
#   TOTAL_TIMESTEPS=2000000, TIMESTEPS_PER_ITERATION=20000, N_ENVS=2,
#   WANDB_PROJECT=thesis, FIRST_CORE=33, MAX_PARALLEL=5,
#   STORAGE=outputs/optuna/journal.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ARM="${1:?usage: run_final_5seeds.sh <arm> [group_suffix]}"
SUFFIX="${2:-}"

SEEDS="${SEEDS:-1 2 3 4 5}"
PREF_BUDGET="${PREF_BUDGET:-5000}"
DEMO_BUDGET="${DEMO_BUDGET:-500}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
TIMESTEPS_PER_ITERATION="${TIMESTEPS_PER_ITERATION:-20000}"
N_ENVS="${N_ENVS:-2}"
WANDB_ENTITY="${WANDB_ENTITY:-andrea02polimi-politecnico-di-milano}"
WANDB_PROJECT="${WANDB_PROJECT:-thesis}"
FIRST_CORE="${FIRST_CORE:-33}"
CORES_PER_RUN="${CORES_PER_RUN:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
STORAGE="${STORAGE:-outputs/optuna/journal.log}"
STUDY_SUFFIX="${STUDY_SUFFIX:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

GROUP="${ARM}${SUFFIX}"
mkdir -p logs "outputs/final/${GROUP}"

echo "Exporting best config for $ARM (pref=$PREF_BUDGET, demo=$DEMO_BUDGET)..."
OVERRIDES="$(cd scripts && "$PYTHON_BIN" export_best_config.py \
    --arm "$ARM" --study-suffix "$STUDY_SUFFIX" --format full --storage-path "../$STORAGE" \
    --pref-budget "$PREF_BUDGET" --demo-budget "$DEMO_BUDGET")"
echo "  $OVERRIDES"

slot=0
for SEED in $SEEDS; do
    lo=$((FIRST_CORE + slot * CORES_PER_RUN))
    hi=$((lo + CORES_PER_RUN - 1))
    # shellcheck disable=SC2086
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        nohup taskset -c "${lo}-${hi}" "$PYTHON_BIN" scripts/train_hybrid_sac.py \
            $OVERRIDES \
            run.seed="$SEED" \
            run.output_dir="outputs/final/${GROUP}" \
            run.name="${GROUP}-seed${SEED}" \
            run.group="$GROUP" \
            wandb.entity="$WANDB_ENTITY" \
            wandb.project="$WANDB_PROJECT" \
            wandb.tags="[final,${ARM}]" \
            env.n_envs="$N_ENVS" \
            train.kwargs.total_timesteps="$TOTAL_TIMESTEPS" \
            train.kwargs.timesteps_per_iteration="$TIMESTEPS_PER_ITERATION" \
            > "logs/final_${GROUP}_seed${SEED}.log" 2>&1 &
    echo "  seed $SEED on cores ${lo}-${hi} (pid $!)"
    slot=$(( (slot + 1) % MAX_PARALLEL ))
    # When all slots are busy, wait for the whole wave before reusing cores.
    if [[ "$slot" -eq 0 ]]; then wait; fi
done
wait
echo "All seeds finished for $GROUP."
