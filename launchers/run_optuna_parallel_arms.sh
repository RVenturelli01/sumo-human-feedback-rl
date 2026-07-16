#!/usr/bin/env bash
# Tune several arms IN PARALLEL, one sequential Optuna worker per arm.
#
# One worker per arm means each study runs one trial at a time: after the
# random startup trials, every new trial is informed by ALL completed ones
# (best use of TPE). Each worker pins its training subprocess to 3 cores
# (n_envs=2 -> 2 SUMO processes + 1 single-threaded learner).
#
# Usage:
#   ./launchers/run_optuna_parallel_arms.sh <n_trials_per_arm> [first_core] [arm ...]
#
# Defaults: first_core=33, arms = the 4 baselines + hybrid_demo_1
# (5 workers x 3 cores = cores 33..47). Launch hybrid_demo_2 with
# run_optuna_workers.sh (or this script) when a slot frees up.
#
# Extra tune_hybrid_sac.py args go in TUNE_ARGS, e.g.:
#   TUNE_ARGS="--total-timesteps 1000000" ./launchers/run_optuna_parallel_arms.sh 30
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

N_TRIALS="${1:?usage: run_optuna_parallel_arms.sh <n_trials_per_arm> [first_core] [arm ...]}"
FIRST_CORE="${2:-33}"
shift $(( $# > 2 ? 2 : $# ))
ARMS=("$@")
[[ ${#ARMS[@]} -eq 0 ]] && ARMS=(pref_soft pref_bernoulli demo_1 demo_2 hybrid_demo_1)

CORES_PER_WORKER="${CORES_PER_WORKER:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TUNE_ARGS="${TUNE_ARGS:-}"

mkdir -p logs outputs/optuna

echo "Tuning ${#ARMS[@]} arms in parallel, $N_TRIALS trials each," \
     "cores ${FIRST_CORE}..$((FIRST_CORE + ${#ARMS[@]} * CORES_PER_WORKER - 1))"

i=0
for arm in "${ARMS[@]}"; do
    lo=$((FIRST_CORE + i * CORES_PER_WORKER))
    hi=$((lo + CORES_PER_WORKER - 1))
    # shellcheck disable=SC2086
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        nohup "$PYTHON_BIN" scripts/tune_hybrid_sac.py \
            --arm "$arm" \
            --n-trials "$N_TRIALS" \
            --cores "${lo}-${hi}" \
            $TUNE_ARGS \
            > "logs/optuna_${arm}.log" 2>&1 &
    echo "  $arm on cores ${lo}-${hi} (pid $!)"
    i=$((i + 1))
done

echo "Tail progress with:  tail -f logs/optuna_*.log"
wait
echo "All arm workers exited."
