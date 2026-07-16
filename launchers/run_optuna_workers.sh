#!/usr/bin/env bash
# Launch N Optuna workers for one tuning arm, each pinned to its own core slice.
#
# Each worker runs scripts/tune_hybrid_sac.py, which spawns one training
# subprocess at a time (test_hybrid_SAC.py with n_envs=2). A trial keeps busy
# the 2 SUMO subprocesses during rollouts and the single-threaded learner
# during gradient steps, so 3 pinned cores per worker is the right ratio.
#
# Usage:
#   ./launchers/run_optuna_workers.sh <arm> <n_trials_total> [n_workers] [first_core] [extra tune_hybrid_sac.py args...]
#
# NOTE: prefer run_optuna_parallel_arms.sh (one sequential worker per arm,
# fully-informed TPE). Use this one to put several workers on a single arm.
#
# Example (server fis3, allowed cores 33-47 -> 5 workers x 3 cores):
#   ./launchers/run_optuna_workers.sh pref_soft 40 5 33
#   ./launchers/run_optuna_workers.sh demo_2 40 5 33 --total-timesteps 1000000
#
# Workers share the study through outputs/optuna/journal.log, so the total
# trial budget is split across them. Logs go to logs/optuna_<arm>_w<i>.log.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ARM="${1:?usage: run_optuna_workers.sh <arm> <n_trials_total> [n_workers] [first_core] [extra args...]}"
N_TRIALS_TOTAL="${2:?missing n_trials_total}"
N_WORKERS="${3:-5}"
FIRST_CORE="${4:-33}"
shift $(( $# > 4 ? 4 : $# ))
EXTRA_ARGS=("$@")

CORES_PER_WORKER="${CORES_PER_WORKER:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p logs outputs/optuna

# Split the trial budget across workers (first workers take the remainder).
BASE=$((N_TRIALS_TOTAL / N_WORKERS))
REMAINDER=$((N_TRIALS_TOTAL % N_WORKERS))

echo "Arm $ARM: $N_TRIALS_TOTAL trials on $N_WORKERS workers," \
     "cores ${FIRST_CORE}..$((FIRST_CORE + N_WORKERS * CORES_PER_WORKER - 1))"

for i in $(seq 0 $((N_WORKERS - 1))); do
    lo=$((FIRST_CORE + i * CORES_PER_WORKER))
    hi=$((lo + CORES_PER_WORKER - 1))
    n_trials=$((BASE + (i < REMAINDER ? 1 : 0)))
    [[ "$n_trials" -eq 0 ]] && continue
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        nohup "$PYTHON_BIN" scripts/tune_hybrid_sac.py \
            --arm "$ARM" \
            --n-trials "$n_trials" \
            --cores "${lo}-${hi}" \
            "${EXTRA_ARGS[@]}" \
            > "logs/optuna_${ARM}_w${i}.log" 2>&1 &
    echo "  worker $i on cores ${lo}-${hi}: $n_trials trials (pid $!)"
done

echo "Tail progress with:  tail -f logs/optuna_${ARM}_w*.log"
wait
echo "All workers exited for arm $ARM."
