#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

seeds=(0 1 2 3 4)
core_sets=("25-28" "29-32" "33-36" "37-40" "41-44")

LOG_DIR="outputs/demo_sac_robustness_4m_alberto_no_collision/launcher_logs"
mkdir -p "$LOG_DIR"

pids=()

for i in "${!seeds[@]}"; do
    seed="${seeds[$i]}"
    cores="${core_sets[$i]}"

    echo "Avvio seed=$seed sui core=$cores"

    env \
        SEED="$seed" \
        OUTPUT_DIR="outputs/demo_sac_robustness_4m_alberto_no_collision" \
        WANDB_PROJECT="demo-sac-robustness-4m" \
        TOTAL_TIMESTEPS=4000000 \
        TIMESTEPS_PER_ITERATION=20000 \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        taskset -c "$cores" \
        bash launchers/run_demo_SAC_robustness_4m.sh \
        >"$LOG_DIR/seed_${seed}.log" 2>&1 &

    pids+=("$!")
done

trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "Tutti i training sono terminati."