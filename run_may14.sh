#!/bin/bash
# Esperimento 14 maggio: seg × fragmenter_type × seed
# ChristianoAlgorithm (no demo), fragmenter_type ∈ {active, random}
# seg ∈ {1, 2, 5, 20, 50, 80, 1000} × 3 seed = 42 run totali
# comps = initial_comps = 2000 (200 per seg=1000), comparison_timesteps=20k
# Core 0-11 (slots 0-2) + core 30-45 (slots 3-6) → 7 slot paralleli
# Checkpoint → /storage/fis3/checkpoints/may14/
set -euo pipefail

PYTHON="python"
SCRIPT="test_may14.py"
LOG_DIR="/storage/fis3/logs_may14"

mkdir -p "$LOG_DIR"

CORE_GROUPS=("0-3" "4-7" "8-11" "30-33" "34-37" "38-41" "42-45")
N_SLOTS=${#CORE_GROUPS[@]}

declare -a SLOT_PIDS
for i in $(seq 0 $((N_SLOTS - 1))); do SLOT_PIDS[$i]=-1; done

SEGS=(1 5 20)
FRAGS=("random")
SEEDS=(10)

COMBOS=()
for seg in "${SEGS[@]}"; do
    for frag in "${FRAGS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            COMBOS+=("$seg $frag $seed")
        done
    done
done

TOTAL=${#COMBOS[@]}
echo "Total runs: $TOTAL  (${#SEGS[@]} seg × ${#FRAGS[@]} frag × ${#SEEDS[@]} seed)"
echo "Parallel slots: $N_SLOTS (core groups: ${CORE_GROUPS[*]})"
echo "------------------------------------------------------"

get_free_slot() {
    while true; do
        for i in $(seq 0 $((N_SLOTS - 1))); do
            pid=${SLOT_PIDS[$i]}
            if [ "$pid" -eq -1 ] || ! kill -0 "$pid" 2>/dev/null; then
                echo "$i"
                return
            fi
        done
        sleep 10
    done
}

launched=0
for combo in "${COMBOS[@]}"; do
    read -r seg frag seed <<< "$combo"

    slot=$(get_free_slot)
    cores=${CORE_GROUPS[$slot]}
    logfile="$LOG_DIR/seg${seg}_frag${frag}_seed${seed}.log"

    launched=$((launched + 1))
    echo "[$(date '+%H:%M:%S')] ($launched/$TOTAL) slot=$slot cores=$cores | seg=$seg frag=$frag seed=$seed"

    taskset -c "$cores" $PYTHON $SCRIPT \
        --seg  "$seg"  \
        --frag "$frag" \
        --seed "$seed" \
        > "$logfile" 2>&1 &

    SLOT_PIDS[$slot]=$!
done

wait
echo "======================================================"
echo "All $TOTAL runs completed."