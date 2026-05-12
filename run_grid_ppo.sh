#!/bin/bash
set -euo pipefail

PYTHON="python"
SCRIPT="test_chri_PPO_grid.py"
LOG_DIR="/storage/fis3/logs_grid_ppo"
mkdir -p "$LOG_DIR"

CORE_GROUPS=("30-35" "36-41" "42-47")
N_SLOTS=${#CORE_GROUPS[@]}

declare -a SLOT_PIDS
for i in $(seq 0 $((N_SLOTS - 1))); do SLOT_PIDS[$i]=-1; done

# Build all 420 combinations: "seg net_arch steps comps seed"
COMBOS=()
for seg in 1 2 3 10 20 50 200; do
    for net in "64,64" "128,128" "256,256"; do
        for seed in 0 1 2; do
            COMBOS+=("$seg $net 20000 500  $seed")
            COMBOS+=("$seg $net 20000 500  $seed")
            COMBOS+=("$seg $net 20000 2000 $seed")
            COMBOS+=("$seg $net 4000  100  $seed")
            COMBOS+=("$seg $net 4000  400  $seed")
        done
    done
done

TOTAL=${#COMBOS[@]}
echo "Total runs: $TOTAL"
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
    read -r seg net steps comps seed <<< "$combo"
    net_str="${net//,/x}"

    slot=$(get_free_slot)
    cores=${CORE_GROUPS[$slot]}
    logfile="$LOG_DIR/seg${seg}_net${net_str}_steps${steps}_comps${comps}_seed${seed}.log"

    launched=$((launched + 1))
    echo "[$(date '+%H:%M:%S')] ($launched/$TOTAL) slot=$slot cores=$cores | seg=$seg net=$net_str steps=$steps comps=$comps seed=$seed"

    taskset -c "$cores" $PYTHON $SCRIPT \
        --seg "$seg" \
        --net-arch "$net" \
        --steps "$steps" \
        --comps "$comps" \
        --seed "$seed" \
        > "$logfile" 2>&1 &

    SLOT_PIDS[$slot]=$!
done

wait
echo "======================================================"
echo "All $TOTAL runs completed."
