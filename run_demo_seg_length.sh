#!/bin/bash
# Esperimento demo: lunghezze di segmento (1 2 5 10 20 1000) × 5 seed
# ZhangAlgorithm (demo-rew) con expert PPO pretrained.
# comps_per_iter = seg*100, cappato a 20 per seg=1000
# WandB group = demo_seg={seg}_comps={comps}  (esclude seed)
# Checkpoint → /storage/fis3/checkpoints/demo-seg-study/
set -euo pipefail

PYTHON="python"
SCRIPT="test_demo_seg_length.py"
LOG_DIR="/storage/fis3/logs_demo_seg_length"
EXPERT_PATH="/storage/fis3/pretrained/ppo_pretrained_ppo-fast.zip"   # ← modifica se necessario

mkdir -p "$LOG_DIR"

CORE_GROUPS=("0-3" "4-7" "8-11")
N_SLOTS=${#CORE_GROUPS[@]}

declare -a SLOT_PIDS
for i in $(seq 0 $((N_SLOTS - 1))); do SLOT_PIDS[$i]=-1; done

# comps_per_iter per segmento (deve corrispondere a COMPS_FOR_SEG in test_demo_seg_length.py)
declare -A COMPS_FOR_SEG
COMPS_FOR_SEG[1]=100
COMPS_FOR_SEG[2]=200
COMPS_FOR_SEG[5]=500
COMPS_FOR_SEG[10]=1000
COMPS_FOR_SEG[20]=2000
COMPS_FOR_SEG[1000]=20

SEGS=(1 2 5 10 20 1000)
SEEDS=(0 1 2 3 4)

# Costruisce la lista di combinazioni "seg seed"
COMBOS=()
for seg in "${SEGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        COMBOS+=("$seg $seed")
    done
done

TOTAL=${#COMBOS[@]}
echo "Total runs: $TOTAL  (${#SEGS[@]} seg × ${#SEEDS[@]} seed)"
echo "Parallel slots: $N_SLOTS (core groups: ${CORE_GROUPS[*]})"
echo "Expert: $EXPERT_PATH"
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
    read -r seg seed <<< "$combo"
    comps=${COMPS_FOR_SEG[$seg]}

    slot=$(get_free_slot)
    cores=${CORE_GROUPS[$slot]}
    logfile="$LOG_DIR/seg${seg}_comps${comps}_seed${seed}.log"

    launched=$((launched + 1))
    echo "[$(date '+%H:%M:%S')] ($launched/$TOTAL) slot=$slot cores=$cores | seg=$seg comps=$comps seed=$seed"

    taskset -c "$cores" $PYTHON $SCRIPT \
        --seg         "$seg"         \
        --seed        "$seed"        \
        --expert-path "$EXPERT_PATH" \
        > "$logfile" 2>&1 &

    SLOT_PIDS[$slot]=$!
done

wait
echo "======================================================"
echo "All $TOTAL runs completed."
