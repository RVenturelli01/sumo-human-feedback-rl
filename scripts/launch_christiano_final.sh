#!/usr/bin/env bash
# ============================================================
#  Christiano final experiments — 512 runs
#  Grid: lr_rew × gradient_steps_rew × batch_size_rew
#        × fragment_length × fragmenter_type × net_arch
#        × (labels_type × total_queries)
#
#  Parallelism: 8 slots × 6 cores = 48 cores (0-47)
#  initial_queries = 2 × total_queries / 100
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── WandB ──────────────────────────────────────────────────────────────────
ENTITY="andrea02polimi-politecnico-di-milano"
PROJECT="christiano_final_experiments"

# ── Paths ──────────────────────────────────────────────────────────────────
OUTPUT_DIR="/storage/fis3"
LOG_DIR="$OUTPUT_DIR/logs"

# ── Fixed hyperparameters ──────────────────────────────────────────────────
SEED=0
TOTAL_TIMESTEPS=2000000
TIMESTEPS_PER_IT=20000
CHECKPOINT_INTERVAL=10

# ── Parallelism ─────────────────────────────────────────────────────────────
# 6 cores per run: 4 for SubprocVecEnv workers + 2 for main/reward-model
# 8 slots × 6 cores = 48 cores (0-47)
CORES_PER_RUN=6
MAX_PARALLEL=$(( 48 / CORES_PER_RUN ))

# ── Hyperparameter grid ─────────────────────────────────────────────────────
LR_VALS=("3e-4" "1e-3")
GRAD_VALS=(100 300)
BSZ_VALS=(100 500)
FLEN_VALS=(1 2 25 null)
FTYPE_VALS=("active" "random")
ARCH_VALS=("[128,128]" "[32,32]")

# labels_type + total_queries pairs  (format: labels:tq:initial_queries)
# initial_queries = 2 × total_queries / 100  (= 2 queries-per-iteration)
LABEL_TQ_PAIRS=(
    "soft:10000:200"
    "binary_bernoulli:100000:2000"
    "binary_bernoulli:50000:1000"
    "binary_bernoulli:10000:200"
)

# ── Build experiment list ────────────────────────────────────────────────────
declare -a EXPERIMENTS=()

for lr in "${LR_VALS[@]}"; do
  for grad in "${GRAD_VALS[@]}"; do
    for bsz in "${BSZ_VALS[@]}"; do
      for flen in "${FLEN_VALS[@]}"; do
        for ftype in "${FTYPE_VALS[@]}"; do
          for arch in "${ARCH_VALS[@]}"; do
            for ltp in "${LABEL_TQ_PAIRS[@]}"; do
              IFS=':' read -r labels tq iq <<< "$ltp"
              EXPERIMENTS+=("$lr|$grad|$bsz|$flen|$ftype|$arch|$labels|$tq|$iq")
            done
          done
        done
      done
    done
  done
done

TOTAL=${#EXPERIMENTS[@]}

echo "============================================================"
echo " Christiano final experiments"
echo " Total runs  : $TOTAL"
echo " Parallel    : $MAX_PARALLEL slots × $CORES_PER_RUN cores (cores 0-47)"
echo " Output      : $OUTPUT_DIR"
echo " WandB       : $ENTITY / $PROJECT"
echo " Started at  : $(date)"
echo "============================================================"
echo ""

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

# ── Job pool ─────────────────────────────────────────────────────────────────
declare -a SLOT_PID=()
for ((s = 0; s < MAX_PARALLEL; s++)); do SLOT_PID[$s]=0; done

submitted=0
completed=0

while (( completed < TOTAL )); do
    for ((s = 0; s < MAX_PARALLEL; s++)); do
        pid=${SLOT_PID[$s]}

        # ── Reap finished slot ────────────────────────────────────────────
        if (( pid != 0 )) && ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" && rc=0 || rc=$?
            if (( rc == 0 )); then
                echo "[$(date '+%H:%M:%S')] DONE  slot=$s  pid=$pid  completed=$((completed + 1))/$TOTAL"
            else
                echo "[$(date '+%H:%M:%S')] FAIL  slot=$s  pid=$pid  rc=$rc  completed=$((completed + 1))/$TOTAL"
            fi
            SLOT_PID[$s]=0
            pid=0
            (( completed++ ))
        fi

        # ── Submit next run into free slot ────────────────────────────────
        if (( pid == 0 && submitted < TOTAL )); then
            (( submitted++ ))
            exp="${EXPERIMENTS[$((submitted - 1))]}"
            IFS='|' read -r lr grad bsz flen ftype arch labels tq iq <<< "$exp"

            c0=$(( s * CORES_PER_RUN ))
            c1=$(( c0 + CORES_PER_RUN - 1 ))

            # Build a readable filename tag for the log
            arch_tag=$(echo "$arch" | tr -d '[]' | tr ',' 'x')
            log="$LOG_DIR/$(printf '%04d' "$submitted")_lr${lr}_g${grad}_b${bsz}_fl${flen}_${ftype}_${arch_tag}_${labels}_tq${tq}.log"

            echo "[$(date '+%H:%M:%S')] START run $submitted/$TOTAL  slot=$s  cores=${c0}-${c1}  lr=$lr g=$grad b=$bsz fl=$flen ft=$ftype arch=$arch labels=$labels tq=$tq iq=$iq"

            (
                taskset -c "${c0}-${c1}" python scripts/test_chri_PPO.py \
                    --config-path ../configs \
                    --config-name  test_chri_PPO \
                    run.seed="$SEED" \
                    run.output_dir="$OUTPUT_DIR" \
                    run.baseline=false \
                    wandb.entity="$ENTITY" \
                    wandb.project="$PROJECT" \
                    wandb.tags=null \
                    algo.kwargs.lr_rew="$lr" \
                    algo.kwargs.gradient_steps_rew="$grad" \
                    algo.kwargs.batch_size_rew="$bsz" \
                    "algo.kwargs.fragment_length=$flen" \
                    algo.kwargs.fragmenter_type="$ftype" \
                    algo.kwargs.labels_type="$labels" \
                    "algo.kwargs.reward_model_kwargs.net_arch=$arch" \
                    algo.kwargs.initial_queries="$iq" \
                    train.kwargs.total_timesteps="$TOTAL_TIMESTEPS" \
                    train.kwargs.total_queries="$tq" \
                    train.kwargs.timesteps_per_iteration="$TIMESTEPS_PER_IT" \
                    train.kwargs.checkpoint_interval="$CHECKPOINT_INTERVAL" \
                    >> "$log" 2>&1
            ) &
            SLOT_PID[$s]=$!
        fi
    done

    sleep 5
done

echo ""
echo "============================================================"
echo " All $TOTAL experiments finished — $(date)"
echo "============================================================"
