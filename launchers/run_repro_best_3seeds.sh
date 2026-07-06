#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Reproduce the 5 best configs (from the W&B sweep survey) on 3 seeds each.
#
#   1. maxent_2           (SAC)  -> scripts/test_demo_SAC.py
#   2. demo               (SAC)  -> scripts/test_demo_SAC.py
#   3. maxent_corrected   (SAC)  -> scripts/test_demo_SAC.py
#   4. chri_soft          (PPO)  -> scripts/test_chri_PPO.py
#   5. chri_binary_bernoulli (PPO)-> scripts/test_chri_PPO.py
#
# 5 configs x 3 seeds = 15 runs, logged to a NEW W&B project, each config in its
# own W&B *group* (seeds aggregate) so a grouped plot with x-axis
# `agent/time/total_timesteps` shows the mean +/- std band across seeds.
#
# Pins each run to cores in [24, 47] and activates the `sumo-rlhf` conda env.
#
# Run it detached so it survives disconnects, e.g.:
#     tmux new -s repro 'bash launchers/run_repro_best_3seeds.sh'
#   or
#     nohup bash launchers/run_repro_best_3seeds.sh > repro.out 2>&1 &
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---- conda -----------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-sumo-rlhf}"
for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
  if [[ -f "$c/etc/profile.d/conda.sh" ]]; then source "$c/etc/profile.d/conda.sh"; break; fi
done
conda activate "$CONDA_ENV"
echo "conda env: $(which python)"

# ---- experiment knobs ------------------------------------------------------
WANDB_ENTITY="${WANDB_ENTITY:-andrea02polimi-politecnico-di-milano}"
WANDB_PROJECT="${WANDB_PROJECT:-repro_best_3seeds}"
SEEDS=(${SEEDS:-0 1 2})
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/repro_best_3seeds}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

# ---- core pool -------------------------------------------------------------
CORE_START="${CORE_START:-24}"
CORE_END="${CORE_END:-47}"
CORES_PER_JOB="${CORES_PER_JOB:-1}"          # raise to run fewer jobs at once
NCORES=$(( CORE_END - CORE_START + 1 ))
NSLOTS=$(( NCORES / CORES_PER_JOB ))
[[ $NSLOTS -lt 1 ]] && { echo "CORES_PER_JOB too large for the pool"; exit 1; }

# One thread per allotted core; avoids oversubscription across parallel jobs.
export OMP_NUM_THREADS="$CORES_PER_JOB"
export MKL_NUM_THREADS="$CORES_PER_JOB"
export OPENBLAS_NUM_THREADS="$CORES_PER_JOB"
export NUMEXPR_NUM_THREADS="$CORES_PER_JOB"

CONFIGS=(maxent_2 demo maxent_corrected chri_soft chri_binary_bernoulli)

# Populate globals SCRIPT and OVERRIDES (array) for a given config name.
build_cmd() {
  local cfg="$1"
  OVERRIDES=(
    "wandb.entity=$WANDB_ENTITY"
    "wandb.project=$WANDB_PROJECT"
    "wandb.tags=[repro,best,3seeds]"
  )
  case "$cfg" in
    maxent_2|demo)
      SCRIPT="scripts/test_demo_SAC.py"
      OVERRIDES+=(
        "algo.kwargs.loss_type=$cfg"
        algo.kwargs.lr_rew=0.0001 algo.kwargs.gradient_steps_rew=200 algo.kwargs.l2_rew=0
        algo.kwargs.temperature=1 algo.kwargs.batch_size_expert=64 algo.kwargs.batch_size_model=64
        algo.kwargs.relabel_rewards=true algo.kwargs.normalize_agent_reward=true
        algo.kwargs.initial_agent_timesteps=20000
        "algo.kwargs.reward_model_kwargs.net_arch=[64,64]"
        agent.kwargs.learning_rate=0.0001242983309370202 agent.kwargs.ent_coef=auto
        agent.kwargs.gamma=0.999 agent.kwargs.tau=0.0031102923983872435
        agent.kwargs.batch_size=256 agent.kwargs.train_freq=8 agent.kwargs.gradient_steps=32
        agent.kwargs.buffer_size=300000 agent.kwargs.learning_starts=2000
        train.kwargs.total_timesteps=2000000 train.kwargs.timesteps_per_iteration=20000
      ) ;;
    maxent_corrected)
      SCRIPT="scripts/test_demo_SAC.py"
      OVERRIDES+=(
        algo.kwargs.loss_type=maxent_corrected
        algo.kwargs.lr_rew=0.0003 algo.kwargs.gradient_steps_rew=20 algo.kwargs.l2_rew=0.05
        algo.kwargs.temperature=1 algo.kwargs.fragment_length=null
        algo.kwargs.batch_size_expert=64 algo.kwargs.batch_size_model=64
        algo.kwargs.relabel_rewards=true algo.kwargs.normalize_agent_reward=false
        algo.kwargs.initial_agent_timesteps=20000
        "algo.kwargs.reward_model_kwargs.net_arch=[64,64]"
        agent.kwargs.learning_rate=0.0001242983309370202 agent.kwargs.ent_coef=auto
        agent.kwargs.gamma=0.995 agent.kwargs.tau=0.005
        agent.kwargs.batch_size=256 agent.kwargs.train_freq=8 agent.kwargs.gradient_steps=64
        agent.kwargs.buffer_size=300000 agent.kwargs.learning_starts=2000
        train.kwargs.total_timesteps=2000000 train.kwargs.timesteps_per_iteration=20000
      ) ;;
    chri_soft)
      SCRIPT="scripts/test_chri_PPO.py"
      OVERRIDES+=(
        algo.kwargs.labels_type=soft
        algo.kwargs.lr_rew=0.0003 algo.kwargs.gradient_steps_rew=100 algo.kwargs.l2_rew=0.0001
        algo.kwargs.temperature=20 algo.kwargs.fragment_length=null
        algo.kwargs.fragmenter_type=active algo.kwargs.initial_queries=200
        algo.kwargs.train_comparison_frac=0.8
        "algo.kwargs.reward_model_kwargs.net_arch=[128,128]"
        agent.kwargs.learning_rate=0.0003 agent.kwargs.ent_coef=0 agent.kwargs.gamma=0.997
        agent.kwargs.batch_size=64 agent.kwargs.n_steps=1000 agent.kwargs.n_epochs=10
        train.kwargs.total_timesteps=2000000 train.kwargs.total_queries=10000
        train.kwargs.timesteps_per_iteration=20000
      ) ;;
    chri_binary_bernoulli)
      SCRIPT="scripts/test_chri_PPO.py"
      OVERRIDES+=(
        algo.kwargs.labels_type=binary_bernoulli
        algo.kwargs.lr_rew=0.0003 algo.kwargs.gradient_steps_rew=100 algo.kwargs.l2_rew=0.0001
        algo.kwargs.temperature=20 algo.kwargs.fragment_length=2
        algo.kwargs.fragmenter_type=active algo.kwargs.initial_queries=2000
        algo.kwargs.train_comparison_frac=0.8
        "algo.kwargs.reward_model_kwargs.net_arch=[128,128]"
        agent.kwargs.learning_rate=0.0003 agent.kwargs.ent_coef=0 agent.kwargs.gamma=0.997
        agent.kwargs.batch_size=64 agent.kwargs.n_steps=1000 agent.kwargs.n_epochs=10
        train.kwargs.total_timesteps=2000000 train.kwargs.total_queries=100000
        train.kwargs.timesteps_per_iteration=20000
      ) ;;
    *) echo "unknown config $cfg" >&2; exit 1 ;;
  esac
}

launch() {
  local cfg="$1" seed="$2" slot="$3"
  local c0=$(( CORE_START + slot * CORES_PER_JOB ))
  local c1=$(( c0 + CORES_PER_JOB - 1 ))
  build_cmd "$cfg"
  local log="$LOG_DIR/${cfg}_seed${seed}.log"
  echo "-> $cfg seed=$seed cores ${c0}-${c1}  (log: $log)"
  taskset -c "${c0}-${c1}" python "$SCRIPT" \
    "${OVERRIDES[@]}" \
    run.seed="$seed" \
    "run.output_dir=$OUTPUT_ROOT/${cfg}" \
    > "$log" 2>&1 &
}

# ---- build the job list (config x seed) ------------------------------------
jobs=()
for cfg in "${CONFIGS[@]}"; do
  for s in "${SEEDS[@]}"; do jobs+=("$cfg|$s"); done
done
total=${#jobs[@]}
echo "launching $total runs on cores ${CORE_START}-${CORE_END} (${NSLOTS} slots, ${CORES_PER_JOB} core/job)"

# ---- run in waves of NSLOTS ------------------------------------------------
i=0
fail=0
while [ $i -lt $total ]; do
  pids=()
  specs=()
  for (( slot=0; slot<NSLOTS && i<total; slot++, i++ )); do
    spec="${jobs[$i]}"
    launch "${spec%%|*}" "${spec##*|}" "$slot"
    pids+=($!)
    specs+=("$spec")
  done
  # wait for the whole wave; report failures but keep going
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "!! FAILED: ${specs[$idx]} (see log)"; fail=$(( fail + 1 ))
    fi
  done
done

echo "done. $((total - fail))/$total succeeded. logs in $LOG_DIR"
[[ $fail -eq 0 ]]
