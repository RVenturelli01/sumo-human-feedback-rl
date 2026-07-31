PYTHON_BIN=python
STORAGE=outputs/optuna/journal.log
WANDB_ENTITY=andrea02polimi-politecnico-di-milano
WANDB_PROJECT=tuning-thesis-fixed-demo-subseed
DEMO_SUBSAMPLE_SEED=1000
LEVELS="2723 1000 500 200 100 50"
SEEDS="1 2 3"
CORE_SLOTS=("32-33")

OVERRIDES="$(
  cd scripts && "$PYTHON_BIN" export_best_config.py \
    --arm demo_2 \
    --format full \
    --storage-path "../$STORAGE"
)"

slot=0
for LEVEL in $LEVELS; do
  for SEED in $SEEDS; do
    GROUP="budget_demo_2_${LEVEL}"
    RUN_NAME="${GROUP}-seed${SEED}"
    range="${CORE_SLOTS[$slot]}"

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      nohup taskset -c "$range" "$PYTHON_BIN" scripts/train_hybrid_sac.py \
        $OVERRIDES \
        run.n_expert_trajectories="$LEVEL" \
        run.demo_subsample_seed="$DEMO_SUBSAMPLE_SEED" \
        run.seed="$SEED" \
        run.output_dir="outputs/budget_curves/${GROUP}" \
        run.name="$RUN_NAME" \
        run.group="$GROUP" \
        wandb.entity="$WANDB_ENTITY" \
        wandb.project="$WANDB_PROJECT" \
        wandb.tags="[budget_curve,demo_2,fixed_demo_subsample_seed]" \
        env.n_envs=2 \
        train.kwargs.total_timesteps=2000000 \
        train.kwargs.timesteps_per_iteration=20000 \
        > "logs/${GROUP}_seed${SEED}.log" 2>&1 &

    echo "$RUN_NAME on cores $range"
    slot=$(( (slot + 1) % ${#CORE_SLOTS[@]} ))
    if [[ "$slot" -eq 0 ]]; then wait; fi
  done
done
wait



PYTHON_BIN=/home/fis3/miniconda3/envs/sumo-rlhf/bin/python

nohup "$PYTHON_BIN" scripts/tune_hybrid_sac.py \
  --arm pref_bernoulli \
  --n-trials 30 \
  --cores 30-31 \
  --pref-budget 100000 \
  --study-suffix _q100k_temp \
  --total-timesteps 1000000 \
  > logs/optuna_pref_bernoulli_q100k_temp.log 2>&1 &