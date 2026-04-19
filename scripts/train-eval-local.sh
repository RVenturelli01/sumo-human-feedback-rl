run_exp() {
    local name="$1"
    shift
    echo ""
    echo "============================================================"
    echo "  EXPERIMENT: $name"
    echo "============================================================"
    echo "y" | python train.py \
        run.name="$name" \
        run.seed=0 \
        "$@"
}

COMMON=(
    run.output_dir=outputs
    wandb.kwargs.project=local-experiments
    wandb.kwargs.sync_tensorboard=false
    env.kwargs.ego=continuous
    env.kwargs.reward=fast
    env.n_envs=4
)

run_exp dagger \
  "${COMMON[@]}" \
  algo=dagger \
  algo.expert_id=ppo-v0 \
  algo.train.kwargs.n_rounds=25
