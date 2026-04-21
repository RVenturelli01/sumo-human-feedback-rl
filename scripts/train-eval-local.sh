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
    wandb.kwargs.project=christiano-experiments
    wandb.kwargs.sync_tensorboard=false
    env.kwargs.ego=continuous
    env.kwargs.reward=fast
    env.n_envs=4
)

#run_exp local_PPO_SFT_short \
#    "${COMMON[@]}" \
#    algo=christiano_demo \
#    algo/agent=PPO \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=0 \
#    algo.train.kwargs.sft_batch_size=0 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=1 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.demo_loss_weight=0.0 \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.agent.kwargs.n_epochs=10 \
#    algo.agent.kwargs.n_steps=1000

run_exp local_SAC_SFT_short \
    "${COMMON[@]}" \
    algo=christiano_demo \
    algo/agent=SAC \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.n_initial_queries=200 \
    algo.train.kwargs.n_queries_per_iter=10 \
    algo.train.kwargs.reward_model_train_steps=200 \
    algo.train.kwargs.reward_model_batch_size=64 \
    algo.train.kwargs.n_sft_steps=2000 \
    algo.train.kwargs.sft_batch_size=200 \
    algo.kwargs.reward_model_n_networks=3 \
    algo.kwargs.reward_model_hidden_size=256 \
    algo.kwargs.reward_model_lr=3e-4 \
    algo.kwargs.reward_model_l2=1e-4 \
    algo.kwargs.segment_length=1 \
    algo.kwargs.preference_dataset_max_size=3000 \
    algo.kwargs.query_schedule=constant \
    algo.kwargs.demo_loss_weight=0.0 \
    algo.kwargs.expert_batch_size=64 \
    algo.kwargs.expert_dataset_max_size=3000 \
    algo.agent.kwargs.learning_rate=3e-4 \
    algo.agent.kwargs.buffer_size=100000 \
    algo.agent.kwargs.learning_starts=100 \
    algo.agent.kwargs.batch_size=256 \
    algo.agent.kwargs.gradient_steps=1000 \
    algo.agent.kwargs.train_freq=4000
