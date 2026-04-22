PROJECT_DIR="/work/fis3/sumo-human-feedback-rl"
OUTPUT_DIR="/storage/fis3"
CONDA_ENV="sumo-rlhf"
CORES="36-47"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

COMMON=(
    run.output_dir="$OUTPUT_DIR"
    wandb.kwargs.project=christiano-experiments
    wandb.kwargs.sync_tensorboard=false
    env.kwargs.ego=continuous
    env.kwargs.reward=fast
    env.n_envs=8
)

# ---------------------------------------------------------------------------
# Helper: run a single experiment
# ---------------------------------------------------------------------------
run_exp() {
    local name="$1"
    shift
    echo ""
    echo "============================================================"
    echo "  EXPERIMENT: $name"
    echo "============================================================"
    echo "y" | taskset -c "$CORES" python scripts/train.py \
        run.name="$name" \
        run.seed=0 \
        "$@"
}

run_exp PPO_dagger_pretrain_short \
    "${COMMON[@]}" \
    algo=christiano_demo \
    algo/agent=PPO \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.n_initial_queries=200 \
    algo.train.kwargs.n_queries_per_iter=10 \
    algo.train.kwargs.reward_model_train_steps=200 \
    algo.train.kwargs.reward_model_batch_size=64 \
    algo.train.kwargs.n_sft_steps=0 \
    algo.train.kwargs.sft_batch_size=0 \
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
    algo.agent.kwargs.batch_size=64 \
    algo.agent.kwargs.n_epochs=10 \
    algo.agent.kwargs.n_steps=1000

run_exp SAC_dagger_pretrain_short \
    "${COMMON[@]}" \
    algo=christiano_demo \
    algo.expert_id=sac-fast \
    algo/agent=SAC \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.n_initial_queries=200 \
    algo.train.kwargs.n_queries_per_iter=10 \
    algo.train.kwargs.reward_model_train_steps=200 \
    algo.train.kwargs.reward_model_batch_size=64 \
    algo.train.kwargs.n_sft_steps=200 \
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
    algo.agent.kwargs.train_freq=8000
#
#
#run_exp PPO_demo_logsigmoid_short \
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
#    algo.kwargs.demo_loss_weight=1.0 \
#    algo.kwargs.demo_loss_type=logsigmoid \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.agent.kwargs.n_epochs=10 \
#    algo.agent.kwargs.n_steps=1000
#
#run_exp SAC_demo_logsigmoid_short \
#    "${COMMON[@]}" \
#    algo=christiano_demo \
#    algo.expert_id=sac-fast \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=1 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.demo_loss_weight=1.0 \
#    algo.kwargs.demo_loss_type=logsigmoid \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp PPO_demo_softplus_short \
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
#    algo.kwargs.demo_loss_weight=1.0 \
#    algo.kwargs.demo_loss_type=constant_grad \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.agent.kwargs.n_epochs=10 \
#    algo.agent.kwargs.n_steps=1000
#
#run_exp SAC_demo_softplus_short \
#    "${COMMON[@]}" \
#    algo=christiano_demo \
#    algo.expert_id=sac-fast \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=1 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.demo_loss_weight=1.0 \
#    algo.kwargs.demo_loss_type=constant_grad \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp PPO_dagger_pretrain_short \
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
#
#run_exp SAC_dagger_pretrain_short \
#    "${COMMON[@]}" \
#    algo=christiano_demo \
#    algo.expert_id=sac-fast \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
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
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp SAC_demo_logsigmoid_short_PPO_expert \
#    "${COMMON[@]}" \
#    algo=christiano_demo \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=1 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.demo_loss_weight=1.0 \
#    algo.kwargs.demo_loss_type=logsigmoid \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp PPO_demo_logsigmoid_short_SAC_expert \
#    "${COMMON[@]}" \
#    algo=christiano_demo \
#    algo.expert_id=sac-fast \
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
#    algo.kwargs.demo_loss_weight=1.0 \
#    algo.kwargs.demo_loss_type=logsigmoid \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.agent.kwargs.n_epochs=10 \
#    algo.agent.kwargs.n_steps=1000

#run_exp SAC_long \
#    "${COMMON[@]}" \
#    algo=christiano \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=None \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp PPO_long \
#    "${COMMON[@]}" \
#    algo=christiano \
#    algo/agent=PPO \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=None \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.agent.kwargs.n_epochs=10 \
#    algo.agent.kwargs.n_steps=1000
#
#
#run_exp SAC_short \
#    "${COMMON[@]}" \
#    algo=christiano \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=1 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp PPO_short \
#    "${COMMON[@]}" \
#    algo=christiano \
#    algo/agent=PPO \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=1 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.agent.kwargs.n_epochs=10 \
#    algo.agent.kwargs.n_steps=1000
#
#run_exp SAC_10 \
#    "${COMMON[@]}" \
#    algo=christiano \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=10 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp PPO_10 \
#    "${COMMON[@]}" \
#    algo=christiano \
#    algo/agent=PPO \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=10 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.agent.kwargs.n_epochs=10 \
#    algo.agent.kwargs.n_steps=1000
#
#run_exp SAC_SFT_short \
#    "${COMMON[@]}" \
#    algo=christiano_demo \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
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
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp PPO_SFT_short \
#    "${COMMON[@]}" \
#    algo=christiano \
#    algo/agent=PPO \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
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
#
#run_exp SAC_SFT_long \
#    "${COMMON[@]}" \
#    algo=christiano_demo \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=None \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.demo_loss_weight=0.0 \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp PPO_SFT_long \
#    "${COMMON[@]}" \
#    algo=christiano \
#    algo/agent=PPO \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=None \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.demo_loss_weight=0.0 \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.agent.kwargs.n_epochs=10 \
#    algo.agent.kwargs.n_steps=1000
#
#run_exp SAC_SFT_10 \
#    "${COMMON[@]}" \
#    algo=christiano_demo \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=10 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.demo_loss_weight=0.0 \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp PPO_SFT_10 \
#    "${COMMON[@]}" \
#    algo=christiano \
#    algo/agent=PPO \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=10 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.demo_loss_weight=0.0 \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.agent.kwargs.n_epochs=10 \
#    algo.agent.kwargs.n_steps=1000
#
#run_exp SAC_demo_short \
#    "${COMMON[@]}" \
#    algo=christiano_demo \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=1 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.demo_loss_weight=1.0 \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp PPO_demo_short \
#    "${COMMON[@]}" \
#    algo=christiano \
#    algo/agent=PPO \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=1 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.demo_loss_weight=1.0 \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.agent.kwargs.n_epochs=10 \
#    algo.agent.kwargs.n_steps=1000
#
#run_exp SAC_demo_long \
#    "${COMMON[@]}" \
#    algo=christiano_demo \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=None \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.demo_loss_weight=1.0 \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp PPO_demo_long \
#    "${COMMON[@]}" \
#    algo=christiano \
#    algo/agent=PPO \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=None \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.demo_loss_weight=1.0 \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.agent.kwargs.n_epochs=10 \
#    algo.agent.kwargs.n_steps=1000
#
#run_exp SAC_demo_10 \
#    "${COMMON[@]}" \
#    algo=christiano_demo \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=10 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.demo_loss_weight=1.0 \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp PPO_demo_10 \
#    "${COMMON[@]}" \
#    algo=christiano \
#    algo/agent=PPO \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.train.kwargs.n_sft_steps=200 \
#    algo.train.kwargs.sft_batch_size=200 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3e-4 \
#    algo.kwargs.reward_model_l2=1e-4 \
#    algo.kwargs.segment_length=10 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.demo_loss_weight=1.0 \
#    algo.kwargs.expert_batch_size=64 \
#    algo.kwargs.expert_dataset_max_size=3000 \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.agent.kwargs.n_epochs=10 \
#    algo.agent.kwargs.n_steps=1000