set -e

# ---------------------------------------------------------------------------
# Paths / environment
# ---------------------------------------------------------------------------
PROJECT_DIR="/work/fis3/sumo-human-feedback-rl"
OUTPUT_DIR="/storage/fis3"
CONDA_ENV="sumo-rlhf"
CORES="36-47"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ---------------------------------------------------------------------------
# Shared overrides (apply to every run)
# ---------------------------------------------------------------------------
COMMON=(
    run.output_dir="$OUTPUT_DIR"
    wandb.kwargs.project=experiments
    wandb.kwargs.sync_tensorboard=false
    env.kwargs.ego=continuous
    env.kwargs.reward=fast
    env.n_envs=8
    #algo=christiano
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

#run_exp christiano_SAC \
#    "${COMMON_SAC[@]}" \
#    algo.kwargs.segment_length=25 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.train.kwargs.total_timesteps=1000000

#run_exp christiano_PPO \
#    "${COMMON_PPO[@]}" \
#    algo.kwargs.segment_length=25 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.agent.kwargs.n_steps=1000 \
#    algo.train.kwargs.n_rollout_steps=1000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.train.kwargs.total_timesteps=1000000

#run_exp christiano_SAC_small_ensemble \
#    "${COMMON_SAC[@]}" \
#    algo.kwargs.segment_length=25 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=2 \
#    algo.train.kwargs.total_timesteps=1000000

#run_exp christiano_PPO_small_ensemble \
#    "${COMMON_PPO[@]}" \
#    algo.kwargs.segment_length=25 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.agent.kwargs.n_steps=1000 \
#    algo.train.kwargs.n_rollout_steps=1000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=2 \
#    algo.train.kwargs.total_timesteps=1000000


# con 8 env bisogna dimezzare n_rollout_steps

# provare con seg length = 1
#run_exp christiano_SAC_seg_len_1 \
#    "${COMMON_SAC[@]}" \
#    algo.kwargs.segment_length=1 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.n_rollout_steps=1000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.train.kwargs.total_timesteps=1000000

#run_exp christiano_PPO_seg_len1 \
#    "${COMMON_PPO[@]}" \
#    algo.kwargs.segment_length=1 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.agent.kwargs.n_steps=1000 \
#    algo.train.kwargs.n_rollout_steps=1000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.train.kwargs.total_timesteps=1000000

#run_exp christiano_SAC_seg_len_1_small_ensemble \
#    "${COMMON_SAC[@]}" \
#    algo.kwargs.segment_length=1 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.n_rollout_steps=1000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=2 \
#    algo.train.kwargs.total_timesteps=1000000

#run_exp christiano_SAC_seg_len_1_small_ensemble_moreQueries \
#    "${COMMON_SAC[@]}" \
#    algo.kwargs.segment_length=1 \
#    algo.train.kwargs.n_queries_per_iter=20 \
#    algo.train.kwargs.n_rollout_steps=1000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=2 \
#    algo.train.kwargs.total_timesteps=1000000

#run_exp christiano_SAC_seg_len_1_small_ensemble_400rolloutsteps \
#    "${COMMON_SAC[@]}" \
#    algo.kwargs.segment_length=1 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.n_rollout_steps=400 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=2 \
#    algo.train.kwargs.total_timesteps=1000000

#run_exp christiano_PPO_seg_len1_small_ensemble \
#    "${COMMON_PPO[@]}" \
#    algo.kwargs.segment_length=1 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.agent.kwargs.n_steps=1000 \
#    algo.train.kwargs.n_rollout_steps=1000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=2 \
#    algo.train.kwargs.total_timesteps=1000000

#run_exp christiano_PPO_long_seg_small_ensemble_and_rolloutSteps \
#    "${COMMON_PPO[@]}" \
#    algo.kwargs.segment_length=100 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.agent.kwargs.n_steps=400 \
#    algo.train.kwargs.n_rollout_steps=400 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=2 \
#    algo.train.kwargs.total_timesteps=1000000

#run_exp christiano_PPO_long_seg_small_ensemble_and_rolloutSteps_more_queries \
#    "${COMMON_PPO[@]}" \
#    algo.kwargs.segment_length=100 \
#    algo.train.kwargs.n_queries_per_iter=20 \
#    algo.agent.kwargs.n_steps=400 \
#    algo.train.kwargs.n_rollout_steps=400 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=2 \
#    algo.train.kwargs.total_timesteps=1000000

#run_exp christiano_SAC_seg_len_1_3ensemble \
#    "${COMMON_SAC[@]}" \
#    algo.kwargs.segment_length=1 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.n_rollout_steps=1000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.train.kwargs.total_timesteps=1000000
#
#run_exp christiano_PPO_seg_len1_smallensemble_moreQueriws \
#    "${COMMON_PPO[@]}" \
#    algo.kwargs.segment_length=1 \
#    algo.train.kwargs.n_queries_per_iter=20 \
#    algo.agent.kwargs.n_steps=1000 \
#    algo.train.kwargs.n_rollout_steps=1000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=2 \
#    algo.train.kwargs.total_timesteps=1000000
#
#run_exp christiano_PPO_seg_len1_3ensemble_moreQueriws \
#    "${COMMON_PPO[@]}" \
#    algo.kwargs.segment_length=1 \
#    algo.train.kwargs.n_queries_per_iter=20 \
#    algo.agent.kwargs.n_steps=1000 \
#    algo.train.kwargs.n_rollout_steps=1000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.train.kwargs.total_timesteps=1000000

# NEW

#run_exp christiano_SAC_seg_len_1_3ensemble \
#    "${COMMON[@]}" \
#    algo/agent=SAC \
#    algo.kwargs.segment_length=1 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.agent.kwargs.train_freq=8000 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.train.kwargs.total_timesteps=1000000
#
#run_exp christiano_PPO_seg_len1 \
#    "${COMMON[@]}" \
#    algo/agent=PPO \
#    algo.kwargs.segment_length=1 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.agent.kwargs.n_steps=1000 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.train.kwargs.total_timesteps=1000000

# New

#run_exp christiano_SAC_traj_len_withNormalization \
#    "${COMMON[@]}" \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=0.0003 \
#    algo.kwargs.reward_model_l2=0.0001 \
#    algo.kwargs.segment_length=None \
#    algo.kwargs.episode_length_estimate=10 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.agent.kwargs.learning_rate=0.0003 \
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp christiano_PPO_traj_len_withNormalization \
#    "${COMMON[@]}" \
#    algo/agent=PPO \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=0.0003 \
#    algo.kwargs.reward_model_l2=0.0001 \
#    algo.kwargs.segment_length=None \
#    algo.kwargs.episode_length_estimate=10 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.agent.kwargs.learning_rate=0.0003 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.agent.kwargs.n_epochs=10 \
#    algo.agent.kwargs.n_steps=1000
#
#
#run_exp christiano_SAC_shortSeg \
#    "${COMMON[@]}" \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=0.0003 \
#    algo.kwargs.reward_model_l2=0.0001 \
#    algo.kwargs.segment_length=1 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.agent.kwargs.learning_rate=0.0003 \
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp christiano_PPO_shortSeg \
#    "${COMMON[@]}" \
#    algo/agent=PPO \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=0.0003 \
#    algo.kwargs.reward_model_l2=0.0001 \
#    algo.kwargs.segment_length=1 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.agent.kwargs.learning_rate=0.0003 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.agent.kwargs.n_epochs=10 \
#    algo.agent.kwargs.n_steps=1000

#run_exp christiano_SAC_seg_len1 \
#    "${COMMON[@]}" \
#    algo/agent=SAC \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.n_initial_queries=200 \
#    algo.train.kwargs.n_queries_per_iter=10 \
#    algo.train.kwargs.reward_model_train_steps=200 \
#    algo.train.kwargs.reward_model_batch_size=64 \
#    algo.kwargs.reward_model_n_networks=3 \
#    algo.kwargs.reward_model_hidden_size=256 \
#    algo.kwargs.reward_model_lr=3.0e-4 \
#    algo.kwargs.reward_model_l2=1.0e-4 \
#    algo.kwargs.segment_length=1 \
#    algo.kwargs.episode_length_estimate=None \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.agent.kwargs.learning_rate=3.0e-4 \
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp SAC_trajLen_est1 \
#    "${COMMON[@]}" \
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
#    algo.kwargs.episode_length_estimate=1 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.buffer_size=100000 \
#    algo.agent.kwargs.learning_starts=100 \
#    algo.agent.kwargs.batch_size=256 \
#    algo.agent.kwargs.gradient_steps=1000 \
#    algo.agent.kwargs.train_freq=8000
#
#run_exp SAC_seg10 \
#    "${COMMON[@]}" \
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
#run_exp PPO_seg1 \
#    "${COMMON[@]}" \
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
#run_exp PPO_seg10 \
#    "${COMMON[@]}" \
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
#run_exp PPO_trajLen_est1 \
#    "${COMMON[@]}" \
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
#    algo.kwargs.episode_length_estimate=1 \
#    algo.kwargs.preference_dataset_max_size=3000 \
#    algo.kwargs.query_schedule=constant \
#    algo.agent.kwargs.learning_rate=3e-4 \
#    algo.agent.kwargs.batch_size=64 \
#    algo.agent.kwargs.n_epochs=10 \
#    algo.agent.kwargs.n_steps=1000

# with demo

run_exp demo_PPO_short_seg \
    "${COMMON[@]}" \
    algo=christiano_demo \
    algo/agent=PPO \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.n_initial_queries=200 \
    algo.train.kwargs.n_queries_per_iter=10 \
    algo.train.kwargs.reward_model_train_steps=200 \
    algo.train.kwargs.reward_model_batch_size=64 \
    algo.train.kwargs.n_sft_steps=200 \
    algo.train.kwargs.sft_batch_size=200 \
    algo.kwargs.reward_model_n_networks=2 \
    algo.kwargs.reward_model_hidden_size=256 \
    algo.kwargs.reward_model_lr=3e-4 \
    algo.kwargs.reward_model_l2=1e-4 \
    algo.kwargs.segment_length=1 \
    algo.kwargs.episode_length_estimate=None \
    algo.kwargs.preference_dataset_max_size=3000 \
    algo.kwargs.query_schedule=constant \
    algo.kwargs.demo_loss_weight=0.5 \
    algo.kwargs.expert_batch_size=64 \
    algo.kwargs.expert_dataset_max_size=3000 \
    algo.agent.kwargs.learning_rate=3e-4 \
    algo.agent.kwargs.batch_size=64 \
    algo.agent.kwargs.n_epochs=10 \
    algo.agent.kwargs.n_steps=1000

run_exp demo_SAC_short_seg \
    "${COMMON[@]}" \
    algo=christiano_demo \
    algo/agent=SAC \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.n_initial_queries=200 \
    algo.train.kwargs.n_queries_per_iter=10 \
    algo.train.kwargs.reward_model_train_steps=200 \
    algo.train.kwargs.reward_model_batch_size=64 \
    algo.train.kwargs.n_sft_steps=200 \
    algo.train.kwargs.sft_batch_size=200 \
    algo.kwargs.reward_model_n_networks=2 \
    algo.kwargs.reward_model_hidden_size=256 \
    algo.kwargs.reward_model_lr=3e-4 \
    algo.kwargs.reward_model_l2=1e-4 \
    algo.kwargs.segment_length=1 \
    algo.kwargs.episode_length_estimate=None \
    algo.kwargs.preference_dataset_max_size=3000 \
    algo.kwargs.query_schedule=constant \
    algo.kwargs.demo_loss_weight=0.5 \
    algo.kwargs.expert_batch_size=64 \
    algo.kwargs.expert_dataset_max_size=3000 \
    algo.agent.kwargs.learning_rate=3e-4 \
    algo.agent.kwargs.buffer_size=100000 \
    algo.agent.kwargs.learning_starts=100 \
    algo.agent.kwargs.batch_size=256 \
    algo.agent.kwargs.gradient_steps=1000 \
    algo.agent.kwargs.train_freq=8000

run_exp demo_PPO_full_traj \
    "${COMMON[@]}" \
    algo=christiano_demo \
    algo/agent=PPO \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.n_initial_queries=200 \
    algo.train.kwargs.n_queries_per_iter=10 \
    algo.train.kwargs.reward_model_train_steps=200 \
    algo.train.kwargs.reward_model_batch_size=64 \
    algo.train.kwargs.n_sft_steps=200 \
    algo.train.kwargs.sft_batch_size=200 \
    algo.kwargs.reward_model_n_networks=2 \
    algo.kwargs.reward_model_hidden_size=256 \
    algo.kwargs.reward_model_lr=3e-4 \
    algo.kwargs.reward_model_l2=1e-4 \
    algo.kwargs.segment_length=None \
    algo.kwargs.episode_length_estimate=10 \
    algo.kwargs.preference_dataset_max_size=3000 \
    algo.kwargs.query_schedule=constant \
    algo.kwargs.demo_loss_weight=0.5 \
    algo.kwargs.expert_batch_size=64 \
    algo.kwargs.expert_dataset_max_size=3000 \
    algo.agent.kwargs.learning_rate=3e-4 \
    algo.agent.kwargs.batch_size=64 \
    algo.agent.kwargs.n_epochs=10 \
    algo.agent.kwargs.n_steps=1000

run_exp demo_SAC_full_traj \
    "${COMMON[@]}" \
    algo=christiano_demo \
    algo/agent=SAC \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.n_initial_queries=200 \
    algo.train.kwargs.n_queries_per_iter=10 \
    algo.train.kwargs.reward_model_train_steps=200 \
    algo.train.kwargs.reward_model_batch_size=64 \
    algo.train.kwargs.n_sft_steps=200 \
    algo.train.kwargs.sft_batch_size=200 \
    algo.kwargs.reward_model_n_networks=2 \
    algo.kwargs.reward_model_hidden_size=256 \
    algo.kwargs.reward_model_lr=3e-4 \
    algo.kwargs.reward_model_l2=1e-4 \
    algo.kwargs.segment_length=None \
    algo.kwargs.episode_length_estimate=10 \
    algo.kwargs.preference_dataset_max_size=3000 \
    algo.kwargs.query_schedule=constant \
    algo.kwargs.demo_loss_weight=0.5 \
    algo.kwargs.expert_batch_size=64 \
    algo.kwargs.expert_dataset_max_size=3000 \
    algo.agent.kwargs.learning_rate=3e-4 \
    algo.agent.kwargs.buffer_size=100000 \
    algo.agent.kwargs.learning_starts=100 \
    algo.agent.kwargs.batch_size=256 \
    algo.agent.kwargs.gradient_steps=1000 \
    algo.agent.kwargs.train_freq=8000

run_exp PPO_only_SFTdemo_short_seg \
    "${COMMON[@]}" \
    algo=christiano_demo \
    algo/agent=PPO \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.n_initial_queries=200 \
    algo.train.kwargs.n_queries_per_iter=10 \
    algo.train.kwargs.reward_model_train_steps=200 \
    algo.train.kwargs.reward_model_batch_size=64 \
    algo.train.kwargs.n_sft_steps=200 \
    algo.train.kwargs.sft_batch_size=200 \
    algo.kwargs.reward_model_n_networks=2 \
    algo.kwargs.reward_model_hidden_size=256 \
    algo.kwargs.reward_model_lr=3e-4 \
    algo.kwargs.reward_model_l2=1e-4 \
    algo.kwargs.segment_length=1 \
    algo.kwargs.episode_length_estimate=None \
    algo.kwargs.preference_dataset_max_size=3000 \
    algo.kwargs.query_schedule=constant \
    algo.kwargs.demo_loss_weight=0.0 \
    algo.kwargs.expert_batch_size=64 \
    algo.kwargs.expert_dataset_max_size=3000 \
    algo.agent.kwargs.learning_rate=3e-4 \
    algo.agent.kwargs.batch_size=64 \
    algo.agent.kwargs.n_epochs=10 \
    algo.agent.kwargs.n_steps=1000

run_exp SAC_only_SFTdemo_short_seg \
    "${COMMON[@]}" \
    algo=christiano_demo \
    algo/agent=SAC \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.n_initial_queries=200 \
    algo.train.kwargs.n_queries_per_iter=10 \
    algo.train.kwargs.reward_model_train_steps=200 \
    algo.train.kwargs.reward_model_batch_size=64 \
    algo.train.kwargs.n_sft_steps=200 \
    algo.train.kwargs.sft_batch_size=200 \
    algo.kwargs.reward_model_n_networks=2 \
    algo.kwargs.reward_model_hidden_size=256 \
    algo.kwargs.reward_model_lr=3e-4 \
    algo.kwargs.reward_model_l2=1e-4 \
    algo.kwargs.segment_length=1 \
    algo.kwargs.episode_length_estimate=None \
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

run_exp PPO_only_SFTdemo_long_seg \
    "${COMMON[@]}" \
    algo=christiano_demo \
    algo/agent=PPO \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.n_initial_queries=200 \
    algo.train.kwargs.n_queries_per_iter=10 \
    algo.train.kwargs.reward_model_train_steps=200 \
    algo.train.kwargs.reward_model_batch_size=64 \
    algo.train.kwargs.n_sft_steps=200 \
    algo.train.kwargs.sft_batch_size=200 \
    algo.kwargs.reward_model_n_networks=2 \
    algo.kwargs.reward_model_hidden_size=256 \
    algo.kwargs.reward_model_lr=3e-4 \
    algo.kwargs.reward_model_l2=1e-4 \
    algo.kwargs.segment_length=None \
    algo.kwargs.episode_length_estimate=10 \
    algo.kwargs.preference_dataset_max_size=3000 \
    algo.kwargs.query_schedule=constant \
    algo.kwargs.demo_loss_weight=0.0 \
    algo.kwargs.expert_batch_size=64 \
    algo.kwargs.expert_dataset_max_size=3000 \
    algo.agent.kwargs.learning_rate=3e-4 \
    algo.agent.kwargs.batch_size=64 \
    algo.agent.kwargs.n_epochs=10 \
    algo.agent.kwargs.n_steps=1000

run_exp SAC_only_SFTdemo_long_seg \
    "${COMMON[@]}" \
    algo=christiano_demo \
    algo/agent=SAC \
    algo.train.kwargs.total_timesteps=1000000 \
    algo.train.kwargs.n_initial_queries=200 \
    algo.train.kwargs.n_queries_per_iter=10 \
    algo.train.kwargs.reward_model_train_steps=200 \
    algo.train.kwargs.reward_model_batch_size=64 \
    algo.train.kwargs.n_sft_steps=200 \
    algo.train.kwargs.sft_batch_size=200 \
    algo.kwargs.reward_model_n_networks=2 \
    algo.kwargs.reward_model_hidden_size=256 \
    algo.kwargs.reward_model_lr=3e-4 \
    algo.kwargs.reward_model_l2=1e-4 \
    algo.kwargs.segment_length=None \
    algo.kwargs.episode_length_estimate=10 \
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