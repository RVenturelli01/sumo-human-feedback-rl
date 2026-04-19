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

run_exp exp_07_christiano-rew_model-pref-SAC-seg_len_10-ens_3 \
  "${COMMON[@]}" \
  algo=christiano \
  algo/agent=SAC \
  algo.train.kwargs.total_timesteps=1000000 \
  algo.train.kwargs.n_initial_queries=200 \
  algo.train.kwargs.n_queries_per_iter=10 \
  algo.train.kwargs.reward_model_train_steps=200 \
  algo.train.kwargs.reward_model_batch_size=64 \
  algo.kwargs.reward_model_n_networks=3 \
  algo.kwargs.reward_model_hidden_size=256 \
  algo.kwargs.reward_model_lr=3e-4 \
  algo.kwargs.reward_model_l2=1e-4 \
  algo.kwargs.segment_length=10 \
  algo.kwargs.episode_length_estimate=None \
  algo.kwargs.preference_dataset_max_size=3000 \
  algo.kwargs.query_schedule=constant \
  algo.agent.kwargs.learning_rate=3e-4 \
  algo.agent.kwargs.buffer_size=100000 \
  algo.agent.kwargs.learning_starts=100 \
  algo.agent.kwargs.batch_size=256 \
  algo.agent.kwargs.gradient_steps=1000 \
  algo.agent.kwargs.train_freq=8000