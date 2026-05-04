#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

#python scripts/train.py \
#  algo=christiano_demo \
#  wandb.kwargs.project=christiano_demo \
#  algo/agent=PPO \
#  env.kwargs.ego=continuous \
#  run.name=testDemo1 \
#  run.output_dir="$REPO_ROOT/outputs" \
#  wandb.enabled=true \


#python scripts/train.py \
#    algo=christiano \
#    wandb.kwargs.project=regularization_analysis \
#    algo/agent=PPO \
#    env.kwargs.ego=continuous \
#    run.name=segLen15-withRegularization0.5-PPO \
#    run.seed=0 \
#    run.output_dir="$REPO_ROOT/outputs" \
#    wandb.enabled=true \
#    algo.kwargs.n_ensembles_rew=3 \
#    algo.kwargs.lr_rew=3e-4 \
#    algo.kwargs.batch_size_rew=64 \
#    algo.kwargs.n_ephochs_rew=3 \
#    algo.kwargs.n_iterations=50 \
#    algo.kwargs.train_comparison_frac=0.8 \
#    algo.kwargs.fragment_length=15 \
#    algo.kwargs.transition_oversampling=2.0 \
#    algo.kwargs.initial_comparison_frac=0.1 \
#    algo.kwargs.initial_epoch_multiplier=1.0 \
#    algo.kwargs.query_schedule=constant \
#    algo.kwargs.use_reward_reg=true \
#    algo.kwargs.reward_mean_reg=0.5 \
#    algo.train.kwargs.total_timesteps=1000000 \
#    algo.train.kwargs.total_comparisons=2500

#COMP=(2500 2000 1500 1000 500 100 50)
#
#for COMP in "${COMP[@]}"; do
#    RUN_NAME="TOTAL_COMPARISONS_${COMP}"
#
#    echo "===================================================="
#    echo "Running experiment: $RUN_NAME"
#    echo "TOTAL_COMPARISONS_=$COMP"
#    echo "===================================================="
#
#    echo "y" | python scripts/train.py \
#        algo=christiano \
#        algo/agent=PPO \
#        env.kwargs.ego=continuous \
#        run.name="$RUN_NAME" \
#        run.seed=0 \
#        run.output_dir="$REPO_ROOT/outputs" \
#        wandb.enabled=true \
#        algo.kwargs.n_ensembles_rew=1 \
#        algo.kwargs.lr_rew=3e-4 \
#        algo.kwargs.batch_size_rew=64 \
#        algo.kwargs.n_ephochs_rew=20 \
#        algo.kwargs.n_iterations=50 \
#        algo.kwargs.train_comparison_frac=0.8 \
#        algo.kwargs.fragment_length=1 \
#        algo.kwargs.transition_oversampling=10.0 \
#        algo.kwargs.initial_comparison_frac=0.1 \
#        algo.kwargs.initial_epoch_multiplier=10.0 \
#        algo.kwargs.query_schedule=constant \
#        algo.train.kwargs.total_timesteps=1000000 \
#        algo.train.kwargs.total_comparisons="$COMP"
#done

SEG_LENS=(1 2 10 20 50)

for SEG_LEN in "${SEG_LENS[@]}"; do
    RUN_NAME="with_regularization_debug_seg_${SEG_LEN}"

    echo "===================================================="
    echo "Running experiment: $RUN_NAME"
    echo "fragment_length=$SEG_LEN"
    echo "===================================================="

    echo "y" | python scripts/train.py \
        algo=christiano \
        algo/agent=PPO \
        env.kwargs.ego=continuous \
        run.name="$RUN_NAME" \
        run.seed=0 \
        run.output_dir="$REPO_ROOT/outputs" \
        wandb.enabled=true \
        wandb.kwargs.project=debug \
        algo.kwargs.n_ensembles_rew=3 \
        algo.kwargs.lr_rew=3e-4 \
        algo.kwargs.batch_size_rew=64 \
        algo.kwargs.n_ephochs_rew=3 \
        algo.kwargs.n_iterations=50 \
        algo.kwargs.train_comparison_frac=0.8 \
        algo.kwargs.fragment_length="$SEG_LEN" \
        algo.kwargs.transition_oversampling=1.0 \
        algo.kwargs.initial_comparison_frac=0.1 \
        algo.kwargs.initial_epoch_multiplier=1.0 \
        algo.kwargs.use_reward_reg=true \
        algo.kwargs.reward_mean_reg=0.5 \
        algo.kwargs.query_schedule=constant \
        algo.train.kwargs.total_timesteps=1000000 \
        algo.train.kwargs.total_comparisons=2500
done