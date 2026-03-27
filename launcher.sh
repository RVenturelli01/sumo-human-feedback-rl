
python scripts/eval.py \
    source.model_path=/home/ricca/projects/sumo-human-feedback-rl/outputs/train/2026-03-28_00-11-41_train_christiano_v0/model.zip \
    env.kwargs.ego=discrete \
    run.n_episodes=100 \
    run.seed=0 \
    run.output_dir=/home/ricca/projects/sumo-human-feedback-rl/sumo-rl-ego/outputs \
    run.name=eval_christiano_n100 \
    wandb.enabled=False \

# python scripts/train.py \
#     algo=christiano \
#     env.kwargs.ego=discrete \
#     run.seed=0 \
#     run.output_dir=/home/ricca/projects/sumo-human-feedback-rl/outputs \
#     run.name=train_christiano_v0 \
#     wandb.enabled=true \
#     wandb.kwargs.sync_tensorboard=false \


