import argparse
import random

import numpy as np
import torch
import sumo_rl_ego as sre
from stable_baselines3 import PPO
from human_feedback_rl.algorithms.demo_rew.demo_rew_algorithm import ZhangAlgorithm
import wandb


# comps_per_iteration per ogni lunghezza di segmento.
# seg=1000 è cappato a 20 per mantenere num_steps ragionevole
# (num_steps = transition_oversampling × comps × seg_length, quindi 1×20×1000 = 20k step/iter).
COMPS_FOR_SEG = {
    1:    100,
    2:    200,
    5:    500,
    10:   1000,
    20:   2000,
    1000: 20,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg",  type=int, required=True, choices=list(COMPS_FOR_SEG))
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)

    comps_per_iter      = COMPS_FOR_SEG[args.seg]
    initial_comparisons = 2 * comps_per_iter

    run_name = f"demo-PPO seg={args.seg} comps={comps_per_iter} seed={args.seed}"
    # Group includes all varying hyperparameters except the seed.
    group    = f"demo_seg={args.seg}_comps={comps_per_iter}"

    wandb.init(
        project="meeting-demo-seg-study",
        entity="andrea02polimi-politecnico-di-milano",
        name=run_name,
        group=group,
        config={
            "segment_length":          args.seg,
            "comparisons_per_iter":    comps_per_iter,
            "initial_comparisons":     initial_comparisons,
            "comparison_queue_size":   50_000,
            "net_arch_rew":            "128x128",
            "n_ensembles_rew":         3,
            "timesteps_per_iteration": 20_000,
            "total_timesteps":         1_000_000,
            "expert":                  "ppo-fast",
            "seed":                    args.seed,
        },
    )

    print(f"Starting: {run_name}")
    print(f"  seg={args.seg}  comps_per_iter={comps_per_iter}  "
          f"initial_comparisons={initial_comparisons}  seed={args.seed}")

    env = sre.make_vec_env(
        "HighwayEgo-v0",
        n_envs=4,
        base_seed=args.seed,
        ego="continuous",
        reward="fast",
    )

    agent = PPO(
        env=env,
        policy="MlpPolicy",
        n_steps=1000,
        n_epochs=10,
        learning_rate=3e-4,
        batch_size=64,
        ent_coef=0.01,
        gae_lambda=0.95,
        gamma=0.995,
        policy_kwargs={"net_arch": [64, 64]},
        device="cpu",
        seed=args.seed,
    )

    print("Loading expert ppo-fast via sre.load_policy...")
    expert = sre.load_policy("ppo-fast", env=env)

    algo = ZhangAlgorithm(
        env=env,
        agent=agent,
        expert=expert,
        rng=np.random.default_rng(args.seed),
        lr_rew=3e-4,
        n_epochs_rew=1,
        batch_size_rew=128,
        l2_rew=1e-4,
        decay_rew=0.0,
        fragment_length=args.seg,
        transition_oversampling=1,
        initial_comparisons=initial_comparisons,
        initial_epoch_multiplier=1,
        query_schedule="constant",
        comparison_queue_size=50_000,
        n_ensembles_rew=3,
        train_comparison_frac=0.8,
        net_arch_rew=[128, 128],
    )

    checkpoint_dir = f"/storage/fis3/checkpoints/demo-seg-study/seg{args.seg}_comps{comps_per_iter}_seed{args.seed}"

    algo.train(
        total_timesteps=1_000_000,
        comparisons_per_iteration=comps_per_iter,
        timesteps_per_iteration=20_000,
        checkpoint_interval=10,
        checkpoint_dir=checkpoint_dir,
    )

    wandb.finish()
    print(f"Done: {run_name}")
