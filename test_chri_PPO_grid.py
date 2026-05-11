import argparse
import random

import numpy as np
import torch
import sumo_rl_ego as sre
from stable_baselines3 import PPO
from human_feedback_rl.algorithms import ChristianoAlgorithm
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg", type=int, required=True)
    parser.add_argument("--net-arch", type=str, required=True, help="e.g. 64,64")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--comps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def parse_net_arch(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parse_args()
    net_arch = parse_net_arch(args.net_arch)
    net_arch_str = "x".join(str(x) for x in net_arch)

    set_seeds(args.seed)

    config = {
        "env": {
            "id": "HighwayEgo-v0",
            "kwargs": {
                "n_envs": 4,
                "base_seed": args.seed,
                "ego": "continuous",
                "reward": "fast",
            },
        },
        "agent": {
            "policy": "MlpPolicy",
            "n_steps": 1000,
            "n_epochs": 10,
            "learning_rate": 0.0003,
            "batch_size": 64,
            "ent_coef": 0.01,
            "gae_lambda": 0.95,
            "gamma": 0.995,
            "policy_kwargs": {"net_arch": [64, 64]},
            "device": "cpu",
            "seed": args.seed,
        },
        "algo": {
            "lr_rew": 0.0003,
            "n_epochs_rew": 1,
            "batch_size_rew": 128,
            "l2_rew": 0.0001,
            "fragment_length": args.seg,
            "transition_oversampling": 1,
            "initial_comparison_frac": 0.1,
            "initial_epoch_multiplier": 1,
            "comparison_queue_size": 20_000,
            "n_ensembles_rew": 3,
            "train_comparison_frac": 0.8,
            "net_arch_rew": net_arch,
        },
        "train": {
            "total_timesteps": 1_000_000,
            "comparisons_per_iteration": args.comps,
            "timesteps_per_iteration": args.steps,
        },
    }

    run_name = (
        f"PPO seg={args.seg} net={net_arch_str}"
        f" steps={args.steps} comps={args.comps} seed={args.seed}"
    )
    group = f"seg={args.seg}_net={net_arch_str}_steps={args.steps}_comps={args.comps}"

    wandb.init(
        project="temp",
        entity="andrea02polimi-politecnico-di-milano",
        name=run_name,
        group=group,
        config={
            "segment_length": args.seg,
            "net_arch": net_arch_str,
            "timesteps_per_iteration": args.steps,
            "comparisons_per_iteration": args.comps,
            "seed": args.seed,
            "comparison_queue_size": config["algo"]["comparison_queue_size"],
            "n_ensembles_rew": config["algo"]["n_ensembles_rew"],
            "total_timesteps": config["train"]["total_timesteps"],
        },
    )

    print(f"Starting: {run_name}")

    env = sre.make_vec_env(config["env"]["id"], **config["env"]["kwargs"])
    agent = PPO(env=env, **config["agent"])
    algo = ChristianoAlgorithm(env=env, agent=agent, **config["algo"])
    algo.train(**config["train"])

    wandb.finish()
