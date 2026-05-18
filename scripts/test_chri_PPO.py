import random

import hydra
from omegaconf import DictConfig, OmegaConf

from pathlib import Path

import numpy as np
import torch as th

import sumo_rl_ego as sre
from stable_baselines3 import PPO
from human_feedback_rl.algorithms import ChristianoAlgorithm
from sumo_rl_ego.utils import CustomLoggingCallback
import wandb

from human_feedback_rl.common.loggers import WandbWriter, PrefixedLogger, Logger
from stable_baselines3.common.vec_env import VecMonitor


def make_run_dir(output_dir: Path, name: str) -> Path:
    candidate = output_dir / name
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate
    i = 1
    while True:
        candidate = output_dir / f"{name}_{i:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        i += 1


def get_name(cfg):
    type = "baseline" if cfg.run.baseline else f"chri"
    
    total_queries = cfg.train.kwargs.total_queries
    segment_length = cfg.algo.kwargs.fragment_length
    hard_labels = cfg.algo.kwargs.hard_labels
    seed = cfg.run.seed

    group_name = (
        f"ppo_{type}"
        f" seg_len={segment_length}"
        f" tot_queries={total_queries}"
        f" hard_labels={hard_labels}"
    )

    run_name = group_name + f" seed={seed}"
    
    return group_name, run_name



@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    
    seed = cfg.run.seed
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    rng = np.random.default_rng(seed)

    group_name, run_name = get_name(cfg)
    run_dir = make_run_dir(Path(cfg.run.output_dir), run_name)

    config = OmegaConf.to_container(cfg, resolve=True)
    config["group_name"] = group_name

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=config,
        group=group_name,
        name=run_name,
        dir=str(run_dir),
    )

    print(cfg)

    print("Creating environment...")
    env = sre.make_vec_env(cfg.env.id, n_envs=cfg.env.n_envs, base_seed=seed, **OmegaConf.to_container(cfg.env.kwargs, resolve=True))

    print("Initializing agent...")
    agent = PPO(env=env, seed=seed, **OmegaConf.to_container(cfg.agent.kwargs, resolve=True))

    if cfg.run.baseline:
        agent.set_env(VecMonitor(env))
        logger = Logger(folder=None, output_formats=[WandbWriter()])
        agent.set_logger(PrefixedLogger(logger, "agent"))

        agent.learn(
            total_timesteps=cfg.train.kwargs.total_timesteps,
            callback=CustomLoggingCallback(),
        )

    else:
        print("Initializing algorithm...")
        algo = ChristianoAlgorithm(env=env, agent=agent, rng=rng, **OmegaConf.to_container(cfg.algo.kwargs, resolve=True))

        print("Starting training...")
        train_kwargs = OmegaConf.to_container(cfg.train.kwargs, resolve=True)
        train_kwargs["checkpoint_dir"] = str(run_dir)
        agent = algo.train(**train_kwargs)


if __name__ == '__main__':
    main()
