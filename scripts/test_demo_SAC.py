import pickle
import random
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch as th

import sumo_rl_ego as sre
from stable_baselines3 import SAC
from human_feedback_rl.algorithms import DemoAlgorithm
import wandb


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
    seed = cfg.run.seed
    group_name = "sac_demo_irl"
    run_name = group_name + f" seed={seed}"
    return group_name, run_name


def load_expert_trajectories(path: str) -> list:
    """Load trajectories from a directory of .pkl files or a single .pkl file."""
    p = Path(path)
    if p.is_dir():
        trajectories = []
        for pkl_file in sorted(p.glob("*.pkl")):
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            trajectories.extend(data if isinstance(data, list) else [data])
        return trajectories
    with open(p, "rb") as f:
        data = pickle.load(f)
    return data if isinstance(data, list) else [data]


@hydra.main(version_base=None, config_path="../configs", config_name="test_demo_SAC")
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

    print(OmegaConf.to_yaml(cfg))

    print(f"Loading expert trajectories...")
    expert_trajectories_path = Path(__file__).parent.parent / "data_for_training/expert_trajectories.pkl"

    with open(expert_trajectories_path, "rb") as f:
        expert_trajectories = pickle.load(f)

    print(f"Loaded {len(expert_trajectories)} expert trajectories")

    print("Creating environment...")
    env = sre.make_vec_env(
        cfg.env.id,
        n_envs=cfg.env.n_envs,
        base_seed=seed,
        **OmegaConf.to_container(cfg.env.kwargs, resolve=True),
    )

    debug_dataset_path = Path(__file__).parent.parent / "data_for_training/debug_dataset.pkl"
    debug_dataset = None
    if debug_dataset_path.exists():
        with open(debug_dataset_path, "rb") as f:
            debug_dataset = pickle.load(f)

    print("Initializing agent...")
    agent = SAC(env=env, seed=seed, **OmegaConf.to_container(cfg.agent.kwargs, resolve=True))

    print("Initializing DemoAlgorithm...")
    algo = DemoAlgorithm(
        env=env,
        agent=agent,
        expert_trajectories=expert_trajectories,
        rng=rng,
        debug_dataset=debug_dataset,
        **OmegaConf.to_container(cfg.algo.kwargs, resolve=True),
    )

    print("Starting training...")
    train_kwargs = OmegaConf.to_container(cfg.train.kwargs, resolve=True)
    train_kwargs["checkpoint_dir"] = str(run_dir)
    algo.train(**train_kwargs)


if __name__ == '__main__':
    main()
