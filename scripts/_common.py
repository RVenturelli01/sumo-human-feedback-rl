"""Shared helpers for the training entry-point scripts.

Keeps the boilerplate (run-dir creation, seeding, W&B init, data loading) in one
place so the per-algorithm scripts only contain what is specific to them.
"""

import pickle
import random
from pathlib import Path

import numpy as np
import torch as th
import wandb
from omegaconf import DictConfig, OmegaConf

from human_feedback_rl.common.loggers import configure_wandb_metrics

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"


def make_run_dir(output_dir: Path, name: str) -> Path:
    """Create `output_dir/name`, appending `_NN` if it already exists."""
    output_dir = Path(output_dir)
    candidate = output_dir / name
    i = 0
    while candidate.exists():
        i += 1
        candidate = output_dir / f"{name}_{i:02d}"
    candidate.mkdir(parents=True)
    return candidate


def seed_everything(seed: int) -> np.random.Generator:
    """Seed python, numpy and torch; return a seeded numpy Generator."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    return np.random.default_rng(seed)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_expert_trajectories(name: str = "expert_trajectories_no_collision.pkl") -> list:
    """Load the expert trajectories pickle from data_for_training/."""
    return load_pickle(DATA_DIR / name)


def load_debug_dataset(name: str = "debug_dataset.pkl"):
    """Load the optional debug dataset, or None if it is not present."""
    path = DATA_DIR / name
    return load_pickle(path) if path.exists() else None


def init_wandb_run(cfg: DictConfig, group_name: str, run_name: str, run_dir: Path) -> None:
    """Initialise a W&B run, recording the resolved Hydra config."""
    config = OmegaConf.to_container(cfg, resolve=True)
    config["group_name"] = group_name
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=config,
        group=group_name,
        name=run_name,
        dir=str(run_dir),
    )
    configure_wandb_metrics(run)
