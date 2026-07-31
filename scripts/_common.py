"""Shared helpers for the training entry-point scripts.

Keeps the boilerplate (run-dir creation, seeding, W&B init, data loading) in one
place so the per-algorithm scripts only contain what is specific to them.
"""

import pickle
import random
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch as th
import wandb
from omegaconf import DictConfig, OmegaConf

import sumo_gym_ego as sge
from sumo_gym_ego import EgoStatus
from sumo_rl_ego.utils import run_episode

from human_feedback_rl.common.demo_subsampling import (
    DEMO_SUBSAMPLE_SEED,
    select_demo_indices,
    subsample_manifest,
)
from human_feedback_rl.common.loggers import configure_wandb_metrics

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"
MAX_RUN_DIR_NAME_LENGTH = 180


def _shorten_path_component(name: str, max_length: int = MAX_RUN_DIR_NAME_LENGTH) -> str:
    """Keep run directory names below common filesystem component limits."""
    if len(name) <= max_length:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{name[: max_length - len(digest) - 1]}_{digest}"


def make_run_dir(output_dir: Path, name: str) -> Path:
    """Create `output_dir/name`, appending `_NN` if it already exists."""
    output_dir = Path(output_dir)
    name = _shorten_path_component(str(name))
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


def load_expert_trajectories(
    name: str = "expert_trajectories_no_collision.pkl",
    n_trajectories: Optional[int] = None,
    seed: Optional[int] = None,
    n_transitions: Optional[int] = None,
    return_manifest: bool = False,
):
    """Load the expert trajectories pickle from datasets/.

    With ``n_trajectories`` set, keep a seeded random subset: the prefix of one
    permutation, so at a fixed seed smaller budgets are nested inside larger
    ones (clean demo-budget curves).

    With ``n_transitions`` set instead, budget the demonstrations in
    TRANSITIONS rather than trajectories: take whole trajectories from the same
    seeded permutation while the cumulative length stays within the cap (always
    at least one). One expert trajectory is ~175 transitions here, so a
    nominally "homogeneous" split like 500 preferences + 500 trajectories
    actually feeds the demo channel ~175x more data than the preference one;
    this knob makes the two channels comparable per transition. Nesting is
    preserved for this budget too (same permutation prefix).

    The two budgets are mutually exclusive.

    ``seed=None`` means :data:`DEMO_SUBSAMPLE_SEED`, the shared constant that
    makes the demo-only and hybrid arms read the same demonstrations at the
    same budget. It is deliberately NOT the run seed. With
    ``return_manifest=True`` the selection is returned alongside the
    trajectories as a dict (indices, fingerprint, dataset identity) for the
    run to persist.
    """
    trajectories = load_pickle(DATA_DIR / name)
    lengths = [len(trajectory) for trajectory in trajectories]
    indices = select_demo_indices(
        n_available=len(trajectories),
        lengths=lengths,
        n_trajectories=n_trajectories,
        n_transitions=n_transitions,
        seed=seed,
    )
    subset = [trajectories[i] for i in indices]
    # --- EXTENSION PLACEHOLDER: demonstration noise -------------------------
    # Planned experiment ("rumore sulle dimostrazioni"): corrupt the expert
    # data right here, after subsampling, via a new knob, e.g.
    # ``demo_noise: float = 0.0`` (seeded with the same rng):
    #   * action noise — add clipped Gaussian noise to each transition's
    #     action (std = demo_noise * action range), i.e. a sloppier expert;
    #   * or trajectory swap — replace a demo_noise fraction of expert
    #     trajectories with random agent-quality ones (mislabeled demos).
    # Loading is the single choke point every consumer goes through, so the
    # corruption applies uniformly to the IRL losses AND to the
    # demos-as-preferences pairs. With demo_noise=0.0 behaviour is unchanged.
    if not return_manifest:
        return subset
    manifest = subsample_manifest(
        indices=indices,
        lengths=lengths,
        seed=seed,
        n_trajectories=n_trajectories,
        n_transitions=n_transitions,
        dataset_name=name,
    )
    return subset, manifest


def load_debug_dataset(name: str = "debug_dataset.pkl"):
    """Load the optional debug dataset, or None if it is not present."""
    path = DATA_DIR / name
    return load_pickle(path) if path.exists() else None


class _SB3PolicyAdapter:
    """Wraps an SB3 model to match the policy interface used by run_episode."""

    def __init__(self, model):
        self.model = model

    def reset(self):
        pass

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


def evaluate(model, env_id: str, env_kwargs: dict, n_episodes: int, seed: int) -> dict:
    """Run `n_episodes` deterministic episodes on a fresh env; return mean metrics."""
    env = sge.make_env(env_id, seed=seed, **env_kwargs)
    policy = _SB3PolicyAdapter(model)
    fast_returns, comfort_returns, speeds, lengths = [], [], [], []
    successes, collisions, off_roads, timeouts = [], [], [], []
    try:
        for ep in range(n_episodes):
            # Vary the eval seed per episode so we average over distinct scenarios.
            info = run_episode(env, policy, seed=seed + ep)
            ep_metrics = info.get("metrics", {}).get("episode", {})
            fast_returns.append(float(ep_metrics.get("rewards/ep_fast_return", np.nan)))
            comfort_returns.append(float(ep_metrics.get("rewards/ep_comfort_return", np.nan)))
            speeds.append(float(ep_metrics.get("performance/ep_avg_speed", np.nan)))
            lengths.append(float(info.get("step", 0)))

            # The four terminal statuses are mutually exclusive and partition the
            # outcomes, so their rates characterise *how* the agent fails, not just
            # that it does.
            status = info.get("ego_status", EgoStatus.RUNNING)
            successes.append(int(status == EgoStatus.ARRIVED.value))
            collisions.append(int(status == EgoStatus.COLLIDED.value))
            off_roads.append(int(status == EgoStatus.OFF_ROAD.value))
            timeouts.append(int(status == EgoStatus.TIMEOUT.value))
    finally:
        env.close()
    return {
        "eval/mean_fast_return": float(np.nanmean(fast_returns)),
        "eval/mean_comfort_return": float(np.nanmean(comfort_returns)),
        "eval/mean_speed": float(np.nanmean(speeds)),
        "eval/mean_ep_length": float(np.mean(lengths)),
        "eval/success_rate": float(np.mean(successes)),
        "eval/collision_rate": float(np.mean(collisions)),
        "eval/off_road_rate": float(np.mean(off_roads)),
        "eval/timeout_rate": float(np.mean(timeouts)),
    }


def log_sweep_summary(metrics: dict) -> dict:
    """Mirror eval/<k> metrics as sweep/<k> in the W&B log and run summary.

    Same naming convention as the baseline scripts, so sweeps/best_params.py
    and existing sweep configs work unchanged on these runs.
    """
    summary = {f"sweep/{k[len('eval/'):]}": v for k, v in metrics.items()}
    wandb.log(summary)
    for k, v in summary.items():
        wandb.run.summary[k] = v
    return summary


def init_wandb_run(cfg: DictConfig, group_name: str, run_name: str, run_dir: Path) -> None:
    """Initialise a W&B run, recording the resolved Hydra config."""
    config = OmegaConf.to_container(cfg, resolve=True)
    config["group_name"] = group_name
    raw_tags = cfg.wandb.get("tags", None)
    tags = OmegaConf.to_container(raw_tags, resolve=True) if raw_tags is not None else None
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=config,
        group=group_name,
        name=run_name,
        dir=str(run_dir),
        tags=tags,
    )
    configure_wandb_metrics(run)
