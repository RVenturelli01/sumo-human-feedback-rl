"""Train Guided Cost Learning on the SUMO highway environment."""

import hashlib
import re
from pathlib import Path

import hydra
import torch as th
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, SAC

import sumo_gym_ego as sge
from human_feedback_rl.algorithms import GuidedCostLearning

from _common import init_wandb_run, load_expert_trajectories, make_run_dir, seed_everything


AGENTS = {
    "SAC": SAC,
    "PPO": PPO,
}


def get_name(cfg: DictConfig) -> tuple[str, str]:
    group_name = f"gcl_{cfg.agent.algo.lower()}_{cfg.env.kwargs.reward}"
    overrides = HydraConfig.get().overrides.task
    if not overrides:
        return group_name, f"{group_name} seed={cfg.run.seed}"

    parts = []
    for override in overrides:
        key, _, value = override.partition("=")
        parts.append(f"{key.split('.')[-1]}={value}")
    raw = "_".join(parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    short = "_".join(parts[:5])
    safe = re.sub(r"[^A-Za-z0-9_.=-]+", "-", short).strip("-")
    return group_name, f"{safe}_{digest}"[:120]


def select_expert_trajectories(trajectories, max_trajectories: int, rng):
    if max_trajectories is None or max_trajectories <= 0:
        return list(trajectories)
    if max_trajectories >= len(trajectories):
        return list(trajectories)
    indices = rng.choice(len(trajectories), size=max_trajectories, replace=False)
    return [trajectories[int(i)] for i in indices]


@hydra.main(version_base=None, config_path="../configs", config_name="guided_cost_learning")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.run.seed)
    rng = seed_everything(seed)

    group_name, run_name = get_name(cfg)
    run_dir = make_run_dir(Path(cfg.run.output_dir), run_name)
    init_wandb_run(cfg, group_name, run_name, run_dir)

    print(OmegaConf.to_yaml(cfg))

    print("Loading expert demonstrations...")
    expert_trajectories = load_expert_trajectories(cfg.demonstrations.dataset)
    expert_trajectories = select_expert_trajectories(
        expert_trajectories,
        int(cfg.demonstrations.max_trajectories),
        rng,
    )
    print(f"Loaded {len(expert_trajectories)} expert trajectories.")

    print("Creating SUMO vector environment...")
    env = sge.make_vec_env(
        cfg.env.id,
        n_envs=int(cfg.env.n_envs),
        base_seed=seed,
        **OmegaConf.to_container(cfg.env.kwargs, resolve=True),
    )

    print(f"Initializing {cfg.agent.algo} agent...")
    agent_cls = AGENTS[cfg.agent.algo]
    agent = agent_cls(
        env=env,
        seed=seed,
        **OmegaConf.to_container(cfg.agent.kwargs, resolve=True),
    )

    checkpoint_dir = run_dir / "checkpoints"
    print("Initializing Guided Cost Learning...")
    algo = GuidedCostLearning(
        env=env,
        agent=agent,
        expert_trajectories=expert_trajectories,
        rng=rng,
        log_folder=str(run_dir / "logs"),
        **OmegaConf.to_container(cfg.algo.kwargs, resolve=True),
    )

    print("Starting training...")
    train_kwargs = OmegaConf.to_container(cfg.train.kwargs, resolve=True)
    train_kwargs["checkpoint_dir"] = str(checkpoint_dir)
    trained_agent = algo.train(**train_kwargs)

    final_agent_path = run_dir / "agent_final"
    final_cost_path = run_dir / "cost_model_final.pt"
    trained_agent.save(str(final_agent_path))
    th.save(algo.cost_model.state_dict(), final_cost_path)
    print(f"Saved final agent -> {final_agent_path}.zip")
    print(f"Saved final cost model -> {final_cost_path}")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
