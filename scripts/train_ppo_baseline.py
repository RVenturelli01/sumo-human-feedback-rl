"""
PPO baseline trained on the environment's TRUE reward (no reward model).
"""

import hashlib

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import numpy as np

import sumo_gym_ego as sge
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from human_feedback_rl.common.custom_logging_callback import CustomLoggingCallback
from sumo_gym_ego import EgoStatus
from sumo_rl_ego.utils import run_episode
import wandb

from human_feedback_rl.common.loggers import WandbWriter, PrefixedLogger, Logger

from _common import init_wandb_run, make_run_dir, seed_everything


def get_name():
    group_name = "ppo_baseline"
    overrides = HydraConfig.get().overrides.task
    if not overrides:
        return group_name, group_name

    digest = hashlib.sha1("|".join(overrides).encode("utf-8")).hexdigest()[:8]
    visible_keys = {
        "learning_rate",
        "gamma",
        "n_steps",
        "n_epochs",
        "batch_size",
        "ent_coef",
    }
    parts = []
    for override in overrides:
        key, _, value = override.partition("=")
        leaf = key.split(".")[-1]
        if leaf in visible_keys:
            parts.append(f"{leaf}={value}")
    label = "_".join(parts[:4])
    run_name = f"{group_name}_{label}_{digest}" if label else f"{group_name}_{digest}"
    return group_name, run_name


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
            info = run_episode(env, policy, seed=seed + ep)
            ep_metrics = info.get("metrics", {}).get("episode", {})
            fast_returns.append(float(ep_metrics.get("rewards/ep_fast_return", np.nan)))
            comfort_returns.append(float(ep_metrics.get("rewards/ep_comfort_return", np.nan)))
            speeds.append(float(ep_metrics.get("performance/ep_avg_speed", np.nan)))
            lengths.append(float(info.get("step", 0)))

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


def train_and_eval_one_seed(cfg, seed: int, env_kwargs: dict):
    """Train a fresh PPO agent on one seed and return (agent, eval_metrics)."""
    seed_everything(seed)

    env = sge.make_vec_env(cfg.env.id, n_envs=cfg.env.n_envs, base_seed=seed, **env_kwargs)
    agent = PPO(env=env, seed=seed, **OmegaConf.to_container(cfg.agent.kwargs, resolve=True))

    agent.set_env(VecMonitor(env))
    logger = Logger(folder=None, output_formats=[WandbWriter()])
    agent.set_logger(PrefixedLogger(logger, "agent"))

    agent.learn(
        total_timesteps=cfg.train.kwargs.total_timesteps,
        log_interval=cfg.train.kwargs.log_interval,
        callback=CustomLoggingCallback(),
    )

    agent.env.close()
    metrics = evaluate(agent, cfg.env.id, env_kwargs, cfg.eval.n_episodes, cfg.eval.seed)
    return agent, metrics


@hydra.main(version_base=None, config_path="../configs", config_name="train_ppo_baseline")
def main(cfg: DictConfig) -> None:
    group_name, run_name = get_name()
    run_dir = make_run_dir(cfg.run.output_dir, run_name or group_name)
    init_wandb_run(cfg, group_name, run_name, run_dir)

    print(OmegaConf.to_yaml(cfg))
    env_kwargs = OmegaConf.to_container(cfg.env.kwargs, resolve=True)
    seeds = list(cfg.run.seeds)

    per_seed_metrics = []
    for seed in seeds:
        print(f"\n===== Seed {seed} ({seeds.index(seed) + 1}/{len(seeds)}) =====")
        agent, metrics = train_and_eval_one_seed(cfg, seed, env_kwargs)
        per_seed_metrics.append(metrics)
        wandb.log({f"per_seed/seed{seed}/{k[len('eval/'):]}": v for k, v in metrics.items()})
        print(f"  seed {seed}: {metrics}")
        agent.save(str(run_dir / f"agent_seed{seed}"))

    aggregated = {}
    for key in per_seed_metrics[0]:
        short = key[len("eval/"):]
        values = [m[key] for m in per_seed_metrics]
        aggregated[f"sweep/{short}"] = float(np.mean(values))
        aggregated[f"sweep/{short}_std"] = float(np.std(values))
    wandb.log(aggregated)
    for k, v in aggregated.items():
        wandb.run.summary[k] = v
    print(f"\nAggregated over {len(seeds)} seeds: {aggregated}")

    wandb.finish()


if __name__ == "__main__":
    main()
