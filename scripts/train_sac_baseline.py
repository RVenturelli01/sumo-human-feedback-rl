"""
SAC baseline trained on the environment's TRUE reward (no reward model).
"""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import numpy as np

import sumo_gym_ego as sge
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecMonitor
from human_feedback_rl.common.custom_logging_callback import (
    CustomLoggingCallback,
    FixedIntervalDumpCallback,
)
from sumo_gym_ego import EgoStatus
from sumo_rl_ego.utils import run_episode
import wandb

from human_feedback_rl.common.loggers import WandbWriter, PrefixedLogger, Logger

from _common import init_wandb_run, make_run_dir, seed_everything


def get_name():
    group_name = "sac_baseline"
    # Name each run after the Hydra overrides it received. Under a sweep these are
    # exactly the swept hyperparameters for this trial, so every trial gets a
    # distinct, informative name (e.g. "learning_rate=0.0003_gamma=0.997") that
    # works automatically at every stage, whatever that stage sweeps. Standalone
    # runs (no overrides) fall back to the group name. All runs share the group,
    # so baseline runs stay together and separable from the demo-IRL experiments.
    overrides = HydraConfig.get().overrides.task
    parts = []
    for o in overrides:
        key, _, value = o.partition("=")
        # keep only the leaf of the dotpath to keep names short
        parts.append(f"{key.split('.')[-1]}={value}")
    run_name = "_".join(parts) if parts else group_name
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


def train_and_eval_one_seed(cfg, seed: int, env_kwargs: dict):
    """Train a fresh SAC agent on one seed and return (agent, eval_metrics).

    Each seed gets its own env that is fully closed before evaluation, since
    libsumo holds a single global SUMO instance — only one env may be open at a
    time. Training metrics are logged under an `agent_seed{seed}` prefix so the
    per-seed learning curves stay separate in W&B.
    """
    seed_everything(seed)

    env = sge.make_vec_env(cfg.env.id, n_envs=cfg.env.n_envs, base_seed=seed, **env_kwargs)
    agent = SAC(env=env, seed=seed, **OmegaConf.to_container(cfg.agent.kwargs, resolve=True))

    agent.set_env(VecMonitor(env))
    logger = Logger(folder=None, output_formats=[WandbWriter()])
    agent.set_logger(PrefixedLogger(logger, f"agent_seed{seed}"))

    callbacks = [CustomLoggingCallback()]
    log_interval = cfg.train.kwargs.log_interval
    dump_interval = cfg.train.get("agent_log_timestep_interval", None)
    if dump_interval is not None:
        callbacks.append(FixedIntervalDumpCallback(dump_interval))
        log_interval = None

    agent.learn(
        total_timesteps=cfg.train.kwargs.total_timesteps,
        log_interval=log_interval,
        callback=callbacks,
    )

    # Release the single global SUMO instance before opening the eval env.
    agent.env.close()
    metrics = evaluate(agent, cfg.env.id, env_kwargs, cfg.eval.n_episodes, cfg.eval.seed)
    return agent, metrics


@hydra.main(version_base=None, config_path="../configs", config_name="train_sac_baseline")
def main(cfg: DictConfig) -> None:
    group_name, run_name = get_name()
    run_dir = make_run_dir(cfg.run.output_dir, run_name or group_name)
    init_wandb_run(cfg, group_name, run_name, run_dir)

    print(OmegaConf.to_yaml(cfg))
    env_kwargs = OmegaConf.to_container(cfg.env.kwargs, resolve=True)
    seeds = list(cfg.run.seeds)

    # Train each seed sequentially and collect its deterministic eval metrics.
    # The sweep optimises the across-seed mean, so a lucky single seed can't win.
    per_seed_metrics = []
    for seed in seeds:
        print(f"\n===== Seed {seed} ({seeds.index(seed) + 1}/{len(seeds)}) =====")
        agent, metrics = train_and_eval_one_seed(cfg, seed, env_kwargs)
        per_seed_metrics.append(metrics)
        wandb.log({f"per_seed/seed{seed}/{k[len('eval/'):]}": v for k, v in metrics.items()})
        print(f"  seed {seed}: {metrics}")
        agent.save(str(run_dir / f"agent_seed{seed}"))

    # Aggregate across seeds: sweep/<metric> is the mean (the optimisation target),
    # sweep/<metric>_std is the dispersion across seeds.
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
