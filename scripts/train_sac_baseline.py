"""
SAC baseline trained on the environment's TRUE reward (no reward model).
"""

import hashlib

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
import wandb

from human_feedback_rl.common.loggers import WandbWriter, PrefixedLogger, Logger

from _common import evaluate, init_wandb_run, make_run_dir, seed_everything


def get_name():
    group_name = "sac_baseline"
    # Keep the filesystem/W&B run name short. The full resolved config is stored
    # in W&B; the hash only disambiguates override-heavy launcher or sweep runs.
    overrides = HydraConfig.get().overrides.task
    if not overrides:
        return group_name, group_name

    digest = hashlib.sha1("|".join(overrides).encode("utf-8")).hexdigest()[:8]
    visible_keys = {
        "learning_rate",
        "gamma",
        "tau",
        "ent_coef",
        "train_freq",
        "gradient_steps",
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


def train_and_eval_one_seed(cfg, seed: int, env_kwargs: dict):
    """Train a fresh SAC agent on one seed and return (agent, eval_metrics).

    Each seed gets its own env that is fully closed before evaluation, since
    libsumo holds a single global SUMO instance — only one env may be open at a
    time. Training metrics use the same `agent` prefix as DemoAlgorithm runs so
    W&B panels can compare the baseline against learned-reward agents.
    """
    seed_everything(seed)

    env = sge.make_vec_env(cfg.env.id, n_envs=cfg.env.n_envs, base_seed=seed, **env_kwargs)
    agent = SAC(env=env, seed=seed, **OmegaConf.to_container(cfg.agent.kwargs, resolve=True))

    agent.set_env(VecMonitor(env))
    logger = Logger(folder=None, output_formats=[WandbWriter()])
    agent.set_logger(PrefixedLogger(logger, "agent"))

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
