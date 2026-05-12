import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from pathlib import Path

import sumo_rl_ego as sre
from stable_baselines3 import PPO
from human_feedback_rl.algorithms import ChristianoAlgorithm
from sumo_rl_ego.utils import CustomLoggingCallback
import wandb

from human_feedback_rl.common.loggers import WandbWriter, PrefixedLogger
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import VecMonitor


def get_name(cfg):
    total_timesteps = cfg["train"]["total_timesteps"]
    timesteps_per_it = cfg["train"]["timesteps_per_iteration"]
    n_envs = cfg["env"]["kwargs"]["n_envs"]
    n_steps = cfg["agent"]["n_steps"]

    base = (
        f"PPO"
        f" {n_envs}x{n_steps}={n_envs * n_steps}"
        f" tot_steps={total_timesteps}"
        f" steps_per_it={timesteps_per_it}"
    )

    if cfg["baseline"]:
        return base + " (baseline)"

    comparisons_per_it = cfg["train"]["comparisons_per_iteration"]
    segment_length = cfg["algo"]["fragment_length"]

    return base + f" comp_per_it={comparisons_per_it} seg_len={segment_length} (Christiano synch)"


def print_summary(cfg):
    train = cfg["train"]
    agent = cfg["agent"]
    env = cfg["env"]["kwargs"]

    steps_per_it = train["timesteps_per_iteration"]
    tot_steps = train["total_timesteps"]
    rollout_len = env["n_envs"] * agent["n_steps"]
    n_it = int(tot_steps / steps_per_it)

    print("=" * 60)
    print(f"  {get_name(cfg)}")
    print("=" * 60)
    print(f"  Iterations:       {n_it}")
    print(f"  Timesteps:        {steps_per_it:>4} / it    total: {steps_per_it * n_it:>10}")
    print(f"  Rollout length:   {rollout_len}  ({env['n_envs']} envs x {agent['n_steps']} steps)")

    if not cfg["baseline"]:
        algo = cfg["algo"]
        comp_per_it = train["comparisons_per_iteration"]
        seg_len = algo["fragment_length"]
        print(f"  Comparisons:      {comp_per_it:>4} / it    total: {comp_per_it * n_it:>10}")
        print(f"  Segment length:   {seg_len}")
        print("-" * 60)
        print("  Reward model:")
        print(f"    lr:                    {algo['lr_rew']}")
        print(f"    n_epochs:        {algo['n_epochs_rew']}")
        print(f"    batch size:            {algo['batch_size_rew']}")
        print(f"    comparison_queue_size: {algo['comparison_queue_size']}")

    print("=" * 60)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    wandb.init(
        project="temp",
        entity="rventurelli-politecnico-di-milano",
        config=OmegaConf.to_container(cfg, resolve=True),
        name=get_name(cfg),
        dir=str(run_dir),
    )

    print_summary(cfg)

    print("Creating environment...")
    env = sre.make_vec_env(cfg.env.id, **OmegaConf.to_container(cfg.env.kwargs, resolve=True))

    print("Initializing agent...")
    agent = PPO(env=env, **OmegaConf.to_container(cfg.agent, resolve=True))

    if cfg.baseline:
        agent.set_env(VecMonitor(env))
        logger = Logger(folder=None, output_formats=[WandbWriter()])
        agent.set_logger(PrefixedLogger(logger, "agent"))

        agent.learn(
            total_timesteps=cfg.train.total_timesteps,
            callback=CustomLoggingCallback(),
        )

    else:
        print("Initializing algorithm...")
        algo = ChristianoAlgorithm(env=env, agent=agent, **OmegaConf.to_container(cfg.algo, resolve=True))

        print("Starting training...")
        train_kwargs = OmegaConf.to_container(cfg.train, resolve=True)
        train_kwargs["checkpoint_dir"] = str(run_dir)
        agent = algo.train(**train_kwargs)


if __name__ == '__main__':
    main()
