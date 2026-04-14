import random

import hydra
import numpy as np
import torch
import sumo_rl_ego as sre

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from human_feedback_rl.algorithms import ChristianoAlgorithm, ChristianoPPOAlgorithm
from scripts.eval import EvalMetrics

from sumo_rl_ego.utils import (
    init_wandb,
    confirm_cfg,
    save_outputs,
    CustomLoggingCallback,
)

ALGO_REGISTRY = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}


def set_global_seeds(seed: int) -> None:
    """Set all global RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_train_cfg(cfg):
    print(f"\n========== TRAIN CONFIG ==========\n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("================== Summary ==================\n")
    print(f"Environment: {cfg.env.id} (x{cfg.env.n_envs} envs)")
    print(f"Environment arguments: {cfg.env.kwargs}")
    print(f"Algorithm: {cfg.algo.name}")
    print("\n=============================================\n")


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    _ = HydraConfig.get().runtime.output_dir

    set_global_seeds(cfg.run.seed)
    print_train_cfg(cfg)
    confirm_cfg()

    run = init_wandb(cfg)
    env = None

    try:
        print("Creating environment...")
        env = sre.make_vec_env(
            cfg.env.id,
            n_envs=cfg.env.n_envs,
            base_seed=cfg.run.seed,
            **cfg.env.kwargs,
        )

        if cfg.algo.name == "christiano":
            print("Initializing agent...")
            algo_cls = ALGO_REGISTRY[cfg.algo.agent.algo]
            agent = algo_cls(
                env=env,
                seed=cfg.run.seed,
                **OmegaConf.to_container(cfg.algo.agent.kwargs, resolve=True),
            )

            print("Initializing algorithm...")
            algo = ChristianoAlgorithm(
                env=env,
                agent=agent,
                rng=np.random.default_rng(cfg.run.seed),
                **OmegaConf.to_container(cfg.algo.kwargs, resolve=True),
            )
            with open_dict(cfg):
                cfg.model = {"algo": cfg.algo.agent.algo}

        elif cfg.algo.name == "christiano_ppo":
            print("Initializing agent...")
            agent = PPO(
                env=env,
                seed=cfg.run.seed,
                **OmegaConf.to_container(cfg.algo.agent.kwargs, resolve=True),
            )

            print("Initializing algorithm...")
            algo = ChristianoPPOAlgorithm(
                env=env,
                agent=agent,
                rng=np.random.default_rng(cfg.run.seed),
                **OmegaConf.to_container(cfg.algo.kwargs, resolve=True),
            )
            with open_dict(cfg):
                cfg.model = {"algo": "PPO"}

        else:
            raise ValueError(f"Unknown algorithm: {cfg.algo.name!r}")

        print("Starting training...")
        algo.train(**OmegaConf.to_container(cfg.algo.train.kwargs, resolve=True))

        print("\nTraining finished.")
        save_outputs(cfg, agent)
        print("Run completed successfully.\n")

        # eval
        print("Loading environment for evaluation...")
        eval_env = sre.make_env(
            cfg.env.id,
            seed=cfg.run.seed,
            **cfg.env.kwargs
        )

        try:
            policy = sre.ModelPolicy(agent)

            metrics = EvalMetrics()

            print("Running evaluation...")
            for _ in range(cfg.run.n_episodes_eval):
                info = sre.run_episode(
                    eval_env,
                    policy,
                    seed=cfg.run.seed,
                )

                metrics.add_episode(info)

            if run is not None:
                metrics.log_metrics()

            metrics.print_metrics()
        finally:
            eval_env.close()

    finally:
        if env is not None:
            env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
