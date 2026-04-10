import hydra
import numpy as np
import sumo_rl_ego as sre

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from human_feedback_rl.algorithms import ChristianoAlgorithm

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

        else:
            raise ValueError(f"Unknown algorithm: {cfg.algo.name!r}")

        print("Starting training...")
        algo.train(**OmegaConf.to_container(cfg.algo.train.kwargs, resolve=True))

        print("\nTraining finished.")
        save_outputs(cfg, agent)
        print("Run completed successfully.\n")

    finally:
        if env is not None:
            env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
