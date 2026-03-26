import hydra
import sumo_rl_ego as sre

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3


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
            **cfg.env.kwargs
        )

        if cfg.algorithm.name == "christiano":
            print("Initializing agent...")
            algo_cls = ALGO_REGISTRY[cfg.algorithm.agent.algo]
            agent = algo_cls(**cfg.algorithm.agent.kwargs)

            print("Initializing algorithm...")
            algo = ChristianoAlgorithm(
                env=env,
                agent=agent,
                **cfg.algorithm.kwargs,
            )

        elif cfg.algorithm.name == "dagger":
            print("Initializing algorithm...")
            algo = DaggerAlgorithm(env, **cfg.algorithm.kwargs)

        elif cfg.algorithm.name == "humlrn-v0":
            print("Initializing agent...")
            algo_cls = ALGO_REGISTRY[cfg.algorithm.agent.algo]
            agent = algo_cls(**cfg.algorithm.agent.kwargs)

            print("Initializing algorithm...")
            algo = HumLrnAlgorithm_v0(
                env=env,
                agent=agent,
                **cfg.algorithm.kwargs,
            )


        print("Starting training...")
        agent = algo.train(**cfg.algorithm.train.kwargs)

        
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
