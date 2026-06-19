import hydra
from omegaconf import DictConfig, OmegaConf

import sumo_rl_ego as sre
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from sumo_rl_ego.utils import CustomLoggingCallback
from human_feedback_rl.algorithms import PreferenceAlgorithm
from human_feedback_rl.common.loggers import WandbWriter, PrefixedLogger, Logger

from _common import (
    init_wandb_run,
    load_debug_dataset,
    make_run_dir,
    seed_everything,
)


def get_name(cfg) -> tuple[str, str]:
    if cfg.run.baseline:
        group_name = "ppo_baseline"
    else:
        group_name = f"ppo_chri_{cfg.algo.kwargs.labels_type} seg={cfg.algo.kwargs.fragment_length}"
    return group_name, f"{group_name} seed={cfg.run.seed}"


@hydra.main(version_base=None, config_path="../configs", config_name="test_chri_PPO")
def main(cfg: DictConfig) -> None:
    seed = cfg.run.seed
    rng = seed_everything(seed)

    group_name, run_name = get_name(cfg)
    run_dir = make_run_dir(cfg.run.output_dir, run_name)
    init_wandb_run(cfg, group_name, run_name, run_dir)

    print(OmegaConf.to_yaml(cfg))

    print("Creating environment...")
    env = sre.make_vec_env(
        cfg.env.id,
        n_envs=cfg.env.n_envs,
        base_seed=seed,
        **OmegaConf.to_container(cfg.env.kwargs, resolve=True),
    )

    print("Initializing agent...")
    agent = PPO(env=env, seed=seed, **OmegaConf.to_container(cfg.agent.kwargs, resolve=True))

    if cfg.run.baseline:
        # Plain PPO on the true reward, used as the preference-learning baseline.
        agent.set_env(VecMonitor(env))
        agent.set_logger(PrefixedLogger(Logger(folder=None, output_formats=[WandbWriter()]), "agent"))
        agent.learn(
            total_timesteps=cfg.train.kwargs.total_timesteps,
            callback=CustomLoggingCallback(),
        )
        return

    print("Initializing algorithm...")
    algo = PreferenceAlgorithm(
        env=env,
        agent=agent,
        rng=rng,
        debug_dataset=load_debug_dataset(),
        **OmegaConf.to_container(cfg.algo.kwargs, resolve=True),
    )

    print("Starting training...")
    train_kwargs = OmegaConf.to_container(cfg.train.kwargs, resolve=True)
    train_kwargs["checkpoint_dir"] = str(run_dir)
    algo.train(**train_kwargs)


if __name__ == "__main__":
    main()
