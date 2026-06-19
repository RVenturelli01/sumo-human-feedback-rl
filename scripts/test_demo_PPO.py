import hydra
from omegaconf import DictConfig, OmegaConf

import sumo_rl_ego as sre
from stable_baselines3 import PPO
from human_feedback_rl.algorithms import DemoAlgorithm

from _common import (
    init_wandb_run,
    load_debug_dataset,
    load_expert_trajectories,
    make_run_dir,
    seed_everything,
)


@hydra.main(version_base=None, config_path="../configs", config_name="test_demo_PPO")
def main(cfg: DictConfig) -> None:
    seed = cfg.run.seed
    rng = seed_everything(seed)

    group_name = "ppo_demo_irl"
    run_name = f"{group_name} seed={seed}"
    run_dir = make_run_dir(cfg.run.output_dir, run_name)
    init_wandb_run(cfg, group_name, run_name, run_dir)

    print(OmegaConf.to_yaml(cfg))

    print("Loading expert trajectories...")
    expert_trajectories = load_expert_trajectories()
    print(f"Loaded {len(expert_trajectories)} expert trajectories")

    print("Creating environment...")
    env = sre.make_vec_env(
        cfg.env.id,
        n_envs=cfg.env.n_envs,
        base_seed=seed,
        **OmegaConf.to_container(cfg.env.kwargs, resolve=True),
    )

    print("Initializing agent...")
    agent = PPO(env=env, seed=seed, **OmegaConf.to_container(cfg.agent.kwargs, resolve=True))

    print("Initializing DemoAlgorithm...")
    algo = DemoAlgorithm(
        env=env,
        agent=agent,
        expert_trajectories=expert_trajectories,
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
