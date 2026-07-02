import hydra
from omegaconf import DictConfig, OmegaConf

import sumo_gym_ego as sge
import sumo_rl_ego as sre
from human_feedback_rl.algorithms import DemoAlgorithm2

from _common import (
    init_wandb_run,
    load_debug_dataset,
    load_expert_trajectories,
    make_run_dir,
    seed_everything,
)


@hydra.main(version_base=None, config_path="../configs", config_name="test_demo2")
def main(cfg: DictConfig) -> None:
    seed = cfg.run.seed
    rng = seed_everything(seed)

    group_name = "demo2_weighted_bc"
    loss_type = cfg.algo.kwargs.loss_type
    weight_temperature = cfg.algo.kwargs.weight_temperature
    run_name = f"{group_name} loss={loss_type} eta={weight_temperature} seed={seed}"
    run_dir = make_run_dir(cfg.run.output_dir, run_name)
    init_wandb_run(cfg, group_name, run_name, run_dir)

    print(OmegaConf.to_yaml(cfg))

    print("Loading expert trajectories...")
    expert_trajectories = load_expert_trajectories()
    print(f"Loaded {len(expert_trajectories)} expert trajectories")

    print("Creating environment...")
    env = sge.make_vec_env(
        cfg.env.id,
        n_envs=cfg.env.n_envs,
        base_seed=seed,
        **OmegaConf.to_container(cfg.env.kwargs, resolve=True),
    )
    rollout_env = sge.make_vec_env(
        cfg.env.id,
        n_envs=cfg.env.n_envs,
        base_seed=seed + 10_000,
        **OmegaConf.to_container(cfg.env.kwargs, resolve=True),
    )

    # No SAC/PPO: DemoAlgorithm2 owns a standalone SquashedGaussianPolicy and
    # trains it purely by weighted behavior cloning (section 13).
    print("Initializing DemoAlgorithm2...")
    algo = DemoAlgorithm2(
        env=env,
        expert_trajectories=expert_trajectories,
        rng=rng,
        debug_dataset=load_debug_dataset(),
        rollout_env=rollout_env,
        **OmegaConf.to_container(cfg.algo.kwargs, resolve=True),
    )

    print("Starting training...")
    train_kwargs = OmegaConf.to_container(cfg.train.kwargs, resolve=True)
    train_kwargs["checkpoint_dir"] = str(run_dir)
    algo.train(**train_kwargs)


if __name__ == "__main__":
    main()
