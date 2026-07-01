import hydra
from omegaconf import DictConfig, OmegaConf

import sumo_gym_ego as sge
import sumo_rl_ego as sre
from human_feedback_rl.algorithms import DaggerAlgorithm
from human_feedback_rl.common import BCPolicy

from _common import (
    init_wandb_run,
    make_run_dir,
    seed_everything,
)


@hydra.main(version_base=None, config_path="../configs", config_name="dagger")
def main(cfg: DictConfig) -> None:
    seed = cfg.run.seed
    rng = seed_everything(seed)

    group_name = "dagger"
    run_name = f"{group_name} expert={cfg.expert.id} seed={seed}"
    run_dir = make_run_dir(cfg.run.output_dir, run_name)
    init_wandb_run(cfg, group_name, run_name, run_dir)

    print(OmegaConf.to_yaml(cfg))

    print("Creating environment...")
    env = sge.make_vec_env(
        cfg.env.id,
        n_envs=cfg.env.n_envs,
        base_seed=seed,
        **OmegaConf.to_container(cfg.env.kwargs, resolve=True),
    )

    print(f"Loading expert ({cfg.expert.id})...")
    expert = sre.load_policy(cfg.expert.id, env=env)

    print("Initializing agent (BCPolicy)...")
    agent = BCPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: cfg.algo.kwargs.bc_lr,
        **OmegaConf.to_container(cfg.agent.kwargs.policy_kwargs, resolve=True),
    )

    print("Initializing DaggerAlgorithm...")
    algo = DaggerAlgorithm(
        env=env,
        agent=agent,
        expert=expert,
        rng=rng,
        log_folder=str(run_dir),
        **OmegaConf.to_container(cfg.algo.kwargs, resolve=True),
    )

    print("Starting training...")
    agent = algo.train(**OmegaConf.to_container(cfg.train.kwargs, resolve=True))

    policy_path = run_dir / "bc_policy.pt"
    agent.save(policy_path)
    print(f"Saved trained policy -> {policy_path}")


if __name__ == "__main__":
    main()
