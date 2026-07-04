"""Entry point for hybrid (demonstrations + preferences) reward learning.

Runs :class:`HybridAlgorithm`, which trains a single shared reward model from a
Guided-Cost-Learning demonstration loss and a Bradley-Terry preference loss. The
experimental mode (pref_only / demo_only / hybrid) is chosen entirely by the
``algo.kwargs.lambda_demo`` and ``algo.kwargs.lambda_pref`` config values.
"""

import hydra
from omegaconf import DictConfig, OmegaConf

import sumo_gym_ego as sge
import sumo_rl_ego as sre
from stable_baselines3 import SAC
from human_feedback_rl.algorithms import HybridAlgorithm
from human_feedback_rl.common.replay_buffers import (
    RewardDiagnosticsReplayBuffer,
    RewardRelabelReplayBuffer,
)

from _common import (
    init_wandb_run,
    load_debug_dataset,
    load_expert_trajectories,
    make_run_dir,
    seed_everything,
)


@hydra.main(version_base=None, config_path="../configs", config_name="test_hybrid_SAC")
def main(cfg: DictConfig) -> None:
    seed = cfg.run.seed
    rng = seed_everything(seed)

    lambda_demo = cfg.algo.kwargs.lambda_demo
    lambda_pref = cfg.algo.kwargs.lambda_pref
    mode = HybridAlgorithm._resolve_mode(float(lambda_demo), float(lambda_pref))

    group_name = f"sac_hybrid_{mode}"
    run_name = (
        f"{group_name} ldemo={lambda_demo} lpref={lambda_pref} "
        f"loss={cfg.algo.kwargs.loss_type} seed={seed}"
    )
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

    relabel_rewards = cfg.algo.kwargs.relabel_rewards

    print("Initializing agent...")
    agent = SAC(
        env=env,
        seed=seed,
        replay_buffer_class=(
            RewardRelabelReplayBuffer if relabel_rewards else RewardDiagnosticsReplayBuffer
        ),
        **OmegaConf.to_container(cfg.agent.kwargs, resolve=True),
    )

    print("Initializing HybridAlgorithm...")
    algo = HybridAlgorithm(
        env=env,
        agent=agent,
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

    print("Running final evaluation...")
    algo.evaluate(n_episodes=20, log_prefix="eval_final")


if __name__ == "__main__":
    main()
