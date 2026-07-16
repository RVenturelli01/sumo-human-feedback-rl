import json

import hydra
from omegaconf import DictConfig, OmegaConf

import sumo_gym_ego as sge
from stable_baselines3 import SAC
import wandb

from human_feedback_rl.algorithms import HybridAlgorithm
from human_feedback_rl.common.loggers import (
    JsonlWriter,
    WandbWriter,
    make_human_output_format,
)
from human_feedback_rl.common.replay_buffers import (
    RewardDiagnosticsReplayBuffer,
    RewardRelabelReplayBuffer,
)

from _common import (
    evaluate,
    init_wandb_run,
    load_debug_dataset,
    load_expert_trajectories,
    log_sweep_summary,
    make_run_dir,
    seed_everything,
)


@hydra.main(version_base=None, config_path="../configs", config_name="test_hybrid_SAC")
def main(cfg: DictConfig) -> None:
    seed = cfg.run.seed
    rng = seed_everything(seed)

    loss_type = cfg.algo.kwargs.loss_type
    demo_weight = cfg.algo.kwargs.demo_weight
    group_name = f"sac_hybrid_{loss_type}"
    run_name = f"{group_name} demo_weight={demo_weight} seed={seed}"
    run_dir = make_run_dir(cfg.run.output_dir, run_name)
    init_wandb_run(cfg, group_name, run_name, run_dir)

    print(OmegaConf.to_yaml(cfg))

    print("Loading expert trajectories...")
    demo_subsample_seed = cfg.run.get("demo_subsample_seed", None)
    expert_trajectories = load_expert_trajectories(
        n_trajectories=cfg.run.get("n_expert_trajectories", None),
        seed=seed if demo_subsample_seed is None else demo_subsample_seed,
    )
    n_expert_transitions = sum(len(traj) for traj in expert_trajectories)
    print(
        f"Loaded {len(expert_trajectories)} expert trajectories "
        f"({n_expert_transitions} transitions)"
    )
    wandb.config.update(
        {
            "expert_n_trajectories": len(expert_trajectories),
            "expert_n_transitions": n_expert_transitions,
        },
        allow_val_change=True,
    )

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
            RewardRelabelReplayBuffer
            if relabel_rewards
            else RewardDiagnosticsReplayBuffer
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
        # The JSONL sink lets an external monitor (e.g. an Optuna worker) follow
        # per-iteration progress without going through W&B.
        output_formats=[
            make_human_output_format(),
            WandbWriter(),
            JsonlWriter(run_dir / "metrics.jsonl"),
        ],
        **OmegaConf.to_container(cfg.algo.kwargs, resolve=True),
    )

    print("Starting training...")
    train_kwargs = OmegaConf.to_container(cfg.train.kwargs, resolve=True)
    train_kwargs["checkpoint_dir"] = str(run_dir)
    trained_agent = algo.train(**train_kwargs)

    # libsumo allows one simulation per process: release both training envs
    # before evaluate() opens its own.
    env.close()
    rollout_env.close()

    print("Running final held-out evaluation...")
    env_kwargs = OmegaConf.to_container(cfg.env.kwargs, resolve=True)
    metrics = evaluate(
        trained_agent, cfg.env.id, env_kwargs, cfg.eval.n_episodes, cfg.eval.seed
    )
    print(f"Final evaluation: {metrics}")
    log_sweep_summary(metrics)
    with open(run_dir / "final_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)
    trained_agent.save(str(run_dir / "agent_final"))

    wandb.finish()


if __name__ == "__main__":
    main()
