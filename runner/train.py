"""Train one arm. The single entry point.

    python runner/train.py arm=hybrid_soft budget=1000 run.seed=3

An arm is a file under configs/arm/, the shared settings live in
configs/protocol/, and Hydra composes them over configs/train.yaml. All seven
arms are the same algorithm with one or both feedback channels on: the
preference-only ones set demo_weight=0, the demonstration-only one
total_queries=0.

Writes metrics.jsonl, final_eval.json and agent_final.zip in the run directory.
For a grid, use Hydra's --multirun; there is no scheduler here.
"""

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

from human_feedback_rl.common.demo_subsampling import DEMO_SUBSAMPLE_SEED

from utils.budget import register_resolvers
from utils.common import (
    evaluate,
    init_wandb_run,
    load_debug_dataset,
    load_expert_trajectories,
    log_sweep_summary,
    make_run_dir,
    seed_everything,
)


register_resolvers()


def start_run(cfg: DictConfig, seed: int):
    """Name the run, make its directory, and open it on W&B."""
    loss_type = cfg.algo.kwargs.loss_type
    demo_weight = cfg.algo.kwargs.demo_weight
    group_name = cfg.run.get("group", None) or f"sac_hybrid_{loss_type}"
    run_name = (
        cfg.run.get("name", None)
        or f"{group_name} demo_weight={demo_weight} seed={seed}"
    )
    run_dir = make_run_dir(cfg.run.output_dir, run_name)
    init_wandb_run(cfg, group_name, run_name, run_dir)
    return run_dir


def load_demonstrations(cfg: DictConfig, run_dir):
    """Read the demonstrations for this budget, and record which ones they were.

    The subsample seed is NOT the run seed: every arm at a given budget must read
    the same demonstrations, so a budget curve compares algorithms and not
    datasets. null falls back to the shared DEMO_SUBSAMPLE_SEED.
    """
    expert_trajectories, manifest = load_expert_trajectories(
        n_trajectories=cfg.run.get("n_expert_trajectories", None),
        seed=cfg.run.get("demo_subsample_seed", None),
        n_transitions=cfg.run.get("n_expert_transitions", None),
        return_manifest=True,
    )
    n_transitions = manifest["n_transitions_selected"]
    print(
        f"Loaded {len(expert_trajectories)} expert trajectories "
        f"({n_transitions} transitions), "
        f"subsample seed {manifest['subsample_seed']}, "
        f"fingerprint {manifest['fingerprint'][:12]}"
    )
    if manifest["subsample_seed"] != DEMO_SUBSAMPLE_SEED:
        # Legitimate for an ablation that varies the demonstrations on purpose,
        # but it means this run is NOT comparable to arms on the shared seed.
        print(
            f"WARNING: demo subsample seed {manifest['subsample_seed']} is not "
            f"the shared {DEMO_SUBSAMPLE_SEED}; this run sees different "
            f"demonstrations from the other arms at the same budget."
        )
    # Keep the full selection beside the run's other artifacts, and put the
    # fingerprint on W&B, so two runs can be checked against each other without
    # access to the machine that produced them.
    with open(run_dir / "demo_subsample.json", "w") as f:
        json.dump(manifest, f, indent=2)
    wandb.config.update(
        {
            "expert_n_trajectories": len(expert_trajectories),
            "expert_n_transitions": n_transitions,
            "demo_subsample_fingerprint": manifest["fingerprint"],
            "demo_subsample_seed_used": manifest["subsample_seed"],
            "demo_dataset_fingerprint": manifest["dataset_fingerprint"],
        },
        allow_val_change=True,
    )
    return expert_trajectories


def build_envs(cfg: DictConfig, seed: int):
    """The training environment, and the rollout one when it is not shared.

    A rollout_env of None puts HybridAlgorithm on the shared-environment branch:
    the buffering wrapper sits inside the reward wrapper on the agent's own
    environment, so every step of learn() is recorded with the true reward and
    sample() reuses those instead of collecting fresh trajectories.
    """
    env_kwargs = OmegaConf.to_container(cfg.env.kwargs, resolve=True)
    env = sge.make_vec_env(
        cfg.env.id, n_envs=cfg.env.n_envs, base_seed=seed, **env_kwargs
    )
    shared = bool(OmegaConf.select(cfg, "env.shared_rollout_env") or False)
    rollout_env = None if shared else sge.make_vec_env(
        cfg.env.id, n_envs=cfg.env.n_envs, base_seed=seed + 10_000, **env_kwargs
    )
    print(f"Rollout env: {'shared with training' if shared else 'dedicated'}")
    return env, rollout_env


def build_agent(cfg: DictConfig, env, seed: int) -> SAC:
    """The SAC agent, with the replay buffer the relabelling setting asks for."""
    relabel = cfg.algo.kwargs.relabel_rewards
    return SAC(
        env=env,
        seed=seed,
        replay_buffer_class=(
            RewardRelabelReplayBuffer if relabel else RewardDiagnosticsReplayBuffer
        ),
        **OmegaConf.to_container(cfg.agent.kwargs, resolve=True),
    )


def build_algorithm(cfg: DictConfig, env, agent, expert_trajectories, rng,
                    rollout_env, run_dir) -> HybridAlgorithm:
    """The algorithm, wired to its three logging sinks."""
    return HybridAlgorithm(
        env=env,
        agent=agent,
        expert_trajectories=expert_trajectories,
        rng=rng,
        debug_dataset=load_debug_dataset(),
        rollout_env=rollout_env,
        # The JSONL sink lets an external monitor follow per-iteration progress
        # without going through W&B.
        output_formats=[
            make_human_output_format(),
            WandbWriter(),
            JsonlWriter(run_dir / "metrics.jsonl"),
        ],
        **OmegaConf.to_container(cfg.algo.kwargs, resolve=True),
    )


def final_evaluation(cfg: DictConfig, trained_agent, run_dir) -> None:
    """Evaluate the trained policy on held-out episodes, and save both."""
    env_kwargs = OmegaConf.to_container(cfg.env.kwargs, resolve=True)
    metrics = evaluate(
        trained_agent, cfg.env.id, env_kwargs, cfg.eval.n_episodes, cfg.eval.seed
    )
    print(f"Final evaluation: {metrics}")
    log_sweep_summary(metrics)
    with open(run_dir / "final_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)
    trained_agent.save(str(run_dir / "agent_final"))


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    seed = cfg.run.seed
    rng = seed_everything(seed)
    run_dir = start_run(cfg, seed)

    print(OmegaConf.to_yaml(cfg))

    print("Loading expert trajectories...")
    expert_trajectories = load_demonstrations(cfg, run_dir)

    print("Creating environment...")
    env, rollout_env = build_envs(cfg, seed)

    print("Initializing agent...")
    agent = build_agent(cfg, env, seed)

    print("Initializing HybridAlgorithm...")
    algo = build_algorithm(cfg, env, agent, expert_trajectories, rng,
                           rollout_env, run_dir)

    print("Starting training...")
    train_kwargs = OmegaConf.to_container(cfg.train.kwargs, resolve=True)
    train_kwargs["checkpoint_dir"] = str(run_dir)
    trained_agent = algo.train(**train_kwargs)

    # libsumo allows one simulation per process: release both training envs
    # before evaluate() opens its own.
    env.close()
    if rollout_env is not None:
        rollout_env.close()

    print("Running final held-out evaluation...")
    final_evaluation(cfg, trained_agent, run_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
