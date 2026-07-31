"""Train HybridAlgorithm with a SAC agent (the single training entry point).

Configured by configs/train_hybrid_sac.yaml plus Hydra overrides. Covers every
experiment arm: hybrid (both sources), pref-only (``demo_weight=0``) and
demo-only (``total_queries=0``). Writes per-iteration ``metrics.jsonl``, a
final held-out ``final_eval.json`` and ``agent_final.zip`` in the run dir,
and mirrors the final evaluation as ``sweep/*`` metrics on W&B.
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

from _common import (
    evaluate,
    init_wandb_run,
    load_debug_dataset,
    load_expert_trajectories,
    log_sweep_summary,
    make_run_dir,
    seed_everything,
)


@hydra.main(version_base=None, config_path="../configs", config_name="train_hybrid_sac")
def main(cfg: DictConfig) -> None:
    seed = cfg.run.seed
    rng = seed_everything(seed)

    loss_type = cfg.algo.kwargs.loss_type
    demo_weight = cfg.algo.kwargs.demo_weight
    group_name = cfg.run.get("group", None) or f"sac_hybrid_{loss_type}"
    run_name = (
        cfg.run.get("name", None)
        or f"{group_name} demo_weight={demo_weight} seed={seed}"
    )
    run_dir = make_run_dir(cfg.run.output_dir, run_name)
    init_wandb_run(cfg, group_name, run_name, run_dir)

    print(OmegaConf.to_yaml(cfg))

    print("Loading expert trajectories...")
    # The subsample seed is NOT the run seed: every arm at a given budget must
    # read the same demonstrations, so a budget curve compares algorithms and
    # not datasets. null falls back to the shared DEMO_SUBSAMPLE_SEED.
    expert_trajectories, demo_manifest = load_expert_trajectories(
        n_trajectories=cfg.run.get("n_expert_trajectories", None),
        seed=cfg.run.get("demo_subsample_seed", None),
        n_transitions=cfg.run.get("n_expert_transitions", None),
        return_manifest=True,
    )
    n_expert_transitions = demo_manifest["n_transitions_selected"]
    print(
        f"Loaded {len(expert_trajectories)} expert trajectories "
        f"({n_expert_transitions} transitions), "
        f"subsample seed {demo_manifest['subsample_seed']}, "
        f"fingerprint {demo_manifest['fingerprint'][:12]}"
    )
    if demo_manifest["subsample_seed"] != DEMO_SUBSAMPLE_SEED:
        # Legitimate for an ablation that varies the demonstrations on purpose,
        # but it means this run is NOT comparable to arms on the shared seed.
        print(
            f"WARNING: demo subsample seed {demo_manifest['subsample_seed']} is not "
            f"the shared {DEMO_SUBSAMPLE_SEED}; this run sees different "
            f"demonstrations from the other arms at the same budget."
        )
    # Persist the full selection next to the run's other artifacts, and put the
    # fingerprint on W&B so two runs can be checked against each other without
    # access to the machine that produced them.
    with open(run_dir / "demo_subsample.json", "w") as f:
        json.dump(demo_manifest, f, indent=2)
    wandb.config.update(
        {
            "expert_n_trajectories": len(expert_trajectories),
            "expert_n_transitions": n_expert_transitions,
            "demo_subsample_fingerprint": demo_manifest["fingerprint"],
            "demo_subsample_seed_used": demo_manifest["subsample_seed"],
            "demo_dataset_fingerprint": demo_manifest["dataset_fingerprint"],
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
