#!/usr/bin/env python3
"""Pre-train an SB3 agent with human_feedback_rl.common.PreTrainAgent.

Example:
    python scripts/pre_train_agent.py --agent PPO --expert-id ppo-fast
    python scripts/pre_train_agent.py --agent SAC --expert-id sac-fast --n-envs 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm


ROOT_DIR = Path(__file__).resolve().parents[1]
for package_dir in (ROOT_DIR / "human-feedback-rl", ROOT_DIR / "sumo-rl-ego"):
    package_path = str(package_dir)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)

import sumo_rl_ego as sre  # noqa: E402
from human_feedback_rl.common.pre_train_agent import PreTrainAgent  # noqa: E402


AGENTS = {
    "PPO": PPO,
    "SAC": SAC,
}


class VectorExpertPolicy:
    """Adapter for expert policies that may only accept one observation."""

    def __init__(self, policy):
        self.policy = policy

    def predict(self, obs: np.ndarray) -> np.ndarray:
        try:
            actions = self.policy.predict(obs)
        except Exception:
            actions = [self.policy.predict(single_obs) for single_obs in obs]

        return np.asarray(actions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-train a PPO/SAC agent using PreTrainAgent."
    )
    parser.add_argument("--agent", choices=AGENTS.keys(), default="PPO")
    parser.add_argument("--expert-id", default="ppo-fast")
    parser.add_argument("--env-id", default="HighwayEgo-v0")
    parser.add_argument("--ego", choices=["continuous", "discrete"], default="continuous")
    parser.add_argument("--reward", choices=["fast", "comfort"], default="fast")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--segment-length", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO rollout steps.")
    parser.add_argument("--n-epochs", type=int, default=5, help="PPO epochs.")
    parser.add_argument(
        "--critic-warmup-rollout-steps",
        type=int,
        default=10_000,
        help="SAC only: steps used to fill the replay buffer after DAgger.",
    )
    parser.add_argument(
        "--critic-warmup-train-steps",
        type=int,
        default=1_000,
        help="SAC only: critic-only gradient steps after rollout collection.",
    )
    return parser.parse_args()


def make_agent(args: argparse.Namespace, env):
    common_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gamma": 0.995,
        "seed": args.seed,
        "device": args.device,
        "verbose": 1,
        "policy_kwargs": {"net_arch": [256, 128]},
    }

    if args.agent == "PPO":
        return PPO(
            **common_kwargs,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            ent_coef=0.01,
        )

    return SAC(
        **common_kwargs,
        buffer_size=100_000,
        learning_starts=100,
        train_freq=1000,
        gradient_steps=1000,
    )


def default_save_path(args: argparse.Namespace) -> Path:
    name = f"{args.agent.lower()}_pretrained_{args.expert_id}"
    return ROOT_DIR / "outputs" / "pretrained" / name


def main() -> None:
    args = parse_args()
    save_path = Path(args.save_path) if args.save_path else default_save_path(args)

    print("Creating vector environment...")
    env = sre.make_vec_env(
        args.env_id,
        n_envs=args.n_envs,
        base_seed=args.seed,
        ego=args.ego,
        reward=args.reward,
    )

    try:
        print(f"Loading expert policy '{args.expert_id}'...")
        expert = VectorExpertPolicy(sre.load_policy(args.expert_id, env=env))

        print(f"Initializing {args.agent} agent...")
        agent = make_agent(args, env)

        agent._last_obs = env.reset()
        if isinstance(agent, OnPolicyAlgorithm):
            agent._last_episode_starts = np.ones((env.num_envs,), dtype=bool)

        pretrainer = PreTrainAgent(
            env=env,
            agent=agent,
            rng=np.random.default_rng(args.seed),
            expert_policy=expert,
            segment_length=args.segment_length,
        )

        print("Starting pre-training...")
        pretrainer.train(
            save_path=str(save_path),
            n_critic_warmup_rollout_steps=args.critic_warmup_rollout_steps,
            n_critic_warmup_train_steps=args.critic_warmup_train_steps,
        )

        print(f"Done. Saved model: {save_path}.zip")

    finally:
        env.close()


if __name__ == "__main__":
    main()
