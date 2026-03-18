"""
Evaluate a policy trained with scripts/train.py.

Usage:
    python scripts/eval.py \
        run.dir=output/christiano/2026-03-18_10-13-21 \
        agent.model=output/christiano/2026-03-18_10-13-21/models/policy_christiano

    # override number of evaluation episodes
    python scripts/eval.py run.dir=... eval.episodes=100
"""

import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sumo_rl_ego as sre
from stable_baselines3 import A2C as SB3A2C


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_metrics(log: dict):
    """Pretty-print the environment's metric groups."""
    groups: dict = {}
    for key, value in log.items():
        group, name = key.split("/", 1)
        groups.setdefault(group, {})[name] = value

    for group, items in groups.items():
        print(f"[{group}]")
        for name, value in items.items():
            if isinstance(value, float):
                value = round(value, 4)
            print(f"  {name:26s}: {value}")
        print()


def _run_episodes(env, policy, episodes: int):
    """Roll out `policy` for `episodes` episodes; return per-episode rewards and lengths."""
    print(f"\n=== Evaluating — {episodes} episodes ===")

    ep_rewards = []
    ep_lengths = []

    for _ in tqdm(range(episodes)):
        obs, _ = env.reset()
        terminated = truncated = False
        ep_reward = 0.0
        ep_len = 0

        while not (terminated or truncated):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_len += 1

        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_len)

    return ep_rewards, ep_lengths


def _save_plots(ep_rewards: list, ep_lengths: list, metrics_log: dict, plot_dir: Path) -> None:
    """Generate and save evaluation plots to plot_dir."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    episodes = list(range(1, len(ep_rewards) + 1))
    window = max(1, len(ep_rewards) // 10)

    # ── Reward lineplot with moving average ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, ep_rewards, alpha=0.4, color="steelblue", label="episode reward")
    if len(ep_rewards) >= window:
        ma = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            list(range(window, len(ep_rewards) + 1)),
            ma,
            color="steelblue",
            lw=2,
            label=f"moving avg (w={window})",
        )
    ax.axhline(np.mean(ep_rewards), color="red", ls="--", lw=1.0, label=f"mean={np.mean(ep_rewards):.2f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Evaluation — reward per episode")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(plot_dir / "reward_per_episode.png"), dpi=120)
    plt.close(fig)

    # ── Episode length lineplot ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, ep_lengths, alpha=0.4, color="darkorange", label="episode length")
    if len(ep_lengths) >= window:
        ma_len = np.convolve(ep_lengths, np.ones(window) / window, mode="valid")
        ax.plot(
            list(range(window, len(ep_lengths) + 1)),
            ma_len,
            color="darkorange",
            lw=2,
            label=f"moving avg (w={window})",
        )
    ax.axhline(np.mean(ep_lengths), color="red", ls="--", lw=1.0, label=f"mean={np.mean(ep_lengths):.1f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Evaluation — episode length")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(plot_dir / "episode_length.png"), dpi=120)
    plt.close(fig)

    # ── Aggregate metrics barplot ─────────────────────────────────────────────
    if metrics_log:
        flat = {}
        for key, value in metrics_log.items():
            if isinstance(value, (int, float)):
                flat[key.replace("/", "\n")] = float(value)

        if flat:
            fig, ax = plt.subplots(figsize=(max(6, len(flat) * 0.8), 5))
            keys = list(flat.keys())
            vals = [flat[k] for k in keys]
            bars = ax.bar(keys, vals, color="teal", alpha=0.75)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
            ax.set_title("Evaluation — aggregate metrics")
            ax.set_ylabel("Value")
            ax.tick_params(axis="x", labelsize=8)
            fig.tight_layout()
            fig.savefig(str(plot_dir / "aggregate_metrics.png"), dpi=120)
            plt.close(fig)

    print(f"\n[eval] Plots saved to {plot_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):

    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # PROJECT_ROOT = sumo-human-feedback-rl/ (root of the repo)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # ── Load training config to reproduce the exact environment ──────────────
    run_dir = PROJECT_ROOT / cfg.run.dir
    train_cfg = OmegaConf.load(run_dir / "config.yaml")

    expert_cfg = OmegaConf.load(
        PROJECT_ROOT / train_cfg.env.expert_model / ".hydra" / "config.yaml"
    )

    print(f"[eval] Training run : {run_dir}")
    print(f"[eval] Scenario     : {expert_cfg.env}")

    # ── Build single (non-vectorized) environment ─────────────────────────────
    env = sre.make_env(expert_cfg.env, seed=train_cfg.seed)

    # ── Load policy (SB3 A2C .zip format) ────────────────────────────────────
    agent_path = PROJECT_ROOT / cfg.agent.model
    print(f"[eval] Policy path  : {agent_path}")

    policy = SB3A2C.load(str(agent_path), device="cpu")

    # ── Run evaluation ────────────────────────────────────────────────────────
    ep_rewards, ep_lengths = _run_episodes(env, policy, cfg.eval.episodes)

    # ── Print environment metrics ─────────────────────────────────────────────
    log = env.metrics_tracker.get_log_metrics()
    _print_metrics(log)

    # ── Save plots ────────────────────────────────────────────────────────────
    plot_dir = run_dir / "eval_plots"
    _save_plots(ep_rewards, ep_lengths, log, plot_dir)

    print(f"\n[eval] Mean reward : {np.mean(ep_rewards):.3f} ± {np.std(ep_rewards):.3f}")
    print(f"[eval] Mean length : {np.mean(ep_lengths):.1f}")

    env.close()


if __name__ == "__main__":
    main()
