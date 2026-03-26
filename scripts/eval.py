"""
Evaluate a policy trained with scripts/train.py.

Usage:
    python scripts/eval.py \
        run.dir=output/christiano/2026-03-18_10-13-21 \
        source.model_path=output/christiano/2026-03-18_10-13-21/models/policy_christiano

    # override number of episodes or seed
    python scripts/eval.py run.dir=... source.model_path=... run.n_episodes=200 run.seed=42
"""

import numpy as np
import hydra
import wandb
import sumo_rl_ego as sre

from collections import defaultdict
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import A2C as SB3A2C
from sumo_gym_ego import EgoStatus

from sumo_rl_ego.utils import (
    init_wandb,
    resolve_paths,
    confirm_cfg,
    check_source_cfg,
)

from human_feedback_rl.common.utils.env_setup import _load_policy_cfg, _env_kwargs


# ─────────────────────────────────────────────────────────────────────────────
# Policy wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _DeterministicPolicy:
    """
    Wraps an SB3 model to match the sre.run_episode policy interface:
        policy.reset()
        policy.predict(obs) -> action
    Uses deterministic=True so evaluation is reproducible.
    """
    def __init__(self, model):
        self._model = model

    def reset(self):
        pass

    def predict(self, obs):
        action, _ = self._model.predict(obs, deterministic=True)
        return action


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

class EvalMetrics:

    def __init__(self):
        self.data = defaultdict(list)

    def add_episode(self, info):
        ep = info.get("metrics", {}).get("episode", {})

        for key, value in ep.items():
            self.data[key].append(value)

        ep_length   = info.get("step", 0)
        ep_duration = info.get("sim_time", 0.0)
        self.data["performance/ep_length"].append(float(ep_length))
        self.data["performance/ep_duration"].append(float(ep_duration))

        ego_status = info.get("ego_status", EgoStatus.RUNNING)
        self.data["event_rate/collisions"].append(int(ego_status == EgoStatus.COLLIDED.value))
        self.data["event_rate/off_road"].append(int(ego_status == EgoStatus.OFF_ROAD.value))
        self.data["event_rate/timeouts"].append(int(ego_status == EgoStatus.TIMEOUT.value))
        self.data["event_rate/successes"].append(int(ego_status == EgoStatus.ARRIVED.value))

    def print_metrics(self):
        current = ""
        for key in sorted(self.data.keys()):
            values = self.data[key]
            if not values:
                continue
            mean = np.mean(values)
            sec, name = key.split("/", 1)
            if sec != current:
                current = sec
                print(f"\n=== {sec} ===")
            print(f"  {name}: {mean:.3f}")

    def log_metrics(self):
        sre.utils.log_histogram(
            data=self.data["performance/ep_duration"],
            value="duration",
            title="Duration over episodes")

        sre.utils.log_histogram(
            data=self.data["performance/ep_avg_speed"],
            value="avg_speed",
            title="Average Speed Distribution")

        sre.utils.log_histogram(
            data=self.data["rewards/ep_fast_return"],
            value="fast_return",
            title="Fast Return Distribution")

        sre.utils.log_histogram(
            data=self.data["rewards/ep_comfort_return"],
            value="comfort_return",
            title="Comfort Return Distribution")

        sre.utils.log_bar_plot(
            data=[
                ["collisions", np.mean(self.data["event_rate/collisions"])],
                ["off_road",   np.mean(self.data["event_rate/off_road"])],
                ["timeouts",   np.mean(self.data["event_rate/timeouts"])],
                ["successes",  np.mean(self.data["event_rate/successes"])],
            ],
            value="rate",
            title="Event Rates",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_eval_cfg(cfg, env_id, env_kw):
    print(f"\n========== EVAL CONFIG ==========\n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("================== Summary ==================\n")
    print(f"Environment  : {env_id}")
    print(f"Env kwargs   : {env_kw}")
    print(f"Policy path  : {cfg.source.model_path}")
    print(f"Episodes     : {cfg.run.n_episodes}")
    print(f"Seed         : {cfg.run.seed}")
    print("\n=============================================\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    resolve_paths(cfg)
    check_source_cfg(cfg)

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Load training config to reproduce the exact environment used during training.
    run_dir    = PROJECT_ROOT / cfg.run.dir
    train_cfg  = OmegaConf.load(run_dir / "config.yaml")
    expert_cfg = _load_policy_cfg(train_cfg.env.expert_model)
    env_id     = expert_cfg.env.id
    env_kw     = _env_kwargs(expert_cfg)

    _print_eval_cfg(cfg, env_id, env_kw)
    confirm_cfg()

    run = init_wandb(cfg)
    env = None

    try:
        env = sre.make_env(env_id, seed=cfg.run.seed, **env_kw)

        model  = SB3A2C.load(str(PROJECT_ROOT / cfg.source.model_path), device="cpu")
        policy = _DeterministicPolicy(model)

        metrics = EvalMetrics()

        print(f"Running evaluation — {cfg.run.n_episodes} episodes…")
        for _ in range(cfg.run.n_episodes):
            info = sre.run_episode(env, policy, seed=cfg.run.seed)
            metrics.add_episode(info)

        if run is not None:
            metrics.log_metrics()

        metrics.print_metrics()

    finally:
        if env is not None:
            env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
