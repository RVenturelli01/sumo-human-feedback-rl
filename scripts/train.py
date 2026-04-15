from collections import defaultdict

import random

import hydra
import numpy as np
import torch
import sumo_rl_ego as sre

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3

from sumo_gym_ego import EgoStatus
from human_feedback_rl.algorithms import ChristianoAlgorithm

from sumo_rl_ego.utils import (
    init_wandb,
    confirm_cfg,
    save_outputs,
)


class EvalMetrics:

    def __init__(self):
        self.data = defaultdict(list)

    def add_episode(self, info):
        ep = info.get("metrics", {}).get("episode", {})

        for key, value in ep.items():
            self.data[key].append(value)

        # --- external metrics ---
        ep_length = info.get("step", 0)
        ep_duration = info.get("sim_time", 0.0)
        self.data["performance/ep_length"].append(float(ep_length))
        self.data["performance/ep_duration"].append(float(ep_duration))

        # --- events ---
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

            print(f"{name}: {mean:.3f}")

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
                ["off_road", np.mean(self.data["event_rate/off_road"])],
                ["timeouts", np.mean(self.data["event_rate/timeouts"])],
                ["successes", np.mean(self.data["event_rate/successes"])],
            ],
            value="rate",
            title="Event Rates",
        )


ALGO_REGISTRY = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}


def set_global_seeds(seed: int) -> None:
    """Set all global RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_train_cfg(cfg):
    print(f"\n========== TRAIN CONFIG ==========\n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("================== Summary ==================\n")
    print(f"Environment: {cfg.env.id} (x{cfg.env.n_envs} envs)")
    print(f"Environment arguments: {cfg.env.kwargs}")
    print(f"Algorithm: {cfg.algo.name}")
    print("\n=============================================\n")


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    _ = HydraConfig.get().runtime.output_dir

    set_global_seeds(cfg.run.seed)
    print_train_cfg(cfg)
    confirm_cfg()

    run = init_wandb(cfg)
    env = None

    try:
        print("Creating environment...")
        env = sre.make_vec_env(
            cfg.env.id,
            n_envs=cfg.env.n_envs,
            base_seed=cfg.run.seed,
            **cfg.env.kwargs,
        )

        if cfg.algo.name == "christiano":
            print("Initializing agent...")
            algo_cls = ALGO_REGISTRY[cfg.algo.agent.algo]
            agent = algo_cls(
                env=env,
                seed=cfg.run.seed,
                **OmegaConf.to_container(cfg.algo.agent.kwargs, resolve=True),
            )

            print("Initializing algorithm...")
            algo = ChristianoAlgorithm(
                env=env,
                agent=agent,
                rng=np.random.default_rng(cfg.run.seed),
                **OmegaConf.to_container(cfg.algo.kwargs, resolve=True),
            )
            with open_dict(cfg):
                cfg.model = {"algo": cfg.algo.agent.algo}
        else:
            raise ValueError(f"Unknown algorithm: {cfg.algo.name!r}")

        print("Starting training...")
        algo.train(**OmegaConf.to_container(cfg.algo.train.kwargs, resolve=True))

        print("\nTraining finished.")
        save_outputs(cfg, agent)
        print("Run completed successfully.\n")

        # -------------- eval -----------------------
        print("Loading environment for evaluation...")
        eval_env = sre.make_env(
            cfg.env.id,
            seed=cfg.run.seed,
            **cfg.env.kwargs
        )

        try:
            policy = sre.ModelPolicy(agent)

            metrics = EvalMetrics()

            print("Running evaluation...")
            for _ in range(cfg.run.n_episodes_eval):
                info = sre.run_episode(
                    eval_env,
                    policy,
                    seed=cfg.run.seed,
                )

                metrics.add_episode(info)

            if run is not None:
                metrics.log_metrics()

            metrics.print_metrics()
        finally:
            eval_env.close()

    finally:
        if env is not None:
            env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
