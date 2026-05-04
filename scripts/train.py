from collections import defaultdict
import pickle
from pathlib import Path

import hydra
import numpy as np
import wandb

import sumo_rl_ego as sre

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from human_feedback_rl.algorithms import ChristianoAlgorithm, DaggerAlgorithm
from human_feedback_rl.algorithms.christiano.christiano_algorithm_demo import ChristianoAlgorithmDemo
from human_feedback_rl.common import BCPolicy

from sumo_gym_ego import EgoStatus

from sumo_rl_ego.utils import (
    init_wandb, 
    confirm_cfg,
    save_outputs,
    CustomLoggingCallback,
)

class _BCPolicyAdapter:
    """Wraps BCPolicy to match the sre.Policy interface expected by run_episode."""

    def __init__(self, bc_policy: BCPolicy):
        self._policy = bc_policy

    def reset(self):
        pass

    def predict(self, obs):
        action, _ = self._policy.predict(obs, deterministic=True)
        return action


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
    # test
    _ = HydraConfig.get().runtime.output_dir

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
            **cfg.env.kwargs
        )

        if cfg.algo.name == "christiano":
            print("Initializing agent...")
            algo_cls = ALGO_REGISTRY[cfg.algo.agent.algo]
            agent = algo_cls(
                env=env,
                **cfg.algo.agent.kwargs
            )

            print("Initializing algorithm...")
            algo = ChristianoAlgorithm(
                env=env,
                agent=agent,
                **cfg.algo.kwargs,
            )
            with open_dict(cfg):
                cfg.model = {"algo": cfg.algo.agent.algo}

        if cfg.algo.name == "christiano_demo":
            print("Initializing agent...")
            algo_cls = ALGO_REGISTRY[cfg.algo.agent.algo]
            agent = algo_cls(
                env=env,
                **cfg.algo.agent.kwargs
            )

            print(f"Loading expert ({cfg.algo.expert_id})...")
            expert = sre.load_policy(cfg.algo.expert_id)

            print("Initializing algorithm...")
            algo = ChristianoAlgorithmDemo(
                env=env,
                agent=agent,
                expert_policy=expert,
                **cfg.algo.kwargs,
            )
        elif cfg.algo.name == "dagger":
            print("Initializing agent (BCPolicy)...")
            agent = BCPolicy(
                observation_space=env.observation_space,
                action_space=env.action_space,
                lr_schedule=lambda _: cfg.algo.kwargs.bc_lr,
                **OmegaConf.to_container(cfg.algo.policy_kwargs, resolve=True),
            )

            print(f"Loading expert ({cfg.algo.expert_id})...")
            expert = sre.load_policy(cfg.algo.expert_id)

            print("Initializing algorithm...")
            algo = DaggerAlgorithm(
                env=env,
                agent=agent,
                expert=expert,
                **cfg.algo.kwargs,
            )

        print("Starting training...")
        agent = algo.train(**cfg.algo.train.kwargs)

        
        print("\nTraining finished.")
        save_outputs(cfg, agent)

        if cfg.algo.name == "christiano" or  cfg.algo.name == "christiano_demo":
            _output_dir = Path(HydraConfig.get().runtime.output_dir)

            rm_path = _output_dir / "reward_model.pt"
            algo.save_reward_model(rm_path)
            if cfg.wandb.enabled and wandb.run is not None:
                wandb.save(str(rm_path))

            print("Collecting debug trajectories...")
            debug_data = algo.collect_debug_data(n_steps=cfg.run.get("n_steps_debug", 2000))
            debug_path = _output_dir / "debug_data.pkl"
            with open(debug_path, "wb") as _f:
                pickle.dump(debug_data, _f)
            print(f"Saved debug data → {debug_path}  ({len(debug_data['trajectories'])} episodes)")

        print("Run completed successfully.\n")

        # -------------- eval -----------------------
        print("Loading environment for evaluation...")
        eval_env = sre.make_env(
            cfg.env.id,
            seed=cfg.run.seed,
            **cfg.env.kwargs
        )

        try:
            if cfg.algo.name == "dagger":
                policy = _BCPolicyAdapter(agent)
            else:
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
