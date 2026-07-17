import numpy as np
import hydra
import sumo_rl_ego as sre

from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from sumo_gym_ego import EgoStatus

from sumo_rl_ego.utils import (
    init_wandb,
    resolve_paths,
    confirm_cfg,
    check_source_cfg,
    load_policy_from_cfg,
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



def print_eval_cfg(cfg):
    print("\n========== EVAL CONFIG ==========\n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("================== Summary ==================\n")
    print(f"Environment: {cfg.env.id}")
    print(f"Environment arguments: {cfg.env.kwargs}")

    # Print all non-null source fields
    for key, value in cfg.source.items():
        if value is not None:
            if key == "policy_class" and value.get("__target__") is None:
                continue
            print(f"{key}: {value}")

    print(f"Number of episodes: {cfg.run.n_episodes}")
    print("\n=============================================\n")





@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    resolve_paths(cfg)
    check_source_cfg(cfg)

    print_eval_cfg(cfg)
    confirm_cfg()


    run = init_wandb(cfg)
    env = None

    try:
        print("Loading environment...")
        env = sre.make_env(
            cfg.env.id, 
            seed=cfg.run.seed, 
            **cfg.env.kwargs
        )

        print("Loading policy...")
        policy = load_policy_from_cfg(cfg, env=env)

        metrics = EvalMetrics()

        print("Running evaluation...")
        for _ in range(cfg.run.n_episodes):

            info = sre.run_episode(
                env,
                policy,
                seed=cfg.run.seed,
            )
            
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
