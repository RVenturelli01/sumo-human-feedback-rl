import hydra
import sumo_gym_ego as sge
import sumo_rl_ego as sre

from omegaconf import DictConfig, OmegaConf

from sumo_rl_ego.utils import (
    confirm_cfg,
    check_source_cfg,
    load_policy_from_cfg,
    resolve_paths,
)


def print_play_cfg(cfg, title):
    print(f"\n========== PLAYCONFIG ==========\n")
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

    print(f"manual: {cfg.run.manual}")
    print("\n=============================================\n")



@hydra.main(version_base=None, config_path="../configs", config_name="play.yaml")
def main(cfg: DictConfig):
    resolve_paths(cfg)
    check_source_cfg(cfg)

    print_play_cfg(cfg, "PLAY")
    confirm_cfg()

    print("Loading environment...")
    env = sge.make_env(
        cfg.env.id, 
        seed=cfg.run.seed, 
        **cfg.env.kwargs,
        use_gui=True,
    )

    print("Loading policy...")
    policy = load_policy_from_cfg(cfg)

    print("Starting play...")
    sre.play_policy(
        env,
        policy,
        manual=cfg.run.manual,
    )

    env.close()


if __name__ == "__main__":
    main()
