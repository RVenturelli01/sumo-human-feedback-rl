"""
Visualise a policy trained with scripts/train.py in the SUMO GUI.

Usage:
    python scripts/play.py \
        agent.model=output/christiano/2026-03-18_10-13-21/models/policy_christiano \
        run.dir=output/christiano/2026-03-18_10-13-21

Optional overrides:
    eval.episodes=5          number of episodes to play (default: infinite loop)
    play.step_delay=0.0      seconds to wait between steps (0 = as fast as possible)
    play.interactive=false   if true, press Enter before each step
"""

import time

import hydra
import traci
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import sumo_rl_ego as sre
from stable_baselines3 import A2C as SB3A2C


@hydra.main(version_base=None, config_path="../configs", config_name="play.yaml")
def main(cfg: DictConfig):

    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # PROJECT_ROOT = sumo-human-feedback-rl/ (root of the repo)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # ── Reproduce training environment ───────────────────────────────────────
    run_dir   = PROJECT_ROOT / cfg.run.dir
    train_cfg = OmegaConf.load(run_dir / "config.yaml")

    from human_feedback_rl.common.utils.env_setup import _load_policy_cfg
    expert_cfg = _load_policy_cfg(train_cfg.env.expert_model)

    print(f"[play] Scenario : {expert_cfg.env}")
    print(f"[play] Policy   : {cfg.agent.model}")

    env = sre.make_env(expert_cfg.env.id, seed=train_cfg.seed, use_gui=True)

    # ── Load policy (SB3 A2C .zip format) ────────────────────────────────────
    agent_path = PROJECT_ROOT / cfg.agent.model
    policy = SB3A2C.load(str(agent_path), device="cpu")

    # ── GUI setup ─────────────────────────────────────────────────────────────
    obs, _ = env.reset()
    traci.gui.setSchema("View #0", "real world")
    traci.gui.trackVehicle("View #0", "ego")
    traci.gui.setZoom("View #0", 1700)

    max_episodes = cfg.eval.get("episodes", None)   # None = run forever
    episode = 0

    print("\nSUMO GUI aperta. Ctrl+C per uscire.\n")

    # ── Rollout loop ─────────────────────────────────────────────────────────
    while max_episodes is None or episode < max_episodes:

        terminated = truncated = False
        ep_reward  = 0.0
        step       = 0

        while not (terminated or truncated):

            if cfg.play.interactive:
                input(f"  [ep={episode} step={step}] Press Enter to step…")

            action, _ = policy.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step      += 1

            if cfg.play.step_delay > 0:
                time.sleep(cfg.play.step_delay)

        status = info.get("status", "?")
        event  = info.get("event",  "?")
        print(f"[ep={episode}] steps={step}  reward={ep_reward:.2f}"
              f"  status={status}  event={event}")

        episode += 1
        obs, _ = env.reset()

        # Re-centre camera after reset (vehicle ID changes on new episode)
        try:
            traci.gui.trackVehicle("View #0", "ego")
        except Exception:
            pass

    env.close()


if __name__ == "__main__":
    main()
