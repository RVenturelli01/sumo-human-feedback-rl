"""
Entry point for RLHF training.

Usage:
    python scripts/train.py
    python scripts/train.py christiano.oracle=qnet
    python scripts/train.py christiano.use_demo_preferences=true christiano.db_train_maxlen=6000
    python scripts/train.py christiano.label_mode=soft wandb.tags='[soft,env_reward]'
    python scripts/train.py algorithm=dqn
"""
import multiprocessing as mp
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    # On Linux use forkserver: faster startup than spawn (no full re-import),
    # safer than fork (avoids deadlocks when the parent has active threads from
    # wandb.init or PyTorch). On macOS spawn is required.
    method = "forkserver" if sys.platform == "linux" else "spawn"
    mp.set_start_method(method, force=True)

    output_dir = HydraConfig.get().runtime.output_dir

    if cfg.algorithm == "christiano":
        from human_feedback_rl.algorithms.christiano import ChristianoRLHF

        algo = ChristianoRLHF(
            expert_model_path   = cfg.expert_model,
            seed                = cfg.seed,
            n_envs              = cfg.christiano.n_envs,
            device              = cfg.device,
            oracle              = cfg.christiano.oracle,
            label_mode          = cfg.christiano.label_mode,
            oracle_temperature  = cfg.christiano.oracle_temperature,
            use_demonstrations  = cfg.christiano.use_demonstrations,
            use_demo_preferences= cfg.christiano.use_demo_preferences,
            n_reward_predictors = cfg.christiano.n_reward_predictors,
            rp_lr               = cfg.christiano.rp_lr,
            rp_val_interval     = cfg.christiano.rp_val_interval,
            demo_weight         = cfg.christiano.demo_weight,
            demo_margin         = cfg.christiano.demo_margin,
            policy_lr           = cfg.christiano.policy_lr,
            gamma               = cfg.christiano.gamma,
            rollout_steps       = cfg.christiano.rollout_steps,
            entropy_coef        = cfg.christiano.entropy_coef,
            value_coef          = cfg.christiano.value_coef,
            max_gradient_norm   = cfg.christiano.max_gradient_norm,
            initial_prefs       = cfg.christiano.initial_prefs,
            segment_len         = cfg.christiano.segment_len,
            max_segs            = cfg.christiano.max_segs,
            db_train_maxlen     = cfg.christiano.db_train_maxlen,
            db_val_maxlen       = cfg.christiano.db_val_maxlen,
            seg_pipe_maxsize    = cfg.christiano.seg_pipe_maxsize,
            demo_seg_pipe_maxsize = cfg.christiano.demo_seg_pipe_maxsize,
            demo_db_maxlen      = cfg.christiano.demo_db_maxlen,
            disagreement_candidates = cfg.christiano.disagreement_candidates,
            max_query_interval  = cfg.christiano.max_query_interval,
            total_env_steps     = cfg.christiano.total_env_steps,
            rp_reload_interval  = cfg.christiano.rp_reload_interval,
            rp_retrain_min_new_prefs = cfg.christiano.rp_retrain_min_new_prefs,
            policy_save_interval= cfg.christiano.policy_save_interval,
            torch_num_threads   = cfg.christiano.torch_num_threads,
            wandb_project       = cfg.wandb.project,
            wandb_entity        = cfg.wandb.get("entity") or None,
            wandb_tags          = list(cfg.wandb.get("tags", [])),
        )
        algo.train(output_dir=output_dir)

    elif cfg.algorithm == "dqn":
        import torch
        import wandb
        import sumo_rl_ego as sre
        from stable_baselines3 import DQN
        from stable_baselines3.common.vec_env import VecMonitor
        from stable_baselines3.common.callbacks import BaseCallback

        torch.set_num_threads(cfg.dqn.torch_num_threads)

        class _WandbLogger(BaseCallback):
            """Logs episode stats to wandb every ~10k env steps."""
            def _on_step(self):
                n = self.training_env.num_envs
                if self.num_timesteps % 10000 < n:
                    buf = self.model.ep_info_buffer
                    if buf:
                        wandb.log({
                            "policy/mean_episode_length":       sum(e["l"] for e in buf) / len(buf),
                            "policy/mean_episode_avg_true_rew": sum(e["r"] for e in buf) / len(buf),
                        }, step=self.num_timesteps)
                return True

        tags = list(cfg.wandb.get("tags", []))
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity") or None,
            name=tags[0] if tags else None,
            tags=tags,
            config={**dict(cfg.dqn), "seed": cfg.seed},
        )

        env = sre.make_vec_env(cfg.dqn.env_id, n_envs=cfg.dqn.n_envs, base_seed=cfg.seed)
        env = VecMonitor(env)

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=cfg.dqn.learning_rate,
            buffer_size=cfg.dqn.buffer_size,
            learning_starts=cfg.dqn.learning_starts,
            batch_size=cfg.dqn.batch_size,
            tau=cfg.dqn.tau,
            gamma=cfg.dqn.gamma,
            train_freq=cfg.dqn.train_freq,
            gradient_steps=cfg.dqn.gradient_steps,
            target_update_interval=cfg.dqn.target_update_interval,
            exploration_fraction=cfg.dqn.exploration_fraction,
            exploration_initial_eps=cfg.dqn.exploration_initial_eps,
            exploration_final_eps=cfg.dqn.exploration_final_eps,
            max_grad_norm=cfg.dqn.max_grad_norm,
            policy_kwargs={"net_arch": list(cfg.dqn.net_arch)},
            seed=cfg.seed,
            verbose=0,
        )
        model.learn(total_timesteps=cfg.dqn.total_env_steps, callback=_WandbLogger())

        Path(output_dir, "models").mkdir(parents=True, exist_ok=True)
        model.save(str(Path(output_dir) / "models" / "dqn_baseline"))
        wandb.finish()

    else:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm!r}")


if __name__ == "__main__":
    main()
