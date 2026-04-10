import hydra
import numpy as np
import sumo_rl_ego as sre

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from human_feedback_rl.algorithms import ChristianoAlgorithm, ChristianoSACAlgorithm, DaggerAlgorithm
from human_feedback_rl.algorithms.humlrn import HumLrnAlgorithm
from human_feedback_rl.common import BCPolicy


from sumo_rl_ego.utils import (
    init_wandb, 
    confirm_cfg,
    save_outputs,
    CustomLoggingCallback,
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
                rng=np.random.default_rng(cfg.run.seed),
                **cfg.algo.kwargs,
            )
            with open_dict(cfg):
                cfg.model = {"algo": cfg.algo.agent.algo}

        elif cfg.algo.name == "christiano-sac":
            print("Initializing agent...")
            algo_cls = ALGO_REGISTRY[cfg.algo.agent.algo]
            agent = algo_cls(
                env=env,
                **OmegaConf.to_container(cfg.algo.agent.kwargs, resolve=True)
            )

            print("Initializing algorithm...")
            algo = ChristianoSACAlgorithm(
                env=env,
                agent=agent,
                **cfg.algo.kwargs,
            )
            with open_dict(cfg):
                cfg.model = {"algo": cfg.algo.agent.algo}

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

        elif cfg.algo.name == "humlrn-v0":
            print("Initializing agent...")
            algo_cls = ALGO_REGISTRY[cfg.algo.agent.algo]
            agent = algo_cls(
                env=env,
                **cfg.algo.agent.kwargs,
            )

            print(f"Loading expert ({cfg.algo.expert_id})...")
            expert = sre.load_policy(cfg.algo.expert_id)

            print("Initializing algorithm...")
            algo = HumLrnAlgorithm(
                env=env,
                agent=agent,
                expert=expert,
                **cfg.algo.kwargs,
            )
            with open_dict(cfg):
                cfg.model = {"algo": cfg.algo.agent.algo}

        elif cfg.algo.name == "imitation-preference":
            print("Initializing agent...")
            algo_cls = ALGO_REGISTRY[cfg.algo.agent.algo]

            agent = algo_cls(
                env=env,
                **cfg.algo.agent.kwargs,
            )

            print("Importing imitation modules...")
            from imitation.algorithms.preference_comparisons import PreferenceComparisons
            from imitation.rewards.reward_nets import BasicRewardNet
            from imitation.util.networks import RunningNorm
            from imitation.data.wrappers import RolloutInfoWrapper
            from imitation.algorithms.preference_comparisons import (
                SyntheticGatherer,
                RandomFragmenter,
            )

            print("Wrapping environment...")
            env_wrapper = RolloutInfoWrapper(env)

            print("Initializing reward model...")
            reward_net = BasicRewardNet(
                observation_space=env.observation_space,
                action_space=env.action_space
            )

            from imitation.algorithms.preference_comparisons import AgentTrainer

            trajectory_generator = AgentTrainer(
                algorithm=agent,
                reward_fn=reward_net,
                venv=env_wrapper,
                rng=np.random.default_rng(cfg.run.seed),
            )

            print("Initializing preference components...")
            fragmenter = RandomFragmenter(rng=np.random.default_rng(cfg.run.seed),)
            preference_gatherer = SyntheticGatherer(rng=np.random.default_rng(cfg.run.seed),)

            print("Initializing preference comparisons...")
            algo = PreferenceComparisons(
                trajectory_generator=trajectory_generator,
                reward_model=reward_net,
                num_iterations=cfg.algo.kwargs.num_iterations,
                fragmenter=fragmenter,
                preference_gatherer=preference_gatherer,
                fragment_length=cfg.algo.kwargs.fragment_length,
                transition_oversampling=cfg.algo.kwargs.transition_oversampling,
                initial_comparison_frac=cfg.algo.kwargs.initial_comparison_frac,
                initial_epoch_multiplier=cfg.algo.kwargs.initial_epoch_multiplier,
                query_schedule=cfg.algo.kwargs.query_schedule,
                rng=np.random.default_rng(cfg.run.seed),
                allow_variable_horizon=True,
            )

            with open_dict(cfg):
                cfg.model = {"algo": cfg.algo.agent.algo}


        print("Starting training...")
        algo.train(**OmegaConf.to_container(cfg.algo.train.kwargs, resolve=True))

        
        print("\nTraining finished.")
        save_outputs(cfg, agent)
        print("Run completed successfully.\n")

    finally:
        if env is not None:
            env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
