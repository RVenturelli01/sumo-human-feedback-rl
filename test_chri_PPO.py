import sumo_rl_ego as sre
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from human_feedback_rl.algorithms import ChristianoAlgorithm
from sumo_rl_ego.utils import CustomLoggingCallback
import wandb

from human_feedback_rl.common.loggers import WandbWriter, PrefixedLogger
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import VecMonitor


config = {
    "baseline": False,
    "env": {
        "id": "HighwayEgo-v0",
        "kwargs": {
            "n_envs": 16,
            "base_seed": 0,
            "ego": "continuous",
            "reward": "fast",
        },
    },
    "agent": {
        "policy": "MlpPolicy",
        "n_steps": 1024,
        "n_epochs": 5,
        "learning_rate": 0.0003,
        "batch_size": 256,
        "ent_coef": 0.01,
        "gae_lambda": 0.95,
        "gamma": 0.995,
        "policy_kwargs": {"net_arch": [256, 128]},
        "device": "cpu",
        "seed": 0,
    },
    "algo": {
        "lr_rew": 0.005,
        "n_gradient_steps_rew": 10,
        "batch_size_rew": 200,
        "l2_rew": 0.01,
        "fragment_length": 10,
        "transition_oversampling": 1,
        "initial_comparison_frac": 0.01,
        "initial_epoch_multiplier": 2,
        "comparison_queue_size": 1_000,
    },
    "train": {
        "comparisons_per_iteration": 100,
        "timesteps_per_iteration": 16 * 1024,
        "total_timesteps": 128,
    },
}


def get_name(config):
    total_timesteps = config["train"]["total_timesteps"]
    timesteps_per_it = config["train"]["timesteps_per_iteration"]
    n_envs = config["env"]["kwargs"]["n_envs"]
    n_steps = config["agent"]["n_steps"]

    base = (
        f"PPO"
        f" {n_envs}x{n_steps}={n_envs * n_steps}"
        f" tot_steps={total_timesteps}"
        f" steps_per_it={timesteps_per_it}"
    )

    if config["baseline"]:
        return base + " (baseline)"
    
    comparisons_per_it = config["train"]["comparisons_per_iteration"]
    segment_length = config["algo"]["fragment_length"]
    
    return base + f" comp_per_it={comparisons_per_it} seg_len={segment_length} (Christiano synch)"


def print_summary(config):
    train = config["train"]
    agent = config["agent"]
    env = config["env"]["kwargs"]

    steps_per_it = train["timesteps_per_iteration"]
    tot_steps = train["total_timesteps"]
    rollout_len = env["n_envs"] * agent["n_steps"]
    n_it = int(tot_steps / steps_per_it)

    print("=" * 60)
    print(f"  {get_name(config)}")
    print("=" * 60)
    print(f"  Iterations:       {n_it}")
    print(f"  Timesteps:        {steps_per_it:>4} / it    total: {steps_per_it * n_it:>10}")
    print(f"  Rollout length:   {rollout_len}  ({env['n_envs']} envs x {agent['n_steps']} steps)")

    if not config["baseline"]:
        algo = config["algo"]
        comp_per_it = train["comparisons_per_iteration"]
        seg_len = algo["fragment_length"]
        print(f"  Comparisons:      {comp_per_it:>4} / it    total: {comp_per_it * n_it:>10}")
        print(f"  Segment length:   {seg_len}")
        print("-" * 60)
        print("  Reward model:")
        print(f"    lr:                    {algo['lr_rew']}")
        print(f"    gradient steps:        {algo['n_gradient_steps_rew']}")
        print(f"    batch size:            {algo['batch_size_rew']}")
        print(f"    comparison_queue_size: {algo['comparison_queue_size']}")

    print("=" * 60)




if __name__ == '__main__':
    wandb.init(
        project="temp",
        entity="rventurelli-politecnico-di-milano",
        config=config,
        name=get_name(config),
    )

    print_summary(config)

    print("Creating environment...")
    env = sre.make_vec_env(config["env"]["id"], **config["env"]["kwargs"])

    print("Initializing agent...")
    agent = PPO(env=env, **config["agent"])

    if config["baseline"]:
        agent.set_env(VecMonitor(env))
        logger = Logger(folder=None, output_formats=[WandbWriter()])
        agent.set_logger(PrefixedLogger(logger, "agent"))

        total_timesteps = config["train"]["n_iterations"] * config["train"]["timesteps_per_iteration"]

        agent.learn(
            total_timesteps=total_timesteps,
            callback=CustomLoggingCallback(),
        )

    else:
        print("Initializing algorithm...")
        algo = ChristianoAlgorithm(env=env, agent=agent, **config["algo"])

        print("Starting training...")
        agent = algo.train(**config["train"])
