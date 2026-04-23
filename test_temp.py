
import sumo_rl_ego as sre
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from human_feedback_rl.algorithms import ChristianoAlgorithm
import wandb
    
wandb.init(
    project="temp",
    entity="rventurelli-politecnico-di-milano",
    config={
        "algorithm": "ChristianoAlgorithm",
        "agent": "PPO",
        "total_comparisons": 100,
        "total_timesteps": 10_000,
        "n_envs": 1,
        "ego": "continuous",
        "reward": "fast",
    }
)


print("Creating environment...")
env = sre.make_vec_env(
    "HighwayEgo-v0", 
    n_envs=1, 
    base_seed=0,
    ego="continuous",
    reward="fast"
)

print("Initializing agent...")
agent = PPO(
    policy="MlpPolicy",
    env=env
)

print("Initializing algorithm...")
algo = ChristianoAlgorithm(
    env=env,
    agent=agent,
)

print("Starting training...")
agent = algo.train(
    total_comparisons=100,
    total_timesteps=10_000,
)

