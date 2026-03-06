import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import traci

from human_feedback_rl.concrete_experts.Concrete_demonstration_expert import (
    ConcreteStepDemonstrationExpert,
)
from human_feedback_rl.Core import Step

from sumo_rl_ego.infra.builders.env_factory import build_env
from sumo_rl_ego.infra.builders.model_factory import load_model
from sumo_rl_ego.infra.loaders.config_loader import load_config_from_model

from torch.utils.tensorboard import SummaryWriter


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ImitationTrainer:

    def __init__(self, env, policy, expert, optimizer):

        self.env = env
        self.policy = policy
        self.expert = expert
        self.optimizer = optimizer

        self.all_episode_rewards = []
        self.all_episode_lengths = []

        self.obs, _ = env.reset()

        self.reset_metrics()

        self.writer = SummaryWriter("runs/demonstration_feedback")

    def reset_metrics(self):

        self.running_loss = 0
        self.correct_actions = 0
        self.total_actions = 0

        self.episode_reward = 0
        self.episode_length = 0

    '''def train_step(self):

        state = self.obs[0] if len(self.obs.shape) > 1 else self.obs

        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0)

        logits = self.policy(state_tensor)

        agent_action = torch.argmax(logits, dim=1).item()

        step = Step(state, agent_action)
        feedback = self.expert.query(step)

        expert_action = int(np.array(feedback.value).item())

        target = torch.tensor([expert_action], dtype=torch.long)

        loss = torch.nn.functional.cross_entropy(logits, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if agent_action == expert_action:
            self.correct_actions += 1

        self.total_actions += 1
        self.running_loss += loss.item()

        self.obs, reward, terminated, truncated, info = self.env.step(agent_action)

        self.episode_reward += reward
        self.episode_length += 1

        done = terminated or truncated

        if done:
            self.all_episode_rewards.append(self.episode_reward)
            self.all_episode_lengths.append(self.episode_length)

            print(
                "episode finished | length:",
                self.episode_length,
                "reward:",
                self.episode_reward,
                "collided:",
                self.env.metrics_tracker.ep_collided,
                info
            )

            episode_id = len(self.all_episode_rewards)

            self.writer.add_scalar(
                "episode/reward",
                float(self.episode_reward),
                episode_id
            )

            self.writer.add_scalar(
                "episode/length",
                int(self.episode_length),
                episode_id
            )

            self.obs, _ = self.env.reset()

            #traci.gui.setSchema("View #0", "real world")
            #traci.gui.trackVehicle("View #0", "ego")
            #traci.gui.setZoom("View #0", 5000)

            self.episode_reward = 0
            self.episode_length = 0'''

    def train_step(self):

        state = self.obs[0] if len(self.obs.shape) > 1 else self.obs

        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0)

        logits = self.policy(state_tensor)

        agent_action = torch.argmax(logits, dim=1).item()

        step = Step(state, agent_action)
        feedback = self.expert.query(step)

        expert_action = int(np.array(feedback.value).item())

        target = torch.tensor([expert_action], dtype=torch.long)

        loss = torch.nn.functional.cross_entropy(logits, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if agent_action == expert_action:
            self.correct_actions += 1

        self.total_actions += 1
        self.running_loss += loss.item()

        self.obs, reward, terminated, truncated, info = self.env.step(agent_action)

        self.episode_reward += reward
        self.episode_length += 1

        return terminated or truncated

    '''def log_metrics(self, step):

        avg_loss = self.running_loss / self.total_actions
        accuracy = self.correct_actions / self.total_actions

        collision_rate = self.env.metrics_tracker.get_log_metrics().get("cause/collision_rate")

        if collision_rate is not None:
            collision_rate = float(np.array(collision_rate).item())

        print(
            "step:", step,
            "loss:", avg_loss,
            "accuracy:", accuracy,
            "collided:", collision_rate,
            "global collision:", self.env.metrics_tracker.global_collisions,
            "window:", self.env.metrics_tracker.window
        )

        self.writer.add_scalar("train/loss", float(avg_loss), step)
        self.writer.add_scalar("train/accuracy", float(accuracy), step)

        if collision_rate is not None:
            self.writer.add_scalar(
                "env/collision_rate",
                float(collision_rate),
                step
            )

        self.running_loss = 0
        self.correct_actions = 0
        self.total_actions = 0'''

    def log_metrics(self, episode):

        avg_loss = self.running_loss / self.total_actions
        accuracy = self.correct_actions / self.total_actions

        metrics = self.env.metrics_tracker.get_log_metrics()

        collision_rate = metrics.get("cause/collision_rate")

        if collision_rate is not None:
            collision_rate = float(np.array(collision_rate).item())

        print(
            "episode:", episode,
            "loss:", avg_loss,
            "accuracy:", accuracy,
            "collision_rate:", collision_rate
        )

        self.writer.add_scalar("train/loss", float(avg_loss), episode)
        self.writer.add_scalar("train/accuracy", float(accuracy), episode)

        if collision_rate is not None:
            self.writer.add_scalar(
                "env/collision_rate",
                collision_rate,
                episode
            )

        self.running_loss = 0
        self.correct_actions = 0
        self.total_actions = 0

    '''def train(self, steps=20000, log_interval=1000):

        for step in range(steps):

            self.train_step()

            if step % log_interval == 0 and step > 0:
                self.log_metrics(step)

        if len(self.all_episode_rewards) > 0:
            mean_reward = np.mean(self.all_episode_rewards)
            std_reward = np.std(self.all_episode_rewards)

            mean_length = np.mean(self.all_episode_lengths)
            std_length = np.std(self.all_episode_lengths)

            print("\n===== TRAINING SUMMARY =====")
            print("episodes:", len(self.all_episode_rewards))
            print("mean reward:", mean_reward)
            print("std reward:", std_reward)
            print("mean episode length:", mean_length)
            print("std episode length:", std_length)

        self.writer.add_scalar("summary/mean_reward", mean_reward)
        self.writer.add_scalar("summary/std_reward", std_reward)
        self.writer.add_scalar("summary/mean_episode_length", mean_length)
        self.writer.add_scalar("summary/std_episode_length", std_length)

        self.writer.close()'''

    def train(self, total_episodes=10000, log_interval=1000):

        episode = 0

        while episode < total_episodes:

            self.obs, _ = self.env.reset()

            self.episode_reward = 0
            self.episode_length = 0

            done = False

            while not done:

                done = self.train_step()

            episode += 1

            print("episode:", episode)

            self.all_episode_rewards.append(self.episode_reward)
            self.all_episode_lengths.append(self.episode_length)

            episode_id = episode

            self.writer.add_scalar(
                "episode/reward",
                float(self.episode_reward),
                episode_id
            )

            self.writer.add_scalar(
                "episode/length",
                int(self.episode_length),
                episode_id
            )

            if episode % log_interval == 0:

                self.log_metrics(episode)

        self.writer.close()


def main():

    model_dir = "2026-03-04_19-04-17_test_dqn_highway/model.zip"
    cfg = load_config_from_model(model_dir)
    env = build_env(cfg, seed=0)

    expert_model = load_model(
        env,
        cfg,
        load_path=model_dir,
        seed=0
    )
    expert = ConcreteStepDemonstrationExpert(expert_model)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNetwork(obs_dim, n_actions)

    optimizer = optim.Adam(
        policy.parameters(),
        lr=3e-4
    )

    trainer = ImitationTrainer(
        env=env,
        policy=policy,
        expert=expert,
        optimizer=optimizer
    )

    trainer.train(
        total_episodes=200,
        log_interval=20
    )


    env.close()


if __name__ == "__main__":
    main()