import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 

from human_feedback_rl.concrete_experts.Concrete_demonstration_expert import (
    ConcreteStepDemonstrationExpert,
)
from human_feedback_rl.Core import Step



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


def sample_random_state(obs_dim):
    return np.random.uniform(-1, 1, size=obs_dim)


def evaluate(policy, expert, obs_dim, n_samples=1000):
    correct = 0

    for _ in range(n_samples):

        state = sample_random_state(obs_dim)

        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0)

        logits = policy(state_tensor)
        agent_action = torch.argmax(logits, dim=1).item()

        step = Step(state, agent_action)
        feedback = expert.query(step)

        expert_action = feedback.value

        if agent_action == expert_action:
            correct += 1

    return correct / n_samples


def main():
    expert_model = torch.load("expert_policies/model.zip")
    expert = ConcreteStepDemonstrationExpert(expert_model)

    obs_dim = 10
    n_actions = 5

    policy = PolicyNetwork(obs_dim, n_actions)

    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    for step_id in range(20000):
        state = sample_random_state(obs_dim)

        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0)

        logits = policy(state_tensor)

        agent_action = torch.argmax(logits, dim=1).item()

        step = Step(state, agent_action)
        feedback = expert.query(step)

        expert_action = feedback.value

        target = torch.tensor(
            [expert_action],
            dtype=torch.long
        )

        loss = torch.nn.functional.cross_entropy(
            logits,
            target
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_id % 1000 == 0:

            acc = evaluate(policy, expert, obs_dim)

            print(
                "step:",
                step_id,
                "loss:",
                loss.item(),
                "accuracy:",
                acc
            )


if __name__ == "__main__":
    main()