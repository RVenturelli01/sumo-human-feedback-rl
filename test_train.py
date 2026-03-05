import torch

from human_feedback_rl.concrete_experts.Concrete_demonstration_expert import ConcreteStepDemonstrationExpert
from human_feedback_rl.Core import Step

from sumo_rl_ego.infra.loaders import load_config_from_model
from sumo_rl_ego.infra.builders.env_factory import build_env
from sumo_rl_ego.infra.builders.model_factory import load_model

def main():
    
    model_dir = "sumo-rl-ego/outputs/best/2026-03-04_19-04-17_test_dqn_highway/model.zip"

    cfg = load_config_from_model(model_dir)

    env = build_env(cfg, seed=0)

    expert_model = load_model(env, cfg, load_path=model_dir, seed=0)

    expert = ConcreteStepDemonstrationExpert(expert_model)

    print(expert)


if __name__ == "__main__":
    main()