"""Launch the confirmed demo-anchor reliability ablation on the server.

This intentionally reuses the server's gradient-diagnostics launcher and the
winning ``hybrid_demo_2_hom_soft`` Optuna export. Only experiment identity,
gradient fusion, and the confirmed 10% initial-query rule differ.
"""

import os
import subprocess

from omegaconf import OmegaConf

import launch_grad_diagnostics as base


SOFT_LANE = "demo_anchor_soft"
BERN_LANE = "demo_anchor_bern"
DEMO_LANE = "demo_no_norm_compare"
SOFT_CURVE_ARM = "hybrid_demo_2_soft_hom_demo_anchor_rel"
BERN_CURVE_ARM = "hybrid_demo_2_bern_hom_demo_anchor_rel"

base.SLOTS[SOFT_LANE] = tuple(str(core) for core in range(24, 33))
base.LANES[SOFT_LANE] = {
    "arm": "hybrid_demo_2",
    "preferences": True,
    "study_suffix": "_hom_soft",
    "preference_labels": "soft",
    "curve_arm": SOFT_CURVE_ARM,
    "pref_temperature": 20.0,
    "budgets": (10, 100, 1000),
    "initial_queries": {10: 1, 100: 10, 1000: 100},
    "normalize_agent_reward": True,
    "tuned": {
        "gradient_steps_rew": 139,
        "batch_size_pref": 256,
        "batch_size_expert": 16,
    },
}
base.SLOTS[BERN_LANE] = tuple(str(core) for core in range(33, 39))
base.LANES[BERN_LANE] = {
    "arm": "hybrid_demo_2",
    "preferences": True,
    "study_suffix": "_hom_bern",
    "preference_labels": "binary_bernoulli",
    "curve_arm": BERN_CURVE_ARM,
    "pref_temperature": 3.0595414013726767,
    "budgets": (1000, 2723),
    "initial_queries": {1000: 100, 2723: 272},
    "normalize_agent_reward": True,
    "tuned": {
        "gradient_steps_rew": 78,
        "batch_size_pref": 256,
        "batch_size_expert": 64,
        "initial_agent_timesteps": 40000,
    },
}
base.SLOTS[DEMO_LANE] = tuple(str(core) for core in range(39, 45))
base.LANES[DEMO_LANE] = {
    "arm": "demo_2",
    "preferences": False,
    "study_suffix": "_no_norm",
    "preference_labels": "auto",
    "curve_arm": "demo_2_no_norm",
    "pref_temperature": None,
    "budgets": (100, 1000),
    "initial_queries": {100: 0, 1000: 0},
    "normalize_agent_reward": False,
    "tuned": {
        "gradient_steps_rew": 100,
        "batch_size_expert": 16,
        "batch_size_model": 64,
        "initial_agent_timesteps": 20000,
    },
}

_base_task_overrides = base.task_overrides
_base_validate = base.validate


def task_overrides(task):
    if task.lane == DEMO_LANE:
        return (
            *_base_task_overrides(task),
            f"wandb.tags=[grad_diagnostics,demo_2_no_norm,demo_control,B{task.budget}]",
        )
    family = (
        "hybrid_demo_2_soft_hom"
        if task.lane == SOFT_LANE
        else "hybrid_demo_2_bern_hom"
    )
    return (
        *_base_task_overrides(task),
        "algo.kwargs.gradient_fusion=demo_anchor_reliability",
        "algo.kwargs.reliability_ema_beta=0.9",
        "algo.kwargs.reliability_lambda_max=1.0",
        f"wandb.tags=[grad_diagnostics,demo_anchor_reliability,"
        f"{family},hom,B{task.budget}]",
    )


def validate(lane, tasks, exports):
    records = _base_validate(lane, tasks, exports)
    seen = set()
    for task in tasks:
        if task.budget in seen:
            continue
        seen.add(task.budget)
        command = base.full_command(task, exports[task.budget])
        command[2:2] = ["--cfg", "job", "--resolve"]
        result = subprocess.run(
            command,
            cwd=base.REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        cfg = OmegaConf.create(result.stdout)
        checks = {
            "initial_queries": (cfg.algo.kwargs.initial_queries, task.initial_queries),
            "total_timesteps": (cfg.train.kwargs.total_timesteps, 2_000_000),
        }
        if lane != DEMO_LANE:
            checks.update({
                "gradient_fusion": (
                    cfg.algo.kwargs.gradient_fusion,
                    "demo_anchor_reliability",
                ),
                "reliability_ema_beta": (cfg.algo.kwargs.reliability_ema_beta, 0.9),
                "reliability_lambda_max": (
                    cfg.algo.kwargs.reliability_lambda_max,
                    1.0,
                ),
            })
        if lane == BERN_LANE:
            checks.update({
                "labels_type": (cfg.algo.kwargs.labels_type, "binary_bernoulli"),
                "pref_temperature": (
                    cfg.algo.kwargs.pref_temperature,
                    3.0595414013726767,
                ),
                "lr_rew": (cfg.algo.kwargs.lr_rew, 0.0003080841576274553),
                "l2_rew": (cfg.algo.kwargs.l2_rew, 0.0005307422191330497),
                "reward_net_arch": (
                    list(cfg.algo.kwargs.reward_model_kwargs.net_arch),
                    [32, 32],
                ),
            })
        if lane == DEMO_LANE:
            checks.update({
                "total_queries": (cfg.algo.kwargs.total_queries, 0),
                "train_total_queries": (cfg.train.kwargs.total_queries, 0),
                "normalize_agent_reward": (
                    cfg.algo.kwargs.normalize_agent_reward,
                    False,
                ),
                "lr_rew": (cfg.algo.kwargs.lr_rew, 0.0009187069964354143),
                "l2_rew": (cfg.algo.kwargs.l2_rew, 5.061862748858848e-06),
                "reward_net_arch": (
                    list(cfg.algo.kwargs.reward_model_kwargs.net_arch),
                    [64, 64],
                ),
            })
        bad = {name: values for name, values in checks.items() if values[0] != values[1]}
        if bad:
            raise RuntimeError(f"Demo-anchor config mismatch at B{task.budget}: {bad}")
        for record in records:
            if record["budget"] == task.budget:
                record["checks"].update({name: values[0] for name, values in checks.items()})
    return records


base.task_overrides = task_overrides
base.validate = validate

# Every process is pinned to one core; keep numerical libraries from creating
# extra worker pools before taskset takes effect.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


if __name__ == "__main__":
    raise SystemExit(base.main())
