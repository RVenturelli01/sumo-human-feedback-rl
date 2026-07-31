#!/usr/bin/env python3
"""Run the approved budget-curve completion matrix on CPU cores 24-47.

The matrix contains 19 new points x seeds 1, 2, 3 = 57 runs.  Hybrid points
are queued first because those arms currently have no budget-curve points.

Safety properties:

* exact Optuna studies/trials/parameters are validated before any launch;
* every task has deterministic output, log, group, name, and W&B run ID;
* the already-running two-core cohort is drained without refill or repinning;
* subsequent runs use one core each (24 concurrent slots over cores 24-47);
* transition state survives manager restarts and skips completed outputs;
* retries are bounded and recorded;
* dry-run and preflight modes never launch training.

Usage:
    python scripts/schedule_budget_curves_completion.py --preflight
    python scripts/schedule_budget_curves_completion.py --dry-run
    python scripts/schedule_budget_curves_completion.py
    python scripts/schedule_budget_curves_completion.py --status
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import math
import netrc
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/home/fis3/miniconda3/envs/sumo-rlhf/bin/python")
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_hybrid_sac.py"
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export_best_config.py"
JOURNAL = REPO_ROOT / "outputs" / "optuna" / "journal.log"
DATASET = REPO_ROOT / "datasets" / "expert_trajectories_no_collision.pkl"

STATE_ROOT = REPO_ROOT / "outputs" / "budget_curves_completion"
RUN_ROOT = STATE_ROOT / "runs"
LOG_ROOT = STATE_ROOT / "logs"
MARKER_ROOT = STATE_ROOT / "markers"
MANIFEST_PATH = STATE_ROOT / "manifest.json"
TRANSITION_MANIFEST_PATH = STATE_ROOT / "transition_manifest.json"
TRANSITION_STATE_PATH = STATE_ROOT / "transition_state.json"
LOCK_PATH = STATE_ROOT / "orchestrator.lock"

WANDB_ENTITY = "andrea02polimi-politecnico-di-milano"
WANDB_PROJECT = "tuning-thesis-budget-curves-completion"

SEEDS = (1, 2, 3)
LEGACY_PAIR_SLOTS = tuple(
    f"{core}-{core + 1}" for core in range(24, 48, 2)
)
SINGLE_SLOTS = tuple(str(core) for core in range(24, 48))
# New launches always use singleton slots.  Keep the short alias only for
# command/manifest compatibility with the first orchestrator version.
SLOTS = SINGLE_SLOTS
ALL_ADOPTABLE_SLOTS = LEGACY_PAIR_SLOTS + SINGLE_SLOTS
TRANSITION_VERSION = 1
PHASE_DRAIN = "draining_legacy_pairs"
PHASE_SINGLE = "single_core"
TOTAL_TIMESTEPS = 2_000_000
TIMESTEPS_PER_ITERATION = 20_000
N_ITERATIONS = TOTAL_TIMESTEPS // TIMESTEPS_PER_ITERATION
N_ENVS = 2
EVAL_EPISODES = 20
MAX_ATTEMPTS = 2
LOOP_SECONDS = 10
CPU_SAMPLE_SECONDS = 1
CPU_MEAN_BUSY = 35.0
CPU_MAX_BUSY = 70.0
IDLE_SCANS_REQUIRED = 2
CANARY_SECONDS = 60
MIN_FREE_BYTES = 10 * 1024**3
PREF_BERNOULLI_POINTS = ((1_000, 200), (1_000_000, 200_000))
LEGACY_PREF_BERNOULLI_POINTS = ((1_000, 200), (15_000, 3_000))

STOP_REQUESTED = False


@dataclass(frozen=True)
class ConfigSource:
    key: str
    arm: str
    study_suffix: str
    expected_trial: int
    preference_labels: str
    expected_params: dict[str, Any]

    @property
    def study_name(self) -> str:
        return f"hybrid_sac_{self.arm}{self.study_suffix}"


CONFIG_SOURCES = {
    "demo_1": ConfigSource(
        key="demo_1",
        arm="demo_1",
        study_suffix="",
        expected_trial=24,
        preference_labels="auto",
        expected_params={
            "lr_rew": 0.0014456657756301045,
            "gradient_steps_rew": 20,
            "l2_rew": 0.0004360175899125305,
            "reward_net_arch": "[64,64]",
            "initial_agent_timesteps": 20000,
            "batch_size_expert": 32,
            "batch_size_model": 32,
        },
    ),
    "demo_2_no_norm": ConfigSource(
        key="demo_2_no_norm",
        arm="demo_2",
        study_suffix="_no_norm",
        expected_trial=26,
        preference_labels="auto",
        expected_params={
            "lr_rew": 0.0009187069964354143,
            "gradient_steps_rew": 100,
            "l2_rew": 5.061862748858848e-06,
            "reward_net_arch": "[64,64]",
            "initial_agent_timesteps": 20000,
            "batch_size_expert": 16,
            "batch_size_model": 64,
        },
    ),
    "pref_soft": ConfigSource(
        key="pref_soft",
        arm="pref_soft",
        study_suffix="",
        expected_trial=3,
        preference_labels="soft",
        expected_params={
            "lr_rew": 0.0018373242658509387,
            "gradient_steps_rew": 23,
            "l2_rew": 0.00012704069184662418,
            "reward_net_arch": "[32,32]",
            "initial_agent_timesteps": 40000,
            "batch_size_pref": 64,
            "query_schedule": "constant",
            "initial_queries": 250,
            "fragmenter_type": "random",
        },
    ),
    "pref_bernoulli": ConfigSource(
        key="pref_bernoulli",
        arm="pref_bernoulli",
        study_suffix="_q100k_temp",
        expected_trial=21,
        preference_labels="binary_bernoulli",
        expected_params={
            "lr_rew": 0.0008519268053820848,
            "gradient_steps_rew": 99,
            "l2_rew": 1.1190973215409014e-06,
            "reward_net_arch": "[128,128]",
            "initial_agent_timesteps": 20000,
            "batch_size_pref": 64,
            "pref_temperature": 3.0595414013726767,
            "query_schedule": "hyperbolic",
            "initial_queries": 20000,
            "fragmenter_type": "active",
        },
    ),
    "hybrid_demo_2_soft": ConfigSource(
        key="hybrid_demo_2_soft",
        arm="hybrid_demo_2",
        study_suffix="",
        expected_trial=10,
        preference_labels="soft",
        expected_params={
            "lr_rew": 0.00046178781677192805,
            "gradient_steps_rew": 147,
            "l2_rew": 1.981463066451472e-05,
            "reward_net_arch": "[128,128]",
            "initial_agent_timesteps": 20000,
            "batch_size_pref": 256,
            "batch_size_expert": 128,
            "demo_weight": 9.662826870596577,
        },
    ),
    "hybrid_demo_2_bernoulli": ConfigSource(
        key="hybrid_demo_2_bernoulli",
        arm="hybrid_demo_2",
        study_suffix="_bernoulli_norm",
        expected_trial=11,
        preference_labels="binary_bernoulli",
        expected_params={
            "lr_rew": 0.002432388041110311,
            "gradient_steps_rew": 28,
            "l2_rew": 0.0005634790500586056,
            "reward_net_arch": "[8,8]",
            "initial_agent_timesteps": 20000,
            "batch_size_pref": 128,
            "pref_temperature": 25.115553877134268,
            "initial_queries": 10000,
            "batch_size_expert": 128,
            "demo_weight": 3.410517404291985,
        },
    ),
}


@dataclass(frozen=True)
class Point:
    arm: str
    budget: int
    source_key: str
    pref_budget: int
    demo_budget: int | None
    initial_queries: int
    normalize_agent_reward: bool
    labels_type: str
    loss_type: str
    query_schedule: str
    fragmenter_type: str
    pref_temperature: float
    demo_weight: float
    # Optional demo budget counted in TRANSITIONS instead of trajectories.
    # When set it takes precedence over demo_budget in the emitted overrides
    # (run.n_expert_trajectories=null + run.n_expert_transitions=<value>).
    # Defaults to None so every existing Point is unchanged.
    demo_transitions: int | None = None

    @property
    def group(self) -> str:
        return f"budget_{self.arm}_{self.budget}"


@dataclass(frozen=True)
class Task:
    point: Point
    seed: int

    @property
    def arm(self) -> str:
        return self.point.arm

    @property
    def budget(self) -> int:
        return self.point.budget

    @property
    def group(self) -> str:
        return self.point.group

    @property
    def run_name(self) -> str:
        return f"{self.group}-seed{self.seed}"

    @property
    def key(self) -> str:
        return f"{self.arm}_b{self.budget}_s{self.seed}"

    @property
    def wandb_run_id(self) -> str:
        payload = f"{WANDB_PROJECT}/{self.run_name}".encode()
        return "bc" + hashlib.sha1(payload).hexdigest()[:18]

    @property
    def output_root(self) -> Path:
        return RUN_ROOT / self.group

    @property
    def log_path(self) -> Path:
        return LOG_ROOT / f"{self.key}.log"

    @property
    def demo_subsample_seed(self) -> int | None:
        if self.point.demo_budget is None:
            return None
        return 1000 + self.seed

    def manifest_record(self) -> dict[str, Any]:
        record = {
            "key": self.key,
            "arm": self.arm,
            "budget": self.budget,
            "pref_budget": self.point.pref_budget,
            "demo_budget": self.point.demo_budget,
            "initial_queries": self.point.initial_queries,
            "seed": self.seed,
            "demo_subsample_seed": self.demo_subsample_seed,
            "source": self.point.source_key,
            "normalize_agent_reward": self.point.normalize_agent_reward,
            "labels_type": self.point.labels_type,
            "loss_type": self.point.loss_type,
            "query_schedule": self.point.query_schedule,
            "fragmenter_type": self.point.fragmenter_type,
            "pref_temperature": self.point.pref_temperature,
            "demo_weight": self.point.demo_weight,
            "group": self.group,
            "run_name": self.run_name,
            "wandb_run_id": self.wandb_run_id,
            "output_root": str(self.output_root.relative_to(REPO_ROOT)),
            "log_path": str(self.log_path.relative_to(REPO_ROOT)),
        }
        # Emitted only when set: task_matrix_sha256() hashes these records, so
        # an unconditional key would change the matrix signature of every
        # existing task and invalidate the stored transition state.
        if self.point.demo_transitions is not None:
            record["demo_transitions"] = self.point.demo_transitions
        return record


def _hybrid_point(arm: str, budget: int, labels_type: str) -> Point:
    half = budget // 2
    initial_by_pref_budget = {
        1: 1,
        10: 1,
        100: 10,
        1000: 100,
        2723: 272,
    }
    temperature = 20.0 if labels_type == "soft" else 25.115553877134268
    return Point(
        arm=arm,
        budget=budget,
        source_key=arm,
        pref_budget=half,
        demo_budget=half,
        initial_queries=initial_by_pref_budget[half],
        normalize_agent_reward=True,
        labels_type=labels_type,
        loss_type="demo_2",
        query_schedule="constant",
        fragmenter_type="active",
        pref_temperature=temperature,
        demo_weight=1.0,
    )


def build_points(
    pref_bernoulli_points: tuple[tuple[int, int], ...] = (
        PREF_BERNOULLI_POINTS
    ),
    include_max_hybrid_bernoulli: bool = True,
) -> tuple[Point, ...]:
    """Return points in launch priority order, three seeds per point later."""
    points: list[Point] = []

    # Two complete hybrid curves first, alternating arms at every budget.
    for budget in (2, 20, 200, 2000):
        points.append(_hybrid_point("hybrid_demo_2_soft", budget, "soft"))
        points.append(
            _hybrid_point(
                "hybrid_demo_2_bernoulli", budget, "binary_bernoulli"
            )
        )
    if include_max_hybrid_bernoulli:
        points.append(
            _hybrid_point(
                "hybrid_demo_2_bernoulli", 5_446, "binary_bernoulli"
            )
        )

    # Then complete the low-budget side of the four existing arms.
    demo_common = {
        "pref_budget": 0,
        "initial_queries": 0,
        "labels_type": "soft",
        "query_schedule": "constant",
        "fragmenter_type": "active",
        "pref_temperature": 20.0,
        "demo_weight": 1.0,
    }
    for budget in (1, 10, 20):
        points.extend(
            [
                Point(
                    arm="demo_1",
                    budget=budget,
                    source_key="demo_1",
                    demo_budget=budget,
                    normalize_agent_reward=True,
                    loss_type="demo_1",
                    **demo_common,
                ),
                Point(
                    arm="demo_2_no_norm",
                    budget=budget,
                    source_key="demo_2_no_norm",
                    demo_budget=budget,
                    normalize_agent_reward=False,
                    loss_type="demo_2",
                    **demo_common,
                ),
            ]
        )

    for budget, initial_queries in ((10, 2), (100, 20), (250, 50)):
        points.append(
            Point(
                arm="pref_soft",
                budget=budget,
                source_key="pref_soft",
                pref_budget=budget,
                demo_budget=None,
                initial_queries=initial_queries,
                normalize_agent_reward=True,
                labels_type="soft",
                loss_type="demo_2",
                query_schedule="constant",
                fragmenter_type="random",
                pref_temperature=20.0,
                demo_weight=0.0,
            )
        )

    for budget, initial_queries in pref_bernoulli_points:
        points.append(
            Point(
                arm="pref_bernoulli",
                budget=budget,
                source_key="pref_bernoulli",
                pref_budget=budget,
                demo_budget=None,
                initial_queries=initial_queries,
                normalize_agent_reward=True,
                labels_type="binary_bernoulli",
                loss_type="demo_2",
                query_schedule="hyperbolic",
                fragmenter_type="active",
                pref_temperature=3.0595414013726767,
                demo_weight=0.0,
            )
        )
    return tuple(points)


def build_tasks(
    pref_bernoulli_points: tuple[tuple[int, int], ...] = (
        PREF_BERNOULLI_POINTS
    ),
    include_max_hybrid_bernoulli: bool = True,
) -> tuple[Task, ...]:
    return tuple(
        Task(point, seed)
        for point in build_points(
            pref_bernoulli_points,
            include_max_hybrid_bernoulli=include_max_hybrid_bernoulli,
        )
        for seed in SEEDS
    )


def slot_cores(slot: str) -> frozenset[int]:
    values = slot.split("-")
    if len(values) == 1:
        lower = upper = int(values[0])
    elif len(values) == 2:
        lower, upper = (int(value) for value in values)
    else:
        raise ValueError(f"Invalid CPU slot {slot!r}.")
    if lower > upper:
        raise ValueError(f"Invalid CPU slot {slot!r}.")
    return frozenset(range(lower, upper + 1))


def task_matrix_sha256(tasks: tuple[Task, ...]) -> str:
    payload = [task.manifest_record() for task in tasks]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_task_matrix(tasks: tuple[Task, ...]) -> dict[str, Any]:
    if len(tasks) != 60:
        raise RuntimeError(f"Expected 60 tasks, got {len(tasks)}.")
    if len(build_points()) != 20:
        raise RuntimeError(f"Expected 20 points, got {len(build_points())}.")

    for attribute in ("key", "run_name", "wandb_run_id"):
        values = [getattr(task, attribute) for task in tasks]
        if len(values) != len(set(values)):
            raise RuntimeError(f"Duplicate task {attribute}.")

    groups: dict[str, list[int]] = {}
    for task in tasks:
        groups.setdefault(task.group, []).append(task.seed)
        point = task.point
        if not 0 <= point.initial_queries <= point.pref_budget:
            raise RuntimeError(f"Invalid initial_queries for {task.key}.")
        if point.demo_budget is not None and not 1 <= point.demo_budget <= 2723:
            raise RuntimeError(f"Invalid demo budget for {task.key}.")
        if point.arm.startswith("hybrid_"):
            if point.pref_budget + (point.demo_budget or 0) != point.budget:
                raise RuntimeError(f"Invalid hybrid split for {task.key}.")
            if point.demo_weight != 1.0 or not point.normalize_agent_reward:
                raise RuntimeError(f"Invalid approved hybrid overrides for {task.key}.")
    if any(sorted(seeds) != list(SEEDS) for seeds in groups.values()):
        raise RuntimeError("Every point must contain seeds 1, 2, 3 exactly once.")

    for label, slots in (
        ("legacy", LEGACY_PAIR_SLOTS),
        ("single", SINGLE_SLOTS),
    ):
        all_cores = [core for slot in slots for core in slot_cores(slot)]
        if all_cores != list(range(24, 48)) or len(all_cores) != len(
            set(all_cores)
        ):
            raise RuntimeError(
                f"{label} CPU slots must partition cores 24-47 exactly."
            )

    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.arm] = counts.get(task.arm, 0) + 1
    expected_counts = {
        "demo_1": 9,
        "demo_2_no_norm": 9,
        "pref_soft": 9,
        "pref_bernoulli": 6,
        "hybrid_demo_2_soft": 12,
        "hybrid_demo_2_bernoulli": 15,
    }
    if counts != expected_counts:
        raise RuntimeError(f"Unexpected per-arm task counts: {counts}.")
    return {
        "tasks": len(tasks),
        "points": len(groups),
        "per_arm": counts,
        "matrix_sha256": task_matrix_sha256(tasks),
    }


def params_equal(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    if actual.keys() != expected.keys():
        return False
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, float):
            if not isinstance(actual_value, (int, float)) or not math.isclose(
                float(actual_value),
                expected_value,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                return False
        elif actual_value != expected_value:
            return False
    return True


def validate_optuna_sources() -> list[dict[str, Any]]:
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    storage = JournalStorage(
        JournalFileBackend(
            str(JOURNAL), lock_obj=JournalFileOpenLock(str(JOURNAL))
        )
    )
    validated = []
    for source in CONFIG_SOURCES.values():
        study = optuna.load_study(study_name=source.study_name, storage=storage)
        trial = study.best_trial
        if trial.number != source.expected_trial:
            raise RuntimeError(
                f"{source.study_name}: expected best trial "
                f"#{source.expected_trial}, found #{trial.number}."
            )
        if not params_equal(trial.params, source.expected_params):
            raise RuntimeError(
                f"{source.study_name}: best params differ from the approved snapshot.\n"
                f"expected={source.expected_params}\nactual={trial.params}"
            )
        stored_labels = study.user_attrs.get("preference_labels")
        if (
            source.preference_labels != "auto"
            and stored_labels is not None
            and stored_labels != source.preference_labels
        ):
            raise RuntimeError(
                f"{source.study_name}: preference_labels={stored_labels!r}, "
                f"expected {source.preference_labels!r}."
            )
        validated.append(
            {
                "key": source.key,
                "study": source.study_name,
                "trial": trial.number,
                "objective": trial.value,
                "preference_labels": stored_labels,
                "params": trial.params,
            }
        )
    return validated


def validate_runtime_query_schedules(
    tasks: tuple[Task, ...],
) -> dict[str, Any]:
    from human_feedback_rl.common.base_reward_learning_algorithm import (
        BaseRewardLearningAlgorithm,
        QUERY_SCHEDULES,
    )

    expected_module = (
        REPO_ROOT
        / "human-feedback-rl"
        / "human_feedback_rl"
        / "common"
        / "base_reward_learning_algorithm.py"
    ).resolve()
    loaded_module = Path(
        inspect.getsourcefile(BaseRewardLearningAlgorithm) or ""
    ).resolve()
    if loaded_module != expected_module:
        raise RuntimeError(
            "The training environment imports human_feedback_rl from the "
            f"wrong path: {loaded_module}; expected {expected_module}."
        )

    cases = {
        (
            task.point.query_schedule,
            task.point.pref_budget,
            task.point.initial_queries,
        )
        for task in tasks
        if task.point.pref_budget > 0
    }
    records = []
    for schedule_name, total, initial in sorted(cases):
        holder = SimpleNamespace(
            query_schedule=QUERY_SCHEDULES[schedule_name],
            query_schedule_name=schedule_name,
            initial_queries=initial,
        )
        schedule = BaseRewardLearningAlgorithm.build_query_schedule(
            holder, N_ITERATIONS, total
        )
        if (
            len(schedule) != N_ITERATIONS
            or sum(schedule) != total
            or any(count < 0 for count in schedule)
        ):
            raise RuntimeError(
                f"Invalid {schedule_name} schedule for total={total}, "
                f"initial={initial}: len={len(schedule)} sum={sum(schedule)}"
            )
        if schedule_name == "constant":
            main_loop = list(schedule)
            main_loop[0] -= initial
            remaining = total - initial
            cumulative = 0
            for iteration, count in enumerate(main_loop, start=1):
                cumulative += count
                target = iteration * remaining // N_ITERATIONS
                if cumulative != target:
                    raise RuntimeError(
                        f"Non-uniform constant schedule for total={total}, "
                        f"initial={initial}, iteration={iteration}: "
                        f"{cumulative} != {target}"
                    )
        records.append(
            {
                "name": schedule_name,
                "total": total,
                "initial": initial,
                "remaining": total - initial,
                "minimum_main_loop": min(
                    [
                        count - initial if index == 0 else count
                        for index, count in enumerate(schedule)
                    ]
                ),
                "maximum_main_loop": max(
                    [
                        count - initial if index == 0 else count
                        for index, count in enumerate(schedule)
                    ]
                ),
            }
        )
    return {"module": str(loaded_module), "cases": records}


@lru_cache(maxsize=None)
def export_overrides(
    source_key: str, pref_budget: int, demo_budget: int | None
) -> tuple[str, ...]:
    source = CONFIG_SOURCES[source_key]
    command = [
        str(PYTHON),
        str(EXPORT_SCRIPT),
        "--arm",
        source.arm,
        "--format",
        "full",
        "--storage-path",
        str(JOURNAL.relative_to(REPO_ROOT)),
        "--pref-budget",
        str(pref_budget or 5000),
        "--demo-budget",
        str(demo_budget or 500),
        "--preference-labels",
        source.preference_labels,
    ]
    if source.study_suffix:
        command.extend(["--study-suffix", source.study_suffix])
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    overrides = tuple(shlex.split(result.stdout.strip()))
    if not overrides:
        raise RuntimeError(f"No overrides exported for {source.study_name}.")
    return overrides


def task_overrides(task: Task) -> tuple[str, ...]:
    point = task.point
    overrides = [
        f"algo.kwargs.total_queries={point.pref_budget}",
        f"train.kwargs.total_queries={point.pref_budget}",
        f"algo.kwargs.initial_queries={point.initial_queries}",
        f"algo.kwargs.normalize_agent_reward={str(point.normalize_agent_reward).lower()}",
        f"algo.kwargs.labels_type={point.labels_type}",
        f"algo.kwargs.loss_type={point.loss_type}",
        f"algo.kwargs.query_schedule={point.query_schedule}",
        f"algo.kwargs.fragmenter_type={point.fragmenter_type}",
        f"algo.kwargs.pref_temperature={point.pref_temperature}",
        f"algo.kwargs.demo_weight={point.demo_weight}",
    ]
    if point.demo_transitions is not None:
        # Transition budget wins over the trajectory one. The exported config
        # block already carries run.n_expert_trajectories=<demo_budget>; these
        # come after it and Hydra keeps the last occurrence.
        overrides.extend(
            [
                "run.n_expert_trajectories=null",
                f"run.n_expert_transitions={point.demo_transitions}",
                f"run.demo_subsample_seed={task.demo_subsample_seed}",
            ]
        )
    elif point.demo_budget is not None:
        overrides.extend(
            [
                f"run.n_expert_trajectories={point.demo_budget}",
                f"run.demo_subsample_seed={task.demo_subsample_seed}",
            ]
        )
    overrides.extend(
        [
            f"run.seed={task.seed}",
            f"run.output_dir={task.output_root.relative_to(REPO_ROOT)}",
            f"run.name={task.run_name}",
            f"run.group={task.group}",
            f"wandb.entity={WANDB_ENTITY}",
            f"wandb.project={WANDB_PROJECT}",
            f"wandb.tags=[budget_curve,completion,{task.arm}]",
            f"env.n_envs={N_ENVS}",
            f"eval.n_episodes={EVAL_EPISODES}",
            f"train.kwargs.total_timesteps={TOTAL_TIMESTEPS}",
            f"train.kwargs.timesteps_per_iteration={TIMESTEPS_PER_ITERATION}",
        ]
    )
    return tuple(overrides)


def build_training_command(
    task: Task,
    slot: str,
    base_overrides: tuple[str, ...] | None = None,
) -> list[str]:
    if slot not in SLOTS:
        raise ValueError(f"Unknown CPU slot {slot!r}.")
    if base_overrides is None:
        base_overrides = export_overrides(
            task.point.source_key,
            task.point.pref_budget,
            task.point.demo_budget,
        )
    return [
        "taskset",
        "-c",
        slot,
        str(PYTHON),
        "scripts/train_hybrid_sac.py",
        *base_overrides,
        *task_overrides(task),
    ]


def validate_resolved_configs(
    tasks: tuple[Task, ...],
) -> list[dict[str, Any]]:
    from omegaconf import OmegaConf

    representatives: dict[str, Task] = {}
    for task in tasks:
        representatives.setdefault(task.group, task)

    records = []
    for task in representatives.values():
        command = [
            str(PYTHON),
            "scripts/train_hybrid_sac.py",
            "--cfg",
            "job",
            "--resolve",
            *export_overrides(
                task.point.source_key,
                task.point.pref_budget,
                task.point.demo_budget,
            ),
            *task_overrides(task),
        ]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        cfg = OmegaConf.create(result.stdout)
        point = task.point
        checks = {
            "run.seed": (cfg.run.seed, task.seed),
            "run.name": (cfg.run.name, task.run_name),
            "run.group": (cfg.run.group, task.group),
            "wandb.project": (cfg.wandb.project, WANDB_PROJECT),
            "env.n_envs": (cfg.env.n_envs, N_ENVS),
            "eval.n_episodes": (cfg.eval.n_episodes, EVAL_EPISODES),
            "algo.total_queries": (
                cfg.algo.kwargs.total_queries,
                point.pref_budget,
            ),
            "train.total_queries": (
                cfg.train.kwargs.total_queries,
                point.pref_budget,
            ),
            "algo.initial_queries": (
                cfg.algo.kwargs.initial_queries,
                point.initial_queries,
            ),
            "algo.normalize_agent_reward": (
                cfg.algo.kwargs.normalize_agent_reward,
                point.normalize_agent_reward,
            ),
            "algo.labels_type": (
                cfg.algo.kwargs.labels_type,
                point.labels_type,
            ),
            "algo.loss_type": (cfg.algo.kwargs.loss_type, point.loss_type),
            "algo.query_schedule": (
                cfg.algo.kwargs.query_schedule,
                point.query_schedule,
            ),
            "algo.fragmenter_type": (
                cfg.algo.kwargs.fragmenter_type,
                point.fragmenter_type,
            ),
            "algo.pref_temperature": (
                cfg.algo.kwargs.pref_temperature,
                point.pref_temperature,
            ),
            "algo.demo_weight": (
                cfg.algo.kwargs.demo_weight,
                point.demo_weight,
            ),
            "train.total_timesteps": (
                cfg.train.kwargs.total_timesteps,
                TOTAL_TIMESTEPS,
            ),
            "train.timesteps_per_iteration": (
                cfg.train.kwargs.timesteps_per_iteration,
                TIMESTEPS_PER_ITERATION,
            ),
        }
        if point.demo_transitions is not None:
            # Same precedence as task_overrides: without this branch the check
            # below would demand n_expert_trajectories == demo_budget and find
            # the null we deliberately set, failing a correct config.
            checks.update(
                {
                    "run.n_expert_transitions": (
                        cfg.run.get("n_expert_transitions", None),
                        point.demo_transitions,
                    ),
                    "run.n_expert_trajectories": (
                        cfg.run.n_expert_trajectories,
                        None,
                    ),
                    "run.demo_subsample_seed": (
                        cfg.run.demo_subsample_seed,
                        task.demo_subsample_seed,
                    ),
                }
            )
        elif point.demo_budget is not None:
            checks.update(
                {
                    "run.n_expert_trajectories": (
                        cfg.run.n_expert_trajectories,
                        point.demo_budget,
                    ),
                    "run.demo_subsample_seed": (
                        cfg.run.demo_subsample_seed,
                        task.demo_subsample_seed,
                    ),
                }
            )
        mismatches = {
            key: {"actual": actual, "expected": expected}
            for key, (actual, expected) in checks.items()
            if actual != expected
        }
        if mismatches:
            raise RuntimeError(
                f"Resolved Hydra config mismatch for {task.group}: {mismatches}"
            )
        records.append(
            {
                "group": task.group,
                "source": point.source_key,
                "sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
            }
        )
    return records


def training_environment(task: Task) -> dict[str, str]:
    environment = dict(os.environ)
    # Do not inherit settings that could silently keep these runs local.
    environment.pop("WANDB_DISABLED", None)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            "WANDB_MODE": "online",
            "WANDB_RUN_ID": task.wandb_run_id,
            "WANDB_RESUME": "allow",
        }
    )
    return environment


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def process_record(process: "ProcessInfo") -> dict[str, Any]:
    return {
        "pid": process.pid,
        "ppid": process.ppid,
        "pgid": process.pgid,
        "sid": process.sid,
        "start_time": process.start_time,
        "state": process.state,
        "affinity": sorted(process.affinity),
        "command": process.command,
    }


def transition_member(
    task: Task, process: "ProcessInfo", slot: str
) -> dict[str, Any]:
    return {
        "task_key": task.key,
        "run_name": task.run_name,
        "wandb_run_id": task.wandb_run_id,
        "pid": process.pid,
        "start_time": process.start_time,
        "pgid": process.pgid,
        "sid": process.sid,
        "slot": slot,
        "affinity": sorted(process.affinity),
        "attempts": attempts(task),
    }


def validate_transition_state(
    state: dict[str, Any], tasks: tuple[Task, ...]
) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise RuntimeError("Transition state must be a JSON object.")
    if state.get("version") != TRANSITION_VERSION:
        raise RuntimeError(
            f"Unsupported transition state version {state.get('version')!r}."
        )
    expected_hash = task_matrix_sha256(tasks)
    if state.get("matrix_sha256") != expected_hash:
        raise RuntimeError(
            "Transition state task matrix does not match this orchestrator."
        )
    phase = state.get("phase")
    if phase not in (PHASE_DRAIN, PHASE_SINGLE):
        raise RuntimeError(f"Unknown transition phase {phase!r}.")
    if phase == PHASE_SINGLE and state.get("rollout") not in (
        "canary",
        "full",
    ):
        raise RuntimeError("Single-core transition has an invalid rollout state.")
    cohort = state.get("cohort")
    if not isinstance(cohort, list) or not cohort:
        raise RuntimeError("Transition state has no legacy cohort.")

    task_by_key = {task.key: task for task in tasks}
    seen_keys: set[str] = set()
    seen_names: set[str] = set()
    seen_pids: set[int] = set()
    seen_sessions: set[tuple[int, int]] = set()
    seen_slots: set[str] = set()
    required_member_keys = {
        "task_key",
        "run_name",
        "wandb_run_id",
        "pid",
        "start_time",
        "pgid",
        "sid",
        "slot",
        "affinity",
        "attempts",
    }
    for member in cohort:
        if not isinstance(member, dict) or not required_member_keys.issubset(
            member
        ):
            raise RuntimeError("Malformed transition cohort member.")
        task = task_by_key.get(member["task_key"])
        if (
            task is None
            or member["run_name"] != task.run_name
            or member["wandb_run_id"] != task.wandb_run_id
        ):
            raise RuntimeError(
                f"Transition member does not match task {member.get('task_key')!r}."
            )
        slot = member["slot"]
        if slot not in LEGACY_PAIR_SLOTS:
            raise RuntimeError(f"Unexpected legacy slot {slot!r}.")
        affinity = [int(core) for core in member["affinity"]]
        if frozenset(affinity) != slot_cores(slot):
            raise RuntimeError(
                f"Transition affinity {affinity} does not match slot {slot}."
            )
        integers = ("pid", "start_time", "pgid", "sid", "attempts")
        if any(
            not isinstance(member[key], int) or member[key] < 0
            for key in integers
        ):
            raise RuntimeError(f"Invalid process identity for {task.run_name}.")
        identity_values = (
            member["task_key"],
            member["run_name"],
            member["pid"],
            (member["pgid"], member["sid"]),
            slot,
        )
        sets = (
            seen_keys,
            seen_names,
            seen_pids,
            seen_sessions,
            seen_slots,
        )
        if any(value in seen for value, seen in zip(identity_values, sets)):
            raise RuntimeError("Duplicate identity in transition cohort.")
        for value, seen in zip(identity_values, sets):
            seen.add(value)

    manager = state.get("legacy_manager")
    if not isinstance(manager, dict) or not {
        "pid",
        "start_time",
        "command",
    }.issubset(manager):
        raise RuntimeError("Transition state has no valid legacy manager.")
    return state


def load_transition_state(tasks: tuple[Task, ...]) -> dict[str, Any]:
    try:
        state = json.loads(TRANSITION_STATE_PATH.read_text())
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Missing prepared transition state: {TRANSITION_STATE_PATH}"
        ) from error
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(
            f"Unreadable transition state: {TRANSITION_STATE_PATH}"
        ) from error
    return validate_transition_state(state, tasks)


def capture_transition_state(
    tasks: tuple[Task, ...],
    processes: list["ProcessInfo"],
    legacy_manager_pid: int,
    expected_live: int,
    allow_stopped_manager: bool = False,
) -> dict[str, Any]:
    if TRANSITION_STATE_PATH.exists():
        raise RuntimeError(
            f"Transition state already exists: {TRANSITION_STATE_PATH}"
        )
    manager = process_by_pid(processes, legacy_manager_pid)
    if manager is None and allow_stopped_manager:
        try:
            lock_pid = int(LOCK_PATH.read_text().strip())
        except (OSError, ValueError) as error:
            raise RuntimeError(
                "Cannot verify the stopped legacy manager from its lock file."
            ) from error
        if lock_pid != legacy_manager_pid:
            raise RuntimeError(
                f"Stale lock PID {lock_pid} does not match "
                f"{legacy_manager_pid}."
            )
        unexpected_managers = other_orchestrators(processes)
        if unexpected_managers:
            detail = [
                (process.pid, process.command)
                for process in unexpected_managers
            ]
            raise RuntimeError(
                f"Another manager is active while capturing: {detail}"
            )
        manager_record = {
            "pid": legacy_manager_pid,
            "start_time": None,
            "command": "scripts/schedule_budget_curves_completion.py",
            "state": "already_stopped_before_capture",
            "stale_lock_verified": True,
        }
    elif (
        manager is None
        or "python" not in Path(manager.command.split(maxsplit=1)[0]).name
        or "scripts/schedule_budget_curves_completion.py"
        not in manager.command
    ):
        raise RuntimeError(
            f"PID {legacy_manager_pid} is not the legacy orchestrator."
        )
    else:
        manager_record = process_record(manager)

    task_by_name = {task.run_name: task for task in tasks}
    live = live_training_runs(processes)
    managed_cores = frozenset(range(24, 48))
    unknown = [
        (name, process.pid, sorted(process.affinity))
        for name, process in live.items()
        if process.affinity & managed_cores and name not in task_by_name
    ]
    if unknown:
        raise RuntimeError(f"Unknown training runs on managed cores: {unknown}")

    members = []
    for name, process in live.items():
        task = task_by_name.get(name)
        if task is None:
            continue
        slots = [
            slot
            for slot in LEGACY_PAIR_SLOTS
            if process.affinity == slot_cores(slot)
        ]
        if len(slots) != 1:
            raise RuntimeError(
                f"Legacy task {name} has unexpected affinity "
                f"{sorted(process.affinity)}."
            )
        if process.pid != process.pgid or process.pid != process.sid:
            raise RuntimeError(
                f"Legacy task {name} is not its session/process-group leader."
            )
        members.append(transition_member(task, process, slots[0]))

    members.sort(key=lambda item: LEGACY_PAIR_SLOTS.index(item["slot"]))
    if len(members) != expected_live:
        raise RuntimeError(
            f"Expected {expected_live} live legacy runs, found {len(members)}."
        )
    if len({member["slot"] for member in members}) != len(members):
        raise RuntimeError("Multiple legacy tasks share a CPU pair.")

    state = {
        "version": TRANSITION_VERSION,
        "matrix_sha256": task_matrix_sha256(tasks),
        "phase": PHASE_DRAIN,
        "captured_at": timestamp(),
        "switched_at": None,
        "legacy_manager": manager_record,
        "cohort": members,
        "history": [
            {
                "at": timestamp(),
                "event": "captured_legacy_cohort",
                "count": len(members),
            }
        ],
    }
    validate_transition_state(state, tasks)
    atomic_json(TRANSITION_STATE_PATH, state)
    return state


def switch_to_single_phase(
    state: dict[str, Any], tasks: tuple[Task, ...]
) -> dict[str, Any]:
    validate_transition_state(state, tasks)
    if state["phase"] == PHASE_SINGLE:
        return state
    updated = {
        **state,
        "phase": PHASE_SINGLE,
        "switched_at": timestamp(),
        "rollout": "canary",
        "canary_task_key": None,
        "canary_pid": None,
        "canary_started_at_epoch": None,
        "canary_log_offset": None,
        "history": [
            *state.get("history", []),
            {"at": timestamp(), "event": "switched_to_single_core"},
        ],
    }
    atomic_json(TRANSITION_STATE_PATH, updated)
    return validate_transition_state(updated, tasks)


def update_transition_state(
    state: dict[str, Any],
    tasks: tuple[Task, ...],
    event: str,
    **changes: Any,
) -> dict[str, Any]:
    updated = {
        **state,
        **changes,
        "history": [
            *state.get("history", []),
            {"at": timestamp(), "event": event},
        ],
    }
    atomic_json(TRANSITION_STATE_PATH, updated)
    return validate_transition_state(updated, tasks)


def marker_path(task: Task) -> Path:
    return MARKER_ROOT / f"{task.key}.json"


def read_marker(task: Task) -> dict[str, Any]:
    try:
        return json.loads(marker_path(task).read_text())
    except (OSError, ValueError, TypeError):
        return {}


def write_marker(task: Task, state: str, **extra: Any) -> None:
    previous = read_marker(task)
    atomic_json(
        marker_path(task),
        {
            **previous,
            "key": task.key,
            "state": state,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **extra,
        },
    )


def attempts(task: Task) -> int:
    value = read_marker(task).get("attempts", 0)
    return int(value) if isinstance(value, (int, float, str)) else 0


def valid_final_output(directory: Path) -> bool:
    metrics_path = directory / "final_eval.json"
    agent_path = directory / "agent_final.zip"
    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(metrics, dict) or not zipfile.is_zipfile(agent_path):
        return False
    for key in ("eval/mean_fast_return", "eval/success_rate"):
        value = metrics.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            return False
    return 0.0 <= float(metrics["eval/success_rate"]) <= 1.0


def final_output_dirs(task: Task) -> list[Path]:
    return sorted(
        path.parent
        for path in task.output_root.glob(f"{task.run_name}*/final_eval.json")
        if valid_final_output(path.parent)
    )


def migrate_pref_bernoulli_budget(
    tasks: tuple[Task, ...],
) -> dict[str, Any]:
    """Atomically replace the never-launched 15k point with the 1M point."""
    try:
        state = json.loads(TRANSITION_STATE_PATH.read_text())
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Missing prepared transition state: {TRANSITION_STATE_PATH}"
        ) from error
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(
            f"Unreadable transition state: {TRANSITION_STATE_PATH}"
        ) from error

    new_hash = task_matrix_sha256(tasks)
    if state.get("matrix_sha256") == new_hash:
        return validate_transition_state(state, tasks)

    include_max_hybrid_bernoulli = any(
        task.arm == "hybrid_demo_2_bernoulli" and task.budget == 5_446
        for task in tasks
    )
    legacy_tasks = build_tasks(
        LEGACY_PREF_BERNOULLI_POINTS,
        include_max_hybrid_bernoulli=include_max_hybrid_bernoulli,
    )
    legacy_hash = task_matrix_sha256(legacy_tasks)
    if state.get("matrix_sha256") != legacy_hash:
        raise RuntimeError(
            "Transition state matches neither the approved 15k matrix nor "
            "the replacement 1M matrix."
        )
    validate_transition_state(state, legacy_tasks)

    history = state.get("history")
    if not isinstance(history, list):
        raise RuntimeError("Transition state history is malformed.")

    old_tasks = tuple(
        task
        for task in legacy_tasks
        if task.arm == "pref_bernoulli" and task.budget == 15_000
    )
    new_tasks = tuple(
        task
        for task in tasks
        if task.arm == "pref_bernoulli" and task.budget == 1_000_000
    )
    if len(old_tasks) != len(SEEDS) or len(new_tasks) != len(SEEDS):
        raise RuntimeError("Unexpected pref_bernoulli replacement task count.")

    live_names = set(live_training_runs(iter_processes()))
    artifacts: dict[str, list[str]] = {}
    for task in (*old_tasks, *new_tasks):
        found = []
        if task.run_name in live_names:
            found.append("live_process")
        if marker_path(task).exists():
            found.append(str(marker_path(task).relative_to(REPO_ROOT)))
        if task.log_path.exists():
            found.append(str(task.log_path.relative_to(REPO_ROOT)))
        if task.output_root.exists():
            found.append(str(task.output_root.relative_to(REPO_ROOT)))
        if found:
            artifacts[task.key] = found
    if artifacts:
        raise RuntimeError(
            "Cannot replace a budget point with launch artifacts: "
            f"{artifacts}"
        )

    updated = {
        **state,
        "matrix_sha256": new_hash,
        "history": [
            *history,
            {
                "at": timestamp(),
                "event": (
                    "migrated_pref_bernoulli_budget_15000_to_1000000"
                ),
                "old_budget": 15_000,
                "new_budget": 1_000_000,
                "old_matrix_sha256": legacy_hash,
                "new_matrix_sha256": new_hash,
            },
        ],
    }
    validate_transition_state(updated, tasks)
    atomic_json(TRANSITION_STATE_PATH, updated)
    return validate_transition_state(updated, tasks)


def migrate_hybrid_bernoulli_max_budget(
    tasks: tuple[Task, ...],
) -> dict[str, Any]:
    """Atomically add the approved 2723 preference + 2723 demo point."""
    try:
        state = json.loads(TRANSITION_STATE_PATH.read_text())
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Missing prepared transition state: {TRANSITION_STATE_PATH}"
        ) from error
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(
            f"Unreadable transition state: {TRANSITION_STATE_PATH}"
        ) from error

    new_hash = task_matrix_sha256(tasks)
    if state.get("matrix_sha256") == new_hash:
        return validate_transition_state(state, tasks)

    previous_tasks = build_tasks(include_max_hybrid_bernoulli=False)
    previous_hash = task_matrix_sha256(previous_tasks)
    if state.get("matrix_sha256") != previous_hash:
        raise RuntimeError(
            "Transition state matches neither the previous 57-run matrix "
            "nor the approved 60-run matrix."
        )
    validate_transition_state(state, previous_tasks)

    history = state.get("history")
    if not isinstance(history, list):
        raise RuntimeError("Transition state history is malformed.")

    added_tasks = tuple(
        task
        for task in tasks
        if task.arm == "hybrid_demo_2_bernoulli"
        and task.point.pref_budget == 2_723
        and task.point.demo_budget == 2_723
    )
    if len(added_tasks) != len(SEEDS):
        raise RuntimeError("Unexpected max-budget hybrid task count.")

    live_names = set(live_training_runs(iter_processes()))
    artifacts: dict[str, list[str]] = {}
    for task in added_tasks:
        found = []
        if task.run_name in live_names:
            found.append("live_process")
        if marker_path(task).exists():
            found.append(str(marker_path(task).relative_to(REPO_ROOT)))
        if task.log_path.exists():
            found.append(str(task.log_path.relative_to(REPO_ROOT)))
        if task.output_root.exists():
            found.append(str(task.output_root.relative_to(REPO_ROOT)))
        if found:
            artifacts[task.key] = found
    if artifacts:
        raise RuntimeError(
            "Cannot add max-budget hybrid tasks with launch artifacts: "
            f"{artifacts}"
        )

    updated = {
        **state,
        "matrix_sha256": new_hash,
        "history": [
            *history,
            {
                "at": timestamp(),
                "event": (
                    "added_hybrid_demo_2_bernoulli_pref2723_demo2723"
                ),
                "pref_budget": 2_723,
                "demo_budget": 2_723,
                "old_matrix_sha256": previous_hash,
                "new_matrix_sha256": new_hash,
            },
        ],
    }
    validate_transition_state(updated, tasks)
    atomic_json(TRANSITION_STATE_PATH, updated)
    return validate_transition_state(updated, tasks)


def task_done(task: Task, update_marker: bool = True) -> bool:
    if read_marker(task).get("state") != "done":
        return False
    outputs = final_output_dirs(task)
    if not outputs:
        return False
    if update_marker:
        write_marker(
            task,
            "done",
            attempts=attempts(task),
            output_dirs=[str(path.relative_to(REPO_ROOT)) for path in outputs],
        )
    return True


def mark_done_after_exit(
    task: Task, return_code: int | None, slot: str
) -> bool:
    outputs = final_output_dirs(task)
    # A process launched by this scheduler must exit zero after wandb.finish().
    # For an adopted non-child process Linux cannot provide the exit code;
    # disappearance plus validated outputs is the recovery-path signal.
    if not outputs or return_code not in (0, None):
        return False
    write_marker(
        task,
        "done",
        attempts=attempts(task),
        slot=slot,
        return_code=return_code,
        exit_code_available=return_code is not None,
        output_dirs=[str(path.relative_to(REPO_ROOT)) for path in outputs],
    )
    return True


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    pgid: int
    sid: int
    start_time: int
    state: str
    affinity: frozenset[int]
    command: str


def iter_processes() -> list[ProcessInfo]:
    processes = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        pid = int(item.name)
        try:
            stat = (item / "stat").read_text()
            close_paren = stat.rfind(")")
            if close_paren < 0:
                continue
            fields = stat[close_paren + 2 :].split()
            state = fields[0]
            ppid = int(fields[1])
            pgid = int(fields[2])
            sid = int(fields[3])
            start_time = int(fields[19])
            command = (
                (item / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
                .strip()
            )
            if state == "Z" or not command:
                continue
            affinity = frozenset(os.sched_getaffinity(pid))
        except (OSError, PermissionError, IndexError, ValueError):
            continue
        processes.append(
            ProcessInfo(
                pid=pid,
                ppid=ppid,
                pgid=pgid,
                sid=sid,
                start_time=start_time,
                state=state,
                affinity=affinity,
                command=command,
            )
        )
    return processes


def live_training_runs(processes: list[ProcessInfo]) -> dict[str, ProcessInfo]:
    result: dict[str, ProcessInfo] = {}
    for process in processes:
        if "scripts/train_hybrid_sac.py" not in process.command:
            continue
        # Hydra's config-resolution subprocess carries the same run.name but
        # exits without training.  It must never be adopted as a live run.
        if re.search(r"(?:^|\s)--cfg(?:\s|=)", process.command):
            continue
        match = re.search(r"(?:^|\s)run\.name=([^\s]+)", process.command)
        if match:
            run_name = match.group(1)
            previous = result.get(run_name)
            if previous is not None and previous.pid != process.pid:
                raise RuntimeError(
                    f"Duplicate live run.name={run_name!r}: "
                    f"pids {previous.pid} and {process.pid}."
                )
            result[run_name] = process
    return result


def process_by_pid(
    processes: list[ProcessInfo], pid: int
) -> ProcessInfo | None:
    return next((process for process in processes if process.pid == pid), None)


def process_identity_matches(
    process: ProcessInfo,
    *,
    pid: int,
    start_time: int,
    pgid: int,
    sid: int,
    run_name: str,
) -> bool:
    return (
        process.pid == pid
        and process.start_time == start_time
        and process.pgid == pgid
        and process.sid == sid
        and f"run.name={run_name}" in process.command
        and "scripts/train_hybrid_sac.py" in process.command
    )


def process_group_alive(
    processes: list[ProcessInfo], pgid: int, sid: int
) -> bool:
    return any(
        process.pgid == pgid and process.sid == sid for process in processes
    )


def owned_run_processes(
    processes: list[ProcessInfo],
    *,
    root_pid: int,
    pgid: int,
    sid: int,
    slot: str | None = None,
) -> list[ProcessInfo]:
    """Return the complete observable process tree owned by one run.

    Training workers inherit the run process group.  W&B deliberately starts
    helpers in separate sessions, but records the trainer PID (and its own
    parent PID) in their command lines, so those helpers remain attributable
    even if Linux reparents them during shutdown.
    """
    owned_pids = {
        process.pid
        for process in processes
        if process.pgid == pgid and process.sid == sid
    }
    root_pattern = re.compile(
        rf"(?:^|\s)--pid\s+{re.escape(str(root_pid))}(?:\s|$)"
    )
    changed = True
    while changed:
        changed = False
        for process in processes:
            if process.pid in owned_pids:
                continue
            parent_match = re.search(
                r"(?:^|\s)--parent-pid\s+(\d+)(?:\s|$)",
                process.command,
            )
            command_parent = (
                int(parent_match.group(1)) if parent_match else None
            )
            if (
                process.ppid in owned_pids
                or root_pattern.search(process.command)
                or command_parent in owned_pids
            ):
                owned_pids.add(process.pid)
                changed = True
    if slot is not None:
        expected_affinity = slot_cores(slot)
        for process in processes:
            executable = Path(process.command.split(maxsplit=1)[0]).name
            if (
                process.affinity == expected_affinity
                and executable == "gpu_stats"
                and re.search(
                    r"(?:^|\s)--parent-pid\s+\d+(?:\s|$)",
                    process.command,
                )
            ):
                # gpu_stats may briefly outlive wandb-core.  At that point
                # Linux can reparent it and the old parent is absent from
                # /proc; exact slot affinity keeps the attribution unique.
                owned_pids.add(process.pid)
    return [process for process in processes if process.pid in owned_pids]


def run_process_tree_alive(
    processes: list[ProcessInfo],
    *,
    root_pid: int,
    pgid: int,
    sid: int,
    slot: str | None = None,
) -> bool:
    return bool(
        owned_run_processes(
            processes,
            root_pid=root_pid,
            pgid=pgid,
            sid=sid,
            slot=slot,
        )
    )


def validated_marker_owned_pids(
    task: Task,
    marker: dict[str, Any],
    processes: list[ProcessInfo],
) -> set[int]:
    if marker.get("state") != "running" or marker.get("slot") not in SINGLE_SLOTS:
        return set()
    pid = marker.get("pid")
    start_time = marker.get("start_time")
    pgid = marker.get("pgid")
    sid = marker.get("sid")
    if not all(isinstance(value, int) for value in (pid, pgid, sid)):
        return set()
    owned = owned_run_processes(
        processes,
        root_pid=pid,
        pgid=pgid,
        sid=sid,
        slot=str(marker["slot"]),
    )
    if not owned:
        return set()
    main = process_by_pid(processes, pid)
    if main is not None:
        if not isinstance(start_time, int) or not process_identity_matches(
            main,
            pid=pid,
            start_time=start_time,
            pgid=pgid,
            sid=sid,
            run_name=task.run_name,
        ):
            raise RuntimeError(
                f"PID reuse or marker identity mismatch for {task.run_name}."
            )
        if main.affinity != slot_cores(str(marker["slot"])):
            raise RuntimeError(
                f"Marker affinity changed for {task.run_name}: "
                f"{sorted(main.affinity)}."
            )
    return {process.pid for process in owned}


def constrained_slot_processes(
    processes: list[ProcessInfo], slot: str
) -> list[ProcessInfo]:
    cores = slot_cores(slot)
    return [
        process
        for process in processes
        if len(process.affinity) <= 8 and process.affinity & cores
    ]


def other_orchestrators(processes: list[ProcessInfo]) -> list[ProcessInfo]:
    result = []
    for process in processes:
        if process.pid == os.getpid():
            continue
        executable = process.command.split(maxsplit=1)[0]
        if "python" not in Path(executable).name:
            continue
        if re.search(
            r"(?:^|\s)--(?:status|preflight|dry-run)(?:\s|$)",
            process.command,
        ):
            continue
        if "schedule_" in process.command or "orchestrator" in process.command:
            result.append(process)
    return result


def _core_times() -> tuple[dict[int, int], dict[int, int]]:
    busy: dict[int, int] = {}
    total: dict[int, int] = {}
    for line in Path("/proc/stat").read_text().splitlines():
        match = re.match(r"cpu(\d+)\s", line)
        if not match:
            continue
        values = [int(value) for value in line.split()[1:]]
        core = int(match.group(1))
        total[core] = sum(values)
        busy[core] = total[core] - values[3] - values[4]
    return busy, total


def slot_cpu_busy() -> dict[str, tuple[float, float]]:
    busy0, total0 = _core_times()
    time.sleep(CPU_SAMPLE_SECONDS)
    busy1, total1 = _core_times()
    result = {}
    for slot in SLOTS:
        values = []
        for core in slot_cores(slot):
            delta_total = max(total1[core] - total0[core], 1)
            values.append(100 * (busy1[core] - busy0[core]) / delta_total)
        result[slot] = (sum(values) / len(values), max(values))
    return result


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def git_snapshot() -> dict[str, Any]:
    return {
        "root_head": command_output(["git", "rev-parse", "HEAD"]),
        "submodule_head": command_output(
            ["git", "-C", "human-feedback-rl", "rev-parse", "HEAD"]
        ),
        "root_status": command_output(["git", "status", "--short"]),
        "submodule_status": command_output(
            ["git", "-C", "human-feedback-rl", "status", "--short"]
        ),
    }


def has_wandb_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        credentials = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError):
        return False
    return credentials is not None


def preflight(
    tasks: tuple[Task, ...],
    require_free_cores: bool = True,
    transition_state: dict[str, Any] | None = None,
    allowed_orchestrator_pid: int | None = None,
    defer_resolved_configs: bool = False,
) -> dict[str, Any]:
    matrix = validate_task_matrix(tasks)
    required_paths = (PYTHON, TRAIN_SCRIPT, EXPORT_SCRIPT, JOURNAL, DATASET)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required paths: {missing}")
    if shutil.which("taskset") is None:
        raise RuntimeError("taskset is not available.")
    if not has_wandb_credentials():
        raise RuntimeError("No W&B credentials found in WANDB_API_KEY or ~/.netrc.")
    allowed = frozenset(os.sched_getaffinity(0))
    required_cores = frozenset(range(24, 48))
    if not required_cores.issubset(allowed):
        raise RuntimeError(
            f"Current process cannot use all required cores: "
            f"missing={sorted(required_cores - allowed)}"
        )
    free_bytes = shutil.disk_usage(REPO_ROOT).free
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"Only {free_bytes / 1024**3:.1f} GiB free.")

    processes = iter_processes()
    coordinators = [
        process
        for process in other_orchestrators(processes)
        if process.pid != allowed_orchestrator_pid
    ]
    if coordinators:
        detail = [(process.pid, process.command) for process in coordinators]
        raise RuntimeError(f"Other orchestrators are active: {detail}")
    task_names = {task.run_name for task in tasks}
    live = live_training_runs(processes)
    allowed_pids: set[int] = set()
    owned_slots: set[str] = set()
    if transition_state is not None:
        validate_transition_state(transition_state, tasks)
        if transition_state["phase"] == PHASE_DRAIN:
            member_by_name = {
                member["run_name"]: member
                for member in transition_state["cohort"]
            }
            for member in transition_state["cohort"]:
                allowed_pids.update(
                    process.pid
                    for process in owned_run_processes(
                        processes,
                        root_pid=member["pid"],
                        pgid=member["pgid"],
                        sid=member["sid"],
                        slot=member["slot"],
                    )
                )
            unexpected_names = sorted(
                name
                for name, process in live.items()
                if process.affinity & required_cores
                and name not in member_by_name
            )
            if unexpected_names:
                raise RuntimeError(
                    f"Unexpected live tasks during legacy drain: "
                    f"{unexpected_names}"
                )
            for name, process in live.items():
                if name not in member_by_name:
                    continue
                member = member_by_name[name]
                if not process_identity_matches(
                    process,
                    pid=member["pid"],
                    start_time=member["start_time"],
                    pgid=member["pgid"],
                    sid=member["sid"],
                    run_name=member["run_name"],
                ):
                    raise RuntimeError(
                        f"Live legacy identity changed for {name}."
                    )
                if process.affinity != slot_cores(member["slot"]):
                    raise RuntimeError(
                        f"Live legacy affinity changed for {name}: "
                        f"{sorted(process.affinity)}."
                    )
        else:
            for task in tasks:
                allowed_pids.update(
                    validated_marker_owned_pids(
                        task, read_marker(task), processes
                    )
                )
            for name, process in live.items():
                if name not in task_names:
                    continue
                slots = [
                    slot
                    for slot in SINGLE_SLOTS
                    if process.affinity == slot_cores(slot)
                ]
                if len(slots) != 1:
                    raise RuntimeError(
                        f"Live singleton {name} has unexpected affinity "
                        f"{sorted(process.affinity)}."
                    )
                slot = slots[0]
                if slot in owned_slots:
                    raise RuntimeError(
                        f"Multiple live tasks share singleton slot {slot}."
                    )
                owned_slots.add(slot)
                allowed_pids.update(
                    item.pid
                    for item in owned_run_processes(
                        processes,
                        root_pid=process.pid,
                        pgid=process.pgid,
                        sid=process.sid,
                        slot=slot,
                    )
                )

    occupied = {}
    for slot in SINGLE_SLOTS:
        conflicts = [
            process
            for process in constrained_slot_processes(processes, slot)
            if process.pid not in allowed_pids
        ]
        if conflicts:
            occupied[slot] = conflicts
    if require_free_cores and occupied:
        detail = {
            slot: [(process.pid, process.command) for process in processes]
            for slot, processes in occupied.items()
        }
        raise RuntimeError(f"Required CPU slots are occupied: {detail}")

    sources = validate_optuna_sources()
    query_schedules = validate_runtime_query_schedules(tasks)
    resolved_configs = (
        []
        if defer_resolved_configs
        else validate_resolved_configs(tasks)
    )
    # Materialize every unique exported config now; a failure aborts before
    # state directories or training processes are created.
    for point in build_points():
        export_overrides(point.source_key, point.pref_budget, point.demo_budget)

    return {
        **matrix,
        "project": WANDB_PROJECT,
        "slots": list(SINGLE_SLOTS),
        "legacy_slots": list(LEGACY_PAIR_SLOTS),
        "transition_phase": (
            transition_state["phase"] if transition_state is not None else None
        ),
        "free_gib": round(free_bytes / 1024**3, 1),
        "sources": sources,
        "query_schedules": query_schedules,
        "resolved_configs": resolved_configs,
        "resolved_configs_deferred": defer_resolved_configs,
        "git": git_snapshot(),
    }


def command_records(tasks: tuple[Task, ...]) -> list[dict[str, Any]]:
    records = []
    for index, task in enumerate(tasks):
        slot = SLOTS[index % len(SLOTS)]
        command = build_training_command(task, slot)
        records.append(
            {
                **task.manifest_record(),
                "planned_slot": slot,
                "command": shlex.join(command),
            }
        )
    return records


def write_manifest(
    tasks: tuple[Task, ...],
    preflight_result: dict[str, Any],
    transition_state: dict[str, Any],
) -> None:
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "priority": "hybrid points first; three seeds remain adjacent per point",
        "runtime": {
            "total_timesteps": TOTAL_TIMESTEPS,
            "timesteps_per_iteration": TIMESTEPS_PER_ITERATION,
            "n_iterations": N_ITERATIONS,
            "n_envs": N_ENVS,
            "eval_episodes": EVAL_EPISODES,
            "max_attempts": MAX_ATTEMPTS,
        },
        "slots": list(SINGLE_SLOTS),
        "legacy_slots": list(LEGACY_PAIR_SLOTS),
        "transition": transition_state,
        "original_manifest": (
            {
                "path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
                "sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
            }
            if MANIFEST_PATH.exists()
            else None
        ),
        "preflight": preflight_result,
        "sources": {
            key: {
                **asdict(source),
                "study_name": source.study_name,
            }
            for key, source in CONFIG_SOURCES.items()
        },
        "tasks": command_records(tasks),
    }
    atomic_json(TRANSITION_MANIFEST_PATH, payload)


@dataclass
class Running:
    task: Task
    pid: int
    slot: str
    process: subprocess.Popen[str] | None = None
    start_time: int | None = None
    pgid: int | None = None
    sid: int | None = None
    adopted: bool = False


class Scheduler:
    def __init__(
        self,
        tasks: tuple[Task, ...],
        max_parallel: int,
        max_attempts: int,
        loop_seconds: int,
        transition_state: dict[str, Any],
    ):
        self.tasks = tasks
        self.task_by_key = {task.key: task for task in tasks}
        self.task_by_name = {task.run_name: task for task in tasks}
        self.max_parallel = max_parallel
        self.max_attempts = max_attempts
        self.loop_seconds = loop_seconds
        self.transition_state = validate_transition_state(
            transition_state, tasks
        )
        self.running: dict[str, Running] = {}
        self.idle_scans = {slot: 0 for slot in SINGLE_SLOTS}
        self.last_drain_log = 0.0

    @property
    def phase(self) -> str:
        return str(self.transition_state["phase"])

    @property
    def cohort(self) -> list[dict[str, Any]]:
        return list(self.transition_state["cohort"])

    def validate_live_layout(self, processes: list[ProcessInfo]) -> None:
        coordinators = other_orchestrators(processes)
        if coordinators:
            detail = [(process.pid, process.command) for process in coordinators]
            raise RuntimeError(f"Another orchestrator appeared: {detail}")

        live = live_training_runs(processes)
        managed_cores = frozenset(range(24, 48))
        allowed_pids: set[int] = set()
        if self.phase == PHASE_DRAIN:
            members = {
                member["run_name"]: member for member in self.cohort
            }
            unexpected = sorted(
                name
                for name, process in live.items()
                if process.affinity & managed_cores and name not in members
            )
            if unexpected:
                raise RuntimeError(
                    f"Unexpected live tasks during drain: {unexpected}"
                )
            for member in self.cohort:
                allowed_pids.update(
                    process.pid
                    for process in owned_run_processes(
                        processes,
                        root_pid=member["pid"],
                        pgid=member["pgid"],
                        sid=member["sid"],
                        slot=member["slot"],
                    )
                )
            for name, process in live.items():
                if name not in members:
                    continue
                member = members[name]
                if not process_identity_matches(
                    process,
                    pid=member["pid"],
                    start_time=member["start_time"],
                    pgid=member["pgid"],
                    sid=member["sid"],
                    run_name=member["run_name"],
                ):
                    raise RuntimeError(
                        f"Legacy process identity changed for {name}."
                    )
                if process.affinity != slot_cores(member["slot"]):
                    raise RuntimeError(
                        f"Legacy affinity changed for {name}: "
                        f"{sorted(process.affinity)}."
                    )
        else:
            for task in self.tasks:
                allowed_pids.update(
                    validated_marker_owned_pids(
                        task, read_marker(task), processes
                    )
                )
            used_slots: set[str] = set()
            for name, process in live.items():
                if name not in self.task_by_name:
                    if process.affinity & managed_cores:
                        raise RuntimeError(
                            f"Unknown live training run {name!r} on managed cores."
                        )
                    continue
                matching = [
                    slot
                    for slot in SINGLE_SLOTS
                    if process.affinity == slot_cores(slot)
                ]
                if len(matching) != 1:
                    raise RuntimeError(
                        f"Live task {name} has non-singleton affinity "
                        f"{sorted(process.affinity)}."
                    )
                if matching[0] in used_slots:
                    raise RuntimeError(
                        f"Multiple live tasks share slot {matching[0]}."
                    )
                used_slots.add(matching[0])
                allowed_pids.update(
                    item.pid
                    for item in owned_run_processes(
                        processes,
                        root_pid=process.pid,
                        pgid=process.pgid,
                        sid=process.sid,
                        slot=matching[0],
                    )
                )

        unexpected_processes = [
            process
            for process in processes
            if len(process.affinity) <= 8
            and process.affinity & managed_cores
            and process.pid not in allowed_pids
        ]
        if unexpected_processes and self.phase == PHASE_DRAIN:
            detail = [
                (
                    process.pid,
                    process.pgid,
                    process.sid,
                    sorted(process.affinity),
                    process.command,
                )
                for process in unexpected_processes
            ]
            raise RuntimeError(
                f"Unexpected processes on managed cores: {detail}"
            )

    def adopt_legacy(self, processes: list[ProcessInfo]) -> None:
        for member in self.cohort:
            task = self.task_by_key[member["task_key"]]
            if task.key in self.running:
                continue
            if not run_process_tree_alive(
                processes,
                root_pid=member["pid"],
                pgid=member["pgid"],
                sid=member["sid"],
                slot=member["slot"],
            ):
                self.settle_exit(
                    task,
                    return_code=None,
                    slot=member["slot"],
                    pid=member["pid"],
                    minimum_attempts=max(member["attempts"], 1),
                )
                continue
            main = process_by_pid(processes, member["pid"])
            if main is not None and not process_identity_matches(
                main,
                pid=member["pid"],
                start_time=member["start_time"],
                pgid=member["pgid"],
                sid=member["sid"],
                run_name=member["run_name"],
            ):
                raise RuntimeError(
                    f"PID reuse or identity mismatch for {member['run_name']}."
                )
            self.running[task.key] = Running(
                task=task,
                pid=member["pid"],
                slot=member["slot"],
                start_time=member["start_time"],
                pgid=member["pgid"],
                sid=member["sid"],
                adopted=True,
            )
            count = max(attempts(task), member["attempts"], 1)
            write_marker(
                task,
                "running",
                attempts=count,
                pid=member["pid"],
                start_time=member["start_time"],
                pgid=member["pgid"],
                sid=member["sid"],
                slot=member["slot"],
                adopted=True,
                wandb_run_id=task.wandb_run_id,
            )
            log(
                f"ADOPT legacy {task.run_name} pid={member['pid']} "
                f"slot={member['slot']}"
            )

    def adopt_singletons(self, processes: list[ProcessInfo]) -> None:
        live = live_training_runs(processes)
        for task in self.tasks:
            process = live.get(task.run_name)
            if process is None or task.key in self.running:
                continue
            matching = [
                slot
                for slot in SINGLE_SLOTS
                if process.affinity == slot_cores(slot)
            ]
            if len(matching) != 1:
                raise RuntimeError(
                    f"Live task {task.run_name} has unexpected affinity "
                    f"{sorted(process.affinity)}."
                )
            slot = matching[0]
            if any(item.slot == slot for item in self.running.values()):
                raise RuntimeError(f"Multiple adopted tasks on slot {slot}.")
            if process.pid != process.pgid or process.pid != process.sid:
                raise RuntimeError(
                    f"Live singleton {task.run_name} is not a session leader."
                )
            self.running[task.key] = Running(
                task=task,
                pid=process.pid,
                slot=slot,
                start_time=process.start_time,
                pgid=process.pgid,
                sid=process.sid,
                adopted=True,
            )
            write_marker(
                task,
                "running",
                attempts=max(attempts(task), 1),
                pid=process.pid,
                start_time=process.start_time,
                pgid=process.pgid,
                sid=process.sid,
                slot=slot,
                adopted=True,
                wandb_run_id=task.wandb_run_id,
            )
            log(
                f"ADOPT singleton {task.run_name} "
                f"pid={process.pid} slot={slot}"
            )

        # If the main process exited but workers remain, the persisted marker
        # still identifies the session that must keep the slot reserved.
        for task in self.tasks:
            if task.key in self.running:
                continue
            marker = read_marker(task)
            if marker.get("state") != "running":
                continue
            slot = marker.get("slot")
            pid = marker.get("pid")
            pgid = marker.get("pgid")
            sid = marker.get("sid")
            if (
                slot not in SINGLE_SLOTS
                or not isinstance(pid, int)
                or not isinstance(pgid, int)
                or not isinstance(sid, int)
                or not run_process_tree_alive(
                    processes,
                    root_pid=pid,
                    pgid=pgid,
                    sid=sid,
                    slot=slot,
                )
            ):
                continue
            if any(item.slot == slot for item in self.running.values()):
                raise RuntimeError(f"Multiple process groups share slot {slot}.")
            self.running[task.key] = Running(
                task=task,
                pid=pid,
                slot=slot,
                start_time=marker.get("start_time"),
                pgid=pgid,
                sid=sid,
                adopted=True,
            )
            log(
                f"ADOPT singleton session {task.run_name} "
                f"pgid={pgid} slot={slot}"
            )

    def settle_exit(
        self,
        task: Task,
        return_code: int | None,
        slot: str,
        pid: int,
        minimum_attempts: int = 1,
    ) -> None:
        if task_done(task):
            return
        count = max(attempts(task), minimum_attempts)
        marker = read_marker(task)
        if attempts(task) != count:
            write_marker(
                task,
                marker.get("state", "running"),
                attempts=count,
            )
            marker = read_marker(task)
        if mark_done_after_exit(task, return_code=return_code, slot=slot):
            log(f"DONE {task.run_name} slot={slot} attempt={count}")
        elif count < self.max_attempts:
            if marker.get("state") != "retry_pending":
                write_marker(
                    task,
                    "retry_pending",
                    attempts=count,
                    pid=pid,
                    slot=slot,
                    return_code=return_code,
                )
                log(
                    f"RETRY {task.run_name}: output finale assente "
                    f"dopo attempt={count} return_code={return_code}"
                )
        elif marker.get("state") != "failed":
            write_marker(
                task,
                "failed",
                attempts=count,
                pid=pid,
                slot=slot,
                return_code=return_code,
            )
            log(
                f"FAIL {task.run_name}: tentativi esauriti "
                f"return_code={return_code}"
            )

    def reap(self, processes: list[ProcessInfo]) -> None:
        for key, running in list(self.running.items()):
            if running.process is not None:
                return_code = running.process.poll()
            else:
                return_code = None
            main_alive = (
                running.process is not None and return_code is None
            )
            tree_alive = (
                running.pgid is not None
                and running.sid is not None
                and run_process_tree_alive(
                    processes,
                    root_pid=running.pid,
                    pgid=running.pgid,
                    sid=running.sid,
                    slot=running.slot,
                )
            )
            alive = main_alive or tree_alive
            if alive:
                continue
            del self.running[key]
            self.settle_exit(
                running.task,
                return_code=return_code,
                slot=running.slot,
                pid=running.pid,
            )

    def reconcile_stale_markers(
        self, live_names: set[str], processes: list[ProcessInfo]
    ) -> None:
        for task in self.tasks:
            if (
                task.key in self.running
                or task.run_name in live_names
                or read_marker(task).get("state") != "running"
            ):
                continue
            marker = read_marker(task)
            pid = marker.get("pid")
            pgid = marker.get("pgid")
            sid = marker.get("sid")
            if (
                isinstance(pid, int)
                and isinstance(pgid, int)
                and isinstance(sid, int)
                and run_process_tree_alive(
                    processes,
                    root_pid=pid,
                    pgid=pgid,
                    sid=sid,
                    slot=str(marker.get("slot")),
                )
            ):
                raise RuntimeError(
                    f"Unadopted live process group for {task.run_name}."
                )
            self.settle_exit(
                task,
                return_code=None,
                slot=str(marker.get("slot", "unknown")),
                pid=int(marker.get("pid", 0)),
            )

    def pending(self, live_names: set[str]) -> list[Task]:
        result = []
        for task in self.tasks:
            if task.key in self.running or task.run_name in live_names:
                continue
            if task_done(task):
                continue
            if attempts(task) >= self.max_attempts:
                continue
            result.append(task)
        return result

    def launch(
        self, task: Task, slot: str, processes: list[ProcessInfo]
    ) -> bool:
        conflicts = constrained_slot_processes(processes, slot)
        if conflicts:
            self.idle_scans[slot] = 0
            return False

        count = attempts(task) + 1
        command = build_training_command(task, slot)
        task.output_root.mkdir(parents=True, exist_ok=True)
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_offset = (
            task.log_path.stat().st_size if task.log_path.exists() else 0
        )
        with task.log_path.open("a") as stream:
            stream.write(
                f"\n=== {time.strftime('%F %T')} attempt={count} "
                f"slot={slot} wandb_id={task.wandb_run_id}\n"
            )
            stream.write(shlex.join(command) + "\n")
            stream.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=training_environment(task),
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        launched_info = process_by_pid(iter_processes(), process.pid)
        start_time = (
            launched_info.start_time if launched_info is not None else None
        )
        pgid = launched_info.pgid if launched_info is not None else process.pid
        sid = launched_info.sid if launched_info is not None else process.pid
        self.running[task.key] = Running(
            task=task,
            pid=process.pid,
            slot=slot,
            process=process,
            start_time=start_time,
            pgid=pgid,
            sid=sid,
            adopted=False,
        )
        write_marker(
            task,
            "running",
            attempts=count,
            pid=process.pid,
            start_time=start_time,
            pgid=pgid,
            sid=sid,
            slot=slot,
            adopted=False,
            wandb_run_id=task.wandb_run_id,
            command=shlex.join(command),
        )
        if self.transition_state.get("rollout") == "canary":
            self.transition_state = update_transition_state(
                self.transition_state,
                self.tasks,
                "canary_started",
                canary_task_key=task.key,
                canary_pid=process.pid,
                canary_started_at_epoch=time.time(),
                canary_log_offset=log_offset,
            )
        self.idle_scans[slot] = 0
        log(
            f"START {task.run_name} pid={process.pid} slot={slot} "
            f"attempt={count}"
        )
        return True

    def failed(self) -> list[Task]:
        return [
            task
            for task in self.tasks
            if not task_done(task, update_marker=False)
            and attempts(task) >= self.max_attempts
            and task.key not in self.running
        ]

    def drain_step(self, processes: list[ProcessInfo]) -> bool:
        self.validate_live_layout(processes)
        self.adopt_legacy(processes)
        self.reap(processes)
        latest = iter_processes()
        self.validate_live_layout(latest)
        self.reap(latest)
        self.adopt_legacy(latest)
        live_groups = sum(
            run_process_tree_alive(
                latest,
                root_pid=member["pid"],
                pgid=member["pgid"],
                sid=member["sid"],
                slot=member["slot"],
            )
            for member in self.cohort
        )
        now = time.time()
        if now - self.last_drain_log >= 60:
            log(
                f"DRAIN legacy_live={live_groups}/"
                f"{len(self.cohort)} launch_enabled=false"
            )
            self.last_drain_log = now
        if live_groups:
            return False
        if self.running:
            raise RuntimeError(
                "Legacy process groups are gone but adopted runs remain."
            )
        self.transition_state = switch_to_single_phase(
            self.transition_state, self.tasks
        )
        self.idle_scans = {slot: 0 for slot in SINGLE_SLOTS}
        log(
            "TRANSITION phase=single_core rollout=canary "
            "legacy_live=0 launch_enabled=true"
        )
        return True

    def canary_log_ready(self, task: Task, offset: int) -> bool:
        try:
            with task.log_path.open("rb") as stream:
                stream.seek(offset)
                content = stream.read().decode(errors="replace")
        except OSError:
            return False
        return (
            "wandb: Tracking run with wandb version" in content
            and re.search(r"(?:^|\n)Iteration 0/99(?:\n|$)", content)
            is not None
        )

    def maybe_promote_canary(self) -> None:
        if self.transition_state.get("rollout") != "canary":
            return
        key = self.transition_state.get("canary_task_key")
        started = self.transition_state.get("canary_started_at_epoch")
        log_offset = self.transition_state.get("canary_log_offset")
        if (
            key is None
            or not isinstance(started, (int, float))
            or not isinstance(log_offset, int)
        ):
            return
        task = self.task_by_key.get(str(key))
        healthy = task is not None and (
            task_done(task, update_marker=False)
            or (
                key in self.running
                and self.canary_log_ready(task, log_offset)
            )
        )
        if healthy and time.time() - float(started) >= CANARY_SECONDS:
            self.transition_state = update_transition_state(
                self.transition_state,
                self.tasks,
                "canary_passed",
                rollout="full",
            )
            log("CANARY passed; ramping to 24 singleton slots.")

    def single_step(self, processes: list[ProcessInfo]) -> bool:
        self.validate_live_layout(processes)
        self.adopt_singletons(processes)
        self.reap(processes)
        latest = iter_processes()
        self.validate_live_layout(latest)
        self.adopt_singletons(latest)
        live_names = set(live_training_runs(latest))
        self.reconcile_stale_markers(live_names, latest)
        pending = self.pending(live_names)

        done_count = sum(
            task_done(task, update_marker=False) for task in self.tasks
        )
        failed = self.failed()
        if done_count == len(self.tasks) and not self.running:
            log(f"COMPLETE all {done_count} tasks finished.")
            return True
        if not pending and not self.running:
            if failed:
                names = ", ".join(task.run_name for task in failed)
                raise RuntimeError(f"No runnable tasks; failed: {names}")
            raise RuntimeError(
                "No runnable tasks and no live tasks, but completion is partial."
            )

        self.maybe_promote_canary()
        busy = slot_cpu_busy()
        latest = iter_processes()
        used_slots = {running.slot for running in self.running.values()}
        for slot in SINGLE_SLOTS:
            if slot in used_slots:
                self.idle_scans[slot] = 0
                continue
            mean_busy, max_busy = busy[slot]
            idle = (
                not constrained_slot_processes(latest, slot)
                and mean_busy < CPU_MEAN_BUSY
                and max_busy < CPU_MAX_BUSY
            )
            self.idle_scans[slot] = (
                self.idle_scans[slot] + 1 if idle else 0
            )

        rollout_limit = (
            1
            if self.transition_state.get("rollout") == "canary"
            else self.max_parallel
        )
        capacity = rollout_limit - len(self.running)
        for slot in SINGLE_SLOTS:
            if (
                STOP_REQUESTED
                or capacity <= 0
                or not pending
                or slot in used_slots
                or self.idle_scans[slot] < IDLE_SCANS_REQUIRED
            ):
                continue
            newest = iter_processes()
            self.validate_live_layout(newest)
            if self.launch(pending[0], slot, newest):
                pending.pop(0)
                used_slots.add(slot)
                capacity -= 1
        return False

    def loop(self) -> None:
        global STOP_REQUESTED
        log(
            f"START scheduler pid={os.getpid()} tasks={len(self.tasks)} "
            f"max_parallel={self.max_parallel} phase={self.phase} "
            f"project={WANDB_PROJECT}"
        )
        while not STOP_REQUESTED:
            processes = iter_processes()
            if self.phase == PHASE_DRAIN:
                self.drain_step(processes)
            elif self.single_step(processes):
                return

            if not STOP_REQUESTED:
                time.sleep(self.loop_seconds)
        log("STOP requested; child training processes were left running.")


def status_payload(tasks: tuple[Task, ...]) -> dict[str, Any]:
    processes = iter_processes()
    live = live_training_runs(processes)
    relevant = {
        task.run_name: live[task.run_name]
        for task in tasks
        if task.run_name in live
    }
    transition: dict[str, Any] | None = None
    transition_error: str | None = None
    if TRANSITION_STATE_PATH.exists():
        try:
            transition = load_transition_state(tasks)
        except RuntimeError as error:
            transition_error = str(error)
    cohort_live = (
        sum(
            run_process_tree_alive(
                processes,
                root_pid=member["pid"],
                pgid=member["pgid"],
                sid=member["sid"],
                slot=member["slot"],
            )
            for member in transition["cohort"]
        )
        if transition is not None
        else None
    )
    return {
        "project": WANDB_PROJECT,
        "total": len(tasks),
        "done": sum(task_done(task, update_marker=False) for task in tasks),
        "failed": [
            task.run_name
            for task in tasks
            if read_marker(task).get("state") == "failed"
        ],
        "live": {
            name: {
                "pid": process.pid,
                "affinity": sorted(process.affinity),
                "slot_type": (
                    "single" if len(process.affinity) == 1 else "legacy_pair"
                ),
                "state": process.state,
            }
            for name, process in relevant.items()
        },
        "transition": {
            "phase": (
                transition["phase"] if transition is not None else "unprepared"
            ),
            "rollout": (
                transition.get("rollout")
                if transition is not None
                else None
            ),
            "launch_enabled": (
                transition is not None
                and transition["phase"] == PHASE_SINGLE
            ),
            "cohort_total": (
                len(transition["cohort"])
                if transition is not None
                else None
            ),
            "cohort_live": cohort_live,
            "cohort_finished": (
                len(transition["cohort"]) - int(cohort_live)
                if transition is not None and cohort_live is not None
                else None
            ),
            "error": transition_error,
        },
        "markers": {
            state: sum(
                read_marker(task).get("state") == state for task in tasks
            )
            for state in (
                "running",
                "done",
                "retry_pending",
                "failed",
            )
        },
    }


def handle_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log(f"Received signal {signum}; stopping scheduler only.")


def acquire_lock() -> Any:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    stream = LOCK_PATH.open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("schedule_budget_curves_completion is already running")
    stream.seek(0)
    stream.truncate()
    stream.write(f"{os.getpid()}\n")
    stream.flush()
    return stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--prepare-transition", action="store_true")
    modes.add_argument(
        "--migrate-pref-bernoulli-budget",
        action="store_true",
        help=(
            "Atomically replace the never-launched pref_bernoulli 15k point "
            "with the approved 1M point."
        ),
    )
    modes.add_argument(
        "--migrate-hybrid-bernoulli-max-budget",
        action="store_true",
        help=(
            "Atomically add three runs with 2723 preferences and 2723 "
            "demonstrations."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=len(SINGLE_SLOTS),
        choices=range(1, len(SINGLE_SLOTS) + 1),
    )
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--loop-seconds", type=int, default=LOOP_SECONDS)
    parser.add_argument(
        "--defer-resolved-configs",
        action="store_true",
        help=(
            "Launch after all preflight checks except slow Hydra resolution; "
            "use only when those checks will run concurrently after launch."
        ),
    )
    parser.add_argument(
        "--resume-without-preflight",
        action="store_true",
        help=(
            "Resume an already validated single-core campaign without "
            "repeating the expensive Optuna/config preflight."
        ),
    )
    parser.add_argument("--legacy-manager-pid", type=int)
    parser.add_argument("--expected-legacy-runs", type=int, default=12)
    parser.add_argument(
        "--allow-stopped-legacy-manager",
        action="store_true",
        help=(
            "Prepare from surviving trainers after a verified legacy-manager "
            "exit and stale matching lock."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = build_tasks()
    validate_task_matrix(tasks)

    if args.migrate_pref_bernoulli_budget:
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        lock_stream = acquire_lock()
        try:
            coordinators = other_orchestrators(iter_processes())
            if coordinators:
                detail = [
                    (process.pid, process.command)
                    for process in coordinators
                ]
                raise RuntimeError(
                    f"Cannot migrate while orchestrators are active: {detail}"
                )
            result = migrate_pref_bernoulli_budget(tasks)
        finally:
            lock_stream.close()
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.migrate_hybrid_bernoulli_max_budget:
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        lock_stream = acquire_lock()
        try:
            coordinators = other_orchestrators(iter_processes())
            if coordinators:
                detail = [
                    (process.pid, process.command)
                    for process in coordinators
                ]
                raise RuntimeError(
                    f"Cannot migrate while orchestrators are active: {detail}"
                )
            result = migrate_hybrid_bernoulli_max_budget(tasks)
        finally:
            lock_stream.close()
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.status:
        print(json.dumps(status_payload(tasks), indent=2, sort_keys=True))
        return

    if args.max_attempts <= 0 or args.loop_seconds <= 0:
        raise SystemExit("--max-attempts and --loop-seconds must be positive.")
    if args.expected_legacy_runs <= 0:
        raise SystemExit("--expected-legacy-runs must be positive.")

    if args.prepare_transition:
        if args.legacy_manager_pid is None:
            raise SystemExit(
                "--prepare-transition requires --legacy-manager-pid."
            )
        result = preflight(
            tasks,
            require_free_cores=False,
            allowed_orchestrator_pid=args.legacy_manager_pid,
        )
        state = capture_transition_state(
            tasks,
            iter_processes(),
            legacy_manager_pid=args.legacy_manager_pid,
            expected_live=args.expected_legacy_runs,
            allow_stopped_manager=args.allow_stopped_legacy_manager,
        )
        print(
            json.dumps(
                {
                    "preflight": {
                        "tasks": result["tasks"],
                        "points": result["points"],
                        "matrix_sha256": result["matrix_sha256"],
                    },
                    "transition": state,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    transition_state = (
        load_transition_state(tasks)
        if TRANSITION_STATE_PATH.exists()
        else None
    )
    validation_only = args.preflight or args.dry_run
    if validation_only:
        result = preflight(
            tasks,
            require_free_cores=False,
            transition_state=transition_state,
            allowed_orchestrator_pid=args.legacy_manager_pid,
        )
        if args.preflight:
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        if args.dry_run:
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k != "git"},
                    indent=2,
                )
            )
            for index, task in enumerate(tasks, start=1):
                slot = SLOTS[(index - 1) % len(SLOTS)]
                print(
                    f"{index:02d}/{len(tasks)} {task.key} slot={slot} "
                    f"wandb_id={task.wandb_run_id}"
                )
                print(f"  {shlex.join(build_training_command(task, slot))}")
            return

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    MARKER_ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = acquire_lock()
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    try:
        # Re-read state and preflight only after holding the authoritative
        # manager lock, closing the preflight/start race.
        transition_state = load_transition_state(tasks)
        if args.resume_without_preflight:
            result = {
                **validate_task_matrix(tasks),
                "project": WANDB_PROJECT,
                "slots": list(SINGLE_SLOTS),
                "legacy_slots": list(LEGACY_PAIR_SLOTS),
                "transition_phase": transition_state["phase"],
                "preflight_deferred": "user_approved_recovery_resume",
            }
        else:
            result = preflight(
                tasks,
                require_free_cores=transition_state["phase"] == PHASE_DRAIN,
                transition_state=transition_state,
                defer_resolved_configs=args.defer_resolved_configs,
            )
        write_manifest(tasks, result, transition_state)
        Scheduler(
            tasks,
            max_parallel=args.max_parallel,
            max_attempts=args.max_attempts,
            loop_seconds=args.loop_seconds,
            transition_state=transition_state,
        ).loop()
    finally:
        lock_stream.close()


if __name__ == "__main__":
    main()
