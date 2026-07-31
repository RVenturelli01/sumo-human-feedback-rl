#!/usr/bin/env python3
"""Run the approved no-normalization experiment pipeline on cores 30-47.

The scheduler waits for the pref_bernoulli final launcher to finish, then:

1. tunes demo_1 and demo_2 in fresh ``*_no_norm`` Optuna studies;
2. runs five-seed budget curves with a fixed demo subset;
3. selects the level with the largest seed-mean held-out fast return;
4. runs ten final seeds with paired training/demo-subsample seeds;

It never signals external processes. Launches require a free, CPU-idle slot,
and state is reconstructed from Optuna, output files, and live processes so a
restarted scheduler does not duplicate completed work.

The pref_soft normalize=false ablation is intentionally disabled. Set
``ENABLE_PREF_SOFT_ABLATION=true`` only to opt back into those tasks.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/home/fis3/miniconda3/envs/sumo-rlhf/bin/python"
JOURNAL = REPO_ROOT / "outputs" / "optuna" / "journal.log"
STATE_ROOT = REPO_ROOT / "outputs" / "demo_no_norm_pipeline"
LOG_ROOT = STATE_ROOT / "logs"
RUN_ROOT = STATE_ROOT / "runs"
BEST_ROOT = STATE_ROOT / "best_configs"

ENTITY = "andrea02polimi-politecnico-di-milano"
TUNING_PROJECT = "tuning-thesis"
BUDGET_PROJECT = "tuning-thesis-demo-no-norm-budget-curves"
FINAL_PROJECT = "thesis"
STUDY_SUFFIX = "_no_norm"

ARMS = ("demo_1", "demo_2")
TUNING_TARGET = 30
TUNING_MAX_WORKERS = {"demo_1": 4, "demo_2": 5}
TUNING_DEMO_BUDGET = 500
FIXED_SUBSAMPLE_SEED = 1000
BUDGET_LEVELS = (50, 100, 200, 500, 1000, 2723)
BUDGET_SEEDS = tuple(range(1, 6))
FINAL_SEEDS = tuple(range(1, 11))
PREF_SOFT_BUDGET = 500
SLOTS = tuple(f"{core}-{core + 1}" for core in range(30, 48, 2))
PREF_SOFT_ABLATION_ENABLED = (
    os.environ.get("ENABLE_PREF_SOFT_ABLATION", "false").strip().lower() == "true"
)

TUNING_TIMESTEPS = 1_000_000
RUN_TIMESTEPS = 2_000_000
TIMESTEPS_PER_ITERATION = 20_000
N_ENVS = 2
EVAL_EPISODES = 20

BERNOULLI_GROUP = "pref_bernoulli_q100k_temp"
BERNOULLI_LAUNCHER = "run_final_5seeds.sh pref_bernoulli _q100k_temp"
LOOP_SECONDS = 10
CPU_SAMPLE_SECONDS = 1
CPU_MEAN_BUSY = 35.0
CPU_MAX_BUSY = 70.0
IDLE_SCANS_REQUIRED = 2
MAX_ATTEMPTS = 2


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    state: str
    affinity: frozenset[int]
    command: str


@dataclass(frozen=True)
class Task:
    kind: str
    arm: str
    seed: int
    level: int | None = None

    @property
    def group(self) -> str:
        if self.kind == "budget":
            return f"budget_{self.arm}_{self.level}"
        if self.kind == "final":
            return f"{self.arm}_no_norm"
        return "pref_soft_no_norm_ablation"

    @property
    def run_name(self) -> str:
        return f"{self.group}-seed{self.seed}"

    @property
    def key(self) -> str:
        level = f"_{self.level}" if self.level is not None else ""
        return f"{self.kind}_{self.arm}{level}_seed{self.seed}"

    @property
    def output_root(self) -> Path:
        return RUN_ROOT / self.kind / self.run_name


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def slot_cores(slot: str) -> frozenset[int]:
    lower, upper = (int(value) for value in slot.split("-"))
    return frozenset(range(lower, upper + 1))


def iter_processes() -> list[ProcessInfo]:
    processes = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        pid = int(item.name)
        try:
            fields = (item / "stat").read_text().split()
            state = fields[2]
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
        except (OSError, PermissionError, IndexError):
            continue
        processes.append(ProcessInfo(pid, state, affinity, command))
    return processes


def live_run_names(processes: list[ProcessInfo]) -> dict[str, ProcessInfo]:
    result = {}
    for process in processes:
        if "scripts/train_hybrid_sac.py" not in process.command:
            continue
        match = re.search(r"(?:^|\s)run\.name=([^\s]+)", process.command)
        if match:
            result[match.group(1)] = process
    return result


def option_value(command: str, option: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    try:
        index = tokens.index(option)
    except ValueError:
        return None
    return tokens[index + 1] if index + 1 < len(tokens) else None


def tuning_workers(processes: list[ProcessInfo], arm: str | None = None) -> list[ProcessInfo]:
    workers = []
    for process in processes:
        command = process.command
        if "scripts/tune_hybrid_sac.py" not in command:
            continue
        if option_value(command, "--study-suffix") != STUDY_SUFFIX:
            continue
        worker_arm = option_value(command, "--arm")
        if worker_arm in ARMS and (arm is None or worker_arm == arm):
            workers.append(process)
    return workers


def worker_slots(processes: list[ProcessInfo]) -> set[str]:
    return {
        slot
        for process in tuning_workers(processes)
        if (slot := option_value(process.command, "--cores")) in SLOTS
    }


def bernoulli_is_live(processes: list[ProcessInfo]) -> tuple[bool, list[int]]:
    matches = [
        process.pid
        for process in processes
        if (
            f"run.group={BERNOULLI_GROUP}" in process.command
            or BERNOULLI_LAUNCHER in process.command
        )
    ]
    return bool(matches), sorted(matches)


def other_orchestrators(processes: list[ProcessInfo]) -> list[ProcessInfo]:
    own_name = Path(__file__).name
    result = []
    for process in processes:
        command = process.command
        if process.pid == os.getpid() or own_name in command:
            continue
        if "python" not in command:
            continue
        if "schedule_" in command or "orchestrator" in command:
            result.append(process)
    return result


def constrained_slot_processes(
    processes: list[ProcessInfo], slot: str
) -> list[ProcessInfo]:
    cores = slot_cores(slot)
    return [
        process
        for process in processes
        if len(process.affinity) <= 8 and process.affinity & cores
    ]


def _core_times() -> tuple[dict[int, int], dict[int, int]]:
    busy, total = {}, {}
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


def storage():
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    return JournalStorage(
        JournalFileBackend(
            str(JOURNAL), lock_obj=JournalFileOpenLock(str(JOURNAL))
        )
    )


def study_snapshot(arm: str) -> dict[str, Any]:
    import optuna

    name = f"hybrid_sac_{arm}{STUDY_SUFFIX}"
    try:
        study = optuna.load_study(study_name=name, storage=storage())
    except KeyError:
        return {"name": name, "total": 0, "complete": 0, "best": None}
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    ]
    return {
        "name": name,
        "total": len(study.trials),
        "complete": len(complete),
        "best": max((float(trial.value) for trial in complete), default=None),
    }


def tuning_done(arm: str, processes: list[ProcessInfo]) -> bool:
    snapshot = study_snapshot(arm)
    return (
        snapshot["total"] >= TUNING_TARGET
        and snapshot["complete"] > 0
        and not tuning_workers(processes, arm)
    )


def run_checked(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()


def exported_overrides(
    arm: str,
    study_suffix: str = "",
    pref_budget: int = 5000,
    demo_budget: int = 500,
) -> list[str]:
    command = [
        PYTHON,
        "scripts/export_best_config.py",
        "--arm",
        arm,
        "--study-suffix",
        study_suffix,
        "--format",
        "full",
        "--storage-path",
        str(JOURNAL.relative_to(REPO_ROOT)),
        "--pref-budget",
        str(pref_budget),
        "--demo-budget",
        str(demo_budget),
    ]
    return shlex.split(run_checked(command))


def best_params(arm: str, study_suffix: str = "") -> dict[str, Any]:
    output = run_checked(
        [
            PYTHON,
            "scripts/export_best_config.py",
            "--arm",
            arm,
            "--study-suffix",
            study_suffix,
            "--format",
            "params",
            "--storage-path",
            str(JOURNAL.relative_to(REPO_ROOT)),
        ]
    )
    return json.loads(output)


def save_best_snapshot(arm: str) -> list[str]:
    path = BEST_ROOT / f"{arm}{STUDY_SUFFIX}.json"
    overrides = exported_overrides(
        arm, STUDY_SUFFIX, demo_budget=TUNING_DEMO_BUDGET
    )
    effective = [*overrides, "algo.kwargs.normalize_agent_reward=false"]
    snapshot = study_snapshot(arm)
    atomic_json(
        path,
        {
            **snapshot,
            "arm": arm,
            "study_suffix": STUDY_SUFFIX,
            "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": best_params(arm, STUDY_SUFFIX),
            "exported_overrides": overrides,
            "effective_overrides": effective,
            "normalization_note": (
                "The historical fixed config contains normalize=true; the final "
                "effective override is normalize=false."
            ),
        },
    )
    return effective


def load_demo_overrides(arm: str) -> list[str]:
    path = BEST_ROOT / f"{arm}{STUDY_SUFFIX}.json"
    if not path.exists():
        return save_best_snapshot(arm)
    return json.loads(path.read_text())["effective_overrides"]


def pref_soft_overrides() -> list[str]:
    overrides = exported_overrides("pref_soft", pref_budget=5000)
    params = best_params("pref_soft")
    tuned_initial = int(params.get("initial_queries", PREF_SOFT_BUDGET // 5))
    initial = min(tuned_initial, PREF_SOFT_BUDGET // 5)
    return [
        *overrides,
        f"algo.kwargs.initial_queries={initial}",
        f"algo.kwargs.total_queries={PREF_SOFT_BUDGET}",
        f"train.kwargs.total_queries={PREF_SOFT_BUDGET}",
        "algo.kwargs.normalize_agent_reward=false",
    ]


def task_final_files(task: Task) -> list[Path]:
    if not task.output_root.exists():
        return []
    return sorted(task.output_root.rglob("final_eval.json"))


def task_done(task: Task) -> bool:
    return bool(task_final_files(task))


def attempt_path(task: Task) -> Path:
    return STATE_ROOT / "attempts" / f"{task.key}.txt"


def attempts(task: Task) -> int:
    try:
        return int(attempt_path(task).read_text().strip())
    except (OSError, ValueError):
        return 0


def record_attempt(task: Task) -> int:
    count = attempts(task) + 1
    path = attempt_path(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{count}\n")
    return count


def all_budget_tasks(arm: str) -> list[Task]:
    return [
        Task("budget", arm, seed, level)
        for level in BUDGET_LEVELS
        for seed in BUDGET_SEEDS
    ]


def all_final_tasks(arm: str) -> list[Task]:
    return [Task("final", arm, seed) for seed in FINAL_SEEDS]


def all_ablation_tasks() -> list[Task]:
    if not PREF_SOFT_ABLATION_ENABLED:
        return []
    return [Task("ablation", "pref_soft", seed) for seed in FINAL_SEEDS]


def read_metrics(task: Task) -> dict[str, Any]:
    files = task_final_files(task)
    if not files:
        raise RuntimeError(f"missing final_eval.json for {task.key}")
    return json.loads(files[-1].read_text())


def best_budget_path(arm: str) -> Path:
    return STATE_ROOT / f"best_budget_{arm}.json"


def select_best_budget(arm: str) -> int:
    path = best_budget_path(arm)
    if path.exists():
        return int(json.loads(path.read_text())["selected_budget"])

    tasks = all_budget_tasks(arm)
    if not all(task_done(task) for task in tasks):
        raise RuntimeError(f"budget curves incomplete for {arm}")

    levels = {}
    for level in BUDGET_LEVELS:
        metrics = [
            read_metrics(Task("budget", arm, seed, level))
            for seed in BUDGET_SEEDS
        ]
        returns = [float(item["eval/mean_fast_return"]) for item in metrics]
        successes = [float(item["eval/success_rate"]) for item in metrics]
        levels[str(level)] = {
            "n_seeds": len(returns),
            "return_mean": statistics.fmean(returns),
            "return_std": statistics.pstdev(returns),
            "return_min": min(returns),
            "success_mean": statistics.fmean(successes),
            "success_std": statistics.pstdev(successes),
        }
    selected = max(
        BUDGET_LEVELS,
        key=lambda level: (
            levels[str(level)]["return_mean"],
            levels[str(level)]["success_mean"],
            -level,
        ),
    )
    atomic_json(
        path,
        {
            "arm": arm,
            "criterion": (
                "maximum five-seed mean eval/mean_fast_return; success mean and "
                "smaller budget are tie-breakers"
            ),
            "selected_budget": selected,
            "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "levels": levels,
        },
    )
    log(
        f"BEST BUDGET {arm}: {selected} "
        f"(mean return={levels[str(selected)]['return_mean']:.6f})"
    )
    return selected


def base_training_command(task: Task, slot: str) -> list[str]:
    if task.kind == "ablation":
        overrides = pref_soft_overrides()
        tags = "[final,pref_soft,no_norm,ablation]"
        project = FINAL_PROJECT
    else:
        overrides = load_demo_overrides(task.arm)
        tags = (
            f"[budget_curve,{task.arm},no_norm,fixed_demo_subsample_seed]"
            if task.kind == "budget"
            else f"[final,{task.arm},no_norm,retuned]"
        )
        project = BUDGET_PROJECT if task.kind == "budget" else FINAL_PROJECT

    command = [
        "taskset",
        "-c",
        slot,
        PYTHON,
        "scripts/train_hybrid_sac.py",
        *overrides,
    ]
    if task.kind == "budget":
        command += [
            f"run.n_expert_trajectories={task.level}",
            f"run.demo_subsample_seed={FIXED_SUBSAMPLE_SEED}",
        ]
    elif task.kind == "final":
        command += [
            f"run.n_expert_trajectories={select_best_budget(task.arm)}",
            f"run.demo_subsample_seed={FIXED_SUBSAMPLE_SEED + task.seed}",
        ]
    command += [
        f"run.seed={task.seed}",
        f"run.output_dir={task.output_root.relative_to(REPO_ROOT)}",
        f"run.name={task.run_name}",
        f"run.group={task.group}",
        f"wandb.entity={ENTITY}",
        f"wandb.project={project}",
        f"wandb.tags={tags}",
        f"env.n_envs={N_ENVS}",
        f"eval.n_episodes={EVAL_EPISODES}",
        f"train.kwargs.total_timesteps={RUN_TIMESTEPS}",
        f"train.kwargs.timesteps_per_iteration={TIMESTEPS_PER_ITERATION}",
        "algo.kwargs.normalize_agent_reward=false",
    ]
    return command


def launch_task(task: Task, slot: str) -> int:
    task.output_root.mkdir(parents=True, exist_ok=True)
    attempt = record_attempt(task)
    command = base_training_command(task, slot)
    log_path = LOG_ROOT / f"{task.key}.log"
    with open(log_path, "a") as stream:
        stream.write(
            f"\n=== {time.strftime('%F %T')} attempt={attempt} slot={slot}\n"
        )
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(
        f"START {task.key} pid={process.pid} slot={slot} attempt={attempt}"
    )
    return process.pid


def launch_tuning_worker(arm: str, slot: str) -> int:
    command = [
        PYTHON,
        "scripts/tune_hybrid_sac.py",
        "--arm",
        arm,
        "--n-trials",
        "1",
        "--cores",
        slot,
        "--study-suffix",
        STUDY_SUFFIX,
        "--demo-budget",
        str(TUNING_DEMO_BUDGET),
        "--total-timesteps",
        str(TUNING_TIMESTEPS),
        "--timesteps-per-iteration",
        str(TIMESTEPS_PER_ITERATION),
        "--n-envs",
        str(N_ENVS),
        "--eval-episodes",
        str(EVAL_EPISODES),
        "--wandb-entity",
        ENTITY,
        "--wandb-project",
        TUNING_PROJECT,
        "--override",
        "algo.kwargs.normalize_agent_reward=false",
        "--override",
        f"run.demo_subsample_seed={FIXED_SUBSAMPLE_SEED}",
        "--override",
        f"wandb.tags=[optuna,{arm},no_norm,fixed_demo_subsample_seed]",
    ]
    log_path = LOG_ROOT / f"tuning_{arm}_{slot}.log"
    with open(log_path, "a") as stream:
        stream.write(f"\n=== {time.strftime('%F %T')} slot={slot}\n")
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(f"START tuning {arm} pid={process.pid} slot={slot}")
    return process.pid


def pending_task(
    processes: list[ProcessInfo], live_names: dict[str, ProcessInfo]
) -> Task | None:
    # Critical path first: budget and final demo runs as soon as dependencies
    # are complete. The pref_soft ablation fills only otherwise-unused capacity.
    for arm in ARMS:
        if not tuning_done(arm, processes):
            continue
        if not (BEST_ROOT / f"{arm}{STUDY_SUFFIX}.json").exists():
            save_best_snapshot(arm)
        budget_tasks = all_budget_tasks(arm)
        for task in budget_tasks:
            if (
                not task_done(task)
                and task.run_name not in live_names
                and attempts(task) < MAX_ATTEMPTS
            ):
                return task
        if all(task_done(task) for task in budget_tasks):
            select_best_budget(arm)
            for task in all_final_tasks(arm):
                if (
                    not task_done(task)
                    and task.run_name not in live_names
                    and attempts(task) < MAX_ATTEMPTS
                ):
                    return task

    for task in all_ablation_tasks():
        if (
            not task_done(task)
            and task.run_name not in live_names
            and attempts(task) < MAX_ATTEMPTS
        ):
            return task
    return None


def failed_tasks(processes: list[ProcessInfo] | None = None) -> list[Task]:
    tasks = [
        *(task for arm in ARMS for task in all_budget_tasks(arm)),
        *(task for arm in ARMS for task in all_final_tasks(arm)),
        *all_ablation_tasks(),
    ]
    live_names = live_run_names(processes or [])
    return [
        task
        for task in tasks
        if (
            not task_done(task)
            and task.run_name not in live_names
            and attempts(task) >= MAX_ATTEMPTS
        )
    ]


def pipeline_complete(processes: list[ProcessInfo]) -> bool:
    return (
        all(tuning_done(arm, processes) for arm in ARMS)
        and all(
            task_done(task)
            for arm in ARMS
            for task in (*all_budget_tasks(arm), *all_final_tasks(arm))
        )
        and all(task_done(task) for task in all_ablation_tasks())
        and not tuning_workers(processes)
    )


def write_manifest() -> None:
    atomic_json(
        STATE_ROOT / "manifest.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cores": list(SLOTS),
            "entity": ENTITY,
            "tuning": {
                "project": TUNING_PROJECT,
                "arms": list(ARMS),
                "study_suffix": STUDY_SUFFIX,
                "target_trials_per_arm": TUNING_TARGET,
                "demo_budget": TUNING_DEMO_BUDGET,
                "demo_subsample_seed": FIXED_SUBSAMPLE_SEED,
                "normalize_agent_reward": False,
            },
            "budget_curves": {
                "project": BUDGET_PROJECT,
                "levels": list(BUDGET_LEVELS),
                "seeds": list(BUDGET_SEEDS),
                "groups": "budget_<arm>_<level>",
                "selection": "max five-seed mean eval/mean_fast_return",
            },
            "final": {
                "project": FINAL_PROJECT,
                "groups": ["demo_1_no_norm", "demo_2_no_norm"],
                "training_seeds": list(FINAL_SEEDS),
                "demo_subsample_seeds": [
                    FIXED_SUBSAMPLE_SEED + seed for seed in FINAL_SEEDS
                ],
            },
            "pref_soft_ablation": {
                "enabled": PREF_SOFT_ABLATION_ENABLED,
                "project": FINAL_PROJECT,
                "group": "pref_soft_no_norm_ablation",
                "budget": PREF_SOFT_BUDGET,
                "seeds": list(FINAL_SEEDS),
                "config_source": "hybrid_sac_pref_soft",
                "only_intended_change": "normalize_agent_reward=false",
            },
            "report_commands": [
                (
                    "python scripts/report_budget_curves.py --project "
                    f"{BUDGET_PROJECT}"
                ),
                "python scripts/report_thesis_runs.py --project thesis",
            ],
        },
    )


def status_payload(processes: list[ProcessInfo]) -> dict[str, Any]:
    bernoulli, bernoulli_pids = bernoulli_is_live(processes)
    live_names = live_run_names(processes)
    return {
        "bernoulli_gate": bernoulli,
        "bernoulli_pids": bernoulli_pids,
        "pref_soft_ablation_enabled": PREF_SOFT_ABLATION_ENABLED,
        "tuning": {
            arm: {
                **study_snapshot(arm),
                "workers": [p.pid for p in tuning_workers(processes, arm)],
            }
            for arm in ARMS
        },
        "live_pipeline_runs": sorted(
            name
            for name in live_names
            if (
                "_no_norm" in name
                or name.startswith("budget_demo_1_")
                or name.startswith("budget_demo_2_")
            )
        ),
        "done": {
            "budget": {
                arm: sum(task_done(task) for task in all_budget_tasks(arm))
                for arm in ARMS
            },
            "final": {
                arm: sum(task_done(task) for task in all_final_tasks(arm))
                for arm in ARMS
            },
            "pref_soft_ablation": sum(
                task_done(task) for task in all_ablation_tasks()
            ),
        },
        "failed": [task.key for task in failed_tasks(processes)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print reconstructed state and exit without launching anything.",
    )
    args = parser.parse_args()

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    BEST_ROOT.mkdir(parents=True, exist_ok=True)
    write_manifest()

    if args.status:
        print(json.dumps(status_payload(iter_processes()), indent=2))
        return

    lock_stream = open(STATE_ROOT / "orchestrator.lock", "w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("orchestrator already running")
    lock_stream.write(f"{os.getpid()}\n")
    lock_stream.flush()

    idle_streak = {slot: 0 for slot in SLOTS}
    gate_was_live = False
    last_wait_log = 0.0
    log(
        f"START pid={os.getpid()} slots={','.join(SLOTS)}; "
        f"waiting for {BERNOULLI_GROUP}"
    )

    while True:
        processes = iter_processes()
        bernoulli_live, bernoulli_pids = bernoulli_is_live(processes)
        if bernoulli_live:
            gate_was_live = True
            if time.time() - last_wait_log >= 300 or last_wait_log == 0:
                log(f"WAIT Bernoulli final runs pids={bernoulli_pids}")
                last_wait_log = time.time()
            time.sleep(LOOP_SECONDS)
            continue
        if gate_was_live:
            log("GATE OPEN: Bernoulli final launcher and runs have ended")
            gate_was_live = False

        foreign = other_orchestrators(processes)
        if foreign:
            if time.time() - last_wait_log >= 300:
                log(
                    "WAIT other orchestrator(s): "
                    + ",".join(str(process.pid) for process in foreign)
                )
                last_wait_log = time.time()
            time.sleep(LOOP_SECONDS)
            continue

        if pipeline_complete(processes):
            atomic_json(
                STATE_ROOT / "COMPLETE.json",
                {
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": status_payload(processes),
                },
            )
            log("COMPLETE: all enabled tuning, budget, final, and ablation runs finished")
            return

        failures = failed_tasks(processes)
        if failures:
            atomic_json(
                STATE_ROOT / "FAILED.json",
                {
                    "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tasks": [task.key for task in failures],
                    "status": status_payload(processes),
                },
            )
            raise SystemExit(
                "tasks exhausted retries: "
                + ", ".join(task.key for task in failures)
            )

        live_names = live_run_names(processes)
        reserved = worker_slots(processes)
        for name, process in live_names.items():
            if (
                "_no_norm" in name
                or name.startswith("budget_demo_1_")
                or name.startswith("budget_demo_2_")
            ):
                for slot in SLOTS:
                    if process.affinity & slot_cores(slot):
                        reserved.add(slot)

        cpu = slot_cpu_busy()
        processes = iter_processes()
        for slot in SLOTS:
            occupied = constrained_slot_processes(processes, slot)
            mean_busy, max_busy = cpu[slot]
            if (
                slot not in reserved
                and not occupied
                and mean_busy < CPU_MEAN_BUSY
                and max_busy < CPU_MAX_BUSY
            ):
                idle_streak[slot] += 1
            else:
                idle_streak[slot] = 0

        launched = False
        for slot in SLOTS:
            if idle_streak[slot] < IDLE_SCANS_REQUIRED:
                continue

            processes = iter_processes()
            live_names = live_run_names(processes)
            if constrained_slot_processes(processes, slot):
                idle_streak[slot] = 0
                continue

            # Reserve new Optuna trials one at a time per loop so the shared
            # journal reflects the reservation before another worker starts.
            candidates = []
            for arm in ARMS:
                snapshot = study_snapshot(arm)
                workers = tuning_workers(processes, arm)
                if (
                    snapshot["total"] < TUNING_TARGET
                    and len(workers) < TUNING_MAX_WORKERS[arm]
                ):
                    candidates.append((len(workers), snapshot["total"], arm))
            if candidates:
                _, _, arm = min(candidates)
                launch_tuning_worker(arm, slot)
                idle_streak[slot] = 0
                launched = True
                break

            task = pending_task(processes, live_names)
            if task is not None:
                launch_task(task, slot)
                idle_streak[slot] = 0
                launched = True

        if not launched and time.time() - last_wait_log >= 300:
            payload = status_payload(iter_processes())
            log("STATUS " + json.dumps(payload, separators=(",", ":")))
            last_wait_log = time.time()
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    main()
