#!/usr/bin/env python3
"""Launch final 10-seed Bernoulli hybrid A/B runs after tuning completes.

The manager is deliberately conservative:

* it waits for the tuning COMPLETE marker, the tuning manager to exit, and all
  ``_bernoulli_norm`` tuning workers to disappear;
* it launches only on pairs that are process-free and CPU-idle for two scans;
* it rechecks the tuning gate and slot immediately before every launch;
* it reconstructs completion from local final-eval files and W&B, so a restart
  does not intentionally duplicate completed work.

Strategies use half/full budgets from the definitive single-source baselines:

* Demo 1: A=50k preferences + 500 demos, B=100k + 1000;
* Demo 2: A=50k preferences + 250 demos, B=100k + 500.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/home/fis3/miniconda3/envs/sumo-rlhf/bin/python"
JOURNAL = REPO_ROOT / "outputs" / "optuna" / "journal.log"
STATE_ROOT = REPO_ROOT / "outputs" / "hybrid_bernoulli_final_ab"
LOG_ROOT = STATE_ROOT / "logs"

ENTITY = "andrea02polimi-politecnico-di-milano"
WANDB_PROJECT = "thesis"
STUDY_SUFFIX = "_bernoulli_norm"
TUNING_MANAGER = "scripts/hybrid_bernoulli_tuning_manager.py"
TUNING_COMPLETE = (
    REPO_ROOT / "outputs" / "hybrid_bernoulli_tuning" / "COMPLETE.json"
)
ARMS = ("hybrid_demo_1", "hybrid_demo_2")
TARGET_TRIALS = 30
TUNING_PREF_BUDGET = 100_000

SLOTS = tuple(f"{core}-{core + 1}" for core in range(24, 48, 2))
LOOP_SECONDS = 10
CPU_SAMPLE_SECONDS = 1
CPU_MEAN_BUSY = 35.0
CPU_MAX_BUSY = 70.0
IDLE_SCANS_REQUIRED = 2
MAX_OWN_WORKERS = len(SLOTS)
MAX_ATTEMPTS = 2
RETRY_GRACE_SECONDS = 300
WANDB_REFRESH_SECONDS = 180

TOTAL_TIMESTEPS = 2_000_000
TIMESTEPS_PER_ITERATION = 20_000
N_ENVS = 2
EVAL_EPISODES = 20

SOURCE_FILES = (
    "scripts/train_hybrid_sac.py",
    "scripts/export_best_config.py",
    "scripts/tune_hybrid_sac.py",
)


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    state: str
    affinity: frozenset[int]
    command: str


@dataclass(frozen=True)
class Task:
    arm: str
    strategy: str
    seed: int
    pref_budget: int
    demo_budget: int

    @property
    def group(self) -> str:
        return f"{self.arm}_bernoulli_{self.strategy}"

    @property
    def run_name(self) -> str:
        return f"{self.group}-seed{self.seed}"

    @property
    def key(self) -> str:
        return f"{self.arm}_{self.strategy}_seed{self.seed}"

    @property
    def demo_subsample_seed(self) -> int:
        return 1000 + self.seed

    @property
    def output_parent(self) -> Path:
        return REPO_ROOT / "outputs" / "final" / self.group


def build_tasks() -> list[Task]:
    budgets = {
        "hybrid_demo_1": {
            "A": (50_000, 500),
            "B": (100_000, 1000),
        },
        "hybrid_demo_2": {
            "A": (50_000, 250),
            "B": (100_000, 500),
        },
    }
    tasks = []
    # Keep every partial wave balanced across arms and strategies.
    for seed in range(1, 11):
        for arm in ARMS:
            for strategy in ("A", "B"):
                pref_budget, demo_budget = budgets[arm][strategy]
                tasks.append(
                    Task(arm, strategy, seed, pref_budget, demo_budget)
                )
    return tasks


TASKS = tuple(build_tasks())


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    missing = [
        relative for relative in SOURCE_FILES if not (REPO_ROOT / relative).is_file()
    ]
    if missing:
        raise SystemExit("Missing source files: " + ", ".join(missing))
    return {relative: sha256(REPO_ROOT / relative) for relative in SOURCE_FILES}


def assert_sources_unchanged(expected: dict[str, str]) -> None:
    actual = source_hashes()
    if actual != expected:
        raise SystemExit(
            "Source files changed after manager startup; refusing new launches: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )


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


def tuning_manager_live(processes: list[ProcessInfo]) -> bool:
    return any(TUNING_MANAGER in process.command for process in processes)


def tuning_workers(processes: list[ProcessInfo]) -> list[ProcessInfo]:
    workers = []
    for process in processes:
        if "scripts/tune_hybrid_sac.py" not in process.command:
            continue
        if option_value(process.command, "--study-suffix") != STUDY_SUFFIX:
            continue
        if option_value(process.command, "--arm") in ARMS:
            workers.append(process)
    return workers


def own_task_processes(
    processes: list[ProcessInfo], task: Task | None = None
) -> list[ProcessInfo]:
    names = {item.run_name for item in TASKS} if task is None else {task.run_name}
    return [
        process
        for process in processes
        if any(f"run.name={name}" in process.command for name in names)
    ]


def live_task_count(processes: list[ProcessInfo]) -> int:
    return sum(bool(own_task_processes(processes, task)) for task in TASKS)


def constrained_slot_processes(
    processes: list[ProcessInfo], slot: str
) -> list[ProcessInfo]:
    cores = slot_cores(slot)
    return [
        process
        for process in processes
        if len(process.affinity) <= 8 and process.affinity & cores
    ]


def foreign_coordinators(processes: list[ProcessInfo]) -> list[ProcessInfo]:
    own_name = Path(__file__).name
    pattern = re.compile(r"scripts/\S*(?:manager|schedule|orchestrator)\S*\.py")
    result = []
    for process in processes:
        if process.pid == os.getpid() or own_name in process.command:
            continue
        if TUNING_MANAGER in process.command:
            continue
        if "python" in process.command and pattern.search(process.command):
            result.append(process)
    return result


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
            str(JOURNAL),
            lock_obj=JournalFileOpenLock(str(JOURNAL)),
        )
    )


def study_snapshot(arm: str, include_best: bool = False) -> dict[str, Any]:
    import optuna

    name = f"hybrid_sac_{arm}{STUDY_SUFFIX}"
    study = optuna.load_study(study_name=name, storage=storage())
    counts = {"complete": 0, "pruned": 0, "failed": 0, "running": 0}
    completed = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            counts["complete"] += 1
            if trial.value is not None:
                completed.append(trial)
        elif trial.state == optuna.trial.TrialState.PRUNED:
            counts["pruned"] += 1
        elif trial.state == optuna.trial.TrialState.FAIL:
            counts["failed"] += 1
        elif trial.state in (
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.WAITING,
        ):
            counts["running"] += 1
    payload: dict[str, Any] = {
        "name": name,
        "total": len(study.trials),
        **counts,
        "preference_labels": study.user_attrs.get("preference_labels"),
    }
    if include_best:
        if not completed:
            raise SystemExit(f"No completed trial in {name}")
        best = max(completed, key=lambda trial: float(trial.value))
        payload["best"] = {
            "trial_number": best.number,
            "objective": best.value,
            "params": best.params,
            "eval": {
                key: value
                for key, value in best.user_attrs.items()
                if key.startswith("eval/")
            },
            "run_dir": best.user_attrs.get("run_dir"),
        }
    return payload


def tuning_gate(processes: list[ProcessInfo]) -> tuple[bool, dict[str, Any]]:
    marker = TUNING_COMPLETE.exists()
    manager_live = tuning_manager_live(processes)
    workers = tuning_workers(processes)
    snapshots = {}
    journal_ready = JOURNAL.exists()
    if journal_ready:
        try:
            snapshots = {arm: study_snapshot(arm) for arm in ARMS}
        except Exception as exc:
            snapshots = {"error": str(exc)}
            journal_ready = False
    studies_ready = bool(snapshots) and "error" not in snapshots and all(
        snapshot["total"] >= TARGET_TRIALS
        and snapshot["running"] == 0
        and snapshot["preference_labels"] == "binary_bernoulli"
        for snapshot in snapshots.values()
    )
    payload = {
        "complete_marker": marker,
        "manager_live": manager_live,
        "worker_count": len(workers),
        "worker_pids": [process.pid for process in workers],
        "journal_ready": journal_ready,
        "studies_ready": studies_ready,
        "studies": snapshots,
    }
    return (
        marker
        and not manager_live
        and not workers
        and journal_ready
        and studies_ready,
        payload,
    )


def run_checked(command: list[str]) -> str:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def exported_overrides(task: Task) -> list[str]:
    output = run_checked(
        [
            PYTHON,
            "scripts/export_best_config.py",
            "--arm",
            task.arm,
            "--study-suffix",
            STUDY_SUFFIX,
            "--format",
            "full",
            "--storage-path",
            str(JOURNAL),
            "--pref-budget",
            str(task.pref_budget),
            "--demo-budget",
            str(task.demo_budget),
            "--preference-labels",
            "binary_bernoulli",
        ]
    )
    return shlex.split(output)


def override_map(overrides: list[str]) -> dict[str, str]:
    result = {}
    for override in overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            result[key] = value
    return result


def scaled_initial_queries(task: Task, best_params: dict[str, Any]) -> int:
    tuned = int(best_params["initial_queries"])
    ratio = tuned / TUNING_PREF_BUDGET
    return max(100, round(task.pref_budget * ratio))


def validate_best_snapshot(arm: str, snapshot: dict[str, Any]) -> None:
    if snapshot["preference_labels"] != "binary_bernoulli":
        raise SystemExit(f"{arm} study is not binary_bernoulli")
    best = snapshot.get("best", {})
    params = best.get("params", {})
    required = {
        "pref_temperature",
        "initial_queries",
        "demo_weight",
        "batch_size_pref",
        "batch_size_expert",
    }
    missing = sorted(required - set(params))
    if missing:
        raise SystemExit(f"{arm} best trial misses params: {missing}")
    initial_queries = int(params["initial_queries"])
    if initial_queries not in (2000, 5000, 10000, 20000):
        raise SystemExit(
            f"{arm} has unexpected tuned initial_queries={initial_queries}"
        )


def validate_overrides(
    task: Task,
    overrides: list[str],
    best_params: dict[str, Any],
) -> int:
    scaled_iq = scaled_initial_queries(task, best_params)
    values = override_map([*overrides, f"algo.kwargs.initial_queries={scaled_iq}"])
    expected = {
        "algo.kwargs.labels_type": "binary_bernoulli",
        "algo.kwargs.total_queries": str(task.pref_budget),
        "train.kwargs.total_queries": str(task.pref_budget),
        "run.n_expert_trajectories": str(task.demo_budget),
        "algo.kwargs.loss_type": (
            "demo_1" if task.arm == "hybrid_demo_1" else "demo_2"
        ),
        "algo.kwargs.normalize_agent_reward": "true",
        "algo.kwargs.initial_queries": str(scaled_iq),
        "algo.kwargs.pref_temperature": str(best_params["pref_temperature"]),
        "algo.kwargs.demo_weight": str(best_params["demo_weight"]),
    }
    mismatches = {
        key: {"expected": expected_value, "actual": values.get(key)}
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    }
    if mismatches:
        raise SystemExit(
            f"Invalid overrides for {task.key}: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return scaled_iq


def task_final_files(task: Task) -> list[Path]:
    if not task.output_parent.exists():
        return []
    return sorted(task.output_parent.glob(f"{task.run_name}*/final_eval.json"))


def task_done(task: Task, wandb_finished: set[str] | None = None) -> bool:
    return bool(task_final_files(task)) or (
        wandb_finished is not None and task.run_name in wandb_finished
    )


def attempt_path(task: Task) -> Path:
    return STATE_ROOT / "attempts" / f"{task.key}.json"


def attempt_record(task: Task) -> dict[str, Any]:
    try:
        return json.loads(attempt_path(task).read_text())
    except (OSError, ValueError, TypeError):
        return {"count": 0, "last_started_epoch": 0}


def attempts(task: Task) -> int:
    return int(attempt_record(task).get("count", 0))


def retry_grace_elapsed(task: Task) -> bool:
    last_started = float(attempt_record(task).get("last_started_epoch", 0))
    return time.time() - last_started >= RETRY_GRACE_SECONDS


def record_attempt(task: Task, slot: str) -> int:
    previous = attempt_record(task)
    count = int(previous.get("count", 0)) + 1
    atomic_json(
        attempt_path(task),
        {
            "count": count,
            "last_started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_started_epoch": time.time(),
            "slot": slot,
        },
    )
    return count


def wandb_finished_names() -> set[str]:
    import wandb

    groups = {task.group for task in TASKS}
    return {
        run.name
        for run in wandb.Api(timeout=60).runs(
            f"{ENTITY}/{WANDB_PROJECT}", per_page=500
        )
        if run.group in groups and run.state == "finished"
    }


def build_command(
    task: Task,
    slot: str,
    best_params_by_arm: dict[str, dict[str, Any]],
) -> list[str]:
    best_params = best_params_by_arm[task.arm]
    overrides = exported_overrides(task)
    scaled_iq = validate_overrides(task, overrides, best_params)
    return [
        "taskset",
        "-c",
        slot,
        PYTHON,
        "scripts/train_hybrid_sac.py",
        *overrides,
        f"algo.kwargs.initial_queries={scaled_iq}",
        f"run.seed={task.seed}",
        f"run.demo_subsample_seed={task.demo_subsample_seed}",
        f"run.output_dir=outputs/final/{task.group}",
        f"run.name={task.run_name}",
        f"run.group={task.group}",
        f"wandb.entity={ENTITY}",
        f"wandb.project={WANDB_PROJECT}",
        (
            f"wandb.tags=[final,{task.arm},bernoulli,strategy_{task.strategy},"
            "normalized,paired_demo_subsample]"
        ),
        f"env.n_envs={N_ENVS}",
        f"eval.n_episodes={EVAL_EPISODES}",
        f"train.kwargs.total_timesteps={TOTAL_TIMESTEPS}",
        f"train.kwargs.timesteps_per_iteration={TIMESTEPS_PER_ITERATION}",
        "algo.kwargs.normalize_agent_reward=true",
    ]


def launch_task(
    task: Task,
    slot: str,
    best_params_by_arm: dict[str, dict[str, Any]],
) -> int:
    task.output_parent.mkdir(parents=True, exist_ok=True)
    command = build_command(task, slot, best_params_by_arm)
    attempt = record_attempt(task, slot)
    log_path = LOG_ROOT / f"{task.key}.log"
    with log_path.open("a") as stream:
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
        f"START {task.run_name} pid={process.pid} slot={slot} "
        f"pref={task.pref_budget} demo={task.demo_budget} "
        f"demo_subsample_seed={task.demo_subsample_seed}"
    )
    return process.pid


def pending_task(
    processes: list[ProcessInfo], wandb_finished: set[str]
) -> Task | None:
    for task in TASKS:
        if task_done(task, wandb_finished):
            continue
        if own_task_processes(processes, task):
            continue
        count = attempts(task)
        if count >= MAX_ATTEMPTS:
            continue
        if count and not retry_grace_elapsed(task):
            continue
        return task
    return None


def failed_tasks(
    processes: list[ProcessInfo], wandb_finished: set[str]
) -> list[Task]:
    return [
        task
        for task in TASKS
        if (
            not task_done(task, wandb_finished)
            and not own_task_processes(processes, task)
            and attempts(task) >= MAX_ATTEMPTS
        )
    ]


def status_payload(
    processes: list[ProcessInfo],
    wandb_finished: set[str] | None = None,
) -> dict[str, Any]:
    finished = wandb_finished or set()
    done = [task.run_name for task in TASKS if task_done(task, finished)]
    live = {
        task.run_name: [process.pid for process in own_task_processes(processes, task)]
        for task in TASKS
        if own_task_processes(processes, task)
    }
    gate_open, gate = tuning_gate(processes)
    return {
        "tasks": {
            "total": len(TASKS),
            "done": len(done),
            "live": len(live),
            "pending": len(TASKS) - len(done) - len(live),
            "done_names": done,
            "live_names": live,
        },
        "tuning_gate": {"open": gate_open, **gate},
        "foreign_coordinators": [
            {"pid": process.pid, "command": process.command}
            for process in foreign_coordinators(processes)
        ],
        "occupied_slots": {
            slot: [
                process.pid
                for process in constrained_slot_processes(processes, slot)
            ]
            for slot in SLOTS
            if constrained_slot_processes(processes, slot)
        },
        "attempts": {
            task.run_name: attempts(task) for task in TASKS if attempts(task)
        },
    }


def finalized_setup(
    hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    snapshots = {arm: study_snapshot(arm, include_best=True) for arm in ARMS}
    for arm, snapshot in snapshots.items():
        validate_best_snapshot(arm, snapshot)
    best_params_by_arm = {
        arm: snapshot["best"]["params"] for arm, snapshot in snapshots.items()
    }
    sample_commands = {}
    for task in TASKS[:4]:
        sample_commands[task.key] = shlex.join(
            build_command(task, "24-25", best_params_by_arm)
        )
    setup = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_hashes": hashes,
        "entity": ENTITY,
        "wandb_project": WANDB_PROJECT,
        "study_suffix": STUDY_SUFFIX,
        "tuning_complete_marker": str(TUNING_COMPLETE),
        "slots": list(SLOTS),
        "max_own_workers": MAX_OWN_WORKERS,
        "total_timesteps": TOTAL_TIMESTEPS,
        "eval_episodes": EVAL_EPISODES,
        "normalize_agent_reward": True,
        "initial_queries_policy": (
            "preserve best-trial fraction of the 100000 tuning budget"
        ),
        "best_studies": snapshots,
        "tasks": [
            {
                "arm": task.arm,
                "strategy": task.strategy,
                "seed": task.seed,
                "group": task.group,
                "pref_budget": task.pref_budget,
                "demo_budget": task.demo_budget,
                "demo_subsample_seed": task.demo_subsample_seed,
            }
            for task in TASKS
        ],
        "sample_commands": sample_commands,
    }
    return setup, best_params_by_arm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print reconstructed process/local state and exit without writing.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate files, task matrix, and current tuning gate without launching.",
    )
    args = parser.parse_args()

    if args.status:
        print(json.dumps(status_payload(iter_processes()), indent=2))
        return

    hashes = source_hashes()
    if len(TASKS) != 40 or len({task.run_name for task in TASKS}) != 40:
        raise SystemExit("Invalid or duplicate A/B task matrix")

    processes = iter_processes()
    gate_open, gate = tuning_gate(processes)
    if args.validate:
        payload: dict[str, Any] = {
            "source_hashes": hashes,
            "task_count": len(TASKS),
            "groups": sorted({task.group for task in TASKS}),
            "slots": list(SLOTS),
            "tuning_gate": {"open": gate_open, **gate},
        }
        if gate_open:
            setup, _ = finalized_setup(hashes)
            payload["finalized_setup"] = setup
        print(json.dumps(payload, indent=2))
        return

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (STATE_ROOT / "manager.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("hybrid Bernoulli final A/B manager already running")
    lock_stream.write(f"{os.getpid()}\n")
    lock_stream.flush()

    idle_streak = {slot: 0 for slot in SLOTS}
    wandb_finished: set[str] = set()
    last_wandb_refresh = 0.0
    last_status_log = 0.0
    setup_written = False
    best_params_by_arm: dict[str, dict[str, Any]] = {}
    log(
        f"START pid={os.getpid()} tasks={len(TASKS)} "
        f"slots={','.join(SLOTS)}"
    )

    while True:
        processes = iter_processes()
        gate_open, gate = tuning_gate(processes)
        if not gate_open:
            idle_streak = {slot: 0 for slot in SLOTS}
            payload = status_payload(processes, wandb_finished)
            atomic_json(STATE_ROOT / "status.json", payload)
            if time.time() - last_status_log >= 300 or last_status_log == 0:
                log("WAIT tuning gate " + json.dumps(gate, separators=(",", ":")))
                last_status_log = time.time()
            time.sleep(LOOP_SECONDS)
            continue

        if not setup_written:
            assert_sources_unchanged(hashes)
            setup, best_params_by_arm = finalized_setup(hashes)
            atomic_json(STATE_ROOT / "manifest.json", setup)
            setup_written = True
            log(
                "GATE OPEN: tuning complete; selected trials "
                + ", ".join(
                    f"{arm}=#{setup['best_studies'][arm]['best']['trial_number']}"
                    for arm in ARMS
                )
            )

        if time.time() - last_wandb_refresh >= WANDB_REFRESH_SECONDS:
            try:
                wandb_finished = wandb_finished_names()
            except Exception as exc:
                log(f"WARN W&B refresh failed: {exc}")
            last_wandb_refresh = time.time()

        processes = iter_processes()
        foreign = foreign_coordinators(processes)
        if foreign:
            payload = status_payload(processes, wandb_finished)
            atomic_json(STATE_ROOT / "status.json", payload)
            if time.time() - last_status_log >= 300 or last_status_log == 0:
                log(
                    "WAIT foreign coordinator(s): "
                    + ",".join(str(process.pid) for process in foreign)
                )
                last_status_log = time.time()
            time.sleep(LOOP_SECONDS)
            continue

        live_own = own_task_processes(processes)
        failures = failed_tasks(processes, wandb_finished)
        all_done = all(task_done(task, wandb_finished) for task in TASKS)
        if all_done and not live_own:
            payload = status_payload(processes, wandb_finished)
            atomic_json(
                STATE_ROOT / "COMPLETE.json",
                {
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": payload,
                },
            )
            log("COMPLETE: all 40 Bernoulli hybrid A/B runs finished")
            return
        if failures and not live_own and pending_task(processes, wandb_finished) is None:
            atomic_json(
                STATE_ROOT / "FAILED.json",
                {
                    "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tasks": [task.run_name for task in failures],
                },
            )
            log(
                "FAILED after retries: "
                + ",".join(task.run_name for task in failures)
            )
            raise SystemExit(1)

        cpu = slot_cpu_busy()
        processes = iter_processes()
        for slot in SLOTS:
            occupied = constrained_slot_processes(processes, slot)
            mean_busy, max_busy = cpu[slot]
            if (
                not occupied
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

            # Last-moment checks close the gap between CPU sampling and spawn.
            processes = iter_processes()
            gate_still_open, _ = tuning_gate(processes)
            if not gate_still_open:
                idle_streak = {candidate: 0 for candidate in SLOTS}
                break
            if foreign_coordinators(processes):
                break
            if live_task_count(processes) >= MAX_OWN_WORKERS:
                break
            if constrained_slot_processes(processes, slot):
                idle_streak[slot] = 0
                continue

            task = pending_task(processes, wandb_finished)
            if task is None:
                break
            assert_sources_unchanged(hashes)
            launch_task(task, slot, best_params_by_arm)
            idle_streak[slot] = 0
            launched = True
            # Let /proc expose the reservation before considering another pair.
            break

        payload = status_payload(iter_processes(), wandb_finished)
        atomic_json(STATE_ROOT / "status.json", payload)
        if not launched and (
            time.time() - last_status_log >= 300 or last_status_log == 0
        ):
            log("STATUS " + json.dumps(payload, separators=(",", ":")))
            last_status_log = time.time()
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    main()
