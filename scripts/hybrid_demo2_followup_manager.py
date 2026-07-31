#!/usr/bin/env python3
"""Run hybrid Demo 2 follow-ups beside the Bernoulli tuning manager.

Priority queue:
1. complete the historical soft-label A/B groups with seeds 6-10;
2. run three-seed pilots with the new Demo 2 budget.

The Bernoulli tuning manager always gets first choice of capacity. While it is
live, this manager launches only when all six expected tuning workers are
present, and only on pairs that remain idle for two consecutive scans.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
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
STATE_ROOT = REPO_ROOT / "outputs" / "hybrid_demo2_followups"
LOG_ROOT = STATE_ROOT / "logs"

ENTITY = "andrea02polimi-politecnico-di-milano"
WANDB_PROJECT = "thesis"
ARM = "hybrid_demo_2"
TUNING_MANAGER = "scripts/hybrid_bernoulli_tuning_manager.py"
TUNING_STUDY_SUFFIX = "_bernoulli_norm"
TUNING_TARGET_WORKERS = 6

SLOTS = tuple(f"{core}-{core + 1}" for core in range(24, 48, 2))
MAX_OWN_WORKERS = 6
LOOP_SECONDS = 10
CPU_SAMPLE_SECONDS = 1
CPU_MEAN_BUSY = 35.0
CPU_MAX_BUSY = 70.0
IDLE_SCANS_REQUIRED = 2
MAX_ATTEMPTS = 2
WANDB_REFRESH_SECONDS = 300

TOTAL_TIMESTEPS = 2_000_000
TIMESTEPS_PER_ITERATION = 20_000
N_ENVS = 2
EVAL_EPISODES = 20

EXPECTED_BEST_PARAMS = {
    "lr_rew": 0.00046178781677192805,
    "gradient_steps_rew": 147,
    "l2_rew": 1.981463066451472e-05,
    "reward_net_arch": "[128,128]",
    "initial_agent_timesteps": 20_000,
    "batch_size_pref": 256,
    "batch_size_expert": 128,
    "demo_weight": 9.662826870596577,
}


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    state: str
    affinity: frozenset[int]
    command: str


@dataclass(frozen=True)
class Task:
    phase: str
    variant: str
    seed: int
    pref_budget: int
    demo_budget: int
    group: str
    demo_subsample_seed: int | None

    @property
    def run_name(self) -> str:
        return f"{self.group}-seed{self.seed}"

    @property
    def key(self) -> str:
        return f"{self.phase}_{self.variant}_seed{self.seed}"

    @property
    def output_parent(self) -> Path:
        return REPO_ROOT / "outputs" / "final" / self.group


def build_tasks() -> list[Task]:
    tasks = []
    # Alternating variants keeps partial A/B statistics balanced.
    for seed in range(6, 11):
        tasks.extend(
            [
                Task("old", "A", seed, 250, 25, "hybrid_demo_2_A", None),
                Task("old", "B", seed, 500, 50, "hybrid_demo_2_B", None),
            ]
        )
    for seed in range(1, 4):
        paired_subsample_seed = 1000 + seed
        tasks.extend(
            [
                Task(
                    "pilot",
                    "A",
                    seed,
                    250,
                    250,
                    "hybrid_demo_2_A_d250",
                    paired_subsample_seed,
                ),
                Task(
                    "pilot",
                    "B",
                    seed,
                    500,
                    500,
                    "hybrid_demo_2_B_d500",
                    paired_subsample_seed,
                ),
            ]
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
    result = []
    for process in processes:
        if "scripts/tune_hybrid_sac.py" not in process.command:
            continue
        if option_value(process.command, "--study-suffix") != TUNING_STUDY_SUFFIX:
            continue
        result.append(process)
    return result


def own_task_processes(
    processes: list[ProcessInfo], task: Task | None = None
) -> list[ProcessInfo]:
    names = {item.run_name for item in TASKS} if task is None else {task.run_name}
    return [
        process
        for process in processes
        if any(f"run.name={name}" in process.command for name in names)
    ]


def constrained_slot_processes(
    processes: list[ProcessInfo], slot: str
) -> list[ProcessInfo]:
    cores = slot_cores(slot)
    return [
        process
        for process in processes
        if len(process.affinity) <= 8 and process.affinity & cores
    ]


def unknown_coordinators(processes: list[ProcessInfo]) -> list[ProcessInfo]:
    own_name = Path(__file__).name
    result = []
    for process in processes:
        command = process.command
        if process.pid == os.getpid() or own_name in command:
            continue
        if TUNING_MANAGER in command:
            continue
        if "python" not in command:
            continue
        if "schedule_" in command or "orchestrator" in command:
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


def run_checked(command: list[str]) -> str:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def exported_overrides(pref_budget: int, demo_budget: int) -> list[str]:
    output = run_checked(
        [
            PYTHON,
            "scripts/export_best_config.py",
            "--arm",
            ARM,
            "--format",
            "full",
            "--storage-path",
            str(JOURNAL),
            "--pref-budget",
            str(pref_budget),
            "--demo-budget",
            str(demo_budget),
        ]
    )
    return shlex.split(output)


def best_params() -> dict[str, Any]:
    output = run_checked(
        [
            PYTHON,
            "scripts/export_best_config.py",
            "--arm",
            ARM,
            "--format",
            "params",
            "--storage-path",
            str(JOURNAL),
        ]
    )
    return json.loads(output)


def validate_best_params(params: dict[str, Any]) -> None:
    if set(params) != set(EXPECTED_BEST_PARAMS):
        raise SystemExit(
            "Refusing changed hybrid_demo_2 best-param keys: "
            + json.dumps(params, sort_keys=True)
        )
    for key, expected in EXPECTED_BEST_PARAMS.items():
        actual = params[key]
        if isinstance(expected, float):
            matches = math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=0)
        else:
            matches = actual == expected
        if not matches:
            raise SystemExit(
                f"Refusing changed hybrid_demo_2 best param {key}: "
                f"expected {expected!r}, got {actual!r}"
            )


def override_map(overrides: list[str]) -> dict[str, str]:
    result = {}
    for override in overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            result[key] = value
    return result


def validate_overrides(task: Task, overrides: list[str]) -> None:
    values = override_map(overrides)
    expected = {
        "algo.kwargs.labels_type": "soft",
        "algo.kwargs.total_queries": str(task.pref_budget),
        "train.kwargs.total_queries": str(task.pref_budget),
        "run.n_expert_trajectories": str(task.demo_budget),
        "algo.kwargs.normalize_agent_reward": "true",
        "algo.kwargs.pref_temperature": "20.0",
        "algo.kwargs.demo_weight": str(EXPECTED_BEST_PARAMS["demo_weight"]),
    }
    mismatches = {
        key: {"expected": value, "actual": values.get(key)}
        for key, value in expected.items()
        if values.get(key) != value
    }
    if mismatches:
        raise SystemExit(
            f"Refusing invalid overrides for {task.key}: "
            + json.dumps(mismatches, sort_keys=True)
        )


def task_final_files(task: Task) -> list[Path]:
    if not task.output_parent.exists():
        return []
    return sorted(
        task.output_parent.glob(f"{task.run_name}*/final_eval.json")
    )


def task_done(task: Task, wandb_finished: set[str] | None = None) -> bool:
    return bool(task_final_files(task)) or (
        wandb_finished is not None and task.run_name in wandb_finished
    )


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


def build_command(task: Task, slot: str) -> list[str]:
    overrides = exported_overrides(task.pref_budget, task.demo_budget)
    validate_overrides(task, overrides)
    tags = (
        "[final,hybrid_demo_2,soft,historical_completion]"
        if task.phase == "old"
        else "[final,pilot,hybrid_demo_2,soft,new_demo_budget,normalized]"
    )
    command = [
        "taskset",
        "-c",
        slot,
        PYTHON,
        "scripts/train_hybrid_sac.py",
        *overrides,
        f"run.seed={task.seed}",
    ]
    if task.demo_subsample_seed is not None:
        command.append(f"run.demo_subsample_seed={task.demo_subsample_seed}")
    command += [
        f"run.output_dir=outputs/final/{task.group}",
        f"run.name={task.run_name}",
        f"run.group={task.group}",
        f"wandb.entity={ENTITY}",
        f"wandb.project={WANDB_PROJECT}",
        f"wandb.tags={tags}",
        f"env.n_envs={N_ENVS}",
        f"eval.n_episodes={EVAL_EPISODES}",
        f"train.kwargs.total_timesteps={TOTAL_TIMESTEPS}",
        f"train.kwargs.timesteps_per_iteration={TIMESTEPS_PER_ITERATION}",
        "algo.kwargs.normalize_agent_reward=true",
    ]
    return command


def launch_task(task: Task, slot: str) -> int:
    task.output_parent.mkdir(parents=True, exist_ok=True)
    attempt = record_attempt(task)
    command = build_command(task, slot)
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
        f"pref={task.pref_budget} demo={task.demo_budget}"
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
        if attempts(task) >= MAX_ATTEMPTS:
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
    processes: list[ProcessInfo], wandb_finished: set[str] | None = None
) -> dict[str, Any]:
    finished = wandb_finished or set()
    done = [task.run_name for task in TASKS if task_done(task, finished)]
    live = {
        task.run_name: [process.pid for process in own_task_processes(processes, task)]
        for task in TASKS
        if own_task_processes(processes, task)
    }
    return {
        "tasks": {
            "total": len(TASKS),
            "done": len(done),
            "live": len(live),
            "pending": len(TASKS) - len(done) - len(live),
            "done_names": done,
            "live_names": live,
        },
        "tuning": {
            "manager_live": tuning_manager_live(processes),
            "worker_count": len(tuning_workers(processes)),
            "worker_pids": [process.pid for process in tuning_workers(processes)],
        },
        "unknown_coordinators": [
            {"pid": process.pid, "command": process.command}
            for process in unknown_coordinators(processes)
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
            task.run_name: attempts(task)
            for task in TASKS
            if attempts(task)
        },
    }


def validate_setup() -> dict[str, Any]:
    if not JOURNAL.exists():
        raise SystemExit(f"Missing Optuna journal: {JOURNAL}")
    params = best_params()
    validate_best_params(params)
    commands = {}
    for task in (
        TASKS[0],
        TASKS[1],
        next(task for task in TASKS if task.group == "hybrid_demo_2_A_d250"),
        next(task for task in TASKS if task.group == "hybrid_demo_2_B_d500"),
    ):
        command = build_command(task, "28-29")
        commands[task.key] = shlex.join(command)
    source_hashes = {
        relative: sha256(REPO_ROOT / relative)
        for relative in (
            "scripts/train_hybrid_sac.py",
            "scripts/export_best_config.py",
            "scripts/tune_hybrid_sac.py",
        )
    }
    return {
        "best_params": params,
        "source_hashes": source_hashes,
        "commands": commands,
        "tasks": [
            {
                "phase": task.phase,
                "variant": task.variant,
                "seed": task.seed,
                "group": task.group,
                "pref_budget": task.pref_budget,
                "demo_budget": task.demo_budget,
                "demo_subsample_seed": task.demo_subsample_seed,
            }
            for task in TASKS
        ],
    }


def write_manifest(setup: dict[str, Any]) -> None:
    atomic_json(
        STATE_ROOT / "manifest.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "entity": ENTITY,
            "wandb_project": WANDB_PROJECT,
            "slots": list(SLOTS),
            "max_own_workers": MAX_OWN_WORKERS,
            "tuning_manager": TUNING_MANAGER,
            "tuning_target_workers": TUNING_TARGET_WORKERS,
            "priority": ["historical_completion", "new_budget_pilot"],
            "total_timesteps": TOTAL_TIMESTEPS,
            "eval_episodes": EVAL_EPISODES,
            "normalize_agent_reward": True,
            **setup,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print reconstructed local/process state and exit without writing.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the best config and generated commands without launching.",
    )
    args = parser.parse_args()

    if args.status:
        print(json.dumps(status_payload(iter_processes()), indent=2))
        return

    setup = validate_setup()
    if args.validate:
        print(json.dumps(setup, indent=2))
        return

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    write_manifest(setup)

    lock_stream = (STATE_ROOT / "manager.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("hybrid Demo 2 follow-up manager already running")
    lock_stream.write(f"{os.getpid()}\n")
    lock_stream.flush()

    idle_streak = {slot: 0 for slot in SLOTS}
    wandb_finished: set[str] = set()
    last_wandb_refresh = 0.0
    last_status_log = 0.0
    log(
        f"START pid={os.getpid()} tasks={len(TASKS)} "
        f"slots={','.join(SLOTS)}"
    )

    while True:
        if time.time() - last_wandb_refresh >= WANDB_REFRESH_SECONDS:
            try:
                wandb_finished = wandb_finished_names()
            except Exception as exc:
                log(f"WARN W&B refresh failed: {exc}")
            last_wandb_refresh = time.time()

        processes = iter_processes()
        foreign = unknown_coordinators(processes)
        if foreign:
            if time.time() - last_status_log >= 300 or last_status_log == 0:
                log(
                    "WAIT unknown coordinator(s): "
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
            log("COMPLETE: historical A/B and three-seed pilots finished")
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

        manager_live = tuning_manager_live(processes)
        worker_count = len(tuning_workers(processes))
        tuning_has_priority = manager_live and worker_count < TUNING_TARGET_WORKERS

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
        if not tuning_has_priority:
            for slot in SLOTS:
                if idle_streak[slot] < IDLE_SCANS_REQUIRED:
                    continue

                processes = iter_processes()
                if tuning_manager_live(processes) and (
                    len(tuning_workers(processes)) < TUNING_TARGET_WORKERS
                ):
                    break
                if len(own_task_processes(processes)) >= MAX_OWN_WORKERS:
                    break
                if constrained_slot_processes(processes, slot):
                    idle_streak[slot] = 0
                    continue

                task = pending_task(processes, wandb_finished)
                if task is None:
                    break
                launch_task(task, slot)
                idle_streak[slot] = 0
                launched = True
                # Make the reservation visible before considering another slot.
                break

        payload = status_payload(iter_processes(), wandb_finished)
        atomic_json(STATE_ROOT / "status.json", payload)
        if not launched and (
            time.time() - last_status_log >= 300 or last_status_log == 0
        ):
            reason = (
                "tuning priority"
                if tuning_has_priority
                else "capacity/tasks"
            )
            log(f"STATUS reason={reason} " + json.dumps(payload, separators=(",", ":")))
            last_status_log = time.time()
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    main()
