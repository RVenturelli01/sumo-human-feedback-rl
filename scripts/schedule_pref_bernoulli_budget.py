#!/usr/bin/env python3
"""Schedule pref_bernoulli budget curves after the q100k_temp tuning ends.

The scheduler is deliberately conservative around other launchers. It waits
for the 30-trial tuning study to have no live workers, selects the finished
W&B run with the largest ``sweep/mean_fast_return``, then dispatches the
standard 5 budget levels x 3 seeds onto idle two-core slots in 20-47.

It never signals or otherwise manages external processes. A slot must be
process-free, CPU-idle on two consecutive scans, and unreserved by a known
active orchestrator before a run can be started.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
JOURNAL = REPO_ROOT / "outputs" / "optuna" / "journal.log"
STATE_ROOT = REPO_ROOT / "outputs" / "pref_bernoulli_budget_scheduler"
LOG_ROOT = STATE_ROOT / "logs"
BEST_SNAPSHOT = STATE_ROOT / "best_wandb_run.json"

WANDB_ENTITY = "andrea02polimi-politecnico-di-milano"
TUNING_PROJECT = "tuning-thesis"
TUNING_GROUP = "tune_pref_bernoulli_q100k_temp"
TUNING_STUDY = "hybrid_sac_pref_bernoulli_q100k_temp"
TUNING_TARGET = 30

BUDGET_PROJECT = "tuning-thesis-pref-bernoulli-budget-curves"
BUDGET_LEVELS = (250000, 100000, 50000, 25000, 10000)
SEEDS = (1, 2, 3)
SLOTS = tuple(f"{core}-{core + 1}" for core in range(20, 48, 2))

# The active fixed-demo scheduler may immediately refill these slots. Keep
# them reserved until that scheduler exits, avoiding a check-then-launch race.
FIXED_SCHEDULER = "scripts/schedule_budget_pref_fixed.py"
FIXED_BUDGET_SLOTS = {
    "32-33", "34-35", "36-37", "40-41", "42-43", "46-47"
}
POST_TUNING_SCHEDULER = "scripts/post_tuning_orchestrator.py"

TOTAL_TIMESTEPS = 2_000_000
TIMESTEPS_PER_ITERATION = 20_000
N_ENVS = 2
EVAL_EPISODES = 20
LOOP_SECONDS = 10
CPU_SAMPLE_SECONDS = 2
CPU_MEAN_BUSY = 35.0
CPU_MAX_BUSY = 70.0
IDLE_SCANS_REQUIRED = 2
MAX_ATTEMPTS = 2
RETURN_METRIC = "sweep/mean_fast_return"


@dataclass(frozen=True)
class Task:
    level: int
    seed: int

    @property
    def group(self) -> str:
        return f"budget_pref_bernoulli_{self.level}"

    @property
    def run_name(self) -> str:
        return f"{self.group}-seed{self.seed}"

    @property
    def key(self) -> str:
        return f"{self.level}_seed{self.seed}"

    @property
    def output_root(self) -> Path:
        return REPO_ROOT / "outputs" / "pref_bernoulli_budget_curves" / self.group


@dataclass
class Running:
    task: Task
    pid: int
    slot: str
    process: subprocess.Popen | None = None


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    state: str
    affinity: frozenset[int]
    command: str


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def slot_cores(slot: str) -> frozenset[int]:
    lo, hi = (int(value) for value in slot.split("-"))
    return frozenset(range(lo, hi + 1))


def iter_processes() -> list[ProcessInfo]:
    processes = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        pid = int(item.name)
        try:
            state = (item / "stat").read_text().split()[2]
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


def command_is_live(processes: list[ProcessInfo], needle: str) -> bool:
    return any(needle in process.command for process in processes)


def tuning_workers(processes: list[ProcessInfo]) -> list[ProcessInfo]:
    return [
        process
        for process in processes
        if "tune_hybrid_sac.py" in process.command
        and "--arm pref_bernoulli" in process.command
        and "--study-suffix _q100k_temp" in process.command
    ]


def live_training_runs(processes: list[ProcessInfo]) -> dict[str, ProcessInfo]:
    runs = {}
    for process in processes:
        if "scripts/train_hybrid_sac.py" not in process.command:
            continue
        match = re.search(r"(?:^|\s)run\.name=([^\s]+)", process.command)
        if match:
            runs[match.group(1)] = process
    return runs


def process_alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (OSError, IndexError):
        return False


def study_snapshot() -> tuple[int, float | None]:
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    storage = JournalStorage(
        JournalFileBackend(
            str(JOURNAL), lock_obj=JournalFileOpenLock(str(JOURNAL))
        )
    )
    study = optuna.load_study(study_name=TUNING_STUDY, storage=storage)
    try:
        best_value = float(study.best_value)
    except ValueError:
        best_value = None
    return len(study.trials), best_value


def tuning_complete(processes: list[ProcessInfo]) -> tuple[bool, int, float | None]:
    count, best_value = study_snapshot()
    workers = tuning_workers(processes)
    return count >= TUNING_TARGET and not workers, count, best_value


def select_best_wandb_run(optuna_best: float | None) -> dict[str, Any] | None:
    import wandb

    api = wandb.Api(timeout=90)
    runs = api.runs(
        f"{WANDB_ENTITY}/{TUNING_PROJECT}",
        filters={"group": TUNING_GROUP, "state": "finished"},
        per_page=100,
    )
    candidates = []
    for run in runs:
        value = dict(run.summary).get(RETURN_METRIC)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            candidates.append((float(value), run.id))
    if not candidates:
        log("WAIT: nessuna run W&B finished con sweep/mean_fast_return")
        return None

    value, run_id = max(candidates)
    if optuna_best is None or not math.isclose(value, optuna_best, rel_tol=0, abs_tol=1e-6):
        log(
            "WAIT: best W&B e best Optuna non ancora allineati "
            f"({value} vs {optuna_best})"
        )
        return None

    run = api.run(f"{WANDB_ENTITY}/{TUNING_PROJECT}/{run_id}")
    return {
        "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entity": WANDB_ENTITY,
        "project": TUNING_PROJECT,
        "group": TUNING_GROUP,
        "run_id": run.id,
        "run_name": run.name,
        "url": run.url,
        "metric": RETURN_METRIC,
        "value": value,
        "commit": getattr(run, "commit", None),
        "config": dict(run.config),
    }


def load_or_select_best(optuna_best: float | None) -> dict[str, Any] | None:
    if BEST_SNAPSHOT.exists():
        return json.loads(BEST_SNAPSHOT.read_text())
    snapshot = select_best_wandb_run(optuna_best)
    if snapshot is None:
        return None
    BEST_SNAPSHOT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    log(
        f"BEST W&B congelata: {snapshot['run_name']} id={snapshot['run_id']} "
        f"{RETURN_METRIC}={snapshot['value']:.6f}"
    )
    return snapshot


def hydra_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    return json.dumps(value, separators=(",", ":"))


def flatten_config(value: Any, prefix: str) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        pairs = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            pairs.extend(flatten_config(child, path))
        return pairs
    return [(prefix, value)]


def best_overrides(snapshot: dict[str, Any]) -> list[str]:
    config = snapshot["config"]
    pairs = []
    for root in ("env", "agent", "algo", "train"):
        section = config.get(root)
        if isinstance(section, dict):
            pairs.extend(flatten_config(section, root))
    return [f"{key}={hydra_value(value)}" for key, value in pairs]


def task_final_files(task: Task) -> list[Path]:
    return sorted(task.output_root.glob(f"{task.run_name}*/final_eval.json"))


def marker(task: Task, suffix: str) -> Path:
    return STATE_ROOT / f"task_{task.key}.{suffix}"


def attempts(task: Task) -> int:
    path = marker(task, "attempts")
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return 0


def record_attempt(task: Task) -> int:
    value = attempts(task) + 1
    marker(task, "attempts").write_text(f"{value}\n")
    return value


def task_done(task: Task) -> bool:
    if marker(task, "done").exists():
        return True
    if task_final_files(task):
        marker(task, "done").touch()
        return True
    return False


def task_failed(task: Task) -> bool:
    return marker(task, "failed").exists()


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


def reserved_slots(processes: list[ProcessInfo], tuning_is_done: bool) -> set[str]:
    reserved = set()
    if command_is_live(processes, FIXED_SCHEDULER):
        reserved |= FIXED_BUDGET_SLOTS
        if not tuning_is_done:
            reserved |= {"30-31", "38-39", "44-45"}
    if command_is_live(processes, POST_TUNING_SCHEDULER):
        reserved |= {
            slot for slot in SLOTS if slot_cores(slot) & set(range(33, 48))
        }

    known = (
        FIXED_SCHEDULER,
        POST_TUNING_SCHEDULER,
        "scripts/schedule_pref_bernoulli_budget.py",
    )
    unknown_scheduler = any(
        ("scheduler" in process.command or "orchestrator" in process.command)
        and not any(name in process.command for name in known)
        for process in processes
    )
    if unknown_scheduler:
        reserved |= set(SLOTS)
    return reserved


def slot_has_constrained_process(
    slot: str,
    processes: list[ProcessInfo],
    baseline_affinity: frozenset[int],
) -> bool:
    cores = slot_cores(slot)
    return any(
        process.affinity != baseline_affinity and bool(process.affinity & cores)
        for process in processes
    )


class Scheduler:
    def __init__(self, project: str):
        self.project = project
        self.tasks = [Task(level, seed) for level in BUDGET_LEVELS for seed in SEEDS]
        self.running: dict[str, Running] = {}
        self.idle_scans = {slot: 0 for slot in SLOTS}
        self.baseline_affinity = frozenset(os.sched_getaffinity(0))
        self.snapshot: dict[str, Any] | None = None
        self._last_wait_message: tuple[int, int] | None = None

    def adopt(self, processes: list[ProcessInfo]) -> None:
        live = live_training_runs(processes)
        for task in self.tasks:
            process = live.get(task.run_name)
            if process is None or task.key in self.running:
                continue
            matching = [slot for slot in SLOTS if process.affinity == slot_cores(slot)]
            slot = matching[0] if matching else "unknown"
            self.running[task.key] = Running(task, process.pid, slot)
            log(f"ADOPT {task.run_name} pid={process.pid} slot={slot}")

    def reap(self) -> None:
        for key, running in list(self.running.items()):
            alive = (
                running.process.poll() is None
                if running.process is not None
                else process_alive(running.pid)
            )
            if alive:
                continue
            del self.running[key]
            if task_final_files(running.task):
                marker(running.task, "done").touch()
                log(f"DONE {running.task.run_name} slot={running.slot}")
            elif attempts(running.task) < MAX_ATTEMPTS:
                log(
                    f"RETRY {running.task.run_name}: output finale assente "
                    f"dopo attempt {attempts(running.task)}"
                )
            else:
                marker(running.task, "failed").touch()
                log(f"FAIL {running.task.run_name}: tentativi esauriti")

    def pending(self, live_names: set[str]) -> list[Task]:
        return [
            task
            for task in self.tasks
            if task.key not in self.running
            and task.run_name not in live_names
            and not task_done(task)
            and not task_failed(task)
        ]

    def launch(self, task: Task, slot: str, processes: list[ProcessInfo]) -> bool:
        if slot_has_constrained_process(slot, processes, self.baseline_affinity):
            self.idle_scans[slot] = 0
            return False

        assert self.snapshot is not None
        initial_queries = int(
            self.snapshot["config"].get("algo", {}).get("kwargs", {}).get(
                "initial_queries", 500
            )
        )
        initial_queries = min(initial_queries, task.level // 5)
        command = [
            "taskset", "-c", slot,
            sys.executable,
            "scripts/train_hybrid_sac.py",
            *best_overrides(self.snapshot),
            f"algo.kwargs.total_queries={task.level}",
            f"train.kwargs.total_queries={task.level}",
            f"algo.kwargs.initial_queries={initial_queries}",
            f"run.seed={task.seed}",
            f"run.output_dir={task.output_root.relative_to(REPO_ROOT)}",
            f"run.name={task.run_name}",
            f"run.group={task.group}",
            f"wandb.entity={WANDB_ENTITY}",
            f"wandb.project={self.project}",
            "wandb.tags=[budget_curve,pref_bernoulli,q100k_temp_best]",
            f"env.n_envs={N_ENVS}",
            f"eval.n_episodes={EVAL_EPISODES}",
            f"train.kwargs.total_timesteps={TOTAL_TIMESTEPS}",
            f"train.kwargs.timesteps_per_iteration={TIMESTEPS_PER_ITERATION}",
        ]

        task.output_root.mkdir(parents=True, exist_ok=True)
        attempt = record_attempt(task)
        log_path = LOG_ROOT / f"{task.run_name}.attempt{attempt}.log"
        stream = open(log_path, "a")
        stream.write(
            f"=== {time.strftime('%F %T')} slot={slot}\n{shlex.join(command)}\n"
        )
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
        stream.close()
        self.running[task.key] = Running(task, process.pid, slot, process)
        self.idle_scans[slot] = 0
        log(
            f"START {task.run_name} pid={process.pid} slot={slot} "
            f"queries={task.level} initial_queries={initial_queries}"
        )
        return True

    def loop(self) -> None:
        while True:
            processes = iter_processes()
            self.adopt(processes)
            self.reap()

            done, trial_count, optuna_best = tuning_complete(processes)
            workers = len(tuning_workers(processes))
            if not done:
                status = (trial_count, workers)
                if status != self._last_wait_message:
                    log(
                        f"WAIT tuning: trials={trial_count}/{TUNING_TARGET} "
                        f"workers={workers}"
                    )
                    self._last_wait_message = status
                time.sleep(LOOP_SECONDS)
                continue

            if self.snapshot is None:
                self.snapshot = load_or_select_best(optuna_best)
                if self.snapshot is None:
                    time.sleep(LOOP_SECONDS)
                    continue

            live_names = set(live_training_runs(processes))
            pending = self.pending(live_names)
            if not pending and not self.running:
                failed = [task.run_name for task in self.tasks if task_failed(task)]
                log(
                    "DONE scheduler: tutte le budget curve processate"
                    + (f"; fallite={failed}" if failed else "")
                )
                return

            reserved = reserved_slots(processes, tuning_is_done=True)
            cpu = slot_cpu_busy()
            processes = iter_processes()
            for slot in SLOTS:
                busy = slot_has_constrained_process(
                    slot, processes, self.baseline_affinity
                )
                mean_busy, max_busy = cpu[slot]
                idle = (
                    slot not in reserved
                    and not busy
                    and mean_busy < CPU_MEAN_BUSY
                    and max_busy < CPU_MAX_BUSY
                )
                self.idle_scans[slot] = self.idle_scans[slot] + 1 if idle else 0

            for slot in SLOTS:
                if not pending or self.idle_scans[slot] < IDLE_SCANS_REQUIRED:
                    continue
                # Refresh /proc immediately before Popen to minimize races with
                # manual launchers and other schedulers.
                latest = iter_processes()
                if slot in reserved_slots(latest, tuning_is_done=True):
                    self.idle_scans[slot] = 0
                    continue
                if self.launch(pending[0], slot, latest):
                    pending.pop(0)

            time.sleep(LOOP_SECONDS)


def print_status(project: str) -> None:
    processes = iter_processes()
    done, count, best = tuning_complete(processes)
    reserved = reserved_slots(processes, done)
    print(
        json.dumps(
            {
                "project": project,
                "tuning_trials": count,
                "tuning_target": TUNING_TARGET,
                "tuning_workers": [process.pid for process in tuning_workers(processes)],
                "tuning_complete": done,
                "optuna_best": best,
                "reserved_slots": sorted(reserved),
                "live_runs": {
                    name: {
                        "pid": process.pid,
                        "affinity": sorted(process.affinity),
                    }
                    for name, process in live_training_runs(processes).items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", default=BUDGET_PROJECT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print_status(args.project)
        return

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(exist_ok=True)
    lock_stream = open(STATE_ROOT / "scheduler.lock", "a+")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("schedule_pref_bernoulli_budget gia in esecuzione")
    lock_stream.seek(0)
    lock_stream.truncate()
    lock_stream.write(f"{os.getpid()}\n")
    lock_stream.flush()

    log(
        f"START scheduler pid={os.getpid()} project={args.project} "
        f"tasks={len(BUDGET_LEVELS) * len(SEEDS)} slots={','.join(SLOTS)}"
    )
    Scheduler(args.project).loop()


if __name__ == "__main__":
    main()
