#!/usr/bin/env python3
"""Schedule normalized Bernoulli hybrid tuning without contending for cores.

The existing demo no-normalization pipeline owns cores 30-47 until it writes
its COMPLETE marker. While that pipeline is incomplete, this manager may use
only free pairs in 24-29. Afterwards it expands to every free pair in 24-47.

Each worker reserves one Optuna trial. State is reconstructed from the shared
journal and live processes, so restarting the manager does not duplicate the
30-trial target for either study.
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
STATE_ROOT = REPO_ROOT / "outputs" / "hybrid_bernoulli_tuning"
LOG_ROOT = STATE_ROOT / "logs"

ENTITY = "andrea02polimi-politecnico-di-milano"
WANDB_PROJECT = "tuning-thesis"
STUDY_SUFFIX = "_bernoulli_norm"
ARMS = ("hybrid_demo_1", "hybrid_demo_2")
TARGET_TRIALS = 30
MAX_WORKERS_PER_ARM = 3

PREF_BUDGET = 100_000
DEMO_BUDGET = 500
DEMO_SUBSAMPLE_SEED = 1000
TRAINING_SEED = 0
TOTAL_TIMESTEPS = 1_000_000
TIMESTEPS_PER_ITERATION = 20_000
N_ENVS = 2
EVAL_EPISODES = 20

SLOTS = tuple(f"{core}-{core + 1}" for core in range(24, 48, 2))
LEGACY_SLOTS = frozenset(f"{core}-{core + 1}" for core in range(30, 48, 2))
LEGACY_MANAGER = "scripts/schedule_demo_no_norm_pipeline.py"
LEGACY_COMPLETE = (
    REPO_ROOT / "outputs" / "demo_no_norm_pipeline" / "COMPLETE.json"
)

LOOP_SECONDS = 10
CPU_SAMPLE_SECONDS = 1
CPU_MEAN_BUSY = 35.0
CPU_MAX_BUSY = 70.0
IDLE_SCANS_REQUIRED = 2

EXPECTED_SOURCE_HASHES = {
    "scripts/tune_hybrid_sac.py": (
        "1c633334b601fbd3a4da9679053662aa1c6e3d6488a71aabb4ea4cc393235b42"
    ),
    "scripts/export_best_config.py": (
        "85229c11c98b8e0f90693e180e3ac9f5d7f98a23fc66f8db76fcdc3fe3a4a682"
    ),
}


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    state: str
    affinity: frozenset[int]
    command: str


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


def validate_sources() -> dict[str, str]:
    actual = {
        relative: sha256(REPO_ROOT / relative)
        for relative in EXPECTED_SOURCE_HASHES
    }
    mismatches = {
        relative: {"expected": EXPECTED_SOURCE_HASHES[relative], "actual": value}
        for relative, value in actual.items()
        if value != EXPECTED_SOURCE_HASHES[relative]
    }
    if mismatches:
        raise SystemExit(
            "Refusing to launch with unexpected source files: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return actual


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


def tuning_workers(
    processes: list[ProcessInfo], arm: str | None = None
) -> list[ProcessInfo]:
    workers = []
    for process in processes:
        if "scripts/tune_hybrid_sac.py" not in process.command:
            continue
        if option_value(process.command, "--study-suffix") != STUDY_SUFFIX:
            continue
        worker_arm = option_value(process.command, "--arm")
        if worker_arm in ARMS and (arm is None or worker_arm == arm):
            workers.append(process)
    return workers


def legacy_manager_pids(processes: list[ProcessInfo]) -> list[int]:
    return sorted(
        process.pid
        for process in processes
        if LEGACY_MANAGER in process.command
    )


def unknown_coordinators(processes: list[ProcessInfo]) -> list[ProcessInfo]:
    own_name = Path(__file__).name
    result = []
    for process in processes:
        command = process.command
        if process.pid == os.getpid() or own_name in command:
            continue
        if LEGACY_MANAGER in command:
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
        return {
            "name": name,
            "total": 0,
            "complete": 0,
            "pruned": 0,
            "failed": 0,
            "running": 0,
            "best": None,
            "preference_labels": None,
        }

    counts = {
        "complete": 0,
        "pruned": 0,
        "failed": 0,
        "running": 0,
    }
    complete_values = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            counts["complete"] += 1
            if trial.value is not None:
                complete_values.append(float(trial.value))
        elif trial.state == optuna.trial.TrialState.PRUNED:
            counts["pruned"] += 1
        elif trial.state == optuna.trial.TrialState.FAIL:
            counts["failed"] += 1
        elif trial.state in (
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.WAITING,
        ):
            counts["running"] += 1
    return {
        "name": name,
        "total": len(study.trials),
        **counts,
        "best": max(complete_values, default=None),
        "preference_labels": study.user_attrs.get("preference_labels"),
    }


def protected_slots(processes: list[ProcessInfo]) -> set[str]:
    # The marker is authoritative even if the legacy manager is temporarily
    # restarted: until completion, its entire configured range remains owned.
    protected = set()
    if not LEGACY_COMPLETE.exists():
        protected.update(LEGACY_SLOTS)
    for process in tuning_workers(processes):
        slot = option_value(process.command, "--cores")
        if slot in SLOTS:
            protected.add(slot)
    return protected


def build_worker_command(arm: str, slot: str) -> list[str]:
    return [
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
        "--preference-labels",
        "binary_bernoulli",
        "--pref-budget",
        str(PREF_BUDGET),
        "--demo-budget",
        str(DEMO_BUDGET),
        "--seed",
        str(TRAINING_SEED),
        "--total-timesteps",
        str(TOTAL_TIMESTEPS),
        "--timesteps-per-iteration",
        str(TIMESTEPS_PER_ITERATION),
        "--n-envs",
        str(N_ENVS),
        "--eval-episodes",
        str(EVAL_EPISODES),
        "--wandb-entity",
        ENTITY,
        "--wandb-project",
        WANDB_PROJECT,
        "--override",
        "algo.kwargs.normalize_agent_reward=true",
        "--override",
        f"run.demo_subsample_seed={DEMO_SUBSAMPLE_SEED}",
        "--override",
        f"wandb.tags=[optuna,{arm},bernoulli,normalized,fixed_demo_subsample_seed]",
    ]


def launch_worker(arm: str, slot: str) -> int:
    command = build_worker_command(arm, slot)
    log_path = LOG_ROOT / f"{arm}_{slot}.log"
    with log_path.open("a") as stream:
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
    log(f"START {arm} pid={process.pid} slot={slot}")
    return process.pid


def status_payload(processes: list[ProcessInfo]) -> dict[str, Any]:
    snapshots = {arm: study_snapshot(arm) for arm in ARMS}
    protected = protected_slots(processes)
    return {
        "studies": {
            arm: {
                **snapshots[arm],
                "workers": [
                    process.pid for process in tuning_workers(processes, arm)
                ],
            }
            for arm in ARMS
        },
        "legacy_pipeline": {
            "complete_marker": LEGACY_COMPLETE.exists(),
            "manager_pids": legacy_manager_pids(processes),
            "protected_slots": sorted(protected & LEGACY_SLOTS),
        },
        "unknown_coordinators": [
            {"pid": process.pid, "command": process.command}
            for process in unknown_coordinators(processes)
        ],
        "occupied_slots": {
            slot: [process.pid for process in constrained_slot_processes(processes, slot)]
            for slot in SLOTS
            if constrained_slot_processes(processes, slot)
        },
    }


def studies_complete(processes: list[ProcessInfo]) -> bool:
    if tuning_workers(processes):
        return False
    for arm in ARMS:
        snapshot = study_snapshot(arm)
        if snapshot["total"] < TARGET_TRIALS or snapshot["running"]:
            return False
    return True


def write_manifest(source_hashes: dict[str, str]) -> None:
    atomic_json(
        STATE_ROOT / "manifest.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_hashes": source_hashes,
            "cores": list(SLOTS),
            "legacy_reserved_cores": sorted(LEGACY_SLOTS),
            "legacy_complete_marker": str(LEGACY_COMPLETE),
            "entity": ENTITY,
            "wandb_project": WANDB_PROJECT,
            "arms": list(ARMS),
            "study_suffix": STUDY_SUFFIX,
            "target_trials_per_arm": TARGET_TRIALS,
            "max_workers_per_arm": MAX_WORKERS_PER_ARM,
            "preference_labels": "binary_bernoulli",
            "pref_temperature_search": "log-uniform [1, 50]",
            "initial_queries_search": "2%, 5%, 10%, 20% of pref budget",
            "pref_budget": PREF_BUDGET,
            "demo_budget": DEMO_BUDGET,
            "demo_subsample_seed": DEMO_SUBSAMPLE_SEED,
            "training_seed": TRAINING_SEED,
            "normalize_agent_reward": True,
            "total_timesteps": TOTAL_TIMESTEPS,
            "eval_episodes": EVAL_EPISODES,
        },
    )


def validate_setup() -> dict[str, Any]:
    hashes = validate_sources()
    snapshots = {arm: study_snapshot(arm) for arm in ARMS}
    incompatible = {
        arm: snapshot["preference_labels"]
        for arm, snapshot in snapshots.items()
        if snapshot["preference_labels"] not in (None, "binary_bernoulli")
    }
    if incompatible:
        raise SystemExit(
            "Refusing incompatible existing studies: "
            + json.dumps(incompatible, sort_keys=True)
        )
    return {
        "source_hashes": hashes,
        "studies": snapshots,
        "commands": {
            arm: shlex.join(build_worker_command(arm, "24-25"))
            for arm in ARMS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print reconstructed state and exit without writing or launching.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate sources, studies, and commands without launching.",
    )
    args = parser.parse_args()

    if args.status:
        print(json.dumps(status_payload(iter_processes()), indent=2))
        return
    if args.validate:
        print(json.dumps(validate_setup(), indent=2))
        return

    setup = validate_setup()
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    write_manifest(setup["source_hashes"])

    lock_stream = (STATE_ROOT / "manager.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("hybrid Bernoulli tuning manager already running")
    lock_stream.write(f"{os.getpid()}\n")
    lock_stream.flush()

    idle_streak = {slot: 0 for slot in SLOTS}
    last_status_log = 0.0
    log(
        f"START pid={os.getpid()} slots={','.join(SLOTS)} "
        f"target={TARGET_TRIALS} per arm"
    )

    while True:
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

        if studies_complete(processes):
            payload = status_payload(processes)
            atomic_json(
                STATE_ROOT / "COMPLETE.json",
                {
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": payload,
                },
            )
            log("COMPLETE: both normalized Bernoulli hybrid studies reached target")
            return

        protected = protected_slots(processes)
        cpu = slot_cpu_busy()
        processes = iter_processes()
        for slot in SLOTS:
            occupied = constrained_slot_processes(processes, slot)
            mean_busy, max_busy = cpu[slot]
            if (
                slot not in protected
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
            if slot in protected_slots(processes):
                idle_streak[slot] = 0
                continue
            if constrained_slot_processes(processes, slot):
                idle_streak[slot] = 0
                continue

            candidates = []
            for arm in ARMS:
                snapshot = study_snapshot(arm)
                workers = tuning_workers(processes, arm)
                if (
                    snapshot["total"] < TARGET_TRIALS
                    and len(workers) < MAX_WORKERS_PER_ARM
                ):
                    candidates.append((len(workers), snapshot["total"], arm))
            if not candidates:
                continue

            _, _, arm = min(candidates)
            launch_worker(arm, slot)
            idle_streak[slot] = 0
            launched = True
            # One reservation per loop lets the journal register the RUNNING
            # trial before the next worker is considered.
            break

        payload = status_payload(iter_processes())
        atomic_json(STATE_ROOT / "status.json", payload)
        if not launched and (
            time.time() - last_status_log >= 300 or last_status_log == 0
        ):
            log("STATUS " + json.dumps(payload, separators=(",", ":")))
            last_status_log = time.time()
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    main()
