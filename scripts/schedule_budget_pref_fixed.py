#!/usr/bin/env python3
"""Schedule fixed-demo budget runs and pref_bernoulli Optuna workers."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock


REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_ROOT = REPO_ROOT / "outputs" / "budget_pref_scheduler"
LOG_ROOT = STATE_ROOT / "logs"
JOURNAL = REPO_ROOT / "outputs" / "optuna" / "journal.log"

PREF_STUDY = "hybrid_sac_pref_bernoulli_q100k_temp"
PREF_TARGET = 30
PREF_SLOTS = ("30-31", "38-39", "44-45")
BUDGET_SLOTS = ("32-33", "34-35", "36-37", "40-41", "42-43", "46-47")
BUDGET_TASKS = tuple(
    (level, seed)
    for level in (2723, 1000, 500, 200, 100, 50)
    for seed in (1, 2, 3)
    if (level, seed) != (2723, 1)
)

WANDB_ENTITY = "andrea02polimi-politecnico-di-milano"
BUDGET_PROJECT = "tuning-thesis-fixed-demo-subseed"
PYTHON = "/home/fis3/miniconda3/envs/sumo-rlhf/bin/python"
LOOP_SECONDS = 10


@dataclass
class Running:
    kind: str
    slot: str
    process: subprocess.Popen
    name: str
    finals_before: set[Path] | None = None


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return ""


def proc_state(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2]
    except OSError:
        return None


def live_training_slots() -> dict[str, list[int]]:
    slots = {slot: [] for slot in (*PREF_SLOTS, *BUDGET_SLOTS)}
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        pid = int(item.name)
        if proc_state(pid) == "Z" or "train_hybrid_sac.py" not in proc_cmdline(pid):
            continue
        try:
            affinity = os.sched_getaffinity(pid)
        except (OSError, PermissionError):
            continue
        for slot in slots:
            lo, hi = (int(value) for value in slot.split("-"))
            if affinity.intersection(range(lo, hi + 1)):
                slots[slot].append(pid)
    return slots


def live_run_names() -> set[str]:
    names = set()
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or proc_state(int(item.name)) == "Z":
            continue
        cmd = proc_cmdline(int(item.name))
        if "train_hybrid_sac.py" not in cmd:
            continue
        for token in shlex.split(cmd):
            if token.startswith("run.name="):
                names.add(token.split("=", 1)[1])
    return names


def storage() -> JournalStorage:
    return JournalStorage(
        JournalFileBackend(
            str(JOURNAL), lock_obj=JournalFileOpenLock(str(JOURNAL))
        )
    )


def study() -> optuna.Study:
    return optuna.load_study(study_name=PREF_STUDY, storage=storage())


def finalize_legacy_trial(worker_pid: int, trial_number: int) -> bool:
    marker = STATE_ROOT / f"legacy_trial_{trial_number}.finalized"
    if marker.exists():
        return True

    trial_dir = (
        REPO_ROOT
        / "outputs"
        / "optuna"
        / "hybrid_sac_pref_bernoulli_q100k_temp"
        / f"trial_{trial_number:04d}"
    )
    live = any(
        f"trial_{trial_number:04d}" in proc_cmdline(int(item.name))
        and proc_state(int(item.name)) != "Z"
        for item in Path("/proc").iterdir()
        if item.name.isdigit()
    )
    if live:
        return False

    current_study = study()
    frozen = current_study.trials[trial_number]
    if frozen.state == optuna.trial.TrialState.RUNNING:
        eval_files = sorted(trial_dir.glob("*/final_eval.json"))
        if eval_files:
            metrics = json.loads(eval_files[-1].read_text())
            trial_id = frozen._trial_id
            for key, value in metrics.items():
                current_study._storage.set_trial_user_attr(trial_id, key, value)
            current_study._storage.set_trial_user_attr(
                trial_id, "run_dir", str(eval_files[-1].parent)
            )
            current_study.tell(
                trial_number, float(metrics["eval/mean_fast_return"])
            )
            log(
                f"FINALIZE legacy pref trial {trial_number}: "
                f"eval/mean_fast_return={metrics['eval/mean_fast_return']}"
            )
        else:
            current_study.tell(
                trial_number, state=optuna.trial.TrialState.FAIL
            )
            log(f"FAIL legacy pref trial {trial_number}: final_eval.json assente")

    cmd = proc_cmdline(worker_pid)
    if "tune_hybrid_sac.py" in cmd and "--arm pref_bernoulli" in cmd:
        os.kill(worker_pid, signal.SIGTERM)
        os.kill(worker_pid, signal.SIGCONT)
        log(f"STOP legacy Optuna worker pid={worker_pid}")
    marker.touch()
    return True


def export_demo_overrides() -> list[str]:
    output = subprocess.check_output(
        [
            PYTHON,
            "export_best_config.py",
            "--arm",
            "demo_2",
            "--format",
            "full",
            "--storage-path",
            "../outputs/optuna/journal.log",
        ],
        cwd=REPO_ROOT / "scripts",
        text=True,
    )
    return shlex.split(output)


def budget_final_files(level: int, seed: int) -> set[Path]:
    root = REPO_ROOT / "outputs" / "budget_curves" / f"budget_demo_2_{level}"
    return set(root.glob(f"budget_demo_2_{level}-seed{seed}*/final_eval.json"))


def launch_budget(level: int, seed: int, slot: str, overrides: list[str]) -> Running:
    group = f"budget_demo_2_{level}"
    run_name = f"{group}-seed{seed}"
    command = [
        "taskset",
        "-c",
        slot,
        PYTHON,
        "scripts/train_hybrid_sac.py",
        *overrides,
        f"run.n_expert_trajectories={level}",
        "run.demo_subsample_seed=1000",
        f"run.seed={seed}",
        f"run.output_dir=outputs/budget_curves/{group}",
        f"run.name={run_name}",
        f"run.group={group}",
        f"wandb.entity={WANDB_ENTITY}",
        f"wandb.project={BUDGET_PROJECT}",
        "wandb.tags=[budget_curve,demo_2,fixed_demo_subsample_seed]",
        "env.n_envs=2",
        "train.kwargs.total_timesteps=2000000",
        "train.kwargs.timesteps_per_iteration=20000",
        "algo.kwargs.normalize_agent_reward=false",
    ]
    log_path = LOG_ROOT / f"budget_{level}_seed{seed}.log"
    stream = open(log_path, "a")
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
    log(f"START budget {level}-seed{seed} pid={process.pid} slot={slot}")
    return Running(
        "budget", slot, process, f"{level}-seed{seed}", budget_final_files(level, seed)
    )


def launch_pref(slot: str) -> Running:
    command = [
        PYTHON,
        "scripts/tune_hybrid_sac.py",
        "--arm",
        "pref_bernoulli",
        "--n-trials",
        "1",
        "--cores",
        slot,
        "--pref-budget",
        "100000",
        "--study-suffix",
        "_q100k_temp",
        "--total-timesteps",
        "1000000",
    ]
    log_path = LOG_ROOT / f"pref_worker_{slot}.log"
    stream = open(log_path, "a")
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
    log(f"START pref worker pid={process.pid} slot={slot}")
    return Running("pref", slot, process, f"pref-{slot}")


def reap(running: dict[str, Running]) -> None:
    for slot, job in list(running.items()):
        returncode = job.process.poll()
        if returncode is None:
            continue
        del running[slot]
        if job.kind == "pref":
            log(f"END pref worker slot={slot} exit={returncode}")
            continue

        level_text, seed_text = job.name.split("-seed")
        level, seed = int(level_text), int(seed_text)
        new_finals = budget_final_files(level, seed) - (job.finals_before or set())
        marker = STATE_ROOT / f"budget_{level}_seed{seed}"
        if returncode == 0 and new_finals:
            marker.with_suffix(".done").touch()
            log(f"DONE budget {job.name} slot={slot}")
        else:
            marker.with_suffix(".failed").touch()
            log(f"FAIL budget {job.name} slot={slot} exit={returncode}")


def next_budget(live_names: set[str]) -> tuple[int, int] | None:
    for level, seed in BUDGET_TASKS:
        base = STATE_ROOT / f"budget_{level}_seed{seed}"
        run_name = f"budget_demo_2_{level}-seed{seed}"
        if base.with_suffix(".done").exists() or base.with_suffix(".failed").exists():
            continue
        if run_name not in live_names:
            return level, seed
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-worker-pid", type=int, default=1729765)
    parser.add_argument("--legacy-trial", type=int, default=8)
    args = parser.parse_args()

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(exist_ok=True)
    lock_stream = open(STATE_ROOT / "scheduler.lock", "w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("scheduler gia in esecuzione")
    lock_stream.write(str(os.getpid()))
    lock_stream.flush()

    overrides = export_demo_overrides()
    running: dict[str, Running] = {}
    legacy_done = False
    log(f"START scheduler pid={os.getpid()} pref_target={PREF_TARGET}")

    while True:
        reap(running)
        if not legacy_done:
            legacy_done = finalize_legacy_trial(
                args.legacy_worker_pid, args.legacy_trial
            )

        occupied = live_training_slots()
        for slot in running:
            occupied[slot].append(running[slot].process.pid)

        live_names = live_run_names()
        for slot in BUDGET_SLOTS:
            if occupied[slot] or slot in running:
                continue
            task = next_budget(live_names)
            if task is None:
                break
            level, seed = task
            running[slot] = launch_budget(level, seed, slot, overrides)
            live_names.add(f"budget_demo_2_{level}-seed{seed}")

        # Launch at most one worker per loop so its trial is reserved in the
        # shared journal before the next capacity decision.
        current_trials = len(study().trials)
        if current_trials < PREF_TARGET:
            for slot in PREF_SLOTS:
                if not occupied[slot] and slot not in running:
                    running[slot] = launch_pref(slot)
                    break

        budget_left = next_budget(live_run_names()) is not None
        pref_left = len(study().trials) < PREF_TARGET
        active = bool(running) or any(occupied.values())
        if not budget_left and not pref_left and not active and legacy_done:
            log("DONE scheduler: code completate")
            return
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    main()
