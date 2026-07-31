#!/usr/bin/env python3
"""Resync completed pipeline runs whose online W&B filestream failed.

The watcher is intentionally independent from the experiment scheduler. It
never signals a process and never reads a transaction file while its training
run is alive. A run is eligible only when:

* its W&B internal log contains a fatal filestream upload error;
* its training process is no longer present;
* its local ``final_eval.json`` exists.

Eligible runs are appended to their existing W&B run IDs one at a time.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import wandb


REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = REPO_ROOT / "outputs" / "demo_no_norm_pipeline"
RUN_ROOT = PIPELINE_ROOT / "runs" / "budget"
STATE_ROOT = PIPELINE_ROOT / "wandb_resync"
STATE_PATH = STATE_ROOT / "state.json"
MAIN_COMPLETE = PIPELINE_ROOT / "COMPLETE.json"

WANDB_BIN = "/home/fis3/miniconda3/envs/sumo-rlhf/bin/wandb"
ENTITY = "andrea02polimi-politecnico-di-milano"
PROJECT = "tuning-thesis-demo-no-norm-budget-curves"
RETURN_METRIC = "sweep/mean_fast_return"

POLL_SECONDS = 60
RETRY_SECONDS = 10 * 60
SYNC_TIMEOUT_SECONDS = 30 * 60
FATAL_NEEDLE = "filestream: fatal error"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"runs": {}}


def process_commands() -> list[str]:
    commands = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            state = (item / "stat").read_text().split()[2]
            command = (
                (item / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
                .strip()
            )
        except (OSError, IndexError):
            continue
        if state != "Z" and command:
            commands.append(command)
    return commands


def run_is_live(run_name: str, commands: list[str]) -> bool:
    return any(
        "scripts/train_hybrid_sac.py" in command
        and f"run.name={run_name}" in command
        for command in commands
    )


def run_id(wandb_file: Path) -> str:
    match = re.fullmatch(r"run-([a-zA-Z0-9]+)\.wandb", wandb_file.name)
    if not match:
        raise ValueError(f"unexpected W&B transaction filename: {wandb_file}")
    return match.group(1)


def has_fatal_upload(wandb_file: Path) -> bool:
    internal_log = wandb_file.parent / "logs" / "debug-internal.log"
    try:
        return FATAL_NEEDLE in internal_log.read_text(errors="replace")
    except OSError:
        return False


def final_eval_path(wandb_file: Path) -> Path:
    return wandb_file.parents[2] / "final_eval.json"


def run_name(wandb_file: Path) -> str:
    return wandb_file.parents[2].name


def remote_snapshot(identifier: str) -> dict[str, Any]:
    run = wandb.Api(timeout=90).run(f"{ENTITY}/{PROJECT}/{identifier}")
    summary = dict(run.summary)
    return {
        "id": identifier,
        "name": run.name,
        "state": run.state,
        "return": summary.get(RETURN_METRIC),
        "success": summary.get("sweep/success_rate"),
        "step": summary.get("_step"),
    }


def remote_complete(snapshot: dict[str, Any]) -> bool:
    return snapshot["state"] == "finished" and snapshot["return"] is not None


def wait_for_remote_complete(identifier: str) -> dict[str, Any]:
    snapshot = remote_snapshot(identifier)
    for _ in range(5):
        if remote_complete(snapshot):
            return snapshot
        time.sleep(10)
        snapshot = remote_snapshot(identifier)
    return snapshot


def discover() -> list[Path]:
    return sorted(RUN_ROOT.glob("**/wandb/run-*/run-*.wandb"))


def sync_run(wandb_file: Path, identifier: str) -> subprocess.CompletedProcess:
    command = [
        WANDB_BIN,
        "sync",
        "--append",
        "--include-online",
        "--no-mark-synced",
        "--id",
        identifier,
        "--entity",
        ENTITY,
        "--project",
        PROJECT,
        str(wandb_file),
    ]
    log(f"SYNC {run_name(wandb_file)} id={identifier}")
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=SYNC_TIMEOUT_SECONDS,
        check=False,
    )


def main() -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = open(STATE_ROOT / "watcher.lock", "w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("W&B resync watcher already running")
    lock_stream.write(f"{os.getpid()}\n")
    lock_stream.flush()

    state = load_state()
    log(f"START pid={os.getpid()} project={PROJECT}")

    while True:
        commands = process_commands()
        eligible = []
        pending = []

        for wandb_file in discover():
            if not has_fatal_upload(wandb_file):
                continue
            identifier = run_id(wandb_file)
            name = run_name(wandb_file)
            record = state["runs"].setdefault(identifier, {"run_name": name})

            if record.get("status") == "finished":
                continue
            if run_is_live(name, commands) or not final_eval_path(wandb_file).exists():
                pending.append(name)
                continue
            if time.time() < float(record.get("retry_after", 0)):
                pending.append(name)
                continue
            eligible.append((wandb_file, identifier, record))

        if eligible:
            wandb_file, identifier, record = eligible[0]
            try:
                before = remote_snapshot(identifier)
                if remote_complete(before):
                    record.update(
                        status="finished",
                        verified_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                        remote=before,
                    )
                    log(
                        f"ALREADY FINISHED {run_name(wandb_file)} id={identifier}"
                    )
                else:
                    result = sync_run(wandb_file, identifier)
                    record["attempts"] = int(record.get("attempts", 0)) + 1
                    record["last_output"] = result.stdout[-4000:]
                    record["last_exit_code"] = result.returncode
                    if result.returncode != 0:
                        raise RuntimeError(
                            f"wandb sync exited with {result.returncode}"
                        )
                    after = wait_for_remote_complete(identifier)
                    if not remote_complete(after):
                        raise RuntimeError(
                            f"remote run incomplete after sync: {after}"
                        )
                    record.update(
                        status="finished",
                        verified_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                        remote=after,
                    )
                    log(
                        f"DONE {run_name(wandb_file)} id={identifier} "
                        f"return={after['return']}"
                    )
            except Exception as error:
                # Network/API failures are expected here; keep the watcher
                # alive and retry without touching any training process.
                record.update(
                    status="retry",
                    last_error=str(error),
                    retry_after=time.time() + RETRY_SECONDS,
                )
                log(
                    f"RETRY {run_name(wandb_file)} id={identifier}: {error}"
                )
            atomic_json(STATE_PATH, state)
            time.sleep(POLL_SECONDS)
            continue

        atomic_json(STATE_PATH, state)
        if MAIN_COMPLETE.exists() and not pending:
            log("COMPLETE: pipeline ended and all fatal uploads are synchronized")
            return
        if pending:
            log(f"WAIT active/incomplete: {','.join(sorted(pending))}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
