#!/usr/bin/env python3
"""Two-phase campaign for the hybrid arms at HOMOGENEOUS feedback budgets.

Why this exists
---------------
The published budget curves for ``hybrid_demo_2_soft`` and
``hybrid_demo_2_bernoulli`` sit below every baseline. Three concrete causes,
all verified in the code:

1. ``_hybrid_point()`` in schedule_budget_curves_completion.py hardcodes
   ``demo_weight=1.0``, discarding the tuned values (9.66 soft / 3.41 bern).
2. ``demo_weight`` is not a loss coefficient but a gradient-norm ratio
   (hybrid_algorithm.py): 1.0 means "make the demo gradient exactly as loud as
   the preference gradient" -- with noisy Bernoulli labels, as loud as noise.
3. The Bernoulli hybrid ran at ``pref_temperature=25.1`` while the
   ``pref_bernoulli`` baseline it is plotted against ran at 3.06. Since that
   temperature lives ONLY in the synthetic oracle (gatherers.py), the two arms
   faced annotators of very different noise levels and are not comparable.

Plus the tuning itself used lopsided budgets (5000 pref / 500 demo, and
100k / 500), so of course the optimum boosted the under-represented channel.

What this manager does
----------------------
Phase 1 -- two concurrent Optuna lanes, 12 single-core workers each, 30 trials
each, at homogeneous budgets, with ``demo_weight`` and ``pref_temperature``
PINNED (and therefore removed from the search space):

    bern : hybrid_demo_2 / binary_bernoulli / 2723 pref + 2723 demo / T=3.0595
    soft : hybrid_demo_2 / soft             /  500 pref +  500 demo / T=20.0

Phase 2 -- budget curves from the winners, 3 seeds, 2M timesteps, one wave:

    bern : (2723,2723) (1000,1000) (100,100)
    soft : (2723,2723) (100,100) (10,10) (1,1)

Design notes that are load-bearing
----------------------------------
* Warm starts are enqueued ONCE by this manager under its own lock. Optuna's
  ``enqueue_trial`` docstring warns it "might produce duplicated trials if
  called simultaneously by multiple processes"; 12 workers starting together
  would race. Workers therefore never receive ``--enqueue-params``.
* WAITING is counted separately from RUNNING. ``enqueue_trial`` creates
  WAITING trials, and the budget must be counted on terminal states only,
  otherwise queued warm starts inflate the total and the lane stops early.
* The orphan reaper never kills a tuning worker from the outside: that would
  leave the trial RUNNING forever (JournalStorage has no heartbeat) and
  deadlock the handover, and would orphan the trainer, which lives in its own
  session. Collapse/timeout pruning happens INSIDE tune_hybrid_sac.py.
* The reaper identifies its target through ``worker_token`` and the
  ``trial_runtime.json`` written by the tuner; without that record it refuses
  to guess which process to kill.
* ``Scheduler`` is constructed with the OLD task matrix and used only for
  ``launch()``: its ``__init__`` validates the transition state against the
  matrix and every other method iterates ``self.tasks``. Reaping and
  bookkeeping for the 21 new tasks are done here.

Usage
-----
    python scripts/hybrid_hom_campaign_manager.py --validate
    python scripts/hybrid_hom_campaign_manager.py --status
    python scripts/hybrid_hom_campaign_manager.py --dry-run-phase2
    nohup python scripts/hybrid_hom_campaign_manager.py \
        > outputs/hybrid_hom_campaign/manager.log 2>&1 &
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optuna  # noqa: E402
from optuna.trial import TrialState  # noqa: E402

from scripts import schedule_budget_curves_completion as budget_runner  # noqa: E402
from scripts.tune_hybrid_sac import initial_queries_choices  # noqa: E402

PYTHON = str(budget_runner.PYTHON)
TUNE_SCRIPT = "scripts/tune_hybrid_sac.py"
JOURNAL = budget_runner.JOURNAL
ENTITY = budget_runner.WANDB_ENTITY
TUNING_WANDB_PROJECT = "tuning-thesis"

STATE_ROOT = REPO_ROOT / "outputs" / "hybrid_hom_campaign"
LOG_ROOT = STATE_ROOT / "logs"
LOCK_PATH = STATE_ROOT / "manager.lock"
STATE_PATH = STATE_ROOT / "state.json"
STATUS_PATH = STATE_ROOT / "status.json"
MANIFEST_PATH = STATE_ROOT / "manifest.json"
PHASE2_SOURCES_PATH = STATE_ROOT / "phase2_sources.json"
COMPLETE_PATH = STATE_ROOT / "COMPLETE.json"

SLOTS = budget_runner.SINGLE_SLOTS  # cores 24..47, one run per core
LOOP_SECONDS = budget_runner.LOOP_SECONDS
CPU_MEAN_BUSY = budget_runner.CPU_MEAN_BUSY
CPU_MAX_BUSY = budget_runner.CPU_MAX_BUSY
IDLE_SCANS_REQUIRED = budget_runner.IDLE_SCANS_REQUIRED
MIN_FREE_BYTES = budget_runner.MIN_FREE_BYTES

TUNING_TOTAL_TIMESTEPS = 1_000_000
TUNING_TIMESTEPS_PER_ITERATION = 20_000
N_ENVS = budget_runner.N_ENVS
EVAL_EPISODES = budget_runner.EVAL_EPISODES
DEMO_SUBSAMPLE_SEED_TUNING = 1000
TUNING_SEED = 0

PHASE_TUNING = "tuning"
PHASE_CURVES = "curves"
PHASE_COMPLETE = "complete"
PHASE_BLOCKED = "blocked"          # health / infrastructure fault
PHASE_NEEDS_REVIEW = "needs_review"  # campaign healthy, result too weak

# Handover health gates.
MAX_FAIL_RATIO = 0.25
MIN_COMPLETE = 8
WARN_COMPLETE = 12
GREEN_BEST_VALUE = 59.0

REAP_SCANS_REQUIRED = 2
KILL_GRACE_SECONDS = 20

# Scripts whose content this manager depends on. Recomputed after the
# phase-0 edits; a mismatch means someone changed the contract underneath.
PINNED_SOURCES = (
    "scripts/tune_hybrid_sac.py",
    "scripts/export_best_config.py",
    "scripts/schedule_budget_curves_completion.py",
)


# --------------------------------------------------------------------------
# Lane definitions
# --------------------------------------------------------------------------

BERN_TEMPERATURE = 3.0595414013726767  # == the pref_bernoulli baseline oracle
SOFT_TEMPERATURE = 20.0                # == the pref_soft baseline oracle
DEMO_WEIGHT = 1.0

LANES: dict[str, dict[str, Any]] = {
    "bern": {
        "arm": "hybrid_demo_2",
        "study_suffix": "_hom_bern",
        "labels": "binary_bernoulli",
        "pref_budget": 2723,
        "demo_budget": 2723,
        "pref_temperature": BERN_TEMPERATURE,
        "curve_arm": "hybrid_demo_2_bern_hom",
        "curve_budgets": (2723, 1000, 100),
        # initial_queries is searched for this lane; the tuned fraction is
        # carried to every curve level (resolved at handover).
        "initial_fraction": None,
        "tags": "[optuna,hybrid_demo_2,bernoulli,hom,fixed_demo_weight]",
    },
    "soft": {
        "arm": "hybrid_demo_2",
        "study_suffix": "_hom_soft",
        "labels": "soft",
        "pref_budget": 500,
        "demo_budget": 500,
        "pref_temperature": SOFT_TEMPERATURE,
        "curve_arm": "hybrid_demo_2_soft_hom",
        "curve_budgets": (2723, 100, 10, 1),
        # arm_overrides pins initial_queries = max(100, 0.1*500) = 100 for this
        # lane, i.e. 20% of the tuning budget. Carry that validated fraction.
        "initial_fraction": 0.20,
        "tags": "[optuna,hybrid_demo_2,soft,hom,fixed_demo_weight]",
    },
}

# Warm starts. Previous winners, stripped of demo_weight/pref_temperature
# (no longer sampled: leaving them in would make the enqueued dict diverge
# from the realised params). The Bernoulli winner used initial_queries=10000
# at a 100k budget, i.e. 10%; both the fraction-preserving remap (272) and the
# 20% option (545) are enqueued so the question is settled empirically.
WARM_STARTS: dict[str, list[dict[str, Any]]] = {
    "bern": [
        {
            "lr_rew": 0.002432388041110311,
            "gradient_steps_rew": 28,
            "l2_rew": 0.0005634790500586056,
            "reward_net_arch": "[8,8]",
            "initial_agent_timesteps": 20000,
            "batch_size_pref": 128,
            "initial_queries": 272,
            "batch_size_expert": 128,
        },
        {
            "lr_rew": 0.002432388041110311,
            "gradient_steps_rew": 28,
            "l2_rew": 0.0005634790500586056,
            "reward_net_arch": "[8,8]",
            "initial_agent_timesteps": 20000,
            "batch_size_pref": 128,
            "initial_queries": 545,
            "batch_size_expert": 128,
        },
        # demo_2_no_norm winner, partial: batch_size_pref left to be sampled.
        {
            "lr_rew": 0.0009187069964354143,
            "gradient_steps_rew": 100,
            "l2_rew": 5.061862748858848e-06,
            "reward_net_arch": "[64,64]",
            "initial_agent_timesteps": 20000,
            "batch_size_expert": 16,
        },
    ],
    "soft": [
        {
            "lr_rew": 0.00046178781677192805,
            "gradient_steps_rew": 147,
            "l2_rew": 1.981463066451472e-05,
            "reward_net_arch": "[128,128]",
            "initial_agent_timesteps": 20000,
            "batch_size_pref": 256,
            "batch_size_expert": 128,
        },
        {
            "lr_rew": 0.0009187069964354143,
            "gradient_steps_rew": 100,
            "l2_rew": 5.061862748858848e-06,
            "reward_net_arch": "[64,64]",
            "initial_agent_timesteps": 20000,
            "batch_size_expert": 16,
        },
    ],
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def study_name(lane: str) -> str:
    config = LANES[lane]
    return f"hybrid_sac_{config['arm']}{config['study_suffix']}"


def warm_hash(lane: str) -> str:
    payload = json.dumps(WARM_STARTS[lane], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def storage():
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    path = str(JOURNAL)
    return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


def load_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, ValueError):
        return {"phase": PHASE_TUNING}


def save_state(phase: str, **extra: Any) -> None:
    budget_runner.atomic_json(
        STATE_PATH,
        {
            "phase": phase,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **extra,
        },
    )


# --------------------------------------------------------------------------
# Optuna views
# --------------------------------------------------------------------------

def validate_warm_starts() -> None:
    """Reject warm dicts that would make Optuna fail the trial outright.

    A categorical value outside its choices raises inside
    CategoricalDistribution.to_internal_repr BEFORE the containment check, so
    the trial FAILs and burns a slot.
    """
    for lane, entries in WARM_STARTS.items():
        allowed = initial_queries_choices(LANES[lane]["pref_budget"])
        for index, entry in enumerate(entries):
            for forbidden in ("demo_weight", "pref_temperature"):
                if forbidden in entry:
                    raise SystemExit(
                        f"warm[{lane}][{index}] contains {forbidden}, which is "
                        f"pinned and no longer sampled."
                    )
            queries = entry.get("initial_queries")
            if queries is None:
                continue
            if lane != "bern":
                raise SystemExit(
                    f"warm[{lane}][{index}] sets initial_queries but that lane "
                    f"does not search it."
                )
            if queries not in allowed:
                raise SystemExit(
                    f"warm[{lane}][{index}] initial_queries={queries} not in "
                    f"{allowed}; the trial would fail on enqueue."
                )


def open_study(lane: str, create: bool = False):
    name = study_name(lane)
    if create:
        return optuna.create_study(
            study_name=name, storage=storage(), direction="maximize",
            load_if_exists=True,
        )
    return optuna.load_study(study_name=name, storage=storage())


def snapshot(lane: str) -> dict[str, Any]:
    """Per-state counts plus the RUNNING trials with their worker tokens.

    WAITING is deliberately NOT folded into RUNNING: enqueued warm starts sit
    in WAITING, and the trial budget must be counted on terminal states only.
    """
    try:
        study = open_study(lane)
    except KeyError:
        return {
            "study": study_name(lane), "exists": False,
            "complete": 0, "pruned": 0, "fail": 0, "running": 0, "waiting": 0,
            "terminal": 0, "best_value": None, "best_number": None,
            "running_trials": [], "prune_reasons": {},
        }
    counts = {"complete": 0, "pruned": 0, "fail": 0, "running": 0, "waiting": 0}
    running_trials = []
    prune_reasons: dict[str, int] = {}
    best_value, best_number, best_params = None, None, None
    for trial in study.trials:
        if trial.state == TrialState.COMPLETE:
            counts["complete"] += 1
            if trial.value is not None and (best_value is None or trial.value > best_value):
                best_value, best_number = trial.value, trial.number
                best_params = dict(trial.params)
        elif trial.state == TrialState.PRUNED:
            counts["pruned"] += 1
            reason = str(trial.user_attrs.get("prune_reason", "unknown"))
            prune_reasons[reason] = prune_reasons.get(reason, 0) + 1
        elif trial.state == TrialState.FAIL:
            counts["fail"] += 1
        elif trial.state == TrialState.RUNNING:
            counts["running"] += 1
            running_trials.append(
                {
                    "trial_id": trial._trial_id,
                    "number": trial.number,
                    "worker_token": trial.user_attrs.get("worker_token"),
                }
            )
        elif trial.state == TrialState.WAITING:
            counts["waiting"] += 1
    terminal = counts["complete"] + counts["pruned"] + counts["fail"]
    return {
        "study": study_name(lane), "exists": True, **counts,
        "terminal": terminal,
        "best_value": best_value, "best_number": best_number,
        "best_params": best_params,
        "running_trials": running_trials,
        "prune_reasons": prune_reasons,
    }


def warm_start_init(lane: str) -> dict[str, Any]:
    """Enqueue warm starts exactly once, restart-safe.

    Not "enqueue then assert N WAITING": after a manager restart some warm
    trials may already be RUNNING/COMPLETE/PRUNED. Match on the enqueued
    payload (system_attrs["fixed_params"]) across ALL states instead.
    """
    study = open_study(lane, create=True)
    existing: list[dict[str, Any]] = [
        trial.system_attrs.get("fixed_params")
        for trial in study.trials
        if trial.system_attrs.get("fixed_params")
    ]
    enqueued, already, duplicates = 0, 0, []
    for entry in WARM_STARTS[lane]:
        matches = [item for item in existing if item == entry]
        if len(matches) > 1:
            duplicates.append(entry)
        elif len(matches) == 1:
            already += 1
        else:
            study.enqueue_trial(entry, skip_if_exists=True)
            enqueued += 1
    if duplicates:
        return {"lane": lane, "duplicates": duplicates, "ok": False}
    stored = study.user_attrs.get("warm_start_hash")
    if stored is None:
        study.set_user_attr("warm_start_hash", warm_hash(lane))
    elif stored != warm_hash(lane):
        return {"lane": lane, "warm_hash_mismatch": stored, "ok": False}
    return {"lane": lane, "enqueued": enqueued, "already": already, "ok": True}


# --------------------------------------------------------------------------
# Processes
# --------------------------------------------------------------------------

TOKEN_RE = re.compile(r"--worker-token\s+(\S+)")


def tuning_workers(processes) -> dict[str, Any]:
    """{worker_token: ProcessInfo} for live workers of THIS campaign."""
    result = {}
    for process in processes:
        if TUNE_SCRIPT not in process.command:
            continue
        if not any(
            config["study_suffix"] in process.command for config in LANES.values()
        ):
            continue
        match = TOKEN_RE.search(process.command)
        if match:
            result[match.group(1)] = process
    return result


def lane_of_worker(process) -> str | None:
    for lane, config in LANES.items():
        if f"--study-suffix {config['study_suffix']}" in process.command:
            return lane
    return None


def orphan_trainers(processes, workers: dict[str, Any]) -> list[Any]:
    """Trainers of this campaign whose parent worker is gone."""
    worker_pids = {process.pid for process in workers.values()}
    orphans = []
    for process in processes:
        if "scripts/train_hybrid_sac.py" not in process.command:
            continue
        if not any(
            f"run.group=tune_hybrid_demo_2{config['study_suffix']}" in process.command
            for config in LANES.values()
        ):
            continue
        if process.ppid not in worker_pids and process.ppid == 1:
            orphans.append(process)
    return orphans


def terminate_group(pgid: int) -> bool:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return True
        deadline = time.time() + KILL_GRACE_SECONDS
        while time.time() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return True
            time.sleep(1)
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    return False


def trial_runtime(lane: str, number: int) -> dict[str, Any]:
    path = (
        REPO_ROOT / "outputs" / "optuna" / study_name(lane)
        / f"trial_{number:04d}" / "trial_runtime.json"
    )
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


class Reaper:
    """Fail RUNNING trials whose worker died, after cleaning their trainer.

    Any worker death (OOM, SIGKILL, node hiccup) leaves a RUNNING trial that
    the journal never cleans up, which blocks the `running == 0` handover
    gate forever. The trainer also survives -- it is in its own session -- and
    keeps holding a core.
    """

    def __init__(self) -> None:
        self.absent: dict[tuple[str, int], int] = {}
        self.blocked_reason: str | None = None

    def scan(self, snapshots: dict[str, dict], workers: dict[str, Any]) -> list[dict]:
        actions = []
        live_tokens = set(workers)
        seen: set[tuple[str, int]] = set()
        for lane, snap in snapshots.items():
            for entry in snap["running_trials"]:
                token, number = entry["worker_token"], entry["number"]
                key = (lane, number)
                seen.add(key)
                if token is not None and token in live_tokens:
                    self.absent.pop(key, None)
                    continue
                self.absent[key] = self.absent.get(key, 0) + 1
                if self.absent[key] < REAP_SCANS_REQUIRED:
                    continue
                actions.append(self._reap(lane, entry))
        for key in list(self.absent):
            if key not in seen:
                del self.absent[key]
        return [action for action in actions if action]

    def _reap(self, lane: str, entry: dict) -> dict | None:
        number, trial_id = entry["number"], entry["trial_id"]
        record = trial_runtime(lane, number)
        if not record:
            # Never guess which process to kill on a shared machine.
            self.blocked_reason = (
                f"trial_runtime.json missing for {study_name(lane)} "
                f"trial {number}; refusing to guess the trainer process."
            )
            log(f"BLOCKED {self.blocked_reason}")
            return None
        pgid = record.get("trainer_pgid")
        killed = terminate_group(int(pgid)) if pgid else True
        if not killed:
            self.blocked_reason = (
                f"could not terminate trainer pgid={pgid} of "
                f"{study_name(lane)} trial {number}."
            )
            log(f"BLOCKED {self.blocked_reason}")
            return None

        # Re-read: the worker may have finished between the scan and now,
        # which would make RUNNING -> FAIL an illegal transition.
        store = storage()
        try:
            fresh = store.get_trial(trial_id)
        except (KeyError, ValueError) as error:
            log(f"REAP skip trial {number}: {error}")
            return None
        if fresh.state != TrialState.RUNNING:
            log(f"REAP skip trial {number}: now {fresh.state.name}")
            self.absent.pop((lane, number), None)
            return None
        outcome: Any
        try:
            # The bool means "state changed", not "succeeded"; verify by re-read.
            outcome = store.set_trial_state_values(trial_id, state=TrialState.FAIL)
        except Exception as error:  # noqa: BLE001 - storage raises many types
            log(f"REAP failed to fail trial {number}: {error!r}")
            return None
        after = store.get_trial(trial_id).state.name
        self.absent.pop((lane, number), None)
        log(
            f"REAP {study_name(lane)} trial {number} (trial_id={trial_id}) "
            f"RUNNING -> {after} [returned {outcome!r}] pgid={pgid}"
        )
        return {
            "lane": lane, "trial_number": number, "trial_id": trial_id,
            "state_after": after, "returned": outcome, "trainer_pgid": pgid,
        }


# --------------------------------------------------------------------------
# Phase 1: tuning
# --------------------------------------------------------------------------

def worker_command(lane: str, slot: str, token: str) -> list[str]:
    config = LANES[lane]
    return [
        PYTHON, TUNE_SCRIPT,
        "--arm", config["arm"],
        "--n-trials", "1",
        "--cores", slot,
        "--study-suffix", config["study_suffix"],
        "--preference-labels", config["labels"],
        "--pref-budget", str(config["pref_budget"]),
        "--demo-budget", str(config["demo_budget"]),
        "--seed", str(TUNING_SEED),
        "--total-timesteps", str(TUNING_TOTAL_TIMESTEPS),
        "--timesteps-per-iteration", str(TUNING_TIMESTEPS_PER_ITERATION),
        "--n-envs", str(N_ENVS),
        "--eval-episodes", str(EVAL_EPISODES),
        "--trial-timeout", str(6 * 3600),
        "--wandb-entity", ENTITY,
        "--wandb-project", TUNING_WANDB_PROJECT,
        "--fix-demo-weight", repr(DEMO_WEIGHT),
        "--fix-pref-temperature", repr(config["pref_temperature"]),
        "--worker-token", token,
        "--override", "algo.kwargs.normalize_agent_reward=true",
        "--override", f"run.demo_subsample_seed={DEMO_SUBSAMPLE_SEED_TUNING}",
        "--override", f"wandb.tags={config['tags']}",
    ]


def launch_worker(lane: str, slot: str) -> dict[str, Any]:
    token = f"{lane}-{slot}-{uuid.uuid4().hex[:8]}"
    command = worker_command(lane, slot, token)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"tune_{lane}_{slot}.log"
    environment = dict(os.environ)
    environment.pop("WANDB_DISABLED", None)
    environment.update(
        {
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg", "WANDB_MODE": "online",
        }
    )
    with log_path.open("a") as stream:
        stream.write(f"\n=== {time.strftime('%F %T')} slot={slot} token={token}\n")
        stream.write(" ".join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=environment, text=True,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
        )
    log(f"START tuning lane={lane} slot={slot} pid={process.pid} token={token}")
    return {"lane": lane, "slot": slot, "pid": process.pid, "token": token}


def pick_lane(snapshots: dict[str, dict], workers: dict[str, Any],
              target: int, max_per_lane: int) -> str | None:
    """Choose the lane that is furthest behind and still has budget."""
    live_per_lane: dict[str, int] = {lane: 0 for lane in LANES}
    for process in workers.values():
        lane = lane_of_worker(process)
        if lane:
            live_per_lane[lane] += 1
    candidates = []
    for lane, snap in snapshots.items():
        live = live_per_lane[lane]
        if live >= max_per_lane:
            continue
        # Workers that started but have not registered a trial yet are not in
        # `running`; count them so the lane cannot overshoot its budget.
        pending = max(live - snap["running"], 0)
        if snap["terminal"] + snap["running"] + pending >= target:
            continue
        candidates.append((live, snap["terminal"] + snap["running"], lane))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


# --------------------------------------------------------------------------
# Handover
# --------------------------------------------------------------------------

def handover_ready(snapshots: dict[str, dict], workers: dict[str, Any],
                   orphans: list[Any], target: int) -> tuple[bool, list[str]]:
    reasons = []
    for lane, snap in snapshots.items():
        if snap["terminal"] < target:
            reasons.append(f"{lane}: terminal {snap['terminal']}/{target}")
        if snap["running"]:
            reasons.append(f"{lane}: {snap['running']} RUNNING")
        if snap["waiting"]:
            reasons.append(f"{lane}: {snap['waiting']} WAITING")
    if workers:
        reasons.append(f"{len(workers)} tuning workers alive")
    if orphans:
        reasons.append(f"{len(orphans)} orphan trainers alive")
    return (not reasons), reasons


def health_verdict(snapshots: dict[str, dict], min_best: float) -> dict[str, Any]:
    problems, warnings = [], []
    for lane, snap in snapshots.items():
        terminal = max(snap["terminal"], 1)
        ratio = snap["fail"] / terminal
        if ratio > MAX_FAIL_RATIO:
            problems.append(
                f"{lane}: FAIL ratio {ratio:.0%} > {MAX_FAIL_RATIO:.0%} "
                f"({snap['fail']}/{snap['terminal']})"
            )
        if snap["complete"] < MIN_COMPLETE:
            problems.append(f"{lane}: only {snap['complete']} COMPLETE trials")
        if snap["best_value"] is None:
            problems.append(f"{lane}: no best value")
        elif snap["complete"] < WARN_COMPLETE:
            warnings.append(
                f"{lane}: only {snap['complete']} COMPLETE (< {WARN_COMPLETE}); "
                f"best may be fragile"
            )
    if problems:
        return {"verdict": PHASE_BLOCKED, "problems": problems, "warnings": warnings}
    for lane, snap in snapshots.items():
        value = snap["best_value"]
        if value < min_best:
            problems.append(f"{lane}: best {value:.2f} < {min_best:.2f}")
        elif value < GREEN_BEST_VALUE:
            warnings.append(
                f"{lane}: best {value:.2f} below the green line "
                f"{GREEN_BEST_VALUE:.2f}; proceeding"
            )
    if problems:
        return {"verdict": PHASE_NEEDS_REVIEW, "problems": problems, "warnings": warnings}
    return {"verdict": PHASE_CURVES, "problems": [], "warnings": warnings}


def freeze_phase2_sources(snapshots: dict[str, dict]) -> dict[str, Any]:
    payload = {}
    for lane, snap in snapshots.items():
        config = LANES[lane]
        fraction = config["initial_fraction"]
        if fraction is None:
            tuned = snap["best_params"].get("initial_queries")
            if tuned is None:
                raise RuntimeError(f"{lane}: best trial has no initial_queries.")
            fraction = tuned / config["pref_budget"]
        payload[lane] = {
            "study": snap["study"],
            "trial_number": snap["best_number"],
            "value": snap["best_value"],
            "params": snap["best_params"],
            "initial_fraction": fraction,
            "pref_temperature": config["pref_temperature"],
            "demo_weight": DEMO_WEIGHT,
            "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    return payload


# --------------------------------------------------------------------------
# Phase 2: budget curves
# --------------------------------------------------------------------------

def inject_config_sources(sources: dict[str, Any]) -> None:
    """Register the new arms before the first (lru_cached) export_overrides."""
    for lane, record in sources.items():
        config = LANES[lane]
        key = config["curve_arm"]
        budget_runner.CONFIG_SOURCES[key] = budget_runner.ConfigSource(
            key=key,
            arm=config["arm"],
            study_suffix=config["study_suffix"],
            expected_trial=record["trial_number"],
            preference_labels=config["labels"],
            expected_params=record["params"],
        )


def build_points(sources: dict[str, Any]) -> tuple[Any, ...]:
    points = []
    for lane, record in sources.items():
        config = LANES[lane]
        fraction = record["initial_fraction"]
        for budget in config["curve_budgets"]:
            initial = max(1, min(budget, round(fraction * budget)))
            points.append(
                budget_runner.Point(
                    arm=config["curve_arm"],
                    budget=budget * 2,  # group level = pref + demo
                    source_key=config["curve_arm"],
                    pref_budget=budget,
                    demo_budget=budget,
                    initial_queries=initial,
                    normalize_agent_reward=True,
                    labels_type=config["labels"],
                    loss_type="demo_2",
                    query_schedule="constant",
                    fragmenter_type="active",
                    pref_temperature=config["pref_temperature"],
                    demo_weight=DEMO_WEIGHT,
                )
            )
    return tuple(points)


def build_curve_tasks(sources: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        budget_runner.Task(point, seed)
        for point in build_points(sources)
        for seed in budget_runner.SEEDS
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def build_status(target: int) -> dict[str, Any]:
    processes = budget_runner.iter_processes()
    workers = tuning_workers(processes)
    snapshots = {lane: snapshot(lane) for lane in LANES}
    orphans = orphan_trainers(processes, workers)
    state = load_state()
    status = {
        "phase": state.get("phase", PHASE_TUNING),
        "target_trials": target,
        "lanes": snapshots,
        "workers": [
            {"token": token, "pid": p.pid, "lane": lane_of_worker(p),
             "cores": sorted(p.affinity)}
            for token, p in sorted(workers.items())
        ],
        "orphan_trainers": [{"pid": p.pid, "cmd": p.command[:120]} for p in orphans],
        "zombie_suspect": {
            lane: snap["running"] for lane, snap in snapshots.items()
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if PHASE2_SOURCES_PATH.exists():
        try:
            status["phase2_sources"] = json.loads(PHASE2_SOURCES_PATH.read_text())
        except ValueError:
            pass
    if state.get("phase") == PHASE_CURVES:
        sources = json.loads(PHASE2_SOURCES_PATH.read_text())
        tasks = build_curve_tasks(sources)
        status["curves"] = {
            "total": len(tasks),
            "done": sum(1 for task in tasks if budget_runner.task_done(task, False)),
            "groups": sorted({task.group for task in tasks}),
        }
    return status


class CurveRunner:
    """Own reap/bookkeeping loop for the 21 new tasks.

    Scheduler.__init__ validates the transition state against the OLD task
    matrix and every other Scheduler method iterates self.tasks, so the
    scheduler is built with the old matrix and used only for launch().
    """

    def __init__(self, tasks, max_attempts: int) -> None:
        self.tasks = tasks
        self.max_attempts = max_attempts
        matrix = budget_runner.build_tasks()
        self.scheduler = budget_runner.Scheduler(
            matrix,
            max_parallel=len(SLOTS),
            max_attempts=max_attempts,
            loop_seconds=LOOP_SECONDS,
            transition_state=budget_runner.load_transition_state(matrix),
        )
        self.running: dict[str, Any] = {}
        self.parked: dict[str, str] = {}
        self.started_at: dict[str, float] = {}

    def pending(self):
        for task in self.tasks:
            if task.key in self.running or task.key in self.parked:
                continue
            if budget_runner.task_done(task, update_marker=False):
                continue
            if budget_runner.attempts(task) >= self.max_attempts:
                self.parked[task.key] = "max attempts reached"
                continue
            yield task

    def launch(self, task, slot: str, processes) -> bool:
        if not self.scheduler.launch(task, slot, processes):
            return False
        self.running[task.key] = self.scheduler.running.pop(task.key)
        return True

    def reap(self, processes, max_run_hours: float, collapse_action: str) -> None:
        now = time.time()
        for key, running in list(self.running.items()):
            return_code = running.process.poll() if running.process else None
            main_alive = running.process is not None and return_code is None
            tree_alive = (
                running.pgid is not None
                and running.sid is not None
                and budget_runner.run_process_tree_alive(
                    processes,
                    root_pid=running.pid,
                    pgid=running.pgid,
                    sid=running.sid,
                    slot=running.slot,
                )
            )
            if main_alive or tree_alive:
                elapsed_h = (now - self.started_at.get(key, now)) / 3600.0
                if elapsed_h > max_run_hours:
                    # A collapsed run takes ~27h instead of ~7h and would blow
                    # the deadline. Kill the whole group: SUMO children live in
                    # the same session and would otherwise keep the core.
                    log(
                        f"CAP {running.task.run_name} exceeded "
                        f"{max_run_hours}h; terminating group "
                        f"pgid={running.pgid} (no retry)"
                    )
                    if running.pgid:
                        terminate_group(int(running.pgid))
                    budget_runner.write_marker(
                        running.task, "failed",
                        attempts=self.max_attempts,
                        reason=f"exceeded max_run_hours={max_run_hours}",
                    )
                    self.parked[key] = "wall-clock cap"
                    del self.running[key]
                    self.started_at.pop(key, None)
                continue
            del self.running[key]
            self.started_at.pop(key, None)
            done = budget_runner.mark_done_after_exit(
                running.task, return_code, running.slot
            )
            log(
                f"EXIT {running.task.run_name} rc={return_code} "
                f"done={done} slot={running.slot}"
            )
            if not done:
                budget_runner.write_marker(
                    running.task, "failed",
                    attempts=budget_runner.attempts(running.task),
                    return_code=return_code,
                )

    def all_done(self) -> bool:
        return not self.running and not list(self.pending())


def free_slots(processes, idle_scans: dict[str, int]) -> list[str]:
    """Slots with no process pinned to them AND quiet for enough scans."""
    busy = budget_runner.slot_cpu_busy()
    result = []
    for slot in SLOTS:
        if budget_runner.constrained_slot_processes(processes, slot):
            idle_scans[slot] = 0
            continue
        mean, peak = busy.get(slot, (0.0, 0.0))
        if mean > CPU_MEAN_BUSY or peak > CPU_MAX_BUSY:
            idle_scans[slot] = 0
            continue
        idle_scans[slot] = idle_scans.get(slot, 0) + 1
        if idle_scans[slot] >= IDLE_SCANS_REQUIRED:
            result.append(slot)
    return result


def run_loop(args) -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_file = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"Another manager holds {LOCK_PATH}.")
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()

    validate_warm_starts()
    import shutil

    if shutil.disk_usage(REPO_ROOT).free < MIN_FREE_BYTES:
        raise SystemExit("Less than 10 GiB free on the repo filesystem.")

    budget_runner.atomic_json(
        MANIFEST_PATH,
        {
            "lanes": LANES,
            "warm_starts": WARM_STARTS,
            "target_trials": args.target_trials,
            "max_workers_per_lane": args.max_workers_per_lane,
            "source_sha256": {n: sha256(REPO_ROOT / n) for n in PINNED_SOURCES},
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    state = load_state()
    phase = args.phase if args.phase != "auto" else state.get("phase", PHASE_TUNING)
    if phase in (PHASE_BLOCKED, PHASE_NEEDS_REVIEW):
        raise SystemExit(
            f"State is {phase!r}; inspect status.json and clear it deliberately."
        )

    # Warm starts: once, here, under the lock. Optuna warns that concurrent
    # enqueue_trial calls can duplicate trials, so workers never do this.
    if phase == PHASE_TUNING:
        for lane in LANES:
            result = warm_start_init(lane)
            log(f"WARM {json.dumps(result, sort_keys=True)}")
            if not result["ok"]:
                save_state(PHASE_BLOCKED, reason=f"warm start problem: {result}")
                raise SystemExit(f"Warm start problem on lane {lane}: {result}")

    reaper = Reaper()
    idle_scans: dict[str, int] = {slot: 0 for slot in SLOTS}
    runner: CurveRunner | None = None
    log(f"START manager pid={os.getpid()} phase={phase} target={args.target_trials}")

    while True:
        processes = budget_runner.iter_processes()
        foreign = budget_runner.other_orchestrators(processes)
        if foreign:
            log(f"PAUSE foreign coordinator: {[p.pid for p in foreign]}")
            time.sleep(LOOP_SECONDS)
            continue

        if phase == PHASE_TUNING:
            workers = tuning_workers(processes)
            snapshots = {lane: snapshot(lane) for lane in LANES}
            reaped = reaper.scan(snapshots, workers)
            if reaper.blocked_reason:
                save_state(PHASE_BLOCKED, reason=reaper.blocked_reason)
                raise SystemExit(reaper.blocked_reason)
            if reaped:
                snapshots = {lane: snapshot(lane) for lane in LANES}

            orphans = orphan_trainers(processes, workers)
            ready, reasons = handover_ready(
                snapshots, workers, orphans, args.target_trials
            )
            if ready:
                verdict = health_verdict(snapshots, args.min_best_value)
                for warning in verdict["warnings"]:
                    log(f"WARN {warning}")
                if verdict["verdict"] != PHASE_CURVES:
                    save_state(verdict["verdict"], problems=verdict["problems"],
                               warnings=verdict["warnings"])
                    log(f"STOP {verdict['verdict']}: {verdict['problems']}")
                    raise SystemExit(json.dumps(verdict, indent=2))
                # Freeze once. On a restart phase2_sources.json is re-read and
                # never recomputed, so a late straggler cannot change the
                # winner behind our back.
                if PHASE2_SOURCES_PATH.exists():
                    sources = json.loads(PHASE2_SOURCES_PATH.read_text())
                    log("HANDOVER reusing frozen phase2_sources.json")
                else:
                    sources = freeze_phase2_sources(snapshots)
                    budget_runner.atomic_json(PHASE2_SOURCES_PATH, sources)
                    log(
                        "HANDOVER frozen: "
                        + json.dumps(sources, sort_keys=True, default=str)
                    )
                # A validation failure here must not kill an unattended run at
                # 6am: record it and stop deliberately.
                try:
                    inject_config_sources(sources)
                    tasks = build_curve_tasks(sources)
                    budget_runner.validate_runtime_query_schedules(tasks)
                    budget_runner.validate_resolved_configs(tasks)
                except Exception as error:  # noqa: BLE001
                    reason = f"phase-2 validation failed: {error!r}"
                    log(f"BLOCKED {reason}")
                    save_state(PHASE_BLOCKED, reason=reason)
                    budget_runner.atomic_json(
                        STATUS_PATH, build_status(args.target_trials)
                    )
                    raise SystemExit(reason)
                save_state(PHASE_CURVES, warnings=verdict["warnings"])
                phase = PHASE_CURVES
                continue

            if len(workers) < len(SLOTS):
                lane = pick_lane(
                    snapshots, workers, args.target_trials, args.max_workers_per_lane
                )
                if lane is not None:
                    slots = free_slots(processes, idle_scans)
                    if slots:
                        # One launch per iteration so the journal registers the
                        # RUNNING trial before the next decision.
                        launch_worker(lane, slots[0])
                        idle_scans[slots[0]] = 0

            budget_runner.atomic_json(STATUS_PATH, build_status(args.target_trials))
            time.sleep(LOOP_SECONDS)
            continue

        # ---- phase 2 -----------------------------------------------------
        if runner is None:
            sources = json.loads(PHASE2_SOURCES_PATH.read_text())
            inject_config_sources(sources)
            runner = CurveRunner(
                build_curve_tasks(sources), budget_runner.MAX_ATTEMPTS
            )
            log(f"CURVES {len(runner.tasks)} tasks")

        runner.reap(processes, args.max_run_hours, args.collapse_action)
        if runner.all_done():
            budget_runner.atomic_json(
                COMPLETE_PATH,
                {
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "parked": runner.parked,
                },
            )
            save_state(PHASE_COMPLETE, parked=runner.parked)
            log("COMPLETE campaign finished")
            return

        if len(runner.running) < len(SLOTS):
            slots = free_slots(processes, idle_scans)
            for task in runner.pending():
                if not slots:
                    break
                slot = slots.pop(0)
                if runner.launch(task, slot, budget_runner.iter_processes()):
                    runner.started_at[task.key] = time.time()
                    idle_scans[slot] = 0

        budget_runner.atomic_json(STATUS_PATH, build_status(args.target_trials))
        time.sleep(LOOP_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--dry-run-phase2", action="store_true")
    parser.add_argument("--phase", choices=["auto", "tuning", "curves"], default="auto")
    parser.add_argument("--max-workers-per-lane", type=int, default=12)
    parser.add_argument("--target-trials", type=int, default=30)
    parser.add_argument("--min-best-value", type=float, default=55.0)
    parser.add_argument("--collapse-action", choices=["warn", "kill"], default="warn")
    parser.add_argument("--max-run-hours", type=float, default=10.0)
    args = parser.parse_args()

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    if args.status:
        print(json.dumps(build_status(args.target_trials), indent=2, sort_keys=True, default=str))
        return

    if args.validate:
        validate_warm_starts()
        report = {
            "warm_starts": "ok",
            "source_sha256": {name: sha256(REPO_ROOT / name) for name in PINNED_SOURCES},
            "studies_exist": {
                lane: snapshot(lane)["exists"] for lane in LANES
            },
            "worker_commands": {
                lane: " ".join(worker_command(lane, "24", "TOKEN"))
                for lane in LANES
            },
            "free_bytes": __import__("shutil").disk_usage(REPO_ROOT).free,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.dry_run_phase2:
        sources = json.loads(PHASE2_SOURCES_PATH.read_text())
        inject_config_sources(sources)
        tasks = build_curve_tasks(sources)
        print(json.dumps(
            {
                "sources": sources,
                "n_tasks": len(tasks),
                "tasks": [task.manifest_record() for task in tasks],
            },
            indent=2, sort_keys=True, default=str,
        ))
        budget_runner.validate_runtime_query_schedules(tasks)
        print(json.dumps(budget_runner.validate_resolved_configs(tasks), indent=2, default=str))
        return

    run_loop(args)


if __name__ == "__main__":
    main()
