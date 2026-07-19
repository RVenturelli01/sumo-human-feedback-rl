"""Post-tuning orchestrator: budget curves -> minimum budgets -> final 5-seed runs.

Runs ONE stage at a time (natural serialization: no two heavy stages ever
overlap), picking the highest-priority stage whose dependencies are satisfied
and for which enough core slots are free. Designed to run for days inside
tmux while the remaining tuning workers finish.

Stage graph (deps in parentheses):

    curve_pref_soft        (tuning pref_soft done)        -> X_pref_soft
    curve_demo_1           (tuning demo_1 done)           -> Y_demo_1
    curve_demo_2           (tuning demo_2 done)           -> Y_demo_2
    curve_pref_bernoulli   (tuning pref_bernoulli_q100k)  -> X_pref_bernoulli
    final_pref_soft        (X_pref_soft)
    final_demo_1           (Y_demo_1)
    final_demo_2           (Y_demo_2)
    final_pref_bernoulli   (X_pref_bernoulli)
    final_hybrid_demo_1_A/B (tuning hybrid_demo_1 done + X_pref_soft + Y_demo_1)
    final_hybrid_demo_2_A/B (tuning hybrid_demo_2 done + X_pref_soft + Y_demo_2)

"tuning done" = the arm's Optuna study has >= TRIALS_TARGET trials AND its
worker process has exited. Free slots = the canonical 3-core slots of the
33-47 pool minus those still pinned by tuning workers (the floating
hybrid_demo_2 worker is ignored: it uses idle cycles by design).

Safety rails:
* flock on outputs/post_tuning/orchestrator.lock — a second instance refuses
  to start;
* every stage runs with the free slots computed at launch time and blocks
  until it finishes;
* state markers in outputs/post_tuning/state/<stage>.done|.failed make the
  orchestrator restartable (completed stages are skipped, failed ones are NOT
  retried automatically);
* per-stage logs in outputs/post_tuning/logs/<stage>.log;
* --dry-run prints the full plan (stage, deps, readiness, command) and exits.

Usage (on the server, inside tmux):
    python scripts/post_tuning_orchestrator.py --dry-run   # show the plan
    python scripts/post_tuning_orchestrator.py             # run everything
"""

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = REPO_ROOT / "outputs" / "post_tuning"
STATE_DIR = BASE_DIR / "state"
LOG_DIR = BASE_DIR / "logs"
BUDGETS_JSON = BASE_DIR / "budgets.json"
JOURNAL = REPO_ROOT / "outputs" / "optuna" / "journal.log"

CANONICAL_SLOTS = ("33-35", "36-38", "39-41", "42-44", "45-47")
TRIALS_TARGET = 30
POLL_SECONDS = 300
MIN_SLOTS = 2

# (arm, study_suffix) per il tuning; il suffisso vale solo per bernoulli.
STUDY_SUFFIX = {"pref_bernoulli": "_q100k"}


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------

def tuning_workers():
    """{arm: cores_spec} for every live tune_hybrid_sac.py worker."""
    workers = {}
    if not os.path.isdir("/proc"):  # e.g. dry-run on macOS
        return workers
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
        except OSError:
            continue
        if "tune_hybrid_sac.py" not in cmd:
            continue
        arm = re.search(r"--arm (\S+)", cmd)
        cores = re.search(r"--cores (\S+)", cmd)
        if arm:
            workers[arm.group(1)] = cores.group(1) if cores else "?"
    return workers


def study_n_trials(arm: str) -> int:
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    if not JOURNAL.exists():
        return 0
    name = f"hybrid_sac_{arm}{STUDY_SUFFIX.get(arm, '')}"
    storage = JournalStorage(
        JournalFileBackend(str(JOURNAL), lock_obj=JournalFileOpenLock(str(JOURNAL)))
    )
    try:
        study = optuna.load_study(study_name=name, storage=storage)
    except KeyError:
        return 0
    return len(study.trials)


def tuning_done(arm: str) -> bool:
    if arm in tuning_workers():
        return False
    n = study_n_trials(arm)
    if n < TRIALS_TARGET:
        print(f"[warn] {arm}: worker assente ma solo {n}/{TRIALS_TARGET} trial "
              f"nel journal — resto in attesa (serve intervento manuale?)")
        return False
    return True


def free_slots() -> list:
    """Canonical slots not pinned by a live tuning worker (floating ones ignored)."""
    pinned = set(tuning_workers().values())
    return [s for s in CANONICAL_SLOTS if s not in pinned]


def budgets() -> dict:
    if BUDGETS_JSON.exists():
        return json.loads(BUDGETS_JSON.read_text())
    return {}


def budget_of(arm: str):
    entry = budgets().get(arm)
    return None if entry is None else int(entry["min_budget"])


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

def curve_stage(arm: str):
    suffix = STUDY_SUFFIX.get(arm, "")
    def command(slots):
        env = {"CORE_SLOTS": " ".join(slots), "STUDY_SUFFIX": suffix}
        return [
            ("./launchers/run_budget_curves.sh", [arm], env),
            (sys.executable, ["compute_min_budget.py", "--arm", arm,
                              "--out", str(BUDGETS_JSON)], {"cwd": "scripts"}),
        ]
    return {
        "name": f"curve_{arm}",
        "ready": lambda: tuning_done(arm),
        "deps": f"tuning {arm}{suffix} completo (>= {TRIALS_TARGET} trial, worker uscito)",
        "command": command,
    }


def final_stage(arm: str):
    suffix = STUDY_SUFFIX.get(arm, "")
    is_pref = arm.startswith("pref_")
    def command(slots):
        env = {"CORE_SLOTS": " ".join(slots), "STUDY_SUFFIX": suffix}
        x = budget_of(arm)
        env["PREF_BUDGET" if is_pref else "DEMO_BUDGET"] = str(x)
        return [("./launchers/run_final_5seeds.sh", [arm], env)]
    return {
        "name": f"final_{arm}",
        "ready": lambda: budget_of(arm) is not None,
        "deps": f"budget minimo di {arm} calcolato (curva completata)",
        "command": command,
    }


def hybrid_final_stage(loss_idx: int, strategy: str):
    arm = f"hybrid_demo_{loss_idx}"
    demo_arm = f"demo_{loss_idx}"
    def ready():
        return (tuning_done(arm)
                and budget_of("pref_soft") is not None
                and budget_of(demo_arm) is not None)
    def command(slots):
        x, y = budget_of("pref_soft"), budget_of(demo_arm)
        if strategy == "A":
            x, y = max(1, x // 2), max(1, y // 2)
        env = {"CORE_SLOTS": " ".join(slots),
               "PREF_BUDGET": str(x), "DEMO_BUDGET": str(y)}
        return [("./launchers/run_final_5seeds.sh", [arm, f"_{strategy}"], env)]
    return {
        "name": f"final_{arm}_{strategy}",
        "ready": ready,
        "deps": f"tuning {arm} completo + X_pref_soft + Y_{demo_arm}"
                + (" (usati a metà)" if strategy == "A" else " (pieni)"),
        "command": command,
    }


# Priority order: curves first (pref_soft/demo_1 unblock the hybrids),
# then baseline finals, then hybrid finals.
STAGES = [
    curve_stage("pref_soft"),
    curve_stage("demo_1"),
    curve_stage("demo_2"),
    curve_stage("pref_bernoulli"),
    final_stage("pref_soft"),
    final_stage("demo_1"),
    final_stage("demo_2"),
    final_stage("pref_bernoulli"),
    hybrid_final_stage(1, "A"),
    hybrid_final_stage(1, "B"),
    hybrid_final_stage(2, "A"),
    hybrid_final_stage(2, "B"),
]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def marker(stage_name: str, kind: str) -> Path:
    return STATE_DIR / f"{stage_name}.{kind}"


def run_stage(stage: dict, slots: list) -> bool:
    name = stage["name"]
    log_path = LOG_DIR / f"{name}.log"
    log(f"START {name} su slot {slots} (log: {log_path})")
    with open(log_path, "a") as lf:
        for prog, args, extra in stage["command"](slots):
            env = {**os.environ, "PYTHON_BIN": sys.executable,
                   **{k: v for k, v in extra.items() if k != "cwd"}}
            cwd = REPO_ROOT / extra["cwd"] if "cwd" in extra else REPO_ROOT
            lf.write(f"\n=== {time.strftime('%F %T')} $ {prog} {' '.join(args)} "
                     f"env={ {k: v for k, v in extra.items() if k != 'cwd'} }\n")
            lf.flush()
            result = subprocess.run([prog, *args], cwd=cwd, env=env,
                                    stdout=lf, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                log(f"FAIL {name}: {prog} exit {result.returncode} — vedi {log_path}")
                marker(name, "failed").touch()
                return False
    marker(name, "done").touch()
    log(f"DONE {name}")
    return True


def stage_status(stage: dict) -> str:
    if marker(stage["name"], "done").exists():
        return "done"
    if marker(stage["name"], "failed").exists():
        return "failed"
    return "ready" if stage["ready"]() else "waiting"


def print_plan() -> None:
    workers = tuning_workers()
    slots = free_slots()
    print(f"Worker di tuning attivi: {workers or 'nessuno'}")
    print(f"Slot liberi: {slots} (minimo per partire: {MIN_SLOTS})")
    print(f"Budget calcolati: {budgets() or 'nessuno'}\n")
    print(f"{'stage':28s} {'stato':8s} dipendenze")
    for stage in STAGES:
        print(f"{stage['name']:28s} {stage_status(stage):8s} {stage['deps']}")
    ready = [s for s in STAGES if stage_status(s) == "ready"]
    if ready and len(slots) >= MIN_SLOTS:
        s = ready[0]
        print(f"\nProssimo stage che partirebbe: {s['name']}")
        for prog, args, extra in s["command"](slots):
            envs = " ".join(f"{k}={v!r}" for k, v in extra.items() if k != "cwd")
            print(f"  $ {envs} {prog} {' '.join(args)}")
    else:
        print("\nNessuno stage partirebbe ora (dipendenze non pronte o slot insufficienti).")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit; no locks, no side effects.")
    args = parser.parse_args()

    if args.dry_run:
        print_plan()
        return

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    lock_file = open(BASE_DIR / "orchestrator.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("Un altro orchestratore è già in esecuzione (orchestrator.lock).")
    lock_file.write(str(os.getpid()))
    lock_file.flush()

    log(f"Orchestratore avviato (pid {os.getpid()}). Stage totali: {len(STAGES)}")
    while True:
        statuses = {s["name"]: stage_status(s) for s in STAGES}
        pending = [s for s in STAGES if statuses[s["name"]] in ("ready", "waiting")]
        if not pending:
            done = sum(1 for v in statuses.values() if v == "done")
            failed = [k for k, v in statuses.items() if v == "failed"]
            log(f"Fine: {done}/{len(STAGES)} stage completati."
                + (f" FALLITI: {failed}" if failed else ""))
            break

        ready = [s for s in STAGES if statuses[s["name"]] == "ready"]
        slots = free_slots()
        if ready and len(slots) >= MIN_SLOTS:
            run_stage(ready[0], slots)  # blocking; failures are marked, loop continues
            continue

        waiting_on = {s["name"]: statuses[s["name"]] for s in pending}
        log(f"In attesa ({len(slots)} slot liberi): {waiting_on} — riprovo tra {POLL_SECONDS}s")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
