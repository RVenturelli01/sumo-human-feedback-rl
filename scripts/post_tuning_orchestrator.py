"""Global dynamic scheduler: budget curves -> min budgets -> final 5-seed runs.

v2 — task-granular. Every (stage, level, seed) is ONE task = one pinned
training run; the scheduler keeps a single global view of the 3-core slots
and dispatches any ready task onto any free slot, so runs from different
stages proceed in parallel as soon as their dependencies are met. No
oversubscription: one task per slot, slots recomputed before every dispatch.

Slot accounting (process-aware first, CPU as a conservative confirmation):
* busy if a tuning worker is pinned to it;
* busy if a run this scheduler launched (or adopted) is pinned to it;
* otherwise free only if its mean CPU over a 15 s window is < 50 %
  (guards against anything external; the floating hybrid_demo_2 worker is
  intentionally ignored by the process check - it uses idle cycles).

Task states are recovered from disk, not from memory: a task is DONE iff its
run dir contains final_eval.json, RUNNING iff a live process carries its
run.name — so the scheduler is restartable and adopts runs started by the
previous (serial) orchestrator without touching them.

Dependencies (unchanged): curve_<arm> needs its tuning finished (>= 30 trials
+ worker gone); compute_<arm> needs the 15 curve runs; final_<arm> needs the
arm's min budget; final_hybrid_demo_k_{A,B} need their own tuning plus
X_pref_soft and Y_demo_k (A = halves, B = full). Best config selection is
untouched: max sweep/mean_fast_return over COMPLETE trials.

Usage (server, tmux):
    python scripts/post_tuning_orchestrator.py --dry-run
    python scripts/post_tuning_orchestrator.py
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
LOOP_SECONDS = 60
CPU_WINDOW_S = 15
CPU_BUSY_THRESHOLD = 50.0  # % media sui 3 core dello slot

STUDY_SUFFIX = {"pref_bernoulli": "_q100k"}
CURVE_LEVELS = {
    "pref_soft": [10000, 5000, 2000, 1000, 500],
    "pref_bernoulli": [250000, 100000, 50000, 25000, 10000],
    "demo_1": [2723, 1000, 500, 200, 100, 50],
    "demo_2": [2723, 1000, 500, 200, 100, 50],
}
CURVE_SEEDS = (1, 2, 3)
FINAL_SEEDS = (1, 2, 3, 4, 5)
ARM_PRIORITY = ("pref_soft", "demo_1", "demo_2", "pref_bernoulli")


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _iter_cmdlines():
    if not os.path.isdir("/proc"):
        return
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
        except OSError:
            continue
        yield int(pid), cmd


def tuning_workers() -> dict:
    workers = {}
    for _, cmd in _iter_cmdlines():
        if "tune_hybrid_sac.py" in cmd:
            arm = re.search(r"--arm (\S+)", cmd)
            cores = re.search(r"--cores (\S+)", cmd)
            if arm:
                workers[arm.group(1)] = cores.group(1) if cores else "?"
    return workers


def live_run_names() -> dict:
    """{run_name: pid} for every live training run (any launcher)."""
    runs = {}
    for pid, cmd in _iter_cmdlines():
        if "test_hybrid_SAC.py" in cmd:
            m = re.search(r"run\.name=(\S+)", cmd)
            if m:
                runs[m.group(1)] = pid
    return runs


def pid_slot(pid: int):
    """Canonical slot a pid is pinned to, if any."""
    try:
        out = subprocess.run(["taskset", "-pc", str(pid)],
                             capture_output=True, text=True).stdout
        aff = out.rsplit(":", 1)[1].strip()
    except Exception:
        return None
    return aff if aff in CANONICAL_SLOTS else None


def _core_times():
    busy, total = {}, {}
    for line in open("/proc/stat"):
        m = re.match(r"cpu(\d+) ", line)
        if m:
            f = [int(x) for x in line.split()[1:]]
            total[int(m.group(1))] = sum(f)
            busy[int(m.group(1))] = sum(f) - f[3] - f[4]
    return busy, total


def slot_cpu_busy() -> dict:
    """{slot: mean busy %} over CPU_WINDOW_S (conservative confirmation)."""
    if not os.path.exists("/proc/stat"):
        return {s: 0.0 for s in CANONICAL_SLOTS}
    b0, t0 = _core_times()
    time.sleep(CPU_WINDOW_S)
    b1, t1 = _core_times()
    out = {}
    for slot in CANONICAL_SLOTS:
        lo, hi = (int(x) for x in slot.split("-"))
        cores = range(lo, hi + 1)
        vals = [100 * (b1[c] - b0[c]) / max(t1[c] - t0[c], 1) for c in cores]
        out[slot] = sum(vals) / len(vals)
    return out


def study_n_trials(arm: str) -> int:
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    if not JOURNAL.exists():
        return 0
    storage = JournalStorage(
        JournalFileBackend(str(JOURNAL), lock_obj=JournalFileOpenLock(str(JOURNAL)))
    )
    try:
        study = optuna.load_study(
            study_name=f"hybrid_sac_{arm}{STUDY_SUFFIX.get(arm, '')}", storage=storage)
    except KeyError:
        return 0
    return len(study.trials)


_tuning_done_cache = {}

def tuning_done(arm: str) -> bool:
    if _tuning_done_cache.get(arm):
        return True
    if arm in tuning_workers():
        return False
    ok = study_n_trials(arm) >= TRIALS_TARGET
    if ok:
        _tuning_done_cache[arm] = True
    return ok


def budgets() -> dict:
    return json.loads(BUDGETS_JSON.read_text()) if BUDGETS_JSON.exists() else {}


def budget_of(arm: str):
    entry = budgets().get(arm)
    return None if entry is None else int(entry["min_budget"])


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class Task:
    """One pinned training run (or a light compute step)."""

    def __init__(self, name, run_name, out_subdir, ready, env, launcher, args,
                 needs_slot=True):
        self.name = name          # es. curve_pref_soft_L2000_s3
        self.run_name = run_name  # es. budget_pref_soft_2000-seed3 (None per compute)
        self.out_subdir = out_subdir
        self.ready = ready
        self.env = env            # callable -> dict (valutato al dispatch)
        self.launcher = launcher
        self.args = args
        self.needs_slot = needs_slot

    def done(self) -> bool:
        if (STATE_DIR / f"{self.name}.done").exists():
            return True
        if self.run_name is None:
            return False
        base = REPO_ROOT / self.out_subdir
        if list(base.glob(f"{self.run_name}/final_eval.json")) or \
           list(base.glob(f"{self.run_name}_[0-9][0-9]/final_eval.json")):
            (STATE_DIR / f"{self.name}.done").touch()
            return True
        return False

    def failed(self) -> bool:
        return (STATE_DIR / f"{self.name}.failed").exists()


def make_tasks():
    tasks = []
    for arm in ARM_PRIORITY:
        suffix = STUDY_SUFFIX.get(arm, "")
        for level in CURVE_LEVELS[arm]:
            for seed in CURVE_SEEDS:
                tasks.append(Task(
                    name=f"curve_{arm}_L{level}_s{seed}",
                    run_name=f"budget_{arm}_{level}-seed{seed}",
                    out_subdir=f"outputs/budget_curves/budget_{arm}_{level}",
                    ready=(lambda a=arm: tuning_done(a)),
                    env=(lambda a=arm, l=level, s=seed, sf=suffix: {
                        "LEVELS": str(l), "SEEDS": str(s), "STUDY_SUFFIX": sf}),
                    launcher="./launchers/run_budget_curves.sh", args=[arm],
                ))
        curve_names = [f"curve_{arm}_L{l}_s{s}"
                       for l in CURVE_LEVELS[arm] for s in CURVE_SEEDS]
        tasks.append(Task(
            name=f"compute_{arm}",
            run_name=None, out_subdir="",
            ready=(lambda names=tuple(curve_names):
                   all((STATE_DIR / f"{n}.done").exists() for n in names)),
            env=(lambda: {}),
            launcher=sys.executable,
            args=["scripts/compute_min_budget.py", "--arm", arm,
                  "--out", str(BUDGETS_JSON)],
            needs_slot=False,
        ))

    def final_tasks(arm, group_suffix, ready, env_fn):
        for seed in FINAL_SEEDS:
            group = f"{arm}{group_suffix}"
            yield Task(
                name=f"final_{group}_s{seed}",
                run_name=f"{group}-seed{seed}",
                out_subdir=f"outputs/final/{group}",
                ready=ready,
                env=(lambda s=seed, fn=env_fn: {**fn(), "SEEDS": str(s)}),
                launcher="./launchers/run_final_5seeds.sh",
                args=[arm] + ([group_suffix] if group_suffix else []),
            )

    for arm in ARM_PRIORITY:
        suffix = STUDY_SUFFIX.get(arm, "")
        key = "PREF_BUDGET" if arm.startswith("pref_") else "DEMO_BUDGET"
        tasks.extend(final_tasks(
            arm, "",
            ready=(lambda a=arm: budget_of(a) is not None),
            env_fn=(lambda a=arm, k=key, sf=suffix:
                    {k: str(budget_of(a)), "STUDY_SUFFIX": sf}),
        ))

    for idx in (1, 2):
        arm, demo_arm = f"hybrid_demo_{idx}", f"demo_{idx}"
        for strat in ("A", "B"):
            def env_fn(a=arm, d=demo_arm, st=strat):
                x, y = budget_of("pref_soft"), budget_of(d)
                if st == "A":
                    x, y = max(1, x // 2), max(1, y // 2)
                return {"PREF_BUDGET": str(x), "DEMO_BUDGET": str(y)}
            tasks.extend(final_tasks(
                arm, f"_{strat}",
                ready=(lambda a=arm, d=demo_arm:
                       tuning_done(a) and budget_of("pref_soft") is not None
                       and budget_of(d) is not None),
                env_fn=env_fn,
            ))
    return tasks


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class Scheduler:
    def __init__(self, tasks):
        self.tasks = tasks
        self.running = {}   # name -> (popen|None, pid, slot, task)

    def adopt_external(self):
        """Adopt live runs (e.g. started by the previous orchestrator)."""
        live = live_run_names()
        for task in self.tasks:
            if task.run_name in live and task.name not in self.running:
                pid = live[task.run_name]
                slot = pid_slot(pid)
                self.running[task.name] = (None, pid, slot, task)
                log(f"ADOPT {task.name}: pid={pid} slot={slot} (run esterna in corso)")

    def reap(self):
        for name in list(self.running):
            popen, pid, slot, task = self.running[name]
            alive = popen.poll() is None if popen else os.path.isdir(f"/proc/{pid}")
            if alive:
                continue
            del self.running[name]
            if task.done():
                log(f"DONE {name}")
            else:
                (STATE_DIR / f"{name}.failed").touch()
                log(f"FAIL {name} — vedi {LOG_DIR / (name + '.log')}")

    def busy_slots(self):
        busy = set(s for s in tuning_workers().values() if s in CANONICAL_SLOTS)
        busy |= {slot for _, _, slot, _ in self.running.values() if slot}
        return busy

    def free_slots(self):
        candidates = [s for s in CANONICAL_SLOTS if s not in self.busy_slots()]
        if not candidates:
            return []
        cpu = slot_cpu_busy()
        return [s for s in candidates if cpu[s] < CPU_BUSY_THRESHOLD]

    def pending_ready(self):
        out = []
        for task in self.tasks:
            if task.name in self.running or task.done() or task.failed():
                continue
            if task.ready():
                out.append(task)
        return out

    def launch(self, task, slot):
        env = {**os.environ, "PYTHON_BIN": sys.executable, **task.env()}
        if task.needs_slot:
            env["CORE_SLOTS"] = slot
        log_file = open(LOG_DIR / f"{task.name}.log", "a")
        log_file.write(f"\n=== {time.strftime('%F %T')} slot={slot} "
                       f"env={task.env()} $ {task.launcher} {' '.join(task.args)}\n")
        log_file.flush()
        popen = subprocess.Popen([task.launcher, *task.args], cwd=REPO_ROOT,
                                 env=env, stdout=log_file, stderr=subprocess.STDOUT)
        self.running[task.name] = (popen, popen.pid, slot if task.needs_slot else None, task)
        log(f"START {task.name} su slot {slot}")

    def loop(self):
        while True:
            self.adopt_external()
            self.reap()
            ready = self.pending_ready()
            pending = [t for t in self.tasks
                       if not t.done() and not t.failed() and t.name not in self.running]
            if not pending and not self.running:
                failed = [t.name for t in self.tasks if t.failed()]
                log("Fine: tutti i task processati."
                    + (f" FALLITI: {failed}" if failed else ""))
                return
            # compute tasks: nessuno slot richiesto, esegui subito
            for task in [t for t in ready if not t.needs_slot]:
                self.launch(task, slot="-")
                self.reap_blocking(task.name)
            heavy = [t for t in ready if t.needs_slot]
            if heavy:
                for slot in self.free_slots():
                    if not heavy:
                        break
                    self.launch(heavy.pop(0), slot)
            time.sleep(LOOP_SECONDS)

    def reap_blocking(self, name):
        popen, _, _, task = self.running[name]
        popen.wait()
        del self.running[name]
        if popen.returncode == 0:
            (STATE_DIR / f"{name}.done").touch()
            log(f"DONE {name}")
        else:
            (STATE_DIR / f"{name}.failed").touch()
            log(f"FAIL {name}")


def print_plan(tasks):
    sched = Scheduler(tasks)
    sched.adopt_external()
    counts = {"done": 0, "failed": 0, "running": 0, "ready": 0, "waiting": 0}
    rows = []
    for t in tasks:
        if t.name in sched.running:
            st = "running"
        elif t.done():
            st = "done"
        elif t.failed():
            st = "failed"
        else:
            st = "ready" if t.ready() else "waiting"
        counts[st] += 1
        if st != "done":
            rows.append((t.name, st))
    print(f"Task totali: {len(tasks)}  {counts}")
    print(f"Worker tuning: {tuning_workers() or 'nessuno'}")
    print(f"Slot occupati (processi): {sorted(sched.busy_slots())}")
    print(f"Slot liberi (processi + CPU<{CPU_BUSY_THRESHOLD:.0f}% su {CPU_WINDOW_S}s): "
          f"{sched.free_slots()}")
    print(f"Budget: {budgets() or 'nessuno'}\n")
    for name, st in rows:
        print(f"  {name:34s} {st}")
    ready = [t for t in tasks if t.name not in sched.running
             and not t.done() and not t.failed() and t.ready() and t.needs_slot]
    free = sched.free_slots()
    print(f"\nPartirebbero subito: {[t.name for t in ready[:len(free)]]} sui slot {free}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    tasks = make_tasks()

    if args.dry_run:
        print_plan(tasks)
        return

    lock_file = open(BASE_DIR / "orchestrator.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("Un altro scheduler è già in esecuzione (orchestrator.lock).")
    lock_file.write(str(os.getpid()))
    lock_file.flush()

    log(f"Scheduler v2 avviato (pid {os.getpid()}). Task: {len(tasks)}")
    Scheduler(tasks).loop()


if __name__ == "__main__":
    main()
