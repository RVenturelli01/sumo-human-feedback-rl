#!/usr/bin/env python3
"""Curve a unita' di feedback appaiata (evento equivalente di annotazione).

PERCHE'
-------
Le curve di budget esistenti confrontano "500 preferenze" con "500 traiettorie
esperte", ma 500 traiettorie sono ~89.655 transizioni: 179 transizioni esperte
per query di preferenza (circa 90:1 contando le transizioni osservate, dato che
ogni query mostra due frammenti). Le proporzioni di budget, non l'algoritmo,
dominano quel confronto.

Qui il budget totale B e' espresso in EVENTI EQUIVALENTI DI FEEDBACK: una
comparazione fra due frammenti, oppure un'azione esperta dimostrata.

    pref_soft / pref_bernoulli   B query
    demo_2                       traiettorie intere, somma transizioni <= B
    hybrid_soft / hybrid_bern    B/2 query + <= B/2 transizioni esperte

Il "<=" con traiettorie intere e' voluto: demo_2 calcola i return su traiettorie
complete, troncarle cambierebbe la semantica della loss.

NON e' un modello di costo informativo: una transizione esperta contiene
l'azione ottima, una query contiene un bit. Contarle 1:1 favorisce il canale
dimostrazioni. Da dichiarare, non da nascondere.

ISOLAMENTO (requisito vincolante)
---------------------------------
Progetto W&B e directory dedicati, e NESSUN import delle costanti di
schedule_budget_curves_completion: cosi' e' impossibile che un percorso residuo
scriva negli esperimenti esistenti.

Uso:
    python scripts/unit_matched_curves.py --dry-run
    python scripts/unit_matched_curves.py --resolve-check
    python scripts/unit_matched_curves.py --canary
    python scripts/unit_matched_curves.py            # riempie gli slot
    python scripts/unit_matched_curves.py --status
    python scripts/unit_matched_curves.py --report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import optuna  # noqa: E402
from optuna.storages import JournalStorage  # noqa: E402
from optuna.storages.journal import (  # noqa: E402
    JournalFileBackend,
    JournalFileOpenLock,
)

from tune_hybrid_sac import (  # noqa: E402
    FIXED_OVERRIDES,
    arm_overrides,
    fixed_param_overrides,
    params_to_overrides,
)

# --- isolamento --------------------------------------------------------------
PROJECT = "thesis-unit-matched-curves"
ENTITY = "andrea02polimi-politecnico-di-milano"
STATE_ROOT = REPO_ROOT / "outputs" / "unit_matched_curves"
RUN_ROOT = STATE_ROOT / "runs"
LOG_ROOT = STATE_ROOT / "logs"
MARKER_ROOT = STATE_ROOT / "markers"
FORBIDDEN_PATH = "budget_curves_completion"   # non deve comparire da nessuna parte

PYTHON = "/home/fis3/miniconda3/envs/sumo-rlhf/bin/python"
TRAIN = "scripts/train_hybrid_sac.py"
JOURNAL = REPO_ROOT / "outputs" / "optuna" / "journal.log"

CORES = tuple(str(c) for c in range(24, 48))
TOTAL_TIMESTEPS = 1_000_000
TIMESTEPS_PER_ITERATION = 20_000
N_ENVS = 2
EVAL_EPISODES = 20
LOOP_SECONDS = 20

BUDGETS = (200, 1000, 5446, 20000, 50000)
SEEDS = (1, 2, 3)


@dataclass(frozen=True)
class Arm:
    label: str
    tuner_arm: str
    study: str
    labels: str
    rank: int = 0          # 0 = best; 1 = secondo classificato
    uses_pref: bool = True
    uses_demo: bool = True
    sensitivity: bool = False


ARMS = (
    Arm("pref_soft", "pref_soft", "hybrid_sac_pref_soft", "soft",
        uses_demo=False),
    Arm("pref_bernoulli", "pref_bernoulli",
        "hybrid_sac_pref_bernoulli_q100k_temp", "binary_bernoulli",
        uses_demo=False),
    Arm("demo_2", "demo_2", "hybrid_sac_demo_2", "auto", uses_pref=False),
    Arm("hybrid_soft", "hybrid_demo_2", "hybrid_sac_hybrid_demo_2_hom_soft",
        "soft"),
    # rank=1: t18 (best) e' un outlier di configurazione ([32,32]/iat=40000)
    # mentre i rank 2-5 condividono [8,8]/iat=10000/iq=136. L'argmax di 22 stime
    # a singolo seed e' distorto verso l'alto (winner's curse) e t18 supera t5
    # di 0,17 punti. Scelta post-hoc, dichiarata, con sensitivity su t18 sotto.
    Arm("hybrid_bern", "hybrid_demo_2", "hybrid_sac_hybrid_demo_2_hom_bern",
        "binary_bernoulli", rank=1),
    Arm("hybrid_bern_t18", "hybrid_demo_2",
        "hybrid_sac_hybrid_demo_2_hom_bern", "binary_bernoulli",
        rank=0, sensitivity=True),
)
ARM_BY_LABEL = {a.label: a for a in ARMS}


@dataclass(frozen=True)
class Task:
    arm: str
    budget: int
    seed: int

    @property
    def group(self) -> str:
        return f"um_{self.arm}_B{self.budget}"

    @property
    def run_name(self) -> str:
        return f"{self.group}-seed{self.seed}"

    @property
    def key(self) -> str:
        return f"{self.arm}_B{self.budget}_s{self.seed}"

    @property
    def output_root(self) -> Path:
        return RUN_ROOT / self.group

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name

    @property
    def log_path(self) -> Path:
        return LOG_ROOT / f"{self.key}.log"

    @property
    def wandb_run_id(self) -> str:
        payload = f"{PROJECT}/{self.run_name}".encode()
        return "um" + hashlib.sha1(payload).hexdigest()[:18]

    @property
    def sort_key(self) -> tuple:
        # Copertura prima: tutti i seed 1 partono prima di qualunque seed 2.
        # La sensitivity va in coda ai seed 1 principali.
        arm = ARM_BY_LABEL[self.arm]
        return (self.seed, 1 if arm.sensitivity else 0, self.budget, self.arm)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def storage():
    path = str(JOURNAL)
    return JournalStorage(
        JournalFileBackend(path, lock_obj=JournalFileOpenLock(path))
    )


_TUNED: dict[str, dict[str, Any]] = {}


def tuned(arm: Arm) -> dict[str, Any]:
    """Params + pin del trial scelto (per rank) dello studio del braccio."""
    if arm.label in _TUNED:
        return _TUNED[arm.label]
    study = optuna.load_study(study_name=arm.study, storage=storage())
    done = sorted(
        (t for t in study.trials
         if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None),
        key=lambda t: -t.value,
    )
    if len(done) <= arm.rank:
        raise SystemExit(f"{arm.study}: rank {arm.rank} non disponibile.")
    trial = done[arm.rank]
    _TUNED[arm.label] = {
        "trial_number": trial.number,
        "value": trial.value,
        "params": dict(trial.params),
        "fixed_params": study.user_attrs.get("fixed_params") or {},
    }
    return _TUNED[arm.label]


def initial_queries(arm: Arm, pref_budget: int) -> int:
    """Scala la frazione initial_queries/budget validata dal tuning.

    pref_soft   250/5000  = 0,05
    pref_bern   20000/1e5 = 0,20
    hybrid_bern 136/2723  = 0,0499
    hybrid_soft 100/500   = 0,20  (fissato da arm_overrides nel tuning)
    """
    fractions = {
        "pref_soft": 250 / 5000,
        "pref_bernoulli": 20000 / 100000,
        "hybrid_soft": 100 / 500,
        "hybrid_bern": 136 / 2723,
        "hybrid_bern_t18": 136 / 2723,
    }
    fraction = fractions[arm.label]
    return max(1, min(pref_budget, round(fraction * pref_budget)))


def build_overrides(task: Task) -> list[str]:
    arm = ARM_BY_LABEL[task.arm]
    info = tuned(arm)
    B = task.budget

    pref_budget = B if arm.uses_pref and not arm.uses_demo else (
        B // 2 if arm.uses_pref else 0
    )
    demo_transitions = B if arm.uses_demo and not arm.uses_pref else (
        B // 2 if arm.uses_demo else None
    )

    # Blocco tunato, identico a quello che emette export_best_config --format full.
    overrides = list(FIXED_OVERRIDES)
    overrides += arm_overrides(
        arm.tuner_arm,
        pref_budget if pref_budget else 5000,      # inerte se il braccio non usa pref
        demo_transitions if demo_transitions else 500,
        arm.labels,
    )
    overrides += params_to_overrides(info["params"])
    overrides += fixed_param_overrides(
        info["fixed_params"].get("demo_weight"),
        info["fixed_params"].get("pref_temperature"),
    )

    # Override del punto: applicati per ultimi, vincono su tutto.
    point: list[str] = []
    if arm.uses_pref:
        point += [
            f"algo.kwargs.total_queries={pref_budget}",
            f"train.kwargs.total_queries={pref_budget}",
            f"algo.kwargs.initial_queries={initial_queries(arm, pref_budget)}",
        ]
    else:
        point += [
            "algo.kwargs.total_queries=0",
            "train.kwargs.total_queries=0",
            "algo.kwargs.initial_queries=0",
        ]
    if arm.uses_demo:
        point += [
            "run.n_expert_trajectories=null",
            f"run.n_expert_transitions={demo_transitions}",
            f"run.demo_subsample_seed={1000 + task.seed}",
        ]
    point += [
        f"run.seed={task.seed}",
        f"run.name={task.run_name}",
        f"run.group={task.group}",
        f"run.output_dir={task.output_root.relative_to(REPO_ROOT)}",
        f"wandb.entity={ENTITY}",
        f"wandb.project={PROJECT}",
        f"wandb.tags=[unit_matched,{task.arm},B{task.budget}]",
        f"env.n_envs={N_ENVS}",
        f"eval.n_episodes={EVAL_EPISODES}",
        f"train.kwargs.total_timesteps={TOTAL_TIMESTEPS}",
        f"train.kwargs.timesteps_per_iteration={TIMESTEPS_PER_ITERATION}",
    ]
    return overrides + point


def assert_overrides(task: Task, overrides: list[str]) -> None:
    arm = ARM_BY_LABEL[task.arm]
    joined = " ".join(overrides)
    last = {}
    for item in overrides:
        if "=" in item:
            key, value = item.split("=", 1)
            last[key] = value

    if FORBIDDEN_PATH in joined:
        raise AssertionError(f"{task.key}: percorso vietato {FORBIDDEN_PATH!r}")
    if last.get("wandb.project") != PROJECT:
        raise AssertionError(f"{task.key}: progetto {last.get('wandb.project')!r}")
    if not last.get("run.output_dir", "").startswith("outputs/unit_matched_curves"):
        raise AssertionError(f"{task.key}: output_dir {last.get('run.output_dir')!r}")

    tq = int(last["algo.kwargs.total_queries"])
    iq = int(last["algo.kwargs.initial_queries"])
    if iq > tq:
        raise AssertionError(f"{task.key}: initial_queries {iq} > total_queries {tq}")

    if arm.uses_demo:
        if last.get("run.n_expert_trajectories") != "null":
            raise AssertionError(
                f"{task.key}: ultima n_expert_trajectories = "
                f"{last.get('run.n_expert_trajectories')!r}, attesa null"
            )
        expected = task.budget if not arm.uses_pref else task.budget // 2
        if last.get("run.n_expert_transitions") != str(expected):
            raise AssertionError(
                f"{task.key}: n_expert_transitions "
                f"{last.get('run.n_expert_transitions')!r} != {expected}"
            )
        if last.get("run.demo_subsample_seed") in (None, "None", "null"):
            raise AssertionError(f"{task.key}: demo_subsample_seed mancante")
    else:
        if any("n_expert_transitions" in o for o in overrides):
            raise AssertionError(f"{task.key}: braccio pref con n_expert_transitions")
        if last.get("algo.kwargs.demo_weight") != "0.0":
            raise AssertionError(
                f"{task.key}: demo_weight {last.get('algo.kwargs.demo_weight')!r}, atteso 0.0"
            )

    if arm.uses_pref and arm.uses_demo:
        if last.get("algo.kwargs.demo_weight") != "1.0":
            raise AssertionError(
                f"{task.key}: hybrid demo_weight {last.get('algo.kwargs.demo_weight')!r}"
            )
        if tq != task.budget // 2:
            raise AssertionError(f"{task.key}: hybrid total_queries {tq}")


def build_tasks() -> list[Task]:
    tasks = []
    for arm in ARMS:
        seeds = (1,) if arm.sensitivity else SEEDS
        for budget in BUDGETS:
            for seed in seeds:
                tasks.append(Task(arm.label, budget, seed))
    return sorted(tasks, key=lambda t: t.sort_key)


# --- marker / stato ----------------------------------------------------------

def marker_path(task: Task) -> Path:
    return MARKER_ROOT / f"{task.key}.json"


def read_marker(task: Task) -> dict:
    try:
        return json.loads(marker_path(task).read_text())
    except (OSError, ValueError):
        return {}


def write_marker(task: Task, state: str, **extra) -> None:
    MARKER_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {**read_marker(task), "key": task.key, "state": state,
               "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    tmp = marker_path(task).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(marker_path(task))


def final_value(task: Task):
    for candidate in sorted(task.output_root.glob(f"{task.run_name}*/final_eval.json")):
        try:
            return json.loads(candidate.read_text())
        except (OSError, ValueError):
            continue
    return None


def is_done(task: Task) -> bool:
    return final_value(task) is not None


def live_runs() -> dict[str, int]:
    """{run_name: pid} dei training vivi (esclusa la risoluzione Hydra)."""
    out = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if TRAIN not in cmd or "--cfg" in cmd:
            continue
        match = re.search(r"run\.name=(\S+)", cmd)
        if match:
            out[match.group(1)] = int(entry.name)
    return out


def core_of(pid: int) -> str | None:
    try:
        result = subprocess.run(["taskset", "-pc", str(pid)],
                                capture_output=True, text=True)
        value = result.stdout.rsplit(":", 1)[-1].strip()
        return value if value.isdigit() else None
    except Exception:
        return None


def busy_cores() -> set[str]:
    return {c for c in (core_of(p) for p in live_runs().values()) if c}


# --- lancio ------------------------------------------------------------------

def launch(task: Task, core: str) -> int:
    overrides = build_overrides(task)
    assert_overrides(task, overrides)
    command = ["taskset", "-c", core, PYTHON, TRAIN] + overrides
    task.output_root.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("WANDB_DISABLED", None)
    env.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg", "WANDB_MODE": "online",
        "WANDB_RUN_ID": task.wandb_run_id, "WANDB_RESUME": "allow",
    })
    with task.log_path.open("a") as stream:
        stream.write(f"\n=== {time.strftime('%F %T')} core={core}\n")
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, text=True,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
        )
    write_marker(task, "running", pid=process.pid, core=core,
                 wandb_run_id=task.wandb_run_id)
    log(f"START {task.run_name} core={core} pid={process.pid}")
    return process.pid


def run_loop(max_launch: int | None) -> None:
    tasks = build_tasks()
    launched = 0
    while True:
        live = live_runs()
        busy = busy_cores()
        free = [c for c in CORES if c not in busy]
        pending = [t for t in tasks
                   if t.run_name not in live
                   and not is_done(t)
                   and read_marker(t).get("state") != "failed"
                   and not (read_marker(t).get("state") == "running"
                            and t.run_name in live)]
        # marca come finiti quelli che hanno prodotto output
        for t in tasks:
            if is_done(t) and read_marker(t).get("state") != "done":
                write_marker(t, "done")
        pending = [t for t in pending if not is_done(t)]
        if not pending and not live:
            log("COMPLETE: nessun task pendente")
            return
        for task in pending:
            if not free:
                break
            if max_launch is not None and launched >= max_launch:
                break
            core = free.pop(0)
            launch(task, core)
            launched += 1
            time.sleep(2)
        if max_launch is not None and launched >= max_launch:
            log(f"raggiunto --max-launch={max_launch}")
            return
        time.sleep(LOOP_SECONDS)


# --- report ------------------------------------------------------------------

def collect() -> dict:
    out: dict[tuple[str, int], list[float]] = {}
    success: dict[tuple[str, int], list[float]] = {}
    for task in build_tasks():
        payload = final_value(task)
        if not payload:
            continue
        out.setdefault((task.arm, task.budget), []).append(
            payload["eval/mean_fast_return"])
        if "eval/success_rate" in payload:
            success.setdefault((task.arm, task.budget), []).append(
                payload["eval/success_rate"])
    return {"return": out, "success": success}


def transitions_table() -> dict:
    """Consumo reale di transizioni per gruppo, dai log."""
    pattern = re.compile(r"Loaded (\d+) expert trajectories \((\d+) transitions\)")
    out: dict[str, list[tuple[int, int]]] = {}
    for path in sorted(LOG_ROOT.glob("*.log")):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        found = pattern.search(text)
        if found:
            key = path.stem.rsplit("_s", 1)[0]
            out.setdefault(key, []).append(
                (int(found.group(1)), int(found.group(2))))
    return out


def do_report() -> None:
    data = collect()
    print("=== curve a unita' appaiata (media +/- std sui seed) ===")
    print("%-18s %8s %3s %9s %8s %9s" % ("braccio", "B", "n", "return", "std", "success"))
    for arm in ARMS:
        for budget in BUDGETS:
            values = data["return"].get((arm.label, budget))
            if not values:
                continue
            suc = data["success"].get((arm.label, budget), [])
            print("%-18s %8d %3d %9.2f %8.2f %9s" % (
                arm.label, budget, len(values), statistics.mean(values),
                statistics.pstdev(values) if len(values) > 1 else 0.0,
                ("%.2f" % statistics.mean(suc)) if suc else "-"))
    print()
    print("=== consumo reale di transizioni esperte (budget nominale vs reale) ===")
    for key, entries in sorted(transitions_table().items()):
        traj = sorted({e[0] for e in entries})
        trans = sorted({e[1] for e in entries})
        print("  %-30s traiettorie=%s transizioni=%s" % (key, traj, trans))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # noqa: BLE001
        print(f"\n(matplotlib non disponibile: {error})")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for metric, ax, title in (("return", axes[0], "Held-out mean fast return"),
                              ("success", axes[1], "Held-out success rate")):
        for arm in ARMS:
            xs, ys, es = [], [], []
            for budget in BUDGETS:
                values = data[metric].get((arm.label, budget))
                if not values:
                    continue
                xs.append(budget)
                ys.append(statistics.mean(values))
                es.append(statistics.pstdev(values) if len(values) > 1 else 0.0)
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=es, marker="o", markersize=5, capsize=3,
                        linewidth=1.8,
                        linestyle="--" if arm.sensitivity else "-",
                        label=arm.label)
        ax.set_xscale("log")
        ax.set_xlabel("B · eventi equivalenti di feedback (log)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean_fast_return")
    axes[1].set_ylabel("success_rate")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Curve a unita' di feedback appaiata — 1M step, media ± std sui seed",
                 fontsize=11)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        out = STATE_ROOT / f"unit_matched_curves.{suffix}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nfigura: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resolve-check", action="store_true",
                        help="Risolve la config Hydra di un task per braccio.")
    parser.add_argument("--canary", action="store_true",
                        help="Lancia un solo run (hybrid_soft B=200 seed 1).")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--max-launch", type=int, default=None)
    args = parser.parse_args()

    STATE_ROOT.mkdir(parents=True, exist_ok=True)

    if args.report:
        do_report()
        return

    tasks = build_tasks()

    if args.status:
        data = collect()
        live = live_runs()
        done = sum(1 for t in tasks if is_done(t))
        print(f"task {len(tasks)} | finiti {done} | vivi {len(live)}")
        do_report()
        return

    if args.dry_run:
        for task in tasks:
            assert_overrides(task, build_overrides(task))
        print(f"asserzioni PASSATE su {len(tasks)} task")
        print(f"ordine (copertura prima), primi 30:")
        for task in tasks[:30]:
            print(f"   {task.sort_key}  {task.run_name}")
        print()
        for label in ("pref_soft", "demo_2", "hybrid_soft"):
            task = Task(label, 1000, 1)
            keys = [o for o in build_overrides(task)
                    if o.startswith(("algo.kwargs.total_queries",
                                     "algo.kwargs.initial_queries",
                                     "algo.kwargs.demo_weight",
                                     "algo.kwargs.pref_temperature",
                                     "run.n_expert", "run.demo_subsample_seed",
                                     "wandb.project", "run.output_dir"))]
            print(f"{task.run_name}:")
            for k in keys:
                print(f"   {k}")
        return

    if args.resolve_check:
        for arm in ARMS:
            task = Task(arm.label, 1000, 1)
            overrides = build_overrides(task)
            assert_overrides(task, overrides)
            result = subprocess.run(
                [PYTHON, TRAIN, *overrides, "--cfg", "job", "--resolve"],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                print(f"  {arm.label}: RISOLUZIONE FALLITA\n{result.stderr[-800:]}")
                continue
            text = result.stdout
            picks = {}
            for key in ("total_queries", "initial_queries", "n_expert_trajectories",
                        "n_expert_transitions", "demo_weight", "pref_temperature",
                        "demo_subsample_seed", "project"):
                m = re.search(rf"^\s*{key}:\s*(\S+)\s*$", text, re.M)
                picks[key] = m.group(1) if m else "-"
            print(f"  {arm.label:16s} " + "  ".join(
                f"{k}={v}" for k, v in picks.items()))
        return

    if args.canary:
        task = Task("hybrid_soft", 200, 1)
        if is_done(task) or task.run_name in live_runs():
            print("canary gia' presente")
            return
        free = [c for c in CORES if c not in busy_cores()]
        launch(task, free[0])
        return

    run_loop(args.max_launch)


if __name__ == "__main__":
    main()
