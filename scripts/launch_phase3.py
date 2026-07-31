#!/usr/bin/env python3
"""Fase 3: punto (500,500), completamento a 5 seed, e controllo per transizioni.

Quattro blocchi, 18 run a 2M step, una sola ondata sui core 24-47.

E1  soft (500 pref, 500 traiettorie)            seed 1-5   -> 5 run  [punto nuovo]
E2  completamento a 5 seed                      seed 4-5   -> 8 run
      soft x=10, soft x=100, bern x=1000, bern x=2723
E3  soft (500 pref, max 500 TRANSIZIONI)        seed 1-3   -> 3 run  [controllo]
E4  soft x=2723                                 seed 4-5   -> 2 run

Perche' E3 esiste
-----------------
Al budget nominalmente omogeneo (500, 500) i due canali NON ricevono la stessa
quantita' di dati: 500 preferenze su frammenti di lunghezza 1 sono 500
transizioni etichettate, mentre 500 traiettorie esperte sono 89.655 transizioni
(misurato, seed 1001). Rapporto 221:1. E3 appaia i canali per transizioni
(tetto 500 -> 2-4 traiettorie) e sta allo stesso x=500 di E1, cosi' i due punti
sono affiancabili e la differenza fra loro e' il risultato.

Note di implementazione
-----------------------
* Lo scheduler si costruisce con la matrice VECCHIA e si usa solo per launch():
  __init__ valida lo stato di transizione contro quella matrice e ogni altro
  metodo itera self.tasks.
* La guardia anti-duplicato NON puo' usare task.output_root.exists(): quello e'
  il percorso del GRUPPO, che per i seed 4-5 esiste gia' dai seed 1-3 e farebbe
  rifiutare tutto. Si controlla la directory del singolo run.
* Un collasso e' un risultato valido: nessun controllo sul livello di reward.

Uso:
    python scripts/launch_phase3.py --dry-run
    python scripts/launch_phase3.py
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import hybrid_hom_campaign_manager as campaign  # noqa: E402
from scripts import schedule_budget_curves_completion as budget_runner  # noqa: E402

SOURCES_PATH = REPO_ROOT / "outputs" / "hybrid_hom_campaign" / "phase2_sources.json"
MANIFEST_PATH = REPO_ROOT / "outputs" / "hybrid_hom_campaign" / "phase3_manifest.json"

E3_ARM = "hybrid_demo_2_soft_trmatch"
E3_TRANSITIONS = 500

# (corsia, budget per canale, seed) da lanciare.
E1 = [("soft", 500, seed) for seed in (1, 2, 3, 4, 5)]
E2 = [
    ("soft", 10, 4), ("soft", 10, 5),
    ("soft", 100, 4), ("soft", 100, 5),
    ("bern", 1000, 4), ("bern", 1000, 5),
    ("bern", 2723, 4), ("bern", 2723, 5),
]
E4 = [("soft", 2723, 4), ("soft", 2723, 5)]
E3_SEEDS = (1, 2, 3)


def lane_point(lane: str, per_channel: int, sources: dict):
    """Point per un livello della curva, con la stessa ricetta del manager."""
    config = campaign.LANES[lane]
    fraction = sources[lane]["initial_fraction"]
    initial = max(1, min(per_channel, round(fraction * per_channel)))
    return budget_runner.Point(
        arm=config["curve_arm"],
        budget=per_channel * 2,
        source_key=config["curve_arm"],
        pref_budget=per_channel,
        demo_budget=per_channel,
        initial_queries=initial,
        normalize_agent_reward=True,
        labels_type=config["labels"],
        loss_type="demo_2",
        query_schedule="constant",
        fragmenter_type="active",
        pref_temperature=config["pref_temperature"],
        demo_weight=campaign.DEMO_WEIGHT,
    )


def e3_point(sources: dict):
    """Stessa config tunata di soft, ma budget demo contato in transizioni.

    demo_budget resta 500 (mantiene coerenti lo split nominale, la chiave di
    export_overrides e demo_subsample_seed); demo_transitions ha la precedenza
    negli override emessi.
    """
    fraction = sources["soft"]["initial_fraction"]
    return budget_runner.Point(
        arm=E3_ARM,
        budget=1000,
        source_key=campaign.LANES["soft"]["curve_arm"],
        pref_budget=500,
        demo_budget=500,
        initial_queries=max(1, min(500, round(fraction * 500))),
        normalize_agent_reward=True,
        labels_type="soft",
        loss_type="demo_2",
        query_schedule="constant",
        fragmenter_type="active",
        pref_temperature=campaign.LANES["soft"]["pref_temperature"],
        demo_weight=campaign.DEMO_WEIGHT,
        demo_transitions=E3_TRANSITIONS,
    )


def build_tasks(sources: dict) -> list:
    tasks = []
    for block, entries in (("E1", E1), ("E2", E2), ("E4", E4)):
        for lane, per_channel, seed in entries:
            point = lane_point(lane, per_channel, sources)
            tasks.append((block, budget_runner.Task(point, seed)))
    point3 = e3_point(sources)
    for seed in E3_SEEDS:
        tasks.append(("E3", budget_runner.Task(point3, seed)))
    return tasks


def already_present(task, live_names: set) -> str | None:
    """Motivo per cui il run esiste gia', altrimenti None.

    NON usa task.output_root.exists(): quello e' il percorso del gruppo, che per
    i seed 4-5 esiste gia' dai seed precedenti.
    """
    if task.run_name in live_names:
        return "processo vivo"
    if budget_runner.read_marker(task):
        return "marker presente"
    # Stesso glob di final_output_dirs: copre anche le dir dei ritenti (-seedN_01).
    existing = list(task.output_root.glob(f"{task.run_name}*"))
    if existing:
        return f"directory esistente: {existing[0].name}"
    if task.log_path.exists():
        return "log esistente"
    return None


def assert_overrides(block: str, task, overrides: tuple) -> None:
    """Le asserzioni documentate nel piano."""
    trajectory_keys = [o for o in overrides if o.startswith("run.n_expert_trajectories=")]
    transition_keys = [o for o in overrides if o.startswith("run.n_expert_transitions=")]
    seed_keys = [o for o in overrides if o.startswith("run.demo_subsample_seed=")]

    if block == "E3":
        if not trajectory_keys or trajectory_keys[-1] != "run.n_expert_trajectories=null":
            raise AssertionError(
                f"{task.run_name}: l'ultimo n_expert_trajectories deve essere null, "
                f"trovato {trajectory_keys}"
            )
        if transition_keys != [f"run.n_expert_transitions={E3_TRANSITIONS}"]:
            raise AssertionError(
                f"{task.run_name}: atteso un solo n_expert_transitions={E3_TRANSITIONS}, "
                f"trovato {transition_keys}"
            )
    else:
        if transition_keys:
            raise AssertionError(
                f"{task.run_name}: {block} non deve avere n_expert_transitions, "
                f"trovato {transition_keys}"
            )
        if not trajectory_keys or trajectory_keys[-1].endswith("=null"):
            raise AssertionError(
                f"{task.run_name}: {block} deve terminare con un budget in "
                f"traiettorie, trovato {trajectory_keys}"
            )
    if not seed_keys or seed_keys[-1].endswith("=None"):
        raise AssertionError(
            f"{task.run_name}: demo_subsample_seed mancante o None: {seed_keys}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-launch", type=int, default=None,
                        help="Lancia al massimo N run (per un avvio a scaglioni).")
    args = parser.parse_args()

    sources = json.loads(SOURCES_PATH.read_text())
    # Prima di qualsiasi export_overrides: e' @lru_cache.
    campaign.inject_config_sources(sources)
    budget_runner.CONFIG_SOURCES[E3_ARM] = budget_runner.CONFIG_SOURCES[
        campaign.LANES["soft"]["curve_arm"]
    ]

    tasks = build_tasks(sources)
    processes = budget_runner.iter_processes()
    live_names = set(budget_runner.live_training_runs(processes))

    pending, skipped = [], []
    for block, task in tasks:
        overrides = budget_runner.export_overrides(
            task.point.source_key, task.point.pref_budget, task.point.demo_budget
        ) + budget_runner.task_overrides(task)
        assert_overrides(block, task, overrides)
        reason = already_present(task, live_names)
        if reason:
            skipped.append((block, task, reason))
        else:
            pending.append((block, task, overrides))

    print(f"task totali {len(tasks)} | da lanciare {len(pending)} | gia' presenti {len(skipped)}")
    for block, task, reason in skipped:
        print(f"  SKIP  [{block}] {task.run_name}: {reason}")
    print()
    for block, task, overrides in pending:
        marks = [o for o in overrides
                 if o.startswith(("run.n_expert_", "run.demo_subsample_seed=",
                                  "algo.kwargs.total_queries=",
                                  "algo.kwargs.initial_queries="))]
        print(f"  [{block}] {task.run_name}")
        print(f"        {' '.join(marks)}")
    print()
    print("asserzioni sugli override: PASSATE per tutti i task")

    if args.dry_run:
        print("\n--dry-run: nessun lancio.")
        return

    free = [slot for slot in budget_runner.SINGLE_SLOTS
            if not budget_runner.constrained_slot_processes(processes, slot)]
    print(f"slot liberi: {len(free)} -> {free}")

    matrix = budget_runner.build_tasks()
    scheduler = budget_runner.Scheduler(
        matrix,
        max_parallel=len(budget_runner.SINGLE_SLOTS),
        max_attempts=budget_runner.MAX_ATTEMPTS,
        loop_seconds=budget_runner.LOOP_SECONDS,
        transition_state=budget_runner.load_transition_state(matrix),
    )

    limit = args.max_launch if args.max_launch is not None else len(pending)
    launched = []
    for (block, task, _), slot in zip(pending[:limit], free):
        if not scheduler.launch(task, slot, budget_runner.iter_processes()):
            print(f"  slot {slot} occupato prima del lancio, salto {task.run_name}")
            continue
        running = scheduler.running.pop(task.key)
        launched.append({"block": block, "run_name": task.run_name,
                         "group": task.group, "seed": task.seed,
                         "slot": slot, "pid": running.pid,
                         "demo_transitions": task.point.demo_transitions})
        print(f"  START [{block}] {task.run_name} slot={slot} pid={running.pid}")

    budget_runner.atomic_json(MANIFEST_PATH, {
        "launched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "launched": launched,
        "not_launched": [t.run_name for _, t, _ in pending[len(launched):]],
    })
    print(f"\nlanciati {len(launched)}/{len(pending)} | manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
