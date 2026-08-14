#!/usr/bin/env python
"""Coda di run: lancia man mano che i core si liberano.

``launch_thesis_runs.py`` pretende abbastanza core liberi subito e rifiuta
altrimenti. Qui invece i task restano in coda e partono appena c'e' posto,
cosi' si puo' accodare lavoro dietro una campagna gia' in corso.

Due cautele sull'occupazione dei core:

* i worker del tuning lanciano un trial alla volta con ``taskset``, quindi fra
  un trial e il successivo il loro core sembra libero per qualche secondo.
  Leggere solo l'affinita' dei processi di training farebbe rubare quel core.
  I core dichiarati da un worker vivo (``--first-core``) sono percio' trattati
  come RISERVATI, non liberi.
* si parte comunque da un solo task per giro, con una verifica subito dopo:
  meglio riempire la macchina lentamente che sovrapporre due run sullo stesso
  core senza accorgersene.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "launch_thesis_runs", SCRIPTS / "launch_thesis_runs.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["launch_thesis_runs"] = module
    spec.loader.exec_module(module)
    return module


LTR = _load_launcher()


def reserved_by_tuning() -> set[int]:
    """Core dichiarati dai worker di tuning vivi, anche se ora inattivi."""
    reserved = set()
    out = subprocess.run(["ps", "-eo", "args"], text=True,
                         stdout=subprocess.PIPE).stdout.splitlines()
    for line in out:
        if "tune_thesis.py" not in line or "--first-core" not in line:
            continue
        parts = line.split()
        try:
            reserved.add(int(parts[parts.index("--first-core") + 1]))
        except (ValueError, IndexError):
            continue
    return reserved


def free_cores(first: int, last: int) -> list[int]:
    busy = LTR.busy_cores() | reserved_by_tuning()
    return [c for c in range(last, first - 1, -1) if c not in busy]


def launch_one(task, core: int) -> dict:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = LTR.build_command(task.arm, task.budget, task.seed, core)
    with open(task.log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] lanciata {task.run_name} pid={proc.pid} core={core}", flush=True)
    return {"arm": task.arm, "budget": task.budget, "seed": task.seed,
            "run_name": task.run_name, "pid": proc.pid, "core": core,
            "started": stamp}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Piu' bracci in UN solo scheduler: due processi separati si
    # contenderebbero gli stessi core liberi e potrebbero assegnarne uno due
    # volte, perche' ciascuno decide guardando lo stato un istante prima.
    ap.add_argument("--arms", nargs="+", required=True,
                    choices=sorted(LTR.ARMS))
    ap.add_argument("--budgets", nargs="+", type=int, default=[10, 100, 1000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--campaign", default="fix",
                    help="etichetta nel nome del gruppo: th_<campaign>_<arm>_B<budget>")
    # 32 e non CORE_FIRST(=16): i core 16-31 restano ai due studi di tuning,
    # che ne hanno 8 ciascuno. Il presidio dei core riservati resta comunque
    # attivo, ma cosi' non serve fidarsene.
    ap.add_argument("--first-core", type=int, default=32)
    ap.add_argument("--last-core", type=int, default=LTR.CORE_LAST)
    ap.add_argument("--poll", type=int, default=60, help="secondi fra due controlli")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    LTR.CAMPAIGN = args.campaign          # entra nel nome del gruppo e delle cartelle

    # Ordine intrecciato fra i bracci: se qualcosa si ferma a meta' si resta
    # con una copertura parziale di tutti, non con un braccio completo e gli
    # altri a zero.
    tutti = [LTR.Task(a, b, s)
             for b in args.budgets for s in args.seeds for a in args.arms]
    # Ripresa: una run gia' lanciata ha la sua cartella di output. Senza questo
    # controllo un riavvio dello scheduler rilancerebbe cio' che sta gia'
    # girando, con lo stesso nome W&B e la stessa cartella.
    gia_partite = [t for t in tutti if t.output_dir.exists()]
    pending = [t for t in tutti if not t.output_dir.exists()]
    for t in gia_partite:
        print(f"salto {t.run_name}: gia' lanciata", flush=True)
    print(f"in coda: {len(pending)} run ({', '.join(args.arms)}, budget "
          f"{args.budgets}, seed {args.seeds}) come th_{args.campaign}_*  "
          f"[core {args.first_core}-{args.last_core}]", flush=True)

    if args.dry_run:
        for t in pending:
            print(f"  {t.run_name}")
        print("\nlibero ora:", free_cores(args.first_core, args.last_core))
        return 0

    # Validazione anticipata: un override sbagliato si scopre adesso, non fra
    # sei ore quando finalmente si libera un core.
    for arm in args.arms:
        for budget in args.budgets:
            LTR.validate(arm, budget, args.seeds[0])
    print("config validate", flush=True)

    manifest_path = (LTR.OUTPUT_ROOT
                     / f"manifest_queue_{args.campaign}_{'_'.join(args.arms)}.json")
    started: list[dict] = []
    while pending:
        libere = free_cores(args.first_core, args.last_core)
        if libere:
            started.append(launch_one(pending.pop(0), libere[0]))
            manifest_path.write_text(json.dumps(
                {"campaign": args.campaign, "runs": started,
                 "pending": [t.run_name for t in pending]}, indent=2))
            time.sleep(5)         # lascia che taskset compaia prima del prossimo giro
            continue
        time.sleep(args.poll)

    print(f"tutte lanciate. manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
