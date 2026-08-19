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
import dataclasses
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
    ap.add_argument("--batch-size-pref", type=int, default=None,
                    help="sovrascrive batch_size_pref dei bracci scelti. Serve a "
                         "confrontare la vecchia scelta (256, quella delle run "
                         "th_v2) con la nuova simmetrica (64) a parita' di codice: "
                         "quel valore entra in alpha come B = min(batch_size, N), "
                         "quindi a B=1000 sposta alpha da ~0.25 a ~0.54.")
    ap.add_argument("--initial-queries", type=int, default=None,
                    help="numero fisso di query al bootstrap, invece della quota "
                         "per braccio. Serve a sbloccare alpha fin dall'inizio: "
                         "sotto ALPHA_MIN_PREFS confronti la stima non parte e il "
                         "canale preferenze viene scartato del tutto, cosa che a "
                         "B=10 dura fino all'iterazione 44 su 100.")
    ap.add_argument("--initial-agent-timesteps", type=int, default=None,
                    help="sovrascrive il warmup del braccio. pref_soft era tarato "
                         "a 40000 e l'uniformazione a 20000 e' uno dei due "
                         "candidati per il suo crollo a B=100.")
    ap.add_argument("--like-report", action="store_true",
                    help="riproduce il protocollo delle lane del report "
                         "(n_envs=2, train_freq=8, iperparametri originali), "
                         "lasciando pero' shared_rollout_env e n_ensembles come "
                         "sono nel PROTOCOL corrente.")
    ap.add_argument("--set-arm-field", action="append", default=[],
                    metavar="BRACCIO.CAMPO=VALORE",
                    help="sovrascrive un campo del braccio (dataclasses.replace), "
                         "es. hybrid_soft.gradient_steps_rew=100 oppure "
                         "hybrid_soft.uses_pref=false. Ripetibile. A differenza "
                         "di --extra-override, arm_overrides e validate() "
                         "leggono il valore nuovo e restano coerenti.")
    ap.add_argument("--n-ensembles", type=int, default=None,
                    help="sovrascrive n_ensembles del PROTOCOL")
    # Queste due passano dal PROTOCOL e non da --extra-override: validate()
    # legge le attese proprio dal PROTOCOL, quindi un override in coda le
    # contraddirebbe e il lancio si fermerebbe.
    ap.add_argument("--shared-rollout-env", choices=("true", "false"), default=None,
                    help="sovrascrive shared_rollout_env del PROTOCOL. false = "
                         "ambiente di rollout dedicato, come le lane del report.")
    ap.add_argument("--query-schedule", default=None,
                    choices=("constant", "hyperbolic", "inverse_quadratic"),
                    help="forma della distribuzione delle query nel tempo. Riscrive "
                         "il PROTOCOL invece di aggiungersi in coda, cosi' Hydra non "
                         "riceve due volte la stessa chiave. Nota che quando il budget "
                         "e' piu' piccolo del numero di iterazioni tutte le quote "
                         "vanno a zero e i due schedule decrescenti coincidono: a B=10 "
                         "entrambi mettono le query nelle prime iterazioni.")
    ap.add_argument("--n-envs", type=int, default=None,
                    help="numero di ambienti paralleli. Con 2 o piu' si passa da "
                         "DummyVecEnv a SubprocVecEnv e i seed ambiente sono "
                         "base_seed..base_seed+n-1. validate() impone comunque "
                         "replay ratio 2.0 = gradient_steps/(train_freq*n_envs).")
    ap.add_argument("--train-freq", type=int, default=None,
                    help="passi ambiente fra due aggiornamenti SAC. Da scegliere "
                         "insieme a --n-envs per tenere il replay ratio a 2.0.")
    ap.add_argument("--total-queries", type=int, default=None,
                    help="scollega il numero di confronti dal budget delle "
                         "dimostrazioni. Sotto ALPHA_MIN_PREFS (5) alpha resta "
                         "fissato a 1 per tutta la corsa: serve a isolare il solo "
                         "effetto della normalizzazione del gradiente. Con 0 NON "
                         "funziona, perche' _reward_step salta la fusione.")
    ap.add_argument("--total-timesteps", type=int, default=None,
                    help="durata della run. Il numero di iterazioni ne discende: "
                         "n_iterations = total_timesteps / timesteps_per_iteration, "
                         "quindi 1000000 con i 20000 di serie da 50 iterazioni "
                         "invece di 100, e altrettanti round di reward learning.")
    ap.add_argument("--bootstrap", choices=("true", "false", "null"), default=None,
                    help="fissa bootstrap_comparisons. null = decide n_ensembles "
                         "(il default). true con n_ensembles=1 riproduce il "
                         "bootstrap a un solo membro delle lane del report.")
    ap.add_argument("--extra-override", action="append", default=[],
                    help="override Hydra aggiuntivo, ripetibile")
    ap.add_argument("--poll", type=int, default=60, help="secondi fra due controlli")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    LTR.CAMPAIGN = args.campaign          # entra nel nome del gruppo e delle cartelle
    if args.initial_queries is not None:
        # Sostituire la funzione, non aggiungere un override in coda: cosi'
        # arm_overrides e validate leggono lo stesso numero e non possono
        # divergere.
        fisso = args.initial_queries
        LTR.initial_queries = lambda arm, budget, _n=fisso: _n
        print(f"initial_queries forzato a {fisso}", flush=True)
    if args.n_ensembles is not None:
        chiave = "algo.kwargs.reward_model_kwargs.n_ensembles"
        LTR.PROTOCOL = tuple(f"{chiave}={args.n_ensembles}" if o.startswith(chiave + "=")
                             else o for o in LTR.PROTOCOL)
        print(f"n_ensembles forzato a {args.n_ensembles}", flush=True)
    if args.n_envs is not None:
        chiave = "env.n_envs"
        LTR.PROTOCOL = tuple(f"{chiave}={args.n_envs}" if o.startswith(chiave + "=")
                             else o for o in LTR.PROTOCOL)
        print(f"n_envs forzato a {args.n_envs}", flush=True)
    if args.train_freq is not None:
        chiave = "agent.kwargs.train_freq"
        LTR.PROTOCOL = tuple(f"{chiave}={args.train_freq}" if o.startswith(chiave + "=")
                             else o for o in LTR.PROTOCOL)
        print(f"train_freq forzato a {args.train_freq}", flush=True)
    if args.total_queries is not None:
        LTR.QUERIES_OVERRIDE = args.total_queries
        print(f"total_queries forzato a {args.total_queries} "
              f"(alpha resta fissato a 1 se < 5)", flush=True)
    if args.query_schedule is not None:
        chiave = "algo.kwargs.query_schedule"
        LTR.PROTOCOL = tuple(f"{chiave}={args.query_schedule}"
                             if o.startswith(chiave + "=") else o
                             for o in LTR.PROTOCOL)
        print(f"query_schedule forzato a {args.query_schedule}", flush=True)
    if args.total_timesteps is not None:
        chiave = "train.kwargs.total_timesteps"
        LTR.PROTOCOL = tuple(f"{chiave}={args.total_timesteps}"
                             if o.startswith(chiave + "=") else o
                             for o in LTR.PROTOCOL)
        tpi = next(int(o.split("=")[1]) for o in LTR.PROTOCOL
                   if o.startswith("train.kwargs.timesteps_per_iteration="))
        print(f"total_timesteps forzato a {args.total_timesteps} "
              f"({args.total_timesteps // tpi} iterazioni da {tpi} passi)", flush=True)
    if args.shared_rollout_env is not None:
        chiave = "env.shared_rollout_env"
        LTR.PROTOCOL = tuple(f"{chiave}={args.shared_rollout_env}"
                             if o.startswith(chiave + "=") else o
                             for o in LTR.PROTOCOL)
        print(f"shared_rollout_env forzato a {args.shared_rollout_env}", flush=True)
    if args.bootstrap is not None:
        chiave = "algo.kwargs.bootstrap_comparisons"
        resto = tuple(o for o in LTR.PROTOCOL if not o.startswith(chiave + "="))
        LTR.PROTOCOL = resto + (f"{chiave}={args.bootstrap}",)
        print(f"bootstrap_comparisons forzato a {args.bootstrap}", flush=True)
    if args.extra_override:
        # Si avvolge arm_overrides invece di toccare il launcher: cosi' anche
        # validate() risolve la config con gli stessi override applicati.
        _orig = LTR.arm_overrides
        LTR.arm_overrides = (lambda a, b, s_, _o=_orig, _e=list(args.extra_override):
                             _o(a, b, s_) + _e)
        print("override aggiuntivi: " + ", ".join(args.extra_override), flush=True)
    if args.like_report:
        # Valori delle lane demo2_1net / pref_soft_1net / p1_soft_bexp64 /
        # bern_ls_p1_alphavar. Restano fuori shared_rollout_env e n_ensembles,
        # che sono le due variabili che vogliamo cambiare.
        LTR.PROTOCOL = tuple(
            "env.n_envs=2" if o.startswith("env.n_envs=") else
            "agent.kwargs.train_freq=8" if o.startswith("agent.kwargs.train_freq=") else o
            for o in LTR.PROTOCOL)
        for arm, campi in (("hybrid_soft", dict(net_arch="[64,64]")),
                           ("hybrid_bern", dict(gradient_steps_rew=78,
                                                initial_agent_timesteps=40000)),
                           ("pref_soft",   dict(initial_agent_timesteps=40000))):
            if arm in args.arms:
                LTR.ARMS[arm] = dataclasses.replace(LTR.ARMS[arm], **campi)
        print("protocollo del report: n_envs=2, train_freq=8, iperparametri originali",
              flush=True)
    # Dopo --like-report, cosi' un --set-arm-field puo' correggerne i valori.
    for spec in args.set_arm_field:
        target, sep, valore = spec.partition("=")
        arm_name, _, campo = target.partition(".")
        if not sep or arm_name not in LTR.ARMS:
            raise SystemExit(f"--set-arm-field non valido: {spec!r} "
                             f"(atteso BRACCIO.CAMPO=VALORE, bracci: {', '.join(LTR.ARMS)})")
        vecchio = getattr(LTR.ARMS[arm_name], campo)   # AttributeError se il campo non esiste
        if isinstance(vecchio, bool):
            nuovo = valore.lower() in ("true", "1", "yes")
        elif isinstance(vecchio, int):
            nuovo = int(valore)
        elif isinstance(vecchio, float):
            nuovo = float(valore)
        else:                       # str oppure None (es. net_arch, labels)
            nuovo = valore
        LTR.ARMS[arm_name] = dataclasses.replace(LTR.ARMS[arm_name], **{campo: nuovo})
        print(f"{arm_name}.{campo}: {vecchio} -> {nuovo}", flush=True)
    if args.initial_agent_timesteps is not None:
        for arm in args.arms:
            LTR.ARMS[arm] = dataclasses.replace(
                LTR.ARMS[arm], initial_agent_timesteps=args.initial_agent_timesteps)
        print(f"initial_agent_timesteps forzato a {args.initial_agent_timesteps} "
              f"per {', '.join(args.arms)}", flush=True)
    if args.batch_size_pref is not None:
        # Sostituire la voce in ARMS invece di aggiungere un override in coda:
        # cosi' anche validate() controlla il valore nuovo, e non resta un
        # override che contraddice in silenzio la definizione del braccio.
        for arm in args.arms:
            LTR.ARMS[arm] = dataclasses.replace(
                LTR.ARMS[arm], batch_size_pref=args.batch_size_pref)
        print(f"batch_size_pref forzato a {args.batch_size_pref} "
              f"per {', '.join(args.arms)}", flush=True)

    # Ordine intrecciato fra i bracci: se qualcosa si ferma a meta' si resta
    # con una copertura parziale di tutti, non con un braccio completo e gli
    # altri a zero.
    # Ordine a seed maggiore: prima tutta la griglia (bracci x budget) del
    # seed 1, poi quella del seed 2, e cosi' via. Lanciare a blocchi per
    # braccio darebbe dopo quattro ore un solo braccio completo e nulla degli
    # altri; cosi' invece dopo la prima tornata si ha gia' una riga intera di
    # risultati confrontabili, e un problema si vede subito su tutti.
    tutti = [LTR.Task(a, b, s)
             for s in args.seeds for b in args.budgets for a in args.arms]
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
