#!/usr/bin/env python
"""Taratura dei due ibridi sotto la fusione "prova 1".

Perche' un tuner nuovo invece di ``tune_hybrid_sac.py``: quello e' rimasto alla
campagna precedente. Congela ``train_comparison_frac``, chiave che non esiste
piu' (Hydra fallirebbe subito), fissa ``n_ensembles=3``, ``train_freq=8`` e
``normalize_agent_reward=true``, e non conosce ne' ``gcl_fusion`` ne'
``shared_rollout_env``. Ripararlo significherebbe riscriverne meta'.

Questo importa ``PROTOCOL`` e ``ARMS`` da ``launch_thesis_runs``: la taratura
gira **per costruzione** nello stesso protocollo della campagna finale, e non
esiste il modo di farli divergere per dimenticanza.

Cosa si tara, e perche' solo questo
-----------------------------------
Quattro parametri: ``lr_rew``, ``l2_rew``, ``gradient_steps_rew`` e
l'architettura della rete di reward.

I due tassi sono i piu' importanti perche' prova 1 ha cambiato la scala del
gradiente: ad Adam arriva un vettore di norma ~1 invece di ~20, e il weight
decay (lambda*theta) NON e' invariante a quel riscalamento, quindi pesa circa
venti volte di piu' a parita' di lambda. I valori attuali vengono da una
taratura fatta sotto ``norm_balance``, cioe' per un gradiente venti volte piu'
grande. Da qui i range volutamente larghi.

Restano fuori dallo spazio di ricerca:

* ``batch_size_pref`` e ``batch_size_expert`` -- entrano in alpha come
  ``B = min(batch_size, N)``, quindi tararli significa tarare alpha, cioe'
  ottimizzare il contributo invece di misurarlo. Fissati a 64/64.
* ``pref_temperature`` e ``label_smoothing`` -- descrivono l'oracolo, cioe' il
  problema. Tararli per braccio darebbe all'ibrido etichette diverse da quelle
  della baseline con cui viene confrontato.
* protocollo, SAC, ``n_ensembles``, ``query_schedule``, ``initial_queries``,
  ``demo_weight`` (inerte sotto prova 1).

Coordinamento fra worker
------------------------
Storage su journal file, che regge piu' processi sulla stessa macchina senza
server. Ogni worker prende un core con ``taskset`` e chiede trial allo studio
condiviso finche' il budget globale non e' esaurito.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load_launcher():
    """Il launcher e' la fonte unica di protocollo e iperparametri per braccio."""
    spec = importlib.util.spec_from_file_location(
        "launch_thesis_runs", SCRIPTS / "launch_thesis_runs.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["launch_thesis_runs"] = module      # il dataclass ha bisogno del modulo
    spec.loader.exec_module(module)
    return module


LTR = _load_launcher()

TUNE_ROOT = REPO_ROOT / "outputs" / "tuning_thesis"
STORAGE = TUNE_ROOT / "journal"
WANDB_PROJECT = "tuning-thesis-prova1"

# Seed di taratura, DISGIUNTO da quelli di valutazione (1, 2, 3): una
# configurazione scelta sugli stessi seed su cui viene poi misurata si
# valuterebbe da sola.
TUNE_SEED = 11

NET_ARCH_CHOICES = ["[32,32]", "[64,64]", "[128,128]"]


def suggest(trial) -> dict:
    """I quattro parametri, con i range allargati per la nuova scala."""
    return {
        # 1e-5..1e-2: il valore ereditato (~1e-3) era per un gradiente ~20x
        "lr_rew": trial.suggest_float("lr_rew", 1e-5, 1e-2, log=True),
        # 1e-7..1e-2: il weight decay e' l'altro parametro che la
        # normalizzazione di prova 1 ha spostato
        "l2_rew": trial.suggest_float("l2_rew", 1e-7, 1e-2, log=True),
        "gradient_steps_rew": trial.suggest_int("gradient_steps_rew", 20, 200),
        "reward_net_arch": trial.suggest_categorical("reward_net_arch", NET_ARCH_CHOICES),
    }


def trial_overrides(arm_name: str, budget: int, params: dict, trial_number: int) -> list[str]:
    """Override della run: quelli del braccio, con i 4 parametri sostituiti."""
    base = LTR.arm_overrides(arm_name, budget, TUNE_SEED)

    run_name = f"tune_{arm_name}_B{budget}_t{trial_number:03d}"
    out_dir = (TUNE_ROOT / arm_name / run_name).relative_to(REPO_ROOT)
    replace = {
        "algo.kwargs.lr_rew": params["lr_rew"],
        "algo.kwargs.l2_rew": params["l2_rew"],
        "algo.kwargs.gradient_steps_rew": params["gradient_steps_rew"],
        "algo.kwargs.reward_model_kwargs.net_arch": params["reward_net_arch"],
        "run.output_dir": out_dir,
        "run.name": run_name,
        "run.group": f"tune_{arm_name}_B{budget}",
        "wandb.project": WANDB_PROJECT,
        "wandb.tags": f"[tuning,prova1,{arm_name},B{budget}]",
    }

    out, seen = [], set()
    for override in base:
        key = override.split("=", 1)[0]
        if key in replace:
            out.append(f"{key}={replace[key]}")
            seen.add(key)
        else:
            out.append(override)
    for key, value in replace.items():        # chiavi non presenti nel base
        if key not in seen:
            out.append(f"{key}={value}")
    return out


def run_trial(arm_name: str, budget: int, params: dict, trial_number: int, core: int):
    """Esegue una run. Restituisce il return finale, oppure None se e' fallita.

    None e non un valore-sentinella: un crash non e' una configurazione cattiva.
    Registrarlo come COMPLETE con -1e9 insegnerebbe a TPE a evitare quella zona
    di iperparametri per un motivo che non ha nulla a che vedere con essi, e lo
    renderebbe indistinguibile per stato da un trial legittimamente pessimo.
    """
    overrides = trial_overrides(arm_name, budget, params, trial_number)
    run_name = f"tune_{arm_name}_B{budget}_t{trial_number:03d}"
    log_dir = TUNE_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"

    cmd = ["taskset", "-c", str(core), sys.executable,
           str(SCRIPTS / "train_hybrid_sac.py"), *overrides]
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)

    # train_hybrid_sac crea una sottocartella col nome della run dentro output_dir
    eval_path = TUNE_ROOT / arm_name / run_name / run_name / "final_eval.json"
    if proc.returncode != 0 or not eval_path.exists():
        return None
    data = json.loads(eval_path.read_text())
    value = data.get("eval/mean_fast_return")
    return float(value) if value is not None else None


def worker(arm_name: str, budget: int, n_trials: int, core: int, study_name: str) -> None:
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    STORAGE.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(STORAGE / f"{study_name}.log")))
    study = optuna.load_study(study_name=study_name, storage=storage)

    for _ in range(n_trials):
        if len([t for t in study.get_trials(deepcopy=False)
                if t.state.name in ("COMPLETE", "RUNNING")]) >= study.user_attrs["budget"]:
            break
        trial = study.ask()
        params = suggest(trial)
        t0 = time.time()
        value = run_trial(arm_name, budget, params, trial.number, core)
        if value is None:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            esito = "FALLITO"
        else:
            study.tell(trial, value)
            esito = f"ret={value:8.2f}"
        print(f"[core {core}] trial {trial.number:3d}  {esito}  "
              f"({time.time()-t0:.0f}s)  {params}", flush=True)


def create_study(arm_name: str, budget: int, total_trials: int, study_name: str):
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    STORAGE.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(STORAGE / f"{study_name}.log")))
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        load_if_exists=True,
        # n_startup_trials=8: con 8 worker la prima tornata e' comunque casuale;
        # da li' in poi TPE modella su trial realmente conclusi.
        sampler=optuna.samplers.TPESampler(seed=TUNE_SEED, n_startup_trials=8),
    )
    study.set_user_attr("budget", total_trials)
    study.set_user_attr("arm", arm_name)
    study.set_user_attr("budget_B", budget)
    study.set_user_attr("tune_seed", TUNE_SEED)
    study.set_user_attr("space", "lr_rew,l2_rew,gradient_steps_rew,reward_net_arch")
    return study


def best(study_name: str) -> None:
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    storage = JournalStorage(JournalFileBackend(str(STORAGE / f"{study_name}.log")))
    study = optuna.load_study(study_name=study_name, storage=storage)
    done = [t for t in study.get_trials(deepcopy=False) if t.state.name == "COMPLETE"]
    done.sort(key=lambda t: (t.value if t.value is not None else -1e9), reverse=True)
    print(f"{study_name}: {len(done)} trial completati su {study.user_attrs.get('budget')}")
    for t in done[:10]:
        print(f"  #{t.number:3d}  {t.value:8.2f}   {t.params}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["hybrid_soft", "hybrid_bern"], required=True)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--trials", type=int, default=40, help="budget globale dello studio")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--first-core", type=int, required=False,
                    help="primo core; i worker prendono core consecutivi")
    ap.add_argument("--create", action="store_true", help="crea lo studio ed esce")
    ap.add_argument("--best", action="store_true", help="mostra i migliori trial ed esce")
    ap.add_argument("--dry-run", action="store_true", help="stampa gli override di un trial")
    args = ap.parse_args()

    study_name = f"prova1_{args.arm}_B{args.budget}"

    if args.best:
        best(study_name)
        return 0

    if args.dry_run:
        params = {"lr_rew": 1e-3, "l2_rew": 1e-5,
                  "gradient_steps_rew": 100, "reward_net_arch": "[64,64]"}
        for o in trial_overrides(args.arm, args.budget, params, 0):
            print(" ", o)
        return 0

    if args.create:
        create_study(args.arm, args.budget, args.trials, study_name)
        print(f"studio {study_name} creato: budget {args.trials} trial")
        return 0

    if args.first_core is None:
        ap.error("--first-core e' richiesto per lanciare i worker")

    create_study(args.arm, args.budget, args.trials, study_name)
    procs = []
    per_worker = -(-args.trials // args.workers)     # ceil
    for i in range(args.workers):
        core = args.first_core + i
        cmd = [sys.executable, __file__, "--arm", args.arm, "--budget", str(args.budget),
               "--trials", str(args.trials), "--workers", str(args.workers),
               "--first-core", str(core), "--_worker", str(per_worker)]
        procs.append(subprocess.Popen(cmd, cwd=REPO_ROOT))
        print(f"worker {i} su core {core} (pid {procs[-1].pid})")
    return 0


if __name__ == "__main__":
    # ramo worker: invocato da main() con --_worker
    if "--_worker" in sys.argv:
        idx = sys.argv.index("--_worker")
        n = int(sys.argv[idx + 1])
        del sys.argv[idx:idx + 2]
        a = argparse.ArgumentParser()
        a.add_argument("--arm"); a.add_argument("--budget", type=int)
        a.add_argument("--trials", type=int); a.add_argument("--workers", type=int)
        a.add_argument("--first-core", type=int)
        ns, _ = a.parse_known_args()
        worker(ns.arm, ns.budget, n, ns.first_core, f"prova1_{ns.arm}_B{ns.budget}")
        sys.exit(0)
    sys.exit(main())
