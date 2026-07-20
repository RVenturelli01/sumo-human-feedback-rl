"""Export the best Optuna trial of an arm as Hydra overrides / JSON params.

Reads the shared journal, loads the study of the given arm and prints the
best trial in the requested format:

* ``overrides`` (default) — space-separated Hydra overrides (arm presets +
  best params), ready to append to a ``train_hybrid_sac.py`` command line.
* ``full``     — like ``overrides`` but prefixed with the tuner's fixed
  overrides too: a complete, self-contained config for final runs.
* ``params``   — the raw Optuna param dict as JSON. A list of the top-k dicts
  is valid input for ``tune_hybrid_sac.py --enqueue-params`` (warm starts).
* ``summary``  — human-readable report of the top-k trials.

With ``--save-dir`` the best trial is also archived as a self-contained JSON
(params, full overrides, objective, run metadata) — the durable record of the
selected hyperparameters, independent of the Optuna journal. Recommended at
the end of the tuning phase:

    python scripts/export_best_config.py --arm pref_soft --save-dir configs/best

Usage:
    python scripts/export_best_config.py --arm pref_soft
    python scripts/export_best_config.py --arm demo_2 --format params --top-k 3 > warm.json
"""

import argparse
import datetime
import json
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

from tune_hybrid_sac import (
    ARMS,
    FIXED_OVERRIDES,
    OBJECTIVE_METRIC,
    arm_overrides,
    params_to_overrides,
)


def completed_trials_sorted(study):
    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    return sorted(done, key=lambda t: t.value, reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--storage-path", default="outputs/optuna/journal.log")
    parser.add_argument("--format", choices=["overrides", "full", "params", "summary"],
                        default="overrides")
    parser.add_argument("--study-suffix", default="",
                        help="Suffix of the study to read (e.g. '_q100k').")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--pref-budget", type=int, default=5000)
    parser.add_argument("--demo-budget", type=int, default=500)
    parser.add_argument("--save-dir", default=None,
                        help="Also archive the best trial as <dir>/<arm><suffix>.json "
                             "(params + full overrides + metadata).")
    args = parser.parse_args()

    storage = JournalStorage(
        JournalFileBackend(args.storage_path, lock_obj=JournalFileOpenLock(args.storage_path))
    )
    study = optuna.load_study(
        study_name=f"hybrid_sac_{args.arm}{args.study_suffix}", storage=storage
    )
    trials = completed_trials_sorted(study)[: args.top_k]
    if not trials:
        raise SystemExit(f"No completed trials for arm {args.arm}.")

    if args.save_dir:
        best = trials[0]
        record = {
            "arm": args.arm,
            "study": f"hybrid_sac_{args.arm}{args.study_suffix}",
            "trial_number": best.number,
            "objective": best.value,
            "params": best.params,
            "overrides": (
                FIXED_OVERRIDES
                + arm_overrides(args.arm, args.pref_budget, args.demo_budget)
                + params_to_overrides(best.params)
            ),
            "pref_budget": args.pref_budget,
            "demo_budget": args.demo_budget,
            "eval": {k: v for k, v in best.user_attrs.items() if k.startswith("eval/")},
            "run_dir": best.user_attrs.get("run_dir"),
            "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{args.arm}{args.study_suffix}.json"
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"saved: {path}")

    if args.format == "params":
        payload = [t.params for t in trials]
        print(json.dumps(payload if args.top_k > 1 else payload[0], indent=2))
        return

    if args.format == "summary":
        for t in trials:
            print(f"trial #{t.number}: {OBJECTIVE_METRIC}={t.value:.3f}")
            print(f"  params: {t.params}")
            print(f"  success_rate={t.user_attrs.get('eval/success_rate')} "
                  f"collision_rate={t.user_attrs.get('eval/collision_rate')}")
            print(f"  run_dir: {t.user_attrs.get('run_dir')}")
        return

    best = trials[0]
    overrides = (
        arm_overrides(args.arm, args.pref_budget, args.demo_budget)
        + params_to_overrides(best.params)
    )
    if args.format == "full":
        overrides = FIXED_OVERRIDES + overrides
    print(" ".join(overrides))


if __name__ == "__main__":
    main()
