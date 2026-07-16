"""Export the best Optuna trial of an arm as Hydra overrides / JSON params.

Reads the shared journal, loads the study of the given arm and prints the
best trial in the requested format:

* ``overrides`` (default) — space-separated Hydra overrides (arm presets +
  best params), ready to append to a ``test_hybrid_SAC.py`` command line.
* ``full``     — like ``overrides`` but prefixed with the tuner's fixed
  overrides too: a complete, self-contained config for final runs.
* ``params``   — the raw Optuna param dict as JSON. A list of the top-k dicts
  is valid input for ``tune_hybrid_sac.py --enqueue-params`` (warm starts).
* ``summary``  — human-readable report of the top-k trials.

Usage:
    python scripts/export_best_config.py --arm pref_soft
    python scripts/export_best_config.py --arm demo_2 --format params --top-k 3 > warm.json
"""

import argparse
import json

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
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--pref-budget", type=int, default=5000)
    parser.add_argument("--demo-budget", type=int, default=500)
    args = parser.parse_args()

    storage = JournalStorage(
        JournalFileBackend(args.storage_path, lock_obj=JournalFileOpenLock(args.storage_path))
    )
    study = optuna.load_study(study_name=f"hybrid_sac_{args.arm}", storage=storage)
    trials = completed_trials_sorted(study)[: args.top_k]
    if not trials:
        raise SystemExit(f"No completed trials for arm {args.arm}.")

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
