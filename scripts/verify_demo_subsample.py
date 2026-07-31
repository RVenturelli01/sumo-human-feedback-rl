"""Check that every arm sees the same demonstrations at the same budget.

Two modes:

``budgets`` (default)
    Recompute the selection for a list of budgets straight from the dataset
    and print one row per budget: how many demonstrations, how many
    transitions, and the fingerprint of the selected SET. Also verifies that
    the budgets are nested, so a larger budget only ever adds demonstrations.
    This is the "same 10 demos at n=10, whatever the arm" property at its
    source: the selection depends on the budget and the shared seed, and on
    nothing else — not the arm, not the run seed.

``runs``
    Read ``demo_subsample.json`` from finished run directories and group the
    fingerprints by budget. Every run at a given budget must show the same
    fingerprint; anything else is reported as a mismatch, which is how past
    runs launched with inconsistent seeds can be found.

Examples::

    python scripts/verify_demo_subsample.py
    python scripts/verify_demo_subsample.py --budgets 10 100 1000 --seeds 1000 1001
    python scripts/verify_demo_subsample.py runs outputs/*/
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

from human_feedback_rl.common.demo_subsampling import (
    DEMO_SUBSAMPLE_SEED,
    dataset_fingerprint,
    indices_fingerprint,
    select_demo_indices,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"
DEFAULT_BUDGETS = (1, 10, 20, 50, 100, 200, 500, 1000, 2723)


def load_lengths(name: str) -> list:
    with open(DATA_DIR / name, "rb") as f:
        trajectories = pickle.load(f)
    return [len(trajectory) for trajectory in trajectories]


def check_budgets(args) -> int:
    lengths = load_lengths(args.dataset)
    n_available = len(lengths)
    print(f"dataset            {args.dataset}")
    print(f"trajectories       {n_available}")
    print(f"transitions        {sum(lengths)}")
    print(f"dataset fingerprint {dataset_fingerprint(lengths)[:12]}")
    print()

    failures = 0
    for seed in args.seeds:
        label = "shared seed" if seed == DEMO_SUBSAMPLE_SEED else "seed"
        print(f"--- {label} {seed} ---")
        header = f"{'budget':>8}  {'kept':>6}  {'transitions':>12}  {'fingerprint':<14}  nested"
        print(header)
        print("-" * len(header))

        previous_set, previous_budget = None, None
        for budget in args.budgets:
            # A trajectory budget cannot exceed the dataset; a transition
            # budget can, and simply keeps everything.
            if not args.transitions and budget > n_available:
                continue
            indices = select_demo_indices(
                n_available=n_available,
                lengths=lengths,
                n_trajectories=None if args.transitions else budget,
                n_transitions=budget if args.transitions else None,
                seed=seed,
            )
            current = set(int(i) for i in indices)
            if previous_set is None:
                nested = "-"
            elif previous_set <= current:
                nested = "yes"
            else:
                missing = len(previous_set - current)
                nested = f"NO ({missing} dropped from {previous_budget})"
                failures += 1
            print(
                f"{budget:>8}  {len(indices):>6}  "
                f"{sum(lengths[i] for i in indices):>12}  "
                f"{indices_fingerprint(indices)[:12]:<14}  {nested}"
            )
            previous_set, previous_budget = current, budget
        print()

    if len(args.seeds) > 1:
        print(
            "Different seeds SHOULD give different fingerprints; that is what\n"
            "the shared constant protects against. Same seed + same budget\n"
            "always gives the same fingerprint, for every arm."
        )
    return 1 if failures else 0


def check_runs(args) -> int:
    manifests = []
    for directory in args.run_dirs:
        path = Path(directory) / "demo_subsample.json"
        if not path.exists():
            print(f"skip (no demo_subsample.json): {directory}")
            continue
        with open(path) as f:
            manifests.append((Path(directory).name, json.load(f)))

    if not manifests:
        print("No manifests found. Runs launched before this instrumentation")
        print("do not have one; relaunch them to get a verifiable fingerprint.")
        return 1

    by_budget = defaultdict(list)
    for name, manifest in manifests:
        key = (
            manifest.get("budget_n_trajectories"),
            manifest.get("budget_n_transitions"),
            manifest.get("dataset_fingerprint"),
        )
        by_budget[key].append((name, manifest))

    failures = 0
    for (n_traj, n_trans, _), entries in sorted(
        by_budget.items(), key=lambda item: str(item[0])
    ):
        budget = f"n_trajectories={n_traj}" if n_traj is not None else f"n_transitions={n_trans}"
        fingerprints = {manifest["fingerprint"] for _, manifest in entries}
        status = "OK" if len(fingerprints) == 1 else "MISMATCH"
        print(f"{budget}  ({len(entries)} runs)  {status}")
        if len(fingerprints) == 1:
            print(f"    fingerprint {fingerprints.pop()[:12]}")
            continue
        failures += 1
        for name, manifest in sorted(entries):
            print(
                f"    {manifest['fingerprint'][:12]}  "
                f"seed={manifest['subsample_seed']}  {name}"
            )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="mode")

    budgets = subparsers.add_parser("budgets", help="recompute selections from the dataset")
    budgets.add_argument("--dataset", default="expert_trajectories_no_collision.pkl")
    budgets.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    budgets.add_argument(
        "--seeds", type=int, nargs="+", default=[DEMO_SUBSAMPLE_SEED],
        help=f"subsample seeds to report (default: the shared {DEMO_SUBSAMPLE_SEED})",
    )
    budgets.add_argument(
        "--transitions", action="store_true",
        help="read the budgets as transition counts instead of trajectory counts",
    )
    budgets.set_defaults(func=check_budgets)

    runs = subparsers.add_parser("runs", help="compare manifests of finished runs")
    runs.add_argument("run_dirs", nargs="+")
    runs.set_defaults(func=check_runs)

    args = parser.parse_args()
    if args.mode is None:
        args = parser.parse_args(["budgets"])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
