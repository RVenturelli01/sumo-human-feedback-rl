"""Compute the minimum viable budget (90% rule) for one arm and record it.

Reads the finished ``budget_<arm>_<level>`` groups from W&B (same logic as
report_budget_curves.py), applies the rule — smallest level whose seed-mean of
BOTH sweep/mean_fast_return and sweep/success_rate retains >= 90% of the
largest level's improvement, with the next level up also passing — and merges
the result into a JSON file (default outputs/post_tuning/budgets.json):

    {"pref_soft": {"min_budget": 2000, "levels": {...}}, ...}

The orchestrator reads this file to start the final 5-seed runs.

Usage:
    python scripts/compute_min_budget.py --arm pref_soft
    python scripts/compute_min_budget.py --arm demo_1 --out ../outputs/post_tuning/budgets.json
"""

import argparse
import datetime
import json
from pathlib import Path

from report_budget_curves import collect, level_frame, minimum_budget


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True,
                        choices=["pref_soft", "pref_bernoulli", "demo_1", "demo_2"])
    parser.add_argument("--project", default="tuning-thesis")
    parser.add_argument("--out", default="../outputs/post_tuning/budgets.json",
                        help="JSON to merge into (relative to scripts/ by default).")
    parser.add_argument("--min-seeds", type=int, default=3,
                        help="Fail unless every level has at least this many finished seeds.")
    args = parser.parse_args()

    per_arm = collect(args.project)
    if args.arm not in per_arm:
        raise SystemExit(f"No budget_{args.arm}_* groups found in {args.project}.")
    df = level_frame(per_arm[args.arm])
    if not len(df):
        raise SystemExit(f"No finished runs in the budget groups for {args.arm}.")
    incomplete = df[df["n_seeds"] < args.min_seeds]
    if len(incomplete):
        raise SystemExit(
            f"Levels with fewer than {args.min_seeds} seeds for {args.arm}: "
            f"{list(incomplete.index)} — curve not complete yet."
        )

    minimo = minimum_budget(df)
    print(df.to_string())
    print(f"\n{args.arm}: minimum budget (90% rule) = {minimo}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(out_path.read_text()) if out_path.exists() else {}
    data[args.arm] = {
        "min_budget": int(minimo),
        "levels": {str(k): round(float(v), 3) for k, v in df["return_mean"].items()},
        "computed_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
