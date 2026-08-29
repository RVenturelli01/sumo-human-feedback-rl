#!/usr/bin/env python
"""Ask the index how many runs and seeds exist per combination.

    python plots/scripts/list_runs.py --by arm
    python plots/scripts/list_runs.py --filter arm_family=hybrid --by arm demo_loss
    python plots/scripts/list_runs.py --filter arm=pref_soft --by query_budget --state any
"""
import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from rtplots.index import load_index
from rtplots.select import coverage, select_runs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--filter", nargs="*", default=[], help="es. arm=demo_1,demo_2 seed>=1")
    p.add_argument("--by", nargs="*", default=["arm", "budget_level"])
    p.add_argument("--state", default="finished", help="finished | any | crashed,running")
    p.add_argument("--full", action="store_true", help='print one line per run')
    args = p.parse_args()

    df = load_index()
    sel = select_runs(df, args.filter, state=args.state)
    print(f"[list] {len(sel)} run selezionati su {len(df)}")

    if args.full:
        cols = [c for c in ["run_id", "arm", "arm_family", "demo_loss", "pref_labels",
                            "query_budget", "demo_budget", "budget_level", "seed",
                            "state", "group"] if c in sel.columns]
        with pd.option_context("display.max_rows", None, "display.width", 250):
            print(sel[cols].sort_values(cols[1:5]).to_string(index=False))
        return

    cov = coverage(sel, args.by)
    with pd.option_context("display.max_rows", None, "display.width", 250):
        print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
