#!/usr/bin/env python
"""Budget curves: final evaluation against budget level, on a panel grid.

Each run is worth one number, the final held-out evaluation, aggregated over
seeds per budget level. The x axis is logarithmic and the spread is drawn as
error bars rather than a band, because the levels are few and discrete.

It also prints the minimum budget per series under the 90% rule: the smallest
level where the chosen metric stays at 90% of its best, with the level after it
passing as well.

    python plots/scripts/plot_budget.py --list-metrics

Examples:

    # every preference and hybrid method: query budget, final return
    python plots/scripts/plot_budget.py --filter arm_family=pref,hybrid \
        --name budget_pref

    # the demonstration-only methods: trajectory budget, success rate
    python plots/scripts/plot_budget.py --filter arm_family=demo \
        --metric sweep/success_rate --budget-x demo_budget --name budget_demo
"""
import argparse

import matplotlib

matplotlib.use("Agg")  # no display here: never an interactive backend

import _bootstrap  # noqa: F401,E402
from _common import (add_aggregation_args, add_grid_args, add_output_args,  # noqa: E402
                     add_selection_args, report, spec_from_args)
from rtplots import figure as F  # noqa: E402
from rtplots import style as S  # noqa: E402
from rtplots.budget import minimum_budget  # noqa: E402
from rtplots.index import load_index  # noqa: E402
from rtplots.metrics import DEFAULT_SUMMARY_METRIC, METRIC_GROUPS, metric_info  # noqa: E402


def print_metrics():
    for group, items in METRIC_GROUPS:
        printed = False
        for key, label, _, kind, _ in items:
            if kind == "summary":
                if not printed:
                    print(f"\n{group}")
                    printed = True
                print(f"  {key:<30} {label}")


def print_minimum_budgets(series: F.Series):
    """The minimum budget for each series drawn, under the 90% rule."""
    print('\n[budget] minimum budget per series, under the 90% rule:')
    for label, g in series.agg.groupby("label"):
        g = g.sort_values("step")
        smallest = minimum_budget(g["step"], {"metric": g["mean"]})
        levels = ", ".join(str(int(v)) for v in sorted(g["step"].unique()))
        print(f"  {label}: levels [{levels}] -> minimum {smallest}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric", default=None,
                   help=f"cosa plottare (default {DEFAULT_SUMMARY_METRIC}); "
                        "--list-metrics per l'elenco")
    p.add_argument("--list-metrics", action="store_true", help="stampa le metriche note ed esce")
    p.add_argument("--budget-x", default=None, choices=list(F.BUDGET_X_CHOICES),
                   help="asse x: budget_level (default, robusto per ogni arm), "
                        "query_budget o demo_budget")
    add_selection_args(p)
    add_aggregation_args(p)
    add_grid_args(p)
    add_output_args(p, default_name="budget_curves")
    p.add_argument("--ylabel", default=None, help="etichetta asse y (default: dalla metrica)")
    args = p.parse_args()
    if args.list_metrics:
        return print_metrics()

    if args.metric and metric_info(args.metric)["kind"] != "summary":
        raise SystemExit(f"«{args.metric}» e' una metrica di curva, non di eval finale: "
                         f"usa plot_curves.py, non plot_budget.py.")

    S.apply_style(args.font_scale)
    spec = spec_from_args(args, kind="budget")
    try:
        fig, series = F.build(load_index(), spec)
    except ValueError as exc:
        raise SystemExit(str(exc))
    report(series, args, S.save(fig, args.outdir, args.name, formats=args.formats))
    print_minimum_budgets(series)


if __name__ == "__main__":
    main()
