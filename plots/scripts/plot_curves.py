#!/usr/bin/env python
"""Learning curves: a metric against environment timesteps, on a panel grid.

The general script: choose what to filter, what to plot, what goes on rows and
columns, and what the colour separates. Curves come from the W&B history, so the
first time each run costs a request; after that it is cached.

    python plots/scripts/plot_curves.py --list-metrics

With --runs-file the figure is exactly the one previewed in the selector.

Examples:

    # one series per method, colours chosen automatically
    python plots/scripts/plot_curves.py --name learning_all

    # the two hybrids, columns by preference label
    python plots/scripts/plot_curves.py --filter arm_family=hybrid \
        --cols pref_labels --hue demo_loss --name hybrid_learning

    # a reward-model diagnostic instead of the return
    python plots/scripts/plot_curves.py --metric reward/loss_pref_val \
        --filter arm_family=pref --name pref_reward_loss
"""
import argparse

import matplotlib

matplotlib.use("Agg")  # no display here: never an interactive backend

import _bootstrap  # noqa: F401,E402
from _common import (add_aggregation_args, add_grid_args, add_output_args,  # noqa: E402
                     add_selection_args, report, spec_from_args)
from rtplots import figure as F  # noqa: E402
from rtplots import style as S  # noqa: E402
from rtplots.index import load_index  # noqa: E402
from rtplots.metrics import DEFAULT_CURVE_METRIC, METRIC_GROUPS, metric_info  # noqa: E402


def print_metrics():
    for group, items in METRIC_GROUPS:
        print(f"\n{group}")
        for key, label, _, kind, _ in items:
            if kind == "curve":
                print(f"  {key:<55} {label}")
    print("\nQualsiasi altra chiave loggata su W&B e' comunque accettata "
          "(l'asse x si indovina dal prefisso: agent/* -> timesteps, il resto -> iterations).")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric", default=None,
                   help=f"cosa plottare (default {DEFAULT_CURVE_METRIC}); "
                        "--list-metrics per l'elenco")
    p.add_argument("--list-metrics", action="store_true", help="stampa le metriche note ed esce")
    add_selection_args(p)
    add_aggregation_args(p)
    p.add_argument("--smooth", type=int, default=None,
                   help='moving-average window over seeds (default 5)')
    p.add_argument("--grid-points", type=int, default=None)
    p.add_argument("--xmax", type=float, default=None, help="in timestep, es. 1e6")
    add_grid_args(p)
    p.add_argument("--xscale", type=float, default=1e6)
    add_output_args(p, default_name="curves")
    p.add_argument("--ylabel", default=None, help="etichetta asse y (default: dalla metrica)")
    args = p.parse_args()
    if args.list_metrics:
        return print_metrics()

    if args.metric and metric_info(args.metric)["kind"] != "curve":
        raise SystemExit(f"«{args.metric}» e' una metrica di eval finale (sweep/*): "
                         f"usa plot_budget.py, non plot_curves.py.")

    S.apply_style(args.font_scale)
    spec = spec_from_args(args, kind="curve")
    try:
        fig, series = F.build(load_index(), spec)
    except ValueError as exc:
        raise SystemExit(str(exc))
    report(series, args, S.save(fig, args.outdir, args.name, formats=args.formats))


if __name__ == "__main__":
    main()
