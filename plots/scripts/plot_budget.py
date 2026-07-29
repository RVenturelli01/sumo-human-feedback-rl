#!/usr/bin/env python
"""Curve di budget (eval finale vs livello di budget) su griglia di pannelli.

Ogni run vale un solo numero (la valutazione finale held-out, `sweep/*`),
aggregato sui seed per livello di budget — asse x logaritmico, errorbar invece
di banda continua (pochi livelli discreti, non una serie nel tempo). Stampa
anche il budget minimo per serie con la regola del 90%: il livello piu' piccolo
per cui la metrica scelta resta >=90% del livello massimo, con il successivo
che passa anche lui (stessa regola di `scripts/report_budget_curves.py`, qui
generalizzata a qualunque selezione/hue invece dei soli 4 bracci baseline).

L'elenco completo delle metriche disponibili (le `sweep/*`):
    python plots/scripts/plot_budget.py --list-metrics

Esempi:
    # Tutti gli arm pref/hybrid: budget query, return finale
    python plots/scripts/plot_budget.py --filter arm_family=pref,hybrid --name budget_pref

    # Solo i due demo-only: budget traiettorie, success rate
    python plots/scripts/plot_budget.py --filter arm_family=demo \
        --metric sweep/success_rate --budget-x demo_budget --name budget_demo_success
"""
import argparse

import matplotlib

matplotlib.use("Agg")  # niente display su questa macchina: mai un backend interattivo

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
    """Il budget minimo per ogni serie disegnata, con la regola del 90%."""
    print("\n[budget] budget minimo per serie (regola del 90%, vedi docstring):")
    for label, g in series.agg.groupby("label"):
        g = g.sort_values("step")
        minimo = minimum_budget(g["step"], {"metric": g["mean"]})
        levels = ", ".join(str(int(v)) for v in sorted(g["step"].unique()))
        print(f"  {label}: livelli [{levels}] -> minimo {minimo}")


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
