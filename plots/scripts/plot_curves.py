#!/usr/bin/env python
"""Curve di apprendimento (metrica vs environment timesteps) su griglia di pannelli.

E' lo script generico: si sceglie cosa filtrare, cosa plottare (--metric), cosa
mettere sulle righe, sulle colonne e cosa distinguere con il colore. Le run vanno
prese dalla history W&B (niente file locali raggiungibili da qui, vedi
`docs/analysis-pipeline-guide.md`): la prima volta ogni run costa una richiesta,
poi resta in cache in `plots/.cache/curves/`.

L'elenco completo delle metriche disponibili:
    python plots/scripts/plot_curves.py --list-metrics

Con --runs-file la figura e' esattamente quella dell'anteprima del selettore.

Esempi:
    # Una serie per arm, colore automatico (tutti gli arm della campagna)
    python plots/scripts/plot_curves.py --name learning_all

    # Solo i due hybrid, colonne = etichette di preferenza
    python plots/scripts/plot_curves.py --filter arm_family=hybrid \
        --cols pref_labels --hue demo_loss --name hybrid_learning

    # Una diagnostica del reward model invece del return
    python plots/scripts/plot_curves.py --metric reward/loss_pref_val \
        --filter arm_family=pref --name pref_reward_loss
"""
import argparse

import matplotlib

matplotlib.use("Agg")  # niente display su questa macchina: mai un backend interattivo

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
                   help="finestra della media mobile sui seed (default 5)")
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
