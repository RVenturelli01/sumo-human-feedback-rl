#!/usr/bin/env python
"""Scarica in cache curve e/o summary, cosi' selettore e script non aspettano W&B.

Tutto qui passa da W&B (niente file locali raggiungibili, vedi
`docs/analysis-pipeline-guide.md`): la prima apertura di una figura su una
selezione grande puo' pagare una richiesta per run. Questo script riempie la
cache in un colpo solo, in parallelo.

    python plots/scripts/prefetch_curves.py                       # curva default + tutte le sweep/*
    python plots/scripts/prefetch_curves.py --filter arm_family=hybrid
    python plots/scripts/prefetch_curves.py --metric reward/loss_pref_val --no-summary
"""
import argparse

import _bootstrap  # noqa: F401
from rtplots.budget import load_summaries
from rtplots.curves import DEFAULT_CURVE_METRIC, load_curves
from rtplots.index import load_index
from rtplots.select import select_runs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--filter", nargs="*", default=[], help="filtri come negli altri script")
    p.add_argument("--metric", default=DEFAULT_CURVE_METRIC,
                   help="metrica-curva da scaricare (default: return di apprendimento)")
    p.add_argument("--state", default="finished")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-curve", action="store_true", help="salta la curva di apprendimento")
    p.add_argument("--no-summary", action="store_true",
                   help="salta l'eval finale (sweep/*, serve alle curve di budget)")
    args = p.parse_args()

    sel = select_runs(load_index(), args.filter, state=args.state)
    print(f"[prefetch] {len(sel)} run selezionati")

    if not args.no_curve:
        curves = load_curves(sel, metric=args.metric, workers=args.workers)
        print(f"[prefetch] curve «{args.metric}»: {curves.run_id.nunique()}/{len(sel)} run in cache")

    if not args.no_summary:
        summaries = load_summaries(sel, workers=args.workers)
        print(f"[prefetch] summary (sweep/*): {len(summaries)}/{len(sel)} run in cache")


if __name__ == "__main__":
    main()
