"""Toglie dalla cache le curve scaricate mentre la run era ancora in corso.

Fino a quando `curves.curve_from_wandb` non ha imparato a guardare lo stato,
una curva scaricata a run in esecuzione veniva salvata parziale e non piu'
riscaricata. Il danno non e' evidente: `curves.aggregate` fissa la griglia
comune sulla run **piu' corta** del gruppo, quindi un solo file parziale
accorcia in silenzio l'intera serie.

Come si riconosce un file avvelenato: la run risulta `finished` nell'indice ma
la sua curva finisce molto prima di quella delle sorelle dello stesso gruppo,
sulla stessa metrica. Il confronto e' fra pari invece che contro un valore
atteso, cosi' funziona su entrambi gli assi x (timestep dell'agente e
iterazioni) senza doverli distinguere.

    python plots/scripts/clean_curve_cache.py            # elenca soltanto
    python plots/scripts/clean_curve_cache.py --apply    # cancella
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

import pandas as pd

import _bootstrap  # noqa: F401  (mette plots/ nel path)
from rtplots.index import load_index
from rtplots.paths import CURVE_DIR

# project__runid__metrica.parquet, con la metrica che ha gli slash sostituiti
CACHE_RE = re.compile(r"^(?P<project>.+?)__(?P<run_id>[^_]+)__(?P<metric>.+)\.parquet$")
# Sotto questa frazione della sorella piu' lunga il file e' considerato parziale.
KEEP_RATIO = 0.9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true",
                        help="cancella davvero (senza, elenca soltanto)")
    parser.add_argument("--ratio", type=float, default=KEEP_RATIO)
    args = parser.parse_args()

    index = load_index()
    meta = index.set_index("run_id")[["group", "state"]].to_dict("index")

    # (gruppo, metrica) -> [(file, run_id, ultimo step)]
    buckets: dict[tuple, list] = defaultdict(list)
    for path in sorted(CURVE_DIR.glob("*.parquet")):
        m = CACHE_RE.match(path.name)
        if not m:
            continue
        info = meta.get(m.group("run_id"))
        if info is None:
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if df.empty or "step" not in df.columns:
            continue
        buckets[(info["group"], m.group("metric"))].append(
            (path, m.group("run_id"), float(df.step.max()), info["state"])
        )

    suspect = []
    for (group, metric), rows in buckets.items():
        if len(rows) < 2:
            continue                      # senza sorelle non c'e' un metro di paragone
        longest = max(r[2] for r in rows)
        if longest <= 0:
            continue
        for path, run_id, last, state in rows:
            # Una run interrotta e' legittimamente piu' corta: si guardano solo
            # quelle che il registro dice concluse.
            if state == "finished" and last < args.ratio * longest:
                suspect.append((path, group, metric, last, longest))

    if not suspect:
        print(f"{sum(len(v) for v in buckets.values())} curve in cache, nessuna parziale")
        return 0

    print(f"{len(suspect)} curve parziali di run concluse:\n")
    for path, group, metric, last, longest in sorted(suspect, key=lambda r: r[1]):
        print(f"  {path.name}")
        print(f"    {group} · {metric}: finisce a {last:,.0f} contro {longest:,.0f}"
              f" delle sorelle ({last / longest:.0%})")
    if not args.apply:
        print("\n(elenco soltanto; --apply per cancellarle e farle riscaricare)")
        return 0
    for path, *_ in suspect:
        path.unlink()
    print(f"\ncancellate {len(suspect)}: si riscaricano alla prossima figura")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
