"""Drop cached curves that were downloaded while the run was still going.

A curve fetched from a running run used to be saved partial and never fetched
again. The damage is quiet: `curves.aggregate` fits the shared grid to the
shortest run in the group, so one partial file shortens the whole series.

A poisoned file looks like this: the run is finished in the index, but its curve
ends well before its siblings on the same metric. Comparing peers rather than
some expected value means this works on both x axes without telling them apart.

    python plots/scripts/clean_curve_cache.py            # list only
    python plots/scripts/clean_curve_cache.py --apply    # delete
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
# Below this fraction of the longest sibling, a file counts as partial.
KEEP_RATIO = 0.9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true",
                        help='actually delete; without it, only list')
    parser.add_argument("--ratio", type=float, default=KEEP_RATIO)
    args = parser.parse_args()

    index = load_index()
    meta = index.set_index("run_id")[["group", "state"]].to_dict("index")

    # (group, metric) -> [(file, run_id, last step)]
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
            continue                      # with no siblings there is nothing to compare
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
              f" of its siblings ({last / longest:.0%})")
    if not args.apply:
        print("\n(elenco soltanto; --apply per cancellarle e farle riscaricare)")
        return 0
    for path, *_ in suspect:
        path.unlink()
    print(f"\ndeleted {len(suspect)}: they will be refetched by the next figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
