"""Run metadata from W&B, one row per run, cached on disk.

The cache lives in `plots/.cache/`. Updates are incremental: the run list is
always refetched, the full config only for runs that are new or still running,
since a running run can still change its config. Runs deleted from W&B leave
the index at the next update.

How a single run is read is in `source.py`; this module only downloads, merges
and caches.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import source
from .paths import DEFAULT_PROJECTS, INDEX_CSV, INDEX_PARQUET, ensure_dirs, wandb_path


def build_index(force: bool = False, workers: int = 8,
                projects: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Rebuild the index over the given projects and cache it."""
    import wandb

    ensure_dirs()
    projects = list(projects or DEFAULT_PROJECTS)
    cached = pd.DataFrame()
    if INDEX_PARQUET.exists() and not force:
        cached = pd.read_parquet(INDEX_PARQUET)

    api = wandb.Api()
    runs = []  # (run, progetto)
    for project in projects:
        found = list(api.runs(wandb_path(project), per_page=200))
        runs += [(r, project) for r in found]
        if verbose:
            print(f"[index] {len(found)} run su {project}")

    known = set()
    if not cached.empty:
        # Unfinished runs must be reread: their config changes until the end.
        known = set(cached.loc[cached.state == "finished", "run_id"])
    todo = [t for t in runs if t[0].id not in known]
    if verbose:
        print(f"[index] config da scaricare: {len(todo)} (in cache: {len(known)})")

    def fetch(item):
        run, project = item
        try:
            # Required: api.runs() returns run.config == {} until the full
            # config is loaded explicitly.
            run.load(force=True)
            return source.row(run, project)
        except Exception as exc:  # a broken or deleted run must not stop the index
            return dict(run_id=run.id, name=run.name, state=run.state, project=project,
                        tags=",".join(run.tags or []), error=str(exc))

    rows = []
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, row in enumerate(pool.map(fetch, todo), 1):
                rows.append(row)
                if verbose and i % 50 == 0:
                    print(f"[index]   {i}/{len(todo)}")

    new = pd.DataFrame(rows)
    df = pd.concat([cached, new], ignore_index=True) if not cached.empty else new
    df = df.drop_duplicates(subset="run_id", keep="last")

    # State and tags always come fresh from the list, which is cheap. The right
    # join doubles as cleanup: what is gone from W&B leaves the index. Only for
    # the projects just refetched, or --projects would wipe the rest.
    live = pd.DataFrame([{"run_id": r.id, "state": r.state,
                          "tags": ",".join(r.tags or [])} for r, _ in runs])
    other = df[~df.project.isin(projects)]
    fresh = (df[df.project.isin(projects)]
             .drop(columns=["state", "tags"]).merge(live, on="run_id", how="right"))
    df = pd.concat([other, fresh], ignore_index=True) if not other.empty else fresh

    df.to_parquet(INDEX_PARQUET, index=False)
    df.to_csv(INDEX_CSV, index=False)
    if verbose:
        print(f"[index] scritto {INDEX_PARQUET} ({len(df)} run)")
    return df


def load_index(auto_build: bool = True) -> pd.DataFrame:
    """Load the index from the cache, building it if there is none."""
    if not INDEX_PARQUET.exists():
        if not auto_build:
            raise FileNotFoundError(
                f"Indice assente: {INDEX_PARQUET}. Lancia plots/scripts/build_index.py"
            )
        return build_index()
    return pd.read_parquet(INDEX_PARQUET)
