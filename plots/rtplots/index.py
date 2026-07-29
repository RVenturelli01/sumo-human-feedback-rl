"""Indice dei run: metadati (una riga per run) letti da W&B e messi in cache.

La cache sta in `plots/.cache/` (override `RTPLOTS_CACHE`). L'aggiornamento e'
incrementale: la lista dei run (id/nome/stato/tag) viene sempre riscaricata, la
config completa solo per i run non ancora in cache o rimasti non finiti (la
config di una run "running" puo' ancora cambiare). Le run cancellate da W&B
escono dall'indice al primo aggiornamento.

Come si legge una run sta in `rtplots/source.py`: qui ci si limita a scaricare,
unire e mettere in cache.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import source
from .paths import DEFAULT_PROJECTS, INDEX_CSV, INDEX_PARQUET, ensure_dirs, wandb_path


def build_index(force: bool = False, workers: int = 8,
                projects: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    """(Ri)costruisce l'indice dei run sui progetti indicati e lo salva in cache."""
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
        # I run non finiti vanno riletti: la config cambia fino alla fine.
        known = set(cached.loc[cached.state == "finished", "run_id"])
    todo = [t for t in runs if t[0].id not in known]
    if verbose:
        print(f"[index] config da scaricare: {len(todo)} (in cache: {len(known)})")

    def fetch(item):
        run, project = item
        try:
            # Indispensabile: api.runs() restituisce run.config == {} finche' non
            # si forza il caricamento della config completa.
            run.load(force=True)
            return source.row(run, project)
        except Exception as exc:  # run corrotto o rimosso: non blocca l'indice
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

    # stato/tag sempre aggiornati dalla lista (economici, nessun load()). Il join
    # a destra fa anche da pulizia: quello che non c'e' piu' su W&B esce
    # dall'indice. Vale solo per i progetti appena riletti: gli altri restano in
    # cache come sono, altrimenti --projects cancellerebbe il resto dell'indice.
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
    """Carica l'indice dalla cache (costruendolo se manca)."""
    if not INDEX_PARQUET.exists():
        if not auto_build:
            raise FileNotFoundError(
                f"Indice assente: {INDEX_PARQUET}. Lancia plots/scripts/build_index.py"
            )
        return build_index()
    return pd.read_parquet(INDEX_PARQUET)
