"""Caricamento e aggregazione delle curve di budget.

A differenza delle curve di apprendimento (`curves.py`, una serie storica per
run), qui ogni run vale **un solo numero**: la valutazione finale held-out,
scritta da `log_sweep_summary` come `run.summary["sweep/<metrica>"]` (vedi
`scripts/_common.py:evaluate`). Si aggrega sui seed per livello di budget,
non nel tempo — stessa logica di `scripts/report_budget_curves.py`, generalizzata
per passare dalla stessa selezione/griglia del resto del motore invece di un
regex sul nome del gruppo.

`run.summary` e' disponibile senza il costoso `run.load(force=True)` che serve
invece per la config (vedi `source.py`): una richiesta per run, messa in cache
su disco perche' non cambia piu' una volta che la run e' `finished`.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import CACHE_DIR, ensure_dirs, wandb_path

SUMMARY_DIR = CACHE_DIR / "summary"
PASS_RATIO = 0.90  # regola del budget minimo: entrambe le metriche >=90% del livello massimo


def _cache_file(project: str, run_id: str) -> Path:
    return SUMMARY_DIR / f"{project}__{run_id}.json"


def load_summary(run_id: str, project: str, state: str = "finished") -> dict:
    """Il summary completo di una run (tutte le chiavi `sweep/*` incluse).

    Le run non ancora `finished` non si cachano: il summary puo' ancora cambiare.
    """
    ensure_dirs()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_file(project, run_id)
    if state == "finished" and cache_file.exists():
        return json.loads(cache_file.read_text())
    import wandb

    api = wandb.Api()
    run = api.run(f"{wandb_path(project)}/{run_id}")
    data = dict(run.summary or {})
    if run.state == "finished":
        cache_file.write_text(json.dumps(data, default=str))
    return data


def load_summaries(index: pd.DataFrame, workers: int = 8, verbose: bool = True) -> pd.DataFrame:
    """Summary di tutte le run dell'indice, in formato tidy [run_id, <metrica>...]."""
    def fetch(rec):
        return rec["run_id"], load_summary(rec["run_id"], rec["project"], rec.get("state", ""))

    records = index[["run_id", "project", "state"]].to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pairs = list(pool.map(fetch, records))
    rows = [{"run_id": run_id, **summary} for run_id, summary in pairs if summary]
    missing = len(pairs) - len(rows)
    if verbose and missing:
        print(f"[budget] summary assente per {missing}/{len(pairs)} run")
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["run_id"])


def aggregate(index: pd.DataFrame, group_cols, x_col: str, metric: str,
             band: str = "se") -> pd.DataFrame:
    """Media sui seed di `metric` per ogni combinazione di (*group_cols, x_col).

    Ritorna [*group_cols, step, mean, lo, hi, n_seeds] — stesse colonne
    dell'output di `curves.aggregate`, cosi' `grid.py` disegna entrambe con lo
    stesso codice (qui `step` e' il livello di budget, non il tempo).
    """
    group_cols = list(group_cols)
    summaries = load_summaries(index)
    if metric not in summaries.columns:
        return pd.DataFrame(columns=[*group_cols, "step", "mean", "lo", "hi", "n_seeds"])
    df = index[["run_id", x_col, *group_cols]].merge(summaries[["run_id", metric]], on="run_id")
    df = df.dropna(subset=[x_col, metric])
    out = []
    for key, g in df.groupby([x_col, *group_cols], dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        x_val = key[0]
        vals = g[metric].to_numpy(dtype=float)
        n = len(vals)
        mean = vals.mean()
        if band == "std":
            half = vals.std(ddof=1) if n > 1 else 0.0
            lo, hi = mean - half, mean + half
        elif band == "ci95":
            se = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            lo, hi = mean - 1.96 * se, mean + 1.96 * se
        elif band == "iqr":
            lo, hi = np.percentile(vals, 25), np.percentile(vals, 75)
        elif band == "minmax":
            lo, hi = vals.min(), vals.max()
        elif band == "none":
            lo = hi = mean
        else:  # 'se' (default)
            se = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            lo, hi = mean - se, mean + se
        row = {"step": x_val, "mean": mean, "lo": lo, "hi": hi, "n_seeds": n}
        for col, val in zip(group_cols, key[1:]):
            row[col] = val
        out.append(row)
    if not out:
        return pd.DataFrame(columns=[*group_cols, "step", "mean", "lo", "hi", "n_seeds"])
    return pd.DataFrame(out).sort_values("step")


def _relative_score(series: pd.Series, reference: float) -> pd.Series:
    """Frazione del miglioramento sul livello peggiore ancora presente.

    Robusto a return negativi: 1.0 al livello di riferimento (il piu' alto),
    0.0 al livello peggiore.
    """
    lo = series.min()
    span = reference - lo
    if span <= 0:
        return pd.Series(1.0, index=series.index)
    return (series - lo) / span


def minimum_budget(levels: pd.Series, metrics: dict[str, pd.Series]) -> float | None:
    """Il livello piu' piccolo per cui TUTTE le metriche in `metrics` restano
    >=90% del livello massimo, con anche il livello successivo che passa
    (regola del 90%, come `scripts/report_budget_curves.py:minimum_budget`)."""
    if len(levels) < 2:
        return None
    full = levels.max()
    passing = pd.Series(True, index=levels.index)
    for series in metrics.values():
        ref = series.loc[levels == full]
        if ref.empty:
            return None
        passing &= _relative_score(series, ref.iloc[0]) >= PASS_RATIO
    ordered = sorted(levels.unique())
    passing_by_level = passing.groupby(levels).any()
    for i, level in enumerate(ordered):
        upper_ok = passing_by_level.get(ordered[i + 1], True) if i + 1 < len(ordered) else True
        if passing_by_level.get(level, False) and upper_ok:
            return level
    return full
