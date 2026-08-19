"""Caricamento e aggregazione delle curve di apprendimento.

A differenza del progetto di ispirazione (file `.npz` locali + fallback W&B),
qui la sola fonte e' la history W&B: i log locali (`metrics.jsonl`) vivono sul
server dove giri il training, non sulla macchina da cui si analizza (vedi
`docs/analysis-pipeline-guide.md`). Ogni run non ancora in cache costa quindi
una richiesta di rete; i risultati restano in cache su disco (`plots/.cache/`)
e in memoria per il processo del selettore, che ridisegna in continuazione.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .metrics import DEFAULT_CURVE_METRIC, metric_info
from .paths import CURVE_DIR, ensure_dirs, wandb_path

# Cache in memoria condivisa (serve al selettore, che ridisegna in continuazione).
# La chiave include il progetto: il run_id e' unico solo al suo interno.
_MEM: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
MAX_MEM_CURVES = 4000


def clear_cache() -> None:
    _MEM.clear()


def _mem_get(key):
    df = _MEM.get(key)
    if df is not None:
        _MEM.move_to_end(key)
    return df


def _mem_put(key, df) -> None:
    _MEM[key] = df
    while len(_MEM) > MAX_MEM_CURVES:
        _MEM.popitem(last=False)


def curve_from_wandb(run_id: str, project: str, metric: str = DEFAULT_CURVE_METRIC,
                     cache: bool = True, samples: int = 2000,
                     state: str = "") -> pd.DataFrame:
    """Curva [step, ret] di una run, sul suo asse x naturale (vedi metrics.py).

    Una run ancora in corso viene scaricata ma **non** salvata su disco: la sua
    curva e' destinata ad allungarsi, e una cache parziale resterebbe li' per
    sempre accorciando in silenzio ogni figura che la usa (`aggregate` fissa la
    griglia comune sulla run piu' corta del gruppo). Stessa regola gia' adottata
    da `budget.load_summary` per i summary.
    """
    ensure_dirs()
    info = metric_info(metric)
    step_key = info["step_key"]
    cache_file = CURVE_DIR / f"{project}__{run_id}__{metric.replace('/', '_')}.parquet"
    if cache and cache_file.exists():
        return pd.read_parquet(cache_file)
    import wandb

    api = wandb.Api()
    run = api.run(f"{wandb_path(project)}/{run_id}")
    hist = run.history(keys=[step_key, metric], samples=samples, pandas=True)
    if hist is None or hist.empty or step_key not in hist.columns or metric not in hist.columns:
        df = pd.DataFrame(columns=["step", "ret"])
    else:
        df = (pd.DataFrame({"step": hist[step_key].astype(float), "ret": hist[metric].astype(float)})
              .dropna(subset=["step", "ret"]).sort_values("step"))
    if cache and (state or run.state) == "finished":
        df.to_parquet(cache_file, index=False)
    return df


def load_curve(run_id: str, metric: str = DEFAULT_CURVE_METRIC,
              project: str | None = None, state: str = "") -> pd.DataFrame | None:
    project = project or ""
    mem_key = (project, run_id, metric)
    cached = _mem_get(mem_key)
    if cached is not None:
        return cached if not cached.empty else None
    try:
        df = curve_from_wandb(run_id, project=project, metric=metric, state=state)
    except Exception:
        return None
    _mem_put(mem_key, df)
    return df if not df.empty else None


def load_curves(index: pd.DataFrame, metric: str = DEFAULT_CURVE_METRIC,
                verbose: bool = True, workers: int = 8) -> pd.DataFrame:
    """Curve di tutti i run dell'indice, in formato tidy [run_id, step, ret]."""
    run_ids = list(index.run_id)
    projects = dict(zip(index.run_id, index.project)) if "project" in index.columns else {}
    states = dict(zip(index.run_id, index.state)) if "state" in index.columns else {}

    def fetch(run_id):
        return run_id, load_curve(run_id, metric=metric, project=projects.get(run_id),
                                  state=states.get(run_id, ""))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pairs = list(pool.map(fetch, run_ids))

    frames, missing = [], []
    for run_id, c in pairs:
        if c is None or c.empty:
            missing.append(run_id)
            continue
        frames.append(c.assign(run_id=run_id))
    if verbose and missing:
        print(f"[curves] «{metric}» mancante per {len(missing)}/{len(run_ids)} run "
              f"(chiave non loggata da quel run, o run troppo corta)")
    if not frames:
        return pd.DataFrame(columns=["run_id", "step", "ret"])
    return pd.concat(frames, ignore_index=True)


# --- aggregazione sui seed --------------------------------------------------

# Sotto questa frazione della run piu' lunga del gruppo il taglio va segnalato.
TRUNCATION_RATIO = 0.9


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    return pd.Series(y).rolling(window, min_periods=1).mean().to_numpy()


def aggregate(curves: pd.DataFrame, meta: pd.DataFrame, group_cols,
              band: str = "se", smooth: int = 1, grid_points: int | None = None,
              xmax: float | None = None) -> pd.DataFrame:
    """Media sui seed con banda di incertezza.

    Ogni run viene interpolato su una griglia comune al gruppo, poi lisciato con
    media mobile. band: se | std | ci95 | iqr | minmax.
    Ritorna [*group_cols, step, mean, lo, hi, n_seeds].
    """
    group_cols = list(group_cols)
    df = curves.merge(meta[["run_id", *group_cols]], on="run_id", how="inner")
    out = []
    # Una run piu' corta delle sorelle accorcia tutta la serie, e finora lo
    # faceva in silenzio: chi guarda la figura vede una curva interrotta senza
    # sapere perche'.
    truncated = []
    # Una metrica che nasce a meta' run (alpha/* prima che le preferenze
    # bastino a stimarne la dispersione) esiste solo da un certo step in poi.
    # La griglia deve partire da li': `np.interp` FUORI dall'intervallo dei dati
    # non estrapola, replica il primo valore, quindi una griglia che parte da 0
    # disegnava un tratto piatto lungo quanto il buco, indistinguibile da una
    # misura vera. Il bordo destro prendeva gia' l'intersezione fra i seed; ora
    # anche il sinistro.
    late_start = []
    for key, g in df.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        runs = list(g.groupby("run_id"))
        ends = {rid: float(r["step"].max()) for rid, r in runs}
        starts = {rid: float(r["step"].min()) for rid, r in runs}
        t_end = min(ends.values())
        t_start = max(starts.values())
        longest = max(ends.values())
        if longest > 0 and t_end < TRUNCATION_RATIO * longest:
            truncated.append({"run_id": min(ends, key=ends.get),
                              "end": t_end, "longest": longest})
        if t_start > min(starts.values()):
            late_start.append({"run_id": max(starts, key=starts.get),
                               "start": t_start, "earliest": min(starts.values())})
        if xmax is not None:
            t_end = min(t_end, xmax)
        if not t_end > t_start:
            # Nessun intervallo comune ai seed: una serie di un punto solo
            # ingannerebbe piu' di quanto informi.
            continue
        n_pts = grid_points or int(np.median([len(r) for _, r in runs]))
        n_pts = max(int(n_pts), 2)
        grid = np.linspace(t_start, t_end, n_pts)
        mat = np.vstack([
            _smooth(np.interp(grid, r["step"].to_numpy(), r["ret"].to_numpy()), smooth)
            for _, r in runs
        ])
        mean = mat.mean(axis=0)
        n = mat.shape[0]
        if band == "std":
            half = mat.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
            lo, hi = mean - half, mean + half
        elif band == "ci95":
            se = mat.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
            lo, hi = mean - 1.96 * se, mean + 1.96 * se
        elif band == "iqr":
            lo, hi = np.percentile(mat, 25, axis=0), np.percentile(mat, 75, axis=0)
        elif band == "minmax":
            lo, hi = mat.min(axis=0), mat.max(axis=0)
        elif band == "none":
            lo = hi = mean
        else:  # 'se' (default)
            se = mat.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
            lo, hi = mean - se, mean + se
        block = pd.DataFrame({"step": grid, "mean": mean, "lo": lo, "hi": hi, "n_seeds": n})
        for col, val in zip(group_cols, key):
            block[col] = val
        out.append(block)
    if not out:
        result = pd.DataFrame(columns=[*group_cols, "step", "mean", "lo", "hi", "n_seeds"])
    else:
        result = pd.concat(out, ignore_index=True)
    result.attrs["truncated"] = truncated
    result.attrs["late_start"] = late_start
    return result
