"""Percorsi condivisi da tutti gli script di plotting.

A differenza del progetto da cui questo toolkit e' ispirato (che gira su un
cluster condiviso con cache in /storage), qui l'analisi si fa dalla stessa
macchina o dal Mac via API pubblica W&B (vedi docs/analysis-pipeline-guide.md):
niente storage condiviso, niente file locali (metrics.jsonl/evaluations.npz)
raggiungibili — quelli vivono sul server dove giri il training. La cache
dell'indice e delle curve sta quindi dentro la repo (`plots/.cache/`,
ignorata da git), sovrascrivibile con `RTPLOTS_CACHE`.
"""
from __future__ import annotations

import os
from pathlib import Path

# Radice della repo (plots/rtplots/paths.py -> plots/ -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_ROOT = REPO_ROOT / "plots"

# Cache: sovrascrivibile con RTPLOTS_CACHE
CACHE_DIR = Path(os.environ.get("RTPLOTS_CACHE", str(PLOTS_ROOT / ".cache")))
INDEX_PARQUET = CACHE_DIR / "run_index.parquet"
INDEX_CSV = CACHE_DIR / "run_index.csv"
CURVE_DIR = CACHE_DIR / "curves"
# Selezione salvata dal selettore interattivo (plots/scripts/selector.py):
# selection.json e' sempre l'ultima salvata, selections/ tiene lo storico per nome.
SELECTION_JSON = CACHE_DIR / "selection.json"
SELECTIONS_DIR = CACHE_DIR / "selections"

# Output figure: sovrascrivibile con RTPLOTS_OUTPUT o --outdir
OUTPUT_DIR = Path(os.environ.get("RTPLOTS_OUTPUT", str(PLOTS_ROOT / "output")))

WANDB_ENTITY = os.environ.get("RTPLOTS_WANDB_ENTITY", "andrea02polimi-politecnico-di-milano")

# Progetti indicizzati di default. Oggi la campagna di budget curves vive tutta
# in un progetto solo; se in futuro si allarga (es. l'ultima run finale a 5
# seed finisce in un progetto "thesis" separato) e' una riga in piu' qui, non
# un file nuovo: la convenzione di lettura (rtplots/source.py) e' la stessa
# per ogni progetto di questa campagna.
DEFAULT_PROJECTS = os.environ.get(
    "RTPLOTS_WANDB_PROJECTS", "tuning-thesis-budget-curves-completion"
).split(",")


def wandb_path(project: str) -> str:
    """entity/project per un progetto dato."""
    return project if "/" in project else f"{WANDB_ENTITY}/{project}"


def ensure_dirs() -> None:
    for d in (CACHE_DIR, CURVE_DIR, SELECTIONS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
