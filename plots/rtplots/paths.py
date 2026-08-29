"""Paths shared by every plotting script.

The analysis runs from any machine through the W&B API: there is no shared
storage, and the files written during training stay on the machine that ran it.
The index and curve caches therefore live inside the repository, in
`plots/.cache/`, which git ignores. `RTPLOTS_CACHE` overrides it.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repository root: plots/rtplots/paths.py -> plots/ -> repo/
REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_ROOT = REPO_ROOT / "plots"

# Cache directory, overridable with RTPLOTS_CACHE
CACHE_DIR = Path(os.environ.get("RTPLOTS_CACHE", str(PLOTS_ROOT / ".cache")))
INDEX_PARQUET = CACHE_DIR / "run_index.parquet"
INDEX_CSV = CACHE_DIR / "run_index.csv"
CURVE_DIR = CACHE_DIR / "curves"
# Selections saved by the interactive selector: selection.json is always the
# most recent one, and selections/ keeps the earlier ones by name.
SELECTION_JSON = CACHE_DIR / "selection.json"
SELECTIONS_DIR = CACHE_DIR / "selections"

# Where figures are written, overridable with RTPLOTS_OUTPUT or --outdir
OUTPUT_DIR = Path(os.environ.get("RTPLOTS_OUTPUT", str(PLOTS_ROOT / "output")))

WANDB_ENTITY = os.environ.get("RTPLOTS_WANDB_ENTITY", "andrea02polimi-politecnico-di-milano")

# Projects indexed by default. They are all read the same way, because they all
# come from the same entry point, so adding one is a single line here.
#
#   thesis-final       the reference runs, one protocol, th_* groups
#   tuning-*           the budget-curve campaign
#   *-grad-diagnostics fusion schemes, normalization ablation, frozen probe
#   thesis             single-source ablations
#
# They are needed together: comparing hybrid against the single-channel
# baselines means reading from more than one project.
DEFAULT_PROJECTS = os.environ.get(
    "RTPLOTS_WANDB_PROJECTS",
    "thesis-final,thesis-grad-diagnostics,"
    "tuning-thesis-budget-curves-completion,thesis",
).split(",")


def wandb_path(project: str) -> str:
    """entity/project per un progetto dato."""
    return project if "/" in project else f"{WANDB_ENTITY}/{project}"


def ensure_dirs() -> None:
    for d in (CACHE_DIR, CURVE_DIR, SELECTIONS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
