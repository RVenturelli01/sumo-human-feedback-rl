"""Saved selections: reading, writing, and the history.

A selection is what the selector holds when you press Save: the runs chosen, the
filters you reached them with, and the `FigureSpec` the page was drawing with.
Saving therefore saves the figure, not only the list of runs, and
`--runs-file` redraws exactly that figure.

`selection.json` is always the most recent one, the one the scripts use with no
arguments; `selections/<slug>.json` keeps the earlier ones by name.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .figure import SPEC_VERSION, FigureSpec
from .paths import SELECTION_JSON, SELECTIONS_DIR


def slugify(name: str) -> str:
    slug = "".join(c if c.isalnum() else "-" for c in name.strip().lower())
    slug = "-".join(p for p in slug.split("-") if p)[:60]
    return slug or datetime.now().strftime("selection-%Y%m%d-%H%M%S")


def free_slug(slug: str, name: str) -> str:
    """A free slug for `name`, without overwriting a different selection.

    Saving again under the same name should update the existing selection, which
    is how a saved one gets corrected. But two different names that collapse to
    the same slug must not quietly erase each other.
    """
    candidate, n = slug, 1
    while True:
        stored = path_for(candidate)
        if not stored.exists():
            return candidate
        try:
            if json.loads(stored.read_text()).get("name") == name:
                return candidate          # same selection: updating on purpose
        except (json.JSONDecodeError, OSError):
            return candidate              # file illeggibile: rimpiazzarlo va bene
        n += 1
        candidate = f"{slug}-{n}"


def read(path: str | Path) -> dict:
    """Read a selection from a file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Selection not found: {path} (save one from the selector)")
    data = json.loads(path.read_text())
    data.setdefault("version", SPEC_VERSION)
    return data


def spec_from(path: str | Path) -> tuple[FigureSpec, dict]:
    """(FigureSpec, selection) from a file: what the scripts need."""
    data = read(path)
    spec = FigureSpec.from_dict(data.get("spec") or {})
    if not spec.run_ids:
        spec.run_ids = data.get("run_ids")
    return spec, data


# --- history ----------------------------------------------------------------

def path_for(slug: str) -> Path:
    return SELECTIONS_DIR / f"{slugify(slug)}.json"


def listing() -> list[dict]:
    """The saved selections, most recent first."""
    items = []
    for path in SELECTIONS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        items.append({
            "slug": path.stem,
            "name": data.get("name") or path.stem,
            "saved_at": data.get("saved_at", ""),
            "n_runs": data.get("n_runs", 0),
            "summary": " ".join(data.get("filter_args") or []) or "nessun filtro",
            "path": str(path),
        })
    return sorted(items, key=lambda d: d["saved_at"], reverse=True)


def write(payload: dict) -> Path:
    """Save the selection and make it the active one."""
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    stored = path_for(payload["slug"])
    text = json.dumps(payload, indent=1, default=str)
    stored.write_text(text)
    SELECTION_JSON.write_text(text)
    return stored


def activate(slug: str) -> dict:
    """Make a stored selection the active one, and return it."""
    data = read(path_for(slug))
    SELECTION_JSON.write_text(json.dumps(data, indent=1, default=str))
    return data


def rename(slug: str, name: str) -> dict:
    """Cambia solo l'etichetta: slug e nome del file restano quelli di partenza."""
    stored = path_for(slug)
    data = read(stored)
    data["name"] = name
    stored.write_text(json.dumps(data, indent=1, default=str))
    if SELECTION_JSON.exists():
        try:
            active = json.loads(SELECTION_JSON.read_text())
        except json.JSONDecodeError:
            active = {}
        if active.get("slug") == slugify(slug):
            SELECTION_JSON.write_text(json.dumps(data, indent=1, default=str))
    return data


def delete(slug: str) -> None:
    path_for(slug).unlink(missing_ok=True)
