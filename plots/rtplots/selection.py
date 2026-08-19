"""Selezioni salvate: lettura, scrittura, storico.

Una selezione e' quello che il selettore ha in mano quando premi «Salva»: le
run scelte, i filtri con cui ci sei arrivato e — la parte che conta per gli
script — lo `FigureSpec` con cui la pagina stava disegnando. Salvare e' quindi
salvare la figura, non solo l'elenco delle run: `plot_curves.py --runs-file`
(o `plot_budget.py --runs-file`) rifa' esattamente quella figura.

`selection.json` e' sempre l'ultima salvata (quella che gli script usano senza
argomenti); `selections/<slug>.json` e' lo storico per nome.
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
    return slug or datetime.now().strftime("selezione-%Y%m%d-%H%M%S")


def free_slug(slug: str, name: str) -> str:
    """Slug libero per `name`, senza sovrascrivere una selezione diversa.

    Risalvare con lo *stesso* nome deve aggiornare la selezione esistente (e'
    la via per correggere una selezione salvata), ma due nomi diversi che
    collassano sullo stesso slug — succede col nome di default, che ha la
    risoluzione al minuto, e con nomi che coincidono nei primi 60 caratteri —
    non devono cancellarsi a vicenda in silenzio.
    """
    candidate, n = slug, 1
    while True:
        stored = path_for(candidate)
        if not stored.exists():
            return candidate
        try:
            if json.loads(stored.read_text()).get("name") == name:
                return candidate          # stessa selezione: aggiornamento voluto
        except (json.JSONDecodeError, OSError):
            return candidate              # file illeggibile: rimpiazzarlo va bene
        n += 1
        candidate = f"{slug}-{n}"


def read(path: str | Path) -> dict:
    """Selezione da file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Selezione non trovata: {path} (salvala dal selettore)")
    data = json.loads(path.read_text())
    data.setdefault("version", SPEC_VERSION)
    return data


def spec_from(path: str | Path) -> tuple[FigureSpec, dict]:
    """(FigureSpec, selezione) da un file: quello che serve agli script."""
    data = read(path)
    spec = FigureSpec.from_dict(data.get("spec") or {})
    if not spec.run_ids:
        spec.run_ids = data.get("run_ids")
    return spec, data


# --- storico ----------------------------------------------------------------

def path_for(slug: str) -> Path:
    return SELECTIONS_DIR / f"{slugify(slug)}.json"


def listing() -> list[dict]:
    """Storico delle selezioni salvate, dalla piu' recente."""
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
    """Salva la selezione e la rende quella attiva (`selection.json`)."""
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    stored = path_for(payload["slug"])
    text = json.dumps(payload, indent=1, default=str)
    stored.write_text(text)
    SELECTION_JSON.write_text(text)
    return stored


def activate(slug: str) -> dict:
    """Rende attiva una selezione dello storico e la restituisce."""
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
