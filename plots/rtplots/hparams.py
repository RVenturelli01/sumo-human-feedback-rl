"""Iperparametri di un gruppo di run, in YAML.

Perche' serve un modulo e non due righe in `api.py`: l'indice
(`rtplots/source.py`) conserva solo le colonne su cui si filtra e si colora --
arm, budget, temperatura, smoothing, architettura. Gli iperparametri che si
vogliono davvero rileggere dopo (``lr_rew``, ``l2_rew``, ``gradient_steps_rew``,
``batch_size_expert``, ``batch_size_model``, ``batch_size_pref``) NON sono
nell'indice. L'unica fonte e' la config Hydra completa, che W&B restituisce solo
dopo ``run.load(force=True)`` -- la stessa trappola gia' documentata in
`source.py`, dove ``api.runs()`` da' ``config == {}``.

Quindi: una richiesta di rete per run, cachata su disco come gia' fanno curve e
summary. Una riga della tabella di copertura e' un GRUPPO (tipicamente tre
seed), quindi lo YAML separa cio' che le run condividono da cio' che le
distingue: se il gruppo e' sano, l'unica differenza e' ``run.seed``.

L'emissione YAML e' fatta a mano invece che con PyYAML per due motivi: il
selettore dichiara di girare con la sola libreria standard piu' quello che i
plot gia' usano (PyYAML non e' in `plots/requirements.txt`), e servono commenti
nel file prodotto, che PyYAML non sa scrivere.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .paths import CACHE_DIR, ensure_dirs, wandb_path

CONFIG_DIR = CACHE_DIR / "configs"

# Sezioni della config Hydra, nell'ordine in cui vale la pena leggerle.
SECTION_ORDER = ("run", "algo", "agent", "env", "train", "eval", "wandb")

# Chiavi che descrivono dove la run e' finita, non come e' stata configurata.
NOISE_KEYS = (
    "run.output_dir", "run.name", "run.group", "wandb.entity", "wandb.project",
    "wandb.tags", "wandb.mode", "wandb.id",
)


def _cache_file(project: str, run_id: str) -> Path:
    return CONFIG_DIR / f"{project}__{run_id}.json"


def load_config(run_id: str, project: str, state: str = "") -> dict:
    """Config Hydra completa di una run, cachata su disco.

    Una run non ancora ``finished`` non viene cachata: e' la stessa regola di
    `budget.load_summary` e `curves.curve_from_wandb`. Qui in realta' la config
    non cambia piu' dopo il lancio, ma tenere una sola regola per tutta la cache
    costa meno che ricordarsi l'eccezione.
    """
    ensure_dirs()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_file(project, run_id)
    if state == "finished" and cache_file.exists():
        return json.loads(cache_file.read_text())
    import wandb

    api = wandb.Api()
    run = api.run(f"{wandb_path(project)}/{run_id}")
    run.load(force=True)                      # senza questo config e' {}
    cfg = dict(run.config or {})
    if run.state == "finished":
        cache_file.write_text(json.dumps(cfg, default=str))
    return cfg


# --- appiattimento ----------------------------------------------------------

def flatten(cfg: dict, prefix: str = "") -> Dict[str, Any]:
    """`{"algo": {"kwargs": {"lr_rew": 1e-3}}}` -> `{"algo.kwargs.lr_rew": 1e-3}`.

    Le liste restano valori: `net_arch: [64, 64]` e' un iperparametro solo, e
    spezzarlo in `net_arch.0` / `net_arch.1` renderebbe illeggibile il diff fra
    due run che usano architetture di profondita' diversa.
    """
    out: Dict[str, Any] = {}
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, prefix=f"{path}."))
        else:
            out[path] = value
    return out


def _section_rank(key: str) -> tuple:
    head = key.split(".", 1)[0]
    return (SECTION_ORDER.index(head) if head in SECTION_ORDER else len(SECTION_ORDER), key)


def split_common(per_run: Dict[str, Dict[str, Any]]) -> tuple[dict, Dict[str, dict]]:
    """Divide le chiavi condivise da tutte le run da quelle che differiscono.

    Una chiave assente in una run e presente in un'altra conta come differenza:
    non e' un dettaglio, e' esattamente il caso in cui una run e' stata lanciata
    con un'opzione che le altre non avevano.
    """
    if not per_run:
        return {}, {}
    keys = set()
    for flat in per_run.values():
        keys.update(flat)
    common, differing = {}, {}
    for key in sorted(keys, key=_section_rank):
        values = [flat.get(key, KeyError) for flat in per_run.values()]
        first = values[0]
        if all(_same(v, first) for v in values):
            common[key] = first
        else:
            differing[key] = {rid: flat.get(key) for rid, flat in per_run.items()}
    return common, differing


def _same(a, b) -> bool:
    if a is KeyError or b is KeyError:
        return a is b
    if isinstance(a, list) and isinstance(b, list):
        return list(a) == list(b)
    return a == b


# --- emissione YAML ---------------------------------------------------------

_PLAIN_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./+")


def _scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        # repr tiene le cifre significative: lr_rew = 0.0011542956981980379 deve
        # restare confrontabile con l'export di Optuna, non diventare 0.00115.
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_scalar(x) for x in v) + "]"
    text = str(v)
    if text and set(text) <= _PLAIN_OK and not text[0].isdigit():
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _emit(mapping: Dict[str, Any], indent: int) -> List[str]:
    pad = " " * indent
    return [f"{pad}{k}: {_scalar(v)}" for k, v in mapping.items()]


def to_yaml(blocks: Sequence[tuple]) -> str:
    """`[(commento|None, chiave|None, mapping)]` -> testo YAML.

    Volutamente minimale: le config Hydra di questo progetto sono mappe annidate
    di scalari e liste, gia' appiattite qui in chiavi puntate.
    """
    lines: List[str] = []
    for comment, key, mapping in blocks:
        if comment:
            lines.extend(f"# {c}" for c in comment.splitlines())
        if key is None:
            lines.extend(_emit(mapping, 0))
        else:
            lines.append(f"{key}:")
            if not mapping:
                lines[-1] += " {}"
            else:
                lines.extend(_emit(mapping, 2))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --- documento finale -------------------------------------------------------

def run_hparams(cfg: dict) -> Dict[str, Any]:
    """Config Hydra -> chiavi puntate, senza quelle che non sono iperparametri.

    ``run.output_dir``, ``run.name``, ``wandb.*`` dicono dove la run e' finita,
    non come e' stata configurata: lasciarle dentro farebbe comparire ogni run
    del gruppo nella sezione "differenze" e seppellirebbe l'unica riga che conta.
    """
    return {k: v for k, v in flatten(cfg).items() if k not in NOISE_KEYS}


def group_yaml(records: Iterable[dict], cells: Sequence[str] = (),
               columns: Sequence[str] = (), loader=None) -> str:
    """YAML degli iperparametri di un gruppo di run della tabella di copertura.

    ``records`` sono righe dell'indice (run_id, project, state, name, seed).
    ``loader`` e' iniettabile per i test; risolto qui e non come default della
    firma, altrimenti resterebbe legato alla funzione vista alla definizione del
    modulo e un monkeypatch su ``load_config`` non avrebbe effetto.
    """
    loader = loader or load_config
    records = list(records)
    per_run: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    for rec in records:
        run_id = rec["run_id"]
        try:
            cfg = loader(run_id, rec.get("project") or "", rec.get("state") or "")
        except Exception as exc:                       # una run rotta non blocca il resto
            errors[run_id] = f"{type(exc).__name__}: {exc}"
            continue
        per_run[run_id] = run_hparams(cfg)

    common, differing = split_common(per_run)

    identita = {}
    for col, cell in zip(columns, cells):
        identita[str(col)] = cell
    identita["n_run"] = len(records)
    identita["run_ids"] = [r["run_id"] for r in records]
    names = [r.get("name") for r in records if r.get("name")]
    if names:
        identita["run_names"] = names

    blocks = [
        ("Iperparametri di questa riga della tabella di copertura.\n"
         "Generato dal selettore (plots/), dalla config Hydra registrata su W&B.",
         "gruppo", identita),
        (f"Condivisi da tutte le {len(per_run)} run del gruppo.", "comune", common),
    ]
    if differing:
        diff_flat = {f"{key} [{rid}]": val
                     for key, per in differing.items() for rid, val in per.items()}
        blocks.append(
            ("Chiavi su cui le run del gruppo NON coincidono.\n"
             "In un gruppo sano qui c'e' solo run.seed.", "differenze", diff_flat))
    if errors:
        blocks.append(("Run che non e' stato possibile leggere.", "errori", errors))
    return to_yaml(blocks)
