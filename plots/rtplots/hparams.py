"""The hyperparameters of a group of runs, as YAML.

The index keeps only the columns used for filtering and colouring. The
hyperparameters worth rereading later, like ``lr_rew`` or ``gradient_steps_rew``,
are not in it: the only source is the full Hydra config, which W&B returns only
after ``run.load(force=True)``. That is one network request per run, cached on
disk like the curves and the summaries.

One row of the coverage table is a group, usually three seeds, so the YAML keeps
what the runs share apart from what tells them apart. In a healthy group the
only difference is ``run.seed``.

The YAML is written by hand rather than with PyYAML: the selector runs on the
standard library plus what the plots already need, and the output has comments,
which PyYAML cannot write.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .paths import CACHE_DIR, ensure_dirs, wandb_path

CONFIG_DIR = CACHE_DIR / "configs"

# Hydra config sections, in the order worth reading them.
SECTION_ORDER = ("run", "algo", "agent", "env", "train", "eval", "wandb")

# Keys saying where the run ended up, not how it was configured.
NOISE_KEYS = (
    "run.output_dir", "run.name", "run.group", "wandb.entity", "wandb.project",
    "wandb.tags", "wandb.mode", "wandb.id",
)


def _cache_file(project: str, run_id: str) -> Path:
    return CONFIG_DIR / f"{project}__{run_id}.json"


def load_config(run_id: str, project: str, state: str = "") -> dict:
    """The full Hydra config of one run, cached on disk.

    A run that is not finished is not cached, the same rule the curves and the
    summaries follow. A config does not really change after launch, but one rule
    for the whole cache costs less than remembering the exception.
    """
    ensure_dirs()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_file(project, run_id)
    if state == "finished" and cache_file.exists():
        return json.loads(cache_file.read_text())
    import wandb

    api = wandb.Api()
    run = api.run(f"{wandb_path(project)}/{run_id}")
    run.load(force=True)                      # without this, config is {}
    cfg = dict(run.config or {})
    if run.state == "finished":
        cache_file.write_text(json.dumps(cfg, default=str))
    return cfg


# --- flattening -------------------------------------------------------------

def flatten(cfg: dict, prefix: str = "") -> Dict[str, Any]:
    """`{"algo": {"kwargs": {"lr_rew": 1e-3}}}` -> `{"algo.kwargs.lr_rew": 1e-3}`.

    Lists stay values: `net_arch: [64, 64]` is one hyperparameter, and splitting
    it into `net_arch.0` and `net_arch.1` would make the diff between two
    different depths unreadable.
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
    """Split the keys every run shares from the keys that differ.

    A key present in one run and missing from another counts as a difference:
    that is exactly the case where one run was launched with an option the
    others did not have.
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


# --- writing the YAML -------------------------------------------------------

_PLAIN_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./+")


def _scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        # repr keeps the significant digits: lr_rew = 0.0011542956981980379
        # has to stay comparable, not become 0.00115.
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
    """`[(comment|None, key|None, mapping)]` -> YAML text.

    Deliberately minimal: these configs are nested maps of scalars and lists,
    already flattened here into dotted keys.
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


# --- the finished document --------------------------------------------------

def run_hparams(cfg: dict) -> Dict[str, Any]:
    """Hydra config -> dotted keys, without the ones that are not parameters.

    ``run.output_dir``, ``run.name`` and ``wandb.*`` say where a run ended up,
    not how it was configured. Leaving them in would put every run of the group
    under "differences" and bury the one line that matters.
    """
    return {k: v for k, v in flatten(cfg).items() if k not in NOISE_KEYS}


def group_yaml(records: Iterable[dict], cells: Sequence[str] = (),
               columns: Sequence[str] = (), loader=None) -> str:
    """The hyperparameters of one coverage-table group, as YAML.

    ``records`` are index rows. ``loader`` is injectable for the tests, and is
    resolved here rather than as a default argument: a default would bind to the
    function seen at import time, and monkeypatching ``load_config`` would have
    no effect.
    """
    loader = loader or load_config
    records = list(records)
    per_run: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    for rec in records:
        run_id = rec["run_id"]
        try:
            cfg = loader(run_id, rec.get("project") or "", rec.get("state") or "")
        except Exception as exc:                       # one broken run must not stop the rest
            errors[run_id] = f"{type(exc).__name__}: {exc}"
            continue
        per_run[run_id] = run_hparams(cfg)

    common, differing = split_common(per_run)

    identity = {}
    for col, cell in zip(columns, cells):
        identity[str(col)] = cell
    identity["n_run"] = len(records)
    identity["run_ids"] = [r["run_id"] for r in records]
    names = [r.get("name") for r in records if r.get("name")]
    if names:
        identity["run_names"] = names

    blocks = [
        ("Hyperparameters of this row of the coverage table.\n"
         "Written by the selector, from the Hydra config recorded on W&B.",
         "group", identity),
        (f"Shared by all {len(per_run)} runs in the group.", "shared", common),
    ]
    if differing:
        diff_flat = {f"{key} [{rid}]": val
                     for key, per in differing.items() for rid, val in per.items()}
        blocks.append(
            ("Keys the runs in the group do NOT agree on.\n"
             "In a healthy group this is only run.seed.", "differences", diff_flat))
    if errors:
        blocks.append(("Runs that could not be read.", "errors", errors))
    return to_yaml(blocks)
