"""The selector handlers: pure functions, index and request in, dict out.

None of them knows HTTP exists, so they can be tested on their own and the
server only routes. The vocabulary of dimensions is not redefined here: it
comes from `schema.py`, the same one that writes the panel titles and the
legend entries.
"""
from __future__ import annotations

import io
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd

from .. import figure as F
from .. import rules as R
from .. import schema, selection, tikz
from ..budget import SUMMARY_DIR
from ..curves import _MEM as CURVE_MEM
from .. import hparams as HP
from ..metrics import DEFAULT_CURVE_METRIC, DEFAULT_SUMMARY_METRIC, metric_info, ui_groups
from .. import formulas
from ..paths import SELECTION_JSON

# Every run not yet cached is a network request. Past this threshold it is
# better to say so than to make the page wait for minutes.
MAX_WANDB_RUNS = 120
MAX_PREVIEW_RUNS = 800
MAX_PANELS = 24


# --- vocabulary of the dimensions -------------------------------------------

def dimension_values(df: pd.DataFrame) -> list[dict]:
    out = []
    for col in schema.UI_DIMENSIONS:
        if col not in df.columns:
            continue
        vals = df[col].dropna().unique().tolist()
        if not vals:
            continue
        try:
            vals = sorted(vals)
        except TypeError:
            vals = sorted(vals, key=str)
        out.append({
            "col": col,
            "title": schema.title(col),
            "values": [{"value": str(v), "label": schema.html_value(col, v),
                        "count": int((df[col].astype(str) == str(v)).sum())} for v in vals],
        })
    return out


# --- filters from the UI ----------------------------------------------------

OPS = ("is", "is_not", "in", "not_in")
NEGATIVE_OPS = ("is_not", "not_in")
SINGLE_OPS = ("is", "is_not")
DEFAULT_OP = "in"


def dim_filter(raw) -> tuple[str, list]:
    if isinstance(raw, dict):
        op = raw.get("op") or DEFAULT_OP
        return (op if op in OPS else DEFAULT_OP), list(raw.get("values") or [])
    return DEFAULT_OP, list(raw or [])


def apply_ui_filters(df: pd.DataFrame, sel: dict, exclusions: bool = True) -> pd.DataFrame:
    """UI filters: no value chosen means the dimension is not filtered."""
    out = df
    for col, raw in (sel.get("dims") or {}).items():
        op, values = dim_filter(raw)
        if not values or col not in out.columns:
            continue
        hit = out[col].astype(str).isin({str(v) for v in values})
        out = out[~hit] if op in NEGATIVE_OPS else out[hit]
    seeds = sel.get("seeds") or {}
    if seeds.get("min") is not None:
        out = out[out.seed >= float(seeds["min"])]
    if seeds.get("max") is not None:
        out = out[out.seed <= float(seeds["max"])]
    excluded = set(sel.get("excluded") or [])
    if exclusions and excluded:
        out = out[~out.run_id.isin(excluded)]
    return out


def live_counts(df: pd.DataFrame, sel: dict) -> dict:
    """For each value, how many runs would be left if it were chosen."""
    dims = sel.get("dims") or {}
    out = {}
    for col in schema.UI_DIMENSIONS:
        if col not in df.columns:
            continue
        others = {k: v for k, v in dims.items() if k != col}
        sub = apply_ui_filters(df, {**sel, "dims": others})
        counts = sub[col].astype(str).value_counts()
        op, _ = dim_filter(dims.get(col))
        if op in NEGATIVE_OPS:
            kept = apply_ui_filters(df, {**sel, "dims": {**others, col: dims[col]}})
            base = len(kept)
            kept_counts = kept[col].astype(str).value_counts()
            out[col] = {str(k): int(base - kept_counts.get(str(k), 0))
                        for k in counts.index}
        else:
            out[col] = {str(k): int(v) for k, v in counts.items()}
    return out


def _clean(value) -> str:
    try:
        f = float(value)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return str(value)


def filter_args(sel: dict, df: pd.DataFrame) -> list[str]:
    """The UI filters, written in the --filter syntax the scripts take."""
    args = []
    for col, raw in (sel.get("dims") or {}).items():
        op, values = dim_filter(raw)
        if not values or col not in df.columns:
            continue
        negative = op in NEGATIVE_OPS
        if not negative and len(values) == df[col].dropna().astype(str).nunique():
            continue
        args.append(f"{col}{'!=' if negative else '='}"
                    + ",".join(_clean(v) for v in values))
    seeds = sel.get("seeds") or {}
    all_seeds = df.seed.dropna()
    if seeds.get("min") is not None and (all_seeds.empty or seeds["min"] > all_seeds.min()):
        args.append(f"seed>={int(seeds['min'])}")
    if seeds.get("max") is not None and (all_seeds.empty or seeds["max"] < all_seeds.max()):
        args.append(f"seed<={int(seeds['max'])}")
    return args


def coverage_rows(sel_df: pd.DataFrame, excluded=(), limit: int = 300) -> dict:
    """Coverage table over the dimensions that vary in the selection."""
    varying = [c for c in schema.GRID_FIELDS
               if c in sel_df.columns and sel_df[c].nunique(dropna=False) > 1]
    if not varying:
        varying = [c for c in ("arm",) if c in sel_df.columns]
    if not varying:
        return {"columns": [], "rows": []}
    excluded = set(excluded)
    g = (sel_df.groupby(varying, dropna=False)
         .agg(n_runs=("run_id", "size"), n_seeds=("seed", "nunique"),
              seeds=("seed", lambda s: ",".join(str(int(x)) for x in sorted(s.dropna().unique()))),
              run_ids=("run_id", list))
         .reset_index())
    try:
        g = g.sort_values(varying)
    except TypeError:
        pass
    rows = []
    for r in g.head(limit).to_dict("records"):
        ids = list(r["run_ids"])
        kept = [i for i in ids if i not in excluded]
        seeds_kept = sel_df[sel_df.run_id.isin(kept)].seed.dropna().unique() if kept else []
        rows.append({
            "cells": [schema.html_value(c, r[c]) for c in varying],
            "n_runs": int(r["n_runs"]), "n_seeds": int(r["n_seeds"]), "seeds": r["seeds"],
            "run_ids": ids,
            "n_kept": len(kept), "n_seeds_kept": int(len(set(seeds_kept))),
            "on": len(kept) > 0,
        })
    return {"columns": [schema.title(c) for c in varying], "rows": rows,
            "truncated": len(g) > limit}


# --- from a page request to a spec ------------------------------------------

def spec_from_payload(payload: dict, sub: pd.DataFrame) -> F.FigureSpec:
    """Page settings -> FigureSpec, the same one the CLI uses."""
    grid = dict(payload.get("grid") or {})
    kind = grid.pop("kind", None) or "curve"
    spec = F.FigureSpec.from_dict({
        **{k: v for k, v in grid.items() if v not in ("", None)},
        "kind": kind,
        "run_ids": list(sub.run_id),
        "state": "any",                    # lo stato l'ha gia' scelto la UI
        "series_overrides": payload.get("series_overrides") or {},
        "row_captions": "auto" if grid.get("rows") else "off",
    })
    return spec


def wandb_cost_guard(sub: pd.DataFrame, kind: str, metric: str | None) -> str | None:
    """These metrics cost one request per run, so past a threshold it stops."""
    metric = metric or (DEFAULT_SUMMARY_METRIC if kind == "budget" else DEFAULT_CURVE_METRIC)
    if kind == "budget":
        projects = dict(zip(sub.run_id, sub.get("project", pd.Series(dtype=str))))
        todo = [r for r in sub.run_id
                if not (SUMMARY_DIR / f"{projects.get(r)}__{r}.json").exists()]
    else:
        projects = dict(zip(sub.run_id, sub.get("project", pd.Series(dtype=str))))
        todo = [r for r in sub.run_id if (projects.get(r), r, metric) not in CURVE_MEM]
    if len(todo) > MAX_WANDB_RUNS:
        info = metric_info(metric)
        return (f"«{info['label']}» va scaricata da W&B: {len(todo)} run non ancora in "
                f"cached (at most {MAX_WANDB_RUNS}). Narrow the selection; after the first "
                f"scaricamento i dati restano in cache.")
    return None


def render(index: pd.DataFrame, sub: pd.DataFrame, payload: dict,
          fmt: str = "png", dpi: int = 110, plot_lock=None) -> dict:
    """Draw the figure and return it as raw bytes."""
    spec = spec_from_payload(payload, sub)
    try:
        series = F.prepare(index[index.run_id.isin(sub.run_id)], spec, verbose=False)
    except ValueError as exc:
        return {"error": str(exc)}
    panels = F.n_panels(series, spec)
    if panels > MAX_PANELS:
        return {"error": f"Too many panels ({panels}): narrow the filters or the dimensions."}

    lock = plot_lock if plot_lock is not None else _NullLock()
    with lock:
        from .. import style as S

        S.apply_style()
        fig = F.draw(series, spec)
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight",
                    facecolor="white" if fmt in ("jpg", "jpeg") else "auto")
        matplotlib.pyplot.close(fig)
    return {"raw": buf.getvalue(),
            "series": int(series.agg.label.nunique()), "panels": int(panels),
            "hue": [schema.title(h) for h in series.hue],
            "auto_hue": not (payload.get("grid") or {}).get("hue"),
            "metric": series.metric_label,
            "merged": [schema.title(c) for c in series.merged],
            "truncated": series.truncated,
            "late_start": series.late_start,
            "series_list": series_list(series, spec),
            "palette": R.palette(),
            "n_seeds": int(series.agg.n_seeds.max()) if len(series.agg) else 0}


def series_list(series: F.Series, spec: F.FigureSpec) -> list[dict]:
    """The series of the figure, with what the page needs to touch them up."""
    overrides = spec.series_overrides or {}
    renamed = {(o.get("name") or "").strip() or k: k for k, o in overrides.items()}
    out = []
    for label in series.order:
        key = renamed.get(label, label)
        style = series.styles.get(label) or {}
        out.append({
            "key": key,
            "label": label,
            "color": style.get("color"),
            "renamed": key != label,
            "recolored": bool((overrides.get(key) or {}).get("color")),
            "rule": rule_snippet(label, series.matches.get(label) or {}, style),
        })
    return out


def rule_snippet(label: str, match: dict, style: dict) -> str:
    """The `[[series]]` block to paste into style.toml to keep a touch-up."""
    pairs = ", ".join(f"{col} = {_toml_value(v)}" for col, v in match.items()
                      if v is not None and v == v)
    lines = [f"match = {{ {pairs} }}"] if pairs else ["match = { }  # da completare"]
    if style.get("color"):
        lines.append(f'color = "{style["color"]}"')
    lines.append(f"name  = {_toml_literal(label)}")
    if style.get("latex"):
        lines.append(f"latex = {_toml_literal(style['latex'])}")
    return "[[series]]\n" + "\n".join(lines)


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float)):
        return str(value)
    return f'"{value}"'


def _toml_literal(text: str) -> str:
    return f"'{text}'" if "'" not in text else f'"{text}"'


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --- names and snippets -----------------------------------------------------

def figure_name(sel_df: pd.DataFrame, kind: str = "curve") -> str:
    """A file name built from the dimensions fixed in the selection."""
    bits = [kind]
    for col in ("arm_family", "demo_loss", "pref_labels"):
        if col not in sel_df.columns:
            continue
        vals = sel_df[col].dropna().unique()
        if len(vals) == 1:
            bits.append(_clean(vals[0]))
    name = "_".join(bits) or 'selection'
    return "".join(c for c in name if c.isalnum() or c in "_")[:60]


MAX_HPARAM_RUNS = 24


def hparams(df: pd.DataFrame, payload: dict) -> dict:
    """The hyperparameters of the runs in one coverage-table row, as YAML.

    One W&B request per run not yet cached, since the full config only arrives
    with ``run.load``. The same caution as the metrics applies: a row is three
    seeds, not three hundred.
    """
    run_ids = [r for r in (payload.get("run_ids") or []) if r]
    if not run_ids:
        return {"error": 'No run in this row.'}
    if len(run_ids) > MAX_HPARAM_RUNS:
        return {"error": f"{len(run_ids)} runs in this row (at most "
                         f"{MAX_HPARAM_RUNS}): narrow the filters."}
    known = df[df.run_id.isin(run_ids)]
    missing = sorted(set(run_ids) - set(known.run_id))
    if missing:
        return {"error": f"Run non presenti nell'indice: {', '.join(missing)}"}

    records = known[[c for c in ("run_id", "project", "state", "name", "seed")
                     if c in known.columns]].to_dict("records")
    # The order of the row's own list, not of the DataFrame: it is the order
    # the page shows the seeds in.
    by_id = {r["run_id"]: r for r in records}
    records = [by_id[r] for r in run_ids if r in by_id]

    text = HP.group_yaml(records, cells=payload.get("cells") or [],
                         columns=payload.get("columns") or [])
    return {"yaml": text, "filename": f"{_hparams_name(known)}.yaml",
            "n_runs": len(records)}


def _hparams_name(sel_df: pd.DataFrame) -> str:
    """File name: the W&B group if the runs share one, else the first run_id."""
    groups = sel_df.get("group")
    if groups is not None:
        vals = groups.dropna().unique()
        if len(vals) == 1:
            return "hparams_" + "".join(
                c if c.isalnum() or c in "_-" else "_" for c in str(vals[0]))[:60]
    return f"hparams_{sel_df.run_id.iloc[0]}"


def _caption(sel_df: pd.DataFrame, info: dict, band: str, kind: str) -> str:
    arms = ", ".join(sorted(sel_df.arm.dropna().unique())) or "?"
    band_txt = {"se": "errore standard", "std": "deviazione standard",
                "ci95": "intervallo di confidenza al 95\\%", "minmax": 'min--max',
                "none": "nessuna banda"}.get(band, band)
    tipo = "budget" if kind == "budget" else "apprendimento"
    return (f"{info.get('metric', '')} — curva di {tipo}, {arms}. Media su "
            f"{info.get('n_seeds', '?')} seed, banda: {band_txt}.")


def tex_panels(index: pd.DataFrame, sub: pd.DataFrame, payload: dict,
              name: str, plot_lock=None) -> dict:
    """One pgfplots source per panel: {files: [(name, code)], latex: ...}."""
    err = tikz.unavailable_reason()
    if err:
        return {"error": err}
    spec = spec_from_payload(payload, sub)
    try:
        series = F.prepare(index[index.run_id.isin(sub.run_id)], spec, verbose=False)
    except ValueError as exc:
        return {"error": str(exc)}
    panels = F.split_panels(series, spec)
    if len(panels) > MAX_PANELS:
        return {"error": f"Too many panels ({len(panels)}): narrow the filters."}

    lock = plot_lock if plot_lock is not None else _NullLock()
    files = []
    with lock:
        from .. import style as S

        S.apply_style()
        for panel in panels:
            fig = F.draw(panel.series, panel.spec)
            header = f"% {name}{' — ' + panel.caption if panel.caption else ''}"
            files.append((f"{name}{'_' + panel.slug if panel.slug else ''}.tex",
                          tikz.figure_to_tex(fig, header=header,
                                             styles=panel.series.styles)))
            matplotlib.pyplot.close(fig)

    grid = payload.get("grid") or {}
    info = {"metric": series.metric_label,
            "n_seeds": int(series.agg.n_seeds.max()) if len(series.agg) else 0}
    ncol = max(1, len({p.col for p in panels}))
    return {"files": files, "n_panels": len(panels), "metric": series.metric_label,
            "latex": tex_snippet(name, panels, [f for f, _ in files], ncol,
                                 _caption(sub, info, grid.get("band", "se"),
                                          grid.get("kind", "curve")))}


def tex_snippet(name: str, panels, filenames: list[str], ncol: int, caption: str) -> str:
    """The assembled figure: one `\\input` per panel, in subfigures if several."""
    if len(filenames) == 1:
        body = f"  \\input{{figures/{filenames[0]}}}\n"
    else:
        width = f"{0.98 / ncol:.2f}".lstrip("0")
        rows = []
        for panel, filename in zip(panels, filenames):
            lines = [f"  \\begin{{subfigure}}[b]{{{width}\\linewidth}}",
                     f"    \\input{{figures/{filename}}}"]
            if panel.caption:
                lines.append(f"    \\caption{{{panel.caption}}}")
            lines.append("  \\end{subfigure}")
            rows.append("\n".join(lines))
            rows.append("  \\\\" if panel.col == ncol - 1 else "  \\hfill")
        body = "\n".join(rows[:-1]) + "\n"
    return (f"% {tikz.PREAMBLE.lstrip('% ')}"
            + (" \\usepackage{subcaption}\n" if len(filenames) > 1 else "\n")
            + "\\begin{figure}[t]\n"
              "  \\centering\n"
            + body
            + f"  \\caption{{{caption}}}\n"
              f"  \\label{{fig:{name}}}\n"
              "\\end{figure}")


def latex_snippet(name: str, sel_df: pd.DataFrame, info: dict, band: str, kind: str) -> str:
    """Blocco figure pronto da incollare, con caption descrittiva."""
    caption = _caption(sel_df, info, band, kind)
    return ("\\begin{figure}[t]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=\\linewidth]{{figures/{name}.pdf}}\n"
            f"  \\caption{{{caption}}}\n"
            f"  \\label{{fig:{name}}}\n"
            "\\end{figure}")


# --- risposte agli endpoint -------------------------------------------------

OP_LABELS = [
    {"op": "is", "label": 'is', "multi": False},
    {"op": "is_not", "label": 'is not', "multi": False},
    {"op": "in", "label": "fra", "multi": True},
    {"op": "not_in", "label": "non fra", "multi": True},
]


def dimensions(df: pd.DataFrame) -> dict:
    seeds = df.seed.dropna()
    return {
        "dimensions": dimension_values(df),
        "ops": OP_LABELS,
        "default_op": DEFAULT_OP,
        "grid_fields": [{"col": c, "title": schema.title(c)}
                        for c in schema.GRID_FIELDS if c in df.columns],
        "budget_x_choices": [{"col": c, "title": schema.title(c)} for c in F.BUDGET_X_CHOICES],
        "seed_min": int(seeds.min()) if len(seeds) else 1,
        "seed_max": int(seeds.max()) if len(seeds) else 10,
        "n_runs": int(len(df)),
        "selection_path": str(SELECTION_JSON),
        "selections": selection.listing(),
        "metrics_curve": ui_groups("curve"),
        "metrics_budget": ui_groups("summary"),
        "default_metric_curve": DEFAULT_CURVE_METRIC,
        "default_metric_budget": DEFAULT_SUMMARY_METRIC,
    }


def query(df: pd.DataFrame, payload: dict) -> dict:
    sub = apply_ui_filters(df, payload)
    unfiltered = apply_ui_filters(df, payload, exclusions=False)
    grid_cols = [c for c in schema.GRID_FIELDS if c in sub.columns]
    return {
        "n_runs": int(len(sub)),
        "n_excluded": int(len(unfiltered) - len(sub)),
        "n_configs": int(sub.groupby(grid_cols, dropna=False).ngroups) if len(sub) else 0,
        "states": sub.state.value_counts().to_dict() if len(sub) else {},
        "coverage": (coverage_rows(unfiltered, payload.get("excluded") or [])
                     if len(unfiltered) else {"columns": [], "rows": []}),
        "counts": live_counts(df, payload),
        "filter_args": filter_args(payload, df),
    }


def preview(df: pd.DataFrame, payload: dict, plot_lock=None) -> dict:
    sub = apply_ui_filters(df, payload)
    if sub.empty:
        return {"error": 'No run selected.'}
    if len(sub) > MAX_PREVIEW_RUNS:
        return {"error": f"{len(sub)} run: troppe per l'anteprima."}
    grid = payload.get("grid") or {}
    err = wandb_cost_guard(sub, grid.get("kind", "curve"), grid.get("metric"))
    if err:
        return {"error": err}
    t0 = time.time()
    res = render(df, sub, payload, fmt="png", dpi=110, plot_lock=plot_lock)
    res["elapsed"] = round(time.time() - t0, 2)
    return res


def compose_with_formula(figure_png: bytes, df: pd.DataFrame, payload: dict,
                         fmt: str = "png", dpi: int = 300) -> bytes:
    """The figure and the definitions panel, side by side in one image.

    Composed as rasters rather than as a single matplotlib figure: `formulas`
    already draws them, and redrawing them inside the grid would mean two
    sources of truth for the same content.
    """
    from PIL import Image

    blocks = _formula_blocks(df, payload)
    if not blocks:
        return figure_png
    left = Image.open(io.BytesIO(figure_png)).convert("RGB")
    right = Image.open(io.BytesIO(formulas.render_png(blocks, dpi=dpi))).convert("RGB")
    # The definitions accompany the figure, they do not dominate it: half its
    # width at most, and scaled down if they are taller.
    max_w = left.width // 2
    if right.width > max_w:
        right = right.resize((max_w, max(1, right.height * max_w // right.width)))
    if right.height > left.height:
        h = left.height
        right = right.resize((max(1, right.width * h // right.height), h))
    gap = max(12, left.width // 60)
    canvas = Image.new("RGB", (left.width + gap + right.width,
                               max(left.height, right.height)), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    out = io.BytesIO()
    canvas.save(out, format="JPEG" if fmt in ("jpg", "jpeg") else "PNG", quality=95)
    return out.getvalue()


def _formula_blocks(df: pd.DataFrame, payload: dict) -> list:
    """The blocks to show for the current selection: metric and fusions."""
    grid = payload.get("grid") or {}
    metric = grid.get("metric") or (
        DEFAULT_SUMMARY_METRIC if grid.get("kind") == "budget" else DEFAULT_CURVE_METRIC)
    try:
        sub = apply_ui_filters(df, payload)
        fusions = ([f for f in sub.fusion.dropna().unique()]
                   if "fusion" in sub.columns else [])
    except Exception:
        fusions = []
    return formulas.blocks(metric, fusions)


def formula(df: pd.DataFrame, payload: dict) -> dict:
    """The formulas for the chosen metric and the selected fusions.

    Rendered server side with matplotlib mathtext, so the page needs no external
    dependencies.
    """
    blocks = _formula_blocks(df, payload)
    if not blocks:
        return {"svg": None}
    return {"svg": formulas.render_svg(blocks)}


def save(df: pd.DataFrame, payload: dict) -> dict:
    sub = apply_ui_filters(df, payload)
    name = (payload.get("name") or "").strip() or \
        datetime.now().strftime('selection %d/%m %H:%M:%S')
    slug = selection.free_slug(selection.slugify(name), name)
    stored = selection.write({
        "version": F.SPEC_VERSION,
        "name": name,
        "slug": slug,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "n_runs": int(len(sub)),
        "filter_args": filter_args(payload, df),
        "dims": payload.get("dims") or {},
        "series_overrides": payload.get("series_overrides") or {},
        "seeds": payload.get("seeds") or {},
        "excluded": list(payload.get("excluded") or []),
        "run_ids": sub.run_id.tolist(),
        "spec": spec_from_payload(payload, sub).to_dict(),
    })
    print(f"[selector] salvata «{name}»: {len(sub)} run -> {stored}")
    return {"ok": True, "path": str(stored), "slug": slug, "name": name,
            "n_runs": int(len(sub)), "items": selection.listing()}
