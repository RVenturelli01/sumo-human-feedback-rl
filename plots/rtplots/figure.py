"""Dalla selezione alla figura: una sola pipeline, usata da CLI e selettore.

Un solo oggetto serializzabile, `FigureSpec`, e' anche il formato con cui il
selettore salva le selezioni: «rifammi questa figura» e' letteralmente
rileggere lo spec.

`kind` sceglie la pipeline dei dati (non solo lo stile del disegno):
  - "curve"   una serie storica per run, aggregata nel tempo (`curves.py`);
  - "budget"  un solo numero per run (l'eval finale), aggregato per livello di
              budget (`budget.py`).
Sono percorsi diversi perche' i dati di partenza sono diversi (una history W&B
campionata nel tempo contro un `run.summary` scalare), non solo perche' il
grafico si disegna diversamente.
"""
from __future__ import annotations

from dataclasses import (asdict, dataclass, field, fields as dataclass_fields)

import pandas as pd

from . import budget as B
from . import curves as C
from . import labels as L
from . import rules as R
from . import schema
from . import style as S
from .grid import GridOptions, draw_grid
from .metrics import DEFAULT_CURVE_METRIC, DEFAULT_SUMMARY_METRIC, metric_info
from .select import select_runs

SPEC_VERSION = 1

# Assi di budget disponibili come x delle curve di budget: budget_level e'
# quello robusto di default (sempre popolato, un solo livello per gruppo,
# indipendente dal fatto che l'arm scali query o traiettorie); query_budget e
# demo_budget restano scelte esplicite per confronti a un solo braccio.
BUDGET_X_CHOICES = ("budget_level", "query_budget", "demo_budget")


@dataclass
class FigureSpec:
    """Tutto quello che serve per disegnare una figura, e nient'altro."""

    kind: str = "curve"                  # curve | budget
    # cosa
    run_ids: list | None = None          # selezione esplicita (dal selettore)
    filters: list = field(default_factory=list)
    state: str | None = "finished"
    metric: str = ""                     # "" = default del kind (vedi __post_init__)
    budget_x: str = "budget_level"       # solo per kind="budget"
    # come si dividono le curve
    rows: str | None = None
    cols: str | None = None
    hue: list | None = None              # None = automatico
    hue_order: list | None = None
    label_fields: list | None = None     # None = come hue
    min_seeds: int = 1
    # aggregazione (i default vengono da plots/style.toml, vedi rules.py)
    band: str = field(default_factory=lambda: R.get("lines", "band"))
    smooth: int = field(default_factory=lambda: int(R.get("lines", "smooth")))
    grid_points: int | None = None
    xmax: float | None = None
    # aspetto
    paper: bool = True
    share: str = field(default_factory=lambda: R.get("figure", "share"))
    panel_size: tuple = field(default_factory=lambda: tuple(R.get("figure", "panel_size")))
    legend: str = field(default_factory=lambda: R.get("legend", "where"))
    legend_loc: str = field(default_factory=lambda: R.get("legend", "loc"))
    legend_ncol: int = field(default_factory=lambda: int(R.get("legend", "ncol")))
    titles: str = "auto"
    label_mode: str = "all"
    sublabels: bool = False
    row_captions: str = "off"
    suptitle: str | None = None
    xscale: float = field(default_factory=lambda: float(R.get("figure", "xscale")))
    ylim: tuple | None = None
    logy: bool = False
    ylabel: str | None = None            # None = quella della metrica
    series_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.metric:
            self.metric = DEFAULT_SUMMARY_METRIC if self.kind == "budget" else DEFAULT_CURVE_METRIC

    # --- serializzazione -----------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["version"] = SPEC_VERSION
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FigureSpec":
        data = dict(data or {})
        data.pop("version", None)
        known = {f.name for f in dataclass_fields(cls)}
        spec = cls(**{k: v for k, v in data.items() if k in known})
        if spec.ylim:
            spec.ylim = tuple(spec.ylim)
        if spec.panel_size:
            spec.panel_size = tuple(spec.panel_size)
        return spec

    # --- opzioni di disegno --------------------------------------------------

    def grid_options(self, hue, ylabel: str) -> GridOptions:
        is_budget = self.kind == "budget"
        return GridOptions(
            rows=self.rows, cols=self.cols, xscale=1.0 if is_budget else self.xscale,
            xmax=self.xmax, ylim=self.ylim, share=self.share,
            panel_size=tuple(self.panel_size),
            legend=self.legend, legend_loc=self.legend_loc, legend_ncol=self.legend_ncol,
            titles=self.titles, label_mode=self.label_mode, sublabels=self.sublabels,
            row_captions=self.row_captions, suptitle=self.suptitle, ylabel=ylabel,
            logy=self.logy, logx=is_budget, style="errorbar" if is_budget else "band",
            xlabel=schema.title(self.budget_x) if is_budget else R.get("figure", "xlabel"),
            paper=self.paper, hue=list(hue),
        )


# --- scelta delle serie -----------------------------------------------------

def _series_styles(agg, ckey_colors: dict) -> dict:
    """Etichetta -> come si disegna quella curva."""
    out = {}
    for rec in agg.drop_duplicates("label").to_dict("records"):
        rule = R.rule_for(rec)
        out[rec["label"]] = {
            "color": rule.get("color") or ckey_colors[rec["ckey"]],
            "width": float(rule.get("width", S.line_width())),
            "style": rule.get("style", "solid"),
            "band_alpha": float(rule.get("band_alpha", S.band_alpha())),
            "latex": rule.get("latex"),
        }
    return out


def _apply_overrides(agg, order, styles, matches, overrides: dict):
    """Applica i ritocchi fatti a mano nell'anteprima (nome e colore di una serie)."""
    overrides = {k: v for k, v in (overrides or {}).items() if k in styles}
    if not overrides:
        return agg, order, styles, matches
    renames = {}
    for original, over in overrides.items():
        new = (over.get("name") or "").strip() or original
        style = dict(styles.pop(original))
        if over.get("color"):
            style["color"] = over["color"]
        if new != original:
            style["latex"] = None
        styles[new] = style
        matches[new] = matches.pop(original, {})
        renames[original] = new
    agg = agg.copy()
    agg["label"] = agg.label.map(lambda lab: renames.get(lab, lab))
    order = [renames.get(lab, lab) for lab in order]
    return agg, order, styles, matches


def auto_hue(df, exclude=()) -> list:
    """Serie = tutte le dimensioni di ablation che variano nella selezione."""
    exclude = {c for c in exclude if c}
    hue = [c for c in schema.SERIES_FIELDS
           if c in df.columns and c not in exclude and df[c].nunique(dropna=False) > 1]
    if not hue:
        return ["arm"]
    n = df.groupby(hue, dropna=False).ngroups
    for col in reversed(list(hue)):
        rest = [c for c in hue if c != col]
        if rest and df.groupby(rest, dropna=False).ngroups == n:
            hue = rest
    return hue


def merged_dims(df, hue, panels=()) -> list:
    """Dimensioni che variano ma non separano le curve: finiscono mediate."""
    covered = set(hue) | {c for c in panels if c}
    return [c for c in schema.SERIES_FIELDS
            if c in df.columns and c not in covered and df[c].nunique(dropna=False) > 1]


def _sort_ascending(df, cols) -> list:
    return [not (df[c].dtype == bool or set(df[c].dropna().unique()) <= {True, False})
            for c in cols]


# --- pipeline ---------------------------------------------------------------

@dataclass
class Series:
    """Risultato della preparazione: i dati e come vanno disegnati."""

    sel: pd.DataFrame           # run selezionati
    agg: pd.DataFrame           # curve aggregate sui seed
    order: list                 # ordine delle serie in legenda
    styles: dict                # etichetta -> {color, width, style, band_alpha, latex}
    matches: dict               # etichetta -> valori che la identificano (per style.toml)
    hue: list                   # dimensioni che decidono il colore
    ylabel: str
    metric_label: str
    merged: list                # dimensioni che variano senza separare le curve


def select(index: pd.DataFrame, spec: FigureSpec) -> pd.DataFrame:
    """Run che la figura deve usare (selezione esplicita + filtri + stato)."""
    df = index[index.run_id.isin(spec.run_ids)] if spec.run_ids is not None else index
    return select_runs(df, spec.filters, state=spec.state)


def _load_aggregate(sel: pd.DataFrame, spec: FigureSpec, group_cols: list,
                    verbose: bool) -> pd.DataFrame:
    if spec.kind == "budget":
        return B.aggregate(sel, group_cols, spec.budget_x, spec.metric, band=spec.band)
    curves = C.load_curves(sel, metric=spec.metric, verbose=verbose)
    if curves.empty:
        return curves
    return C.aggregate(curves, sel, group_cols, band=spec.band, smooth=spec.smooth,
                       grid_points=spec.grid_points, xmax=spec.xmax)


def prepare(index: pd.DataFrame, spec: FigureSpec, verbose: bool = True) -> Series:
    """Selezione -> dati -> aggregazione -> etichette e colori."""
    sel = select(index, spec)
    if sel.empty:
        raise ValueError("Nessun run corrisponde ai filtri.")

    info = metric_info(spec.metric)
    hue = [c for c in (spec.hue or []) if c in sel.columns]
    if spec.kind == "budget":
        # Ogni livello di budget gira con il best-config *di quel livello*
        # (Optuna tunato per punto): quasi ogni iperparametro (initial_queries,
        # pref_temperature, reward_net_arch, ...) covaria col livello stesso.
        # L'auto_hue generico li terrebbe come dimensioni "necessarie" solo
        # perche' proxy del livello — che pero' e' gia' l'asse x, non una
        # dimensione di colore. Qui il default e' un braccio = una serie;
        # un'ablation vera (es. normalize_agent_reward) si chiede con --hue.
        hue = hue or ["arm"]
        merged = []
    else:
        hue = hue or auto_hue(sel, exclude=(spec.rows, spec.cols))
        merged = merged_dims(sel, hue, (spec.rows, spec.cols))
    if verbose:
        print(f"[plot] {len(sel)} run selezionati; serie per: {', '.join(hue)}")
        if merged:
            print(f"[plot] ATTENZIONE: {', '.join(merged)} variano ma non separano le "
                  f"curve: configurazioni diverse finiscono mediate insieme "
                  f"(togli --hue per le serie automatiche)")

    group_cols = sorted(set(hue) | {"arm"} | {c for c in (spec.rows, spec.cols) if c})
    agg = _load_aggregate(sel, spec, group_cols, verbose)
    if agg.empty:
        raise ValueError(f"Nessun dato per «{info['label']}» in questa selezione.")
    agg = agg[agg.n_seeds >= spec.min_seeds].copy()
    if agg.empty:
        raise ValueError(f"Nessuna serie con almeno {spec.min_seeds} seed.")

    label_fields = tuple(spec.label_fields or hue) + ("arm",)
    agg["label"] = [L.series_label(r, fields=label_fields, paper=spec.paper)
                    for r in agg.to_dict("records")]
    hue_cols = [c for c in hue if c in agg.columns]
    agg["ckey"] = agg[hue_cols].astype(str).agg("|".join, axis=1)
    hue_keys = (agg.drop_duplicates("ckey")
                   .sort_values(hue_cols, ascending=_sort_ascending(agg, hue_cols))["ckey"]
                   .tolist())
    ckey_colors = dict(zip(hue_keys, S.color_cycle(len(hue_keys))))
    styles = _series_styles(agg, ckey_colors)
    sort_cols = hue_cols + [c for c in (spec.rows, spec.cols)
                            if c and c in agg.columns and c not in hue_cols]
    order = (agg.drop_duplicates("label")
                .sort_values(sort_cols, ascending=_sort_ascending(agg, sort_cols))
                ["label"].tolist())
    if spec.hue_order:
        order = [o for o in spec.hue_order if o in set(agg.label)]

    matches = {rec["label"]: {c: rec.get(c) for c in hue_cols}
               for rec in agg.drop_duplicates("label").to_dict("records")}
    agg, order, styles, matches = _apply_overrides(
        agg, order, styles, matches, spec.series_overrides)

    ylabel = spec.ylabel or info["ylabel"]
    return Series(sel=sel, agg=agg, order=order, styles=styles, hue=hue,
                  matches=matches, ylabel=ylabel, metric_label=info["label"], merged=merged)


def draw(series: Series, spec: FigureSpec):
    """Figura matplotlib a partire da un `Series` gia' preparato."""
    return draw_grid(series.agg, series.order, series.styles,
                     spec.grid_options(series.hue, series.ylabel))


def build(index: pd.DataFrame, spec: FigureSpec, verbose: bool = True):
    """(figura, Series) — il percorso completo, uguale per CLI e selettore."""
    series = prepare(index, spec, verbose=verbose)
    return draw(series, spec), series


@dataclass
class Panel:
    """Un riquadro della griglia, disegnabile da solo."""

    series: Series
    spec: FigureSpec
    fixed: dict
    row: int
    col: int

    @property
    def slug(self) -> str:
        bits = [f"{c}_{_slug_value(v)}" for c, v in self.fixed.items()]
        return "_".join(bits)

    @property
    def caption(self) -> str:
        return ", ".join(schema.panel_title(c, v) for c, v in self.fixed.items())


def _slug_value(value) -> str:
    text = str(value).replace(".0", "")
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()


def split_panels(series: Series, spec: FigureSpec) -> list:
    """La griglia spacchettata in pannelli indipendenti, uno per figura (export .tex)."""
    from dataclasses import replace as dc_replace

    from .grid import panel_values

    row_vals = panel_values(series.agg, spec.rows)
    col_vals = panel_values(series.agg, spec.cols)
    flat = dc_replace(spec, rows=None, cols=None, suptitle=None,
                      row_captions="off", sublabels=False, label_mode="all")
    panels = []
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            fixed = {}
            agg = series.agg
            for col, value in ((spec.rows, rv), (spec.cols, cv)):
                if not col:
                    continue
                fixed[col] = value
                agg = agg[agg[col] == value]
            if agg.empty:
                continue
            order = [lab for lab in series.order if lab in set(agg.label)]
            panels.append(Panel(
                series=dc_replace(series, agg=agg, order=order),
                spec=flat, fixed=fixed, row=i, col=j))
    return panels


def n_panels(series: Series, spec: FigureSpec) -> int:
    rows = series.agg[spec.rows].nunique() if spec.rows else 1
    cols = series.agg[spec.cols].nunique() if spec.cols else 1
    return max(1, rows) * max(1, cols)
