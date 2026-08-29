"""From a selection to a figure: one pipeline, used by the CLI and the selector.

`FigureSpec` is a single serializable object, and also the format the selector
saves selections in: "draw this again" is literally rereading the spec.

`kind` chooses the data pipeline, not just the drawing style:

    curve    one time series per run, aggregated over time
    budget   one number per run, the final evaluation, aggregated per budget

They are different paths because the data is different: a W&B history sampled
over time against a scalar `run.summary`.
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
from .metrics import (DEFAULT_CURVE_METRIC, DEFAULT_SUMMARY_METRIC, ITER_STEP,
                      metric_info)
from .select import select_runs

SPEC_VERSION = 1

# Which column can act as the x axis of a budget curve. budget_level is the
# safe default: always filled, one level per group, and indifferent to whether
# the method scales queries or trajectories. The other two are explicit
# choices, for comparing one method with itself.
BUDGET_X_CHOICES = ("budget_level", "query_budget", "demo_budget")


@dataclass
class FigureSpec:
    """Everything needed to draw one figure, and nothing else."""

    kind: str = "curve"                  # curve | budget
    # what to draw
    run_ids: list | None = None          # explicit selection, from the selector
    filters: list = field(default_factory=list)
    state: str | None = "finished"
    metric: str = ""                     # "" = default del kind (vedi __post_init__)
    budget_x: str = "budget_level"       # solo per kind="budget"
    # how the curves are split
    rows: str | None = None
    cols: str | None = None
    hue: list | None = None              # None = automatico
    # Budget curves only: adds `fusion` to the series identity, so two fusion
    # schemes of one method become separate curves instead of being averaged
    # together. Learning curves already split on `fusion` by themselves.
    compare_fusion: bool = False
    # Same for the normalization ablation: without it, ON and OFF of one
    # method end up in the same curve.
    compare_norm: bool = False
    # And for the label-smoothing ablation, eps=0 against eps>0.
    compare_smoothing: bool = False
    hue_order: list | None = None
    label_fields: list | None = None     # None = come hue
    min_seeds: int = 1
    # aggregation; the defaults come from plots/style.toml
    band: str = field(default_factory=lambda: R.get("lines", "band"))
    smooth: int = field(default_factory=lambda: int(R.get("lines", "smooth")))
    grid_points: int | None = None
    xmax: float | None = None
    # appearance
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
    ylabel: str | None = None            # None = whatever the metric says
    series_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.metric:
            self.metric = DEFAULT_SUMMARY_METRIC if self.kind == "budget" else DEFAULT_CURVE_METRIC

    # --- serialization -------------------------------------------------------

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

    # --- drawing options -----------------------------------------------------

    def grid_options(self, hue, ylabel: str) -> GridOptions:
        is_budget = self.kind == "budget"
        # `[figure].xscale` and `xlabel` are meant for the agent axis, in
        # millions of timesteps. The reward-learning metrics sit on the
        # `iterations` counter, which runs from 0 to about 100: dividing that
        # by 1e6 flattens it to zero and makes the label a lie.
        on_iteration_axis = (
            not is_budget
            and metric_info(self.metric).get("step_key") == ITER_STEP
        )
        if on_iteration_axis:
            x_scale, x_label = 1.0, "Iterazione"
        elif is_budget:
            x_scale, x_label = 1.0, schema.title(self.budget_x)
        else:
            x_scale, x_label = self.xscale, R.get("figure", "xlabel")
        return GridOptions(
            rows=self.rows, cols=self.cols, xscale=x_scale,
            xmax=self.xmax, ylim=self.ylim, share=self.share,
            panel_size=tuple(self.panel_size),
            legend=self.legend, legend_loc=self.legend_loc, legend_ncol=self.legend_ncol,
            titles=self.titles, label_mode=self.label_mode, sublabels=self.sublabels,
            row_captions=self.row_captions, suptitle=self.suptitle, ylabel=ylabel,
            logy=self.logy, logx=is_budget, style="errorbar" if is_budget else "band",
            xlabel=x_label,
            paper=self.paper, hue=list(hue),
        )


# --- choosing the series ----------------------------------------------------

# Line styles used to separate series that one style.toml rule would colour
# identically; see _decollide.
DASH_CYCLE = ("solid", "dashed", "dotted", "dashdot", (0, (5, 1, 1, 1)))


def _decollide(styles: dict, order: list, matches: dict | None = None) -> None:
    """Tell apart, by line style, series that would come out identical.

    A `style.toml` rule applies the first match, so it cannot say "colour by
    method, dash by ablation". Comparing two configurations of one method would
    colour both the same and the curves would be indistinguishable. Here the
    colour stays what the rule says and only the dash pattern changes.

    Inside a colliding group the baseline configuration keeps the solid line:
    between two booleans the disabled one comes first, so the ablation is always
    the dashed curve and the convention does not move between figures.
    """
    groups: dict = {}
    for lab in order:
        st = styles.get(lab)
        if st is not None:
            groups.setdefault((st["color"], st["style"]), []).append(lab)
    for labels in groups.values():
        if len(labels) < 2:
            continue
        if matches:
            def base_first(lab):
                vals = matches.get(lab) or {}
                return sum(1 for v in vals.values() if v is True or v == "True")
            labels = sorted(labels, key=base_first)   # stabile: parita' = ordine legenda
        for i, lab in enumerate(labels):
            styles[lab]["style"] = DASH_CYCLE[i % len(DASH_CYCLE)]


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
    """Apply the touch-ups made by hand in the preview: a name, a colour."""
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
    """One series per ablation dimension that varies in the selection."""
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
    """Dimensions that vary but do not split the curves, so get averaged."""
    covered = set(hue) | {c for c in panels if c}
    return [c for c in schema.SERIES_FIELDS
            if c in df.columns and c not in covered and df[c].nunique(dropna=False) > 1]


def _sort_ascending(df, cols) -> list:
    return [not (df[c].dtype == bool or set(df[c].dropna().unique()) <= {True, False})
            for c in cols]


# --- pipeline ---------------------------------------------------------------

@dataclass
class Series:
    """The result of preparing: the data, and how to draw it."""

    sel: pd.DataFrame           # the selected runs
    agg: pd.DataFrame           # curve aggregate sui seed
    order: list                 # series order in the legend
    styles: dict                # etichetta -> {color, width, style, band_alpha, latex}
    matches: dict               # etichetta -> valori che la identificano (per style.toml)
    hue: list                   # dimensioni che decidono il colore
    ylabel: str
    metric_label: str
    merged: list                # dimensions that vary without splitting curves
    truncated: list = field(default_factory=list)  # serie accorciate da una run corta
    late_start: list = field(default_factory=list)  # serie che iniziano dopo lo step 0


def select(index: pd.DataFrame, spec: FigureSpec) -> pd.DataFrame:
    """The runs the figure should use: explicit selection, filters and state."""
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


# In budget curves these do not split the data by themselves, and they are not
# proxies for the level either: if they vary without being on hue, rows or
# columns, different configurations really do get averaged together.
BUDGET_ABLATIONS = (("fusion", "compare_fusion"),
                    ("normalize_agent_reward", "compare_norm"),
                    ("label_smoothing", "compare_smoothing"))


def _merged_budget_dims(sel, hue, spec: FigureSpec) -> list:
    """Ablation columns that would be averaged into the same series."""
    separated = set(hue) | {spec.rows, spec.cols}
    return [col for col, _ in BUDGET_ABLATIONS
            if col in sel.columns
            and sel[col].nunique(dropna=False) > 1
            and col not in separated]


def prepare(index: pd.DataFrame, spec: FigureSpec, verbose: bool = True) -> Series:
    """Selection -> data -> aggregation -> labels and colours."""
    sel = select(index, spec)
    if sel.empty:
        raise ValueError("No run matches the filters.")

    info = metric_info(spec.metric)
    hue = [c for c in (spec.hue or []) if c in sel.columns]
    if spec.kind == "budget":
        # Each budget level runs with the best config for that level, so almost
        # every hyperparameter covaries with the level itself. The generic
        # auto_hue would keep them as "needed" dimensions only because they
        # proxy the level, which is already the x axis. The default here is one
        # method per series; a real ablation is asked for with --hue.
        hue = hue or ["arm"] + [col for col, flag in BUDGET_ABLATIONS
                                if getattr(spec, flag)]
        # `merged_dims` resta spento qui per il motivo sopra, ma queste due
        # columns are not proxies for the level: if they vary unseparated,
        # configurazioni diverse finiscono davvero mediate insieme e va detto.
        merged = _merged_budget_dims(sel, hue, spec)
    else:
        hue = hue or auto_hue(sel, exclude=(spec.rows, spec.cols))
        merged = merged_dims(sel, hue, (spec.rows, spec.cols))
    if verbose:
        print(f"[plot] {len(sel)} runs selected; series by: {', '.join(hue)}")
        if merged:
            flags = {col: flag for col, flag in BUDGET_ABLATIONS}
            opts = [f"--{flags[c].replace('_', '-')}" for c in merged if c in flags]
            fix = (f"'use '{' e '.join(opts)}" if len(opts) == len(merged)
                   else 'drop --hue for automatic series')
            print(f"[plot] WARNING: {', '.join(merged)} vary but do not split the "
                  f"curves: different configurations get averaged together ({fix})")

    group_cols = sorted(set(hue) | {"arm"} | {c for c in (spec.rows, spec.cols) if c})
    agg = _load_aggregate(sel, spec, group_cols, verbose)
    if agg.empty:
        raise ValueError(f"No data for «{info['label']}» in this selection.")
    agg = agg[agg.n_seeds >= spec.min_seeds].copy()
    if agg.empty:
        raise ValueError(f"No series with at least {spec.min_seeds} seeds.")

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
    _decollide(styles, order, matches)

    truncated = list(agg.attrs.get("truncated") or [])
    if verbose and truncated:
        for t in truncated:
            print(f"'[plot] WARNING: the series stops at '{t['end']:,.0f} invece di "
                  f"{t['longest']:,.0f}: run {t['run_id']} is shorter than the others "
                  f"in its group, and the shared grid follows the shortest")
    late_start = list(agg.attrs.get("late_start") or [])
    if verbose and late_start:
        for t in late_start:
            print(f"'[plot] the series starts at '{t['start']:,.0f} invece di "
                  f"{t['earliest']:,.0f}: run {t['run_id']} starts logging this metric "
                  f"later than the others in its group")
    ylabel = spec.ylabel or info["ylabel"]
    return Series(sel=sel, agg=agg, order=order, styles=styles, hue=hue,
                  matches=matches, ylabel=ylabel, metric_label=info["label"], merged=merged,
                  late_start=late_start)


def draw(series: Series, spec: FigureSpec):
    """A matplotlib figure from an already prepared `Series`."""
    return draw_grid(series.agg, series.order, series.styles,
                     spec.grid_options(series.hue, series.ylabel))


def build(index: pd.DataFrame, spec: FigureSpec, verbose: bool = True):
    """(figure, Series): the whole path, the same for the CLI and the selector."""
    series = prepare(index, spec, verbose=verbose)
    return draw(series, spec), series


@dataclass
class Panel:
    """One cell of the grid, drawable on its own."""

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
    """The grid unpacked into independent panels, one figure each."""
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
