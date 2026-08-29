"""One declaration per index column.

Each column says once how it is written in the UI, whether it can go on rows,
columns or colours, and what it adds to a legend, instead of repeating the same
lists by hand in the sidebar, the filters and the panel titles.

The columns come from `source.py`. The order below is also the order of the
selector sidebar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# Readable method names, rendered in monospace by labels.py.
ARM_NAMES = {
    "demo_1": "demo_1",
    "demo_2": "demo_2",
    "pref_soft": "pref_soft",
    "pref_bernoulli": "pref_bernoulli",
    "hybrid_demo_1_soft": "hybrid_demo_1 (soft)",
    "hybrid_demo_1_bernoulli": "hybrid_demo_1 (bernoulli)",
    "hybrid_demo_2_soft": "hybrid_demo_2 (soft)",
    "hybrid_demo_2_bernoulli": "hybrid_demo_2 (bernoulli)",
}


# How the two channels are fused (`algo.kwargs.gcl_fusion`). The short names
# are for reading; the raw value stays the one in the code.
FUSION_NAMES = {
    "norm_balance": "norm_balance (baseline)",
    "alpha_norm_single_adam": 'one Adam on the fused gradient',
    "dual_adam_alpha": 'one Adam per channel',
    "dual_adam_sum": "due Adam, somma",
    "dual_adam_alpha_unit": "due Adam, alpha + budget",
    "dual_adam_alpha_unit_nobudget": "due Adam, alpha su direzioni unitarie",
    # Schemes tried and later dropped from VALID_GCL_FUSIONS. Their runs stay
    # in the index, and the name marks them.
    "dual_adam_reliability": 'dual_adam_reliability (earlier)',
    "demo_anchor_inv_var": 'demo_anchor_inv_var (earlier)',
}


def _int(value):
    return int(float(value))


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)


@dataclass(frozen=True)
class Field:
    col: str
    title: str
    ui: bool = False
    grid: bool = False
    series: bool = False
    html: Callable | None = None
    title_of: Callable | None = None
    legend: Callable | None = None


def _bool_html(yes: str = 'yes', no: str = "no"):
    return lambda v: yes if v in (True, "True") else no


def _int_html(suffix: str = ""):
    return lambda v: f"{_int(v)}{suffix}"


def _num_html(fmt: str = "{:g}"):
    """Like _int_html but for floats.

    Raises on a missing value instead of printing 'nan', which float(nan) would
    do quietly where int(nan) raises.
    """
    def _fmt(v):
        if _missing(v):
            raise ValueError("missing")
        return fmt.format(float(v))
    return _fmt


def _eps(value) -> float:
    """The label_smoothing value, 0 for runs that do not declare one."""
    return 0.0 if _missing(value) else float(value)


def _smoothing_html(value) -> str:
    eps = _eps(value)
    return 'no smoothing' if not eps else f"'smoothing (eps='{eps:g})"


def _millions_html(v):
    if _missing(v):
        raise ValueError("missing")
    return f"{float(v) / 1e6:g}M"


FIELDS: list[Field] = [
    Field(
        "arm", 'Method', ui=True, grid=True, series=True,
        html=lambda v: ARM_NAMES.get(v, str(v)), title_of=lambda v: ARM_NAMES.get(v, str(v)),
        legend=lambda v: ARM_NAMES.get(v, str(v)),
    ),
    # Not in the sidebar and not a grid dimension: redundant with "Method",
    # which already lists every combination one by one. They stay real columns,
    # so they can still be filtered from the command line.
    # How the two channels are combined, hybrid only. In the sidebar and among
    # the grid dimensions because it is what tells the grad-diagnostics methods
    # apart: without it they all collapse into one name.
    Field(
        "fusion", 'Gradient fusion', ui=True, grid=True, series=True,
        html=lambda v: FUSION_NAMES.get(v, str(v)),
        title_of=lambda v: FUSION_NAMES.get(v, str(v)),
        legend=lambda v: FUSION_NAMES.get(v, str(v)),
    ),
    Field("arm_family", 'Family', html=str, title_of=str),
    Field("demo_loss", 'Demonstration loss', html=str),
    Field("pref_labels", 'Preference labels', html=str),
    Field("demo_mode", 'Demonstration mode', html=str),
    Field(
        "query_budget", 'Preference budget (one transition)', ui=True, grid=True, series=True,
        html=_int_html(),
        title_of=lambda v: f"query = {_int(v)}",
        legend=lambda v: f"{_int(v)} query",
    ),
    Field(
        "demo_budget", 'Demonstration budget (trajectories)', ui=True, grid=True, series=True,
        html=lambda v: 'whole dataset' if _missing(v) else f"{_int(v)}' trajectories'",
        title_of=lambda v: "dataset intero" if _missing(v) else f"{_int(v)} traiettorie",
        legend=lambda v: None if _missing(v) else f"{_int(v)} traj",
    ),
    # Not in the sidebar, and not a grid dimension: it stays a real column, and
    # is the default x axis of the budget curves, chosen in the toolbar.
    # A grid dimension, one row per budget, but not in the sidebar: filtering
    # happens on query_budget and demo_budget. The title stays neutral, "B =
    # 10": one row spans several methods, and B means ten preferences and ten
    # trajectories for the hybrid but only one of the two for the single-source
    # methods, so that belongs in the caption, not in the panel title.
    Field(
        "budget_level", 'Budget B (from the group)', grid=True,
        html=_int_html(), title_of=lambda v: f"B = {_int(v)}",
        legend=lambda v: f"B={_int(v)}",
    ),
    Field("normalize_agent_reward", 'Reward normalized', ui=True, grid=True, series=True,
          html=_bool_html(), title_of=lambda v: f"normalize_agent_reward = {bool(v)}",
          legend=lambda v: "norm" if v else "no-norm"),
    # The eps value, not a boolean. There is one level today and the UI reads
    # as on or off, but a second eps would split the curves by itself instead
    # of collapsing two configurations into one.
    Field("label_smoothing", "Label smoothing", ui=True, grid=True, series=True,
          html=_smoothing_html, title_of=_smoothing_html,
          legend=lambda v: None if not _eps(v) else f"eps={_eps(v):g}"),
    Field("query_schedule", 'Query schedule', ui=True, grid=True, series=True, html=str),
    Field("fragmenter_type", "Fragmenter", ui=True, grid=True, series=True, html=str),
    # Not in the sidebar, the grid or the legend: these are the best-config
    # hyperparameters of each budget level, not dimensions to filter or split
    # on. They stay real columns, filterable from the command line.
    Field("initial_queries", 'Initial queries (bootstrap)',
          html=_int_html(), title_of=lambda v: f"initial_queries = {_int(v)}"),
    Field("demo_weight", 'Demonstration weight',
          html=_num_html(), title_of=lambda v: f"demo_weight = {float(v):g}"),
    Field("pref_temperature", 'Oracle temperature', html=_num_html()),
    Field("reward_net_arch", 'Reward model network', html=str),
    Field("demo_subsample_seed", 'Demo subsample seed',
          html=lambda v: "= seed" if _missing(v) else str(_int(v))),
    Field("total_timesteps", 'Total timesteps', html=_millions_html),
    Field("state", 'State', ui=True, html=str),
    Field("project", 'W&B project', ui=True, html=str),
    Field("group_tag", 'Group tag', ui=True, html=str),
    # Too many distinct values for the sidebar, but available to the coverage
    # table and the legend.
    Field("group", 'W&B group'),
    Field("seed", "Seed", legend=lambda v: f"seed={_int(v)}"),
]

BY_COL: dict[str, Field] = {f.col: f for f in FIELDS}

UI_DIMENSIONS = [f.col for f in FIELDS if f.ui]
GRID_FIELDS = [f.col for f in FIELDS if f.grid]
# Dimensioni che devono separare le curve: se una varia e nessuno l'ha messa su
# colours, rows or columns, different configurations get averaged together.
SERIES_FIELDS = [f.col for f in FIELDS if f.series]


def title(col: str) -> str:
    f = BY_COL.get(col)
    return f.title if f else col


def html_value(col: str, value) -> str:
    """The value as it should be written in the selector page."""
    f = BY_COL.get(col)
    if f is not None and f.html is not None:
        try:
            return f.html(value)
        except (TypeError, ValueError, KeyError):
            pass
    if _missing(value):
        return "—"
    if value in (True, "True"):
        return 'yes'
    if value in (False, "False"):
        return "no"
    return str(value)


def panel_title(col: str, value, paper: bool = True) -> str:
    """A row or column title in the grid."""
    if _missing(value):
        return ""
    f = BY_COL.get(col)
    if f is not None and f.title_of is not None:
        try:
            return f.title_of(value)
        except (TypeError, ValueError, KeyError):
            pass
    return f"{col} = {value}"


def legend_bit(col: str, value) -> str | None:
    """Frammento che il campo aggiunge alla legenda, o None."""
    if _missing(value):
        return None
    f = BY_COL.get(col)
    if f is None or f.legend is None:
        return None
    try:
        return f.legend(value)
    except (TypeError, ValueError, KeyError):
        return None
