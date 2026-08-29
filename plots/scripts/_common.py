"""Shared CLI arguments, and how they become a `FigureSpec`.

The scripts do not prepare data any more: they build the spec and hand it to
`rtplots.figure`, the same pipeline the selector uses. "Redraw from the command
line the figure I see in the page" is then guaranteed rather than hoped for.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
from rtplots import selection as SEL
from rtplots.figure import FigureSpec
from rtplots.paths import OUTPUT_DIR, SELECTION_JSON


def add_selection_args(p):
    g = p.add_argument_group('selection')
    g.add_argument("--filter", nargs="*", default=[],
                   help='filters on the index, e.g. arm=demo_1,demo_2 query_budget>=5000')
    g.add_argument("--state", default=None, help="finished (default) | any | crashed")
    g.add_argument("--runs-file", nargs="?", const=str(SELECTION_JSON), default=None,
                   metavar="PATH",
                   help="use the selection saved by the interactive selector "
                        f"(with no argument: {SELECTION_JSON})")
    g.add_argument("--hue", nargs="*", default=None,
                   help="colonne che definiscono le serie colorate; per default sono "
                        "automatic (every dimension that varies in the selection, "
                        "except those already on rows or columns)")
    g.add_argument("--hue-order", nargs="*", default=None)
    g.add_argument("--label-fields", nargs="*", default=None,
                   help="campi mostrati in legenda oltre all'arm (default: --hue)")
    g.add_argument("--min-seeds", type=int, default=1)
    g.add_argument("--compare-fusion", action="store_true", default=None,
                   help="curve di budget: separa anche per schema di fusione "
                        "(without it, different schemes of one method are averaged)")
    g.add_argument("--compare-norm", action="store_true", default=None,
                   help="curve di budget: separa anche per normalize_agent_reward "
                        "(ON tratteggiata, OFF continua)")
    g.add_argument("--compare-smoothing", action="store_true", default=None,
                   help="curve di budget: separa anche per label_smoothing "
                        "(with smoothing dashed, without it solid)")
    return p


def add_aggregation_args(p):
    g = p.add_argument_group("aggregazione")
    g.add_argument("--band", default=None,
                   choices=["se", "std", "ci95", "iqr", "minmax", "none"],
                   help="banda/errorbar: errore standard (default), std, IC 95%%, IQR, min-max")
    return p


def add_grid_args(p):
    g = p.add_argument_group("griglia e aspetto")
    g.add_argument("--rows", default=None, help='index column for the rows')
    g.add_argument("--cols", default=None, help='index column for the columns')
    g.add_argument("--ylim", nargs=2, type=float, default=None)
    g.add_argument("--logy", action="store_true")
    g.add_argument("--share", default=None, choices=["none", "row", "col", "all"])
    g.add_argument("--panel-size", nargs=2, type=float, default=None)
    g.add_argument("--legend", default=None,
                   choices=["outside_right", "outside_bottom", "panel", "figure", "first", "none"])
    g.add_argument("--legend-loc", default=None)
    g.add_argument("--legend-ncol", type=int, default=None)
    g.add_argument("--titles", default=None, choices=["auto", "off"])
    g.add_argument("--label-mode", default=None, choices=["all", "edge"])
    g.add_argument("--sublabels", action="store_true", help="(a) (b) (c) sotto i pannelli")
    g.add_argument("--row-captions", default=None, help="off | auto | testi separati da ';'")
    g.add_argument("--suptitle", default=None)
    g.add_argument("--font-scale", type=float, default=1.0)
    return p


def add_output_args(p, default_name="figure"):
    g = p.add_argument_group("output")
    g.add_argument("--name", default=default_name)
    g.add_argument("--outdir", default=str(OUTPUT_DIR))
    g.add_argument("--formats", nargs="*", default=["png", "pdf"])
    g.add_argument("--dump-csv", action="store_true", help="salva anche i dati aggregati")
    return p


def spec_from_args(args, kind: str, metric: str | None = None) -> FigureSpec:
    """Arguments, plus any saved selection, into a FigureSpec.

    Precedence: the command line, then the saved selection, then the defaults.
    """
    spec = FigureSpec(kind=kind)
    runs_file = getattr(args, "runs_file", None)
    if runs_file:
        spec, data = SEL.spec_from(runs_file)
        spec.kind = kind
        print(f"[plot] selection from {data.get('saved_at')}: {data.get('n_runs')} runs "
              f"({' '.join(data.get('filter_args') or []) or 'nessun filtro'})")
    else:
        spec.state = "finished"

    def pick(attr, dest=None, cast=None):
        value = getattr(args, attr, None)
        if value is None:
            return
        setattr(spec, dest or attr, cast(value) if cast else value)

    spec.filters = list(getattr(args, "filter", []) or [])
    pick("state")
    if metric or getattr(args, "metric", None):
        spec.metric = metric or args.metric
    pick("budget_x")
    for attr in ("rows", "cols", "hue", "hue_order", "label_fields", "min_seeds",
                 "compare_fusion", "compare_norm", "compare_smoothing"):
        pick(attr)
    for attr in ("band", "smooth", "grid_points", "xmax"):
        pick(attr)
    for attr in ("share", "legend", "legend_loc", "legend_ncol", "titles",
                 "label_mode", "sublabels", "row_captions", "suptitle", "xscale", "logy"):
        pick(attr)
    pick("ylim", cast=tuple)
    pick("panel_size", cast=tuple)
    pick("ylabel")
    return spec


def report(series, args, paths):
    for p in paths:
        print(f"[plot] scritto {p}")
    if args.dump_csv:
        csv = f"{args.outdir}/{args.name}.csv"
        series.agg.to_csv(csv, index=False)
        print(f"[plot] scritto {csv}")
    print('[plot] seeds per series:\n'
          + series.agg.groupby("label")["n_seeds"].max().to_string())
