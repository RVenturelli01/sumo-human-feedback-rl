"""Disegno della griglia di pannelli (condiviso da plot_curves.py/plot_budget.py
e dal selettore).

Riceve un DataFrame gia' aggregato con colonne [step, mean, lo, hi, n_seeds, label]
piu' le colonne usate per righe/colonne, e produce la figura. Due stili di
disegno (`GridOptions.style`):
  - "band"      curva + banda ombreggiata (curve di apprendimento, step = tempo);
  - "errorbar"  punti + barre d'errore, pensato per l'asse x logaritmico delle
                curve di budget (pochi livelli discreti, non una serie continua).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from . import labels as L
from . import rules as R
from . import style as S

LETTERS = "abcdefghijklmnopqrstuvwxyz"
# posizioni disponibili per `GridOptions.legend`
LEGEND_CHOICES = ("panel", "first", "figure", "outside_right", "outside_bottom", "none")


@dataclass
class GridOptions:
    rows: str | None = None
    cols: str | None = None
    xscale: float = 1e6
    xmax: float | None = None
    ylim: tuple | None = None
    share: str = "row"
    panel_size: tuple = (3.4, 2.5)
    legend: str = "panel"
    legend_loc: str = "best"
    legend_ncol: int = 1
    titles: str = "auto"
    label_mode: str = "all"
    sublabels: bool = False
    row_captions: str = "off"
    suptitle: str | None = None
    xlabel: str = field(default_factory=lambda: R.get("figure", "xlabel"))
    ylabel: str = field(default_factory=lambda: R.get("figure", "ylabel"))
    logy: bool = False
    logx: bool = False
    style: str = "band"          # band | errorbar
    paper: bool = True
    hue: list = field(default_factory=list)


def panel_values(df, col):
    if col is None:
        return [None]
    return sorted(df[col].dropna().unique().tolist())


def _legend_handles(order, styles, opts: GridOptions):
    """Handle sintetiche per l'intera legenda (una per etichetta, nell'ordine
    scelto), invece di leggerle da un pannello a caso: un pannello preso a caso
    (es. axes[0][0]) potrebbe non contenere tutte le serie, se righe/colonne
    dividono i dati — la legenda "figure"/"outside" deve elencarle comunque
    tutte."""
    handles = []
    for lab in order:
        st = styles.get(lab, {})
        color = st.get("color", "black")
        if opts.style == "errorbar":
            h = mlines.Line2D([], [], color=color, lw=st.get("width", 1.4),
                              ls=st.get("style", "solid"), marker="o", markersize=4)
        else:
            h = mlines.Line2D([], [], color=color, lw=st.get("width", 1.4),
                              ls=st.get("style", "solid"))
        handles.append(h)
    return handles, list(order)


def draw_grid(agg, order, styles, opts: GridOptions):
    """`styles`: etichetta -> {color, width, style, band_alpha} (vedi figure.py)."""
    row_vals = panel_values(agg, opts.rows)
    col_vals = panel_values(agg, opts.cols)
    nrow, ncol = len(row_vals), len(col_vals)
    sharey = {"none": False, "row": "row", "col": "col", "all": True}[opts.share]
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=sharey, squeeze=False,
                             figsize=(opts.panel_size[0] * ncol, opts.panel_size[1] * nrow))

    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            ax = axes[i][j]
            sub = agg
            if opts.rows:
                sub = sub[sub[opts.rows] == rv]
            if opts.cols:
                sub = sub[sub[opts.cols] == cv]

            for lab in order:
                g = sub[sub.label == lab].sort_values("step")
                if g.empty:
                    continue
                x = g["step"] if opts.logx else g["step"] / opts.xscale
                st = styles.get(lab, {})
                if opts.style == "errorbar":
                    yerr = [(g["mean"] - g["lo"]).to_numpy(), (g["hi"] - g["mean"]).to_numpy()]
                    ax.errorbar(x, g["mean"], yerr=yerr, color=st["color"],
                               linewidth=st["width"], linestyle=st["style"],
                               marker="o", markersize=4,
                               capsize=3, label=lab, zorder=3)
                else:
                    ax.plot(x, g["mean"], color=st["color"], lw=st["width"],
                            ls=st["style"], label=lab, zorder=3)
                    ax.fill_between(x, g["lo"], g["hi"], color=st["color"],
                                    alpha=st["band_alpha"], lw=0, zorder=2)

            if opts.titles == "auto" and i == 0 and opts.cols:
                title = L.panel_title(opts.cols, cv, opts.paper)
                if title:
                    ax.set_title(title)

            show_x = opts.label_mode == "all" or i == nrow - 1
            show_y = opts.label_mode == "all" or j == 0
            if opts.logx:
                ax.set_xscale("log")
            S.finalize_axes(ax, xmax=(opts.xmax / opts.xscale) if opts.xmax and not opts.logx else None,
                            xlabel=show_x, ylabel=show_y,
                            xlabel_text=opts.xlabel, ylabel_text=opts.ylabel, logx=opts.logx)
            if opts.label_mode == "all":
                ax.tick_params(labelbottom=True, labelleft=True)
            if opts.logy:
                ax.set_yscale("log")
            if opts.ylim:
                ax.set_ylim(*opts.ylim)
            if opts.legend == "panel" or (opts.legend == "first" and i == 0 and j == 0):
                handles, lbls = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, lbls, loc=opts.legend_loc, ncol=opts.legend_ncol)
            if opts.sublabels:
                ax.text(0.5, -0.34, f"({LETTERS[i * ncol + j]})", transform=ax.transAxes,
                        ha="center", va="top")

    if opts.suptitle:
        fig.suptitle(opts.suptitle)

    # Il layout si calcola SOLO sui pannelli, prima di aggiungere una legenda
    # fuori dagli assi: tight_layout() prova a "farle posto" se esiste gia',
    # e finisce per schiacciare gli assi molto piu' del necessario. La legenda
    # esterna si attacca DOPO, in coordinate relative agli assi ormai definitivi
    # — non serve rilanciare il layout, e bbox_inches="tight" al salvataggio
    # (rcParam globale, vedi style.py) allarga da solo il canvas per farcela
    # stare.
    fig.tight_layout()

    if opts.legend == "figure":
        handles, lbls = _legend_handles(order, styles, opts)
        fig.legend(handles, lbls, loc="lower center", ncol=min(len(lbls), 4),
                   bbox_to_anchor=(0.5, -0.03))
    elif opts.legend == "outside_bottom":
        handles, lbls = _legend_handles(order, styles, opts)
        fig.legend(handles, lbls, loc="upper center", ncol=min(len(lbls), 4),
                   bbox_to_anchor=(0.5, -0.06 * nrow))
    elif opts.legend == "outside_right":
        handles, lbls = _legend_handles(order, styles, opts)
        axes[0][-1].legend(handles, lbls, loc="upper left", bbox_to_anchor=(1.02, 1),
                           borderaxespad=0., ncol=opts.legend_ncol)

    if opts.row_captions != "off":
        caps = ([f"({LETTERS[i]}) {L.panel_title(opts.rows, rv, opts.paper)}"
                for i, rv in enumerate(row_vals)] if opts.row_captions == "auto"
                else opts.row_captions.split(";"))
        fig.subplots_adjust(hspace=0.62)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        for i, cap in enumerate(caps[:nrow]):
            y = min(axes[i][j].get_tightbbox(renderer).transformed(inv).y0
                    for j in range(ncol))
            fig.text(0.5, y - 0.012, cap, ha="center", va="top", style="italic")

    return fig
