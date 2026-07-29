"""Stile dei grafici: applica le regole di `plots/style.toml` a matplotlib.

I valori non stanno qui ma nel `.toml`, che si modifica a mano (vedi
`rules.py`): questo modulo li traduce in rcParams e in colori delle serie.

Preferenze di partenza:
  - palette per braccio, coerente con `scripts/_report_common.py:ARM_COLORS`
    (cosi' i grafici di questo motore e le tabelle/figure gia' scritte per la
    tesi usano sempre lo stesso colore per lo stesso braccio);
  - font serif con mathtext, nomi degli algoritmi in monospace;
  - pannelli con box completo (4 spine), tick verso l'esterno, niente griglia;
  - curva di apprendimento = media sui seed, banda ombreggiata semitrasparente;
  - curva di budget = errorbar su asse x logaritmico (vedi grid.py).
"""
from __future__ import annotations

import re
from itertools import cycle

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import rules as R

# Stessa palette di scripts/_report_common.py:ARM_COLORS, nell'ordine
# scripts/_report_common.py:ARM_ORDER (+ una coppia in piu' per bracci non
# ancora previsti li', come "ibarz").
ARM_ORDER = [
    "#2a78d6",  # blu     pref_soft
    "#1baf7a",  # verde acqua  pref_bernoulli
    "#eda100",  # giallo  demo_1
    "#008300",  # verde   demo_2
    "#4a3aa7",  # viola   hybrid_demo_1
    "#e34948",  # rosso   hybrid_demo_2
    "#8a4b08",  # marrone (scorta)
    "#0f7ea6",  # ciano   (scorta)
]


def band_alpha() -> float:
    return float(R.get("lines", "band_alpha"))


def line_width() -> float:
    return float(R.get("lines", "width"))


def baseline_color() -> str:
    return str(R.get("lines", "baseline_color"))


def baseline_width() -> float:
    return float(R.get("lines", "baseline_width"))


def color_cycle(n: int) -> list[str]:
    """n colori distinti dalla palette del file, allungata se non bastano."""
    palette = R.palette() or ARM_ORDER
    if n <= len(palette):
        return palette[:n]
    it = cycle(palette)
    return [next(it) for _ in range(n)]


def apply_style(scale: float | None = None) -> None:
    """rcParams globali dalle regole. `scale` sovrascrive [figure].font_scale."""
    scale = float(R.get("figure", "font_scale") if scale is None else scale)
    mpl.rcParams.update({
        # font
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIX Two Text", "serif"],
        "font.monospace": ["DejaVu Sans Mono", "Courier New", "monospace"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10 * scale,
        "axes.labelsize": 11 * scale,
        "axes.titlesize": 11 * scale,
        "xtick.labelsize": 9 * scale,
        "ytick.labelsize": 9 * scale,
        "legend.fontsize": float(R.get("legend", "font_size")) * scale,
        # assi: box completo, niente griglia
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.axisbelow": True,
        # tick verso l'esterno
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # linee
        "lines.linewidth": line_width(),
        "lines.solid_capstyle": "round",
        # legenda con cornice
        "legend.frameon": bool(R.get("legend", "frame")),
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.edgecolor": "0.3",
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        # figura
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# "_" va escaped: in mathtext e' l'operatore di pedice anche dentro \mathtt{},
# e i nomi degli arm ne sono pieni (hybrid_demo_1, pref_soft, ...).
_MATH_ESCAPE = {"-": r"\text{-}", " ": r"\ ", "_": r"\_"}


def mathtt(text: str) -> str:
    """Nome di un arm in monospace mathtext: 'hybrid_demo_1 (soft)' ->
    '$\\mathtt{hybrid\\_demo\\_1\\ (soft)}$'."""
    escaped = str(text)
    for ch, esc in _MATH_ESCAPE.items():
        escaped = escaped.replace(ch, esc)
    return r"$\mathtt{%s}$" % escaped


def finalize_axes(ax, xmax=None, xlabel=True, ylabel=True,
                  xlabel_text=None, ylabel_text=None, logx: bool = False) -> None:
    """Etichette e limiti coerenti con lo stile scelto."""
    xlabel_text = R.get("figure", "xlabel") if xlabel_text is None else xlabel_text
    ylabel_text = R.get("figure", "ylabel") if ylabel_text is None else ylabel_text
    if xlabel:
        ax.set_xlabel(xlabel_text)
    if ylabel:
        ax.set_ylabel(ylabel_text)
    if xmax is not None and not logx:
        ax.set_xlim(0, xmax)
    if not logx:
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.tick_params(top=False, right=False)


def save(fig, outdir, name: str, formats=("png", "pdf")) -> list[str]:
    """Salva la figura nei formati richiesti; restituisce i path scritti."""
    from pathlib import Path

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = outdir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt)
        written.append(str(path))
    plt.close(fig)
    return written
