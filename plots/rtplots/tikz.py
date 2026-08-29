"""Export a figure to pgfplots (`.tex`) through tikzplotlib, one file per panel.

The `.tex` stands on its own: a `tikzpicture` with its `\\definecolor`, the bands
as filled paths and one `\\addplot` per series. The real data sits in the source
rather than in an included PDF, so it can be recoloured or rescaled from LaTeX
without regenerating anything.

One panel, one file. An m by n grid does not become a `.tex` with m by n axes:
figures are composed in LaTeX, and whoever exports the source wants the single
panels.

tikzplotlib stopped at 0.10.1 and does not cope with recent matplotlib or numpy
2, so `_compat()` puts back the names it expects. They are aliases, not patches
to its logic: if the library is ever replaced, one function goes away.
"""
from __future__ import annotations

import re

# amsmath is needed for \text{} inside legend entries
PREAMBLE = (r"% In the document preamble: \usepackage{pgfplots,amsmath}"
            r" \pgfplotsset{compat=1.18}")


def _compat() -> None:
    """Put back the names tikzplotlib still uses and its dependencies removed."""
    import numpy as np
    import matplotlib.backends.backend_pgf as pgf
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D
    import webcolors

    if not hasattr(pgf, "common_texification"):        # rinominata in mpl 3.6
        pgf.common_texification = pgf._tex_escape
    if not hasattr(Legend, "legendHandles"):           # rinominata in mpl 3.9
        Legend.legendHandles = property(lambda self: self.legend_handles)
    if not hasattr(Legend, "_ncol"):                   # rinominata in mpl 3.6
        Legend._ncol = property(lambda self: self._ncols)
    if not hasattr(Line2D, "_us_dashSeq"):             # merged in mpl 3.6
        # The two attributes became the pair _unscaled_dash_pattern =
        # (offset, sequence). tikzplotlib reads them only inside a branch
        # guarded by is_dashed(), so the solid case is never reached. The
        # sequence stays the tuple _get_dash_pattern returns, so comparing
        # against the default dash pattern still gives the same answer.
        Line2D._us_dashSeq = property(lambda self: self._unscaled_dash_pattern[1])
        Line2D._us_dashOffset = property(lambda self: self._unscaled_dash_pattern[0])
    for old, new in (("float_", "float64"), ("alltrue", "all"), ("bool8", "bool_")):
        if not hasattr(np, old):                       # removed in numpy 2.0
            setattr(np, old, getattr(np, new))
    if not hasattr(webcolors, "CSS3_HEX_TO_NAMES"):    # removed in webcolors 24
        # tikzplotlib looks for the nearest CSS3 name by walking hex -> name and
        # keeping the first at minimum distance, so both the order and which of
        # two synonyms survives matter.
        #
        # Between synonyms the spelling xcolor knows wins: \color{gray},
        # \color{cyan} and \color{magenta} exist, grey, aqua and fuchsia do not,
        # and an unknown name makes the .tex fail to compile. Alphabetical order
        # picks the first, except for the two synonyms CSS3 added.
        PREFERITI = ("cyan", "magenta")
        mappa: dict[str, str] = {}
        for nome in sorted(webcolors.names(webcolors.CSS3)):
            hex_ = webcolors.name_to_hex(nome, spec=webcolors.CSS3)
            if hex_ not in mappa or nome in PREFERITI:
                mappa[hex_] = nome
        webcolors.CSS3_HEX_TO_NAMES = mappa


def unavailable_reason() -> str | None:
    """None if the .tex export is possible, otherwise why it is not."""
    try:
        _compat()
        import tikzplotlib  # noqa: F401
    except Exception as exc:
        return (f"Export LaTeX non disponibile ({type(exc).__name__}: {exc}). "
                f"Serve tikzplotlib: pip install -r requirements.txt")
    return None


TOKEN = "RTPLOTSLABEL%d"


def _use_latex_names(fig, styles: dict) -> dict:
    """Sostituisce le etichette con segnaposto, e dice a cosa vanno rimessi.

    I nomi `latex` di style.toml sono macro (`\\uppo`): matplotlib non le sa
    disegnare e tikzplotlib le escaperebbe. Si passa quindi da un segnaposto
    innocuo, rimpiazzato nel codice generato — cosi' il `.tex` esce con le macro
    del paper mentre l'anteprima continua a mostrare i nomi leggibili.
    """
    mapping = {}
    for ax in fig.axes:
        for i, line in enumerate(ax.get_lines()):
            latex = (styles.get(line.get_label()) or {}).get("latex")
            if not latex:
                continue
            token = TOKEN % len(mapping)
            mapping[token] = str(latex)
            line.set_label(token)
        legend = ax.get_legend()
        if legend is not None and mapping:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc=legend._loc, ncol=legend._ncols)
    return mapping


def _add_axis_options(code: str, options) -> str:
    """Append lines inside `\\begin{axis}[...]`; the last ones win."""
    extra = [str(o).rstrip(",") for o in options if str(o).strip()]
    if not extra:
        return code
    # il blocco di opzioni e' quello che sta fra "\begin{axis}[" e la prima "]"
    # at the start of a line: pgfplots does not nest brackets at that level
    head, sep, rest = code.partition("\\begin{axis}[\n")
    if not sep:
        return code
    body, closer, tail = rest.partition("\n]\n")
    if not closer:
        return code
    return head + sep + body + ",\n" + ",\n".join(extra) + closer + tail


def figure_to_tex(fig, header: str = "", styles: dict | None = None) -> str:
    """The tikzpicture code for a figure, which must have a single axis.

    `styles` e' la mappa etichetta -> stile prodotta da `figure.prepare`: da li'
    carries the LaTeX names of the series.
    """
    _compat()
    import tikzplotlib

    from . import rules as R

    names = _use_latex_names(fig, styles or {})
    code = tikzplotlib.get_tikz_code(figure=fig)
    for token, latex in names.items():
        code = code.replace(token, latex)
    code = _add_axis_options(code, R.get("latex", "axis_options") or [])
    # tikzplotlib scrive "\path [draw=...]": lo spazio dopo il comando e' legale
    # but makes the diff noisy when the same figure is regenerated
    code = re.sub(r"\\(path|addplot) \[", r"\\\1[", code)
    bits = [b for b in (header, R.latex_macros_comment()) if b]
    return "\n".join(bits + [code]) if bits else code
