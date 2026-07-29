"""Lettura di `plots/style.toml`: le regole scritte a mano che governano i grafici.

Un file solo, dichiarativo, riletto quando cambia (mtime): si salva e la figura
successiva — anteprima o `.tex` — e' gia' quella nuova, senza riavviare niente.

Le regole `[[series]]` sono una lista ordinata: vince la prima che combacia.
"""
from __future__ import annotations

from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # backport, questa repo gira anche su 3.10

from .paths import PLOTS_ROOT

RULES_FILE = PLOTS_ROOT / "style.toml"

# Scorte: servono solo se una chiave sparisce dal .toml, che invece le elenca
# tutte. Tenerle qui evita che una riga cancellata per sbaglio rompa la pagina.
FALLBACK = {
    "figure": {"panel_size": [3.4, 2.5],
               "xlabel": "Environment timesteps", "ylabel": "Mean return",
               "xscale": 1e6, "share": "row", "font_scale": 1.0},
    "lines": {"width": 1.4, "band_alpha": 0.18, "band": "se", "smooth": 5,
              "baseline_color": "#000000", "baseline_width": 1.6},
    # "outside_right" di default: con 5-8 serie in un pannello solo (tutti gli
    # arm sovrapposti) una legenda "best" dentro gli assi finisce quasi sempre
    # sopra ai dati.
    "legend": {"where": "outside_right", "loc": "best", "ncol": 1, "frame": True,
               "font_size": 8.5},
    "palette": {"colors": ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                           "#4a3aa7", "#e34948", "#8a4b08", "#0f7ea6"]},
    "latex": {"axis_options": [], "preamble": "", "macros": []},
}

_cache: dict | None = None
_stamp: tuple | None = None


def load(force: bool = False) -> dict:
    """Il file, riletto se e' cambiato sul disco."""
    global _cache, _stamp
    try:
        stat = RULES_FILE.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = None
    if _cache is not None and stamp == _stamp and not force:
        return _cache
    data = {}
    if stamp is not None:
        try:
            data = tomllib.loads(RULES_FILE.read_text())
        except tomllib.TOMLDecodeError as exc:
            print(f"[rules] {RULES_FILE} non e' TOML valido ({exc}): uso i default")
    _cache, _stamp = data, stamp
    return data


def get(section: str, key: str, default=None):
    """Un valore del file, con la scorta di FALLBACK se manca."""
    value = (load().get(section) or {}).get(key)
    if value is not None:
        return value
    if default is not None:
        return default
    return FALLBACK.get(section, {}).get(key)


def series_rules() -> list[dict]:
    return list(load().get("series") or [])


def rule_for(row) -> dict:
    """La prima regola `[[series]]` che combacia con la serie (o {})."""
    for rule in series_rules():
        wanted = rule.get("match") or {}
        if not wanted:
            continue
        if all(_same(row.get(col), value) for col, value in wanted.items()):
            return rule
    return {}


def _same(actual, wanted) -> bool:
    """Confronto tollerante: nell'indice alcune colonne numeriche sono float."""
    if actual is None:
        return False
    if isinstance(wanted, bool) or isinstance(actual, bool):
        return bool(actual) is bool(wanted)
    try:
        return float(actual) == float(wanted)
    except (TypeError, ValueError):
        return str(actual) == str(wanted)


def palette() -> list[str]:
    return list(get("palette", "colors"))


def latex_macros_comment() -> str:
    """Le macro dei nomi, come commento in cima ai .tex esportati."""
    macros = get("latex", "macros") or []
    preamble = get("latex", "preamble") or ""
    lines = []
    if preamble:
        lines.append(f"% {preamble}")
    lines += [f"% {m}" for m in macros]
    return "\n".join(lines)
