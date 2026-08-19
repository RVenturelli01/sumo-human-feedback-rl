"""Schema dei campi dell'indice: un'unica dichiarazione per colonna.

Stessa idea del progetto di ispirazione (vedi `plots/README.md`): ogni colonna
si dichiara una volta sola con come si scrive nella UI, se puo' finire su
righe/colonne/colori e cosa aggiunge alla legenda — invece di ripetere le
stesse liste a mano nella sidebar, nei filtri e nei titoli dei pannelli.

Le colonne vengono da `rtplots/source.py` (config Hydra -> riga dell'indice).
L'ordine qui sotto e' anche l'ordine della sidebar del selettore.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# Nomi leggibili dei bracci (title case, monospace nel grafico via labels.py).
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


# Schemi di fusione dei due canali (`algo.kwargs.gcl_fusion`). I nomi brevi
# sono quelli usati con il relatore, il valore grezzo resta quello del codice.
FUSION_NAMES = {
    "norm_balance": "norm_balance (baseline)",
    "alpha_norm_single_adam": "prova 1 (un Adam sul gradiente fuso)",
    "dual_adam_alpha": "prova 2 (un Adam per canale)",
    "dual_adam_sum": "due Adam, somma",
    "dual_adam_alpha_unit": "due Adam, alpha + budget",
    "dual_adam_alpha_unit_nobudget": "due Adam, alpha su direzioni unitarie",
    # Schemi provati prima della richiesta del relatore e poi rimossi da
    # VALID_GCL_FUSIONS: le run restano nell'indice, il nome le marca.
    "dual_adam_reliability": "dual_adam_reliability (storico)",
    "demo_anchor_inv_var": "demo_anchor_inv_var (storico)",
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


def _bool_html(yes: str = "sì", no: str = "no"):
    return lambda v: yes if v in (True, "True") else no


def _int_html(suffix: str = ""):
    return lambda v: f"{_int(v)}{suffix}"


def _num_html(fmt: str = "{:g}"):
    """Come _int_html ma per i float: alza su valori mancanti invece di stampare
    'nan' (float(nan) non alza da solo, a differenza di int(nan))."""
    def _fmt(v):
        if _missing(v):
            raise ValueError("missing")
        return fmt.format(float(v))
    return _fmt


def _eps(value) -> float:
    """Il valore di label_smoothing, con 0 per le run che non lo dichiarano."""
    return 0.0 if _missing(value) else float(value)


def _smoothing_html(value) -> str:
    eps = _eps(value)
    return "senza smoothing" if not eps else f"con smoothing (eps={eps:g})"


def _millions_html(v):
    if _missing(v):
        raise ValueError("missing")
    return f"{float(v) / 1e6:g}M"


FIELDS: list[Field] = [
    Field(
        "arm", "Algoritmo", ui=True, grid=True, series=True,
        html=lambda v: ARM_NAMES.get(v, str(v)), title_of=lambda v: ARM_NAMES.get(v, str(v)),
        legend=lambda v: ARM_NAMES.get(v, str(v)),
    ),
    # Ne' in sidebar ne' fra le dimensioni di griglia/copertura: ridondanti con
    # "Algoritmo", che gia' elenca le 8 combinazioni (i 4 bracci base + le 4 di
    # hybrid) come pillole selezionabili una per una. Restano colonne vere,
    # quindi restano filtrabili da riga di comando (--filter arm_family=hybrid).
    # Come i due canali vengono combinati (solo hybrid). In sidebar e fra le
    # dimensioni di griglia perche' e' cio' che distingue i bracci di
    # thesis-grad-diagnostics fra loro: senza, baseline, prova 1 e prova 2
    # collassano tutti in "hybrid_demo_2 (soft)".
    Field(
        "fusion", "Fusione dei gradienti", ui=True, grid=True, series=True,
        html=lambda v: FUSION_NAMES.get(v, str(v)),
        title_of=lambda v: FUSION_NAMES.get(v, str(v)),
        legend=lambda v: FUSION_NAMES.get(v, str(v)),
    ),
    Field("arm_family", "Famiglia", html=str, title_of=str),
    Field("demo_loss", "Loss dimostrazioni", html=str),
    Field("pref_labels", "Etichette preferenze", html=str),
    Field("demo_mode", "Modo dimostrazioni", html=str),
    Field(
        "query_budget", "Budget preferenze (1 transizione)", ui=True, grid=True, series=True,
        html=_int_html(),
        title_of=lambda v: f"query = {_int(v)}",
        legend=lambda v: f"{_int(v)} query",
    ),
    Field(
        "demo_budget", "Budget dimostrazioni (traiettoria)", ui=True, grid=True, series=True,
        html=lambda v: "dataset intero" if _missing(v) else f"{_int(v)} traiettorie",
        title_of=lambda v: "dataset intero" if _missing(v) else f"{_int(v)} traiettorie",
        legend=lambda v: None if _missing(v) else f"{_int(v)} traj",
    ),
    # Non in sidebar ne' in copertura/righe/colonne: resta una colonna vera,
    # usata come asse x di default per le curve di budget (vedi "asse budget"
    # nel toolbar del grafico — un controllo a parte, non le dimensioni qui).
    # In griglia (una riga per budget) ma non in sidebar: il filtro si fa gia'
    # con query_budget/demo_budget. Titolo volutamente neutro, "B = 10": una
    # riga attraversa piu' bracci, e B significa 10 preferenze + 10 traiettorie
    # per l'ibrido ma solo una delle due per i bracci a sorgente singola, quindi
    # la precisazione va nella didascalia, non nel titolo del pannello.
    Field(
        "budget_level", "Budget B (dal gruppo)", grid=True,
        html=_int_html(), title_of=lambda v: f"B = {_int(v)}",
        legend=lambda v: f"B={_int(v)}",
    ),
    Field("normalize_agent_reward", "Reward normalizzata", ui=True, grid=True, series=True,
          html=_bool_html(), title_of=lambda v: f"normalize_agent_reward = {bool(v)}",
          legend=lambda v: "norm" if v else "no-norm"),
    # Il valore di eps, non un booleano: oggi c'e' un solo livello (0.1) e in UI
    # si legge come con/senza, ma se un domani arriva un secondo eps le curve si
    # separano da sole invece di collassare due configurazioni in "con".
    Field("label_smoothing", "Label smoothing", ui=True, grid=True, series=True,
          html=_smoothing_html, title_of=_smoothing_html,
          legend=lambda v: None if not _eps(v) else f"eps={_eps(v):g}"),
    Field("query_schedule", "Schedule query", ui=True, grid=True, series=True, html=str),
    Field("fragmenter_type", "Fragmenter", ui=True, grid=True, series=True, html=str),
    # Ne' in sidebar ne' in copertura/righe/colonne/legenda: iperparametri del
    # best-config per livello di budget, non dimensioni su cui filtrare o
    # separare le curve. Restano colonne vere, filtrabili da riga di comando.
    Field("initial_queries", "Query iniziali (bootstrap)",
          html=_int_html(), title_of=lambda v: f"initial_queries = {_int(v)}"),
    Field("demo_weight", "Peso dimostrazioni",
          html=_num_html(), title_of=lambda v: f"demo_weight = {float(v):g}"),
    Field("pref_temperature", "Temperatura oracolo", html=_num_html()),
    Field("reward_net_arch", "Rete reward model", html=str),
    Field("demo_subsample_seed", "Seed subsample demo",
          html=lambda v: "= seed" if _missing(v) else str(_int(v))),
    Field("total_timesteps", "Timesteps totali", html=_millions_html),
    Field("state", "Stato", ui=True, html=str),
    Field("project", "Progetto W&B", ui=True, html=str),
    Field("group_tag", "Tag gruppo", ui=True, html=str),
    # non filtrabile dalla sidebar (troppi valori unici), ma disponibile per la
    # tabella di copertura e la legenda
    Field("group", "Gruppo W&B"),
    Field("seed", "Seed", legend=lambda v: f"seed={_int(v)}"),
]

BY_COL: dict[str, Field] = {f.col: f for f in FIELDS}

UI_DIMENSIONS = [f.col for f in FIELDS if f.ui]
GRID_FIELDS = [f.col for f in FIELDS if f.grid]
# Dimensioni che devono separare le curve: se una varia e nessuno l'ha messa su
# colori/righe/colonne, configurazioni diverse finiscono mediate insieme.
SERIES_FIELDS = [f.col for f in FIELDS if f.series]


def title(col: str) -> str:
    f = BY_COL.get(col)
    return f.title if f else col


def html_value(col: str, value) -> str:
    """Valore come va scritto nella pagina del selettore."""
    f = BY_COL.get(col)
    if f is not None and f.html is not None:
        try:
            return f.html(value)
        except (TypeError, ValueError, KeyError):
            pass
    if _missing(value):
        return "—"
    if value in (True, "True"):
        return "sì"
    if value in (False, "False"):
        return "no"
    return str(value)


def panel_title(col: str, value, paper: bool = True) -> str:
    """Titolo di riga/colonna della griglia."""
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
