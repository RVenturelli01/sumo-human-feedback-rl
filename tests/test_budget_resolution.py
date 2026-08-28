"""The budget rule and the separation between the two config groups.

These cover what the submodule's tests cannot see: how a run is configured,
rather than how the algorithm behaves once it is.
"""
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from utils.budget import ALPHA_MIN_PREFS, demo_budget, initial_queries, total_queries

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "experiments" / "configs"

# Shares taken from the arm files themselves, so the table below is a statement
# about the reference runs and not a copy of the implementation.
QUOTE = {"demo_only": 0.0, "pref_soft": 0.05, "pref_bern": 0.20,
         "hybrid_soft": 0.10, "hybrid_bern": 0.10, "unw_soft": 0.10, "unw_bern": 0.10}

# What the reference campaigns collected, budget by budget, at floor 1. The
# per-arm floor is checked separately.
ATTESI = {
    "demo_only":   (0, 0, 0),
    "pref_soft":   (1, 5, 50),
    "pref_bern":   (2, 20, 200),
    "hybrid_soft": (1, 10, 100),
    "hybrid_bern": (1, 10, 100),
    "unw_soft":    (1, 10, 100),
    "unw_bern":    (1, 10, 100),
}


def _foglie(percorso: Path) -> set[str]:
    """Dotted leaf keys of a config file, ignoring the `defaults` list."""
    cfg = OmegaConf.load(percorso)
    out: set[str] = set()

    def scendi(nodo, pre=""):
        for k, v in nodo.items():
            if k == "defaults":
                continue
            chiave = f"{pre}{k}"
            if hasattr(v, "items"):
                scendi(v, chiave + ".")
            else:
                out.add(chiave)

    scendi(cfg)
    return out


@pytest.mark.parametrize("arm,attesi", ATTESI.items())
def test_le_quote_riproducono_le_query_iniziali_di_riferimento(arm, attesi):
    usa_pref = arm != "demo_only"
    ottenuti = tuple(
        initial_queries(usa_pref, QUOTE[arm], b, floor=1) for b in (10, 100, 1000)
    )
    assert ottenuti == attesi


def test_i_canali_spenti_azzerano_il_budget():
    assert total_queries(False, 1000) == 0
    assert initial_queries(False, 0.10, 1000, floor=5) == 0
    assert demo_budget(False, 1000) is None
    assert demo_budget(True, 1000) == 1000


PAVIMENTI = {"demo_only": 1, "pref_soft": 1, "pref_bern": 1,
             "hybrid_soft": 5, "hybrid_bern": 5, "unw_soft": 5, "unw_bern": 5}


@pytest.mark.parametrize("arm,atteso", PAVIMENTI.items())
def test_il_pavimento_e_dichiarato_dal_braccio(arm, atteso):
    """Cinque solo dove il peso di affidabilita' va stimato; uno altrove."""
    cfg = OmegaConf.load(CONFIGS / "arm" / f"{arm}.yaml")
    assert cfg.initial_queries_min == atteso


def test_a_budget_dieci_il_pavimento_decide_solo_dove_serve():
    assert initial_queries(True, 0.10, 10, floor=ALPHA_MIN_PREFS) == 5   # due canali
    assert initial_queries(True, 0.05, 10, floor=1) == 1                 # solo preferenze
    # Ai budget maggiori la quota supera il pavimento e lo rende ininfluente.
    assert initial_queries(True, 0.10, 100, floor=ALPHA_MIN_PREFS) == 10


def test_bracci_e_protocolli_non_definiscono_la_stessa_chiave():
    """Entrambi i gruppi contengono chiavi `algo.*`: il vincolo e' sulla foglia.

    Il protocollo fissa loss_type, query_schedule, n_ensembles e altre otto;
    i bracci fissano lr_rew, net_arch e il resto. Se una foglia comparisse in
    entrambi, l'ordine di composizione deciderebbe in silenzio quale vince.
    """
    protocollo = _foglie(CONFIGS / "protocol" / "standard.yaml")
    for arm in sorted(QUOTE):
        sovrapposte = protocollo & _foglie(CONFIGS / "arm" / f"{arm}.yaml")
        assert not sovrapposte, f"{arm} ridefinisce chiavi del protocollo: {sorted(sovrapposte)}"


def test_il_protocollo_tocca_davvero_chiavi_algo():
    """Se un giorno smettesse, il test sopra diventerebbe vacuo senza accorgersene."""
    algo = {k for k in _foglie(CONFIGS / "protocol" / "standard.yaml") if k.startswith("algo.")}
    assert len(algo) >= 10, f"attese >=10 chiavi algo.* nel protocollo, trovate {len(algo)}"
