"""The budget rule, the protocol guard, and the separation between the groups.

These cover what the submodule's own tests cannot see: they are about how a run
is *configured*, not about how the algorithm behaves once configured.
"""
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from utils.budget import ALPHA_MIN_PREFS, check_protocol, demo_budget, initial_queries, total_queries

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "experiments" / "configs"

# Shares taken from the arm files themselves, so the table below is a statement
# about the thesis and not a copy of the implementation.
QUOTE = {"demo_only": 0.0, "pref_soft": 0.05, "pref_bern": 0.20,
         "hybrid_soft": 0.10, "hybrid_bern": 0.10, "unw_soft": 0.10, "unw_bern": 0.10}

# What the thesis campaigns actually collected, budget by budget. The B=10 column
# holds the standard floor of 1; the raised floor of 5 is tested separately.
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
def test_le_quote_riproducono_le_query_iniziali_della_tesi(arm, attesi):
    usa_pref = arm != "demo_only"
    ottenuti = tuple(
        initial_queries(usa_pref, QUOTE[arm], b, floor=1) for b in (10, 100, 1000)
    )
    assert ottenuti == attesi


def test_il_pavimento_alzato_vale_solo_a_budget_dieci():
    # A B=10 il pavimento decide; ai budget maggiori la quota lo supera comunque,
    # quindi alzarlo non cambierebbe nulla -- ma la guardia lo vieta lo stesso,
    # perche' un protocollo deve descrivere una cosa sola.
    assert initial_queries(True, 0.10, 10, floor=ALPHA_MIN_PREFS) == 5
    assert initial_queries(True, 0.10, 100, floor=ALPHA_MIN_PREFS) == 10


def test_i_canali_spenti_azzerano_il_budget():
    assert total_queries(False, 1000) == 0
    assert initial_queries(False, 0.10, 1000, floor=5) == 0
    assert demo_budget(False, 1000) is None
    assert demo_budget(True, 1000) == 1000


def _cfg(**kw) -> DictConfig:
    base = dict(budget=10, initial_queries_min=5, arm_name="hybrid_soft",
                uses_preferences=True, uses_demonstrations=True)
    base.update(kw)
    return OmegaConf.create(base)


def test_la_guardia_accetta_la_combinazione_prevista():
    check_protocol(_cfg())


@pytest.mark.parametrize("kw,atteso", [
    (dict(budget=100), "budget=10"),
    (dict(budget=1000), "budget=10"),
    (dict(arm_name="pref_soft", uses_demonstrations=False), "two-channel"),
    (dict(arm_name="demo_only", uses_preferences=False), "two-channel"),
])
def test_la_guardia_rifiuta_le_combinazioni_sbagliate(kw, atteso):
    with pytest.raises(ValueError, match=atteso):
        check_protocol(_cfg(**kw))


def test_il_protocollo_standard_non_attiva_la_guardia():
    check_protocol(_cfg(initial_queries_min=1, budget=1000,
                        arm_name="demo_only", uses_preferences=False))


def test_bracci_e_protocolli_non_definiscono_la_stessa_chiave():
    """Entrambi i gruppi contengono chiavi `algo.*`: il vincolo e' sulla foglia.

    Il protocollo fissa loss_type, query_schedule, n_ensembles e altre otto;
    i bracci fissano lr_rew, net_arch e il resto. Se una foglia comparisse in
    entrambi, l'ordine di composizione deciderebbe in silenzio quale vince.
    """
    protocollo = _foglie(CONFIGS / "protocol" / "thesis.yaml")
    for arm in sorted(QUOTE):
        sovrapposte = protocollo & _foglie(CONFIGS / "arm" / f"{arm}.yaml")
        assert not sovrapposte, f"{arm} ridefinisce chiavi del protocollo: {sorted(sovrapposte)}"


def test_il_protocollo_tocca_davvero_chiavi_algo():
    """Se un giorno smettesse, il test sopra diventerebbe vacuo senza accorgersene."""
    algo = {k for k in _foglie(CONFIGS / "protocol" / "thesis.yaml") if k.startswith("algo.")}
    assert len(algo) >= 10, f"attese >=10 chiavi algo.* nel protocollo, trovate {len(algo)}"
