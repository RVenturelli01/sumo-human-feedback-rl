"""The composed configuration must equal the one that produced the thesis.

The fixtures in `fixtures/thesis_resolved_configs/` were generated from the
*previous* code path -- the base YAML plus the overrides emitted by
`scripts/launch_thesis_runs.py` -- and cross-checked, key by key, against the
configurations Weights & Biases recorded for the runs themselves. They are the
reference, not the output of what they verify.

If one of these tests fails, the reorganisation changed an experiment. That is
the only thing it can mean.
"""
from pathlib import Path
import json

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from utils.budget import register_resolvers

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "experiments" / "configs"
FIXTURES = REPO / "tests" / "fixtures" / "thesis_resolved_configs"

PREFISSI = ("algo.", "env.", "agent.", "train.", "eval.")
ESATTE = {"run.n_expert_trajectories", "run.n_expert_transitions", "run.demo_subsample_seed"}

# At B=10 the four two-channel arms ran under the raised initial-query floor.
IQ5 = {"hybrid_soft", "hybrid_bern", "unw_soft", "unw_bern"}
CELLE = sorted(p.stem for p in FIXTURES.glob("*.json"))


def _piatto(cfg, pre=""):
    out = {}
    for k, v in cfg.items():
        key = f"{pre}{k}"
        if hasattr(v, "items"):
            out.update(_piatto(v, key + "."))
        else:
            out[key] = v
    return out


def _semantico(d):
    """Only the keys that define the experiment.

    Run name, seed, output directory and W&B metadata are deliberately out: they
    differ between two runs of the same cell without changing what is run.
    """
    return {k: v for k, v in d.items() if k.startswith(PREFISSI) or k in ESATTE}


def _componi(arm: str, budget: int):
    register_resolvers()
    protocollo = "thesis_b10" if (budget == 10 and arm in IQ5) else "thesis"
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose("train", overrides=[
            f"arm={arm}", f"protocol={protocollo}", f"budget={budget}", "run.seed=1",
        ])
    return _semantico(_piatto(OmegaConf.to_container(cfg, resolve=True)))


def test_ci_sono_ventuno_fixture():
    assert len(CELLE) == 21, f"attese 21 celle (7 bracci x 3 budget), trovate {len(CELLE)}"


@pytest.mark.parametrize("cella", CELLE)
def test_la_config_composta_coincide_con_quella_della_tesi(cella):
    arm, budget = cella.rsplit("_B", 1)
    atteso = json.loads((FIXTURES / f"{cella}.json").read_text())
    ottenuto = _componi(arm, int(budget))

    mancanti = sorted(set(atteso) - set(ottenuto))
    aggiunte = sorted(set(ottenuto) - set(atteso))
    assert not mancanti, f"{cella}: chiavi perse dalla config: {mancanti}"
    assert not aggiunte, f"{cella}: chiavi comparse dal nulla: {aggiunte}"

    diverse = {k: (atteso[k], ottenuto[k]) for k in atteso if atteso[k] != ottenuto[k]}
    assert not diverse, f"{cella}: valori diversi (tesi, ora): {diverse}"


# --- run identity ------------------------------------------------------------
# These keys are deliberately outside the semantic fixtures: two runs of the same
# cell differ in name, seed and paths without differing as experiments. But they
# are what decides which W&B group a run lands in and where `evaluate.py
# --aggregate` will look for it, so they need their own check.

IDENTITA = [
    ("hybrid_soft", "thesis",     1000, 3, "th_1mh4_hybrid_soft_B1000"),
    ("hybrid_soft", "thesis_b10",   10, 1, "th_1mh4iq5_hybrid_soft_B10"),
    ("unw_bern",    "thesis_b10",   10, 9, "th_1mh4iq5_unw_bern_B10"),
    ("demo_only",   "thesis",      100, 7, "th_1mh4_demo_only_B100"),
    ("pref_bern",   "thesis",       10, 2, "th_1mh4_pref_bern_B10"),
]


def _config_intera(arm: str, protocollo: str, budget: int, seed: int):
    register_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose("train", overrides=[
            f"arm={arm}", f"protocol={protocollo}", f"budget={budget}", f"run.seed={seed}",
        ])
    return OmegaConf.to_container(cfg, resolve=True)


@pytest.mark.parametrize("arm,protocollo,budget,seed,gruppo", IDENTITA)
def test_l_identita_della_run_e_quella_delle_campagne(arm, protocollo, budget, seed, gruppo):
    cfg = _config_intera(arm, protocollo, budget, seed)
    atteso_nome = f"{gruppo}-seed{seed}"
    assert cfg["run"]["group"] == gruppo
    assert cfg["run"]["name"] == atteso_nome
    # Il doppio annidamento non e' un refuso: train.py crea una sottocartella col
    # nome della run dentro output_dir, ed e' il percorso che l'aggregatore cerca.
    assert cfg["run"]["output_dir"] == f"outputs/thesis_runs/{gruppo}/{atteso_nome}"


@pytest.mark.parametrize("arm,protocollo,budget,seed,gruppo", IDENTITA)
def test_l_identita_non_resta_mai_nulla(arm, protocollo, budget, seed, gruppo):
    """Il difetto che questo test esiste per impedire: con `null` la run parte
    nel gruppo di ripiego, o non parte affatto."""
    cfg = _config_intera(arm, protocollo, budget, seed)
    for chiave in ("group", "name", "output_dir"):
        assert cfg["run"][chiave], f"run.{chiave} e' vuoto"
    for chiave in ("entity", "project", "tags"):
        assert cfg["wandb"][chiave], f"wandb.{chiave} e' vuoto"


def test_i_tag_wandb_descrivono_la_cella():
    cfg = _config_intera("hybrid_soft", "thesis_b10", 10, 1)
    assert cfg["wandb"]["tags"] == ["thesis_final", "1mh4iq5", "hybrid_soft", "B10"]
    assert cfg["wandb"]["project"] == "thesis-final"


def test_il_protocollo_b10_cambia_la_campagna():
    """Se le due campagne finissero nello stesso gruppo, le run superate a B=10
    si mescolerebbero con quelle della tesi."""
    normale = _config_intera("hybrid_soft", "thesis", 10, 1)["run"]["group"]
    variante = _config_intera("hybrid_soft", "thesis_b10", 10, 1)["run"]["group"]
    assert normale != variante
    assert "iq5" in variante and "iq5" not in normale
