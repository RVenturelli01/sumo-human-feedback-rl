"""The composed configuration must match the runs that produced the results.

The fixtures come from the launcher this layer replaced, and were checked key by
key against what Weights & Biases recorded for the runs themselves. They are the
reference, not the output of what they verify.

A failure here means a configuration changed, which means an experiment changed.
"""
from pathlib import Path
import json

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from utils.budget import register_resolvers

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "experiments" / "configs"
FIXTURES = REPO / "tests" / "fixtures" / "reference_configs"

PREFISSI = ("algo.", "env.", "agent.", "train.", "eval.")
ESATTE = {"run.n_expert_trajectories", "run.n_expert_transitions", "run.demo_subsample_seed"}

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
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose("train", overrides=[
            f"arm={arm}", f"budget={budget}", "run.seed=1",
        ])
    return _semantico(_piatto(OmegaConf.to_container(cfg, resolve=True)))


def test_ci_sono_ventuno_fixture():
    assert len(CELLE) == 21, f"attese 21 celle (7 bracci x 3 budget), trovate {len(CELLE)}"


@pytest.mark.parametrize("cella", CELLE)
def test_la_config_composta_coincide_con_il_riferimento(cella):
    arm, budget = cella.rsplit("_B", 1)
    atteso = json.loads((FIXTURES / f"{cella}.json").read_text())
    ottenuto = _componi(arm, int(budget))

    mancanti = sorted(set(atteso) - set(ottenuto))
    aggiunte = sorted(set(ottenuto) - set(atteso))
    assert not mancanti, f"{cella}: chiavi perse dalla config: {mancanti}"
    assert not aggiunte, f"{cella}: chiavi comparse dal nulla: {aggiunte}"

    diverse = {k: (atteso[k], ottenuto[k]) for k in atteso if atteso[k] != ottenuto[k]}
    assert not diverse, f"{cella}: valori diversi (riferimento, ora): {diverse}"


# --- run identity ------------------------------------------------------------
# Deliberately outside the fixtures: two runs of the same cell differ in name and
# paths without differing as experiments. But these decide which W&B group a run
# lands in and where evaluate.py will look for it.

IDENTITA = [
    ("hybrid_soft", 1000, 3),
    ("hybrid_soft",   10, 1),
    ("unw_bern",      10, 9),
    ("demo_only",    100, 7),
    ("pref_bern",     10, 2),
]


def _config_intera(arm: str, budget: int, seed: int, campagna: str = "main"):
    register_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose("train", overrides=[
            f"arm={arm}", f"budget={budget}", f"run.seed={seed}", f"campaign={campagna}",
        ])
    return OmegaConf.to_container(cfg, resolve=True)


@pytest.mark.parametrize("arm,budget,seed", IDENTITA)
def test_l_identita_si_deriva_da_cio_che_identifica_la_run(arm, budget, seed):
    cfg = _config_intera(arm, budget, seed)
    gruppo = f"main_{arm}_B{budget}"
    nome = f"{gruppo}-seed{seed}"
    assert cfg["run"]["group"] == gruppo
    assert cfg["run"]["name"] == nome
    # make_run_dir aggiunge il nome della run, quindi qui c'e' solo il gruppo.
    assert cfg["run"]["output_dir"] == f"outputs/runs/{gruppo}"


@pytest.mark.parametrize("arm,budget,seed", IDENTITA)
def test_l_identita_non_resta_mai_nulla(arm, budget, seed):
    """Il difetto che questo test esiste per impedire: con `null` la run parte
    nel gruppo di ripiego, o non parte affatto.

    `wandb.entity` fa eccezione: nullo significa "l'account di chi lancia", che e'
    il default giusto per chiunque non sia il proprietario del progetto originale.
    """
    cfg = _config_intera(arm, budget, seed)
    for chiave in ("group", "name", "output_dir"):
        assert cfg["run"][chiave], f"run.{chiave} e' vuoto"
    for chiave in ("project", "tags"):
        assert cfg["wandb"][chiave], f"wandb.{chiave} e' vuoto"


def test_l_etichetta_di_campagna_separa_le_run():
    """Due campagne nello stesso progetto W&B non devono mescolare i seed."""
    a = _config_intera("hybrid_soft", 10, 1, campagna="main")["run"]["group"]
    b = _config_intera("hybrid_soft", 10, 1, campagna="retune")["run"]["group"]
    assert a != b and a.endswith("_hybrid_soft_B10") and b.endswith("_hybrid_soft_B10")


def test_i_tag_wandb_descrivono_la_cella():
    cfg = _config_intera("hybrid_soft", 10, 1)
    assert cfg["wandb"]["tags"] == ["main", "hybrid_soft", "B10"]
