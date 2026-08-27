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
