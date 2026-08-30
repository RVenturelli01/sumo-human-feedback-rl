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
CONFIGS = REPO / "runner" / "configs"
FIXTURES = REPO / "tests" / "fixtures" / "reference_configs"

PREFIXES = ("algo.", "env.", "agent.", "train.", "eval.")
EXACT = {"run.n_expert_trajectories", "run.n_expert_transitions", "run.demo_subsample_seed"}

CELLS = sorted(p.stem for p in FIXTURES.glob("*.json"))


def _flat(cfg, prefix=""):
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if hasattr(v, "items"):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _semantic(d):
    """Only the keys that define the experiment.

    Run name, seed, output directory and W&B metadata are deliberately out: they
    differ between two runs of the same cell without changing what is run.
    """
    return {k: v for k, v in d.items() if k.startswith(PREFIXES) or k in EXACT}


def _compose(arm: str, budget: int):
    register_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose("train", overrides=[
            f"arm={arm}", f"budget={budget}", "run.seed=1",
        ])
    return _semantic(_flat(OmegaConf.to_container(cfg, resolve=True)))


def test_there_are_twenty_one_fixtures():
    assert len(CELLS) == 21, f"expected 21 cells (7 arms x 3 budgets), found {len(CELLS)}"


@pytest.mark.parametrize("cell", CELLS)
def test_the_composed_config_matches_the_reference(cell):
    arm, budget = cell.rsplit("_B", 1)
    expected = json.loads((FIXTURES / f"{cell}.json").read_text())
    obtained = _compose(arm, int(budget))

    missing = sorted(set(expected) - set(obtained))
    added = sorted(set(obtained) - set(expected))
    assert not missing, f"{cell}: keys lost from the config: {missing}"
    assert not added, f"{cell}: keys appeared from nowhere: {added}"

    differing = {k: (expected[k], obtained[k]) for k in expected if expected[k] != obtained[k]}
    assert not differing, f"{cell}: differing values (reference, now): {differing}"


# --- run identity ------------------------------------------------------------
# Deliberately outside the fixtures: two runs of the same cell differ in name and
# paths without differing as experiments. But these decide which W&B group a run
# lands in and where evaluate.py will look for it.

IDENTITY = [
    ("hybrid_soft", 1000, 3),
    ("hybrid_soft",   10, 1),
    ("unw_bern",      10, 9),
    ("demo_only",    100, 7),
    ("pref_bern",     10, 2),
]


def _whole_config(arm: str, budget: int, seed: int, campaign: str = "main"):
    register_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose("train", overrides=[
            f"arm={arm}", f"budget={budget}", f"run.seed={seed}", f"campaign={campaign}",
        ])
    return OmegaConf.to_container(cfg, resolve=True)


@pytest.mark.parametrize("arm,budget,seed", IDENTITY)
def test_identity_follows_from_what_identifies_the_run(arm, budget, seed):
    cfg = _whole_config(arm, budget, seed)
    group = f"main_{arm}_B{budget}"
    name = f"{group}-seed{seed}"
    assert cfg["run"]["group"] == group
    assert cfg["run"]["name"] == name
    # make_run_dir appends the run name, so this is just the group.
    assert cfg["run"]["output_dir"] == f"outputs/runs/{group}"


@pytest.mark.parametrize("arm,budget,seed", IDENTITY)
def test_identity_is_never_left_null(arm, budget, seed):
    """The defect this test exists to prevent: with `null` the run starts in the
    fallback group, or does not start at all.

    `wandb.entity` is the exception: null means "whoever is launching", which is
    the right default for anyone who is not the owner of the original project.
    """
    cfg = _whole_config(arm, budget, seed)
    for key in ("group", "name", "output_dir"):
        assert cfg["run"][key], f"run.{key} is empty"
    for key in ("project", "tags"):
        assert cfg["wandb"][key], f"wandb.{key} is empty"


def test_the_campaign_label_keeps_runs_apart():
    """Two campaigns in the same W&B project must not mix their seeds."""
    a = _whole_config("hybrid_soft", 10, 1, campaign="main")["run"]["group"]
    b = _whole_config("hybrid_soft", 10, 1, campaign="retune")["run"]["group"]
    assert a != b and a.endswith("_hybrid_soft_B10") and b.endswith("_hybrid_soft_B10")


def test_the_wandb_tags_describe_the_cell():
    cfg = _whole_config("hybrid_soft", 10, 1)
    assert cfg["wandb"]["tags"] == ["main", "hybrid_soft", "B10"]
