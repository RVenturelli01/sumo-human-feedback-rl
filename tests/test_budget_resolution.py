"""The budget rule and the separation between the two config groups.

These cover what the submodule's tests cannot see: how a run is configured,
rather than how the algorithm behaves once it is.
"""
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from utils.budget import ALPHA_MIN_PREFS, demo_budget, initial_queries, total_queries

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "runner" / "configs"

# Shares taken from the arm files themselves, so the table below is a statement
# about the reference runs and not a copy of the implementation.
SHARES = {"demo_only": 0.0, "pref_soft": 0.05, "pref_bern": 0.20,
         "hybrid_soft": 0.10, "hybrid_bern": 0.10, "unw_soft": 0.10, "unw_bern": 0.10}

# What the reference campaigns collected, budget by budget, at floor 1. The
# per-arm floor is checked separately.
EXPECTED = {
    "demo_only":   (0, 0, 0),
    "pref_soft":   (1, 5, 50),
    "pref_bern":   (2, 20, 200),
    "hybrid_soft": (1, 10, 100),
    "hybrid_bern": (1, 10, 100),
    "unw_soft":    (1, 10, 100),
    "unw_bern":    (1, 10, 100),
}


def _leaf_keys(path: Path) -> set[str]:
    """Dotted leaf keys of a config file, ignoring the `defaults` list."""
    cfg = OmegaConf.load(path)
    out: set[str] = set()

    def descend(node, prefix=""):
        for k, v in node.items():
            if k == "defaults":
                continue
            key = f"{prefix}{k}"
            if hasattr(v, "items"):
                descend(v, key + ".")
            else:
                out.add(key)

    descend(cfg)
    return out


@pytest.mark.parametrize("arm,expected", EXPECTED.items())
def test_shares_reproduce_the_reference_initial_queries(arm, expected):
    uses_prefs = arm != "demo_only"
    obtained = tuple(
        initial_queries(uses_prefs, SHARES[arm], b, floor=1) for b in (10, 100, 1000)
    )
    assert obtained == expected


def test_a_channel_that_is_off_gets_no_budget():
    assert total_queries(False, 1000) == 0
    assert initial_queries(False, 0.10, 1000, floor=5) == 0
    assert demo_budget(False, 1000) is None
    assert demo_budget(True, 1000) == 1000


FLOORS = {"demo_only": 1, "pref_soft": 1, "pref_bern": 1,
             "hybrid_soft": 5, "hybrid_bern": 5, "unw_soft": 5, "unw_bern": 5}


@pytest.mark.parametrize("arm,expected", FLOORS.items())
def test_the_floor_is_declared_by_the_arm(arm, expected):
    """Five only where the reliability weight has to be estimated; one elsewhere."""
    cfg = OmegaConf.load(CONFIGS / "arm" / f"{arm}.yaml")
    assert cfg.initial_queries_min == expected


def test_at_budget_ten_the_floor_binds_only_where_it_is_needed():
    assert initial_queries(True, 0.10, 10, floor=ALPHA_MIN_PREFS) == 5   # two channels
    assert initial_queries(True, 0.05, 10, floor=1) == 1                 # preferences only
    # At larger budgets the share clears the floor, which then does nothing.
    assert initial_queries(True, 0.10, 100, floor=ALPHA_MIN_PREFS) == 10


def test_arms_and_protocol_never_define_the_same_key():
    """Both groups write `algo.*` keys: the constraint is on the leaf.

    The protocol fixes loss_type, query_schedule, n_ensembles and eight more;
    the arms fix lr_rew, net_arch and the rest. If a leaf appeared in both, the
    composition order alone would silently decide which value wins.
    """
    protocol = _leaf_keys(CONFIGS / "protocol" / "standard.yaml")
    for arm in sorted(SHARES):
        overlap = protocol & _leaf_keys(CONFIGS / "arm" / f"{arm}.yaml")
        assert not overlap, f"{arm} redefines protocol keys: {sorted(overlap)}"


def test_the_protocol_really_does_touch_algo_keys():
    """If it ever stopped, the test above would go vacuous without saying so."""
    algo = {k for k in _leaf_keys(CONFIGS / "protocol" / "standard.yaml") if k.startswith("algo.")}
    assert len(algo) >= 10, f"expected >=10 algo.* keys in the protocol, found {len(algo)}"
