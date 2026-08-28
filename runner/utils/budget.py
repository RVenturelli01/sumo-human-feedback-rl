"""How the feedback budget B becomes the numbers the algorithm needs.

Kept separate from train.py so the tests can import it without pulling in torch
or the simulator: checking a configuration should not require SUMO.

    total_queries            comparisons the arm may ask for
    n_expert_trajectories    demonstrations it reads
    initial_queries          comparisons collected before the regular schedule
"""
from __future__ import annotations

from omegaconf import OmegaConf

#: Below this many comparisons the reliability weight is not estimable.
#: Mirrors ALPHA_MIN_PREFS in human_feedback_rl.algorithms.hybrid_algorithm.
ALPHA_MIN_PREFS = 5


def total_queries(uses_preferences: bool, budget: int) -> int:
    return int(budget) if uses_preferences else 0


def demo_budget(uses_demonstrations: bool, budget: int) -> int | None:
    """None leaves the key at its default, which is the whole dataset."""
    return int(budget) if uses_demonstrations else None


def initial_queries(uses_preferences: bool, share: float, budget: int, floor: int) -> int:
    """max(floor, round(share * budget)).

    The share and the floor both come from the arm. The floor only bites at small
    budgets: at B=100 a 10% share already clears it. round() is banker's, so
    0.05 * 10 gives 0 and the floor decides.
    """
    if not uses_preferences:
        return 0
    return max(int(floor), round(float(share) * int(budget)))


def register_resolvers() -> None:
    """Expose the three rules to Hydra as ${hfrl.*}.

    Resolving them in the configuration rather than in code is what lets
    `--cfg job --resolve` show the values a run will actually use.
    """
    for name, fn in (("hfrl.total_queries", total_queries),
                     ("hfrl.demo_budget", demo_budget),
                     ("hfrl.initial_queries", initial_queries)):
        OmegaConf.register_new_resolver(name, fn, replace=True)
