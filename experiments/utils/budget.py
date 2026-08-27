"""How the feedback budget B becomes the three numbers the algorithm needs.

This is the only derived logic in the experiment layer, and it lives on its own
so that both `train.py` and the tests can use it without importing torch, wandb
or the simulator: checking a configuration should not require SUMO.

The three quantities:

    total_queries            comparisons the arm may ask for
    n_expert_trajectories    demonstrations it reads
    initial_queries          comparisons collected before the regular schedule

Each arm derives them from its own `uses_preferences`, `uses_demonstrations` and
`initial_queries_frac`, which is why a single rule for all arms would be wrong:
the shares are 0 for the demonstration-only arm, 0.05 and 0.20 for the two
preference-only baselines, and 0.10 for the four two-channel methods.
"""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

#: Comparisons below which the reliability weight is not estimable.
#: Mirrors ALPHA_MIN_PREFS in human_feedback_rl.algorithms.hybrid_algorithm.
ALPHA_MIN_PREFS = 5


def total_queries(uses_preferences: bool, budget: int) -> int:
    """The budget, or none at all without a preference channel."""
    return int(budget) if uses_preferences else 0


def demo_budget(uses_demonstrations: bool, budget: int) -> int | None:
    """The budget, or ``None`` to leave the key at its default (whole dataset)."""
    return int(budget) if uses_demonstrations else None


def initial_queries(uses_preferences: bool, share: float, budget: int, floor: int) -> int:
    """``max(floor, round(share * budget))``, and 0 without a preference channel.

    The floor is 1 under the standard protocol: with 0 the bootstrap would train
    on no feedback at all and the first iteration would have no signal. It is
    :data:`ALPHA_MIN_PREFS` under ``thesis_b10``, where a 10‑comparison budget
    would otherwise start with a single one and leave the reliability weight
    pinned for most of the run.

    ``round`` is Python's banker's rounding, as in the original launcher, so
    ``0.05 * 10`` gives 0 and the floor of 1 decides.
    """
    if not uses_preferences:
        return 0
    return max(int(floor), round(float(share) * int(budget)))


def register_resolvers() -> None:
    """Expose the three rules to Hydra as ``${hfrl.*}`` interpolations.

    Resolving them in the configuration, rather than computing them in code
    after the fact, is what lets ``--cfg job --resolve`` show the values a run
    will actually use — and therefore lets the tests compare a composed
    configuration against the ones that produced the thesis.
    """
    for nome, fn in (("hfrl.total_queries", total_queries),
                     ("hfrl.demo_budget", demo_budget),
                     ("hfrl.initial_queries", initial_queries)):
        OmegaConf.register_new_resolver(nome, fn, replace=True)


def check_protocol(cfg: DictConfig) -> None:
    """Refuse combinations the protocols were never meant to describe.

    ``thesis_b10`` raises the floor to five. That is meaningful only at B=10 and
    only for the two-channel methods; applying it elsewhere would quietly change
    how much feedback an arm collects up front.
    """
    floor = cfg.get("initial_queries_min", 1)
    if floor == 1:
        return
    if int(cfg.budget) != 10:
        raise ValueError(
            f"this protocol raises the initial-query floor to {floor}, which only "
            f"applies at budget=10; got budget={cfg.budget}."
        )
    if not (cfg.uses_preferences and cfg.uses_demonstrations):
        raise ValueError(
            f"this protocol raises the initial-query floor to {floor}, which only "
            f"applies to the two-channel arms; got arm={cfg.arm_name}."
        )
