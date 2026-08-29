"""How one W&B run is read: Hydra config -> one index row.

Every run comes from the same entry point, so the config always has the same
nested shape, available only after `run.load(force=True)`: the list returned by
`api.runs()` gives `config == {}`.

The delicate part is that there is no "algorithm" field. It is derived from the
config:

    uses_pref = algo.kwargs.total_queries > 0
    uses_demo = algo.kwargs.demo_weight  > 0

Both true is hybrid, the first alone is preference-only, the second alone is
demonstration-only. Deriving it from the config rather than from the group name
is deliberate: group names changed over the campaigns, while the config is
always the one the run actually used.
"""
from __future__ import annotations

import re

GROUP_LEVEL_RE = re.compile(r"^budget_(?P<tag>[a-z0-9]+(?:_[a-z0-9]+)*)_(?P<level>\d+)$")
# The grad-diagnostics groups write the budget with a capital B at the end
# (`gd_p2_alpha_B100`) instead of the bare number the budget-curve campaign
# uses. Same meaning, different syntax: without this pattern `budget_level`
# would be empty for that whole project and the budget curves would be blank.
GD_LEVEL_RE = re.compile(r"^gd_(?P<tag>[a-z0-9]+(?:_[a-z0-9]+)*)_B(?P<level>\d+)$")
# The reference runs use the `th_` prefix: `th_hybrid_soft_B1000` becomes
# ("hybrid_soft", 1000). Same syntax as gd_, different prefix because it is a
# different campaign. Without this pattern those runs would have no
# budget_level and would vanish from the budget curves with no error.
TH_LEVEL_RE = re.compile(r"^th_(?P<tag>[a-z0-9]+(?:_[a-z0-9]+)*)_B(?P<level>\d+)$")

# `gcl_fusion` does not exist in runs made before the fusion schemes: there the
# code applied norm balancing, which is the constructor default.
DEFAULT_FUSION = "norm_balance"


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _bool(v):
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return None if v is None else bool(v)


def _positive(v) -> bool:
    n = _num(v)
    return n is not None and n > 0


def parse_group(group: str | None) -> tuple[str | None, float | None]:
    """`budget_hybrid_demo_2_bern_hom_5446` -> ("hybrid_demo_2_bern_hom", 5446.0).

    The trailing number is the budget level the campaign varies, queries or
    trajectories depending on the method; the rest is a free label, useful for
    telling apart variants that are not columns of their own.

    Two other prefixes carry the same level: `gd_<tag>_B<N>` and `th_<tag>_B<N>`.
    """
    for pattern in (GROUP_LEVEL_RE, GD_LEVEL_RE, TH_LEVEL_RE):
        m = pattern.match(group or "")
        if m:
            return m.group("tag"), float(m.group("level"))
    return None, None


def derive_arm(algo_kwargs: dict, run_cfg: dict) -> dict:
    """{arm, arm_family, demo_loss, pref_labels} from the Hydra config.

    The same rule the launcher uses to decide which method it is running, read
    backwards here from a run that already happened.
    """
    uses_pref = _positive(algo_kwargs.get("total_queries"))
    uses_demo = _positive(algo_kwargs.get("demo_weight"))

    demo_loss = algo_kwargs.get("loss_type") if uses_demo else None
    labels_type = algo_kwargs.get("labels_type")
    pref_labels = None
    if uses_pref:
        pref_labels = "bernoulli" if labels_type == "binary_bernoulli" else "soft"

    if uses_pref and uses_demo:
        arm_family = "hybrid"
        arm = f"hybrid_{demo_loss}_{pref_labels}" if demo_loss and pref_labels else None
    elif uses_pref:
        arm_family = "pref"
        arm = f"pref_{pref_labels}" if pref_labels else None
    elif uses_demo:
        arm_family = "demo"
        arm = demo_loss
    else:
        arm_family = None
        arm = None
    return dict(arm=arm, arm_family=arm_family, demo_loss=demo_loss, pref_labels=pref_labels)


def derive_fusion(algo_kwargs: dict, arm_family: str | None) -> str | None:
    """How the two channels are combined into one update.

    Kept out of `arm` on purpose. `arm` answers "which feedback sources does
    this run use", and is what makes two campaigns comparable; the fusion
    answers "how are they combined", and only means anything for the hybrid.
    """
    if arm_family != "hybrid":
        return None
    return algo_kwargs.get("gcl_fusion") or DEFAULT_FUSION


def row(run, project: str) -> dict:
    """One index row for one run."""
    cfg = run.config or {}
    run_cfg = dict(cfg.get("run") or {})
    algo_kwargs = dict((cfg.get("algo") or {}).get("kwargs") or {})
    train_kwargs = dict((cfg.get("train") or {}).get("kwargs") or {})
    reward_kwargs = dict(algo_kwargs.get("reward_model_kwargs") or {})
    tags = list(run.tags or [])

    arm_bits = derive_arm(algo_kwargs, run_cfg)
    group_tag, budget_level = parse_group(run.group)

    query_budget = _num(algo_kwargs.get("total_queries")) or None
    demo_budget = _num(run_cfg.get("n_expert_trajectories"))
    initial_queries = _num(algo_kwargs.get("initial_queries"))
    initial_queries_frac = (
        round(initial_queries / query_budget, 4)
        if initial_queries is not None and query_budget else None
    )
    reward_net_arch = reward_kwargs.get("net_arch")

    return dict(
        run_id=run.id,
        name=run.name,
        group=run.group,
        group_tag=group_tag,
        budget_level=budget_level,
        state=run.state,
        tags=",".join(tags),
        created_at=str(run.created_at),
        project=project,
        **arm_bits,
        fusion=derive_fusion(algo_kwargs, arm_bits["arm_family"]),
        demo_mode=algo_kwargs.get("demo_mode"),
        query_budget=query_budget,
        # None means the whole expert dataset, no subsampling. It is not a
        # missing value, it is "everything there is".
        demo_budget=demo_budget,
        demo_weight=_num(algo_kwargs.get("demo_weight")),
        initial_queries=initial_queries,
        initial_queries_frac=initial_queries_frac,
        query_schedule=algo_kwargs.get("query_schedule"),
        fragmenter_type=algo_kwargs.get("fragmenter_type"),
        pref_temperature=_num(algo_kwargs.get("pref_temperature")),
        # Missing in runs made before label smoothing existed, where the target
        # was the raw label, which is exactly eps = 0.
        label_smoothing=_num(algo_kwargs.get("label_smoothing")) or 0.0,
        normalize_agent_reward=_bool(algo_kwargs.get("normalize_agent_reward")),
        relabel_rewards=_bool(algo_kwargs.get("relabel_rewards")),
        reward_net_arch=str(reward_net_arch) if reward_net_arch else None,
        demo_subsample_seed=_num(run_cfg.get("demo_subsample_seed")),
        seed=_num(run_cfg.get("seed")),
        total_timesteps=_num(train_kwargs.get("total_timesteps")),
    )
