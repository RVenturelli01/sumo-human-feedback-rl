"""Come si legge una run W&B di questa campagna: config Hydra -> riga dell'indice.

Un solo file (a differenza del progetto di ispirazione, che aveva una `RunSource`
per convenzione perche' univa piu' campagne eterogenee): qui tutte le run vengono
da un solo entry point (`scripts/train_hybrid_sac.py`), quindi la config e' sempre
la stessa struttura annidata (`configs/train_hybrid_sac.yaml`: run/env/agent/algo/
train/eval), disponibile solo dopo `run.load(force=True)` (la lista di
`api.runs()` restituisce `config == {}`, verificato empiricamente su questo
progetto: e' la stessa trappola documentata nel progetto di ispirazione per le
run "paper").

Il punto delicato e' che qui non esiste un campo "algoritmo": si deriva da
quattro chiavi, esattamente come fa `scripts/tune_hybrid_sac.py` per decidere
quale arm sta girando:
  - `uses_pref`  = algo.kwargs.total_queries > 0
  - `uses_demo`  = algo.kwargs.demo_weight > 0
  entrambe vere = hybrid, solo la prima = pref-only, solo la seconda = demo-only.
Derivarlo dalla config (non dal nome del gruppo) e' deliberato: il nome dei
gruppi budget_* e' cambiato nel corso della campagna (sono comparse varianti
come `_no_norm_`, `_bern_hom_`, `_soft_trmatch_` non coperte da
`scripts/_report_common.py:base_arm()`, che le colassa silenziosamente
nell'arm base) — la config invece e' sempre corretta perche' e' quella con cui
la run e' stata davvero lanciata.
"""
from __future__ import annotations

import re

GROUP_LEVEL_RE = re.compile(r"^budget_(?P<tag>[a-z0-9]+(?:_[a-z0-9]+)*)_(?P<level>\d+)$")


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

    Il numero finale e' il livello di budget che quella campagna sta facendo
    variare (query o traiettorie a seconda del braccio); il resto e' un'etichetta
    libera, utile per distinguere varianti non ancora modellate come colonne
    proprie senza doverle indovinare.
    """
    m = GROUP_LEVEL_RE.match(group or "")
    if not m:
        return None, None
    return m.group("tag"), float(m.group("level"))


def derive_arm(algo_kwargs: dict, run_cfg: dict) -> dict:
    """{arm, arm_family, demo_loss, pref_labels} dalla config Hydra.

    Stessa logica di `uses_preferences`/`uses_demos` in
    `scripts/tune_hybrid_sac.py`: e' la definizione ufficiale di cosa distingue
    un braccio dall'altro, qui solo letta all'indietro da una run gia' fatta
    invece che decisa prima di lanciarla.
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


def row(run, project: str) -> dict:
    """Riga dell'indice per una run di questo progetto."""
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
        demo_mode=algo_kwargs.get("demo_mode"),
        query_budget=query_budget,
        # None = intero dataset esperto (nessun sottocampionamento): non e' un
        # valore mancante, e' "tutto quello che c'e'".
        demo_budget=demo_budget,
        demo_weight=_num(algo_kwargs.get("demo_weight")),
        initial_queries=initial_queries,
        initial_queries_frac=initial_queries_frac,
        query_schedule=algo_kwargs.get("query_schedule"),
        fragmenter_type=algo_kwargs.get("fragmenter_type"),
        pref_temperature=_num(algo_kwargs.get("pref_temperature")),
        normalize_agent_reward=_bool(algo_kwargs.get("normalize_agent_reward")),
        relabel_rewards=_bool(algo_kwargs.get("relabel_rewards")),
        reward_net_arch=str(reward_net_arch) if reward_net_arch else None,
        demo_subsample_seed=_num(run_cfg.get("demo_subsample_seed")),
        seed=_num(run_cfg.get("seed")),
        total_timesteps=_num(train_kwargs.get("total_timesteps")),
    )
