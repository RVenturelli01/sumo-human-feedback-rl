# Final results — 200 held-out episodes

These are the numbers reported in the thesis. They come from re-evaluating the
saved policies (`agent_final.zip`) on 200 episodes, **not** from the 20-episode
evaluation that runs at the end of training: with 20 episodes a single collision
moves the mean return by about 8 points, more than the differences between
methods.

| file | content |
|---|---|
| `results_200_episodes.csv` | one row per run — 7 methods x 3 budgets x 10 seeds |
| `results_200_episodes_summary.csv` | one row per method-budget cell |

## How to regenerate them

```bash
# 1. re-evaluate every checkpoint on 200 episodes (writes final_eval_200.json)
python experiments/evaluate.py outputs/thesis_runs/th_1mh4*/*/*

# 2. aggregate into the two tables
python experiments/evaluate.py --aggregate outputs/thesis_runs results
```

## The one trap

At `B=10` the four **two-channel** methods — the two reliability-weighted hybrids
(`hybrid_soft`, `hybrid_bern`) and the two norm-balanced ablations (`unw_soft`,
`unw_bern`) — were run in a **separate campaign** with five initial queries
(`th_1mh4iq5_*`). The `th_1mh4_*_B10` groups for those four arms are the
superseded earlier runs and are still on W&B.

The reason differs between the two pairs. The hybrids need five comparisons
because below `ALPHA_MIN_PREFS = 5` the reliability weight cannot be estimated at
all. The norm-balanced ablations do not estimate a reliability weight, so they
have no such requirement: they were given five initial queries so that both
variants collect feedback on the same schedule, which is what makes the ablation
a comparison of the combination rule alone.

Aggregating without that selection silently produces the wrong numbers for the
two norm-balanced cells: 21.3 instead of 25.2 for NB-soft, and -25.8 instead of
-28.2 for NB-Bernoulli. The aggregation in `evaluate.py` encodes the selection.

## `other_rate`

The four outcome rates do not always sum to one: the environment can also report
`teleported` or `removed_unknown`, which `evaluate()` does not record separately.
`other_rate` carries the remainder so that a sum below one reads as what it is
rather than as an aggregation bug. It is non-zero in exactly one run out of 250 —
`th_1mh4_pref_soft_B100-seed4`, 2 episodes out of 200.

## What `plots/` does instead

`plots/` reads Weights & Biases directly and therefore reports the 20-episode
evaluation logged during training. That is deliberate: it keeps the toolkit in
sync with what the runs actually logged. The 200-episode numbers live here.
