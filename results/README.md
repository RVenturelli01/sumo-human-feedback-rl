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

At `B=10` the four arms with a weighted preference channel — `hybrid_soft`,
`hybrid_bern`, `unw_soft`, `unw_bern` — were run in a **separate campaign** with
five initial queries (`th_1mh4iq5_*`), because below `ALPHA_MIN_PREFS = 5` the
reliability weight cannot be estimated. The `th_1mh4_*_B10` groups for those four
arms are the superseded earlier runs and are still on W&B.

Aggregating without that selection silently produces the wrong numbers for the
two norm-balanced cells: 21.3 instead of 25.2 for NB-soft, and -25.8 instead of
-28.2 for NB-Bernoulli. The aggregation in `evaluate.py` encodes the selection.

## What `plots/` does instead

`plots/` reads Weights & Biases directly and therefore reports the 20-episode
evaluation logged during training. That is deliberate: it keeps the toolkit in
sync with what the runs actually logged. The 200-episode numbers live here.
