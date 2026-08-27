# Hybrid reward learning from demonstrations and preferences

Code for the thesis *Reward learning from preferences and demonstrations via
gradient reliability* (Politecnico di Milano). An agent drives an ego vehicle on
a three-lane highway in SUMO, and the reward it optimises is learned from two
sources at once: expert demonstrations and pairwise preferences over trajectory
fragments.

The method combines the two gradients with a reliability weight `α`, estimated at
every iteration from the dispersion of each channel. When the preference channel
is noisy, `α` moves the weight onto the demonstrations; when comparisons become
informative, it moves it back. The baselines are not separate implementations:
they are the same algorithm with a single channel active.

## Layout

```
experiments/          everything needed to run an experiment
  train.py              the entry point: one run
  evaluate.py           re-evaluation on more episodes, and the result tables
  download_datasets.py  fetches the demonstrations
  configs/
    train.yaml            base configuration
    arm/                  one file per method — hyperparameters only
    protocol/             the campaign settings shared by every method
  utils/
    common.py             seeding, W&B, data loading, evaluation
    budget.py             how the budget B becomes the numbers the algorithm needs
tests/                configuration equivalence with the thesis runs
results/              the numbers reported in the thesis
plots/                the toolkit that drew the figures (reads W&B)
notebooks/            how the datasets were built
human-feedback-rl/    submodule: the algorithm
sumo-rl-ego/          submodule: the environment
```

There is no scheduler and no launcher. An experiment is a configuration, and a
grid is `--multirun`.

## Setup

```bash
git clone --recurse-submodules <repo> && cd sumo-human-feedback-rl
./setup.sh
conda activate sumo-rlhf
```

`setup.sh` creates the environment, installs the two submodules in editable mode,
checks SUMO, downloads the demonstrations and verifies their checksums. It cannot
activate the environment for you — that only works in the shell you are sitting
in, which is why activation is a separate line.

The demonstrations live in a **private** Hugging Face repository. Without access
the download stops with a clear message; ask for access to
`Andrea02/sumo-rlhf-datasets` before starting.

## Running one experiment

```bash
python experiments/train.py arm=hybrid_soft protocol=thesis budget=1000 run.seed=3
```

Three things identify a run, and everything else follows from them:

| | |
|---|---|
| `arm` | the method — one of the seven files under `configs/arm/` |
| `protocol` | the campaign settings — `thesis`, or `thesis_b10` |
| `budget` | the feedback budget `B` |

From those, the configuration derives how many comparisons the arm may ask for,
how many it collects up front, how many demonstrations it reads, the W&B group,
the run name and the output directory. Nothing has to be passed by hand, which is
what used to go wrong.

To see the resolved configuration without running anything:

```bash
python experiments/train.py arm=hybrid_soft protocol=thesis budget=1000 --cfg job --resolve
```

## Reproducing the thesis

Seven methods × three budgets × ten seeds = 210 runs of 1M environment steps.

| method | channels |
|---|---|
| `demo_only` | demonstrations |
| `pref_soft`, `pref_bern` | preferences, soft or Bernoulli labels |
| `hybrid_soft`, `hybrid_bern` | both, combined by the reliability weight |
| `unw_soft`, `unw_bern` | both, combined by norm balancing — the ablation |

```bash
# B=100 and B=1000, every method
python experiments/train.py --multirun \
  arm=demo_only,pref_soft,pref_bern,hybrid_soft,hybrid_bern,unw_soft,unw_bern \
  protocol=thesis budget=100,1000 run.seed=1,2,3,4,5,6,7,8,9,10
```

```bash
# B=10: the two-channel methods use the variant with five initial comparisons
python experiments/train.py --multirun \
  arm=hybrid_soft,hybrid_bern,unw_soft,unw_bern \
  protocol=thesis_b10 budget=10 run.seed=1,2,3,4,5,6,7,8,9,10
```

```bash
# B=10: the single-channel baselines keep their usual share
python experiments/train.py --multirun \
  arm=demo_only,pref_soft,pref_bern \
  protocol=thesis budget=10 run.seed=1,2,3,4,5,6,7,8,9,10
```

Hydra's `--multirun` runs these in sequence. Parallelism is yours to arrange —
`taskset -c <core>` around a single-run command is what the original campaigns
used, one core per run.

### Why B=10 is split in two

Below `ALPHA_MIN_PREFS = 5` comparisons the reliability weight cannot be
estimated at all. At `B=10` the usual 10% share would give one, so the
reliability-weighted hybrids collect five up front instead. The norm-balanced
ablations do not estimate a weight and have no such need, but they use the same
floor so that both variants collect feedback on the same schedule — which is what
makes the ablation a comparison of the combination rule alone.

`protocol=thesis_b10` refuses any other budget, and refuses single-channel arms.

### Demonstrations are the same across methods

Each run subsamples `B` demonstrations with a fixed seed
(`run.demo_subsample_seed=1000`, deliberately independent of the training seed),
so every method at a given budget sees the same ones and the subsets are nested
across budgets. Each run prints a fingerprint of what it loaded:

| budget | trajectories | transitions | fingerprint |
|--------|--------------|-------------|-------------|
| 10     | 10           | 1902        | `02641cfbeb14` |
| 100    | 100          | 18977       | `132b848ddafc` |
| 1000   | 1000         | 179079      | `c3705912f3e5` |

A different fingerprint means a different dataset, and the numbers will not match.

## Evaluation

Training ends with a 20-episode evaluation, which is enough to follow a run but
too coarse for the tables: one collision moves the mean by about 8 points, more
than the differences between methods. The reported numbers come from 200
episodes, re-run from the saved policies:

```bash
python experiments/evaluate.py outputs/thesis_runs/th_1mh4*/*/*
python experiments/evaluate.py --aggregate outputs/thesis_runs results
```

The first writes `final_eval_200.json` next to each checkpoint, skipping runs
that already have one. The second builds the two tables in `results/`, and
refuses to write if the 7 × 3 × 10 grid has holes — a cell built from nine seeds
looks exactly like one built from ten.

Evaluation uses episode seeds `seed + i` from a fixed base, so every method faces
the same 200 scenarios, whatever its training seed.

## Figures

```bash
pip install -r plots/requirements.txt
python plots/scripts/build_index.py     # indexes the W&B projects
python plots/scripts/selector.py        # interactive selector on :8770
```

`plots/` reads Weights & Biases and therefore reports the 20-episode evaluation
logged during training. That is deliberate: it keeps the toolkit in sync with
what the runs actually recorded. The 200-episode numbers are in `results/`.

## Checking that the reorganisation preserved the experiments

```bash
pytest tests/
```

`tests/fixtures/thesis_resolved_configs/` holds the 21 configurations that
produced the thesis, taken from the code that ran them and cross-checked against
the configurations Weights & Biases recorded for the runs themselves. The tests
compare what this repository composes today against them, key by key. If one
fails, an experiment changed.

## History

This repository was reorganised after the experiments were finished. The original
execution layer — a launcher, a core scheduler and thirty-odd campaign scripts —
is preserved at the tag **`pre-reorganization`** (commit `e1c8bbf`, also the head
of the `thesis-protocol` branch), together with the README describing how it was
used. The submodule commit that produced the results carries the tag
**`thesis-experiments`**.

Nothing was archived into a folder: git is the archive.
