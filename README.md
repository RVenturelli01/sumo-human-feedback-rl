# Reward learning from demonstrations and preferences

An agent drives an ego vehicle on a three-lane highway in SUMO. The reward it
optimises is not given: it is learned from two sources at once, expert
demonstrations and pairwise preferences over trajectory fragments.

The two gradients are combined with a reliability weight `α`, estimated at every
iteration from the dispersion of each channel. When comparisons are noisy, `α`
shifts weight onto the demonstrations; when they become informative, it shifts
back. The baselines are not separate code: they are the same algorithm with one
channel switched off.

## Layout

```
runner/               everything needed to run something
  train.py              the entry point: one run
  evaluate.py           re-evaluation on more episodes, and the result tables
  download_datasets.py  fetches the demonstrations
  configs/
    train.yaml            defaults
    arm/                  one file per method — hyperparameters only
    protocol/             the settings every method shares
  utils/
    common.py             seeding, W&B, data loading, evaluation
    budget.py             how the budget becomes the numbers the algorithm needs
tests/                configuration equivalence with the runs that produced the results
plots/                figures, read from Weights & Biases
human-feedback-rl/    submodule: the algorithm
sumo-rl-ego/          submodule: the environment
```

## Setup

```bash
git clone --recurse-submodules <repo> && cd sumo-human-feedback-rl
./setup.sh
conda activate sumo-rlhf
```

`setup.sh` creates the environment, installs both submodules in editable mode,
checks SUMO, downloads the demonstrations and verifies their checksums. If an
environment of that name already exists it checks every pin in `environment.yml`
against it and stops on any mismatch, rather than using it: an environment with
the right name and the wrong contents imports fine and quietly changes the
numbers.

Activation is a separate line because a script cannot change the shell that
called it.

## Running one experiment

```bash
python runner/train.py arm=hybrid_soft budget=1000 run.seed=3
```

Three things identify a run:

| | |
|---|---|
| `arm` | the method — one of the seven files in `configs/arm/` |
| `budget` | how much feedback it gets |
| `run.seed` | |

Everything else follows: how many comparisons the arm may ask for, how many it
collects up front, how many demonstrations it reads, the W&B group, the run name
and the output directory. Nothing has to be passed by hand.

To see the resolved configuration without running anything, add
`--cfg job --resolve`.

## Reproducing the reference results

Seven methods × three budgets × ten seeds, 1M environment steps each.

| method | channels |
|---|---|
| `demo_only` | demonstrations |
| `pref_soft`, `pref_bern` | preferences, soft or Bernoulli labels |
| `hybrid_soft`, `hybrid_bern` | both, combined by the reliability weight |
| `unw_soft`, `unw_bern` | both, combined by norm balancing — the ablation |

```bash
python runner/train.py --multirun \
  arm=demo_only,pref_soft,pref_bern,hybrid_soft,hybrid_bern,unw_soft,unw_bern \
  budget=10,100,1000 run.seed=1,2,3,4,5,6,7,8,9,10
```

`--multirun` runs these in sequence. Parallelism is yours to arrange: the
original campaigns wrapped a single-run command in `taskset -c <core>`, one core
per run.

### The fusion ablations

Two further arms combine both channels without estimating the weight. They are
not part of the reference grid above; `evaluate.py` recognises them and reports
them, but never asks for them.

```bash
python runner/train.py --multirun arm=unwh_soft,unwh_bern \
  budget=10,100,1000 run.seed=1,2,3,4,5,6,7,8,9,10 eval.n_episodes=200
```

`unwh_*` pins the weight at one half instead of estimating it. The other control
needs no arm of its own, only the weight decay switched off:

```bash
python runner/train.py --multirun arm=unw_soft,unw_bern \
  budget=10,100,1000 run.seed=1,2,3,4,5,6,7,8,9,10 \
  algo.kwargs.l2_rew=0 campaign=nowd eval.n_episodes=200
```

Give that one its own `campaign`, as above, and keep its output directory apart
from the reference runs. Its directory names still parse as `unw_soft` and
`unw_bern`, so an aggregate over a directory holding both would read them as
twenty seeds of one cell instead of two sets of ten.

`eval.n_episodes=200` evaluates at the end of training, which saves the
`evaluate.py` pass: the two routes were checked against each other on the whole
reference grid and agree on every value.

### Initial comparisons

Each arm collects a share of its budget before the regular schedule starts: none
for `demo_only`, 5% and 20% for the two preference-only baselines, 10% for the
four two-channel methods. Below five comparisons the reliability weight cannot be
estimated at all, so the two-channel methods have a floor of five. It only
matters at `budget=10`, where 10% would give one.

The norm-balanced ablations do not estimate a weight and would not need the
floor, but they use it anyway so that both variants collect feedback on the same
schedule — which is what makes the ablation a comparison of the combination rule
alone.

### Demonstrations are shared across methods

Each run subsamples `B` demonstrations with a fixed seed, deliberately
independent of the training seed, so every method at a given budget sees the same
ones and the subsets are nested across budgets. Each run prints a fingerprint of
what it loaded:

| budget | trajectories | transitions | fingerprint |
|--------|--------------|-------------|-------------|
| 10     | 10           | 1902        | `02641cfbeb14` |
| 100    | 100          | 18977       | `132b848ddafc` |
| 1000   | 1000         | 179079      | `c3705912f3e5` |

A different fingerprint means a different dataset, and the numbers will not
match.

## Evaluation

Training ends with a 20-episode evaluation. That is enough to follow a run and
too coarse for a table: one collision moves the mean by about 8 points, more than
the differences between methods. The reported numbers come from 200 episodes,
re-run from the saved policies:

```bash
python runner/evaluate.py outputs/runs/*/*
python runner/evaluate.py --aggregate outputs/runs results
```

The first writes `final_eval_200.json` next to each checkpoint and skips runs
that already have one. The second builds two CSVs, and refuses to write if the
7 × 3 × 10 grid has holes — a cell built from nine seeds looks exactly like one
built from ten.

Evaluation seeds are `base + i` from a fixed base, so every method faces the same
200 scenarios whatever its training seed.

Training can also do the 200 episodes itself, which saves the first command:

```bash
python runner/train.py arm=hybrid_soft budget=1000 run.seed=3 eval.n_episodes=200
```

It writes the same numbers to `final_eval.json`. Checked on the whole grid: the
policy evaluated in memory at the end of training and the same policy reloaded
from `agent_final.zip` agree on every value. The aggregate step still reads
`final_eval_200.json`, so that route needs `evaluate.py` either way.

## Figures

```bash
pip install -r plots/requirements.txt
python plots/scripts/build_index.py     # indexes the W&B projects
python plots/scripts/selector.py        # interactive selector on :8770
```

`plots/` reads Weights & Biases, so it shows the 20-episode evaluation logged
during training. The 200-episode numbers come from `evaluate.py --aggregate`.
