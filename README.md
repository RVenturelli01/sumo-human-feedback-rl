# Hybrid reward learning from demonstrations and preferences

Code for the thesis *Hybrid reward learning from demonstrations and preferences*
(Politecnico di Milano). An agent drives an ego vehicle on a three-lane highway
in SUMO, and the reward it optimises is learned from two sources at once: expert
demonstrations and pairwise preferences over trajectory fragments.

The method combines the two gradients with a reliability weight `α`, estimated at
every iteration from the dispersion of each channel. When the preference channel
is noisy, `α` moves the weight onto the demonstrations; when comparisons become
informative, it moves it back. The baselines are the same algorithm with a single
channel active.

## Repository layout

```
scripts/            entry points: training, launchers, evaluation helpers
  train_hybrid_sac.py     one run (Hydra config + overrides)
  launch_thesis_runs.py   arms, shared protocol, per-run validation
  queue_runs.py           scheduler: launches runs as cores free up
configs/            Hydra configs (train_hybrid_sac.yaml is the entry config)
human-feedback-rl/  submodule: the reward-learning algorithms
sumo-rl-ego/        submodule: the SUMO environment
datasets/           expert demonstrations (not versioned, see below)
plots/              plotting toolkit used for the thesis figures
```

## Reproducing the thesis experiments

Everything reported in the Experimental Evaluation section comes from two
campaigns, `th_1mh4` and `th_1mh4iq5`, run with the code on this branch. The
commands below are the ones that produced them.

### 1. Code version

```bash
git clone https://github.com/RVenturelli01/sumo-human-feedback-rl.git
cd sumo-human-feedback-rl
git submodule update --init --recursive
```

The submodule pointer is what matters: `human-feedback-rl` must sit at
`c5cd63e`, which is also tagged `thesis-experiments` in that repository. Every
run in the thesis was produced by that commit. `git submodule update` puts it
there automatically; to check:

```bash
git submodule status human-feedback-rl
# c5cd63ee664633a924156c353f80e676e5331cfb human-feedback-rl (thesis-experiments)
```

The tag exists so the commit stays reachable even if the branch it was developed
on is ever removed. If you see a different SHA, the runs will not match.

### 2. Environment

The runs used Python 3.12 with SUMO/libsumo 1.27.1, torch 2.13.0 (CPU),
numpy 2.5.2, gymnasium 1.3.0 and stable-baselines3 2.9.0. Training is CPU-only:
each run is pinned to a single core with `taskset`.

```bash
conda env create -f environment.yml
conda activate sumo-rlhf
pip install -e human-feedback-rl -e sumo-rl-ego
```

SUMO must be installed separately and `SUMO_HOME` set; `libsumo` is imported
directly, so a working SUMO installation is a hard requirement.

Results are logged to Weights & Biases, so `wandb login` is needed before
launching. The project and entity are set in `scripts/launch_thesis_runs.py`
(`WANDB_ENTITY`, `WANDB_PROJECT`) — change them to your own account.

### 3. Demonstrations

The expert dataset is not in the repository (`*.pkl` is ignored). You need:

```
datasets/expert_trajectories_no_collision.pkl
```

2723 collision-free expert trajectories. Each run subsamples `B` of them with a
fixed seed (`run.demo_subsample_seed=1000`, independent of the training seed), so
all methods at a given budget see the same demonstrations and the subsets are
nested across budgets. The run prints a fingerprint of the subset it loaded:

| budget | trajectories | transitions | fingerprint |
|--------|--------------|-------------|-------------|
| 10     | 10           | 1902        | `02641cfbeb14` |
| 100    | 100          | 18977       | `132b848ddafc` |
| 1000   | 1000         | 179079      | `c3705912f3e5` |

If your fingerprints differ, you have a different dataset and the numbers will
not match.

### 4. Protocol

All arms share one protocol, defined in `PROTOCOL` in
`scripts/launch_thesis_runs.py`. The thesis campaigns override part of it on the
command line:

| parameter | value |
|-----------|-------|
| total environment steps | 1,000,000 (50 iterations × 20,000) |
| parallel environments | 4 (`SubprocVecEnv`) |
| SAC `train_freq` | 4 — with `gradient_steps=32` the replay ratio is 2.0 |
| rollout collection | shared with training |
| reward-model ensemble | 1 member, no bootstrap of comparisons |
| query schedule | hyperbolic |
| evaluation | 200 held-out episodes, deterministic policy, fixed seed 1000 |
| seeds | 1–10 |

`validate()` resolves the config with Hydra before anything is launched and
refuses to start if the replay ratio is not 2.0 or if any protocol key does not
land as expected. Per-arm hyperparameters (learning rate, network size, gradient
steps, warm-up) are written explicitly in `ARMS` in the same file.

### 5. Commands

`queue_runs.py` takes the arms and holds the rest of the runs in a queue,
launching each one on a free core as soon as one is available. `--first-core 16`
is specific to the machine used (cores 0–15 were reserved for other users);
adjust it to your hardware.

Run these one at a time, not in parallel: two schedulers read the state of the
cores at the same instant and can assign the same core twice.

**Demonstration-only baseline** (30 runs):

```bash
python scripts/queue_runs.py \
  --arms demo_only --seeds 1 2 3 4 5 6 7 8 9 10 \
  --campaign 1mh4 \
  --n-envs 4 --train-freq 4 --query-schedule hyperbolic \
  --total-timesteps 1000000 --n-ensembles 1 --bootstrap false \
  --first-core 16
```

**Hybrid methods at B=100 and B=1000** (40 runs):

```bash
python scripts/queue_runs.py \
  --arms hybrid_soft hybrid_bern --budgets 100 1000 --seeds 1 2 3 4 5 6 7 8 9 10 \
  --campaign 1mh4 \
  --n-envs 4 --train-freq 4 --query-schedule hyperbolic \
  --total-timesteps 1000000 --n-ensembles 1 --bootstrap false \
  --set-arm-field 'hybrid_soft.net_arch=[64,64]' \
  --set-arm-field hybrid_bern.gradient_steps_rew=78 \
  --set-arm-field hybrid_bern.initial_agent_timesteps=40000 \
  --first-core 16
```

**Hybrid methods at B=10** (20 runs). Separate campaign because the reliability
weight needs at least five comparisons before it can be estimated: with the
default share, `B=10` would start with one. Five initial queries make `α`
available from the first reward-model update.

```bash
python scripts/queue_runs.py \
  --arms hybrid_soft hybrid_bern --budgets 10 --seeds 1 2 3 4 5 6 7 8 9 10 \
  --campaign 1mh4iq5 \
  --n-envs 4 --train-freq 4 --query-schedule hyperbolic \
  --total-timesteps 1000000 --n-ensembles 1 --bootstrap false \
  --initial-queries 5 \
  --set-arm-field 'hybrid_soft.net_arch=[64,64]' \
  --set-arm-field hybrid_bern.gradient_steps_rew=78 \
  --set-arm-field hybrid_bern.initial_agent_timesteps=40000 \
  --first-core 16
```

**Preference-only baselines and norm-balanced ablation** (120 runs):

```bash
python scripts/queue_runs.py \
  --arms pref_soft pref_bern unw_soft unw_bern --seeds 1 2 3 4 5 6 7 8 9 10 \
  --campaign 1mh4 \
  --n-envs 4 --train-freq 4 --query-schedule hyperbolic \
  --total-timesteps 1000000 --n-ensembles 1 --bootstrap false \
  --set-arm-field pref_soft.initial_agent_timesteps=40000 \
  --set-arm-field 'unw_soft.net_arch=[64,64]' \
  --set-arm-field unw_bern.gradient_steps_rew=78 \
  --set-arm-field unw_bern.initial_agent_timesteps=40000 \
  --first-core 16
```

The three `--set-arm-field` lines on `unw_*` are not cosmetic. `unw_soft` and
`unw_bern` are the ablation of the weighting: they must differ from
`hybrid_soft` and `hybrid_bern` **only** in how the two gradients are combined.
Without them the ablation would also change the network size and the number of
gradient steps, and would no longer isolate the combination rule.

Each command runs 20–120 policies of 1M steps each. On 48 cores the full set
takes a few days.

### 6. Checking that a run started correctly

The scheduler prints one line per launch:

```
[10:26:39] lanciata th_1mh4_demo_only_B10-seed1 pid=1775121 core=60
```

and refuses to start if `validate()` fails, so a wrong parameter surfaces before
any compute is spent, not hours later.

In the run's own log (`outputs/thesis_runs/logs/<run-name>.log`) the first
things worth checking are the demonstration fingerprint and the environment
mode:

```
Loaded 1000 expert trajectories (179079 transitions), subsample seed 1000, fingerprint c3705912f3e5
Creating environment...
Rollout env: condiviso col training
- Collecting 20000 bootstrap transitions
- Collecting 20000 agent + 0 exploration transitions
```

If the fingerprint matches the table above and the rollout env is shared, the run
is on the right protocol. From there the log prints one `- Collecting 20000
agent` block per iteration, 50 in total.

At the end each run writes `final_eval.json` in its output directory, with the
return and the success, collision, off-road and timeout rates.

### 7. Evaluating on 200 episodes

The thesis reports the return over 200 held-out episodes. Runs write a 20-episode
evaluation at the end of training, which is enough to follow a run but too coarse
for the final tables: with 20 episodes one collision moves the mean return by
about 8 points, more than the differences between the methods.

`scripts/evaluate_checkpoints.py` re-runs the evaluation from `agent_final.zip`,
without retraining. It calls the same `evaluate()` used at the end of training,
so the procedure is identical: one environment, deterministic policy, episode
seeds `seed + i`. With the default seed the first 20 episodes are exactly the
ones already evaluated, so a longer run extends the series instead of replacing
it — and all methods face the same 200 scenarios.

```bash
# every run of the thesis campaigns, 44 at a time
find outputs/thesis_runs/th_1mh4* -name agent_final.zip -exec dirname {} \; \
  | xargs -P 44 -I{} python scripts/evaluate_checkpoints.py {}
```

It writes `final_eval_200.json` next to each checkpoint and skips runs that
already have one, so it can be interrupted and restarted. One run takes about 20
seconds on a single core.

The evaluation is deterministic within one environment but not across
environments: re-running it on a machine with a different SUMO or torch version
gives returns that agree to about 0.01%, not to the last digit.

The per-run and aggregated results used in the thesis are in
`results_200_episodes.csv` and `results_200_episodes_summary.csv` in the thesis
folder.

## Figures

`plots/` is the toolkit used for the thesis figures. It indexes the W&B runs and
exports each panel as a standalone `pgfplots` source, so the figures in the
thesis contain the real data and are recompiled by LaTeX rather than included as
images. It has its own dependencies:

```bash
pip install -r plots/requirements.txt
python plots/scripts/build_index.py     # indexes the W&B projects
python plots/scripts/selector.py        # interactive selector on :8770
```

`plots/` is independent of the training code — nothing under `scripts/` or
`human-feedback-rl/` imports it — so changing it cannot affect the experiments.
See `plots/README.md` for details.

## Notes

`scripts/` also contains the tuning and earlier campaign launchers used during
development. They are kept because the hyperparameters in `ARMS` come from those
searches, but they are not needed to reproduce the results above.
