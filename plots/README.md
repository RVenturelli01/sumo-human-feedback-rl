# plots/ — an interactive plotting engine for the W&B runs

A toolkit for exploring and plotting the runs recorded on Weights & Biases. It
indexes several projects together, because comparing hybrid against
preference-only and demonstration-only means reading from more than one. The
`project` column is a filter, so one campaign can be isolated with a click.

## What an "arm" is here

There is no "algorithm" field in the config. It is derived from two keys, the
same rule the launcher uses to decide what it is running:

* `algo.kwargs.total_queries > 0` → uses preferences
* `algo.kwargs.demo_weight  > 0` → uses demonstrations

Both true is **hybrid**, the first alone is **pref**, the second alone is
**demo**.

**`arm` is not enough for the gradient diagnostics.** There the methods differ
in *how* they combine the two gradients, not in which sources they use, so they
all share one `arm`. That is what the **`fusion`** column is for, taken from
`algo.kwargs.gcl_fusion` and defaulting to `norm_balance` when the key is
absent, because that is the constructor default. It sits outside `arm` on
purpose: `arm` answers "which sources", `fusion` answers "how are they
combined", and keeping them apart leaves `arm` comparable across projects.

Deriving all of this from the config rather than from the W&B group name is
deliberate. Group names picked up unplanned suffixes over the campaigns, and a
name-based rule collapses those into the base method; reading the config says
what the run actually did.

## Two kinds of figure, two data pipelines

| | learning curve | budget curve |
|---|---|---|
| script | `plot_curves.py` | `plot_budget.py` |
| one run is worth | a time series | a single number |
| source | the W&B history | `run.summary` (`sweep/*`) |
| x axis | time, timesteps or iterations | budget level, **log scale** |
| aggregation | mean and band over seeds, on a shared grid | mean and error bars, one point per level |
| module | `rtplots/curves.py` | `rtplots/budget.py` |

Everything comes from W&B: the files written during training stay on the machine
that ran it. Each run not yet cached costs one network request, and results are
kept in `plots/.cache/`, so only the first time is slow.

### Coverage exclusions

The checkboxes in the **Coverage** table remove single runs from the figure, and
the exclusions **survive a change of filters**: they are stored by `run_id`, so
they stay identifiable whatever filter is applied. Runs that appear when a
filter is widened arrive selected; the ones you removed stay removed. The
*"excluded: N"* badge says how many, and *"include all"* clears them.

### Formulas beside the figure

To the right of the preview the selector shows a **Definitions** panel with the
formula for the chosen metric and the equations of the fusion schemes present in
the selection. The **formulas** checkbox shows and hides it, and decides whether
the exported image carries the panel: figure and definitions land side by side
in one file. The `.tex` export always leaves them out, since there formulas
belong as macros rather than as an image.

The definitions live in [`rtplots/formulas.py`](rtplots/formulas.py). The alpha
formulas are transcribed from the algorithm's `alpha_estimation.py`: if a
formula and the code disagree, the formula here is the wrong one. The
frozen-probe metrics come from an earlier diagnostic branch that is no longer
part of the package; the runs that logged them are still on W&B, and this is
what their axes mean. A test checks that every metric in those groups, and every
fusion still implemented, has an entry.

Rendering goes through **matplotlib mathtext**, not MathJax, so the page needs no
external dependencies. Mathtext covers a subset of LaTeX, with no environments
and no `\text{}`, so each line is a formula on its own.

### `--compare-fusion` and `--compare-norm`, budget curves only

In budget curves the default is **one method, one curve**, deliberately: each
budget level runs with the best config for that level, so almost every
hyperparameter covaries with the level and `auto_hue` would keep it as a real
dimension. But `fusion` is not a proxy for the level: left unsaid, several
schemes of one method end up **averaged into the same curve**. The same goes for
`normalize_agent_reward`, where ON and OFF would share a curve.

The two flags, and the matching checkboxes in the selector, add those columns to
the series identity. With them off — the default — nothing changes, but if one
of the two is about to be averaged a warning names the flag that separates it.

Learning curves need no flag: `auto_hue` splits on `fusion` by itself.

### Series that would come out identical

A `style.toml` rule applies the **first match**, so it cannot express "colour by
method, dash by ablation". Comparing two configurations of one method would
colour both the same and the curves would be indistinguishable.

`figure.py:_decollide` steps in afterwards: the **colour** stays what the rule
says, and the **line style** separates the series that would otherwise coincide.
It applies in every mode and to any dimension.

Inside a colliding group the baseline configuration keeps the solid line:
between two booleans the **disabled** one comes first, so the ablation is always
the dashed curve and the convention does not move between figures. The legend
instead lists booleans with `yes` first — two different orderings, on purpose.

## Layout

```
plots/
├── rtplots/             the library
│   ├── schema.py        index fields: titles, filters, role in a figure
│   ├── source.py        one W&B run -> one index row
│   ├── index.py         run metadata, cached as parquet
│   ├── metrics.py       the metric catalogue, each with its x axis
│   ├── curves.py        learning curves, aggregated over seeds
│   ├── budget.py        final evaluation per budget level, and the 90% rule
│   ├── select.py        filters and coverage counts
│   ├── figure.py        FigureSpec, and the one pipeline: selection -> figure
│   ├── grid.py          drawing the panel grid
│   ├── tikz.py          .tex export (pgfplots), one file per panel
│   ├── selection.py     saved selections
│   ├── labels.py        series names
│   ├── rules.py         reads style.toml
│   ├── style.py         turns the rules into rcParams and colours
│   └── webui/           the selector: api.py the logic, server.py the transport
├── scripts/             the executables
│   ├── build_index.py   build or refresh the metadata cache
│   ├── selector.py      the interactive selector, on a local server
│   ├── list_runs.py     which runs and seeds exist per combination
│   ├── plot_curves.py   learning curves
│   ├── plot_budget.py   budget curves
│   └── prefetch_curves.py  fill the cache before opening the selector
├── style.toml           the plotting rules, edited by hand
├── selector/            the selector page: html, css, js, no dependencies
├── tests/               tests that need no network
├── requirements.txt     what this needs beyond the rest of the repository
└── output/              generated figures (git ignores it)
```

**One pipeline.** The command line and the selector build the same object,
`FigureSpec`, and hand it to `rtplots.figure`. The same selection gives the same
figure either way.

## Getting started

```bash
pip install -r plots/requirements.txt
python plots/scripts/build_index.py      # first time, or after new runs
python plots/scripts/selector.py         # http://127.0.0.1:8770
```

The page offers a filter per dimension, a switch between learning and budget
curves, a "what to plot" dropdown, rows, columns and colours, a live preview, a
coverage table, selections you can name and save, and export to JPEG or LaTeX.

## Filters

Every script takes `--filter key=value …`, all applied together, with the same
syntax as the selector chips (`arm!=demo_1,demo_2`, `query_budget>=5000`).

| column | what it filters |
|---|---|
| `arm` | the single-source methods plus the hybrid combinations |
| `arm_family` | `demo` \| `pref` \| `hybrid` |
| `demo_loss` | `demo_1` \| `demo_2` |
| `pref_labels` | `soft` \| `bernoulli` |
| `demo_mode` | `gcl` \| `preferences` |
| `query_budget` | `algo.kwargs.total_queries` |
| `demo_budget` | `run.n_expert_trajectories`; missing means the whole dataset, not an unknown value |
| `fusion` | how the two gradients are combined, hybrid only |
| `budget_level` | the trailing number of the group name, in either syntax. Robust for any method, even when query and demonstration budgets vary together |
| `normalize_agent_reward`, `initial_queries`, `demo_weight`, `query_schedule`, `fragmenter_type`, `pref_temperature`, `reward_net_arch`, `demo_subsample_seed`, `total_timesteps` | run hyperparameters |
| `state`, `project`, `group_tag` | W&B state, project, and the free label inside the group name |

Only `finished` runs are used by default; `--state any` includes the rest.

## Examples

```bash
# every method, one series each, learning curve
python plots/scripts/plot_curves.py --name learning_all

# the hybrid combinations, one column per label type
python plots/scripts/plot_curves.py --filter arm_family=hybrid \
    --cols pref_labels --hue demo_loss --name hybrid_learning

# budget curves for the preference methods, with the 90% rule
python plots/scripts/plot_budget.py --filter arm_family=pref --name budget_pref

# the fusion schemes compared, normalization off
python plots/scripts/plot_budget.py --compare-fusion --name fusions \
    --filter arm_family=hybrid pref_labels=soft normalize_agent_reward=False

# budget curves for the demonstration-only methods, x axis in trajectories
python plots/scripts/plot_budget.py --filter arm_family=demo \
    --metric sweep/success_rate --budget-x demo_budget --name budget_demo

# what is available
python plots/scripts/list_runs.py --by arm
```

`--list-metrics` on either script prints the catalogue. Any other key logged on
W&B is accepted too: the x axis is guessed from the prefix, `agent/*` meaning
timesteps and everything else iterations.

## Style

The rules live in [`style.toml`](style.toml): one colour per method, so a method
keeps its colour across every figure, plus widths, bands, legends, series names
and LaTeX macros. The hybrid combinations share the colour of their `demo_loss`
and are told apart by dashes. It applies to the preview and to the `.tex`
export, and is reread for every figure, so saving the file is enough.

## Notes

* The cache lives in `plots/.cache/`, overridable with `RTPLOTS_CACHE`, and git
  ignores it. **Only finished runs reach the disk**: the curve of a running run
  will grow, and a partial cache would sit there forever, quietly shortening
  every figure that uses it, because `curves.aggregate` fits the shared grid to
  the shortest run in the group. To clean caches written before that rule:

  ```bash
  python plots/scripts/clean_curve_cache.py          # list
  python plots/scripts/clean_curve_cache.py --apply  # delete
  ```

  It spots suspect files by comparing each run with its siblings in the same
  group on the same metric, so it works on both x axes without telling them
  apart. When a series does get shortened, the preview and the command line say
  so rather than leaving you to guess.
* `RTPLOTS_WANDB_PROJECTS`, a comma-separated list, chooses which projects to
  index. Adding one is a line in `rtplots/paths.py`, not a new module, because
  every project written by the same entry point is read the same way. A group
  with a new level syntax does need adding to `parse_group`, or `budget_level`
  stays empty and that project's budget curves come out blank.
* In `style.toml` the colour rules for `fusion` come **before** those for `arm`:
  the first match wins, and without that order every scheme would share one
  colour.
* `tikzplotlib` stopped at 0.10.1 and does not cope with recent matplotlib,
  numpy or webcolors without the aliases in `rtplots/tikz.py:_compat()`. See
  `plots/requirements.txt` for the compatible versions.
