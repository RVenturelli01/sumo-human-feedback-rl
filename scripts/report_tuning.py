"""Report on the Optuna tuning campaign (W&B project tuning-thesis).

For every tuning group (run.group starting with ``tune_``):

* top-k trials table (params + objective + outcome rates) — Markdown + LaTeX;
* objective-vs-trial scatter (finished/crashed states) per arm, one figure
  with all arms sharing the y-axis — the search-progress picture;
* optional Optuna parameter importances, if a (copy of the) journal file is
  given with ``--journal`` (never point it at the live journal over NFS while
  writing: scp a copy, reading it is enough).

Usage:
    python scripts/report_tuning.py                       # all tune_* groups
    python scripts/report_tuning.py --journal /tmp/journal.log --top-k 5
"""

import argparse
from collections import defaultdict

import pandas as pd

from _report_common import (
    DEFAULT_OUT,
    ENTITY,
    api,
    arm_color,
    new_axes,
    save_figure,
    save_table,
)

PARAM_KEYS = (
    ("lr_rew", "algo.kwargs.lr_rew"),
    ("gsteps_rew", "algo.kwargs.gradient_steps_rew"),
    ("l2_rew", "algo.kwargs.l2_rew"),
    ("net_arch", "algo.kwargs.reward_model_kwargs.net_arch"),
    ("init_agent_ts", "algo.kwargs.initial_agent_timesteps"),
    ("batch_pref", "algo.kwargs.batch_size_pref"),
    ("schedule", "algo.kwargs.query_schedule"),
    ("init_queries", "algo.kwargs.initial_queries"),
    ("fragmenter", "algo.kwargs.fragmenter_type"),
    ("batch_expert", "algo.kwargs.batch_size_expert"),
    ("batch_model", "algo.kwargs.batch_size_model"),
    ("demo_weight", "algo.kwargs.demo_weight"),
)


def dig(config: dict, dotted: str):
    node = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def trial_number(run_name: str):
    if "-t" in run_name:
        try:
            return int(run_name.rsplit("-t", 1)[1])
        except ValueError:
            pass
    return None


def collect_groups(project: str):
    groups = defaultdict(list)
    for run in api().runs(f"{ENTITY}/{project}", per_page=300):
        if (run.group or "").startswith("tune_"):
            groups[run.group].append(run)
    return groups


def group_frame(runs) -> pd.DataFrame:
    rows = []
    for run in runs:
        row = {
            "trial": trial_number(run.name),
            "state": run.state,
            "objective": run.summary.get("sweep/mean_fast_return"),
            "success": run.summary.get("sweep/success_rate"),
            "collision": run.summary.get("sweep/collision_rate"),
            "hours": (run.summary.get("_runtime") or 0) / 3600,
        }
        for label, dotted in PARAM_KEYS:
            value = dig(run.config, dotted)
            if value is not None:
                row[label] = value
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("trial")
    return df


def plot_progress(frames: dict, out_dir):
    from _report_common import base_arm

    fig, ax = new_axes(width=7.5, height=4.2)
    # Same arm at different budgets keeps its hue; variants change linestyle.
    variant_count = {}
    for group in sorted(frames):
        df = frames[group].dropna(subset=["trial", "objective"])
        if not len(df):
            continue
        color = arm_color(group)
        arm = base_arm(group.removeprefix("tune_"))
        style = ["-", "--", ":"][variant_count.get(arm, 0) % 3]
        variant_count[arm] = variant_count.get(arm, 0) + 1
        label = group.removeprefix("tune_")
        ax.plot(df["trial"], df["objective"], color=color, linewidth=1.8,
                linestyle=style, marker="o", markersize=5, label=label, alpha=0.9)
        best = df["objective"].cummax()
        ax.plot(df["trial"], best, color=color, linewidth=0.9, linestyle=":", alpha=0.45)
    ax.set_xlabel("trial")
    ax.set_ylabel("eval/mean_fast_return (held-out)")
    ax.set_title("Andamento della ricerca per braccio (tratteggio: best-so-far)",
                 fontsize=10, color="#0b0b0b")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save_figure(fig, out_dir, "tuning_progress")


def importances(journal_path: str, group: str, out_dir):
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    storage = JournalStorage(JournalFileBackend(journal_path))
    study_name = f"hybrid_sac_{group.removeprefix('tune_')}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        return None
    done = [t for t in study.trials if t.value is not None]
    if len(done) < 5:
        return None
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception as exc:  # fANOVA needs >=2 distinct values per param
        print(f"  ({study_name}: importances not computable: {exc})")
        return None
    return pd.Series(imp, name=group)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", default="tuning-thesis")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--journal", default=None,
                        help="Local COPY of the Optuna journal for param importances.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = __import__("pathlib").Path(args.out)

    groups = collect_groups(args.project)
    if not groups:
        raise SystemExit(f"No tune_* groups found in {args.project}.")

    frames = {}
    for group, runs in sorted(groups.items()):
        df = group_frame(runs)
        frames[group] = df
        finished = df[df["state"] == "finished"].dropna(subset=["objective"])
        print(f"{group}: {len(df)} run, {len(finished)} finite, "
              f"best={finished['objective'].max() if len(finished) else float('nan'):.1f}")
        top = finished.sort_values("objective", ascending=False).head(args.top_k)
        top = top.drop(columns=["state"]).set_index("trial")
        save_table(top, out_dir, f"tuning_top_{group.removeprefix('tune_')}",
                   float_fmt="%.3g")

    plot_progress(frames, out_dir)

    if args.journal:
        series = [s for g in sorted(groups)
                  if (s := importances(args.journal, g, out_dir)) is not None]
        if series:
            imp_df = pd.concat(series, axis=1).fillna(0.0)
            save_table(imp_df, out_dir, "tuning_param_importances", float_fmt="%.3f")


if __name__ == "__main__":
    main()
