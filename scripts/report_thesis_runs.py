"""Report on the final multi-seed runs (W&B project thesis).

Groups runs by ``run.group`` (pref_soft, ..., hybrid_demo_1_A, hybrid_demo_1_B)
and produces:

* the main thesis table: mean ± std across seeds of the held-out eval metrics
  (sweep/*) per group — Markdown + LaTeX (plus a raw numeric CSV);
* learning curves: agent/rewards/ep_fast_return vs environment timesteps,
  seed-mean with a ±1 std band, one line per group. Hybrid budget variants
  share the arm's hue and differ by linestyle (A solid, B dashed).

Usage:
    python scripts/report_thesis_runs.py                    # project thesis
    python scripts/report_thesis_runs.py --project smoke    # schema test
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from _report_common import (
    DEFAULT_OUT,
    ENTITY,
    SWEEP_METRICS,
    api,
    arm_color,
    base_arm,
    interp_curves,
    mean_std_label,
    new_axes,
    save_figure,
    save_table,
)

CURVE_METRIC = "agent/rewards/ep_fast_return"
CURVE_STEP = "agent/time/total_timesteps"


def collect_groups(project: str, states=("finished",)):
    groups = defaultdict(list)
    for run in api().runs(f"{ENTITY}/{project}", per_page=300):
        if run.group and run.state in states:
            groups[run.group].append(run)
    return groups


def summary_tables(groups: dict, out_dir: Path):
    pretty_rows, raw_rows = {}, {}
    for group, runs in sorted(groups.items()):
        values = {m: [r.summary[m] for r in runs if m in r.summary] for m in SWEEP_METRICS}
        pretty, raw = {}, {"n_seeds": len(runs)}
        for metric, vals in values.items():
            if not vals:
                continue
            s = pd.Series(vals, dtype=float)
            short = metric.removeprefix("sweep/")
            fmt = "{:.2f}" if "rate" not in short else "{:.2f}"
            pretty[short] = mean_std_label(s.mean(), s.std(ddof=0), fmt)
            raw[f"{short}_mean"] = s.mean()
            raw[f"{short}_std"] = s.std(ddof=0)
        pretty["n_seeds"] = len(runs)
        pretty_rows[group] = pretty
        raw_rows[group] = raw

    pretty_df = pd.DataFrame(pretty_rows).T
    raw_df = pd.DataFrame(raw_rows).T
    save_table(pretty_df, out_dir, "thesis_main_table", float_fmt="%.2f")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(out_dir / "thesis_main_table_raw.csv")
    print(f"  csv: {out_dir / 'thesis_main_table_raw.csv'}")
    return pretty_df


def learning_curves(groups: dict, out_dir: Path):
    fig, ax = new_axes(width=7.5, height=4.4)
    plotted = 0
    # Same arm keeps its hue; budget/strategy variants (A/B, _q100k) rotate
    # linestyle so identity is never color-alone within one hue.
    variant_count = {}
    for group, runs in sorted(groups.items()):
        histories = []
        for run in runs:
            h = run.history(keys=[CURVE_STEP, CURVE_METRIC], pandas=True)
            if h is None or not len(h):
                continue
            h = h.dropna(subset=[CURVE_STEP, CURVE_METRIC]).sort_values(CURVE_STEP)
            histories.append((h[CURVE_STEP].to_numpy(), h[CURVE_METRIC].to_numpy()))
        grid, mean, std = interp_curves(histories)
        if grid is None:
            print(f"  ({group}: nessuna history utilizzabile)")
            continue
        color = arm_color(group)
        arm = base_arm(group)
        style = ["-", "--", ":"][variant_count.get(arm, 0) % 3]
        variant_count[arm] = variant_count.get(arm, 0) + 1
        ax.plot(grid, mean, color=color, linewidth=1.8,
                linestyle=style, label=group)
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        plotted += 1
    if not plotted:
        print("  nessuna curva: figura saltata")
        return
    ax.set_xlabel("environment timesteps")
    ax.set_ylabel("ep_fast_return (rollout)")
    ax.set_title("Curve di apprendimento (media sui seed, banda ±1 std)",
                 fontsize=10, color="#0b0b0b")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save_figure(fig, out_dir, "thesis_learning_curves")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", default="thesis")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--include-running", action="store_true",
                        help="Include also runs still marked running (partial).")
    args = parser.parse_args()
    out_dir = Path(args.out)

    states = ("finished", "running") if args.include_running else ("finished",)
    groups = collect_groups(args.project, states)
    if not groups:
        raise SystemExit(f"No grouped runs found in project {args.project}.")
    for group, runs in sorted(groups.items()):
        print(f"{group}: {len(runs)} run ({base_arm(group)})")

    df = summary_tables(groups, out_dir)
    print(df.to_string())
    learning_curves(groups, out_dir)


if __name__ == "__main__":
    main()
