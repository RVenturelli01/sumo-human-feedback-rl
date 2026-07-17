"""Report on the sample-budget curves (W&B groups ``budget_<arm>_<level>``).

Aggregates sweep/mean_fast_return and sweep/success_rate across seeds per
(arm, budget level), draws one curve per arm (log-x budget, error bars over
seeds; return and success on separate panels — never a dual axis), and applies
the minimum-budget rule:

    the smallest level whose seed-mean of BOTH metrics is >= 90% of the
    largest level's, with the next level up also passing.

Prints X (pref arms) and Y (demo arms) and writes the per-level table.

Usage:
    python scripts/report_budget_curves.py                  # project tuning-thesis
    python scripts/report_budget_curves.py --project smoke  # schema test
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _report_common import (
    DEFAULT_OUT,
    ENTITY,
    api,
    arm_color,
    save_figure,
    save_table,
    style_axes,
)

GROUP_RE = re.compile(r"^budget_([a-z0-9_]+)_(\d+)$")
RETURN_METRIC = "sweep/mean_fast_return"
SUCCESS_METRIC = "sweep/success_rate"
PASS_RATIO = 0.90


def collect(project: str):
    """{arm: {level: [runs]}} from budget_* groups."""
    per_arm = defaultdict(lambda: defaultdict(list))
    for run in api().runs(f"{ENTITY}/{project}", per_page=300):
        m = GROUP_RE.match(run.group or "")
        if m and run.state == "finished":
            per_arm[m.group(1)][int(m.group(2))].append(run)
    return per_arm


def level_frame(levels: dict) -> pd.DataFrame:
    rows = []
    for level, runs in sorted(levels.items()):
        ret = pd.Series([r.summary[RETURN_METRIC] for r in runs if RETURN_METRIC in r.summary], dtype=float)
        suc = pd.Series([r.summary[SUCCESS_METRIC] for r in runs if SUCCESS_METRIC in r.summary], dtype=float)
        if not len(ret):
            continue
        rows.append({
            "budget": level, "n_seeds": len(ret),
            "return_mean": ret.mean(), "return_std": ret.std(ddof=0),
            "success_mean": suc.mean(), "success_std": suc.std(ddof=0),
        })
    return pd.DataFrame(rows).sort_values("budget").set_index("budget")


def relative_score(series: pd.Series, reference: float) -> pd.Series:
    """Fraction of the improvement over the worst level that is retained.

    Robust to negative returns: 1.0 at the reference (largest budget),
    0.0 at the worst level.
    """
    lo = series.min()
    span = reference - lo
    if span <= 0:
        return pd.Series(1.0, index=series.index)
    return (series - lo) / span


def minimum_budget(df: pd.DataFrame):
    if len(df) < 2:
        return None
    full = df.index.max()
    rel_ret = relative_score(df["return_mean"], df.loc[full, "return_mean"])
    rel_suc = relative_score(df["success_mean"], df.loc[full, "success_mean"])
    passing = (rel_ret >= PASS_RATIO) & (rel_suc >= PASS_RATIO)
    levels = sorted(df.index)
    for i, level in enumerate(levels):
        upper_ok = passing[levels[i + 1]] if i + 1 < len(levels) else True
        if passing[level] and upper_ok:
            return level
    return full


def plot(per_arm_frames: dict, out_dir: Path):
    fig, (ax_ret, ax_suc) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax in (ax_ret, ax_suc):
        style_axes(ax)
        ax.set_xscale("log")
        ax.set_xlabel("budget (log)")
    for arm, df in sorted(per_arm_frames.items()):
        color = arm_color(arm)
        ax_ret.errorbar(df.index, df["return_mean"], yerr=df["return_std"],
                        color=color, linewidth=1.8, marker="o", markersize=5,
                        capsize=3, label=arm)
        ax_suc.errorbar(df.index, df["success_mean"], yerr=df["success_std"],
                        color=color, linewidth=1.8, marker="o", markersize=5,
                        capsize=3, label=arm)
    ax_ret.set_ylabel("mean_fast_return")
    ax_suc.set_ylabel("success_rate")
    ax_suc.set_ylim(-0.02, 1.02)
    ax_ret.legend(frameon=False, fontsize=8)
    fig.suptitle("Curve di budget (media ± std sui seed)", fontsize=10, color="#0b0b0b")
    save_figure(fig, out_dir, "budget_curves")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", default="tuning-thesis")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.out)

    per_arm = collect(args.project)
    if not per_arm:
        raise SystemExit(f"No budget_* groups found in {args.project}.")

    frames = {}
    for arm, levels in sorted(per_arm.items()):
        df = level_frame(levels)
        if not len(df):
            continue
        frames[arm] = df
        save_table(df, out_dir, f"budget_curve_{arm}", float_fmt="%.2f")
        minimo = minimum_budget(df)
        print(f"{arm}: livelli {list(df.index)} -> budget minimo (regola {PASS_RATIO:.0%}): {minimo}")

    if frames:
        plot(frames, out_dir)


if __name__ == "__main__":
    main()
