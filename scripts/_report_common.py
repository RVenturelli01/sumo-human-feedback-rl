"""Shared helpers for the thesis analysis/report scripts.

All report scripts read W&B via the public API and write figures (PNG+PDF)
and tables (Markdown+LaTeX) under reports/thesis/. Colors follow the entity:
every arm keeps the same hue in every figure (colorblind-validated palette).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "andrea02polimi-politecnico-di-milano"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "reports" / "thesis"

# Fixed arm order and hues (validated categorical palette, light surface).
# Hybrid budget variants (_A/_B) share the arm's hue and differ by linestyle.
ARM_ORDER = (
    "pref_soft", "pref_bernoulli", "demo_1", "demo_2", "hybrid_demo_1", "hybrid_demo_2",
)
ARM_COLORS = {
    "pref_soft": "#2a78d6",       # blue
    "pref_bernoulli": "#1baf7a",  # aqua
    "demo_1": "#eda100",          # yellow
    "demo_2": "#008300",          # green
    "hybrid_demo_1": "#4a3aa7",   # violet
    "hybrid_demo_2": "#e34948",   # red
}
FALLBACK_COLOR = "#e87ba4"        # magenta, for unexpected groups

SWEEP_METRICS = (
    "sweep/mean_fast_return",
    "sweep/success_rate",
    "sweep/collision_rate",
    "sweep/off_road_rate",
    "sweep/timeout_rate",
    "sweep/mean_speed",
    "sweep/mean_ep_length",
)


def api():
    return wandb.Api(timeout=60)


def base_arm(group: str) -> str:
    """Map a run group to its arm: tune_pref_soft, hybrid_demo_1_A -> arm name."""
    name = group
    for prefix in ("tune_", "budget_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for arm in sorted(ARM_ORDER, key=len, reverse=True):
        if name == arm or name.startswith(arm):
            return arm
    return group


def arm_color(group: str) -> str:
    return ARM_COLORS.get(base_arm(group), FALLBACK_COLOR)


def new_axes(width=7.0, height=4.0):
    fig, ax = plt.subplots(figsize=(width, height))
    style_axes(ax)
    return fig, ax


def style_axes(ax):
    """Recessive grid and spines: the data carries the figure."""
    ax.grid(axis="y", color="#e3e2de", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)


def save_figure(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure: {out_dir / name}.png (+.pdf)")


def save_table(df, out_dir: Path, name: str, float_fmt: str = "%.2f") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.md").write_text(df.to_markdown(floatfmt=float_fmt.lstrip("%")) + "\n")
    (out_dir / f"{name}.tex").write_text(
        df.to_latex(float_format=lambda v: float_fmt % v, escape=True)
    )
    print(f"  table: {out_dir / name}.md (+.tex)")


def mean_std_label(mean: float, std: float, fmt: str = "{:.1f}") -> str:
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def interp_curves(histories, n_points: int = 100):
    """Interpolate (x, y) curves on a common grid; returns grid, mean, std.

    Uses the smallest common x-range so no curve is extrapolated.
    """
    histories = [(np.asarray(x), np.asarray(y)) for x, y in histories if len(x) > 1]
    if not histories:
        return None, None, None
    lo = max(x.min() for x, _ in histories)
    hi = min(x.max() for x, _ in histories)
    if hi <= lo:
        return None, None, None
    grid = np.linspace(lo, hi, n_points)
    stack = np.vstack([np.interp(grid, x, y) for x, y in histories])
    return grid, stack.mean(axis=0), stack.std(axis=0)
