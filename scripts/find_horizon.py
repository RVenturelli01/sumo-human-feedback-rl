"""Pick the tuning horizon from the W&B history of past pref/demo-only runs.

For every historical pref_only (demo_weight == 0) and demo_only
(total_queries == 0) hybrid run, smooth agent/rewards/ep_fast_return over
timesteps and compute t95: the first timestep after which the smoothed curve
stays at or above 95% of its plateau (mean of the last 20% of points). Then:

* T_final = 2 x max over arms of the median t95, rounded up to 100k
  (baselines converge before the halfway point by construction);
* T_tune  = T_final / 2 (search horizon; final validation runs at T_final).

Fallback when history is missing or inconclusive: T_final=2M, T_tune=1M.

Usage:
    python scripts/find_horizon.py [--entity ...] [--project ...] [--min-points 20]
"""

import argparse
import math

import numpy as np
import wandb

METRIC = "agent/rewards/ep_fast_return"
STEP_KEY = "agent/time/total_timesteps"
FALLBACK_T_FINAL = 2_000_000
SMOOTH_WINDOW = 10
PLATEAU_FRAC = 0.2
PLATEAU_RATIO = 0.95


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    out = np.empty(len(values))
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out[i] = np.median(values[lo:hi])
    return out


def t95(steps: np.ndarray, values: np.ndarray) -> float | None:
    """First timestep after which the smoothed curve stays >= 95% of plateau."""
    smoothed = rolling_median(values, SMOOTH_WINDOW)
    n_tail = max(1, int(PLATEAU_FRAC * len(smoothed)))
    plateau = float(np.mean(smoothed[-n_tail:]))
    baseline = float(np.min(smoothed))
    if not math.isfinite(plateau) or plateau <= baseline:
        return None
    # Threshold on the improvement over the worst point, not the raw value,
    # so negative returns are handled correctly.
    threshold = baseline + PLATEAU_RATIO * (plateau - baseline)
    above = smoothed >= threshold
    for i in range(len(above)):
        if above[i:].all():
            return float(steps[i])
    return None


def classify_arm(config: dict) -> str | None:
    algo = config.get("algo", {}).get("kwargs", {})
    if not algo:
        return None
    if float(algo.get("demo_weight", 1.0)) == 0.0:
        return f"pref_only/{algo.get('labels_type', '?')}"
    if int(algo.get("total_queries", -1)) == 0:
        return f"demo_only/{algo.get('loss_type', '?')}"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entity", default="andrea02polimi-politecnico-di-milano")
    parser.add_argument("--project", default="preference+demonstration")
    parser.add_argument("--min-points", type=int, default=20,
                        help="Skip runs with fewer logged metric points.")
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}", filters={"state": "finished"})

    per_arm: dict[str, list[float]] = {}
    for run in runs:
        arm = classify_arm(run.config)
        if arm is None:
            continue
        history = run.history(keys=[STEP_KEY, METRIC], pandas=True)
        if history is None or len(history) < args.min_points:
            continue
        history = history.dropna(subset=[STEP_KEY, METRIC]).sort_values(STEP_KEY)
        value = t95(history[STEP_KEY].to_numpy(), history[METRIC].to_numpy())
        status = f"t95={value:,.0f}" if value is not None else "no plateau"
        print(f"  [{arm}] {run.name} ({run.id}): {len(history)} pts, {status}")
        if value is not None:
            per_arm.setdefault(arm, []).append(value)

    if not per_arm:
        print("\nNo usable history found. Fallback:")
        print(f"  T_final = {FALLBACK_T_FINAL:,}   T_tune = {FALLBACK_T_FINAL // 2:,}")
        return

    print("\nPer-arm median t95:")
    medians = {}
    for arm, values in sorted(per_arm.items()):
        medians[arm] = float(np.median(values))
        print(f"  {arm}: median {medians[arm]:,.0f} over {len(values)} runs")

    worst = max(medians.values())
    t_final = int(math.ceil(2 * worst / 100_000) * 100_000)
    print(f"\nmax median t95 = {worst:,.0f}")
    print(f"T_final = {t_final:,} (2 x max, rounded up to 100k)")
    print(f"T_tune  = {t_final // 2:,}")
    print("\nPass to the tuner:  --total-timesteps", t_final // 2)


if __name__ == "__main__":
    main()
