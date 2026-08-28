#!/usr/bin/env python
"""Re-evaluate saved policies on more episodes, and aggregate the results.

Training ends with a 20-episode evaluation, enough to follow a run but too coarse
for a table: one collision moves the mean by about 8 points, more than the
differences between methods.

    python experiments/evaluate.py outputs/runs/*/*
    python experiments/evaluate.py --aggregate outputs/runs results

Re-evaluation reuses the evaluate() from training, so the procedure is identical
and the episode seeds line up: with the same base seed the first 20 episodes are
the ones already measured. Results are written next to each checkpoint and
existing ones are skipped, so it can be interrupted.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

ENV_ID = "HighwayEgo-v0"
ENV_KWARGS = {"ego": "continuous", "reward": "fast"}

# --- aggregation -------------------------------------------------------------

#: Groups that were re-run and whose earlier version must not be read back. Both
#: versions are still on disk, and mixing them gives a cell twenty seeds deep --
#: or, if the newer one is missing, quietly wrong numbers.
SUPERSEDED = {
    "th_1mh4_hybrid_soft_B10", "th_1mh4_hybrid_bern_B10",
    "th_1mh4_unw_soft_B10", "th_1mh4_unw_bern_B10",
}

#: Display names.
METHOD_NAMES = {
    "demo_only": "Demo-only", "pref_soft": "Pref-soft", "pref_bern": "Pref-Bernoulli",
    "hybrid_soft": "Hybrid-soft", "hybrid_bern": "Hybrid-Bernoulli",
    "unw_soft": "NB-soft", "unw_bern": "NB-Bernoulli",
}
METHOD_ORDER = list(METHOD_NAMES.values())
#: <campaign>_<arm>_B<budget>-seed<n>. The arm is matched by name because both
#: the campaign label and the arm names contain underscores.
RUN_DIR_RE = re.compile(
    r"(?P<campaign>.+)_(?P<arm>" + "|".join(METHOD_NAMES) + r")_B(?P<budget>\d+)-seed(?P<seed>\d+)$"
)


def other_rate(metrics: dict) -> float:
    """Episodes whose outcome is none of the four recorded ones.

    The environment can also report `teleported` or `removed_unknown`. Runs
    evaluated after this was noticed record `eval/other_rate` directly; for older
    files it is the residual, which is all that can be recovered from them.
    """
    if "eval/other_rate" in metrics:
        return float(metrics["eval/other_rate"])
    residuo = 1.0 - sum(metrics[f"eval/{k}_rate"]
                        for k in ("success", "collision", "off_road", "timeout"))
    return round(residuo, 6) or 0.0          # `or 0.0` normalises rounding's -0.0


def collect(root: Path, episodes: int) -> list[dict]:
    rows = []
    # rglob, not a fixed depth: older runs sit one directory deeper.
    for path in sorted(root.rglob(f"final_eval_{episodes}.json")):
        m = RUN_DIR_RE.fullmatch(path.parent.name)
        if not m:
            continue
        arm, budget, seed = m["arm"], int(m["budget"]), int(m["seed"])
        if f"{m['campaign']}_{arm}_B{budget}" in SUPERSEDED:
            continue
        d = json.loads(path.read_text())
        rows.append({
            "method": METHOD_NAMES[arm], "budget": budget, "seed": seed,
            "mean_return": d["eval/mean_fast_return"],
            "success_rate": d["eval/success_rate"],
            "collision_rate": d["eval/collision_rate"],
            "off_road_rate": d["eval/off_road_rate"],
            "timeout_rate": d["eval/timeout_rate"],
            "other_rate": other_rate(d),
            "mean_speed": d["eval/mean_speed"],
            "mean_ep_length": d["eval/mean_ep_length"],
            "run_name": path.parent.name,
        })
    rows.sort(key=lambda r: (METHOD_ORDER.index(r["method"]), r["budget"], r["seed"]))
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for name in METHOD_ORDER:
        for budget in (10, 100, 1000):
            cell = [r for r in rows if r["method"] == name and r["budget"] == budget]
            if not cell:
                continue
            v = [r["mean_return"] for r in cell]
            out.append({
                "method": name, "budget": budget, "n_seeds": len(cell),
                # A single seed has no spread; leave the field empty rather than
                # write a 0 that reads like a measurement.
                "mean_return": round(st.mean(v), 3),
                "std_return": round(st.stdev(v), 3) if len(v) > 1 else "",
                "median_return": round(st.median(v), 3),
                "min_return": round(min(v), 3), "max_return": round(max(v), 3),
                "seeds_below_zero": sum(1 for x in v if x < 0),
                "success_rate": f"{st.mean(r['success_rate'] for r in cell):.4f}",
                "collision_rate": f"{st.mean(r['collision_rate'] for r in cell):.4f}",
                "timeout_rate": f"{st.mean(r['timeout_rate'] for r in cell):.4f}",
            })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    # lineterminator="\n": the csv module defaults to CRLF, which makes git see
    # every line as trailing whitespace.
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


EXPECTED_BUDGETS = (10, 100, 1000)
EXPECTED_SEEDS = frozenset(range(1, 11))


def check_grid(rows: list[dict]) -> list[str]:
    """Complaints about a grid that is not the full 7 x 3 x 10.

    Aggregating a partial grid is not an error in itself, but doing it silently
    is: a cell built from nine seeds looks exactly like one built from ten, and
    the mean it produces is not the one the reference numbers report.
    """
    problems = []
    seen = {(r["method"], r["budget"]): [] for r in rows}
    for r in rows:
        seen[(r["method"], r["budget"])].append(r["seed"])

    expected_cells = {(m, b) for m in METHOD_ORDER for b in EXPECTED_BUDGETS}
    for cell in sorted(expected_cells - set(seen)):
        problems.append(f"{cell[0]} B={cell[1]}: missing entirely")
    for cell in sorted(set(seen) - expected_cells):
        problems.append(f"{cell[0]} B={cell[1]}: not part of the grid")

    for cell, seeds in sorted(seen.items()):
        if cell not in expected_cells:
            continue
        doubles = sorted({s for s in seeds if seeds.count(s) > 1})
        if doubles:
            problems.append(f"{cell[0]} B={cell[1]}: duplicated seeds {doubles}")
        missing = sorted(EXPECTED_SEEDS - set(seeds))
        if missing:
            problems.append(f"{cell[0]} B={cell[1]}: missing seeds {missing}")
        extra = sorted(set(seeds) - EXPECTED_SEEDS)
        if extra:
            problems.append(f"{cell[0]} B={cell[1]}: unexpected seeds {extra}")
    return problems


def aggregate(root: Path, dest: Path, episodes: int, allow_incomplete: bool = False) -> int:
    rows = collect(root, episodes)
    if not rows:
        print(f"no final_eval_{episodes}.json under {root}")
        return 1

    problems = check_grid(rows)
    if problems:
        head = "aggregating an incomplete grid" if allow_incomplete else "incomplete grid"
        print(f"{head}: {len(problems)} problem(s)")
        for pb in problems:
            print(f"  {pb}")
        if not allow_incomplete:
            print("\nrefusing to write. Pass --allow-incomplete to aggregate anyway.")
            return 1
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / f"results_{episodes}_episodes.csv", rows)
    write_csv(dest / f"results_{episodes}_episodes_summary.csv", summarise(rows))
    print(f"{len(rows)} runs, {len(summarise(rows))} cells -> {dest}")
    return 0


# --- re-evaluation -----------------------------------------------------------

def reevaluate(run_dirs: list[Path], episodes: int, seed: int, force: bool) -> int:
    from stable_baselines3 import SAC          # loads SUMO; not needed to aggregate

    from utils.common import evaluate

    status = 0
    for run_dir in run_dirs:
        out = run_dir / f"final_eval_{episodes}.json"
        if out.exists() and not force:
            print(f"SKIP    {run_dir.name} ({out.name} already there)")
            continue
        ckpt = run_dir / "agent_final.zip"
        if not ckpt.exists():
            print(f"NO-CKPT {run_dir.name}")
            status = 1
            continue
        model = SAC.load(str(ckpt), device="cpu")
        metrics = evaluate(model, ENV_ID, ENV_KWARGS, episodes, seed)
        out.write_text(json.dumps(metrics, indent=2))
        print(f"OK      {run_dir.name} "
              f"return={metrics['eval/mean_fast_return']:.2f} "
              f"success={metrics['eval/success_rate']:.3f} "
              f"collision={metrics['eval/collision_rate']:.3f}")
    return status


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path,
                   help="run directories to evaluate, or ROOT and DEST with --aggregate")
    p.add_argument("--aggregate", action="store_true",
                   help="do not evaluate: read the existing files and write the tables")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=1000,
                   help="seed of the first episode; episode i uses seed+i. The default "
                        "matches the evaluation at the end of training, so the first "
                        "episodes are the same ones.")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="with --aggregate: write the tables even if the 7 x 3 x 10 grid "
                        "has holes. Off by default, because a cell built from nine seeds "
                        "is indistinguishable from one built from ten.")
    args = p.parse_args()

    if args.aggregate:
        if len(args.paths) != 2:
            p.error("--aggregate takes exactly two paths: ROOT and DEST")
        return aggregate(args.paths[0], args.paths[1], args.episodes,
                         allow_incomplete=args.allow_incomplete)
    return reevaluate(args.paths, args.episodes, args.seed, args.force)


if __name__ == "__main__":
    sys.exit(main())
