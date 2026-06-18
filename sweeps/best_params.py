"""
Print the best hyperparameters from a finished W&B sweep.

Ranks the sweep's runs by its own optimisation metric (sweep/mean_fast_return)
and prints the winning agent.kwargs as YAML, ready to paste into the
`# BEST FROM STAGE ...` block of the next stage's sweep file.

Usage:
    python sweeps/best_params.py <entity/project/sweep_id> [--top N]
"""

import argparse

import wandb
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", help="entity/project/sweep_id")
    parser.add_argument("--top", type=int, default=5, help="how many top runs to list")
    args = parser.parse_args()

    api = wandb.Api()
    sweep = api.sweep(args.sweep)
    metric = sweep.config.get("metric", {}).get("name", "sweep/mean_fast_return")
    goal = sweep.config.get("metric", {}).get("goal", "maximize")

    # Keep only finished runs that actually logged the metric.
    runs = [r for r in sweep.runs
            if r.state == "finished" and r.summary.get(metric) is not None]
    runs.sort(key=lambda r: r.summary[metric], reverse=(goal == "maximize"))

    if not runs:
        print("No finished runs with the metric yet.")
        return

    print(f"Sweep {args.sweep}")
    print(f"Optimising {metric} ({goal}) over {len(runs)} finished runs\n")

    print(f"Top {min(args.top, len(runs))} runs:")
    for r in runs[:args.top]:
        std = r.summary.get(metric + "_std")
        std_s = f" ± {std:.3f}" if std is not None else ""
        print(f"  {r.summary[metric]:>10.3f}{std_s}   {r.name}")

    best = runs[0]
    print(f"\nBest run: {best.name}  ({metric} = {best.summary[metric]:.3f})")
    print("\nBest agent.kwargs (paste into the next stage's sweep YAML):\n")
    agent_kwargs = best.config.get("agent", {}).get("kwargs", {})
    print(yaml.safe_dump(agent_kwargs, sort_keys=False, default_flow_style=False))


if __name__ == "__main__":
    main()
