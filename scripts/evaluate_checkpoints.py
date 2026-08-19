#!/usr/bin/env python
"""Re-evaluate a trained policy on more episodes, starting from its checkpoint.

Training ends with a 20-episode evaluation, which is enough to follow a run but
too coarse for the final tables: with 20 episodes a single collision moves the
mean return by about 8 points, which is larger than the differences between the
methods. This script re-runs the evaluation on as many episodes as you want,
reading `agent_final.zip` and without retraining anything.

It calls the same `evaluate()` used at the end of training, so the procedure is
identical: one environment, deterministic policy, episode seeds `seed + i`. With
the same starting seed the first 20 episodes are the ones already evaluated, so
a longer run extends the series instead of replacing it.

    # one run
    python scripts/evaluate_checkpoints.py outputs/thesis_runs/<group>/<run>/<run>

    # every run of the thesis campaigns, 44 at a time
    find outputs/thesis_runs/th_1mh4* -name agent_final.zip -exec dirname {} \\; \\
      | xargs -P 44 -I{} python scripts/evaluate_checkpoints.py {}

Writes `final_eval_<episodes>.json` next to the checkpoint and skips runs that
already have one, so it can be interrupted and restarted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

ENV_ID = "HighwayEgo-v0"
ENV_KWARGS = {"ego": "continuous", "reward": "fast"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="+", type=Path,
                   help="directory containing agent_final.zip")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=1000,
                   help="seed of the first episode; episode i uses seed+i. The "
                        "default matches the evaluation run at the end of training, "
                        "so the first episodes are the same ones.")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = p.parse_args()

    # importati qui: caricano SUMO, inutile pagarlo se gli argomenti sono sbagliati
    from stable_baselines3 import SAC

    from _common import evaluate

    uscita = 0
    for run_dir in args.run_dir:
        out = run_dir / f"final_eval_{args.episodes}.json"
        if out.exists() and not args.force:
            print(f"SKIP    {run_dir.name} ({out.name} already there)")
            continue
        ckpt = run_dir / "agent_final.zip"
        if not ckpt.exists():
            print(f"NO-CKPT {run_dir.name}")
            uscita = 1
            continue

        model = SAC.load(str(ckpt), device="cpu")
        metrics = evaluate(model, ENV_ID, ENV_KWARGS, args.episodes, args.seed)
        out.write_text(json.dumps(metrics, indent=2))
        print(f"OK      {run_dir.name} "
              f"return={metrics['eval/mean_fast_return']:.2f} "
              f"success={metrics['eval/success_rate']:.3f} "
              f"collision={metrics['eval/collision_rate']:.3f}")
    return uscita


if __name__ == "__main__":
    sys.exit(main())
