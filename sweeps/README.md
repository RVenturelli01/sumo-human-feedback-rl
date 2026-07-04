# SAC baseline hyperparameter sweeps

Staged W&B sweeps for the SAC baseline trained on the environment's **true**
reward ([`scripts/train_sac_baseline.py`](../scripts/train_sac_baseline.py)).
Each stage fixes the winners of the previous one and tunes a new group, so the
search stays tractable.

All sweeps maximise `sweep/mean_fast_return` — the mean true (fast) return of the
deterministic policy on a fresh evaluation env, **averaged across the seeds in
`run.seeds`** (default `[0, 1, 2]`). Each trial trains one agent per seed, so a
lucky single seed cannot win a trial.

> **Cost.** Each trial runs `len(run.seeds) × train.kwargs.total_timesteps`
> environment steps (3 × 500k = 1.5M by default). If sweeps are too slow, either
> lower `train.kwargs.total_timesteps`, drop to 2 seeds, or reduce `run_cap` in
> the sweep YAML.

## Prerequisites

Set the W&B project/entity once so the runs land in the right place. Either pass
them on the CLI (`wandb.project=... wandb.entity=...`) or edit
[`configs/train_sac_baseline.yaml`](../configs/train_sac_baseline.yaml).

## Workflow

```bash
# Stage 1 — Core RL (lr, gamma, tau, batch_size, ent_coef), 30-50 trials
wandb sweep sweeps/sweep_core.yaml
wandb agent <entity>/<project>/<sweep_id>     # run until you have enough trials

# Inspect the W&B sweep table, copy the best Core values into
# sweeps/sweep_buffer.yaml (the "BEST FROM STAGE 1" block), then:

# Stage 2 — Buffer / schedule, 30-50 trials
wandb sweep sweeps/sweep_buffer.yaml
wandb agent <entity>/<project>/<sweep_id>

# Copy best buffer/schedule values into sweeps/sweep_arch.yaml, then:

# Stage 3 — Architecture (net_arch), 20-30 trials
wandb sweep sweeps/sweep_arch.yaml
wandb agent <entity>/<project>/<sweep_id>
```

## Running on a multi-core server

A single `wandb agent` runs trials sequentially (one core). To use a many-core
box, launch many agents on the **same** sweep id — each pulls a different trial.
Pin one core per agent and cap math threads, otherwise PyTorch oversubscribes:

```bash
# 1. create the sweep once (prints the sweep id)
wandb sweep --project sac-baseline-tuning sweeps/sweep_core.yaml

# 2. launch e.g. 46 agents on a 48-core box (leaves cores 46-47 free)
./sweeps/run_agents.sh andrea02polimi-politecnico-di-milano/sac-baseline-tuning/<sweep_id> 46
```

[`run_agents.sh`](run_agents.sh) pins each agent with `taskset -c` and sets
`OMP_NUM_THREADS=1`. Agents exit on their own once `run_cap` is reached. Run it
under `tmux`/`nohup` since a full stage takes a while. One core per agent is the
right ratio: the bottleneck is the single-threaded SUMO sim, not the 64x64 nets.

## Reading the results

On the W&B sweep page, sort the runs table by `sweep/mean_fast_return` (or use
the parallel-coordinates plot). Or print the winner from the CLI:

```bash
python sweeps/best_params.py andrea02polimi-politecnico-di-milano/sac-baseline-tuning/<sweep_id>
```

This lists the top runs (with the across-seed std) and dumps the best run's
`agent.kwargs` as YAML, ready to paste into the next stage's
`# BEST FROM STAGE ...` block.

## Final confirmation

Put all the winning values together in
[`configs/train_sac_baseline.yaml`](../configs/train_sac_baseline.yaml) (or a
copy `sac_baseline_best.yaml`), bump `train.kwargs.total_timesteps` back to the
full budget, and run once (the script already trains all `run.seeds`):

```bash
python scripts/train_sac_baseline.py run.output_dir=outputs
```

These winning agent hyperparameters are what the demonstration-based
reward-learning runs (`scripts/test_demo_SAC.py`) should reuse.

## Notes

- The sweeps shorten `total_timesteps` to 500k to get more trials; the final
  confirmation runs use the full budget.
- Run multiple `wandb agent` processes in parallel (one per free CPU) to speed
  up a sweep — the agent is configured with `device: cpu`.
- Per-seed learning curves are logged under `agent_seed{seed}/...`; per-seed
  final eval under `per_seed/seed{seed}/...`; across-seed aggregates (mean + std)
  under `sweep/...`. All baseline runs share the W&B group `sac_baseline`.
