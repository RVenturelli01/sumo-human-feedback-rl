#!/usr/bin/env bash
# Launch N W&B sweep agents in parallel, each pinned to its own CPU core.
#
# The SAC nets are tiny (64x64) and the bottleneck is the single-threaded SUMO
# simulation, so one core per agent is the right ratio. Each process is also
# capped to a single math thread so the pinned agents don't oversubscribe cores.
#
# Usage:
#   ./sweeps/run_agents.sh <entity/project/sweep_id> [n_agents] [first_core]
#
# Example (46 agents on a 48-core box, leaving cores 46-47 free):
#   ./sweeps/run_agents.sh andrea02polimi-politecnico-di-milano/sac-baseline-tuning/ab12cd34 46
#
# Agents stop on their own once the sweep's run_cap is reached. Logs go to logs/.
set -euo pipefail

SWEEP="${1:?usage: run_agents.sh <entity/project/sweep_id> [n_agents] [first_core]}"
N="${2:-16}"
FIRST_CORE="${3:-0}"

mkdir -p logs
echo "Launching $N agents on sweep $SWEEP (cores ${FIRST_CORE}..$((FIRST_CORE + N - 1)))"

for i in $(seq 0 $((N - 1))); do
    core=$((FIRST_CORE + i))
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        taskset -c "$core" wandb agent "$SWEEP" > "logs/agent_${core}.log" 2>&1 &
    echo "  agent on core $core (pid $!)"
done

echo "Tail progress with:  tail -f logs/agent_*.log"
wait
echo "All agents finished (sweep run_cap reached)."
