#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAUNCHER="$SCRIPT_DIR/run_demo_SAC.sh"

# Edit this list to choose the paired seeds.
SEEDS=(0 1 2)
LOSS_TYPE=maxent_corrected
OUTPUT_DIR="outputs/demo_sac_relabel_ablation"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-false}"

# Every extra Hydra override passed to this launcher is applied to both
# conditions. The seed, loss and relabelling flag below remain paired.

run_condition() {
  local seed="$1"
  local relabel="$2"
  shift 2

  if [[ "$DRY_RUN" == "true" ]]; then
    PYTHON_BIN="$PYTHON_BIN" "$BASE_LAUNCHER" \
      "$@" \
      run.seed="$seed" \
      run.output_dir="$OUTPUT_DIR" \
      algo.kwargs.loss_type="$LOSS_TYPE" \
      algo.kwargs.relabel_rewards="$relabel" \
      --cfg job
    return
  fi

  PYTHON_BIN="$PYTHON_BIN" "$BASE_LAUNCHER" \
    "$@" \
    run.seed="$seed" \
    run.output_dir="$OUTPUT_DIR" \
    algo.kwargs.loss_type="$LOSS_TYPE" \
    algo.kwargs.relabel_rewards="$relabel"
}

for seed in "${SEEDS[@]}"; do
  for relabel in true false; do
    echo "============================================================"
    echo "SAC relabelling ablation: seed=$seed relabel_rewards=$relabel"
    echo "============================================================"
    run_condition "$seed" "$relabel" "$@"
  done
done
