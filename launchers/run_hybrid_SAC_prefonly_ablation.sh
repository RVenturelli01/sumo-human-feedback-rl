#!/usr/bin/env bash
set -euo pipefail

# Preference-branch ablation: hybrid (lambda_demo=1) vs pref_only (lambda_demo=0),
# holding *all* preference hyper-parameters fixed. Isolates whether sharing the
# reward net with the GCL demo loss is what caps preference accuracy: if pref_only
# reaches ~Christiano-level accuracy/correlation while hybrid plateaus lower, the
# demo gradient is drowning the preference signal on the shared reward model.
#
# The preference settings below mirror the reference hybrid run (soft labels,
# fragment_length=10, batch=500, 1000 queries/iter) so the only difference between
# the two conditions is lambda_demo.
#
# Usage:
#   ./launchers/run_hybrid_SAC_prefonly_ablation.sh
#   SEEDS="0 1" ./launchers/run_hybrid_SAC_prefonly_ablation.sh
#   DRY_RUN=true ./launchers/run_hybrid_SAC_prefonly_ablation.sh   # print resolved cfg only
# Any extra args are forwarded as Hydra overrides to both conditions.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAUNCHER="$SCRIPT_DIR/run_hybrid_SAC.sh"

SEEDS="${SEEDS:-0}"
OUTPUT_DIR="outputs/hybrid_sac_prefonly_ablation"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-false}"

# Preference hyper-parameters held fixed across both conditions.
PREF_OVERRIDES=(
  algo.kwargs.pref_labels_type=soft
  algo.kwargs.pref_fragment_length=10
  algo.kwargs.pref_temperature=20.0
  algo.kwargs.pref_batch_size=500
  algo.kwargs.queries_per_iteration=1000
)

run_condition() {
  local mode="$1"
  local seed="$2"
  shift 2

  local extra=()
  [[ "$DRY_RUN" == "true" ]] && extra+=(--cfg job)

  MODE="$mode" PYTHON_BIN="$PYTHON_BIN" "$BASE_LAUNCHER" \
    "${PREF_OVERRIDES[@]}" \
    "$@" \
    run.seed="$seed" \
    run.output_dir="$OUTPUT_DIR" \
    "${extra[@]}"
}

for seed in $SEEDS; do
  for mode in hybrid pref_only; do
    echo "============================================================"
    echo "Preference ablation: mode=$mode seed=$seed"
    echo "============================================================"
    run_condition "$mode" "$seed" "$@"
  done
done
