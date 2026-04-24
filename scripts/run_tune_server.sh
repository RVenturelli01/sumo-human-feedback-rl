#!/bin/bash
# Come usarlo:
#   bash scripts/run_tune_server.sh
#
# Come riprendere dopo un'interruzione:
#   Basta rieseguire lo stesso script. Optuna legge il DB SQLite,
#   conta i trial già completati e aggiunge solo i nuovi richiesti.
#
# Come monitorare in tempo reale:
#   tail -f /storage/fis3/christiano-optuna/tune_<timestamp>.log
#
# Output prodotti:
#   /storage/fis3/christiano-optuna/
#     optuna.db          → DB SQLite con tutti i trial (fonte di verità)
#     pareto_front.json  → Pareto front ordinata per fast_return (aggiornata dopo ogni trial)
#     all_trials.json    → tutti i trial completati con value e params
#     tune_<ts>.log      → log completo stdout+stderr

set -euo pipefail

# ── configurazione ─────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="sumo-rlhf"
OUTPUT_DIR="/storage/fis3/christiano-optuna"
STORAGE="sqlite:///${OUTPUT_DIR}/optuna.db"
STUDY_NAME="christiano-optuna-mo3"
POPULATION_SIZE=30

N_TRIALS=60
TOTAL_TIMESTEPS=1000000
N_EVAL_EPISODES=100
SEED=0

# ── attiva conda ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── installa optuna se mancante ────────────────────────────────────────────────
python -c "import optuna" 2>/dev/null || pip install optuna --quiet

# ── vai alla root del repo ─────────────────────────────────────────────────────
cd "$REPO_ROOT"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  Christiano HPO — server bernstein"
echo "  Study  : $STUDY_NAME"
echo "  Storage: $STORAGE"
echo "  Output : $OUTPUT_DIR"
echo "  Trials : $N_TRIALS × ${TOTAL_TIMESTEPS} ts"
echo "========================================"

# ── lancia il tuning (CPU pinned ai core 36-47) ────────────────────────────────
taskset -c 42-47 python scripts/tune_christiano.py \
    --study-name        "$STUDY_NAME"         \
    --storage           "$STORAGE"            \
    --output-dir        "$OUTPUT_DIR"         \
    --n-trials          "$N_TRIALS"           \
    --population-size   "$POPULATION_SIZE"    \
    --total-timesteps   "$TOTAL_TIMESTEPS"    \
    --n-eval-episodes   "$N_EVAL_EPISODES"    \
    --seed              "$SEED"