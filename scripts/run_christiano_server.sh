#!/bin/bash

set -euo pipefail

# ── Configurazione Server ──────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="sumo-rlhf"
OUTPUT_DIR="/storage/fis3/debug_reward_model"

# Inizializzazione e attivazione ambiente Conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

# Definiamo le griglie di iperparametri
AGENTS=(PPO SAC)
SEG_LENS=(1 2 10 20 50 100)
USE_REGS=("true" "false")

echo "Avvio della suite di esperimenti sul server..."
echo "Agenti: ${AGENTS[*]}"
echo "Lunghezze segmento: ${SEG_LENS[*]}"
echo "Regolarizzazione: ${USE_REGS[*]}"
echo "Output Directory: $OUTPUT_DIR"

# Ciclo nidificato per esplorare tutte le combinazioni
for AGENT in "${AGENTS[@]}"; do
for SEG_LEN in "${SEG_LENS[@]}"; do
    for USE_REG in "${USE_REGS[@]}"; do

        # 1. Generazione di un nome dinamico e significativo
        if [ "$USE_REG" = "true" ]; then
            REG_LABEL="reg_ON"
        else
            REG_LABEL="reg_OFF"
        fi

        RUN_NAME="${AGENT}_seg_${SEG_LEN}_${REG_LABEL}"

        # 2. Logging per il terminale
        echo "===================================================="
        echo "Inizio Esperimento: $RUN_NAME"
        echo " - Agent: $AGENT"
        echo " - Fragment Length: $SEG_LEN"
        echo " - Reward Regularization: $USE_REG"
        echo "===================================================="

        # 3. Esecuzione (mantenendo taskset e i parametri specifici del server)
        echo "y" | taskset -c 39-47 python scripts/train.py \
            algo=christiano \
            algo/agent="$AGENT" \
            env.kwargs.ego=continuous \
            run.name="$RUN_NAME" \
            run.seed=0 \
            run.output_dir="$OUTPUT_DIR" \
            wandb.enabled=true \
            wandb.kwargs.project="debug-new-server" \
            algo.kwargs.n_ensembles_rew=3 \
            algo.kwargs.lr_rew=3e-4 \
            algo.kwargs.batch_size_rew=128 \
            algo.kwargs.n_ephochs_rew=1 \
            algo.kwargs.n_iterations=20 \
            algo.kwargs.train_comparison_frac=0.8 \
            algo.kwargs.fragment_length="$SEG_LEN" \
            algo.kwargs.transition_oversampling=1.0 \
            algo.kwargs.initial_comparison_frac=0.1 \
            algo.kwargs.initial_epoch_multiplier=1.0 \
            algo.kwargs.use_reward_reg="$USE_REG" \
            algo.kwargs.reward_mean_reg=0.0001 \
            algo.kwargs.label_smoothing=0.1 \
            algo.kwargs.query_schedule="constant" \
            algo.train.kwargs.total_timesteps=400000 \
            algo.train.kwargs.total_comparisons=46000

    done
done
done

echo "===================================================="
echo "Tutti gli esperimenti sul server sono terminati!"
echo "===================================================="