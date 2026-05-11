#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Definiamo le griglie di iperparametri
SEG_LENS=(1)
USE_REGS=("true")

echo "Avvio della suite di esperimenti..."
echo "Lunghezze segmento: ${SEG_LENS[*]}"
echo "Regolarizzazione: ${USE_REGS[*]}"

# Ciclo nidificato per esplorare tutte le combinazioni
for SEG_LEN in "${SEG_LENS[@]}"; do
    for USE_REG in "${USE_REGS[@]}"; do

        # 1. Generazione di un nome dinamico e significativo
        if [ "$USE_REG" = "true" ]; then
            REG_LABEL="reg_ON"
        else
            REG_LABEL="reg_OFF"
        fi

        RUN_NAME="seg_${SEG_LEN}_${REG_LABEL}"

        # 2. Logging per il terminale
        echo "===================================================="
        echo "Inizio Esperimento: $RUN_NAME"
        echo " - Fragment Length: $SEG_LEN"
        echo " - Reward Regularization: $USE_REG"
        echo "===================================================="

        # 3. Esecuzione (mantenendo i parametri Hydra invariati)
        echo "y" | python scripts/train.py \
            algo=christiano \
            algo/agent=PPO \
            env.kwargs.ego=continuous \
            run.name="$RUN_NAME" \
            run.seed=0 \
            run.output_dir="$REPO_ROOT/outputs" \
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
            algo.kwargs.query_schedule="constant" \
            algo.train.kwargs.total_timesteps=400000 \
            algo.train.kwargs.total_comparisons=46000

    done
done

echo "===================================================="
echo "Tutti gli esperimenti sono terminati con successo!"
echo "===================================================="