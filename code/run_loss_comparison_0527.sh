#!/bin/bash

# =================================================================
# Loss Function Comparison 實驗 0527
# 比較 Ranking Loss vs Cox Loss（基準線）
# 固定最佳配置（Patient Level Top 10 + Slide Level Top 10）
# =================================================================

export CUDA_VISIBLE_DEVICES=0
LR=2e-4
EPOCHS=20
BASE_DIR="/mnt/data2/TCGA_STAD_Project/0527_experiments/results"

FOLDS=(0 1 2 3 4)
LOSS="logistic_hazard"

# 20 個最佳配置
declare -a CONFIGS=(
    "Fusion_Naive ABMIL Midnight 1000 32 binary one_hot"
    "Fusion_Naive MeanMIL Midnight 1000 32 original one_hot"
    "Fusion_Naive MeanMIL UNI 4000 -1 binary one_hot"
    "Fusion_Naive ABMIL Midnight 1000 32 original one_hot"
    "Fusion_Naive MeanMIL Midnight 1000 32 original label_enc"
    "Fusion_Naive ABMIL Midnight 1000 32 original label_enc"
    "Fusion_Naive TransMIL Midnight 1000 32 binary one_hot"
    "Fusion_Naive MeanMIL Midnight 4000 -1 binary one_hot"
    "Fusion_Naive MeanMIL UNI 1000 32 binary one_hot"
    "Fusion_Naive ABMIL UNI 1000 32 binary one_hot"
    "Fusion_Smart TransMIL UNI 4000 32 original label_enc"
    "Fusion_Smart TransMIL UNI 4000 -1 original label_enc"
    "Fusion_Smart TransMIL UNI 1000 -1 original label_enc"
    "Fusion_Smart ABMIL UNI 100 -1 original label_enc"
    "Fusion_Smart TransMIL UNI 1000 32 original label_enc"
    "Fusion_Smart ABMIL UNI 100 32 original label_enc"
    "WSI TransMIL UNI 1000 -1 original label_enc"
    "WSI TransMIL UNI 100 -1 original label_enc"
    "Fusion_Smart MeanMIL UNI 100 -1 original label_enc"
    "WSI TransMIL UNI 4000 32 original label_enc"
)

echo "🚀 Loss Function Comparison: Ranking Loss"
echo "總共：20 配置 × 5 fold = 100 組"
echo "=================================================="

for CONFIG in "${CONFIGS[@]}"; do
    read -r INPUT_MODE MODEL ENCODER PATCHES BS TASK CLIN_ENC <<< "$CONFIG"

    if [ "$BS" -eq -1 ]; then BS_NAME="bsFull"; else BS_NAME="bs${BS}"; fi

    # WSI 模式不用 clin_enc
    if [ "$INPUT_MODE" == "WSI" ]; then
        CLIN_ENC_NAME="None"
    else
        CLIN_ENC_NAME="$CLIN_ENC"
    fi

    for FOLD in "${FOLDS[@]}"; do
        EXP_NAME="LossComp_${LOSS}_${INPUT_MODE}_${MODEL}_${ENCODER}_p${PATCHES}_${BS_NAME}_${TASK}_${CLIN_ENC_NAME}_fold${FOLD}"
        SAVE_PATH="${BASE_DIR}/${EXP_NAME}"

        if [ -f "${SAVE_PATH}/training_log.csv" ]; then
            echo "[SKIP] ${EXP_NAME}"
            continue
        fi

        echo "[RUN] ${EXP_NAME}"

        python3 main_train_loss_0527.py \
            --input_mode "$INPUT_MODE" \
            --model_type "$MODEL" \
            --encoder_name "$ENCODER" \
            --max_patches "$PATCHES" \
            --target_batch_size "$BS" \
            --fold "$FOLD" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --task_label_mode "$TASK" \
            --clinical_encoding "$CLIN_ENC" \
            --loss_function "$LOSS" \
            --save_dir "$SAVE_PATH"

    done
done

echo "🎉 Ranking Loss 實驗完成！"
echo "結果儲存於：/mnt/data2/TCGA_STAD_Project/0527_experiments/experiment_result/loss_comparison_result_0527.csv"
