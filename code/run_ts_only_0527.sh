#!/bin/bash

# =================================================================
# TS Only 實驗 0527
# 只使用 Tumor Slide（TS）訓練和評估
# 比較 TS Only vs All Slides 的差異
# 固定 20 個最佳配置（Patient Top 10 + Slide Top 10）
# =================================================================

export CUDA_VISIBLE_DEVICES=0
LR=2e-4
EPOCHS=20
BASE_DIR="/mnt/data2/TCGA_STAD_Project/0527_experiments/results"
CSV_DIR="/mnt/data2/TCGA_STAD_Project/0527_experiments/csv/splits_ts_only"
FOLDS=(0 1 2 3 4)

declare -a CONFIGS=(
    "Fusion_Naive ABMIL Midnight 1000 32 binary one_hot"
    "Fusion_Naive MeanMIL Midnight 1000 32 original one_hot"
    "Fusion_Naive MeanMIL UNI 4000 -1 binary one_hot"
    "Fusion_Naive ABMIL Midnight 1000 32 original one_hot"
    "Fusion_Naive MeanMIL Midnight 1000 32 original label_enc"
    "Fusion_Naive ABMIL Midnight 1000 32 original label_enc"
    "Fusion_Naive MeanMIL Midnight 1000 -1 binary one_hot"
    "Fusion_Naive MeanMIL Midnight 4000 -1 original label_enc"
    "Fusion_Naive MeanMIL Midnight 100 -1 binary label_enc"
    "WSI MeanMIL Midnight 4000 32 original label_enc"
    "Fusion_Smart MeanMIL UNI 4000 -1 binary label_enc"
    "WSI MeanMIL UNI 4000 -1 original label_enc"
    "Fusion_Smart MeanMIL UNI 4000 -1 original label_enc"
    "Fusion_Smart MeanMIL UNI 4000 32 original label_enc"
    "Fusion_Naive MeanMIL UNI 4000 32 original one_hot"
    "Fusion_Smart TransMIL Midnight 1000 32 original one_hot"
    "Fusion_Smart MeanMIL UNI 4000 32 binary one_hot"
    "Fusion_Naive MeanMIL UNI 4000 -1 original one_hot"
    "WSI ABMIL UNI 4000 -1 original label_enc"
    "Fusion_Smart ABMIL UNI 4000 32 original label_enc"
)

echo "🚀 TS Only 實驗"
echo "CSV DIR: ${CSV_DIR}"
echo "總共：20 配置 × 5 fold = 100 組"
echo "=================================================="

for CONFIG in "${CONFIGS[@]}"; do
    read -r INPUT_MODE MODEL ENCODER PATCHES BS TASK CLIN_ENC <<< "$CONFIG"

    if [ "$BS" -eq -1 ]; then BS_NAME="bsFull"; else BS_NAME="bs${BS}"; fi
    if [ "$INPUT_MODE" == "WSI" ]; then CLIN_ENC_NAME="None"; else CLIN_ENC_NAME="$CLIN_ENC"; fi

    for FOLD in "${FOLDS[@]}"; do
        EXP_NAME="TSOnly_${INPUT_MODE}_${MODEL}_${ENCODER}_p${PATCHES}_${BS_NAME}_${TASK}_${CLIN_ENC_NAME}_fold${FOLD}"
        SAVE_PATH="${BASE_DIR}/${EXP_NAME}"

        if [ -f "${SAVE_PATH}/training_log.csv" ]; then
            echo "[SKIP] ${EXP_NAME}"
            continue
        fi

        echo "[RUN] ${EXP_NAME}"

        python3 main_train_0527.py \
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
            --csv_dir "$CSV_DIR" \
            --save_dir "$SAVE_PATH"
    done
done

echo "🎉 TS Only 實驗完成！"
echo "結果儲存於：${BASE_DIR}/TSOnly_*"
