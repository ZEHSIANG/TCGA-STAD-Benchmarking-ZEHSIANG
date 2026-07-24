#!/bin/bash

# =================================================================
# Sample Efficiency 實驗 0527
# 固定 Val set，改變 Train set 大小（25/50/75/100%）
# 比較不同 FM 在小樣本下的表現
# 固定最佳配置：3 FM × 最佳 MIL
# =================================================================

export CUDA_VISIBLE_DEVICES=0
LR=2e-4
EPOCHS=20
BASE_DIR="/mnt/data2/TCGA_STAD_Project/0527_experiments/results"
FOLDS=(0 1 2 3 4)

# 固定最佳配置（WSI Only，比較 FM 差異）
MODELS=("MeanMIL")
ENCODERS=("UNI" "Virchow" "Midnight")
PATCHES=1000
BS=-1
TASK="original"
CLIN_ENC="label_enc"

echo "🚀 Sample Efficiency 實驗"
echo "固定：MeanMIL, p1000, bsFull, original, WSI Only"
echo "變因：FM（3種）× 樣本比例（4種）× 5 fold = 60 組"
echo "=================================================="

for PCT in 25 50 75 100; do
    if [ "$PCT" -eq 100 ]; then
        CSV_DIR="/mnt/data2/TCGA_STAD_Project/0527_experiments/csv/splits"
    else
        CSV_DIR="/mnt/data2/TCGA_STAD_Project/0527_experiments/csv/splits_sample_${PCT}pct"
    fi

    for ENCODER in "${ENCODERS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            for FOLD in "${FOLDS[@]}"; do

                BS_NAME="bsFull"
                EXP_NAME="SampleEff_${PCT}pct_WSI_${MODEL}_${ENCODER}_p${PATCHES}_${BS_NAME}_${TASK}_fold${FOLD}"
                SAVE_PATH="${BASE_DIR}/${EXP_NAME}"

                if [ -f "${SAVE_PATH}/training_log.csv" ]; then
                    echo "[SKIP] ${EXP_NAME}"
                    continue
                fi

                echo "[RUN] ${EXP_NAME} (CSV: ${PCT}%)"

                python3 main_train_0527.py \
                    --input_mode "WSI" \
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
    done
done

echo "🎉 Sample Efficiency 實驗完成！"
