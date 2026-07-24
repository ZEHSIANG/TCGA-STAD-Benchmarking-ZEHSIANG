#!/bin/bash

# =================================================================
# 全因子實驗 0527
# 改善項目：
# 1. Stratified 5-Fold（無獨立test set）
# 2. age 除以100（不用Z-score）
# 3. stage 統一格式除以4
# 4. best_predictions.csv 有 case_id
# 5. Patient-Level 5-Fold，8:2
# =================================================================

export CUDA_VISIBLE_DEVICES=0
LR=2e-4
EPOCHS=20
BASE_DIR="/mnt/data2/TCGA_STAD_Project/0527_experiments/results"

MODELS=("ABMIL" "TransMIL" "MeanMIL")
ENCODERS=("UNI" "Virchow" "Midnight")
PATCHES_LIST=(100 1000 4000)
BATCH_SIZES=(-1 32)
TASKS=("original" "binary")
CLINICAL_ENCS=("label_enc" "one_hot")
FOLDS=(0 1 2 3 4)

echo "🚀 啟動全因子實驗 0527"

# =================================================================
# Part 1: Only Clinical
# =================================================================
echo ">>> [Part 1] Running Only Clinical Baselines..."

for TASK in "${TASKS[@]}"; do
  for CLIN_ENC in "${CLINICAL_ENCS[@]}"; do
    for FOLD in "${FOLDS[@]}"; do

      EXP_NAME="Clinical_SNN_None_p0_bsFull_${TASK}_${CLIN_ENC}_fold${FOLD}"
      SAVE_PATH="${BASE_DIR}/${EXP_NAME}"

      if [ -f "${SAVE_PATH}/training_log.csv" ]; then
        echo "[SKIP] ${EXP_NAME}"
        continue
      fi

      echo "[RUN] ${EXP_NAME}"

      python3 main_train_0527.py \
        --input_mode "Clinical" \
        --model_type "SNN" \
        --encoder_name "None" \
        --max_patches 0 \
        --target_batch_size -1 \
        --fold "$FOLD" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --task_label_mode "$TASK" \
        --clinical_encoding "$CLIN_ENC" \
        --save_dir "$SAVE_PATH"

    done
  done
done

# =================================================================
# Part 2: Only WSI
# =================================================================
echo ">>> [Part 2] Running Only WSI Baselines..."

for TASK in "${TASKS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for ENCODER in "${ENCODERS[@]}"; do
      for PATCHES in "${PATCHES_LIST[@]}"; do
        for BS in "${BATCH_SIZES[@]}"; do
          for FOLD in "${FOLDS[@]}"; do

            if [ "$BS" -eq -1 ]; then BS_NAME="bsFull"; else BS_NAME="bs${BS}"; fi

            EXP_NAME="WSI_${MODEL}_${ENCODER}_p${PATCHES}_${BS_NAME}_${TASK}_None_fold${FOLD}"
            SAVE_PATH="${BASE_DIR}/${EXP_NAME}"

            if [ -f "${SAVE_PATH}/training_log.csv" ]; then
              echo "[SKIP] ${EXP_NAME}"
              continue
            fi

            echo "[RUN] ${EXP_NAME}"

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
              --clinical_encoding "label_enc" \
              --save_dir "$SAVE_PATH"

          done
        done
      done
    done
  done
done

# =================================================================
# Part 3: Fusion (Naive & Smart)
# =================================================================
echo ">>> [Part 3] Running Fusion Experiments..."

FUSION_MODES=("Fusion_Naive" "Fusion_Smart")

for TASK in "${TASKS[@]}"; do
  for CLIN_ENC in "${CLINICAL_ENCS[@]}"; do
    for FUSION in "${FUSION_MODES[@]}"; do
      for MODEL in "${MODELS[@]}"; do
        for ENCODER in "${ENCODERS[@]}"; do
          for PATCHES in "${PATCHES_LIST[@]}"; do
            for BS in "${BATCH_SIZES[@]}"; do
              for FOLD in "${FOLDS[@]}"; do

                if [ "$BS" -eq -1 ]; then BS_NAME="bsFull"; else BS_NAME="bs${BS}"; fi

                EXP_NAME="${FUSION}_${MODEL}_${ENCODER}_p${PATCHES}_${BS_NAME}_${TASK}_${CLIN_ENC}_fold${FOLD}"
                SAVE_PATH="${BASE_DIR}/${EXP_NAME}"

                if [ -f "${SAVE_PATH}/training_log.csv" ]; then
                  echo "[SKIP] ${EXP_NAME}"
                  continue
                fi

                echo "[RUN] ${EXP_NAME}"

                python3 main_train_0527.py \
                  --input_mode "$FUSION" \
                  --model_type "$MODEL" \
                  --encoder_name "$ENCODER" \
                  --max_patches "$PATCHES" \
                  --target_batch_size "$BS" \
                  --fold "$FOLD" \
                  --lr "$LR" \
                  --epochs "$EPOCHS" \
                  --task_label_mode "$TASK" \
                  --clinical_encoding "$CLIN_ENC" \
                  --save_dir "$SAVE_PATH"

              done
            done
          done
        done
      done
    done
  done
done

echo "🎉 所有實驗完成！"
echo "結果儲存於：${BASE_DIR}"
echo "Master log：/mnt/data2/TCGA_STAD_Project/0527_experiments/experiment_result/all_experiments_result_0527.csv"
