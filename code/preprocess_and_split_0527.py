import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

# ================================================================
# 讀取原始資料
# ================================================================
df = pd.read_csv('/mnt/data2/TCGA_STAD_Project/final_version_1231/csv/clean_splits_0107/splits_0.csv')

# ================================================================
# Step 1：清理欄位
# ================================================================
# 保留需要的欄位
keep_cols = [
    'slide_id', 'case_id', 'slide_type', 'filename',
    'ajcc_pathologic_stage', 'OS_time', 'OS_status',
    'gender', 'age_at_index'
]
df = df[keep_cols].copy()

# 清理 stage（統一格式）
def simplify_stage(x):
    s = str(x).lower()
    if 'iv' in s: return 'Stage_IV'
    if 'iii' in s: return 'Stage_III'
    if 'ii' in s: return 'Stage_II'
    if 'i' in s: return 'Stage_I'
    return 'Unknown'

df['stage_clean'] = df['ajcc_pathologic_stage'].apply(simplify_stage)

# 清理 gender
df['gender_clean'] = df['gender'].str.lower().str.strip()

# 確認無缺失值
print('=== 清理後欄位 ===')
print(df[['stage_clean','gender_clean','OS_status','age_at_index']].isnull().sum())
print()
print('stage_clean 分布:')
print(df['stage_clean'].value_counts())
print()
print('gender_clean 分布:')
print(df['gender_clean'].value_counts())
print()
print('OS_status 分布:')
print(df['OS_status'].value_counts())

# ================================================================
# Step 2：Stratified 5-Fold（Patient Level，以 OS_status 做 stratify）
# ================================================================
# 取 patient level（每個病患只算一次）
patient_df = df.groupby('case_id').agg(
    OS_status=('OS_status', 'first'),
    stage_clean=('stage_clean', 'first')
).reset_index()

print(f'\n總病患數: {len(patient_df)}')
print(f'Dead(1): {(patient_df["OS_status"]==1).sum()}, Alive(0): {(patient_df["OS_status"]==0).sum()}')

# Stratified KFold by OS_status
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
output_dir = '/mnt/data2/TCGA_STAD_Project/0527_experiments/csv/splits'
os.makedirs(output_dir, exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(skf.split(patient_df['case_id'], patient_df['OS_status'])):
    train_patients = set(patient_df.iloc[train_idx]['case_id'])
    val_patients = set(patient_df.iloc[val_idx]['case_id'])

    df_fold = df.copy()
    df_fold['split'] = df_fold['case_id'].apply(
        lambda x: 'train' if x in train_patients else 'val'
    )

    # 確認無 data leakage
    overlap = set(df_fold[df_fold['split']=='train']['case_id']) & \
              set(df_fold[df_fold['split']=='val']['case_id'])
    assert len(overlap) == 0

    # 確認 OS_status 分布是否一致（stratify 效果）
    train_dead_rate = df_fold[df_fold['split']=='train']['OS_status'].mean()
    val_dead_rate = df_fold[df_fold['split']=='val']['OS_status'].mean()

    out_path = os.path.join(output_dir, f'splits_{fold}.csv')
    df_fold.to_csv(out_path, index=False)

    print(f'\nFold {fold}:')
    print(f'  Train: {len(train_patients)} patients, {(df_fold["split"]=="train").sum()} slides | Dead rate: {train_dead_rate:.3f}')
    print(f'  Val:   {len(val_patients)} patients, {(df_fold["split"]=="val").sum()} slides | Dead rate: {val_dead_rate:.3f}')
    print(f'  Data Leakage: {"✅ 無重疊" if len(overlap)==0 else "❌ 有重疊"}')

print('\n✅ 完成！儲存至:', output_dir)
