import os
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
import datetime

BASE_DIR = "/mnt/data2/TCGA_STAD_Project/0527_experiments/results"
OUTPUT = "/mnt/data2/TCGA_STAD_Project/0527_experiments/experiment_result/late_ensemble_result_0527.csv"

MODELS = ["ABMIL", "TransMIL", "MeanMIL"]
PATCHES_LIST = [100, 1000, 4000]
BATCH_SIZES = ["bsFull", "bs32"]
TASKS = ["original", "binary"]
CLIN_ENCS = ["label_enc", "one_hot"]
FOLDS = [0, 1, 2, 3, 4]
ENCODERS = ["UNI", "Virchow", "Midnight"]

FM_COMBOS = [
    ["UNI", "Virchow"],
    ["UNI", "Midnight"],
    ["Virchow", "Midnight"],
    ["UNI", "Virchow", "Midnight"],
]

def calc_ci_slide(df):
    try:
        return concordance_index(df.index, -df["risk_score"].values, df["event"].values)
    except:
        return np.nan

def calc_ci_patient(df):
    try:
        df_p = df.groupby("case_id").agg(
            risk_score=("risk_score","mean"),
            event=("event","first")
        ).reset_index()
        return concordance_index(df_p.index, -df_p["risk_score"].values, df_p["event"].values)
    except:
        return np.nan

def load_pred(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "case_id" not in df.columns:
        return None
    return df

def merge_and_ensemble(dfs, names):
    merged = dfs[0][["case_id","risk_score","event"]].copy()
    merged = merged.rename(columns={"risk_score": f"risk_{names[0]}"})
    for i in range(1, len(dfs)):
        tmp = dfs[i][["case_id","risk_score"]].rename(columns={"risk_score": f"risk_{names[i]}"})
        merged = merged.merge(tmp, on="case_id", how="inner")
    risk_cols = [f"risk_{n}" for n in names]
    merged["risk_score"] = merged[risk_cols].mean(axis=1)
    return merged

def run_ensemble(paths_per_fold, names, ensemble_type, combo_name, model, patches, bs, task, clin_enc, results):
    fold_slide, fold_patient = [], []
    for fold in FOLDS:
        dfs = []
        valid = True
        for path in paths_per_fold[fold]:
            df = load_pred(path)
            if df is None:
                valid = False; break
            dfs.append(df)
        if not valid: continue
        merged = merge_and_ensemble(dfs, names)
        fold_slide.append(calc_ci_slide(merged))
        fold_patient.append(calc_ci_patient(merged))
    if len(fold_slide) == 0: return
    results.append({
        "Ensemble Type": ensemble_type,
        "Encoder Combo": combo_name,
        "Model": model,
        "MaxPatches": patches,
        "BatchSize": bs,
        "Task": task,
        "ClinEnc": clin_enc,
        "Valid Folds": len(fold_slide),
        "Slide Mean": round(np.mean(fold_slide), 4),
        "Slide Std": round(np.std(fold_slide), 4),
        "Patient Mean": round(np.mean(fold_patient), 4),
        "Patient Std": round(np.std(fold_patient), 4),
    })

results = []

# ================================================================
# Part 1: WSI Ensemble
# ================================================================
print(">>> Part 1: WSI Ensemble")
for combo in FM_COMBOS:
    combo_name = "+".join(combo)
    for model in MODELS:
        for patches in PATCHES_LIST:
            for bs in BATCH_SIZES:
                for task in TASKS:
                    paths = {fold: [
                        os.path.join(BASE_DIR, f"WSI_{model}_{enc}_p{patches}_{bs}_{task}_None_fold{fold}", "best_predictions.csv")
                        for enc in combo
                    ] for fold in FOLDS}
                    run_ensemble(paths, combo, "WSI", combo_name, model, patches, bs, task, "N/A", results)

print(f"  完成：{len(results)} 組")

# ================================================================
# Part 2: Fusion_Smart Ensemble
# ================================================================
print(">>> Part 2: Fusion_Smart Ensemble")
c2 = len(results)
for combo in FM_COMBOS:
    combo_name = "+".join(combo)
    for model in MODELS:
        for patches in PATCHES_LIST:
            for bs in BATCH_SIZES:
                for task in TASKS:
                    for clin_enc in CLIN_ENCS:
                        paths = {fold: [
                            os.path.join(BASE_DIR, f"Fusion_Smart_{model}_{enc}_p{patches}_{bs}_{task}_{clin_enc}_fold{fold}", "best_predictions.csv")
                            for enc in combo
                        ] for fold in FOLDS}
                        run_ensemble(paths, combo, "Fusion_Smart", combo_name, model, patches, bs, task, clin_enc, results)

print(f"  完成：{len(results)-c2} 組")

# ================================================================
# Part 3: Fusion_Naive Ensemble
# ================================================================
print(">>> Part 3: Fusion_Naive Ensemble")
c3 = len(results)
for combo in FM_COMBOS:
    combo_name = "+".join(combo)
    for model in MODELS:
        for patches in PATCHES_LIST:
            for bs in BATCH_SIZES:
                for task in TASKS:
                    for clin_enc in CLIN_ENCS:
                        paths = {fold: [
                            os.path.join(BASE_DIR, f"Fusion_Naive_{model}_{enc}_p{patches}_{bs}_{task}_{clin_enc}_fold{fold}", "best_predictions.csv")
                            for enc in combo
                        ] for fold in FOLDS}
                        run_ensemble(paths, combo, "Fusion_Naive", combo_name, model, patches, bs, task, clin_enc, results)

print(f"  完成：{len(results)-c3} 組")

# ================================================================
# Part 4: WSI + Clinical Ensemble
# ================================================================
print(">>> Part 4: WSI + Clinical Ensemble")
c4 = len(results)
for combo in FM_COMBOS:
    combo_name = "Clinical+" + "+".join(combo)
    for model in MODELS:
        for patches in PATCHES_LIST:
            for bs in BATCH_SIZES:
                for task in TASKS:
                    for clin_enc in CLIN_ENCS:
                        paths = {fold: (
                            [os.path.join(BASE_DIR, f"Clinical_SNN_None_p0_bsFull_{task}_{clin_enc}_fold{fold}", "best_predictions.csv")] +
                            [os.path.join(BASE_DIR, f"WSI_{model}_{enc}_p{patches}_{bs}_{task}_None_fold{fold}", "best_predictions.csv") for enc in combo]
                        ) for fold in FOLDS}
                        run_ensemble(paths, ["Clinical"]+combo, "WSI+Clinical", combo_name, model, patches, bs, task, clin_enc, results)

print(f"  完成：{len(results)-c4} 組")

# ================================================================
# Part 5: Cross-Mode（WSI + Fusion_Smart）
# ================================================================
print(">>> Part 5: Cross-Mode WSI + Fusion_Smart")
c5 = len(results)
for encoder in ENCODERS:
    for model in MODELS:
        for patches in PATCHES_LIST:
            for bs in BATCH_SIZES:
                for task in TASKS:
                    for clin_enc in CLIN_ENCS:
                        combo_name = f"WSI({encoder})+FusionSmart({encoder})"
                        paths = {fold: [
                            os.path.join(BASE_DIR, f"WSI_{model}_{encoder}_p{patches}_{bs}_{task}_None_fold{fold}", "best_predictions.csv"),
                            os.path.join(BASE_DIR, f"Fusion_Smart_{model}_{encoder}_p{patches}_{bs}_{task}_{clin_enc}_fold{fold}", "best_predictions.csv"),
                        ] for fold in FOLDS}
                        run_ensemble(paths, [f"WSI_{encoder}", f"FusionSmart_{encoder}"], "WSI+FusionSmart", combo_name, model, patches, bs, task, clin_enc, results)

print(f"  完成：{len(results)-c5} 組")

# ================================================================
# Part 6: Cross-Mode（WSI + Fusion_Naive）
# ================================================================
print(">>> Part 6: Cross-Mode WSI + Fusion_Naive")
c6 = len(results)
for encoder in ENCODERS:
    for model in MODELS:
        for patches in PATCHES_LIST:
            for bs in BATCH_SIZES:
                for task in TASKS:
                    for clin_enc in CLIN_ENCS:
                        combo_name = f"WSI({encoder})+FusionNaive({encoder})"
                        paths = {fold: [
                            os.path.join(BASE_DIR, f"WSI_{model}_{encoder}_p{patches}_{bs}_{task}_None_fold{fold}", "best_predictions.csv"),
                            os.path.join(BASE_DIR, f"Fusion_Naive_{model}_{encoder}_p{patches}_{bs}_{task}_{clin_enc}_fold{fold}", "best_predictions.csv"),
                        ] for fold in FOLDS}
                        run_ensemble(paths, [f"WSI_{encoder}", f"FusionNaive_{encoder}"], "WSI+FusionNaive", combo_name, model, patches, bs, task, clin_enc, results)

print(f"  完成：{len(results)-c6} 組")

# ================================================================
# Part 7: Cross-Mode（Fusion_Smart + Fusion_Naive）
# ================================================================
print(">>> Part 7: Cross-Mode Fusion_Smart + Fusion_Naive")
c7 = len(results)
for encoder in ENCODERS:
    for model in MODELS:
        for patches in PATCHES_LIST:
            for bs in BATCH_SIZES:
                for task in TASKS:
                    for clin_enc in CLIN_ENCS:
                        combo_name = f"FusionSmart({encoder})+FusionNaive({encoder})"
                        paths = {fold: [
                            os.path.join(BASE_DIR, f"Fusion_Smart_{model}_{encoder}_p{patches}_{bs}_{task}_{clin_enc}_fold{fold}", "best_predictions.csv"),
                            os.path.join(BASE_DIR, f"Fusion_Naive_{model}_{encoder}_p{patches}_{bs}_{task}_{clin_enc}_fold{fold}", "best_predictions.csv"),
                        ] for fold in FOLDS}
                        run_ensemble(paths, [f"FusionSmart_{encoder}", f"FusionNaive_{encoder}"], "FusionSmart+FusionNaive", combo_name, model, patches, bs, task, clin_enc, results)

print(f"  完成：{len(results)-c7} 組")

# ================================================================
# 儲存結果
# ================================================================
df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT, index=False)
print(f"\n✅ 總共 {len(df_out)} 組 ensemble 結果")
print(f"儲存至：{OUTPUT}")

print()
print("=== Top 10 Slide Level ===")
print(df_out.sort_values("Slide Mean", ascending=False).head(10)[
    ["Ensemble Type","Encoder Combo","Model","MaxPatches","BatchSize","Task","Slide Mean","Slide Std"]
].to_string(index=False))

print()
print("=== Top 10 Patient Level ===")
print(df_out.sort_values("Patient Mean", ascending=False).head(10)[
    ["Ensemble Type","Encoder Combo","Model","MaxPatches","BatchSize","Task","Patient Mean","Patient Std"]
].to_string(index=False))
