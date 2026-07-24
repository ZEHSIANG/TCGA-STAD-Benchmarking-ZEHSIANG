import os
import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

# =========================================================
# 1. 基本設定
# =========================================================
DEFAULT_CSV_DIR = '/mnt/data2/TCGA_STAD_Project/0527_experiments/csv/splits/'

FEAT_DIRS = {
    'UNI':     '/mnt/data2/TCGA_STAD_Project/raw_features/TCGA_STAD_features/20x_256px_0px_overlap/features_uni_v2/',
    'Virchow': '/mnt/data2/TCGA_STAD_Project/raw_features/TCGA_STAD_features/20x_224px_0px_overlap/features_virchow2/',
    'Midnight':'/mnt/data2/TCGA_STAD_Project/raw_features/TCGA_STAD_features/20x_224px_0px_overlap/features_midnight12k/',
    'None':    '/mnt/data2/TCGA_STAD_Project/raw_features/TCGA_STAD_features/20x_256px_0px_overlap/features_uni_v2/'
}

class TCGADataset(Dataset):
    def __init__(self, csv_dir, feature_dir, fold=0, mode="wsi", split="train",
                 max_patches=15000, input_dim=1024,
                 task_label_mode="original", clinical_encoding="label_enc"):

        self.csv_dir = csv_dir
        self.feature_dir = feature_dir
        self.mode = mode.lower()
        self.max_patches = max_patches
        self.input_dim = input_dim
        self.task_label_mode = task_label_mode
        self.clinical_encoding = clinical_encoding

        # 讀取 CSV
        csv_path = os.path.join(csv_dir, f"splits_{fold}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到 CSV: {csv_path}")

        df_all = pd.read_csv(csv_path)

        if "split" in df_all.columns:
            self.data = df_all[df_all["split"] == split].reset_index(drop=True)
        else:
            self.data = df_all

        # 處理臨床特徵
        self.clinical_features = self.process_clinical_data(self.data)
        self.clin_dim = self.clinical_features.shape[1]

        if split == "train":
            print(f"[{split.upper()}] ClinEnc: {clinical_encoding} | Dim: {self.clin_dim} | Task: {task_label_mode}")

        id_col = "case_id" if "case_id" in self.data.columns else "bcr_patient_barcode"
        self.clin_dict = {
            cid: feat for cid, feat in zip(self.data[id_col], self.clinical_features)
        }

    def get_clin_dim(self):
        return self.clin_dim

    def process_clinical_data(self, df):
        df_proc = df.copy()

        # =============================================
        # Age：除以 100（固定係數，臨床透明）
        # 範圍約 0.30 ~ 0.88
        # =============================================
        age_col = "age_at_index"
        if age_col not in df_proc.columns:
            age_col = "age_at_initial_pathologic_diagnosis"

        if age_col in df_proc.columns:
            df_proc["age_feat"] = df_proc[age_col].fillna(65.0) / 100.0
        else:
            df_proc["age_feat"] = 0.65

        # =============================================
        # Stage：統一格式後除以4 或 One-Hot
        # =============================================
        stage_col = "stage_clean" if "stage_clean" in df_proc.columns else "ajcc_pathologic_stage"

        def simplify_stage(x):
            s = str(x).lower()
            if "iv" in s: return "stage_iv"
            if "iii" in s: return "stage_iii"
            if "ii" in s: return "stage_ii"
            if "i" in s: return "stage_i"
            return "unknown"

        df_proc["stage_unified"] = df_proc[stage_col].apply(simplify_stage)

        # =============================================
        # Gender：統一格式後 Label 或 One-Hot
        # =============================================
        gender_col = "gender_clean" if "gender_clean" in df_proc.columns else "gender"

        def simplify_gender(x):
            s = str(x).lower()
            if "female" in s: return "female"
            if "male" in s: return "male"
            return "unknown"

        df_proc["gender_unified"] = df_proc[gender_col].apply(simplify_gender)

        # =============================================
        # One-Hot Encoding
        # =============================================
        if self.clinical_encoding == "one_hot":
            stage_cats  = ["stage_i", "stage_ii", "stage_iii", "stage_iv", "unknown"]
            gender_cats = ["female", "male", "unknown"]

            df_proc["stage_unified"]  = pd.Categorical(df_proc["stage_unified"],  categories=stage_cats)
            df_proc["gender_unified"] = pd.Categorical(df_proc["gender_unified"], categories=gender_cats)

            df_stage  = pd.get_dummies(df_proc["stage_unified"],  prefix="stage",  dtype=float)
            df_gender = pd.get_dummies(df_proc["gender_unified"], prefix="gender", dtype=float)
            df_age    = df_proc[["age_feat"]].reset_index(drop=True)

            df_final = pd.concat([df_age, df_gender, df_stage], axis=1)
            return df_final.values.astype(np.float32)

        # =============================================
        # Label Encoding（除以固定係數，不依賴訓練集統計量）
        # =============================================
        else:
            stage_map  = {"stage_iv": 1.00, "stage_iii": 0.75,
                          "stage_ii": 0.50, "stage_i":   0.25, "unknown": 0.0}
            gender_map = {"female": 1.0, "male": 0.0, "unknown": 0.5}

            df_proc["stage_feat"]  = df_proc["stage_unified"].map(stage_map)
            df_proc["gender_feat"] = df_proc["gender_unified"].map(gender_map)

            features = []
            for i in range(len(df_proc)):
                f_vec = [
                    df_proc.iloc[i]["age_feat"],
                    df_proc.iloc[i]["gender_feat"],
                    df_proc.iloc[i]["stage_feat"],
                ]
                features.append(np.array(f_vec, dtype=np.float32))
            return np.array(features, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        slide_id = row.get("slide_id", "unknown")
        case_id  = row.get("case_id",  row.get("bcr_patient_barcode", "unknown"))

        # Survival Event
        os_status      = float(row.get("OS_status", row.get("OS", 0)))
        survival_event = torch.tensor(os_status).float()

        # Time
        t_col      = "OS_time" if "OS_time" in row else "time"
        event_time = torch.tensor(float(row[t_col])).float()
        censorship = torch.tensor(1.0 - os_status).float()

        # Stage label（用於 task_label_mode）
        stage_col = "stage_clean" if "stage_clean" in row else "ajcc_pathologic_stage"
        def get_stage_label(x):
            s = str(x).lower()
            if "iv"  in s: return 3
            if "iii" in s: return 2
            if "ii"  in s: return 1
            if "i"   in s: return 0
            return 0

        raw_label = get_stage_label(row.get(stage_col, "unknown"))

        if self.task_label_mode == "binary":
            label = torch.tensor(0 if raw_label <= 1 else 1).long()
        else:
            label = torch.tensor(raw_label).long()

        # 時間離散化（DeepHit / MTLR 用）
        # 使用等距切法（與 pycox LabTransDiscreteTime 一致）
        # OS_time 範圍：14~3720 天，切成 10 個等距區間
        MAX_TIME = 3720.0
        NUM_BINS = 10
        bin_width = MAX_TIME / NUM_BINS
        time_bin_val = int(float(event_time.item()) / bin_width)
        time_bin_val = max(0, min(time_bin_val, NUM_BINS - 1))

        data_dict = {
            "slide_id":       slide_id,
            "case_id":        case_id,
            "label":          label,
            "survival_event": survival_event,
            "event_time":     event_time,
            "censorship":     censorship,
            "time_bin":       torch.tensor(time_bin_val).long(),
        }

        # WSI Loading
        if self.mode in ["wsi", "fusion"]:
            fname     = row.get("filename", f"{slide_id}.h5")
            if not str(fname).endswith(".h5"):
                fname = str(fname) + ".h5"
            full_path = os.path.join(self.feature_dir, fname)
            try:
                with h5py.File(full_path, "r") as f:
                    if "features" in f: wsi = f["features"][:]
                    elif "uni"     in f: wsi = f["uni"][:]
                    elif "midnight"in f: wsi = f["midnight"][:]
                    else:                wsi = f[list(f.keys())[0]][:]

                    if self.max_patches > 0 and wsi.shape[0] > self.max_patches:
                        idx_ = np.sort(np.random.choice(wsi.shape[0], self.max_patches, replace=False))
                        wsi  = wsi[idx_]
                    data_dict["wsi"] = torch.from_numpy(wsi).float()
            except:
                data_dict["wsi"] = torch.zeros((1, self.input_dim)).float()

        # Clinical Loading
        if self.mode in ["clinical", "fusion"]:
            data_dict["clinical"] = torch.from_numpy(self.clin_dict[case_id]).float()

        return data_dict


def collate_mil(batch):
    res = {
        "label":          torch.stack([item["label"]          for item in batch]),
        "survival_event": torch.stack([item["survival_event"] for item in batch]),
        "event_time":     torch.stack([item["event_time"]     for item in batch]),
        "censorship":     torch.stack([item["censorship"]     for item in batch]),
        "case_id":        [item["case_id"] for item in batch],
        "time_bin":       torch.stack([item["time_bin"]       for item in batch]),
    }
    if "wsi" in batch[0]:
        res["wsi"] = torch.nn.utils.rnn.pad_sequence(
            [i["wsi"] for i in batch], batch_first=True, padding_value=0)
    if "clinical" in batch[0]:
        res["clinical"] = torch.stack([i["clinical"] for i in batch])
    return res
