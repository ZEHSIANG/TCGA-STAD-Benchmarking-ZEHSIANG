import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import csv
import datetime
import traceback
from lifelines.utils import concordance_index

import my_models_0527 as my_models
import myCoxLoss_0527
from dataset import TCGADataset, collate_mil, FEAT_DIRS, DEFAULT_CSV_DIR

MASTER_LOG_FILE = '/mnt/data2/TCGA_STAD_Project/0527_experiments/experiment_result/all_experiments_result_0527.csv'
FAILED_LOG_FILE = '/mnt/data2/TCGA_STAD_Project/0527_experiments/experiment_result/failed_experiments_log.csv'

def train_epoch(model, loader, optimizer, criterion, device, accum_steps, mode):
    model.train()
    total_loss = 0
    all_scores, all_events, all_times = [], [], []
    
    optimizer.zero_grad()
    
    for i, batch_data in enumerate(loader):
        time = batch_data['event_time'].to(device)
        
        # 🟢 [關鍵修正] 改用真正的存活狀態 (Dead/Alive)
        event = batch_data['survival_event'].to(device) 
        
        current_bs = time.size(0)
        logits_list = []
        
        if mode == 'WSI':
            wsi_data = batch_data['wsi']
            for j in range(current_bs):
                output = model(wsi_data[j].to(device).unsqueeze(0)) 
                logits_list.append(output)
            logits = torch.cat(logits_list, dim=0)
            
        elif mode == 'Fusion':
            wsi_data = batch_data['wsi']
            clin_data = batch_data['clinical']
            for j in range(current_bs):
                output = model(wsi_data[j].to(device).unsqueeze(0), clin_data[j].to(device).unsqueeze(0))
                logits_list.append(output)
            logits = torch.cat(logits_list, dim=0)
            
        elif mode == 'Clinical':
            logits = model(batch_data['clinical'].to(device))

        loss = criterion(logits, time, event)
        
        if torch.isnan(loss):
            loss = torch.tensor(0.0).to(device)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        all_scores.extend(logits.detach().cpu().numpy().flatten())
        all_events.extend(event.cpu().numpy().flatten())
        all_times.extend(time.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader)
    
    try:
        # C-Index 也要用 survival_event 算才對
        c_index = concordance_index(all_times, -np.array(all_scores), all_events)
    except:
        c_index = 0.5
        
    return avg_loss, c_index

def validate(model, loader, device, mode, criterion):
    model.eval()
    total_loss = 0
    all_scores, all_events, all_times, all_case_ids = [], [], [], []
    
    with torch.no_grad():
        for batch_data in loader:
            time = batch_data['event_time'].to(device)
            # 🟢 [關鍵修正] 改用真正的存活狀態
            event = batch_data['survival_event'].to(device)
            all_case_ids.extend(batch_data['case_id'])
            
            current_bs = time.size(0)
            logits_list = []
            
            if mode == 'WSI':
                wsi_data = batch_data['wsi']
                for j in range(current_bs):
                    output = model(wsi_data[j].to(device).unsqueeze(0))
                    logits_list.append(output)
                logits = torch.cat(logits_list, dim=0)
            elif mode == 'Fusion':
                wsi_data = batch_data['wsi']
                clin_data = batch_data['clinical']
                for j in range(current_bs):
                    output = model(wsi_data[j].to(device).unsqueeze(0), clin_data[j].to(device).unsqueeze(0))
                    logits_list.append(output)
                logits = torch.cat(logits_list, dim=0)
            elif mode == 'Clinical':
                logits = model(batch_data['clinical'].to(device))
            
            loss = criterion(logits, time, event)
            if not torch.isnan(loss):
                total_loss += loss.item()
            
            all_scores.extend(logits.cpu().numpy().flatten())
            all_events.extend(event.cpu().numpy().flatten())
            all_times.extend(time.cpu().numpy().flatten())
            
    avg_loss = total_loss / len(loader)
    
    try:
        c_index = concordance_index(all_times, -np.array(all_scores), all_events)
    except:
        c_index = 0.5
    
    return avg_loss, c_index, all_case_ids, np.array(all_scores), np.array(all_events)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mode', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='ABMIL')
    parser.add_argument('--encoder_name', type=str, default='UNI')
    parser.add_argument('--target_batch_size', type=int, default=32)
    parser.add_argument('--physical_batch_size', type=int, default=16)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--max_patches', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--task_label_mode', type=str, default='original', choices=['original', 'binary'])
    parser.add_argument('--clinical_encoding', type=str, default='label_enc', choices=['label_enc', 'one_hot'])
    parser.add_argument('--csv_dir', type=str, default=None, help='自訂 CSV 路徑，None 則用 DEFAULT_CSV_DIR')
    
    args = parser.parse_args()
    print("="*60)
    print(f"STARTING EXPERIMENT: {datetime.datetime.now()}")
    
    # 🟢 [修正 1] 更穩健的 logic_mode 判斷邏輯
    # 不管你是 Fusion_Smart, Fusion_Naive, Fusion_Binary... 只要有 Fusion 就歸類為 Fusion
    if 'Fusion' in args.input_mode:
        logic_mode = 'Fusion'
    elif 'Clinical' in args.input_mode:
        logic_mode = 'Clinical'
    elif 'WSI' in args.input_mode:
        logic_mode = 'WSI'
    else:
        raise ValueError(f"Unknown input_mode: {args.input_mode}")

    print(f"Logic Mode: {logic_mode} (Original: {args.input_mode})")
    
    # ... (Dataset 初始化部分保持原樣，不用動) ...
    try:
        current_feat_dir = FEAT_DIRS.get(args.encoder_name, FEAT_DIRS['None'])
        
        if args.target_batch_size == -1:
            bs_name = "FullBatch"
            accum_steps = 1 
        else:
            bs_name = str(args.target_batch_size)
            accum_steps = max(1, args.target_batch_size // args.physical_batch_size)

        csv_dir = args.csv_dir if args.csv_dir else DEFAULT_CSV_DIR
        train_set = TCGADataset(csv_dir, current_feat_dir, args.fold, logic_mode, 'train', args.max_patches, 
                                1536 if args.encoder_name in ['UNI', 'Midnight'] else (2560 if args.encoder_name=='Virchow' else 1536),
                                args.task_label_mode, args.clinical_encoding)
        
        val_set = TCGADataset(csv_dir, current_feat_dir, args.fold, logic_mode, 'val', args.max_patches, 
                              train_set.input_dim, args.task_label_mode, args.clinical_encoding)

        train_loader = DataLoader(train_set, batch_size=args.physical_batch_size, shuffle=True, collate_fn=collate_mil)
        val_loader = DataLoader(val_set, batch_size=args.physical_batch_size, shuffle=False, collate_fn=collate_mil)

        if args.target_batch_size == -1: accum_steps = len(train_set) // args.physical_batch_size

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        clin_dim = train_set.get_clin_dim()

        # 🟢 [修正 2] 模型初始化 (這裡邏輯就通了)
        model = None  # 先給個預設值，防呆

        if logic_mode == 'Clinical':
            model = my_models.SNN(input_dim=clin_dim).to(device)
            
        elif logic_mode == 'WSI':
            if args.model_type == 'ABMIL': model = my_models.ABMIL(train_set.input_dim).to(device)
            elif args.model_type == 'TransMIL': model = my_models.TransMIL(train_set.input_dim).to(device)
            elif args.model_type == 'MeanMIL': model = my_models.MeanMIL(train_set.input_dim).to(device)
            elif args.model_type == 'MaxMIL': model = my_models.MaxMIL(train_set.input_dim).to(device)
            
        elif logic_mode == 'Fusion':
            # 設定 Backbone
            backbone = None
            if args.model_type == 'ABMIL': backbone = my_models.ABMIL(train_set.input_dim)
            elif args.model_type == 'TransMIL': backbone = my_models.TransMIL(train_set.input_dim)
            elif args.model_type == 'MeanMIL': backbone = my_models.MeanMIL(train_set.input_dim)
            elif args.model_type == 'MaxMIL': backbone = my_models.MaxMIL(train_set.input_dim)
            
            # 判斷是 Naive 還是 Smart Fusion
            if 'Naive' in args.input_mode: 
                print(f"Build Model: Naive Fusion with {args.model_type}")
                model = my_models.NaiveFusionModel(backbone, clin_dim=clin_dim).to(device)
            else: 
                # 這裡就是 Fusion_Smart 會執行到的地方
                print(f"Build Model: Smart Fusion (SNN) with {args.model_type}")
                model = my_models.FusionModel(backbone, clin_dim=clin_dim).to(device)

        # 🟢 [修正 3] 最後防呆檢查
        if model is None:
            raise RuntimeError(f"Model initialization failed! logic_mode={logic_mode}, input_mode={args.input_mode}")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)    
        criterion = myCoxLoss_0527.CoxPHLoss()
        
        save_path = args.save_dir 
        os.makedirs(save_path, exist_ok=True)
        
        log_csv_path = os.path.join(save_path, "training_log.csv")
        with open(log_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Train C-Index', 'Val Loss', 'Val C-Index'])

        best_c_index = 0
        best_epoch = -1

        for epoch in range(args.epochs):
            train_loss, train_c_index = train_epoch(model, train_loader, optimizer, criterion, device, accum_steps, logic_mode)
            val_loss, val_c_index, val_ids, val_scores, val_events = validate(model, val_loader, device, logic_mode, criterion)
            
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} C-Idx: {train_c_index:.4f} | Val Loss: {val_loss:.4f} C-Idx: {val_c_index:.4f}")
            
            with open(log_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, train_c_index, val_loss, val_c_index])

            if val_c_index > best_c_index:
                best_c_index = val_c_index
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
                pred_df = pd.DataFrame({'case_id': val_ids, 'risk_score': val_scores, 'event': val_events})
                pred_df.to_csv(os.path.join(save_path, "best_predictions.csv"), index=False)
                print(f"  >>> New Best Saved! (C-Index: {best_c_index:.4f})")

        # 無獨立 test set，僅使用 val set 評估（5-Fold CV）

        os.makedirs(os.path.dirname(MASTER_LOG_FILE), exist_ok=True)
        file_exists = os.path.isfile(MASTER_LOG_FILE)
        with open(MASTER_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'Input Mode', 'Model', 'Encoder', 'Fold', 'BatchSize', 'MaxPatches', 'Best Val C-Index', 'Best Epoch', 'Log Path'])
            writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.input_mode, args.model_type, args.encoder_name, args.fold, bs_name, args.max_patches, f"{best_c_index:.4f}", best_epoch, save_path])

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        os.makedirs(os.path.dirname(FAILED_LOG_FILE), exist_ok=True)
        with open(FAILED_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.datetime.now(), args.input_mode, args.model_type, args.fold, str(e)])

if __name__ == '__main__':
    main()
