import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from lifelines.utils import concordance_index
import pandas as pd
import os

BASE = '/mnt/data2/TCGA_STAD_Project/0527_experiments/results'
OUTPUT_DIR = '/mnt/data2/TCGA_STAD_Project/0527_experiments/figures/paper_figures_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.color': '#cccccc',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

def get_mean_ci(prefix):
    cis = []
    for fold in range(5):
        path = os.path.join(BASE, f'{prefix}_fold{fold}', 'best_predictions.csv')
        try:
            df = pd.read_csv(path)
            if 'case_id' not in df.columns: continue
            df_p = df.groupby('case_id').agg(risk_score=('risk_score','mean'),event=('event','first')).reset_index()
            cis.append(concordance_index(df_p.index, -df_p['risk_score'].values, df_p['event'].values))
        except: pass
    return np.mean(cis) if cis else np.nan

def sig_label(p):
    if np.isnan(p): return 'n.s.'
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'

# ============================================================
# Fig 8: Hyperparameter Sensitivity
# ============================================================
print("Fig 8: Hyperparameter Sensitivity...")

# 收集配對資料
patch_pairs = {100:[], 1000:[], 4000:[]}
bs_pairs = {'bs32':[], 'bsFull':[]}
task_pairs = {'original':[], 'binary':[]}
enc_pairs = {'label_enc':[], 'one_hot':[]}

for enc in ['UNI','Virchow','Midnight']:
    for model in ['ABMIL','TransMIL','MeanMIL']:
        for bs in ['bs32','bsFull']:
            for task in ['original','binary']:
                cis_p = {}
                for p in [100,1000,4000]:
                    ci = get_mean_ci(f'WSI_{model}_{enc}_p{p}_{bs}_{task}_None')
                    if not np.isnan(ci): cis_p[p] = ci
                if len(cis_p)==3:
                    for p in [100,1000,4000]: patch_pairs[p].append(cis_p[p])

        for p in [100,1000,4000]:
            for task in ['original','binary']:
                cis_b = {}
                for bs in ['bs32','bsFull']:
                    ci = get_mean_ci(f'WSI_{model}_{enc}_p{p}_{bs}_{task}_None')
                    if not np.isnan(ci): cis_b[bs] = ci
                if len(cis_b)==2:
                    for bs in ['bs32','bsFull']: bs_pairs[bs].append(cis_b[bs])

        for p in [100,1000,4000]:
            for bs in ['bs32','bsFull']:
                cis_t = {}
                for task in ['original','binary']:
                    ci = get_mean_ci(f'WSI_{model}_{enc}_p{p}_{bs}_{task}_None')
                    if not np.isnan(ci): cis_t[task] = ci
                if len(cis_t)==2:
                    for task in ['original','binary']: task_pairs[task].append(cis_t[task])

for fm in ['UNI','Virchow','Midnight']:
    for model in ['ABMIL','TransMIL','MeanMIL']:
        for p in [100,1000,4000]:
            for bs in ['bs32','bsFull']:
                for task in ['original','binary']:
                    cis_c = {}
                    for clin in ['label_enc','one_hot']:
                        ci = get_mean_ci(f'Fusion_Naive_{model}_{fm}_p{p}_{bs}_{task}_{clin}')
                        if not np.isnan(ci): cis_c[clin] = ci
                    if len(cis_c)==2:
                        for clin in ['label_enc','one_hot']: enc_pairs[clin].append(cis_c[clin])

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

# --- Max Patches ---
ax = axes[0]
data_p = [patch_pairs[p] for p in [100,1000,4000]]
labels_p = ['100', '1,000', '4,000']
means_p = [np.mean(d) for d in data_p]
stds_p = [np.std(d) for d in data_p]
colors_p = ['#90CAF9','#2196F3','#0D47A1']

bars = ax.bar(labels_p, means_p, color=colors_p, alpha=0.85, edgecolor='white', width=0.55)
ax.errorbar(labels_p, means_p, yerr=stds_p, fmt='none',
            ecolor='gray', capsize=5, elinewidth=1.5)
for bar, m in zip(bars, means_p):
    ax.text(bar.get_x()+bar.get_width()/2, m+0.001,
            f'{m:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Wilcoxon 標記
n = min(len(patch_pairs[100]), len(patch_pairs[1000]), len(patch_pairs[4000]))
_, p_100_1000 = wilcoxon(patch_pairs[100][:n], patch_pairs[1000][:n])
_, p_100_4000 = wilcoxon(patch_pairs[100][:n], patch_pairs[4000][:n])
_, p_1000_4000 = wilcoxon(patch_pairs[1000][:n], patch_pairs[4000][:n])
y_top = max(means_p) + max(stds_p) + 0.01

def add_sig_bar(ax, x1, x2, y, p, h=0.005):
    label = sig_label(p)
    color = '#333333' if label != 'n.s.' else '#999999'
    ax.plot([x1,x1,x2,x2],[y,y+h,y+h,y], lw=1.2, color=color)
    ax.text((x1+x2)/2, y+h+0.001, label, ha='center', va='bottom',
            fontsize=9, color=color, fontweight='bold' if label!='n.s.' else 'normal')

add_sig_bar(ax, 0, 1, y_top, p_100_1000)
add_sig_bar(ax, 1, 2, y_top+0.018, p_1000_4000)
add_sig_bar(ax, 0, 2, y_top+0.036, p_100_4000)

ax.set_title('Max Patches', fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Patches per WSI', fontsize=10)
ax.set_ylabel('Patient-Level C-Index', fontsize=10)
ax.set_ylim(0.62, 0.69)

# --- Batch Size ---
ax = axes[1]
data_b = [bs_pairs['bs32'], bs_pairs['bsFull']]
labels_b = ['32', 'Full Batch']
means_b = [np.mean(d) for d in data_b]
stds_b = [np.std(d) for d in data_b]
colors_b = ['#A5D6A7','#2E7D32']

bars = ax.bar(labels_b, means_b, color=colors_b, alpha=0.85, edgecolor='white', width=0.45)
ax.errorbar(labels_b, means_b, yerr=stds_b, fmt='none',
            ecolor='gray', capsize=5, elinewidth=1.5)
for bar, m in zip(bars, means_b):
    ax.text(bar.get_x()+bar.get_width()/2, m+0.001,
            f'{m:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

n = min(len(bs_pairs['bs32']), len(bs_pairs['bsFull']))
_, p_bs = wilcoxon(bs_pairs['bs32'][:n], bs_pairs['bsFull'][:n])
y_top_b = max(means_b) + max(stds_b) + 0.01
add_sig_bar(ax, 0, 1, y_top_b, p_bs)

ax.set_title('Batch Size', fontsize=12, fontweight='bold')
ax.set_xlabel('Batch Size Setting', fontsize=10)
ax.set_ylabel('Patient-Level C-Index', fontsize=10)
ax.set_ylim(0.62, 0.69)

# --- Task Label ---
ax = axes[2]
data_t = [task_pairs['original'], task_pairs['binary']]
labels_t = ['Original\n(4-class)', 'Binary\n(Early/Late)']
means_t = [np.mean(d) for d in data_t]
stds_t = [np.std(d) for d in data_t]
colors_t = ['#FFCC80','#E65100']

bars = ax.bar(labels_t, means_t, color=colors_t, alpha=0.85, edgecolor='white', width=0.45)
ax.errorbar(labels_t, means_t, yerr=stds_t, fmt='none',
            ecolor='gray', capsize=5, elinewidth=1.5)
for bar, m in zip(bars, means_t):
    ax.text(bar.get_x()+bar.get_width()/2, m+0.001,
            f'{m:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

n = min(len(task_pairs['original']), len(task_pairs['binary']))
_, p_task = wilcoxon(task_pairs['original'][:n], task_pairs['binary'][:n])
y_top_t = max(means_t) + max(stds_t) + 0.01
add_sig_bar(ax, 0, 1, y_top_t, p_task)

ax.set_title('Stage Label Mode', fontsize=12, fontweight='bold')
ax.set_xlabel('Task Setting', fontsize=10)
ax.set_ylabel('Patient-Level C-Index', fontsize=10)
ax.set_ylim(0.62, 0.69)

# --- Clinical Encoding ---
ax = axes[3]
data_c = [enc_pairs['label_enc'], enc_pairs['one_hot']]
labels_c = ['Label\nEncoding', 'One-Hot\nEncoding']
means_c = [np.mean(d) for d in data_c]
stds_c = [np.std(d) for d in data_c]
colors_c = ['#CE93D8','#6A1B9A']

bars = ax.bar(labels_c, means_c, color=colors_c, alpha=0.85, edgecolor='white', width=0.45)
ax.errorbar(labels_c, means_c, yerr=stds_c, fmt='none',
            ecolor='gray', capsize=5, elinewidth=1.5)
for bar, m in zip(bars, means_c):
    ax.text(bar.get_x()+bar.get_width()/2, m+0.001,
            f'{m:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

n = min(len(enc_pairs['label_enc']), len(enc_pairs['one_hot']))
_, p_enc = wilcoxon(enc_pairs['label_enc'][:n], enc_pairs['one_hot'][:n])
y_top_c = max(means_c) + max(stds_c) + 0.01
add_sig_bar(ax, 0, 1, y_top_c, p_enc)

ax.set_title('Clinical Encoding\n(Concat Fusion)', fontsize=12, fontweight='bold')
ax.set_xlabel('Encoding Method', fontsize=10)
ax.set_ylabel('Patient-Level C-Index', fontsize=10)
ax.set_ylim(0.62, 0.69)

fig.suptitle('Hyperparameter Sensitivity Analysis\n(Patient-Level C-Index, WSI Only unless noted)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig8_hyperparam_sensitivity.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ Fig 8 完成")

# ============================================================
# Fig 9: Loss Function Comparison
# ============================================================
print("Fig 9: Loss Function Comparison...")

configs = [
    ('Fusion_Naive','ABMIL','Midnight','1000','bs32','binary','one_hot'),
    ('Fusion_Naive','MeanMIL','Midnight','1000','bs32','original','one_hot'),
    ('Fusion_Naive','MeanMIL','UNI','4000','bsFull','binary','one_hot'),
    ('Fusion_Naive','ABMIL','Midnight','1000','bs32','original','one_hot'),
    ('Fusion_Naive','MeanMIL','Midnight','1000','bs32','original','label_enc'),
    ('Fusion_Naive','ABMIL','Midnight','1000','bs32','original','label_enc'),
    ('Fusion_Naive','MeanMIL','Midnight','1000','bsFull','binary','one_hot'),
    ('Fusion_Naive','MeanMIL','Midnight','4000','bsFull','original','label_enc'),
    ('Fusion_Naive','MeanMIL','Midnight','100','bsFull','binary','label_enc'),
    ('WSI','MeanMIL','Midnight','4000','bs32','original','label_enc'),
    ('Fusion_Smart','MeanMIL','UNI','4000','bsFull','binary','label_enc'),
    ('WSI','MeanMIL','UNI','4000','bsFull','original','label_enc'),
    ('Fusion_Smart','MeanMIL','UNI','4000','bsFull','original','label_enc'),
    ('Fusion_Smart','MeanMIL','UNI','4000','bs32','original','label_enc'),
    ('Fusion_Naive','MeanMIL','UNI','4000','bs32','original','one_hot'),
    ('Fusion_Smart','TransMIL','Midnight','1000','bs32','original','one_hot'),
    ('Fusion_Smart','MeanMIL','UNI','4000','bs32','binary','one_hot'),
    ('Fusion_Naive','MeanMIL','UNI','4000','bsFull','original','one_hot'),
    ('WSI','ABMIL','UNI','4000','bsFull','original','label_enc'),
    ('Fusion_Smart','ABMIL','UNI','4000','bs32','original','label_enc'),
]

loss_data = {'Cox PH': [], 'Ranking': [], 'Logistic\nHazard': []}
loss_paired = {'Cox PH':[], 'Ranking':[], 'Logistic\nHazard':[]}

for cfg in configs:
    im,model,enc,p,bs,task,clin = cfg
    clin_n = 'None' if im=='WSI' else clin
    base = f'{im}_{model}_{enc}_p{p}_{bs}_{task}_{clin_n}'
    cox_ci = get_mean_ci(base)
    rank_ci = get_mean_ci(f'LossComp_ranking_{base}')
    lh_ci = get_mean_ci(f'LossComp_logistic_hazard_{base}')
    if not any(np.isnan(v) for v in [cox_ci, rank_ci, lh_ci]):
        loss_paired['Cox PH'].append(cox_ci)
        loss_paired['Ranking'].append(rank_ci)
        loss_paired['Logistic\nHazard'].append(lh_ci)
    if not np.isnan(cox_ci): loss_data['Cox PH'].append(cox_ci)
    if not np.isnan(rank_ci): loss_data['Ranking'].append(rank_ci)
    if not np.isnan(lh_ci): loss_data['Logistic\nHazard'].append(lh_ci)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左：Box Plot
ax = axes[0]
data_l = [loss_data['Cox PH'], loss_data['Ranking'], loss_data['Logistic\nHazard']]
labels_l = ['Cox PH\nLoss', 'Ranking\nLoss', 'Logistic\nHazard']
colors_l = ['#2196F3','#4CAF50','#FF9800']

bp = ax.boxplot(data_l, labels=labels_l, patch_artist=True,
                medianprops=dict(color='black',linewidth=2.5),
                whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
                flierprops=dict(marker='o',markersize=4,alpha=0.5), widths=0.5)
for patch, color in zip(bp['boxes'], colors_l):
    patch.set_facecolor(color); patch.set_alpha(0.75)

for i, (d, c) in enumerate(zip(data_l, colors_l)):
    jitter = np.random.uniform(-0.1,0.1,len(d))
    ax.scatter(np.ones(len(d))*(i+1)+jitter, d, color=c, alpha=0.4, s=20, zorder=4)

for i, d in enumerate(data_l):
    m = np.mean(d)
    ax.text(i+1, np.min(d)-0.01, f'μ={m:.4f}', ha='center', va='top',
            fontsize=9, fontweight='bold', color=colors_l[i])

n = min(len(loss_paired['Cox PH']), len(loss_paired['Ranking']), len(loss_paired['Logistic\nHazard']))
_, p_cr = wilcoxon(loss_paired['Cox PH'][:n], loss_paired['Ranking'][:n])
_, p_cl = wilcoxon(loss_paired['Cox PH'][:n], loss_paired['Logistic\nHazard'][:n])

y_top = max([max(d) for d in data_l]) + 0.01

def add_sig_bar2(ax, x1, x2, y, p, h=0.006):
    label = sig_label(p)
    color = '#333333' if label != 'n.s.' else '#999999'
    ax.plot([x1,x1,x2,x2],[y,y+h,y+h,y], lw=1.2, color=color)
    ax.text((x1+x2)/2, y+h+0.002, label, ha='center', va='bottom',
            fontsize=9, color=color, fontweight='bold' if label!='n.s.' else 'normal')

add_sig_bar2(ax, 1, 2, y_top, p_cr)
add_sig_bar2(ax, 1, 3, y_top+0.02, p_cl)

ax.set_ylabel('Patient-Level C-Index', fontsize=11)
ax.set_title('Loss Function Comparison\n(Box Plot)', fontsize=11, fontweight='bold')
ax.set_ylim(0.52, 0.80)

# 右：Delta Bar Chart（vs Cox）
ax2 = axes[1]
deltas = {
    'Ranking\nvs Cox': np.mean(loss_paired['Ranking'][:n]) - np.mean(loss_paired['Cox PH'][:n]),
    'Logistic Hazard\nvs Cox': np.mean(loss_paired['Logistic\nHazard'][:n]) - np.mean(loss_paired['Cox PH'][:n]),
}
delta_colors = ['#4CAF50' if v > 0 else '#F44336' for v in deltas.values()]
bars = ax2.bar(list(deltas.keys()), list(deltas.values()),
               color=delta_colors, alpha=0.8, edgecolor='white', width=0.45)
for bar, v in zip(bars, deltas.values()):
    ax2.text(bar.get_x()+bar.get_width()/2,
             v+0.002 if v > 0 else v-0.003,
             f'{v:+.4f}', ha='center',
             va='bottom' if v > 0 else 'top',
             fontsize=11, fontweight='bold')

ax2.axhline(0, color='gray', linewidth=1.5, linestyle='-')
ax2.axhline(0, color='#2196F3', linewidth=2, linestyle='--', alpha=0.5, label='Cox PH baseline')
ax2.set_ylabel('ΔC-Index (vs Cox PH)', fontsize=11)
ax2.set_title('Performance Difference\nvs Cox PH Loss', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_ylim(-0.12, 0.05)

fig.suptitle('Loss Function Comparison\n(20 configurations, Patient-Level C-Index)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig9_loss_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ Fig 9 完成")

print(f"\n全部儲存至：{OUTPUT_DIR}")
