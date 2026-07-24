import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
import os, shutil

BASE = '/mnt/data2/TCGA_STAD_Project/0527_experiments/results'
OUTPUT = '/mnt/data2/TCGA_STAD_Project/0527_experiments/figures/paper_figures_v2'

def get_mean_ci(prefix):
    cis = []
    for fold in range(5):
        path = os.path.join(BASE, f'{prefix}_fold{fold}', 'best_predictions.csv')
        try:
            df = pd.read_csv(path)
            if 'case_id' not in df.columns: continue
            df_p = df.groupby('case_id').agg(
                risk_score=('risk_score','mean'),
                event=('event','first')
            ).reset_index()
            cis.append(concordance_index(
                df_p.index, -df_p['risk_score'].values, df_p['event'].values))
        except: pass
    return np.mean(cis) if cis else np.nan

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

encoders = ['UNI', 'Virchow2', 'Midnight']
enc_keys = ['UNI', 'Virchow', 'Midnight']
models = ['ABMIL', 'TransMIL', 'MeanMIL']
input_modes = ['WSI', 'Fusion_Naive', 'Fusion_Smart']
mode_labels = ['WSI Only', 'Concat Fusion', 'Projection Fusion']

# 收集所有超參數組合的平均（marginalize 超參數）
def get_avg_ci(mode, enc_key, model):
    cis = []
    for p in [100, 1000, 4000]:
        for bs in ['bs32', 'bsFull']:
            for task in ['original', 'binary']:
                if mode == 'WSI':
                    name = f'WSI_{model}_{enc_key}_p{p}_{bs}_{task}_None'
                    ci = get_mean_ci(name)
                    if not np.isnan(ci): cis.append(ci)
                else:
                    prefix = 'Fusion_Naive' if mode == 'Fusion_Naive' else 'Fusion_Smart'
                    for clin in ['label_enc', 'one_hot']:
                        name = f'{prefix}_{model}_{enc_key}_p{p}_{bs}_{task}_{clin}'
                        ci = get_mean_ci(name)
                        if not np.isnan(ci): cis.append(ci)
    return np.mean(cis) if cis else np.nan

# ============================================================
# Fig 11：全因子 Heatmap（藍色系，邊際化超參數）
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 藍色系 colormap
blue_cmap = plt.cm.Blues

for ax_idx, (mode, mode_label) in enumerate(zip(input_modes, mode_labels)):
    ax = axes[ax_idx]
    matrix = np.zeros((len(encoders), len(models)))

    for i, (enc, enc_key) in enumerate(zip(encoders, enc_keys)):
        for j, model in enumerate(models):
            matrix[i, j] = get_avg_ci(mode, enc_key, model)

    vmin = 0.60
    vmax = 0.71
    im = ax.imshow(matrix, cmap=blue_cmap, vmin=vmin, vmax=vmax, aspect='auto')

    # 數值標注
    global_max = np.nanmax(matrix)
    for i in range(len(encoders)):
        for j in range(len(models)):
            val = matrix[i, j]
            # 顏色：深色背景用白字，淺色用黑字
            text_color = 'white' if val > (vmin + (vmax-vmin)*0.6) else 'black'
            weight = 'bold' if val == global_max else 'normal'
            ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                    fontsize=11, fontweight=weight, color=text_color)

    # 框出每個子圖最高值
    max_positions = np.argwhere(matrix == global_max)
    for pos in max_positions:
        ax.add_patch(plt.Rectangle(
            (pos[1]-0.5, pos[0]-0.5), 1, 1,
            fill=False, edgecolor='#FF6B00', linewidth=3, zorder=5
        ))

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(encoders)))
    ax.set_yticklabels(encoders, fontsize=11, fontweight='bold')
    ax.set_title(mode_label, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('MIL Architecture', fontsize=10)
    if ax_idx == 0:
        ax.set_ylabel('Foundation Model', fontsize=11)

    # 分隔線
    for x in [0.5, 1.5]:
        ax.axvline(x, color='white', linewidth=2)
    for y in [0.5, 1.5]:
        ax.axhline(y, color='white', linewidth=2)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Patient C-Index', fontsize=9)
    cbar.ax.tick_params(labelsize=9)

fig.suptitle('Full-Grid C-Index Heatmap: FM × MIL Architecture\n'
             '(Marginalized over Max Patches, Batch Size, Task Label)',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
out = os.path.join(OUTPUT, 'fig11_fullgrid_heatmap_v2.png')
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
shutil.copy(out, os.path.expanduser('~/fig11_fullgrid_heatmap_v2.png'))
print(f'Fig11 完成：{out}')

# ============================================================
# Fig 4 重做：Sample Efficiency（仿照 paper 格式）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

pcts = [25, 50, 75, 100]
pts = [76, 152, 228, 304]
fm_colors = {'UNI': '#1565C0', 'Virchow': '#1976D2', 'Midnight': '#42A5F5'}
fm_markers = {'UNI': 'o', 'Virchow': 's', 'Midnight': '^'}

fm_data = {}
for enc_key in ['UNI', 'Virchow', 'Midnight']:
    means, stds = [], []
    for pct in pcts:
        name = f'SampleEff_{pct}pct_WSI_MeanMIL_{enc_key}_p1000_bsFull_original'
        m, s, _ = (lambda cis: (np.mean(cis), np.std(cis), cis) if cis else (np.nan, np.nan, []))(
            [concordance_index(
                pd.read_csv(f'{BASE}/{name}_fold{fold}/best_predictions.csv')
                .groupby('case_id').agg(risk_score=('risk_score','mean'),event=('event','first'))
                .reset_index().pipe(lambda df: (df.index, -df['risk_score'].values, df['event'].values))
                [0],
                pd.read_csv(f'{BASE}/{name}_fold{fold}/best_predictions.csv')
                .groupby('case_id').agg(risk_score=('risk_score','mean'),event=('event','first'))
                .reset_index()['risk_score'].values * -1,
                pd.read_csv(f'{BASE}/{name}_fold{fold}/best_predictions.csv')
                .groupby('case_id').agg(risk_score=('risk_score','mean'),event=('event','first'))
                .reset_index()['event'].values
            ) for fold in range(5)
            if os.path.exists(f'{BASE}/{name}_fold{fold}/best_predictions.csv')]
        )
        means.append(m); stds.append(s)
    fm_data[enc_key] = (means, stds)

# 左圖：Line plot with error bars（主趨勢）
ax = axes[0]
for enc_key, (means, stds) in fm_data.items():
    label = 'UNI' if enc_key == 'UNI' else ('Virchow2' if enc_key == 'Virchow' else 'Midnight-12k')
    ax.errorbar(range(4), means, yerr=stds,
                marker=fm_markers[enc_key], color=fm_colors[enc_key],
                linewidth=2.5, markersize=9, capsize=5, capthick=2,
                label=label, zorder=5)

ax.axhline(0.6, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.set_xticks(range(4))
ax.set_xticklabels([f'{p}%\n(n={n})' for p, n in zip(pcts, pts)], fontsize=10)
ax.set_ylabel('Patient-Level C-Index', fontsize=11)
ax.set_xlabel('Training Data Proportion', fontsize=11)
ax.set_title('Sample Efficiency\n(Mean ± SD, 5-Fold)', fontsize=11, fontweight='bold')
ax.legend(title='Foundation Model', fontsize=10, title_fontsize=10)
ax.set_ylim(0.53, 0.73)
ax.grid(True, alpha=0.25)

# 右圖：各比例下 FM 的 C-Index 分布（Box plot）
ax2 = axes[1]
pct_labels = [f'{p}%\n(n={n})' for p, n in zip(pcts, pts)]
x = np.arange(len(pcts))
width = 0.25
colors_bar = {'UNI': '#1565C0', 'Virchow': '#42A5F5', 'Midnight': '#90CAF9'}

for k, enc_key in enumerate(['UNI', 'Virchow', 'Midnight']):
    means = fm_data[enc_key][0]
    stds = fm_data[enc_key][1]
    label = 'UNI' if enc_key == 'UNI' else ('Virchow2' if enc_key == 'Virchow' else 'Midnight-12k')
    bars = ax2.bar(x + (k-1)*width, means, width*0.9,
                   yerr=stds, color=colors_bar[enc_key],
                   alpha=0.85, label=label, capsize=4,
                   error_kw=dict(elinewidth=1.5, ecolor='#333333'))

ax2.set_xticks(x)
ax2.set_xticklabels(pct_labels, fontsize=10)
ax2.set_ylabel('Patient-Level C-Index', fontsize=11)
ax2.set_xlabel('Training Data Proportion', fontsize=11)
ax2.set_title('Sample Efficiency by FM\n(Grouped Bar Chart)', fontsize=11, fontweight='bold')
ax2.legend(title='Foundation Model', fontsize=10, title_fontsize=10)
ax2.set_ylim(0.53, 0.73)
ax2.axhline(0.6, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax2.grid(True, alpha=0.25, axis='y')

fig.suptitle('Sample Efficiency Analysis\n(WSI Only, MeanMIL, Max Patches=1000)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out4 = os.path.join(OUTPUT, 'fig4_sample_efficiency_v2.png')
plt.savefig(out4, dpi=200, bbox_inches='tight', facecolor='white')
shutil.copy(out4, os.path.expanduser('~/fig4_sample_efficiency_v2.png'))
print(f'Fig4 完成：{out4}')
