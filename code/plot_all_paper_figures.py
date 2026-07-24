import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy.stats import wilcoxon
import os

# ============================================================
# 設定
# ============================================================
BASE = '/mnt/data2/TCGA_STAD_Project/0527_experiments/results'
OUTPUT_DIR = '/mnt/data2/TCGA_STAD_Project/0527_experiments/figures/paper_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 白底學術風格
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

COLORS = {
    'UNI': '#2196F3',
    'Virchow': '#4CAF50',
    'Midnight': '#FF9800',
    'ABMIL': '#9C27B0',
    'TransMIL': '#F44336',
    'MeanMIL': '#00BCD4',
    'Clinical': '#795548',
    'WSI': '#2196F3',
    'Fusion_Naive': '#4CAF50',
    'Fusion_Smart': '#FF9800',
}

# ============================================================
# Helper Functions
# ============================================================
def get_fold_cis(prefix, n_folds=5):
    cis = []
    for fold in range(n_folds):
        path = os.path.join(BASE, f'{prefix}_fold{fold}', 'best_predictions.csv')
        try:
            df = pd.read_csv(path)
            if 'case_id' not in df.columns:
                continue
            df_p = df.groupby('case_id').agg(
                risk_score=('risk_score','mean'),
                event=('event','first')
            ).reset_index()
            ci = concordance_index(
                df_p.index, -df_p['risk_score'].values, df_p['event'].values
            )
            cis.append(ci)
        except:
            pass
    return cis

def get_mean_ci(prefix):
    cis = get_fold_cis(prefix)
    return (np.mean(cis), np.std(cis), cis) if cis else (np.nan, np.nan, [])

def wilcoxon_test(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return np.nan
    try:
        _, p = wilcoxon(a[:n], b[:n])
        return p
    except:
        return np.nan

def sig_label(p):
    if np.isnan(p): return 'n.s.'
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'

def add_sig_bracket(ax, x1, x2, y, p, h=0.005):
    label = sig_label(p)
    color = '#333333' if label != 'n.s.' else '#999999'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, color=color)
    ax.text((x1+x2)/2, y+h+0.002, label, ha='center', va='bottom',
            fontsize=9, color=color, fontweight='bold' if label!='n.s.' else 'normal')

# 載入實驗結果
df_full = pd.read_csv('/mnt/data2/TCGA_STAD_Project/0527_experiments/experiment_result/all_experiments_result_0527.csv')

# ============================================================
# Figure 1: Input Mode Comparison (Box Plot)
# ============================================================
print("繪製 Figure 1: Input Mode Comparison...")

fig, ax = plt.subplots(figsize=(8, 6))

modes = ['Clinical', 'WSI', 'Fusion_Naive', 'Fusion_Smart']
mode_labels = ['Clinical\nOnly', 'WSI\nOnly', 'Concatenation\nFusion', 'Projection-based\nFusion']
mode_data = []

for mode in modes:
    sub = df_full[df_full['Input Mode'] == mode]
    seen = set()
    all_cis = []
    for _, row in sub.iterrows():
        exp = '_'.join(row['Log Path'].split('/')[-1].split('_')[:-1])
        if exp in seen: continue
        seen.add(exp)
        cis = get_fold_cis(exp)
        all_cis.extend(cis)
    mode_data.append(all_cis)

bp = ax.boxplot(mode_data, labels=mode_labels, patch_artist=True,
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker='o', markersize=4, alpha=0.5))

mode_colors = [COLORS['Clinical'], COLORS['WSI'], COLORS['Fusion_Naive'], COLORS['Fusion_Smart']]
for patch, color in zip(bp['boxes'], mode_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Random (0.5)')
ax.set_ylabel('Patient-Level C-Index (5-Fold Mean)', fontsize=12)
ax.set_xlabel('Input Modality', fontsize=12)
ax.set_title('Figure 1: C-Index by Input Modality', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0.45, 0.80)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_input_mode_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ Figure 1 完成")

# ============================================================
# Figure 2: Foundation Model Comparison (Box Plot)
# ============================================================
print("繪製 Figure 2: Foundation Model Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
input_modes_plot = ['WSI', 'Fusion_Naive', 'Fusion_Smart']
input_labels = ['WSI Only', 'Concatenation Fusion', 'Projection-based Fusion']
encoders = ['UNI', 'Virchow', 'Midnight']

for ax_idx, (mode, mode_label) in enumerate(zip(input_modes_plot, input_labels)):
    ax = axes[ax_idx]
    fm_data = []
    for enc in encoders:
        sub = df_full[(df_full['Input Mode']==mode) & (df_full['Encoder']==enc)]
        seen = set(); all_cis = []
        for _, row in sub.iterrows():
            exp = '_'.join(row['Log Path'].split('/')[-1].split('_')[:-1])
            if exp in seen: continue
            seen.add(exp)
            cis = get_fold_cis(exp)
            all_cis.extend(cis)
        fm_data.append(all_cis)

    bp = ax.boxplot(fm_data, labels=encoders, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))

    for patch, enc in zip(bp['boxes'], encoders):
        patch.set_facecolor(COLORS[enc])
        patch.set_alpha(0.7)

    # Wilcoxon 配對比較
    fm_pairs_data = {enc: [] for enc in encoders}
    for model in ['ABMIL', 'TransMIL', 'MeanMIL']:
        for patches in [100, 1000, 4000]:
            for bs in ['32', 'FullBatch']:
                for task in ['original', 'binary']:
                    cis_enc = {}
                    for enc in encoders:
                        bs_str = f'bs{bs}' if bs != 'FullBatch' else 'bsFull'
                        if mode == 'WSI':
                            name = f'WSI_{model}_{enc}_p{patches}_{bs_str}_{task}_None'
                        elif mode == 'Fusion_Naive':
                            for clin in ['label_enc', 'one_hot']:
                                name = f'Fusion_Naive_{model}_{enc}_p{patches}_{bs_str}_{task}_{clin}'
                                ci_val, _, _ = get_mean_ci(name)
                                if not np.isnan(ci_val):
                                    cis_enc[enc] = ci_val
                                    break
                            continue
                        else:
                            for clin in ['label_enc', 'one_hot']:
                                name = f'Fusion_Smart_{model}_{enc}_p{patches}_{bs_str}_{task}_{clin}'
                                ci_val, _, _ = get_mean_ci(name)
                                if not np.isnan(ci_val):
                                    cis_enc[enc] = ci_val
                                    break
                            continue
                        ci_val, _, _ = get_mean_ci(name)
                        if not np.isnan(ci_val):
                            cis_enc[enc] = ci_val
                    if len(cis_enc) == 3:
                        for enc in encoders:
                            fm_pairs_data[enc].append(cis_enc[enc])

    # 加 Wilcoxon 標記
    y_max = max([max(d) if d else 0 for d in fm_data]) + 0.01
    pairs = [(0, 1, 'UNI', 'Virchow'), (1, 2, 'Virchow', 'Midnight'), (0, 2, 'UNI', 'Midnight')]
    y_offsets = [y_max, y_max+0.025, y_max+0.05]
    for (x1, x2, a, b), y_off in zip(pairs, y_offsets):
        if fm_pairs_data[a] and fm_pairs_data[b]:
            p = wilcoxon_test(fm_pairs_data[a], fm_pairs_data[b])
            add_sig_bracket(ax, x1+1, x2+1, y_off, p)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title(mode_label, fontsize=11, fontweight='bold')
    ax.set_xlabel('Foundation Model', fontsize=10)
    if ax_idx == 0:
        ax.set_ylabel('Patient-Level C-Index', fontsize=11)
    ax.set_ylim(0.45, 0.85)

fig.suptitle('Figure 2: Foundation Model Comparison across Input Modalities',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_fm_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ Figure 2 完成")

# ============================================================
# Figure 3: MIL Architecture Comparison (Box Plot)
# ============================================================
print("繪製 Figure 3: MIL Architecture Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
mil_models = ['ABMIL', 'TransMIL', 'MeanMIL']

for ax_idx, (mode, mode_label) in enumerate(zip(input_modes_plot, input_labels)):
    ax = axes[ax_idx]
    mil_data = []
    for model in mil_models:
        sub = df_full[(df_full['Input Mode']==mode) & (df_full['Model']==model)]
        seen = set(); all_cis = []
        for _, row in sub.iterrows():
            exp = '_'.join(row['Log Path'].split('/')[-1].split('_')[:-1])
            if exp in seen: continue
            seen.add(exp)
            cis = get_fold_cis(exp)
            all_cis.extend(cis)
        mil_data.append(all_cis)

    bp = ax.boxplot(mil_data, labels=mil_models, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))

    for patch, model in zip(bp['boxes'], mil_models):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.7)

    # Wilcoxon
    mil_pairs_data = {m: [] for m in mil_models}
    for enc in ['UNI', 'Virchow', 'Midnight']:
        for patches in [100, 1000, 4000]:
            for bs in ['32', 'FullBatch']:
                for task in ['original', 'binary']:
                    cis_model = {}
                    for model in mil_models:
                        bs_str = f'bs{bs}' if bs != 'FullBatch' else 'bsFull'
                        if mode == 'WSI':
                            name = f'WSI_{model}_{enc}_p{patches}_{bs_str}_{task}_None'
                            ci_val, _, _ = get_mean_ci(name)
                            if not np.isnan(ci_val):
                                cis_model[model] = ci_val
                        else:
                            for clin in ['label_enc', 'one_hot']:
                                prefix = 'Fusion_Naive' if mode == 'Fusion_Naive' else 'Fusion_Smart'
                                name = f'{prefix}_{model}_{enc}_p{patches}_{bs_str}_{task}_{clin}'
                                ci_val, _, _ = get_mean_ci(name)
                                if not np.isnan(ci_val):
                                    cis_model[model] = ci_val
                                    break
                    if len(cis_model) == 3:
                        for model in mil_models:
                            mil_pairs_data[model].append(cis_model[model])

    y_max = max([max(d) if d else 0 for d in mil_data]) + 0.01
    pairs = [(0,1,'ABMIL','TransMIL'), (1,2,'TransMIL','MeanMIL'), (0,2,'ABMIL','MeanMIL')]
    y_offsets = [y_max, y_max+0.025, y_max+0.05]
    for (x1, x2, a, b), y_off in zip(pairs, y_offsets):
        if mil_pairs_data[a] and mil_pairs_data[b]:
            p = wilcoxon_test(mil_pairs_data[a], mil_pairs_data[b])
            add_sig_bracket(ax, x1+1, x2+1, y_off, p)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title(mode_label, fontsize=11, fontweight='bold')
    ax.set_xlabel('MIL Architecture', fontsize=10)
    if ax_idx == 0:
        ax.set_ylabel('Patient-Level C-Index', fontsize=11)
    ax.set_ylim(0.45, 0.85)

fig.suptitle('Figure 3: MIL Architecture Comparison across Input Modalities',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_mil_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ Figure 3 完成")

# ============================================================
# Figure 4: Sample Efficiency
# ============================================================
print("繪製 Figure 4: Sample Efficiency...")

fig, ax = plt.subplots(figsize=(8, 6))
pcts = [25, 50, 75, 100]
patient_counts = [76, 152, 228, 304]
fm_results = {
    'UNI':     {'means':[], 'stds':[]},
    'Virchow': {'means':[], 'stds':[]},
    'Midnight':{'means':[], 'stds':[]},
}

for enc in ['UNI', 'Virchow', 'Midnight']:
    for pct in pcts:
        name = f'SampleEff_{pct}pct_WSI_MeanMIL_{enc}_p1000_bsFull_original'
        mean, std, _ = get_mean_ci(name)
        fm_results[enc]['means'].append(mean)
        fm_results[enc]['stds'].append(std)

for enc in ['UNI', 'Virchow', 'Midnight']:
    means = fm_results[enc]['means']
    stds = fm_results[enc]['stds']
    ax.plot(pcts, means, marker='o', color=COLORS[enc],
            linewidth=2.5, markersize=8, label=enc, zorder=5)
    ax.fill_between(pcts,
                    [m-s for m,s in zip(means,stds)],
                    [m+s for m,s in zip(means,stds)],
                    alpha=0.15, color=COLORS[enc])

ax.axhline(0.6, color='gray', linestyle=':', alpha=0.6, linewidth=1)
ax.set_xlabel('Training Data Proportion (%)', fontsize=12)
ax.set_ylabel('Patient-Level C-Index', fontsize=12)
ax.set_title('Figure 4: Sample Efficiency across Foundation Models\n(WSI Only, MeanMIL, p=1000)',
             fontsize=13, fontweight='bold')
ax.set_xticks(pcts)
ax.set_xticklabels([f'{p}%\n({n} pts)' for p,n in zip(pcts, patient_counts)])
ax.legend(title='Foundation Model', fontsize=10, title_fontsize=10)
ax.set_ylim(0.53, 0.72)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_sample_efficiency.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ Figure 4 完成")

# ============================================================
# Figure 5: TS Only vs All Slides
# ============================================================
print("繪製 Figure 5: Diagnostic Slides Only vs All Slides...")

configs_20 = [
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

all_cis_list = []
ts_cis_list = []

for cfg in configs_20:
    im, model, enc, p, bs, task, clin = cfg
    clin_name = 'None' if im == 'WSI' else clin
    base = f'{im}_{model}_{enc}_p{p}_{bs}_{task}_{clin_name}'
    ts_base = f'TSOnly_{im}_{model}_{enc}_p{p}_{bs}_{task}_{clin_name}'
    a, _, _ = get_mean_ci(base)
    t, _, _ = get_mean_ci(ts_base)
    if not np.isnan(a) and not np.isnan(t):
        all_cis_list.append(a)
        ts_cis_list.append(t)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左：散點圖
ax = axes[0]
ax.scatter(all_cis_list, ts_cis_list, color='#F44336', s=80,
           alpha=0.8, edgecolors='white', linewidth=0.8, zorder=5)
lim = [min(min(all_cis_list), min(ts_cis_list))-0.01,
       max(max(all_cis_list), max(ts_cis_list))+0.01]
ax.plot(lim, lim, '--', color='gray', linewidth=1.5, alpha=0.7, label='y=x (no change)')
ax.fill_between(lim, [lim[0]]*2, lim, alpha=0.05, color='#4CAF50')
ax.fill_between(lim, lim, [lim[1]]*2, alpha=0.05, color='#F44336')
ax.text(0.62, 0.70, 'All Slides better', color='#F44336', fontsize=9, alpha=0.8)
mean_diff = np.mean([t-a for t,a in zip(ts_cis_list, all_cis_list)])
ax.text(0.03, 0.97,
        f'n = {len(all_cis_list)}/20 configs\nAll better: {len(all_cis_list)}/20\nMean Δ = {mean_diff:+.4f}',
        transform=ax.transAxes, color='#F44336', fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='#fff3f3', alpha=0.8))
ax.set_xlabel('All Slides C-Index', fontsize=11)
ax.set_ylabel('Diagnostic Slides Only C-Index', fontsize=11)
ax.set_title('Scatter: All Slides vs Diagnostic Slides Only', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

# 右：Paired Box Plot
ax2 = axes[1]
bp = ax2.boxplot([all_cis_list, ts_cis_list],
                 labels=['All Slides\n(BS + TS)', 'Diagnostic\nSlides Only (TS)'],
                 patch_artist=True,
                 medianprops=dict(color='black', linewidth=2),
                 whiskerprops=dict(linewidth=1.2),
                 capprops=dict(linewidth=1.2),
                 flierprops=dict(marker='o', markersize=4, alpha=0.5))
bp['boxes'][0].set_facecolor('#2196F3'); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('#FF9800'); bp['boxes'][1].set_alpha(0.7)

# 配對線
for a, t in zip(all_cis_list, ts_cis_list):
    ax2.plot([1, 2], [a, t], color='gray', alpha=0.3, linewidth=0.8)

_, p_val = wilcoxon(all_cis_list, ts_cis_list)
add_sig_bracket(ax2, 1, 2, max(max(all_cis_list), max(ts_cis_list))+0.005, p_val)

ax2.set_ylabel('Patient-Level C-Index', fontsize=11)
ax2.set_title('Paired Comparison (20 configs)', fontsize=11, fontweight='bold')
ax2.set_ylim(0.55, 0.80)

fig.suptitle('Figure 5: Impact of Slide Type Selection\n(Diagnostic Slides Only vs All Slides)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_slide_type_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ Figure 5 完成")

# ============================================================
# Figure 6: Late Fusion Top Results
# ============================================================
print("繪製 Figure 6: Late Fusion Comparison...")

df_late = pd.read_csv('/mnt/data2/TCGA_STAD_Project/0527_experiments/experiment_result/late_ensemble_result_0527.csv')

# 各 Ensemble Type 的分布
fig, ax = plt.subplots(figsize=(10, 6))
ensemble_types = df_late['Ensemble Type'].unique()
ensemble_data = []
ensemble_labels = []

type_map = {
    'WSI': 'WSI\nEnsemble',
    'Fusion_Naive': 'Concat\nFusion Ensemble',
    'Fusion_Smart': 'Projection\nFusion Ensemble',
    'WSI+Clinical': 'WSI+Clinical\nEnsemble',
    'WSI+FusionSmart': 'WSI+\nFusionSmart',
    'WSI+FusionNaive': 'WSI+\nFusionNaive',
    'FusionSmart+FusionNaive': 'FusionSmart+\nFusionNaive',
}

ens_colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336','#00BCD4','#795548']
valid_types = []
for et in ['WSI','Fusion_Naive','Fusion_Smart','WSI+Clinical','WSI+FusionSmart','WSI+FusionNaive','FusionSmart+FusionNaive']:
    sub = df_late[df_late['Ensemble Type']==et]
    if len(sub) > 0:
        ensemble_data.append(sub['Patient Mean'].values)
        ensemble_labels.append(type_map.get(et, et))
        valid_types.append(et)

bp = ax.boxplot(ensemble_data, labels=ensemble_labels, patch_artist=True,
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker='o', markersize=3, alpha=0.4))

for patch, color in zip(bp['boxes'], ens_colors[:len(ensemble_data)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 標記單一最佳模型基準線
best_single = 0.7021
ax.axhline(best_single, color='#F44336', linestyle='--', linewidth=1.5,
           label=f'Best Single Model ({best_single:.4f})')

ax.set_ylabel('Patient-Level C-Index', fontsize=11)
ax.set_xlabel('Ensemble Strategy', fontsize=11)
ax.set_title('Figure 6: Late Fusion Strategy Comparison\n(Patient-Level C-Index Distribution)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0.50, 0.80)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_late_fusion_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ Figure 6 完成")

# ============================================================
# Figure 7: Summary - Best C-Index per Strategy
# ============================================================
print("繪製 Figure 7: Overall Summary...")

fig, ax = plt.subplots(figsize=(10, 6))

strategies = [
    ('Clinical Only', 0.5707, 0.0483, '#795548'),
    ('WSI Only', 0.6913, 0.0367, '#2196F3'),
    ('Concat Fusion', 0.7021, 0.0387, '#4CAF50'),
    ('Projection Fusion', 0.6888, 0.0408, '#FF9800'),
    ('Late Fusion\n(Ensemble)', 0.7010, 0.0592, '#9C27B0'),
]

names = [s[0] for s in strategies]
means = [s[1] for s in strategies]
stds = [s[2] for s in strategies]
colors_list = [s[3] for s in strategies]

bars = ax.barh(names, means, xerr=stds, color=colors_list, alpha=0.8,
               edgecolor='white', linewidth=0.5,
               error_kw=dict(ecolor='gray', capsize=4, elinewidth=1.5))

for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(mean + std + 0.003, i, f'{mean:.4f}', va='center', fontsize=10, fontweight='bold')

ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Random (0.5)')
ax.set_xlabel('Best Patient-Level C-Index (Mean ± SD)', fontsize=11)
ax.set_title('Figure 7: Best Performance per Fusion Strategy\n(TCGA-STAD, 381 patients)',
             fontsize=13, fontweight='bold')
ax.set_xlim(0.45, 0.80)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_summary_best_per_strategy.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ Figure 7 完成")

print()
print(f"全部圖表已儲存至：{OUTPUT_DIR}")
print("圖表列表：")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png'):
        print(f"  {f}")
