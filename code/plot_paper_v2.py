import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy.stats import wilcoxon
import os

BASE = '/mnt/data2/TCGA_STAD_Project/0527_experiments/results'
OUTPUT_DIR = '/mnt/data2/TCGA_STAD_Project/0527_experiments/figures/paper_figures_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    'UNI': '#2196F3', 'Virchow': '#4CAF50', 'Midnight': '#FF9800',
    'ABMIL': '#9C27B0', 'TransMIL': '#F44336', 'MeanMIL': '#00BCD4',
    'Clinical': '#795548', 'WSI': '#2196F3',
    'Fusion_Naive': '#4CAF50', 'Fusion_Smart': '#FF9800',
}

def get_fold_cis(prefix):
    cis = []
    for fold in range(5):
        path = os.path.join(BASE, f'{prefix}_fold{fold}', 'best_predictions.csv')
        try:
            df = pd.read_csv(path)
            if 'case_id' not in df.columns: continue
            df_p = df.groupby('case_id').agg(
                risk_score=('risk_score','mean'), event=('event','first')
            ).reset_index()
            ci = concordance_index(df_p.index, -df_p['risk_score'].values, df_p['event'].values)
            cis.append(ci)
        except: pass
    return cis

def get_mean_ci(prefix):
    cis = get_fold_cis(prefix)
    return (np.mean(cis), np.std(cis), cis) if cis else (np.nan, np.nan, [])

def wilcoxon_p(a, b):
    n = min(len(a), len(b))
    if n < 2: return np.nan
    try:
        _, p = wilcoxon(a[:n], b[:n])
        return p
    except: return np.nan

def sig_label(p):
    if np.isnan(p): return 'n.s.'
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'

def add_sig(ax, x1, x2, y, p, h=0.008):
    label = sig_label(p)
    color = '#333333' if label != 'n.s.' else '#999999'
    ax.plot([x1,x1,x2,x2],[y,y+h,y+h,y], lw=1.2, color=color)
    ax.text((x1+x2)/2, y+h+0.003, label, ha='center', va='bottom',
            fontsize=9, color=color, fontweight='bold' if label!='n.s.' else 'normal')

df_full = pd.read_csv('/mnt/data2/TCGA_STAD_Project/0527_experiments/experiment_result/all_experiments_result_0527.csv')

# ============================================================
# Fig 1: Input Mode Box Plot
# ============================================================
print("Fig 1...")
fig, ax = plt.subplots(figsize=(7,5))
modes = ['Clinical','WSI','Fusion_Naive','Fusion_Smart']
labels = ['Clinical\nOnly','WSI Only','Concat\nFusion','Projection\nFusion']
data = []
for mode in modes:
    sub = df_full[df_full['Input Mode']==mode]
    seen = set(); cis = []
    for _, row in sub.iterrows():
        exp = '_'.join(row['Log Path'].split('/')[-1].split('_')[:-1])
        if exp in seen: continue
        seen.add(exp)
        cis.extend(get_fold_cis(exp))
    data.append(cis)

bp = ax.boxplot(data, labels=labels, patch_artist=True,
                medianprops=dict(color='black',linewidth=2),
                whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
                flierprops=dict(marker='o',markersize=3,alpha=0.4))
for patch, mode in zip(bp['boxes'], modes):
    patch.set_facecolor(COLORS[mode]); patch.set_alpha(0.75)

ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.set_ylabel('Patient-Level C-Index', fontsize=11)
ax.set_xlabel('Input Modality', fontsize=11)
ax.set_title('C-Index by Input Modality', fontsize=12, fontweight='bold')
ax.set_ylim(0.44, 0.82)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig1_input_mode.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅")

# ============================================================
# Fig 2: FM Comparison (3 subplots, fixed label size)
# ============================================================
print("Fig 2...")
fig, axes = plt.subplots(1,3, figsize=(13,5), sharey=True)
input_modes = ['WSI','Fusion_Naive','Fusion_Smart']
input_labels_short = ['WSI Only','Concat Fusion','Projection Fusion']
encoders = ['UNI','Virchow','Midnight']

for ax_idx, (mode, mlabel) in enumerate(zip(input_modes, input_labels_short)):
    ax = axes[ax_idx]
    fm_data = []
    fm_paired = {e:[] for e in encoders}

    for enc in encoders:
        sub = df_full[(df_full['Input Mode']==mode)&(df_full['Encoder']==enc)]
        seen = set(); cis = []
        for _, row in sub.iterrows():
            exp = '_'.join(row['Log Path'].split('/')[-1].split('_')[:-1])
            if exp in seen: continue
            seen.add(exp)
            cis.extend(get_fold_cis(exp))
        fm_data.append(cis)

    # 配對
    for model in ['ABMIL','TransMIL','MeanMIL']:
        for patches in [100,1000,4000]:
            for bs in ['bs32','bsFull']:
                for task in ['original','binary']:
                    cis_enc = {}
                    for enc in encoders:
                        if mode == 'WSI':
                            name = f'WSI_{model}_{enc}_p{patches}_{bs}_{task}_None'
                        else:
                            for clin in ['label_enc','one_hot']:
                                prefix = 'Fusion_Naive' if mode=='Fusion_Naive' else 'Fusion_Smart'
                                name = f'{prefix}_{model}_{enc}_p{patches}_{bs}_{task}_{clin}'
                                v,_,_ = get_mean_ci(name)
                                if not np.isnan(v):
                                    cis_enc[enc] = v; break
                            continue
                        v,_,_ = get_mean_ci(name)
                        if not np.isnan(v): cis_enc[enc] = v
                    if len(cis_enc)==3:
                        for enc in encoders: fm_paired[enc].append(cis_enc[enc])

    bp = ax.boxplot(fm_data, labels=encoders, patch_artist=True,
                    medianprops=dict(color='black',linewidth=2),
                    whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o',markersize=3,alpha=0.4))
    for patch, enc in zip(bp['boxes'], encoders):
        patch.set_facecolor(COLORS[enc]); patch.set_alpha(0.75)

    y_top = max([max(d) if d else 0 for d in fm_data])
    pairs = [(0,1,'UNI','Virchow'),(1,2,'Virchow','Midnight'),(0,2,'UNI','Midnight')]
    y_offs = [y_top+0.01, y_top+0.04, y_top+0.07]
    for (x1,x2,a,b), yo in zip(pairs, y_offs):
        p = wilcoxon_p(fm_paired[a], fm_paired[b])
        add_sig(ax, x1+1, x2+1, yo, p, h=0.007)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title(mlabel, fontsize=11, fontweight='bold')
    ax.set_xlabel('Foundation Model', fontsize=10)
    ax.tick_params(axis='x', labelsize=10)
    if ax_idx==0: ax.set_ylabel('Patient-Level C-Index', fontsize=11)
    ax.set_ylim(0.44, 0.88)

fig.suptitle('Foundation Model Comparison', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig2_fm_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅")

# ============================================================
# Fig 3: MIL Comparison
# ============================================================
print("Fig 3...")
fig, axes = plt.subplots(1,3, figsize=(13,5), sharey=True)
mil_models = ['ABMIL','TransMIL','MeanMIL']

for ax_idx, (mode, mlabel) in enumerate(zip(input_modes, input_labels_short)):
    ax = axes[ax_idx]
    mil_data = []
    mil_paired = {m:[] for m in mil_models}

    for model in mil_models:
        sub = df_full[(df_full['Input Mode']==mode)&(df_full['Model']==model)]
        seen = set(); cis = []
        for _, row in sub.iterrows():
            exp = '_'.join(row['Log Path'].split('/')[-1].split('_')[:-1])
            if exp in seen: continue
            seen.add(exp)
            cis.extend(get_fold_cis(exp))
        mil_data.append(cis)

    for enc in ['UNI','Virchow','Midnight']:
        for patches in [100,1000,4000]:
            for bs in ['bs32','bsFull']:
                for task in ['original','binary']:
                    cis_m = {}
                    for model in mil_models:
                        if mode=='WSI':
                            name = f'WSI_{model}_{enc}_p{patches}_{bs}_{task}_None'
                            v,_,_ = get_mean_ci(name)
                            if not np.isnan(v): cis_m[model] = v
                        else:
                            for clin in ['label_enc','one_hot']:
                                prefix = 'Fusion_Naive' if mode=='Fusion_Naive' else 'Fusion_Smart'
                                name = f'{prefix}_{model}_{enc}_p{patches}_{bs}_{task}_{clin}'
                                v,_,_ = get_mean_ci(name)
                                if not np.isnan(v):
                                    cis_m[model] = v; break
                    if len(cis_m)==3:
                        for model in mil_models: mil_paired[model].append(cis_m[model])

    bp = ax.boxplot(mil_data, labels=mil_models, patch_artist=True,
                    medianprops=dict(color='black',linewidth=2),
                    whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o',markersize=3,alpha=0.4))
    for patch, model in zip(bp['boxes'], mil_models):
        patch.set_facecolor(COLORS[model]); patch.set_alpha(0.75)

    y_top = max([max(d) if d else 0 for d in mil_data])
    pairs = [(0,1,'ABMIL','TransMIL'),(1,2,'TransMIL','MeanMIL'),(0,2,'ABMIL','MeanMIL')]
    y_offs = [y_top+0.01, y_top+0.04, y_top+0.07]
    for (x1,x2,a,b), yo in zip(pairs, y_offs):
        p = wilcoxon_p(mil_paired[a], mil_paired[b])
        add_sig(ax, x1+1, x2+1, yo, p, h=0.007)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title(mlabel, fontsize=11, fontweight='bold')
    ax.set_xlabel('MIL Architecture', fontsize=10)
    if ax_idx==0: ax.set_ylabel('Patient-Level C-Index', fontsize=11)
    ax.set_ylim(0.44, 0.88)

fig.suptitle('MIL Architecture Comparison', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig3_mil_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅")

# ============================================================
# Fig 4: Sample Efficiency（移除 std 陰影，改成 error bar）
# ============================================================
print("Fig 4...")
fig, ax = plt.subplots(figsize=(7,5))
pcts = [25,50,75,100]
pts = [76,152,228,304]
markers = {'UNI':'o','Virchow':'s','Midnight':'^'}

for enc in ['UNI','Virchow','Midnight']:
    means, stds = [], []
    for pct in pcts:
        name = f'SampleEff_{pct}pct_WSI_MeanMIL_{enc}_p1000_bsFull_original'
        m, s, _ = get_mean_ci(name)
        means.append(m); stds.append(s)
    ax.errorbar(range(4), means, yerr=stds, marker=markers[enc],
                color=COLORS[enc], linewidth=2, markersize=8,
                capsize=4, capthick=1.5, label=enc, zorder=5)

ax.axhline(0.6, color='gray', linestyle=':', alpha=0.6, linewidth=1)
ax.set_xticks(range(4))
ax.set_xticklabels([f'{p}%\n({n} pts)' for p,n in zip(pcts,pts)], fontsize=10)
ax.set_ylabel('Patient-Level C-Index', fontsize=11)
ax.set_xlabel('Training Data Proportion', fontsize=11)
ax.set_title('Sample Efficiency', fontsize=12, fontweight='bold')
ax.legend(title='FM', fontsize=10, title_fontsize=10)
ax.set_ylim(0.54, 0.73)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig4_sample_efficiency.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅")

# ============================================================
# Fig 5: TS Only vs All Slides（移除配對線，改成更清楚的呈現）
# ============================================================
print("Fig 5...")
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

all_ci, ts_ci, diffs = [], [], []
for cfg in configs_20:
    im,model,enc,p,bs,task,clin = cfg
    clin_n = 'None' if im=='WSI' else clin
    base = f'{im}_{model}_{enc}_p{p}_{bs}_{task}_{clin_n}'
    a,_,_ = get_mean_ci(base)
    t,_,_ = get_mean_ci(f'TSOnly_{base}')
    if not np.isnan(a) and not np.isnan(t):
        all_ci.append(a); ts_ci.append(t); diffs.append(t-a)

fig, axes = plt.subplots(1,2, figsize=(12,5))

# 左：Box Plot with paired dots
ax = axes[0]
bp = ax.boxplot([all_ci, ts_ci],
                labels=['All Slides\n(BS + TS)', 'Diagnostic\nSlides Only'],
                patch_artist=True,
                medianprops=dict(color='black',linewidth=2.5),
                whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
                flierprops=dict(marker='o',markersize=3,alpha=0.4),
                widths=0.4)
bp['boxes'][0].set_facecolor('#2196F3'); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('#FF9800'); bp['boxes'][1].set_alpha(0.7)

# 個別點（不畫連線）
np.random.seed(42)
for i, (a, t) in enumerate(zip(all_ci, ts_ci)):
    jitter = np.random.uniform(-0.08, 0.08)
    ax.scatter([1+jitter], [a], color='#2196F3', alpha=0.5, s=25, zorder=4)
    ax.scatter([2+jitter], [t], color='#FF9800', alpha=0.5, s=25, zorder=4)

_, p_val = wilcoxon(all_ci, ts_ci)
add_sig(ax, 1, 2, max(max(all_ci), max(ts_ci))+0.005, p_val, h=0.01)
ax.set_ylabel('Patient-Level C-Index', fontsize=11)
ax.set_title('Paired Box Plot', fontsize=11, fontweight='bold')
ax.set_ylim(0.54, 0.80)

# 右：差值分布（直方圖）
ax2 = axes[1]
ax2.hist(diffs, bins=10, color='#9C27B0', alpha=0.75, edgecolor='white')
ax2.axvline(0, color='gray', linestyle='--', linewidth=1.5, label='No change')
ax2.axvline(np.mean(diffs), color='#F44336', linestyle='-', linewidth=2,
            label=f'Mean Δ = {np.mean(diffs):.4f}')
ax2.set_xlabel('C-Index Difference\n(Diagnostic Only − All Slides)', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('Distribution of Differences\n(n=20 configurations)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)

fig.suptitle('Diagnostic Slides Only vs All Slides', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig5_slide_type.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅")

# ============================================================
# Fig 6: Late Fusion - 只顯示各策略的 Top C-Index
# ============================================================
print("Fig 6...")
df_late = pd.read_csv('/mnt/data2/TCGA_STAD_Project/0527_experiments/experiment_result/late_ensemble_result_0527.csv')

type_map = {
    'WSI': 'WSI\nEnsemble',
    'Fusion_Naive': 'Concat\nFusion Ens.',
    'Fusion_Smart': 'Projection\nFusion Ens.',
    'WSI+Clinical': 'WSI+Clinical\nEnsemble',
    'WSI+FusionSmart': 'WSI+Projection\nFusion',
    'WSI+FusionNaive': 'WSI+Concat\nFusion',
    'FusionSmart+FusionNaive': 'Concat+Projection\nFusion',
}
ens_colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336','#00BCD4','#795548']

fig, ax = plt.subplots(figsize=(10,5))
ens_types = ['WSI','Fusion_Naive','Fusion_Smart','WSI+Clinical',
             'WSI+FusionSmart','WSI+FusionNaive','FusionSmart+FusionNaive']
x_pos = range(len(ens_types))

for i, et in enumerate(ens_types):
    sub = df_late[df_late['Ensemble Type']==et]
    if len(sub)==0: continue
    top = sub.nlargest(5, 'Patient Mean')
    top_means = top['Patient Mean'].values
    top_stds = top['Patient Std'].values
    # 只顯示 Top 3
    for j, (m, s) in enumerate(zip(top_means[:3], top_stds[:3])):
        offset = (j-1) * 0.15
        ax.scatter(i+offset, m, color=ens_colors[i], s=60,
                   alpha=0.8-j*0.2, zorder=5,
                   marker=['*','o','s'][j])
        ax.errorbar(i+offset, m, yerr=s, color=ens_colors[i],
                    fmt='none', capsize=3, alpha=0.5)

ax.axhline(0.7021, color='#F44336', linestyle='--', linewidth=1.5,
           label='Best Single Model (0.7021)')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

ax.set_xticks(range(len(ens_types)))
ax.set_xticklabels([type_map[et] for et in ens_types], fontsize=9)
ax.set_ylabel('Patient-Level C-Index', fontsize=11)
ax.set_title('Late Fusion Strategy Comparison\n(★ Top-1, ● Top-2, ■ Top-3 per strategy)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0.62, 0.74)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig6_late_fusion.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅")

# ============================================================
# Fig 7: Summary Bar Chart
# ============================================================
print("Fig 7...")
fig, ax = plt.subplots(figsize=(9,5))
strategies = [
    ('Clinical Only',    0.5707, 0.0483, '#795548'),
    ('WSI Only',         0.6913, 0.0367, '#2196F3'),
    ('Concat Fusion',    0.7021, 0.0387, '#4CAF50'),
    ('Projection Fusion',0.6888, 0.0408, '#FF9800'),
    ('Late Fusion',      0.7010, 0.0592, '#9C27B0'),
]
names  = [s[0] for s in strategies]
means  = [s[1] for s in strategies]
stds   = [s[2] for s in strategies]
colors_list = [s[3] for s in strategies]

bars = ax.barh(names, means, xerr=stds, color=colors_list, alpha=0.8,
               edgecolor='white', linewidth=0.5,
               error_kw=dict(ecolor='gray',capsize=4,elinewidth=1.5),
               height=0.55)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(m+s+0.003, i, f'{m:.4f}', va='center', fontsize=10, fontweight='bold')

ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel('Best Patient-Level C-Index (Mean ± SD)', fontsize=11)
ax.set_title('Best Performance per Strategy\n(TCGA-STAD, 381 patients)', fontsize=12, fontweight='bold')
ax.set_xlim(0.44, 0.78)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig7_summary.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅")

print(f"\n全部儲存至：{OUTPUT_DIR}")
