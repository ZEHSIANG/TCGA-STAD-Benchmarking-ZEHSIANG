import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('#1a1a2e')

colors = {
    'stage_i': '#4ade80', 'stage_ii': '#60a5fa',
    'stage_iii': '#f97316', 'stage_iv': '#f43f5e',
    'dead': '#f43f5e', 'alive': '#4ade80',
    'male': '#60a5fa', 'female': '#f9a8d4',
    'ts': '#818cf8', 'bs': '#fb923c',
    'bar': '#6366f1', 'highlight': '#e879f9',
}

def style_ax(ax, title):
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#94a3b8', labelsize=10)
    for spine in ['bottom','left']:
        ax.spines[spine].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=12)
    ax.yaxis.label.set_color('#94a3b8')
    ax.xaxis.label.set_color('#94a3b8')

# 讀取真實資料
df = pd.read_csv('/mnt/data2/TCGA_STAD_Project/0527_experiments/csv/splits/splits_0.csv')
patient_df = df.drop_duplicates('case_id').copy()

stage_map = {
    'Stage IA':'Stage I','Stage IB':'Stage I','Stage I':'Stage I',
    'Stage IIA':'Stage II','Stage IIB':'Stage II','Stage II':'Stage II',
    'Stage IIIA':'Stage III','Stage IIIB':'Stage III','Stage IIIC':'Stage III','Stage III':'Stage III',
    'Stage IV':'Stage IV'
}
patient_df['stage_simple'] = patient_df['ajcc_pathologic_stage'].map(stage_map)

# 1. Stage 分布
ax1 = fig.add_subplot(4, 3, 1)
stage_counts = patient_df['stage_simple'].value_counts().sort_index()
bar_colors = [colors['stage_i'],colors['stage_ii'],colors['stage_iii'],colors['stage_iv']]
bars = ax1.bar(stage_counts.index, stage_counts.values, color=bar_colors, edgecolor='none', width=0.6)
for bar, cnt in zip(bars, stage_counts.values):
    pct = cnt/len(patient_df)*100
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom', color='white', fontsize=9)
style_ax(ax1, 'Stage Distribution (381 Patients)')
ax1.set_ylabel('Patient Count')
ax1.set_ylim(0, 210)

# 2. OS_status 分布
ax2 = fig.add_subplot(4, 3, 2)
dead = (patient_df['OS_status']==1).sum()
alive = (patient_df['OS_status']==0).sum()
wedges, texts, autotexts = ax2.pie(
    [dead, alive], labels=[f'Dead\n({dead})', f'Alive\n({alive})'],
    colors=[colors['dead'], colors['alive']],
    autopct='%1.1f%%', startangle=90,
    textprops={'color':'white','fontsize':11},
    wedgeprops={'edgecolor':'#1a1a2e','linewidth':2}
)
for at in autotexts: at.set_color('white'); at.set_fontweight('bold')
style_ax(ax2, 'OS Status Distribution')

# 3. OS_time 分布
ax3 = fig.add_subplot(4, 3, 3)
ax3.hist(patient_df['OS_time'], bins=30, color=colors['bar'], edgecolor='#1a1a2e', alpha=0.85)
ax3.axvline(patient_df['OS_time'].median(), color=colors['highlight'], linestyle='--',
            linewidth=2, label=f'Median: {patient_df["OS_time"].median():.0f}d')
ax3.axvline(patient_df['OS_time'].mean(), color='#fbbf24', linestyle='--',
            linewidth=2, label=f'Mean: {patient_df["OS_time"].mean():.0f}d')
ax3.legend(facecolor='#16213e', labelcolor='white', fontsize=9)
style_ax(ax3, 'OS Time Distribution (days)')
ax3.set_xlabel('Overall Survival Time (days)')
ax3.set_ylabel('Patient Count')

# 4. Gender 分布
ax4 = fig.add_subplot(4, 3, 4)
male = (patient_df['gender']=='male').sum()
female = (patient_df['gender']=='female').sum()
wedges, texts, autotexts = ax4.pie(
    [male, female], labels=[f'Male\n({male})', f'Female\n({female})'],
    colors=[colors['male'], colors['female']],
    autopct='%1.1f%%', startangle=90,
    textprops={'color':'white','fontsize':11},
    wedgeprops={'edgecolor':'#1a1a2e','linewidth':2}
)
for at in autotexts: at.set_color('white'); at.set_fontweight('bold')
style_ax(ax4, 'Gender Distribution')

# 5. Age 分布
ax5 = fig.add_subplot(4, 3, 5)
ax5.hist(patient_df['age_at_index'], bins=25, color='#34d399', edgecolor='#1a1a2e', alpha=0.85)
ax5.axvline(patient_df['age_at_index'].mean(), color=colors['highlight'], linestyle='--',
            linewidth=2, label=f'Mean: {patient_df["age_at_index"].mean():.1f}y')
ax5.legend(facecolor='#16213e', labelcolor='white', fontsize=9)
style_ax(ax5, 'Age Distribution')
ax5.set_xlabel('Age at Diagnosis')
ax5.set_ylabel('Patient Count')

# 6. Slide Type 分布
ax6 = fig.add_subplot(4, 3, 6)
ts_count = (df['slide_type']=='TS').sum()
bs_count = (df['slide_type']=='BS').sum()
bars = ax6.bar(['TS\n(Tumor Slide)', 'BS\n(Biopsy Slide)'],
               [ts_count, bs_count],
               color=[colors['ts'], colors['bs']], edgecolor='none', width=0.5)
for bar, cnt in zip(bars, [ts_count, bs_count]):
    pct = cnt/len(df)*100
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
             f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom', color='white', fontsize=10)
style_ax(ax6, f'Slide Type Distribution ({len(df)} Slides)')
ax6.set_ylabel('Slide Count')
ax6.set_ylim(0, 440)

# 7. Slides per Patient
ax7 = fig.add_subplot(4, 3, 7)
slides_pp = df.groupby('case_id').size().value_counts().sort_index()
bar_colors2 = ['#38bdf8', '#818cf8', '#f59e0b']
bars = ax7.bar([str(k) for k in slides_pp.index], slides_pp.values,
               color=bar_colors2[:len(slides_pp)], edgecolor='none', width=0.5)
for bar, cnt in zip(bars, slides_pp.values):
    pct = cnt/len(patient_df)*100
    ax7.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom', color='white', fontsize=10)
style_ax(ax7, 'Slides per Patient')
ax7.set_xlabel('Number of Slides')
ax7.set_ylabel('Patient Count')
ax7.set_ylim(0, 280)

# 8. 5-Fold CV 分布
ax8 = fig.add_subplot(4, 3, 8)
fold_data = []
for fold in range(5):
    df_f = pd.read_csv(f'/mnt/data2/TCGA_STAD_Project/0527_experiments/csv/splits/splits_{fold}.csv')
    train = df_f[df_f['split']=='train']
    val = df_f[df_f['split']=='val']
    fold_data.append({
        'train_pts': train['case_id'].nunique(),
        'val_pts': val['case_id'].nunique(),
        'train_dead': train.drop_duplicates('case_id')['OS_status'].mean(),
        'val_dead': val.drop_duplicates('case_id')['OS_status'].mean(),
    })
x = np.arange(5)
w = 0.35
b1 = ax8.bar(x-w/2, [d['train_pts'] for d in fold_data], w, label='Train', color='#6366f1', edgecolor='none')
b2 = ax8.bar(x+w/2, [d['val_pts'] for d in fold_data], w, label='Val', color='#f59e0b', edgecolor='none')
ax8.set_xticks(x)
ax8.set_xticklabels([f'Fold {i}' for i in range(5)])
ax8.legend(facecolor='#16213e', labelcolor='white', fontsize=9)
ax8_twin = ax8.twinx()
ax8_twin.plot(x-w/2, [d['train_dead']*100 for d in fold_data], 'o--',
              color='#f43f5e', linewidth=1.5, markersize=5, label='Train dead%')
ax8_twin.plot(x+w/2, [d['val_dead']*100 for d in fold_data], 's--',
              color='#4ade80', linewidth=1.5, markersize=5, label='Val dead%')
ax8_twin.set_ylabel('Dead Rate (%)', color='#94a3b8')
ax8_twin.tick_params(colors='#94a3b8')
ax8_twin.set_ylim(35, 45)
ax8_twin.spines['right'].set_color('#334155')
ax8_twin.spines['top'].set_visible(False)
style_ax(ax8, '5-Fold CV Patient Distribution')
ax8.set_ylabel('Patient Count')

# 9. Sample Efficiency 設計
ax9 = fig.add_subplot(4, 3, 9)
pcts = ['25%', '50%', '75%', '100%']
train_patients_se = [76, 152, 228, 304]
train_slides_se = [104, 218, 322, 428]
x = np.arange(4)
w = 0.35
b1 = ax9.bar(x-w/2, train_patients_se, w, label='Patients', color='#818cf8', edgecolor='none')
b2 = ax9.bar(x+w/2, train_slides_se, w, label='Slides', color='#34d399', edgecolor='none')
for bar, val in zip(b1, train_patients_se):
    ax9.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
             str(val), ha='center', va='bottom', color='white', fontsize=9)
for bar, val in zip(b2, train_slides_se):
    ax9.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
             str(val), ha='center', va='bottom', color='white', fontsize=9)
ax9.set_xticks(x)
ax9.set_xticklabels(pcts)
ax9.legend(facecolor='#16213e', labelcolor='white', fontsize=9)
style_ax(ax9, 'Sample Efficiency Experiment Design')
ax9.set_xlabel('Training Data Proportion')
ax9.set_ylabel('Count')
ax9.set_ylim(0, 490)

# 10. TS Only 設計
ax10 = fig.add_subplot(4, 3, 10)
categories = ['All Slides', 'TS Only']
slides_list = [542, 367]
patients_list = [381, 367]
x = np.arange(2)
w = 0.35
b1 = ax10.bar(x-w/2, slides_list, w, label='Slides', color='#818cf8', edgecolor='none')
b2 = ax10.bar(x+w/2, patients_list, w, label='Patients', color='#fb923c', edgecolor='none')
for bar, val in zip(b1, slides_list):
    ax10.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
              str(val), ha='center', va='bottom', color='white', fontsize=10)
for bar, val in zip(b2, patients_list):
    ax10.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
              str(val), ha='center', va='bottom', color='white', fontsize=10)
ax10.set_xticks(x)
ax10.set_xticklabels(categories)
ax10.legend(facecolor='#16213e', labelcolor='white', fontsize=9)
style_ax(ax10, 'Slide Type Experiment Design')
ax10.set_ylabel('Count')
ax10.set_ylim(0, 620)

# 11. Patient Slide Type 組成
ax11 = fig.add_subplot(4, 3, 11)
patient_types = df.groupby('case_id')['slide_type'].apply(set)
both = (patient_types.apply(lambda x: 'BS' in x and 'TS' in x)).sum()
ts_only_pts = (patient_types.apply(lambda x: x == {'TS'})).sum()
bs_only_pts = (patient_types.apply(lambda x: x == {'BS'})).sum()
wedges, texts, autotexts = ax11.pie(
    [ts_only_pts, both, bs_only_pts],
    labels=[f'TS Only\n({ts_only_pts})', f'BS+TS\n({both})', f'BS Only\n({bs_only_pts})'],
    colors=[colors['ts'], '#a78bfa', colors['bs']],
    autopct='%1.1f%%', startangle=90,
    textprops={'color':'white','fontsize':10},
    wedgeprops={'edgecolor':'#1a1a2e','linewidth':2}
)
for at in autotexts: at.set_color('white'); at.set_fontweight('bold')
style_ax(ax11, 'Patient Slide Type Composition')

# 12. 實驗總覽
ax12 = fig.add_subplot(4, 3, 12)
ax12.set_facecolor('#16213e')
ax12.axis('off')
summary = [
    ('EXPERIMENT SUMMARY', 0.95, '#e879f9', 14, 'bold'),
    ('Dataset: TCGA-STAD', 0.85, '#94a3b8', 11, 'normal'),
    ('  381 patients | 542 slides', 0.79, 'white', 10, 'normal'),
    ('  Dead: 152 (39.9%)', 0.73, 'white', 10, 'normal'),
    ('  OS Time: 14 ~ 3720 days', 0.67, 'white', 10, 'normal'),
    ('Experiments:', 0.58, '#94a3b8', 11, 'normal'),
    ('  Full Grid: 2,720 runs', 0.52, '#4ade80', 10, 'normal'),
    ('  Late Ensemble: 1,656 runs', 0.46, '#4ade80', 10, 'normal'),
    ('  Early Ensemble: 900 runs', 0.40, '#4ade80', 10, 'normal'),
    ('  Loss Comparison: 100 runs', 0.34, '#60a5fa', 10, 'normal'),
    ('  TS Only: 100 runs', 0.28, '#f59e0b', 10, 'normal'),
    ('  Sample Efficiency: 60 runs', 0.22, '#f59e0b', 10, 'normal'),
    ('Total: ~5,536 experiments', 0.12, '#e879f9', 12, 'bold'),
]
for text, y, color, size, weight in summary:
    ax12.text(0.05, y, text, transform=ax12.transAxes,
              color=color, fontsize=size, fontweight=weight, va='top')
ax12.set_title('Experiment Overview', color='white', fontsize=13, fontweight='bold', pad=12)

fig.suptitle('TCGA-STAD Dataset Statistics & Experiment Design',
             color='white', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.97])

output_path = '/mnt/data2/TCGA_STAD_Project/0527_experiments/figures/dataset_statistics_0527.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f'儲存完成：{output_path}')
