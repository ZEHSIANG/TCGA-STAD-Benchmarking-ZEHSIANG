import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.patch.set_facecolor('#1a1a2e')

def style_ax(ax, title):
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ['bottom','left']:
        ax.spines[spine].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=12)
    ax.yaxis.label.set_color('#94a3b8')
    ax.xaxis.label.set_color('#94a3b8')

# ============================================================
# 1. TS Only vs All Slides（Scatter Plot）
# ============================================================
ax1 = axes[0]

all_cis = [0.6626,0.6557,0.6843,0.6530,0.6713,0.6714,0.6948,0.6784,
           0.6605,0.6914,0.6974,0.6946,0.6913,0.6968,0.6832,0.6913,
           0.6948,0.7021,0.6759,0.6949]
ts_cis  = [0.6487,0.6355,0.6451,0.6025,0.6177,0.6150,0.6355,0.6189,
           0.5932,0.6205,0.6260,0.6230,0.6186,0.6236,0.5982,0.6038,
           0.6024,0.6074,0.5720,0.5901]
diffs = [t-a for t,a in zip(ts_cis, all_cis)]

ax1.scatter(all_cis, ts_cis, c='#f43f5e', s=80, alpha=0.8, zorder=5, edgecolors='white', linewidth=0.5)

# 對角線（y=x）
lim_min = min(min(all_cis), min(ts_cis)) - 0.01
lim_max = max(max(all_cis), max(ts_cis)) + 0.01
ax1.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='#94a3b8', linewidth=1.5, alpha=0.7, label='y=x (no change)')
ax1.fill_between([lim_min, lim_max], [lim_min, lim_min], [lim_min, lim_max],
                  alpha=0.05, color='#4ade80')
ax1.fill_between([lim_min, lim_max], [lim_min, lim_max], [lim_max, lim_max],
                  alpha=0.05, color='#f43f5e')

ax1.text(0.58, 0.69, 'All Slides\nbetter', color='#f43f5e', fontsize=9, alpha=0.8)
ax1.text(0.685, 0.575, 'TS Only\nbetter', color='#4ade80', fontsize=9, alpha=0.8)

ax1.set_xlabel('All Slides C-Index (Patient Level)', fontsize=10)
ax1.set_ylabel('TS Only C-Index (Patient Level)', fontsize=10)
ax1.legend(facecolor='#16213e', labelcolor='white', fontsize=9)
style_ax(ax1, 'TS Only vs All Slides\n(20 Configurations, 5-Fold)')

# 統計資訊
avg_diff = np.mean(diffs)
ax1.text(0.03, 0.97, f'All ↓ worse ({len(diffs)}/20)\nMean diff: {avg_diff:+.4f}',
         transform=ax1.transAxes, color='#f43f5e', fontsize=10,
         va='top', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

# ============================================================
# 2. TS Only Diff Bar Chart
# ============================================================
ax2 = axes[1]

short_labels = [
    'WSI/ABMIL/UNI/p4000', 'FS/ABMIL/UNI/p4000', 'FN/MeanMIL/UNI/p4000/bs32',
    'WSI/MeanMIL/UNI/p4000', 'FS/MeanMIL/UNI/p4000/bi', 'FN/MeanMIL/UNI/p4000',
    'FN/MeanMIL/Mid/p1000/ori', 'FS/MeanMIL/UNI/p4000/bs32', 'FS/MeanMIL/UNI/p4000/bi/bs32',
    'FN/MeanMIL/Mid/p4000', 'FN/MeanMIL/Mid/p1000/ori/oh', 'FN/MeanMIL/Mid/p1000/bi',
    'FN/MeanMIL/Mid/p100', 'FN/MeanMIL/UNI/p4000/bi', 'FS/MeanMIL/UNI/p4000/ori',
    'WSI/MeanMIL/Mid/p4000', 'FN/ABMIL/Mid/p1000/ori', 'FN/ABMIL/Mid/p1000/bi',
    'FS/TransMIL/Mid/p1000', 'FN/ABMIL/Mid/p1000/ori/oh'
]

colors = ['#f43f5e' if d < 0 else '#4ade80' for d in diffs]
y_pos = range(len(diffs))
bars = ax2.barh(y_pos, diffs, color=colors, edgecolor='none', height=0.7)
ax2.axvline(0, color='#94a3b8', linewidth=1.5, linestyle='-')
ax2.axvline(np.mean(diffs), color='#fbbf24', linewidth=1.5, linestyle='--',
            label=f'Mean: {np.mean(diffs):+.4f}')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(short_labels, fontsize=7)
ax2.legend(facecolor='#16213e', labelcolor='white', fontsize=9)
ax2.set_xlabel('C-Index Difference (TS Only - All Slides)', fontsize=10)
style_ax(ax2, 'TS Only Performance Difference\n(All Slides is Better)')

# ============================================================
# 3. Sample Efficiency
# ============================================================
ax3 = axes[2]

pcts = [25, 50, 75, 100]
fm_data = {
    'UNI':     ([0.6029, 0.6470, 0.6406, 0.6449],
                [0.0569, 0.0430, 0.0144, 0.0356]),
    'Virchow': ([0.5844, 0.6412, 0.6364, 0.6454],
                [0.0340, 0.0391, 0.0605, 0.0649]),
    'Midnight':([0.5869, 0.6401, 0.6498, 0.6726],
                [0.0516, 0.0511, 0.0258, 0.0426]),
}
fm_colors = {'UNI': '#818cf8', 'Virchow': '#34d399', 'Midnight': '#f97316'}
markers = {'UNI': 'o', 'Virchow': 's', 'Midnight': '^'}

for fm, (means, stds) in fm_data.items():
    ax3.plot(pcts, means, marker=markers[fm], color=fm_colors[fm],
             linewidth=2.5, markersize=8, label=fm)
    ax3.fill_between(pcts,
                     [m-s for m,s in zip(means,stds)],
                     [m+s for m,s in zip(means,stds)],
                     alpha=0.15, color=fm_colors[fm])

ax3.axhline(0.6, color='#94a3b8', linestyle=':', linewidth=1, alpha=0.5, label='C-Index=0.6')
ax3.set_xlabel('Training Data Proportion (%)', fontsize=10)
ax3.set_ylabel('Patient Level C-Index', fontsize=10)
ax3.set_xticks(pcts)
ax3.set_xticklabels(['25%\n(76 pts)', '50%\n(152 pts)', '75%\n(228 pts)', '100%\n(304 pts)'])
ax3.legend(facecolor='#16213e', labelcolor='white', fontsize=10)
ax3.set_ylim(0.54, 0.72)
style_ax(ax3, 'Sample Efficiency\n(WSI Only, MeanMIL, p1000)')

# 標注關鍵發現
ax3.annotate('UNI best\nat 25%', xy=(25, 0.6029), xytext=(35, 0.615),
             color='#818cf8', fontsize=9, arrowprops=dict(arrowstyle='->', color='#818cf8'))
ax3.annotate('Midnight best\nat 100%', xy=(100, 0.6726), xytext=(80, 0.685),
             color='#f97316', fontsize=9, arrowprops=dict(arrowstyle='->', color='#f97316'))

fig.suptitle('TS Only vs All Slides & Sample Efficiency Analysis',
             color='white', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()

output = '/mnt/data2/TCGA_STAD_Project/0527_experiments/figures/ts_sample_analysis.png'
os.makedirs(os.path.dirname(output), exist_ok=True)
plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f'儲存完成：{output}')
