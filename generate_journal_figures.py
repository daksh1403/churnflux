#!/usr/bin/env python3
"""Generate journal-grade figures from REAL experiment data."""
import numpy as np, json, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from pathlib import Path

OUT = Path('/Users/dakshagarwal/dbms_research/churnflux_journal/figures')
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif',
    'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 300,
})

def mean_ci(vals):
    n = len(vals); m = np.mean(vals)
    if n < 2: return m, m, m
    se = sp_stats.sem(vals); h = se * sp_stats.t.ppf(0.975, n-1)
    return m, m-h, m+h

# --- Parse AUC trajectory data from exp_v2_log.txt (SIFT-128 section) ---
sift_data = {}
with open('/Users/dakshagarwal/dbms_research/comprehensive_results/exp_v2_log.txt') as f:
    in_sift = False
    for line in f:
        if '=== SIFT-128 ===' in line: in_sift = True; continue
        if '=== FMNIST ===' in line: break
        if not in_sift: continue
        m = re.match(r'\s+Churn\s+(\d+)%:\s+AUC=([\d.]+|N/A),\s+fail=(\d+),\s+avg=([\d.]+),\s+p1=([\d.]+)', line)
        if m:
            cp = m.group(1); auc = float(m.group(2)) if m.group(2) != 'N/A' else None
            if cp not in sift_data: sift_data[cp] = {'auc': [], 'fail': [], 'avg': [], 'p1': []}
            if auc is not None: sift_data[cp]['auc'].append(auc)
            sift_data[cp]['fail'].append(int(m.group(3)))
            sift_data[cp]['avg'].append(float(m.group(4)))
            sift_data[cp]['p1'].append(float(m.group(5)))

# --- Figure 1: AUC vs churn trajectory ---
churns = sorted(sift_data.keys(), key=int)
auc_means = [np.mean(sift_data[c]['auc']) for c in churns]
auc_los = [mean_ci(sift_data[c]['auc'])[1] for c in churns]
auc_his = [mean_ci(sift_data[c]['auc'])[2] for c in churns]

fig, ax = plt.subplots(figsize=(3.5, 2.8))
ax.plot(churns, auc_means, 'o-', color='#1f77b4', lw=1.5, ms=5, label='AUC (mean)')
ax.fill_between(churns, auc_los, auc_his, color='#1f77b4', alpha=0.2, label='95% CI')
ax.axhline(0.9, color='red', ls='--', lw=1, label='0.9 threshold')
ax.set_xlabel('Churn rate (%)')
ax.set_ylabel('Predictor AUC')
ax.set_ylim(0.80, 1.02)
ax.set_xticks(churns)
ax.legend(loc='lower left', frameon=False)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'fig_auc_trajectory.png', bbox_inches='tight')
plt.savefig(OUT / 'fig_auc_trajectory.pdf', bbox_inches='tight')
plt.close()
print("fig_auc_trajectory done")

# --- Figure 2: RAG retrieval quality ---
with open('/Users/dakshagarwal/dbms_research/comprehensive_results/rag_results.json') as f:
    rag = json.load(f)
seeds = list(rag.keys())
cats = ['Baseline', 'Churn 40%', 'ChurnFlux', 'Uniform matched']
vals = {
    'Baseline': [rag[s]['baseline']['recall'] for s in seeds],
    'Churn 40%': [rag[s]['churn']['40']['recall'] for s in seeds],
    'ChurnFlux': [rag[s]['recovery']['adaptive']['recall'] for s in seeds],
    'Uniform matched': [rag[s]['recovery']['uniform']['recall'] for s in seeds],
}
fig, ax = plt.subplots(figsize=(3.5, 2.8))
x = np.arange(len(cats))
means = [np.mean(vals[c]) for c in cats]
los = [mean_ci(vals[c])[1] for c in cats]
his = [mean_ci(vals[c])[2] for c in cats]
colors = ['#7f7f7f', '#d62728', '#2ca02c', '#1f77b4']
bars = ax.bar(x, means, yerr=[np.array(means)-np.array(los), np.array(his)-np.array(means)],
              capsize=4, width=0.6, color=colors, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(cats, rotation=15, ha='right')
ax.set_ylabel('Recall@10')
ax.set_ylim(0.3, 0.6)
ax.grid(axis='y', alpha=0.3)
for i, (m, lo, hi) in enumerate(zip(means, los, his)):
    ax.text(i, hi + 0.01, f'{m:.3f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(OUT / 'fig_rag_retrieval.png', bbox_inches='tight')
plt.savefig(OUT / 'fig_rag_retrieval.pdf', bbox_inches='tight')
plt.close()
print("fig_rag_retrieval done")

# --- Figure 3: Mean vs tail masking ---
fail_pct = [np.mean(sift_data[c]['fail'])/300*100 for c in churns]
avg_rec = [np.mean(sift_data[c]['avg']) for c in churns]
p1_rec = [np.mean(sift_data[c]['p1']) for c in churns]

fig, ax1 = plt.subplots(figsize=(3.5, 2.8))
ax1.plot(churns, avg_rec, 's-', color='#2ca02c', lw=1.5, label='Average recall')
ax1.plot(churns, p1_rec, '^-', color='#d62728', lw=1.5, label='p1 tail recall')
ax1.set_xlabel('Churn rate (%)')
ax1.set_ylabel('Recall', color='black')
ax1.set_ylim(0.4, 1.0)
ax2 = ax1.twinx()
ax2.plot(churns, fail_pct, 'o--', color='#1f77b4', lw=1.2, label='Fail rate (<0.9)')
ax2.set_ylabel('Fail rate (%)', color='#1f77b4')
ax2.tick_params(axis='y', labelcolor='#1f77b4')
ax2.set_ylim(0, 30)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=False, fontsize=8)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'fig_masking.png', bbox_inches='tight')
plt.savefig(OUT / 'fig_masking.pdf', bbox_inches='tight')
plt.close()
print("fig_masking done")

# --- Figure 4: Theta(sqrt(p)) amplification validation ---
# Mean-tail gap vs churn
gap = np.array(avg_rec) - np.array(p1_rec)
fig, ax = plt.subplots(figsize=(3.5, 2.8))
ax.plot(churns, gap, 'o-', color='#9467bd', lw=1.5, label='Mean - p1 gap (observed)')
# Fit sqrt(p)
p_vals = np.array([int(c) for c in churns]) / 100.0
coef = np.polyfit(np.sqrt(p_vals), gap, 1)
fit = np.polyval(coef, np.sqrt(p_vals))
ax.plot(churns, fit, '--', color='black', lw=1.2, label=f'$\\Theta(\\sqrt{{p}})$ fit (R={np.corrcoef(np.sqrt(p_vals), gap)[0,1]:.3f})')
ax.set_xlabel('Churn rate (%)')
ax.set_ylabel('Mean - p1 gap')
ax.set_ylim(0, 0.30)
ax.legend(frameon=False)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'fig_amplification.png', bbox_inches='tight')
plt.savefig(OUT / 'fig_amplification.pdf', bbox_inches='tight')
plt.close()
print("fig_amplification done")

print(f"\nAll figures saved to {OUT}")
