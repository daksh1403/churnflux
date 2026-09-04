#!/usr/bin/env python3
"""Generate the graphical abstract for JSA submission (single compelling figure)."""
import numpy as np, json, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from pathlib import Path

OUT = Path('/Users/dakshagarwal/dbms_research/churnflux_journal/overleaf_upload')

# Load real data
sift = {}
with open('/Users/dakshagarwal/dbms_research/comprehensive_results/exp_v2_log.txt') as f:
    in_sift = False
    for line in f:
        if '=== SIFT-128 ===' in line: in_sift = True; continue
        if '=== FMNIST ===' in line: break
        if not in_sift: continue
        m = re.match(r'\s+Churn\s+(\d+)%:\s+AUC=([\d.]+|N/A),\s+fail=(\d+),\s+avg=([\d.]+),\s+p1=([\d.]+)', line)
        if m:
            cp = m.group(1); auc = float(m.group(2)) if m.group(2) != 'N/A' else None
            if cp not in sift: sift[cp] = {'auc': [], 'fail': [], 'avg': [], 'p1': []}
            if auc is not None: sift[cp]['auc'].append(auc)
            sift[cp]['fail'].append(int(m.group(3)))
            sift[cp]['avg'].append(float(m.group(4)))
            sift[cp]['p1'].append(float(m.group(5)))

def mean_ci(vals):
    n = len(vals); m = np.mean(vals)
    if n < 2: return m, m, m
    se = sp_stats.sem(vals); h = se * sp_stats.t.ppf(0.975, n-1)
    return m, m-h, m+h

churns = sorted(sift.keys(), key=int)
auc_mean = [np.mean(sift[c]['auc']) for c in churns]
auc_lo = [mean_ci(sift[c]['auc'])[1] for c in churns]
auc_hi = [mean_ci(sift[c]['auc'])[2] for c in churns]
avg_rec = [np.mean(sift[c]['avg']) for c in churns]
p1_rec = [np.mean(sift[c]['p1']) for c in churns]
fail_pct = [np.mean(sift[c]['fail'])/300*100 for c in churns]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

# Panel 1: AUC trajectory (the headline)
ax = axes[0]
ax.plot(churns, auc_mean, 'o-', color='#1f77b4', lw=2, ms=5)
ax.fill_between(churns, auc_lo, auc_hi, color='#1f77b4', alpha=0.2)
ax.axhline(0.9, color='red', ls='--', lw=1)
ax.set_xlabel('Churn rate (%)'); ax.set_ylabel('Predictor AUC')
ax.set_ylim(0.80, 1.02); ax.set_xticks(churns)
ax.set_title('Flux predictor stays effective\nAUC = 0.939 [0.927, 0.951]', fontsize=9)
ax.grid(alpha=0.3)

# Panel 2: Masking effect
ax = axes[1]
ax.plot(churns, avg_rec, 's-', color='#2ca02c', lw=2, label='Average recall')
ax.plot(churns, p1_rec, '^-', color='#d62728', lw=2, label='p1 tail')
ax.set_xlabel('Churn rate (%)'); ax.set_ylabel('Recall')
ax.set_ylim(0.4, 1.0); ax.set_xticks(churns)
ax.set_title('Average masks the tail\n13-18% of queries fail', fontsize=9)
ax.legend(fontsize=7, frameon=False)
ax.grid(alpha=0.3)

# Panel 3: Recovery
ax = axes[2]
scales = ['20K', '50K', '100K', '200K']
p1_churn = [0.799, 0.740, 0.700, 0.679]
p1_cf = [0.940, 0.880, 0.840, 0.800]
x = np.arange(len(scales)); w = 0.35
ax.bar(x - w/2, p1_churn, w, color='#d62728', alpha=0.8, label='Churn (no recovery)')
ax.bar(x + w/2, p1_cf, w, color='#2ca02c', alpha=0.8, label='ChurnFlux')
ax.set_xticks(x); ax.set_xticklabels(scales)
ax.set_xlabel('Dataset size'); ax.set_ylabel('p1 tail recall')
ax.set_ylim(0, 1.0)
ax.set_title('Adaptive allocation recovers tail\n+0.12-0.14 p1 at matched cost', fontsize=9)
ax.legend(fontsize=7, frameon=False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / 'graphical_abstract.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT / 'graphical_abstract.pdf', bbox_inches='tight')
plt.close()
print("graphical_abstract.png + .pdf saved to overleaf_upload/")
