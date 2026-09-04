#!/usr/bin/env python3
"""Generate HONEST journal figures from journal_full_results.json (real data)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'; OUT.mkdir(exist_ok=True)
res = json.load(open(ROOT / 'data' / 'journal_full_results.json'))

def pool(rec, path):
    parts = []
    for key in rec:
        node = rec[key]
        for p in path: node = node[p]
        parts.append(np.asarray(node, float))
    return np.concatenate(parts)

datasets = list(res.keys())
churn = [20, 40]
palette = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
lbl = {'SIFT-128':'SIFT-128','Fashion-MNIST':'Fashion-MNIST','GloVe-200':'GloVe-200','SIFT-1M':'SIFT-1M'}

# ---- Fig 1: Masking gap (avg vs p1 vs fail%) across churn levels ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for d, c in zip(datasets, palette):
    traj = {0: pool(res[d], ['baseline', 'recalls'])}
    for cp in churn: traj[cp] = pool(res[d], ['churn', str(cp), 'recalls'])
    xs = sorted(traj)
    axes[0].plot(xs, [np.mean(traj[x]) for x in xs], marker='o', color=c, label=lbl[d])
    axes[1].plot(xs, [np.percentile(traj[x], 1) for x in xs], marker='o', color=c)
    axes[2].plot(xs, [np.mean(np.asarray(traj[x]) < 0.9) * 100 for x in xs], marker='o', color=c)
for ax, t in zip(axes, ['Average recall', 'p1 (tail) recall', 'Queries failing <0.9 recall (%)']):
    ax.set_xlabel('Churn (%)'); ax.set_title(t); ax.grid(True, alpha=.4)
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / 'fig_masking.pdf'); fig.savefig(OUT / 'fig_masking.png', dpi=130)

# ---- Fig 2: Per-query recall histogram under churn (SIFT-128) ----
r = pool(res['SIFT-128'], ['churn', '40', 'recalls'])
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.hist(r, bins=np.linspace(0, 1, 41), color='steelblue', edgecolor='white')
ax.axvline(np.mean(r), color='red', ls='--', label=f"avg={np.mean(r):.3f}")
ax.axvline(np.median(r), color='green', ls=':', label=f"median={np.median(r):.3f}")
ax.set_xlabel('Per-query recall (K=10, ef=50, 40% churn)'); ax.set_ylabel('Query count')
ax.set_title('SIFT-128 recall distribution under churn'); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'fig_histogram.pdf'); fig.savefig(OUT / 'fig_histogram.png', dpi=130)

# ---- Fig 3: Predictor ROC (pool AUC) ----
fig, ax = plt.subplots(figsize=(5, 4))
for d, c in zip(datasets, palette):
    inst = pool(res[d], ['predictor', 'instability'])
    lab = pool(res[d], ['predictor', 'labels'])
    if lab.sum() == 0 or lab.sum() == len(lab): continue
    fpr, tpr, _ = roc_curve(lab, inst)
    ax.plot(fpr, tpr, color=c, label=f"{lbl[d]} (AUC={np.trapz(tpr, fpr):.2f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=.5)
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title('Instability predictor (no ground truth)'); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'fig_predictor.pdf'); fig.savefig(OUT / 'fig_predictor.png', dpi=130)

# ---- Fig 4: Matched-cost recovery (failures vs avg ef) ----
fig, ax = plt.subplots(figsize=(6, 4))
for d, c in zip(datasets, palette):
    xs, ys = [], []
    for e in sorted(res[d][list(res[d])[0]]['recovery']['uniform'], key=lambda x: int(x[2:])):
        ur = pool(res[d], ['recovery', 'uniform', e, 'recalls'])
        xs.append(int(e[2:])); ys.append(np.mean(np.asarray(ur) < 0.9) * 100)
    cf = pool(res[d], ['recovery', 'churnflux', 'recalls'])
    cfef = np.mean([res[d][k]['recovery']['avg_ef'] for k in res[d]])
    ax.plot(xs, ys, marker='o', color=c, ls='-', label=lbl[d])
    ax.plot(cfef, np.mean(np.asarray(cf) < 0.9) * 100, marker='*', ms=14, color=c)
ax.set_xlabel('Average effort (ef)'); ax.set_ylabel('Failures <0.9 recall (%)')
ax.set_title('Uniform ef vs ChurnFlux (star) — matched cost'); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'fig_recovery.pdf'); fig.savefig(OUT / 'fig_recovery.png', dpi=130)

print("figures written to", OUT)
