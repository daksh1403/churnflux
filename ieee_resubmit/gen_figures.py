#!/usr/bin/env python3
"""Generate IEEE-style figures for the ChurnFlux paper from paper_results.json."""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = __import__('pathlib').Path(__file__).resolve().parent
RES = json.loads((ROOT / 'paper_results.json').read_text())
CHURNS = ['5', '10', '20', '30', '40', '50']
CHURN_X = [5, 10, 20, 30, 40, 50]
COLORS = {'SIFT-1M': '#0072BD', 'GloVe-200': '#D95319', 'Fashion-MNIST': '#EDB120'}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})


def plot_auc():
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for dname in ['SIFT-1M', 'GloVe-200', 'Fashion-MNIST']:
        key = f'HNSW_{dname}'
        if key not in RES:
            continue
        means = [RES[key]['summary'][c]['auc']['mean'] for c in CHURNS]
        lo = [RES[key]['summary'][c]['auc']['ci'][0] for c in CHURNS]
        hi = [RES[key]['summary'][c]['auc']['ci'][1] for c in CHURNS]
        ax.plot(CHURN_X, means, '-o', color=COLORS[dname], label=dname, ms=4)
        ax.fill_between(CHURN_X, lo, hi, color=COLORS[dname], alpha=0.15)
    ax.set_xlabel('Churn rate (%)')
    ax.set_ylabel('AUC of instability predictor')
    ax.set_ylim(0.7, 1.02)
    ax.legend(frameon=False, loc='lower left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / 'fig_auc.png')
    plt.close(fig)
    print('fig_auc.png')


def plot_p1():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4), sharey=True)
    for ax, dname in zip(axes, ['SIFT-1M', 'GloVe-200', 'Fashion-MNIST']):
        key = f'HNSW_{dname}'
        if key not in RES:
            continue
        churn_p1 = [RES[key]['summary'][c]['churn']['p1']['mean'] for c in CHURNS]
        adapt_p1 = [RES[key]['summary'][c]['adapt']['p1']['mean'] for c in CHURNS]
        unif_p1 = [RES[key]['summary'][c]['uniform']['p1']['mean'] for c in CHURNS]
        ax.plot(CHURN_X, churn_p1, '--s', color='#555555', label='Churned (ef=50)', ms=3)
        ax.plot(CHURN_X, adapt_p1, '-o', color='#0072BD', label='ChurnFlux', ms=4)
        ax.plot(CHURN_X, unif_p1, '-^', color='#D95319', label='Uniform (matched)', ms=4)
        ax.set_title(dname)
        ax.set_xlabel('Churn (%)')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('p1 tail recall')
    axes[0].legend(frameon=False, fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(ROOT / 'fig_p1.png')
    plt.close(fig)
    print('fig_p1.png')


def plot_rag():
    rag = RES.get('RAG_downstream', {}).get('summary', {})
    if not rag:
        return
    labels = ['ChurnFlux', 'Uniform\n(matched ef)']
    rec = [rag['adaptive_recall']['mean'], rag['uniform_recall']['mean']]
    rec_lo = [rag['adaptive_recall']['ci'][0], rag['uniform_recall']['ci'][0]]
    rec_hi = [rag['adaptive_recall']['ci'][1], rag['uniform_recall']['ci'][1]]
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    x = np.arange(2)
    bars = ax.bar(x, rec, 0.5, color=['#0072BD', '#D95319'], yerr=[rec[i]-rec_lo[i] for i in range(2)])
    ax.errorbar(x, rec, yerr=[np.array([rec[i]-rec_lo[i] for i in range(2)]), np.array([rec_hi[i]-rec[i] for i in range(2)])],
                fmt='none', ecolor='black', capsize=3)
    for i, v in enumerate(rec):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Recall@10 (downstream)')
    ax.set_ylim(0.4, 0.6)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(ROOT / 'fig_rag.png')
    plt.close(fig)
    print('fig_rag.png')


if __name__ == '__main__':
    plot_auc()
    plot_p1()
    plot_rag()
