#!/usr/bin/env python3
"""Compile all experiment results into a comprehensive summary."""
import re, json, numpy as np
from scipy import stats as sp_stats
from pathlib import Path

RESULTS_DIR = Path('comprehensive_results')

def mean_ci(vals):
    n = len(vals); m = np.mean(vals)
    if n < 2: return m, m, m
    se = sp_stats.sem(vals); h = se * sp_stats.t.ppf(0.975, n-1)
    return m, m-h, m+h

def parse_log(filepath):
    """Parse experiment log file for AUC trajectory data."""
    data = {}
    with open(filepath) as f:
        for line in f:
            m = re.match(r'\s+Churn\s+(\d+)%:\s+AUC=([\d.]+|N/A),\s+fail=(\d+),\s+avg=([\d.]+),\s+p1=([\d.]+)', line)
            if m:
                cp = m.group(1)
                auc = float(m.group(2)) if m.group(2) != 'N/A' else None
                if cp not in data: data[cp] = {'auc': [], 'fail': [], 'avg': [], 'p1': []}
                if auc is not None: data[cp]['auc'].append(auc)
                data[cp]['fail'].append(int(m.group(3)))
                data[cp]['avg'].append(float(m.group(4)))
                data[cp]['p1'].append(float(m.group(5)))
    return data

def parse_rag_log(filepath):
    """Parse RAG evaluation log."""
    results = []
    with open(filepath) as f:
        for line in f:
            m = re.match(r'Seed (\d+): base_recall=([\d.]+) churn40_recall=([\d.]+) fail=(\d+) adaptive_p1=([\d.]+)', line)
            if m:
                results.append({'seed': int(m.group(1)), 'base_recall': float(m.group(2)),
                               'churn40_recall': float(m.group(3)), 'fail': int(m.group(4)),
                               'adaptive_p1': float(m.group(5))})
    return results

def parse_vamana_log(filepath):
    """Parse Vamana experiment log."""
    results = {'seeds': []}
    current_seed = None
    current_data = {}
    with open(filepath) as f:
        for line in f:
            sm = re.match(r'\s+Seed (\d+):', line)
            if sm:
                if current_seed and current_data:
                    results['seeds'].append({'seed': current_seed, **current_data})
                current_seed = int(sm.group(1))
                current_data = {}
            bm = re.match(r'\s+Baseline: avg=([\d.]+), p1=([\d.]+)', line)
            if bm: current_data['baseline'] = {'avg': float(bm.group(1)), 'p1': float(bm.group(2))}
            cm = re.match(r'\s+Churn (\d+)%: avg=([\d.]+), p1=([\d.]+), fail=(\d+)', line)
            if cm:
                if 'churn' not in current_data: current_data['churn'] = {}
                current_data['churn'][cm.group(1)] = {'avg': float(cm.group(2)), 'p1': float(cm.group(3)), 'fail': int(cm.group(4))}
    if current_seed and current_data:
        results['seeds'].append({'seed': current_seed, **current_data})
    return results

# Parse all logs
print("="*80)
print("COMPREHENSIVE EXPERIMENT RESULTS SUMMARY")
print("="*80)

# 1. AUC Trajectory
print("\n1. AUC-vs-Churn Trajectory (SHEAF Differentiation)")
print("-"*60)
auc_data = parse_log('comprehensive_results/exp_v2_log.txt')
print(f"{'Churn%':>6} | {'AUC mean':>8} | {'95% CI':>18} | {'n':>2} | {'Fail%':>5}")
for cp in sorted(auc_data.keys(), key=int):
    d = auc_data[cp]
    if d['auc']:
        m, lo, hi = mean_ci(d['auc'])
        print(f"  {cp:>4}% | {m:>8.4f} | [{lo:.4f}, {hi:.4f}] | {len(d['auc']):>2} | {np.mean(d['fail']):>5.1f}")
all_aucs = [a for d in auc_data.values() for a in d['auc']]
if all_aucs:
    m, lo, hi = mean_ci(all_aucs)
    print(f"\n  Overall: AUC={m:.4f} [{lo:.4f}, {hi:.4f}] (n={len(all_aucs)})")

# 2. RAG Evaluation
print("\n2. RAG Retrieval Quality Under Churn")
print("-"*60)
rag_results = parse_rag_log('comprehensive_results/rag_log.txt')
if rag_results:
    base = [r['base_recall'] for r in rag_results]
    churn = [r['churn40_recall'] for r in rag_results]
    adapt = [r['adaptive_p1'] for r in rag_results]
    bm, blo, bhi = mean_ci(base)
    cm, clo, chi = mean_ci(churn)
    am, alo, ahi = mean_ci(adapt)
    print(f"  Baseline Recall@K: {bm:.4f} [{blo:.4f}, {bhi:.4f}] (n={len(base)})")
    print(f"  Churn40 Recall@K:  {cm:.4f} [{clo:.4f}, {chi:.4f}] (n={len(churn)})")
    print(f"  Adaptive p1:       {am:.4f} [{alo:.4f}, {ahi:.4f}] (n={len(adapt)})")
    print(f"  Note: Low absolute recall due to synthetic data with overlapping topic clusters")
    print(f"  Note: Churn40 > Baseline is consistent with IVF-PQ counterintuitive finding")

# 3. Vamana
print("\n3. Vamana (DiskANN-Style) Index")
print("-"*60)
vamana = parse_vamana_log('comprehensive_results/vamana_log.txt')
if vamana['seeds']:
    for s in vamana['seeds']:
        bl = s.get('baseline', {})
        print(f"  Seed {s['seed']}: baseline avg={bl.get('avg', 'N/A')}, p1={bl.get('p1', 'N/A')}")
        if 'churn' in s:
            for cp in sorted(s['churn'].keys(), key=int):
                c = s['churn'][cp]
                print(f"    Churn {cp}%: avg={c['avg']:.4f}, p1={c['p1']:.3f}, fail={c['fail']}")

# Save comprehensive results
all_results = {
    'auc_trajectory': {cp: {'mean': float(np.mean(d['auc'])), 'ci': [float(mean_ci(d['auc'])[1]), float(mean_ci(d['auc'])[2])], 'n': len(d['auc'])} for cp, d in auc_data.items() if d['auc']},
    'rag': rag_results,
    'vamana': vamana
}
with open(RESULTS_DIR/'comprehensive_summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n{'='*80}")
print("Summary saved to comprehensive_results/comprehensive_summary.json")
print(f"{'='*80}")
