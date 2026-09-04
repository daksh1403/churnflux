#!/usr/bin/env python3
"""Compute honest pooled statistics from journal_results_master.json (REAL data)."""
import json, numpy as np
from pathlib import Path
from math import comb

ROOT = Path(__file__).resolve().parents[1]
m = json.load(open(ROOT / 'data' / 'journal_results_master.json'))

def boot(x, stat=lambda a: np.mean(a), N=2000):
    x = np.asarray(x, float); rng = np.random.RandomState(0)
    v = [stat(rng.choice(x, len(x), replace=True)) for _ in range(N)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def met(r):
    r = np.asarray(r, float)
    return dict(avg=float(np.mean(r)), p1=float(np.percentile(r, 1)),
                p5=float(np.percentile(r, 5)), rob09=float(np.mean(r >= 0.9)),
                rob08=float(np.mean(r >= 0.8)), fail09=int(np.sum(r < 0.9)),
                fail05=int(np.sum(r < 0.5)))

def mcnemar(a, b):
    oa = int((a & ~b).sum()); ob = int((~a & b).sum()); n = oa + ob
    if n == 0: return 1.0
    k = min(oa, ob)
    return min(1.0, 2.0 * sum(comb(n, i) * 0.5 ** n for i in range(k + 1)))

summary = {}
for name, rec in m.items():
    s0 = rec[list(rec)[0]]; n = s0['scale']; dim = s0['dim']; seeds = list(rec)
    qper = len(rec[seeds[0]]['hnsw']['churn']['40']['recalls'])
    nq = qper * len(seeds)
    e = dict(name=name, n=n, dim=dim, seeds=seeds, qper=qper, nq=nq,
             base_avg=float(np.mean([rec[s]['hnsw']['baseline']['avg'] for s in seeds])))
    e['traj'] = {}
    for cl in sorted({int(k) for s in seeds for k in rec[s]['hnsw']['churn']}):
        r = np.concatenate([rec[s]['hnsw']['churn'][str(cl)]['recalls'] for s in seeds])
        mm = met(r); lo, hi = boot(r)
        e['traj'][cl] = dict(avg=mm['avg'], ci=[lo, hi], p1=mm['p1'],
                             rob09=mm['rob09'], fail09=mm['fail09'])
    aucs = [float(rec[s]['hnsw']['predictor']['auc']) for s in seeds]
    e['auc'] = float(np.mean(aucs)); e['auc_seeds'] = aucs
    e['auc_ci'] = list(boot(np.asarray(aucs)))
    e['nfail40_seeds'] = [rec[s]['hnsw']['churn']['40']['metrics']['n_fail_09'] for s in seeds]
    cf = np.concatenate([rec[s]['hnsw']['recovery']['churnflux']['recalls'] for s in seeds])
    cf_ab = cf < 0.9; cfm = met(cf)
    e['cf_avg'] = cfm['avg']; e['cf_p1'] = cfm['p1']; e['cf_fail'] = cfm['fail09']
    e['cf_avg_ef'] = float(np.mean([rec[s]['hnsw']['recovery']['avg_ef'] for s in seeds]))
    efk = sorted(rec[seeds[0]]['hnsw']['recovery']['uniform'], key=lambda x: int(x[2:]))
    best = None
    for kk in efk:
        ur = np.concatenate([rec[s]['hnsw']['recovery']['uniform'][kk]['recalls'] for s in seeds])
        um = met(ur); unA = ur < 0.9
        if best is None or abs(e['cf_avg_ef'] - int(kk[2:])) < abs(e['cf_avg_ef'] - int(best[0][2:])):
            best = (kk, um, unA)
    kk, um, unA = best
    e['matched_ef'] = int(kk[2:]); e['uni_avg'] = um['avg']; e['uni_p1'] = um['p1']
    e['uni_fail'] = um['fail09']
    e['fail_reduce'] = (unA.sum() - cf_ab.sum()) / unA.sum() * 100 if unA.sum() > 0 else 0.0
    e['mcnemar_p'] = mcnemar(cf_ab, unA)
    summary[name] = e
    print(f"=== {name} (n={n},dim={dim},queries/seed={qper},seeds={seeds}) ===")
    print(f"  baseline avg={e['base_avg']:.4f}")
    for cl, t in e['traj'].items():
        print(f"  churn{cl:>2}: avg={t['avg']:.4f} CI=[{t['ci'][0]:.4f},{t['ci'][1]:.4f}] "
              f"p1={t['p1']:.3f} rob09={t['rob09']:.3f} fail09={t['fail09']}")
    print(f"  AUC mean={e['auc']:.3f} CI={[round(x,3) for x in e['auc_ci']]} seeds={[round(a,3) for a in aucs]}")
    print(f"  nfail40/seeds={e['nfail40_seeds']}")
    print(f"  CF(ef~{e['cf_avg_ef']:.0f}): avg={e['cf_avg']:.4f} p1={e['cf_p1']:.3f} fail={e['cf_fail']}")
    print(f"  uniform ef{e['matched_ef']}: avg={e['uni_avg']:.4f} p1={e['uni_p1']:.3f} fail={e['uni_fail']} "
          f"| failRed={e['fail_reduce']:.1f}% McNemar p={e['mcnemar_p']:.4f}")
    print()

json.dump(summary, open(ROOT / 'data' / 'honest_summary.json', 'w'), indent=1)
print("[written] data/honest_summary.json")
