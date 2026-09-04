#!/usr/bin/env python3
"""
ChurnFlux Journal - statistics & analytics.

Loads journal_results_master.json (per-query recalls across seeds) and computes:
  - pooled per-query metrics and bootstrap 95% CIs
  - predictor AUC (+ per-seed spread)
  - matched-cost recovery, McNemar significance ChurnFlux vs comparable uniform
  - failure-reduction effect sizes
Writes reports/statistics.md and reports/paper_numbers.json.
"""
import json, math
from pathlib import Path
import numpy as np
from math import comb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'journal_results_master.json'
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)

RNG = np.random.RandomState(0)
NBOOT = 2000


def load():
    with open(DATA) as f:
        return json.load(f)


def gather(rec, path):
    """Concatenate per-query arrays at dict-path 'path' across seeds."""
    parts = []
    for s in rec.values():
        node = s
        for p in path:
            node = node[p]
        parts.append(np.asarray(node, dtype=np.float64))
    return np.concatenate(parts)


def boot_ci(x, stat):
    x = np.asarray(x, dtype=np.float64)
    vals = np.empty(NBOOT)
    for i in range(NBOOT):
        s = RNG.choice(x, size=len(x), replace=True)
        vals[i] = stat(s)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mean_ci(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = boot_ci(x, np.mean)
    return float(np.mean(x)), lo, hi


def analyze():
    master = load()
    lines = []
    lines.append("# ChurnFlux Journal - Statistical Analysis (REAL DATA)\n")
    lines.append("Pooled across 3 seeds (seeds 41,42,43). "
                 "Bootstrap 95% CIs over queries (2000 resamples).\n")

    summary = {}
    for name, rec in master.items():
        scale = rec['41']['scale']; dim = rec['41']['dim']
        lines.append(f"\n## {name}  (n={scale}, dim={dim}, K=10)\n")

        b_avgs = [rec[s]['hnsw']['baseline']['avg'] for s in rec]
        b_avg = float(np.mean(b_avgs))
        churn_levels = sorted({int(k) for s in rec.values() for k in s['hnsw']['churn']})
        avg_curves = {}
        lines.append("### Churn trajectory (avg recall, ef=50)\n")
        lines.append("| Churn % | avg | 95% CI |")
        lines.append("|---|---|---|")
        lines.append(f"| 0 | {b_avg:.4f} | (seed-mean) |")
        for cl in churn_levels:
            av, lo, hi = mean_ci(gather(rec, ['hnsw','churn', str(cl), 'recalls']))
            avg_curves[cl] = av
            lines.append(f"| {cl} | {av:.4f} | [{lo:.4f}, {hi:.4f}] |")

        c40_fail = [rec[s]['hnsw']['churn']['40']['metrics']['n_fail_09'] for s in rec]
        c40_rob = [rec[s]['hnsw']['churn']['40']['metrics']['rob_09'] for s in rec]
        aucs = [float(rec[s]['hnsw']['predictor']['auc']) for s in rec
                if not (isinstance(rec[s]['hnsw']['predictor']['auc'], float)
                        and math.isnan(rec[s]['hnsw']['predictor']['auc']))]
        auc_mean = float(np.mean(aucs)) if aucs else float('nan')
        lines.append(f"\n**Churn 40%:** failures(<0.9)/seed = {c40_fail}; "
                     f"Rob-0.9 = {[round(r,3) for r in c40_rob]}")
        lines.append(f"**Predictor AUC** per seed = "
                     f"{[round(float(rec[s]['hnsw']['predictor']['auc']),3) for s in rec]}; "
                     f"mean = {auc_mean:.3f}")

        cf_rec = gather(rec, ['hnsw','recovery', 'churnflux', 'recalls'])
        cf_fail = cf_rec < 0.9
        cf_avg, cf_lo, cf_hi = mean_ci(cf_rec)
        cf_avg_ef = float(np.mean([rec[s]['hnsw']['recovery']['avg_ef'] for s in rec]))
        lines.append(f"\n**ChurnFlux:** avg={cf_avg:.4f} CI=[{cf_lo:.4f}, {cf_hi:.4f}], "
                     f"failures={int(cf_fail.sum())}/{len(cf_fail)}, avg_ef={cf_avg_ef:.1f}\n")

        lines.append("| Method | avg recall | failures(<0.9) |")
        lines.append("|---|---|---|")
        lines.append(f"| ChurnFlux (avg ef≈{cf_avg_ef:.0f}) | {cf_avg:.4f} | {int(cf_fail.sum())} |")
        ef_keys = sorted({ef for s in rec.values()
                          for ef in s['hnsw']['recovery']['uniform']},
                         key=lambda x: int(x[2:]))
        uniform_rows = {}
        for efk in ef_keys:
            ur = gather(rec, ['hnsw','recovery', 'uniform', efk, 'recalls'])
            ua, _, _ = mean_ci(ur)
            uniform_rows[efk] = {'avg': ua, 'fail': int((ur < 0.9).sum())}
            lines.append(f"| uniform {efk} | {ua:.4f} | {int((ur<0.9).sum())} |")

        cf_un = min(ef_keys, key=lambda ef: abs(cf_avg_ef - int(ef[2:])))
        un_fail = gather(rec, ['hnsw','recovery', 'uniform', cf_un, 'recalls']) < 0.9
        pval, only_cf, only_un = mcnemar(cf_fail, un_fail)
        un_total = int(un_fail.sum())
        eff = (un_total - int(cf_fail.sum())) / un_total if un_total > 0 else float('nan')
        lines.append(f"\n**Matched-cost (ChurnFlux avg_ef={cf_avg_ef:.1f} vs uniform {cf_un}):** "
                     f"ChurnFlux failures={int(cf_fail.sum())}, uniform failures={un_total}. "
                     f"Reduction={eff*100:.0f}%  (discordant only-CF={only_cf}, only-unif={only_un}) "
                     f"**McNemar p={pval:.4f}**")

        summary[name] = {
            'scale': scale, 'dim': dim,
            'base_avg': b_avg, 'avg_curves': avg_curves,
            'churn40_fail': c40_fail, 'auc': auc_mean, 'auc_seeds': aucs,
            'cf_avg': cf_avg, 'cf_fail': int(cf_fail.sum()), 'cf_avg_ef': cf_avg_ef,
            'uniform': uniform_rows,
            'matched_ef': cf_un, 'mcnemar_p': pval, 'failure_reduction': eff,
            'n_queries': int(len(cf_rec)),
        }

    out = "\n".join(lines)
    (REPORTS / 'statistics.md').write_text(out)
    with open(REPORTS / 'paper_numbers.json', 'w') as f:
        json.dump(summary, f, indent=1)
    print(out)
    print("\n[written] reports/statistics.md, reports/paper_numbers.json")


if __name__ == '__main__':
    analyze()



if __name__ == '__main__':
    analyze()

    only_a = int((a_fail & ~b_fail).sum())
    only_b = int((~a_fail & b_fail).sum())
    n_disc = only_a + only_b
    if n_disc == 0:
        return 1.0, only_a, only_b
    k = min(only_a, only_b)
    p = 0.5
    pval = min(1.0, 2.0 * sum(comb(n_disc, i) * (p ** n_disc) for i in range(k + 1)))
    return pval, only_a, only_b
