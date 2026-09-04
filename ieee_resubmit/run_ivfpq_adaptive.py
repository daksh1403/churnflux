#!/usr/bin/env python3
"""IVF-PQ adaptive recovery: apply ChurnFlux probe-and-boost to IVF-PQ by
boosting nprobe for flagged queries, vs matched-cost uniform nprobe.
Same churn protocol as the HNSW experiments. 5 seeds, SIFT-1M + Fashion-MNIST.
"""
import numpy as np, faiss, h5py, json, time, os, sys
from pathlib import Path
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent
DATADIR = ROOT.parent
K = 10
NPROBE_BASE = 10
NPROBE_BOOST = 40
CHURNS = [5, 10, 20, 30, 40, 50]
SEEDS = list(range(42, 47))
NT = os.cpu_count()
DATASETS = {
    'SIFT-1M': ('sift-1m-euclidean.hdf5', None),
    'Fashion-MNIST': ('fashion-mnist-784-euclidean.hdf5', None),
}


def flush(*a, **kw):
    print(*a, **kw, flush=True)


def load_ds(fn, maxv=None, maxq=300):
    with h5py.File(DATADIR / fn, 'r') as f:
        tr = np.array(f['train'])
        te = np.array(f['test'])[:maxq]
        if maxv and maxv < len(tr):
            tr = tr[np.random.RandomState(0).choice(len(tr), maxv, replace=False)]
    return tr.astype(np.float32), te.astype(np.float32)


def gt(vectors, labels, queries, k):
    g = np.empty((len(queries), k), dtype=np.int64)
    nq = len(queries)
    chunk = max(1, int(np.ceil(nq / NT)))
    for i in range(0, nq, chunk):
        qb = queries[i:i + chunk]
        d = np.sum(qb ** 2, axis=1, keepdims=True) + np.sum(vectors ** 2, axis=1) - 2 * qb @ vectors.T
        ix = np.argpartition(d, k, axis=1)[:, :k]
        for r in range(len(ix)):
            g[i + r] = labels[ix[r][np.argsort(d[r, ix[r]])]]
    return g


def metrics(r):
    return {'avg': float(np.mean(r)), 'p1': float(np.percentile(r, 1)),
            'p5': float(np.percentile(r, 5)), 'rob_09': float(np.mean(r >= 0.9)),
            'n_fail': int(np.sum(r < 0.9))}


def mean_ci(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    n = len(vals)
    m = float(np.mean(vals))
    if n < 2:
        return {'mean': round(m, 4), 'ci': None, 'std': None}
    se = sp_stats.sem(vals)
    h = se * sp_stats.t.ppf(0.975, n - 1)
    return {'mean': round(m, 4), 'ci': [round(m - h, 4), round(m + h, 4)], 'std': round(float(np.std(vals)), 4)}


def instability_ivf(idx, qz, queries, k):
    """Answer-set flux for IVF-PQ: Jaccard disagreement between nprobe=10 and nprobe=20."""
    idx.nprobe = NPROBE_BASE
    _, r1 = idx.search(queries, k)
    idx.nprobe = NPROBE_BASE * 2
    _, r2 = idx.search(queries, k)
    inst = np.zeros(len(queries))
    for i in range(len(queries)):
        s1, s2 = set(r1[i]), set(r2[i])
        inst[i] = 1 - (len(s1 & s2) / len(s1 | s2) if s1 | s2 else 1)
    return inst


def exp_ivfpq_adaptive(dname):
    flush(f"\n{'#'*70}\nIVF-PQ ADAPTIVE: {dname}\n{'#'*70}")
    fn, maxv = DATASETS[dname]
    results = {str(c): {'churn': [], 'adapt': [], 'uniform': []} for c in CHURNS}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        data, queries = load_ds(fn, maxv, 300)
        n = len(data)
        dim = data.shape[1]
        labels = np.arange(n, dtype=np.int64)
        nlist = min(1000, max(100, n // 200))
        m_pq = 32
        while dim % m_pq != 0 and m_pq > 1:
            m_pq -= 1
        t0 = time.time()
        for cp in CHURNS:
            dn = int(cp / 100 * n)
            did = rng.choice(labels, size=dn, replace=False)
            nd = data[rng.choice(n, size=dn, replace=False)].copy()
            nd += rng.normal(0, 0.1, size=nd.shape).astype(np.float32)
            nl = np.arange(n, n + dn, dtype=np.int64)
            keep = ~np.isin(labels, did)
            lv = np.concatenate([data[keep], nd])
            ll = np.concatenate([labels[keep], nl])
            qz = faiss.IndexFlatL2(dim)
            ic = faiss.IndexIVFPQ(qz, dim, nlist, m_pq, 8)
            ic.train(lv)
            ic.add_with_ids(lv, ll)
            gl = gt(lv, ll, queries, K)
            # churn baseline
            ic.nprobe = NPROBE_BASE
            _, res = ic.search(queries, K)
            rc = np.array([len(set(res[i]) & set(gl[i])) / K for i in range(len(queries))])
            # instability
            inst = instability_ivf(ic, qz, queries, K)
            tau = np.percentile(inst, 80)
            unstable = inst >= tau
            nu = int(np.sum(unstable))
            # adaptive: boost nprobe for unstable (index-safe: keep original query order)
            idx_unstable = np.where(unstable)[0]
            idx_stable = np.where(~unstable)[0]
            ar = np.zeros((len(queries), K), dtype=np.int64)
            if len(idx_unstable) > 0:
                ic.nprobe = NPROBE_BOOST
                _, rb = ic.search(queries[idx_unstable], K)
                ar[idx_unstable] = rb
            if len(idx_stable) > 0:
                ic.nprobe = NPROBE_BASE
                _, rs = ic.search(queries[idx_stable], K)
                ar[idx_stable] = rs
            rcf = np.array([len(set(ar[i]) & set(gl[i])) / K for i in range(len(queries))])
            avg_nprobe = NPROBE_BASE + (NPROBE_BOOST - NPROBE_BASE) * nu / len(queries)
            u_nprobe = int(round(avg_nprobe))
            ic.nprobe = u_nprobe
            _, ru = ic.search(queries, K)
            runif = np.array([len(set(ru[i]) & set(gl[i])) / K for i in range(len(queries))])
            key = str(cp)
            results[key]['churn'].append(metrics(rc))
            results[key]['adapt'].append({**metrics(rcf), 'avg_nprobe': avg_nprobe})
            results[key]['uniform'].append({**metrics(runif), 'nprobe': u_nprobe})
            flush(f"  IVF-ADAPT {dname} seed {seed} churn {cp:2d}%: churn_avg={np.mean(rc):.4f} "
                  f"adapt_p1={np.percentile(rcf,1):.3f} unif_p1={np.percentile(runif,1):.3f} "
                  f"unstable={nu} avg_nprobe={avg_nprobe:.1f} [{time.time()-t0:.0f}s]")
    summary = {}
    for key in results:
        r = results[key]
        summary[key] = {
            'churn': {m: mean_ci([s[m] for s in r['churn']]) for m in ['avg', 'p1', 'p5', 'rob_09', 'n_fail']},
            'adapt': {m: mean_ci([s[m] for s in r['adapt']]) for m in ['avg', 'p1', 'p5', 'rob_09', 'n_fail']},
            'uniform': {m: mean_ci([s[m] for s in r['uniform']]) for m in ['avg', 'p1', 'p5', 'rob_09', 'n_fail']},
        }
    return {'config': {'index': 'IVF-PQ', 'nlist': nlist, 'm': m_pq, 'k': K, 'seeds': SEEDS,
                       'nprobe_base': NPROBE_BASE, 'nprobe_boost': NPROBE_BOOST, 'churns': CHURNS},
            'summary': summary, 'raw': results}


def main():
    out = {}
    for dname in DATASETS:
        out[f'IVFPQ_ADAPT_{dname}'] = exp_ivfpq_adaptive(dname)
    (ROOT / 'ivfpq_adaptive_results.json').write_text(json.dumps(out, indent=2))
    flush(f"\nSaved -> {ROOT / 'ivfpq_adaptive_results.json'}")


if __name__ == '__main__':
    main()
