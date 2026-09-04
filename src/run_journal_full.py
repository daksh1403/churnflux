#!/usr/bin/env python3
"""
ChurnFlux JSA journal — HONEST real-data experiment at system-appropriate scale.

Runs baseline + churn(+20,40%) + instability predictor + matched-cost adaptive
recovery for 4 real datasets x N seeds at 100K vectors / 2000 queries (Fashion 60K).
Saves per-query recalls incrementally to data/journal_full_results.json.
"""
import os, time, json, numpy as np, hnswlib, h5py
from math import comb
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]        # dbms_research
HERE = Path(__file__).resolve().parent             # churnflux_journal/src
DATA = HERE.parent / 'data'
OUT = DATA / 'journal_full_results.json'

DATASETS = {
    'SIFT-128':     ('sift-128-euclidean.hdf5',       100000),
    'Fashion-MNIST':('fashion-mnist-784-euclidean.hdf5', 60000),
    'GloVe-200':    ('glove-200-angular.hdf5',        100000),
    'SIFT-1M':      ('sift-1m-euclidean.hdf5',        100000),
}
K = 10
EF = 50                      # base effort
QUERIES = 2000
CHURN = [20, 40]
st = time.time()

def load(fn, maxv, maxq, seed):
    rng = np.random.RandomState(seed)
    with h5py.File(ROOT / fn, 'r') as h:
        tr = h['train'][:]; te = h['test'][:maxq]
    if maxv < len(tr):
        ix = rng.choice(len(tr), maxv, replace=False); tr = tr[ix]
    return np.ascontiguousarray(tr, np.float32), np.ascontiguousarray(te, np.float32)

def gt(bv, bl, q, k, batch=200):
    g = np.empty((q.shape[0], k), np.int64)
    for i in range(0, q.shape[0], batch):
        qb = q[i:i+batch]
        d = np.sum(qb**2, 1, keepdims=True) + np.sum(bv**2, 1) - 2*(qb @ bv.T)
        ix = np.argpartition(d, k, 1)[:, :k]
        for r in range(ix.shape[0]):
            g[i+r] = bl[ix[r][np.argsort(d[r, ix[r]])]]
    return g

def build_idx(data, labels, dim):
    idx = hnswlib.Index(space='l2', dim=dim)
    idx.init_index(max_elements=len(data)*2, ef_construction=200, M=16,
                   allow_replace_deleted=True)
    idx.add_items(data, labels, num_threads=1)
    return idx

def recall(idx, q, gl, k, ef):
    idx.set_ef(int(ef))
    res, _ = idx.knn_query(q, k=k, num_threads=1)
    return np.array([len(set(res[i]) & set(gl[i])) / k for i in range(q.shape[0])])

def metrics(r):
    return dict(avg=float(np.mean(r)), p1=float(np.percentile(r, 1)),
                p5=float(np.percentile(r, 5)), rob09=float(np.mean(r >= 0.9)),
                rob08=float(np.mean(r >= 0.8)), fail09=int(np.sum(r < 0.9)),
                fail05=int(np.sum(r < 0.5)))

def run_one(name, data, q, seed):
    rng = np.random.RandomState(seed + 1000)
    n, dim = data.shape
    labels = np.arange(n, dtype=np.int64)
    out = {}
    # ---- baseline (no churn) ----
    idx = build_idx(data, labels, dim)
    gl = gt(data, labels, q, K)
    r0 = recall(idx, q, gl, K, EF)
    out['baseline'] = {**metrics(r0), 'recalls': r0}
    # ---- churn ----
    out['churn'] = {}
    for cp in CHURN:
        del_n = int(cp/100*n)
        ded = rng.choice(labels, del_n, replace=False)
        idx = build_idx(data, labels, dim)
        for d in ded: idx.mark_deleted(int(d))
        new_data = data[rng.choice(np.arange(n), del_n, replace=False)] + \
                   rng.randn(del_n, dim).astype(np.float32)*0.3
        new_labels = np.arange(n, n+del_n, dtype=np.int64)
        idx.add_items(new_data, new_labels, replace_deleted=True, num_threads=1)
        keep = ~np.isin(labels, ded)
        live_v = np.concatenate([data[keep], new_data])
        live_l = np.concatenate([labels[keep], new_labels])
        gl = gt(live_v, live_l, q, K)
        r = recall(idx, q, gl, K, EF)
        out['churn'][str(cp)] = {**metrics(r), 'recalls': r}
    # ---- predictor + recovery at 40% (reuse last built 40% index) ----
    idx.set_ef(30); rl, _ = idx.knn_query(q, k=K, num_threads=1)
    idx.set_ef(60); rh, _ = idx.knn_query(q, k=K, num_threads=1)
    inst = np.array([1 - len(set(rl[i])&set(rh[i])) / len(set(rl[i])|set(rh[i]))
                     if set(rl[i])|set(rh[i]) else 1.0 for i in range(len(q))])
    rc = out['churn']['40']['recalls']
    lab = (rc < 0.9).astype(int)
    n_fail = int(lab.sum())
    auc = float(roc_auc_score(lab, inst)) if 0 < n_fail < len(q) else float('nan')
    mh = float(np.mean(inst[lab == 0])) if (lab == 0).any() else float('nan')
    mf = float(np.mean(inst[lab == 1])) if n_fail > 0 else float('nan')
    out['predictor'] = dict(auc=auc, n_fail=n_fail, gap=mf-mh, instability=inst, labels=lab)
    # uniform + adaptive at matched cost
    uniform = {}
    for ef in [30, 50, 60, 80]:
        uniform[f'ef{ef}'] = {**metrics(recall(idx, q, gl, K, ef))}
    tau = np.percentile(inst, 80)
    un = inst >= tau
    allres = np.zeros((len(q), K), np.int64)
    idx.set_ef(100); rb, _ = idx.knn_query(q[un], k=K, num_threads=1)
    idx.set_ef(50);  rs, _ = idx.knn_query(q[~un], k=K, num_threads=1)
    allres[un] = rb; allres[~un] = rs
    rc2 = np.array([len(set(allres[i]) & set(gl[i])) / K for i in range(len(q))])
    avg_ef = 50 + 50*un.sum()/len(q)
    out['recovery'] = dict(churnflux={**metrics(rc2), 'recalls': rc2},
                           avg_ef=float(avg_ef), n_unstable=int(un.sum()),
                           uniform=uniform)
    return out

def main():
    ap = _argparse()
    args = ap.parse_args()
    res = {}
    if OUT.exists():
        res = json.load(open(OUT))
    only = args.only
    for seed in args.seeds:
        for name, (fn, maxv) in DATASETS.items():
            if only and name != only: continue
            data, q = load(fn, maxv, args.queries, seed)
            res.setdefault(name, {})
            key = f"n{maxv}_q{args.queries}_{seed}"
            if key in res[name]: print(f"skip {name} {key}"); continue
            t = time.time()
            res[name][key] = run_one(name, data, q, seed)
            json.dump(res, open(OUT, 'w'), default=_def)
            print(f"[{time.time()-st:6.0f}s] {name} {key} done in {time.time()-t:.0f}s "
                  f"auc={res[name][key]['predictor']['auc']}", flush=True)
    json.dump(res, open(OUT, 'w'), default=_def)
    print("ALL DONE", flush=True)

def _argparse():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43, 44])
    p.add_argument('--queries', type=int, default=QUERIES)
    p.add_argument('--only', default=None)
    p.add_argument('--seed', type=int, default=None)
    return p

def _def(o):
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.bool_): return bool(o)
    return str(o)

if __name__ == '__main__':
    main()
