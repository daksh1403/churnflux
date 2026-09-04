#!/usr/bin/env python3
"""
ChurnFlux Journal — authoritative real-data experiment pipeline.

Runs the core ChurnFlux study on REAL HDF5 datasets (sizes specified below),
saving per-query recall arrays (for CIs / histograms / significance tests),
aggregate metrics, latency, and predictor scores for every seed.

Protocol follows the known-good implementation in run_jsa_experiments_v3.py /
diagnostic_churn_v2.py, parameterized by scale and seed.

Usage:  python run_journal_experiments.py --seeds 41 42 43
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import hnswlib, faiss, h5py
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]          # dbms_research
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

DATASETS = {
    'SIFT-128':     ('sift-128-euclidean.hdf5',       100_000, 600),
    'Fashion-MNIST':('fashion-mnist-784-euclidean.hdf5', 60_000, 600),
    'GloVe-200':    ('glove-200-angular.hdf5',        100_000, 600),
    'SIFT-1M':      ('sift-1m-euclidean.hdf5',        100_000, 600),
}
CHURN_LEVELS = [0, 10, 20, 30, 40]
K = 10
EF_BASE = 50
BOOST_EF = 100
STABLE_EF = 50
FAST_FLAG = False
FAST_CFG = {'churn': [20, 40], 'efs': [30, 60, 80]}


def load_dataset(filename, max_vectors, max_queries, seed):
    rng = np.random.RandomState(seed)
    with h5py.File(ROOT / filename, 'r') as f:
        train = np.array(f['train'], dtype=np.float32)
        test = np.array(f['test'], dtype=np.float32)[:max_queries]
    if max_vectors and max_vectors < len(train):
        idx = rng.choice(len(train), max_vectors, replace=False)
        train = train[idx]
    return np.ascontiguousarray(train, dtype=np.float32), np.ascontiguousarray(test, dtype=np.float32)


def gt(bv, bl, q, k, batch=200):
    """Exact ground-truth k-NN returning LABELS."""
    g = np.empty((q.shape[0], k), dtype=np.int64)
    for i in range(0, q.shape[0], batch):
        qb = q[i:i+batch]
        d = np.sum(qb**2, axis=1, keepdims=True) + np.sum(bv**2, axis=1) - 2.0 * (qb @ bv.T)
        ix = np.argpartition(d, k, axis=1)[:, :k]
        for r in range(ix.shape[0]):
            order = np.argsort(d[r, ix[r]])
            g[i+r] = bl[ix[r][order]]
    return g


def recall_at(idx, q, gl, k, ef):
    idx.set_ef(int(ef))
    res, _ = idx.knn_query(q, k=k, num_threads=1)
    return np.array([len(set(res[i]) & set(gl[i])) / k for i in range(q.shape[0])], dtype=np.float64)


def metrics(recalls):
    return {
        'avg': float(np.mean(recalls)), 'p1': float(np.percentile(recalls, 1)),
        'p5': float(np.percentile(recalls, 5)), 'median': float(np.median(recalls)),
        'rob_09': float(np.mean(recalls >= 0.9)), 'rob_08': float(np.mean(recalls >= 0.8)),
        'n_fail_09': int(np.sum(recalls < 0.9)), 'n_fail_05': int(np.sum(recalls < 0.5)),
    }


def instability_scores(idx, queries, k, ef_low=30, ef_high=60):
    idx.set_ef(ef_low); r1, _ = idx.knn_query(queries, k=k, num_threads=1)
    idx.set_ef(ef_high); r2, _ = idx.knn_query(queries, k=k, num_threads=1)
    inst = np.zeros(len(queries))
    for i in range(len(queries)):
        s1, s2 = set(r1[i]), set(r2[i])
        u = s1 | s2
        inst[i] = 1.0 - (len(s1 & s2) / len(u) if len(u) else 1.0)
    return inst


def build_hnsw(data, labels, dim):
    idx = hnswlib.Index(space='l2', dim=dim)
    idx.init_index(max_elements=len(data) * 2 + len(data), ef_construction=200, M=16,
                   allow_replace_deleted=True)
    idx.add_items(data, labels, num_threads=1)
    return idx


def run_hnsw(name, data, queries, seed):
    rng = np.random.RandomState(seed + 1000)
    n, dim = len(data), data.shape[1]
    labels = np.arange(n, dtype=np.int64)
    out = {'baseline': {}, 'churn': {}, 'predictor': {}, 'recovery': {}}

    # ---- baseline ----
    idx = build_hnsw(data, labels, dim)
    gl0 = gt(data, labels, queries, K)
    r0 = recall_at(idx, queries, gl0, K, EF_BASE)
    out['baseline'] = metrics(r0)

    # ---- churn trajectory ----
    churn_levels = FAST_CFG['churn'] if FAST_FLAG else [lvl for lvl in CHURN_LEVELS if lvl > 0]
    saved40 = None
    for cp in churn_levels:
        del_n = int(cp / 100.0 * n)
        idx = build_hnsw(data, labels, dim)
        del_ids = rng.choice(labels, size=del_n, replace=False)
        for d in del_ids:
            idx.mark_deleted(int(d))
        new_data = data[del_ids] + rng.randn(del_n, dim).astype(np.float32) * 0.1
        new_labels = np.arange(n, n + del_n, dtype=np.int64)
        idx.add_items(new_data, new_labels, replace_deleted=True, num_threads=1)
        keep = ~np.isin(labels, del_ids)
        live_vectors = np.concatenate([data[keep], new_data])
        live_labels = np.concatenate([labels[keep], new_labels])
        gl = gt(live_vectors, live_labels, queries, K)
        rc = recall_at(idx, queries, gl, K, EF_BASE)
        out['churn'][str(cp)] = {'metrics': metrics(rc), 'recalls': rc}
        if cp == 40:
            saved40_idx, saved40_gl = idx, gl

    # ---- predictor + recovery at 40% churn (reuse the 40% index) ----
    idx, gl = saved40_idx, saved40_gl
    rc = out['churn']['40']['recalls']
    labels_bin = (rc < 0.9).astype(int)


    inst = instability_scores(idx, queries, K)
    n_fail = int(labels_bin.sum())
    if 0 < n_fail < len(queries):
        auc = float(roc_auc_score(labels_bin, inst))
    else:
        auc = float('nan')
    mean_h = float(np.mean(inst[labels_bin == 0])) if (labels_bin == 0).any() else float('nan')
    mean_f = float(np.mean(inst[labels_bin == 1])) if n_fail > 0 else float('nan')
    gap = mean_f - mean_h
    out['predictor'] = {'auc': auc, 'gap': gap, 'n_failing': n_fail,
                        'mean_healthy': mean_h, 'mean_failing': mean_f,
                        'instability': inst, 'labels_binary': labels_bin}



    # uniform baselines
    uniform = {}
    ef_list = FAST_CFG['efs'] if FAST_FLAG else [30, 50, 60, 80, 100]
    for ef in ef_list:
        ru = recall_at(idx, queries, gl, K, ef)
        uniform[f'ef{ef}'] = {'metrics': metrics(ru), 'recalls': ru}
    # ChurnFlux adaptive
    tau = np.percentile(inst, 80)
    unstable = inst >= tau
    n_un = int(unstable.sum())
    all_res = np.zeros((len(queries), K), dtype=np.int64)
    idx.set_ef(BOOST_EF); res_b, _ = idx.knn_query(queries[unstable], k=K, num_threads=1)
    idx.set_ef(STABLE_EF); res_s, _ = idx.knn_query(queries[~unstable], k=K, num_threads=1)
    all_res[unstable] = res_b; all_res[~unstable] = res_s
    rec_cf = np.array([len(set(all_res[i]) & set(gl[i])) / K for i in range(len(queries))],
                      dtype=np.float64)
    avg_ef = STABLE_EF + (BOOST_EF - STABLE_EF) * n_un / len(queries)
    out['recovery'] = {'churnflux': {'metrics': metrics(rec_cf), 'recalls': rec_cf},
                       'avg_ef': float(avg_ef), 'n_unstable': n_un,
                       'uniform': uniform}
    return out


def pick_m(dim, cap=32):
    """Largest M <= cap that divides dim (required by faiss IVF-PQ)."""
    for m in range(cap, 0, -1):
        if dim % m == 0:
            return m
    return 1


def run_ivfpq(name, data, queries, seed):
    rng = np.random.RandomState(seed + 5000)
    n, dim = len(data), data.shape[1]
    labels_all = np.arange(n, dtype=np.int64)
    nlist = min(256, max(8, int(np.sqrt(n))))
    m = min(32, dim // 4)
    m = pick_m(dim, m)                      # must divide dim (784, 200, ...)
    quantizer = faiss.IndexFlatL2(dim)
    idx = faiss.IndexIVFPQ(quantizer, dim, nlist, max(m, 1), 8)
    idx.train(data); idx.add(data)
    gt_data = gt(data, labels_all, queries, K)
    out = {'baseline': {}}
    idx.nprobe = 10
    _, res = idx.search(queries, K)
    r0 = np.array([len(set(res[i]) & set(gt_data[i])) / K for i in range(len(queries))],
                  dtype=np.float64)
    out['baseline'] = metrics(r0)
    out['churn'] = {}
    for cp in [20, 40]:   # lightweight secondary sweep to keep runtime in-budget
        n_churn = int(cp / 100.0 * n)
        churned = data.copy()
        ci = rng.choice(n, n_churn, replace=False)
        churned[ci] += rng.randn(n_churn, dim).astype(np.float32) * 0.5
        quant2 = faiss.IndexFlatL2(dim)
        idx_c = faiss.IndexIVFPQ(quant2, dim, nlist, max(m, 1), 8)
        idx_c.train(churned); idx_c.add(churned)
        gt_c = gt(churned, labels_all, queries, K)
        idx_c.nprobe = 10
        _, rc = idx_c.search(queries, K)
        r = np.array([len(set(rc[i]) & set(gt_c[i])) / K for i in range(len(queries))],
                     dtype=np.float64)
        out['churn'][str(cp)] = {'metrics': metrics(r), 'recalls': r}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43])
    ap.add_argument('--only', choices=list(DATASETS.keys()), default=None)
    ap.add_argument('--scale', type=int, default=None, help='override vector scale')
    ap.add_argument('--queries', type=int, default=500, help='number of query vectors')
    ap.add_argument('--fast', action='store_true',
                    help='reduced churn/ef sweep for heavy-dimension datasets')

    args = ap.parse_args()
    global FAST_FLAG
    FAST_FLAG = args.fast

    all_results = {}
    master_path = DATA_DIR / 'journal_results_master.json'
    if master_path.exists():
        with open(master_path) as f:
            all_results = json.load(f)
    for name, (fn, max_vec, _max_q) in DATASETS.items():
        if args.only and name != args.only:
            continue
        if args.scale:
            max_vec = args.scale
        all_results.setdefault(name, {})
        for seed in args.seeds:
            print(f"\n=== {name} seed={seed} (scale={max_vec}) ===", flush=True)
            data, queries = load_dataset(fn, max_vec, args.queries, seed)
            h = run_hnsw(name, data, queries, seed)
            i = run_ivfpq(name, data, queries, seed)
            all_results[name][str(seed)] = {'hnsw': h, 'ivfpq': i,
                                            'scale': len(data), 'dim': data.shape[1]}
            with open(master_path, 'w') as f:
                json.dump(all_results, f, indent=1, default=_json_default)
            print(f"  -> {name} seed={seed} AUC={h['predictor']['auc']:.3f} "
                  f"n_fail40={h['churn']['40']['metrics']['n_fail_09']} "
                  f"CF_fail={h['recovery']['churnflux']['metrics']['n_fail_09']}", flush=True)
    print("\nALL DONE")


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


if __name__ == '__main__':
    main()

