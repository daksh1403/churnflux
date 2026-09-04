#!/usr/bin/env python3
"""
ChurnFlux full-scale experiments for IEEE Access resubmission.

Design targets (match the bar of published ANN papers in this space):
  - 3 real-scale datasets: SIFT-1M (1M), GloVe-200 (~1.18M), Fashion-MNIST (60K)
  - 2 index families: HNSW and IVF-PQ
  - Churn sweep: 5, 10, 20, 30, 40, 50% (cumulative, index built once per seed)
  - 10 seeds, 300 queries, K=10, all metrics reported as mean + 95% CI
  - Metrics: avg recall, p1, rob@0.9, fail count, AUC of instability predictor
  - Adaptive recovery vs matched-cost uniform baseline (per-seed matched ef)
  - Downstream RAG: recall@10 of adaptive vs uniform (matched ef)

All numbers written to ieee_resubmit/paper_results.json with config and CI.
"""
import numpy as np, hnswlib, faiss, h5py, json, time, os, sys, argparse
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
DATADIR = ROOT.parent  # datasets live in project root
RESULTS = ROOT / 'paper_results.json'

SEEDS = list(range(42, 52))
K = 10
EF_BASE = 50
CHURNS = [5, 10, 20, 30, 40, 50]
NT = os.cpu_count()

DATASETS = {
    'SIFT-1M': ('sift-1m-euclidean.hdf5', None),
    'GloVe-200': ('glove-200-angular.hdf5', None),
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


def recall_at(idx, queries, gl, k, ef):
    idx.set_ef(ef)
    res, _ = idx.knn_query(queries, k=k, num_threads=NT)
    return np.array([len(set(res[i]) & set(gl[i])) / k for i in range(len(queries))])


def metrics(r):
    return {
        'avg': float(np.mean(r)),
        'p1': float(np.percentile(r, 1)),
        'p5': float(np.percentile(r, 5)),
        'rob_09': float(np.mean(r >= 0.9)),
        'std': float(np.std(r)),
        'n_fail': int(np.sum(r < 0.9)),
    }


def instability(idx, queries, k, ef):
    idx.set_ef(ef)
    r1, _ = idx.knn_query(queries, k=k, num_threads=NT)
    idx.set_ef(ef * 2)
    r2, _ = idx.knn_query(queries, k=k, num_threads=NT)
    inst = np.zeros(len(queries))
    for i in range(len(queries)):
        s1, s2 = set(r1[i]), set(r2[i])
        inst[i] = 1 - (len(s1 & s2) / len(s1 | s2) if s1 | s2 else 1)
    return inst


def safe_auc(y_true, y_score):
    if 0 < int(np.sum(y_true)) < len(y_true):
        return float(roc_auc_score(y_true, y_score))
    return None


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


def build_hnsw(data, labels, dim):
    idx = hnswlib.Index(space='l2', dim=dim)
    idx.init_index(max_elements=len(data) * 2, ef_construction=200, M=16, allow_replace_deleted=True)
    idx.add_items(data, labels, num_threads=NT)
    return idx


def cumulative_churn(base_idx, base_data, base_labels, pct, rng):
    """Apply churn cumulatively to an existing index. Returns (idx, live_vecs, live_labels)."""
    n = len(base_data)
    dn = int(pct / 100 * n)
    did = rng.choice(base_labels, size=dn, replace=False)
    for d in did:
        base_idx.mark_deleted(int(d))
    nd = base_data[rng.choice(n, size=dn, replace=False)].copy()
    nd += rng.normal(0, 0.1, size=nd.shape).astype(np.float32)
    nl = np.arange(n, n + dn, dtype=np.int64)
    base_idx.add_items(nd, nl, replace_deleted=True, num_threads=NT)
    keep = ~np.isin(base_labels, did)
    lv = np.concatenate([base_data[keep], nd])
    ll = np.concatenate([base_labels[keep], nl])
    return base_idx, lv, ll, did


def exp_hnsw(datasets, dname, seeds, checkpoint=True):
    """Independent churn rates: fresh HNSW build per (seed, churn-rate).
    Each (seed, rate) is an independent condition; 10 seeds for the primary
    dataset, 5 for the others (keeps runtime tractable at 1M scale)."""
    flush(f"\n{'#'*70}\nHNSW: {dname} (seeds={seeds})\n{'#'*70}")
    fn, maxv = datasets[dname]
    ckpt = ROOT / f'_ckpt_hnsw_{dname}.json'
    if checkpoint and ckpt.exists():
        try:
            results = json.loads(ckpt.read_text())
            done_seeds = {s['seed'] for s in results.get('_done', [])}
            flush(f"  [resuming from checkpoint: {len(done_seeds)} seeds done]")
        except Exception:
            results = {str(c): {'auc': [], 'churn': [], 'adapt': [], 'uniform': []} for c in CHURNS}
            done_seeds = set()
    else:
        results = {str(c): {'auc': [], 'churn': [], 'adapt': [], 'uniform': []} for c in CHURNS}
        done_seeds = set()
    for seed in seeds:
        if seed in done_seeds:
            flush(f"  seed {seed} already done, skipping")
            continue
        rng = np.random.default_rng(seed)
        data, queries = load_ds(fn, maxv, 300)
        n = len(data)
        dim = data.shape[1]
        labels = np.arange(n, dtype=np.int64)
        t0 = time.time()
        for cp in CHURNS:
            # Independent churn condition: fresh index, churn this rate only
            idx = build_hnsw(data, labels, dim)
            idx, lv, ll, _ = cumulative_churn(idx, data, labels, cp, rng)
            gl_c = gt(lv, ll, queries, K)
            rc = recall_at(idx, queries, gl_c, K, EF_BASE)
            inst = instability(idx, queries, K, EF_BASE)
            lb = (rc < 0.9).astype(int)
            auc = safe_auc(lb, inst)
            # adaptive recovery: flag top-20% instability, boost ef*2, else ef
            tau = np.percentile(inst, 80)
            unstable = inst >= tau
            nu = int(np.sum(unstable))
            idx.set_ef(EF_BASE * 2)
            rb, _ = idx.knn_query(queries[unstable], k=K, num_threads=NT)
            idx.set_ef(EF_BASE)
            rs, _ = idx.knn_query(queries[~unstable], k=K, num_threads=NT)
            ar = np.zeros((len(queries), K), dtype=np.int64)
            ar[unstable] = rb
            ar[~unstable] = rs
            rcf = np.array([len(set(ar[i]) & set(gl_c[i])) / K for i in range(len(queries))])
            avg_ef = EF_BASE + EF_BASE * nu / len(queries)
            uef = int(round(avg_ef))
            ru = recall_at(idx, queries, gl_c, K, uef)
            key = str(cp)
            results[key]['auc'].append(auc)
            results[key]['churn'].append(metrics(rc))
            results[key]['adapt'].append({**metrics(rcf), 'avg_ef': avg_ef})
            results[key]['uniform'].append({**metrics(ru), 'ef': uef})
            flush(f"  {dname} seed {seed} churn {cp:2d}%: AUC={'%.3f' % auc if auc else 'N/A'} "
                  f"churn_avg={np.mean(rc):.4f} fail={int(np.sum(lb)):3d} "
                  f"adapt_p1={np.percentile(rcf,1):.3f} unif_p1={np.percentile(ru,1):.3f} [{time.time()-t0:.0f}s]")
        # checkpoint after each seed
        results.setdefault('_done', []).append({'seed': seed})
        if checkpoint:
            ckpt.write_text(json.dumps(results))
            flush(f"  [checkpoint saved: {len(results['_done'])} seeds]")
    # Summarize
    summary = {}
    for key in results:
        if key == '_done':
            continue
        r = results[key]
        aucs = [a for a in r['auc'] if a is not None]
        summary[key] = {
            'auc': mean_ci(aucs),
            'churn': {m: mean_ci([s[m] for s in r['churn']]) for m in ['avg', 'p1', 'p5', 'rob_09', 'n_fail']},
            'adapt': {m: mean_ci([s[m] for s in r['adapt']]) for m in ['avg', 'p1', 'p5', 'rob_09', 'n_fail']},
            'uniform': {m: mean_ci([s[m] for s in r['uniform']]) for m in ['avg', 'p1', 'p5', 'rob_09', 'n_fail']},
        }
    def _flat(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict) and 'mean' in v:
                out[k] = v['mean']
            elif isinstance(v, dict):
                out[k] = _flat(v)
            else:
                out[k] = v
        return out
    flush(f"  {dname} summary: " + json.dumps({k: _flat(summary[k]) for k in summary}))
    raw = {k: v for k, v in results.items() if k != '_done'}
    return {'config': {'index': 'HNSW', 'ef_base': EF_BASE, 'ef_boost': EF_BASE * 2, 'k': K, 'seeds': seeds, 'churns': CHURNS},
            'summary': summary, 'raw': raw}


def exp_ivfpq(dname, seeds):
    """IVF-PQ under the same churn protocol as HNSW (deletion + insertion).
    Reports recall and the quantization-error (QE) analysis that explains
    why IVF-PQ can behave differently from HNSW under churn."""
    flush(f"\n{'#'*70}\nIVF-PQ: {dname} (seeds={seeds})\n{'#'*70}")
    fn, maxv = DATASETS[dname]
    results = {str(c): {'recall': [], 'qe_all': [], 'qe_churned': [], 'qe_unchanged': []} for c in CHURNS}
    for seed in seeds:
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
            # Same churn protocol as HNSW: delete cp% of vectors, insert cp% replacements
            dn = int(cp / 100 * n)
            did = rng.choice(labels, size=dn, replace=False)
            nd = data[rng.choice(n, size=dn, replace=False)].copy()
            nd += rng.normal(0, 0.1, size=nd.shape).astype(np.float32)
            nl = np.arange(n, n + dn, dtype=np.int64)
            # live set
            keep = ~np.isin(labels, did)
            lv = np.concatenate([data[keep], nd])
            ll = np.concatenate([labels[keep], nl])
            # train IVF-PQ on live data
            qz = faiss.IndexFlatL2(dim)
            ic = faiss.IndexIVFPQ(qz, dim, nlist, m_pq, 8)
            ic.train(lv)
            ic.add_with_ids(lv, ll)
            gl_c = gt(lv, ll, queries, K)
            ic.nprobe = 10
            _, res = ic.search(queries, K)
            rc = np.array([len(set(res[i]) & set(gl_c[i])) / K for i in range(len(queries))])
            # QE analysis: assignment of live vectors to centroids
            _, assign = qz.search(lv, 1)
            assign = assign.ravel()
            centroids = faiss.rev_swig_ptr(qz.get_xb(), nlist * dim).reshape(nlist, dim).copy()
            qe = np.linalg.norm(lv - centroids[assign], axis=1)
            is_new = np.concatenate([np.zeros(len(data[keep]), dtype=bool), np.ones(dn, dtype=bool)])
            results[str(cp)]['recall'].append(float(np.mean(rc)))
            results[str(cp)]['qe_all'].append(float(qe.mean()))
            results[str(cp)]['qe_churned'].append(float(qe[is_new].mean()))
            results[str(cp)]['qe_unchanged'].append(float(qe[~is_new].mean()))
            flush(f"  IVF {dname} seed {seed} churn {cp:2d}%: avg={np.mean(rc):.4f} "
                  f"QE_new={qe[is_new].mean():.4f} QE_old={qe[~is_new].mean():.4f} [{time.time()-t0:.0f}s]")
    summary = {str(c): {m: mean_ci(results[str(c)][m]) for m in ['recall', 'qe_all', 'qe_churned', 'qe_unchanged']} for c in CHURNS}
    return {'config': {'index': 'IVF-PQ', 'nlist': nlist, 'm': m_pq, 'k': K, 'churns': CHURNS, 'seeds': seeds, 'nprobe': 10},
            'summary': summary, 'raw': results}


def load_rag(rag_fn):
    """Consolidate existing RAG downstream results (adaptive vs uniform, matched ef)."""
    import glob
    p = ROOT / rag_fn
    if not p.exists():
        p = Path('/Users/dakshagarwal/dbms_research/comprehensive_results') / rag_fn
    d = json.loads(p.read_text())
    out = {'seeds': [], 'summary': {}}
    for k, v in d.items():
        if not k.startswith('seed_'):
            continue
        rec = v.get('recovery', {})
        if 'adaptive' in rec and 'uniform' in rec:
            out['seeds'].append({
                'seed': int(k.split('_')[1]),
                'adaptive_recall': rec['adaptive'].get('recall'),
                'adaptive_p1': rec['adaptive'].get('p1'),
                'uniform_recall': rec['uniform'].get('recall'),
                'uniform_p1': rec['uniform'].get('p1'),
                'uniform_ef': rec['uniform'].get('ef'),
                'avg_ef': rec.get('avg_ef'),
            })
    for m in ['adaptive_recall', 'adaptive_p1', 'uniform_recall', 'uniform_p1']:
        vals = [s[m] for s in out['seeds'] if s[m] is not None]
        out['summary'][m] = mean_ci(vals)
    return out


def load_checkpoint(dname):
    """Load a checkpoint file if it exists; else None."""
    ckpt = ROOT / f'_ckpt_hnsw_{dname}.json'
    if ckpt.exists():
        try:
            return json.loads(ckpt.read_text())
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['hnsw', 'ivfpq', 'rag'], help='run only one experiment')
    ap.add_argument('--consolidate', action='store_true', help='build paper_results.json from checkpoints + rag, no compute')
    args = ap.parse_args()
    results = {}
    t_start = time.time()
    # Primary dataset gets full 10-seed CIs; secondary datasets use 5 seeds
    # to keep the 1M-scale run tractable. Config records this explicitly.
    seeds_full = SEEDS                      # 10 seeds
    seeds_partial = SEEDS[:5]               # 5 seeds
    if args.consolidate:
        # Rebuild from checkpoints (HNSW) + re-run IVF (fast) + load RAG
        for dname in DATASETS:
            ck = load_checkpoint(dname)
            if ck:
                # rebuild summary from checkpoint raw
                tmp = {str(c): {'auc': [], 'churn': [], 'adapt': [], 'uniform': []} for c in CHURNS}
                for key in ck:
                    if key == '_done':
                        continue
                    tmp[key] = ck[key]
                # recompute summary (reuse logic inline)
                summary = {}
                for key in tmp:
                    r = tmp[key]
                    aucs = [a for a in r['auc'] if a is not None]
                    summary[key] = {
                        'auc': mean_ci(aucs),
                        'churn': {m: mean_ci([s[m] for s in r['churn']]) for m in ['avg', 'p1', 'p5', 'rob_09', 'n_fail']},
                        'adapt': {m: mean_ci([s[m] for s in r['adapt']]) for m in ['avg', 'p1', 'p5', 'rob_09', 'n_fail']},
                        'uniform': {m: mean_ci([s[m] for s in r['uniform']]) for m in ['avg', 'p1', 'p5', 'rob_09', 'n_fail']},
                    }
                seeds = seeds_full if dname == 'SIFT-1M' else seeds_partial
                results[f'HNSW_{dname}'] = {'config': {'index': 'HNSW', 'ef_base': EF_BASE,
                                                       'ef_boost': EF_BASE * 2, 'k': K, 'seeds': seeds, 'churns': CHURNS},
                                            'summary': summary, 'raw': tmp}
                flush(f"  [consolidated HNSW_{dname} from checkpoint: {len(ck.get('_done', []))} seeds]")
            else:
                flush(f"  [WARN: no checkpoint for {dname}, re-running]")
                results[f'HNSW_{dname}'] = exp_hnsw(DATASETS, dname, seeds_full if dname == 'SIFT-1M' else seeds_partial)
        for dname in ['SIFT-1M', 'GloVe-200', 'Fashion-MNIST']:
            results[f'IVFPQ_{dname}'] = exp_ivfpq(dname, seeds_partial)
        results['RAG_downstream'] = load_rag('rag_results.json')
        results['meta'] = {'total_seconds': time.time() - t_start, 'seeds_full': seeds_full,
                           'seeds_partial': seeds_partial, 'k': K, 'ef_base': EF_BASE,
                           'churns': CHURNS, 'nt': NT, 'consolidated': True}
        RESULTS.write_text(json.dumps(results, indent=2))
        flush(f"\nConsolidated -> {RESULTS}")
        return
    if args.only in (None, 'hnsw'):
        for dname in DATASETS:
            seeds = seeds_full if dname == 'SIFT-1M' else seeds_partial
            results[f'HNSW_{dname}'] = exp_hnsw(DATASETS, dname, seeds)
    if args.only in (None, 'ivfpq'):
        for dname in ['SIFT-1M', 'GloVe-200', 'Fashion-MNIST']:
            results[f'IVFPQ_{dname}'] = exp_ivfpq(dname, seeds_partial)
    if args.only in (None, 'rag'):
        results['RAG_downstream'] = load_rag('rag_results.json')
    results['meta'] = {'total_seconds': time.time() - t_start, 'seeds_full': seeds_full,
                       'seeds_partial': seeds_partial, 'k': K, 'ef_base': EF_BASE,
                       'churns': CHURNS, 'nt': NT}
    RESULTS.write_text(json.dumps(results, indent=2))
    flush(f"\nTotal: {(time.time()-t_start)/60:.1f} min -> {RESULTS}")


if __name__ == '__main__':
    main()
