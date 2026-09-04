#!/usr/bin/env python3
"""Optimized experiments: AUC trajectory, scale, multi-seed, IVF-PQ analysis."""
import numpy as np, hnswlib, faiss, h5py, json, time, sys
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path(__file__).parent / 'comprehensive_results'
RESULTS_DIR.mkdir(exist_ok=True)
SEEDS = list(range(42, 52))
K = 10
EF_BASE = 50

def flush_print(*a, **kw):
    print(*a, **kw, flush=True)

def load_ds(fn, maxv=None, maxq=300):
    with h5py.File(fn, 'r') as f:
        tr = np.array(f['train']); te = np.array(f['test'])[:maxq]
        if maxv and maxv < len(tr):
            idx = np.random.choice(len(tr), maxv, replace=False); tr = tr[idx]
    return tr.astype(np.float32), te.astype(np.float32)

def gt(vectors, labels, queries, k):
    g = np.empty((queries.shape[0], k), dtype=np.int64)
    for i in range(0, queries.shape[0], 200):
        qb = queries[i:i+200]
        d = np.sum(qb**2, axis=1, keepdims=True) + np.sum(vectors**2, axis=1) - 2 * qb @ vectors.T
        ix = np.argpartition(d, k, axis=1)[:, :k]
        for r in range(ix.shape[0]): g[i+r] = labels[ix[r][np.argsort(d[r, ix[r]])]]
    return g

def recall_at(idx, queries, gl, k, ef):
    idx.set_ef(ef); res, _ = idx.knn_query(queries, k=k, num_threads=1)
    return np.array([len(set(res[i]) & set(gl[i])) / k for i in range(queries.shape[0])])

def metrics(r):
    return {'avg': float(np.mean(r)), 'p1': float(np.percentile(r,1)), 'p5': float(np.percentile(r,5)),
            'rob_09': float(np.mean(r>=0.9)), 'std': float(np.std(r)), 'var': float(np.var(r)),
            'n_fail': int(np.sum(r<0.9))}

def instability(idx, queries, k, ef):
    idx.set_ef(ef); r1, _ = idx.knn_query(queries, k=k, num_threads=1)
    idx.set_ef(ef*2); r2, _ = idx.knn_query(queries, k=k, num_threads=1)
    inst = np.zeros(len(queries))
    for i in range(len(queries)):
        s1, s2 = set(r1[i]), set(r2[i])
        inst[i] = 1 - (len(s1&s2)/len(s1|s2) if s1|s2 else 1)
    return inst

def apply_churn(data, labels, n, dim, pct, rng):
    dn = int(pct/100*n); did = rng.choice(labels, size=dn, replace=False)
    idx = hnswlib.Index(space='l2', dim=dim)
    idx.init_index(max_elements=n*2, ef_construction=200, M=16, allow_replace_deleted=True)
    idx.add_items(data, labels, num_threads=1)
    for d in did: idx.mark_deleted(int(d))
    nd = data[rng.choice(n, size=dn, replace=False)].copy()
    nd += rng.normal(0, 0.1, size=nd.shape).astype(np.float32)
    nl = np.arange(n, n+dn, dtype=np.int64)
    idx.add_items(nd, nl, replace_deleted=True, num_threads=1)
    keep = ~np.isin(labels, did)
    lv = np.concatenate([data[keep], nd]); ll = np.concatenate([labels[keep], nl])
    return idx, lv, ll, did

def mean_ci(vals):
    n = len(vals); m = np.mean(vals)
    if n < 2: return m, m, m
    se = sp_stats.sem(vals); h = se * sp_stats.t.ppf(0.975, n-1)
    return m, m-h, m+h

def safe_auc(y_true, y_score):
    if 0 < int(np.sum(y_true)) < len(y_true): return float(roc_auc_score(y_true, y_score))
    return None

# ── EXPERIMENT 1: AUC TRAJECTORY ──
def exp_auc_traj():
    flush_print("\n"+"#"*70)
    flush_print("EXP 1: AUC-vs-Churn Trajectory (SHEAF differentiation)")
    flush_print("#"*70)
    datasets = {'SIFT-128': ('sift-128-euclidean.hdf5', 50000), 'FMNIST': ('fashion-mnist-784-euclidean.hdf5', 50000)}
    churns = [5, 10, 20, 30, 40, 50]
    results = {}
    for dname, (fn, maxv) in datasets.items():
        flush_print(f"\n  === {dname} ===")
        res = {str(c): {'auc': [], 'nfail': [], 'avg': [], 'p1': [], 'var': []} for c in churns}
        for seed in SEEDS:
            rng = np.random.default_rng(seed); np.random.seed(seed)
            data, queries = load_ds(fn, maxv, 300)
            n = len(data); dim = data.shape[1]; labels = np.arange(n, dtype=np.int64)
            flush_print(f"    Seed {seed} ({n} vec, {len(queries)} q):")
            for cp in churns:
                t0 = time.time()
                idx, lv, ll, _ = apply_churn(data, labels, n, dim, cp, rng)
                gl = gt(lv, ll, queries, K)
                rc = recall_at(idx, queries, gl, K, EF_BASE)
                inst = instability(idx, queries, K, EF_BASE)
                lb = (rc < 0.9).astype(int)
                auc = safe_auc(lb, inst)
                key = str(cp)
                res[key]['auc'].append(auc); res[key]['nfail'].append(int(np.sum(lb)))
                res[key]['avg'].append(float(np.mean(rc))); res[key]['p1'].append(float(np.percentile(rc,1)))
                res[key]['var'].append(float(np.var(rc)))
                auc_s = f"{auc:.3f}" if auc else "N/A"
                flush_print(f"      Churn {cp:2d}%: AUC={auc_s}, fail={int(np.sum(lb))}, avg={np.mean(rc):.4f}, p1={np.percentile(rc,1):.3f} [{time.time()-t0:.0f}s]")
        summary = {}
        for key in res:
            aucs = [a for a in res[key]['auc'] if a is not None]
            if aucs:
                m, lo, hi = mean_ci(aucs)
                summary[key] = {'auc_mean': round(m,4), 'auc_ci': [round(lo,4), round(hi,4)], 'auc_std': round(float(np.std(aucs)),4),
                                'nfail_mean': round(float(np.mean(res[key]['nfail'])),1), 'avg_mean': round(float(np.mean(res[key]['avg'])),4),
                                'p1_mean': round(float(np.mean(res[key]['p1'])),4), 'var_mean': round(float(np.mean(res[key]['var'])),6)}
            else:
                summary[key] = {'auc_mean': None}
        flush_print(f"\n  {dname} Summary:")
        flush_print(f"  {'Churn%':>6} | {'AUC':>8} | {'95% CI':>16} | {'Fail%':>6}")
        for k in sorted(summary.keys(), key=int):
            s = summary[k]
            if s['auc_mean']:
                flush_print(f"  {k:>5}% | {s['auc_mean']:>8.4f} | [{s['auc_ci'][0]:.4f}, {s['auc_ci'][1]:.4f}] | {s['nfail_mean']:>5.1f}")
        results[dname] = {'summary': summary, 'raw': res}
    return results

# ── EXPERIMENT 2: SCALE ──
def exp_scale():
    flush_print("\n"+"#"*70)
    flush_print("EXP 2: Scale Experiment (SIFT-128)")
    flush_print("#"*70)
    scales = [20000, 50000, 100000, 200000]
    results = {}
    for sc in scales:
        flush_print(f"\n  === Scale {sc} ===")
        seed_res = []
        for seed in SEEDS[:5]:  # 5 seeds for speed
            rng = np.random.default_rng(seed); np.random.seed(seed)
            data, queries = load_ds('sift-128-euclidean.hdf5', sc, 200)
            n = len(data); dim = data.shape[1]; labels = np.arange(n, dtype=np.int64)
            t0 = time.time()
            idx, lv, ll, _ = apply_churn(data, labels, n, dim, 40, rng)
            gl = gt(lv, ll, queries, K)
            rc = recall_at(idx, queries, gl, K, EF_BASE)
            inst = instability(idx, queries, K, EF_BASE)
            lb = (rc < 0.9).astype(int)
            auc = safe_auc(lb, inst)
            tau = np.percentile(inst, 80); unstable = inst >= tau; nu = int(np.sum(unstable))
            idx.set_ef(EF_BASE*2); rb, _ = idx.knn_query(queries[unstable], k=K, num_threads=1)
            idx.set_ef(EF_BASE); rs, _ = idx.knn_query(queries[~unstable], k=K, num_threads=1)
            ar = np.zeros((len(queries), K), dtype=np.int64); ar[unstable] = rb; ar[~unstable] = rs
            rcf = np.array([len(set(ar[i])&set(gl[i]))/K for i in range(len(queries))])
            avg_ef = EF_BASE + EF_BASE*nu/len(queries); uef = int(round(avg_ef))
            idx.set_ef(uef); ru, _ = idx.knn_query(queries, k=K, num_threads=1)
            runif = np.array([len(set(ru[i])&set(gl[i]))/K for i in range(len(queries))])
            seed_res.append({'seed': seed, 'churn': metrics(rc), 'cf': metrics(rcf), 'unif': metrics(runif), 'auc': auc, 'avg_ef': avg_ef, 't': time.time()-t0})
            flush_print(f"    Seed {seed}: n={n}, fail={int(np.sum(lb))}, AUC={'%.3f'%auc if auc else 'N/A'}, p1={np.percentile(rc,1):.3f} [{time.time()-t0:.0f}s]")
        def ci_metric(key, stat):
            vals = [s[key][stat] for s in seed_res]; m, lo, hi = mean_ci(vals)
            return {'mean': round(m,4), 'ci': [round(lo,4), round(hi,4)], 'std': round(float(np.std(vals)),4)}
        results[str(sc)] = {'p1_churn': ci_metric('churn','p1'), 'p1_cf': ci_metric('cf','p1'), 'rob_churn': ci_metric('churn','rob_09'),
                            'rob_cf': ci_metric('cf','rob_09'), 'seeds': seed_res}
        flush_print(f"    Summary: p1_churn={results[str(sc)]['p1_churn']['mean']:.3f}, p1_cf={results[str(sc)]['p1_cf']['mean']:.3f}")
    return results

# ── EXPERIMENT 3: MULTI-SEED ──
def exp_multiseed():
    flush_print("\n"+"#"*70)
    flush_print("EXP 3: Multi-Seed Full Pipeline")
    flush_print("#"*70)
    datasets = {'SIFT-128': ('sift-128-euclidean.hdf5', 100000), 'FMNIST': ('fashion-mnist-784-euclidean.hdf5', 50000)}
    results = {}
    for dname, (fn, maxv) in datasets.items():
        flush_print(f"\n  === {dname} ({maxv} vectors) ===")
        seeds = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed); np.random.seed(seed)
            data, queries = load_ds(fn, maxv, 300)
            n = len(data); dim = data.shape[1]; labels = np.arange(n, dtype=np.int64)
            t0 = time.time()
            # Baseline
            idx_b = hnswlib.Index(space='l2', dim=dim)
            idx_b.init_index(max_elements=n*2, ef_construction=200, M=16, allow_replace_deleted=True)
            idx_b.add_items(data, labels, num_threads=1)
            gt0 = gt(data, labels, queries, K); r0 = recall_at(idx_b, queries, gt0, K, EF_BASE)
            # Churn 40%
            idx, lv, ll, _ = apply_churn(data, labels, n, dim, 40, rng)
            gl = gt(lv, ll, queries, K); rc = recall_at(idx, queries, gl, K, EF_BASE)
            inst = instability(idx, queries, K, EF_BASE)
            lb = (rc < 0.9).astype(int); auc = safe_auc(lb, inst)
            # Adaptive
            tau = np.percentile(inst, 80); unstable = inst >= tau; nu = int(np.sum(unstable))
            idx.set_ef(EF_BASE*2); rb, _ = idx.knn_query(queries[unstable], k=K, num_threads=1)
            idx.set_ef(EF_BASE); rs, _ = idx.knn_query(queries[~unstable], k=K, num_threads=1)
            ar = np.zeros((len(queries), K), dtype=np.int64); ar[unstable] = rb; ar[~unstable] = rs
            rcf = np.array([len(set(ar[i])&set(gl[i]))/K for i in range(len(queries))])
            avg_ef = EF_BASE + EF_BASE*nu/len(queries); uef = int(round(avg_ef))
            idx.set_ef(uef); ru, _ = idx.knn_query(queries, k=K, num_threads=1)
            runif = np.array([len(set(ru[i])&set(gl[i]))/K for i in range(len(queries))])
            idx.set_ef(40); r40, _ = idx.knn_query(queries, k=K, num_threads=1)
            rec40 = np.array([len(set(r40[i])&set(gl[i]))/K for i in range(len(queries))])
            seeds.append({'seed': seed, 'base': metrics(r0), 'churn': metrics(rc), 'cf': metrics(rcf), 'unif': metrics(runif), 'ef40': metrics(rec40), 'auc': auc, 't': time.time()-t0})
            flush_print(f"    Seed {seed}: base_avg={np.mean(r0):.4f} -> churn_avg={np.mean(rc):.4f}, p1={np.percentile(rc,1):.3f}, fail={int(np.sum(lb))}, AUC={'%.3f'%auc if auc else 'N/A'} [{time.time()-t0:.0f}s]")
        # Compute CIs
        def ci(stat, key=None):
            if key: vals = [s[key][stat] for s in seeds]
            else: vals = [s[stat] for s in seeds if s[stat] is not None]
            if not vals: return None
            m, lo, hi = mean_ci(vals)
            return {'mean': round(m,4), 'ci': [round(lo,4), round(hi,4)], 'std': round(float(np.std(vals)),4)}
        summary = {}
        for k in ['churn', 'cf', 'ef40']:
            for st in ['avg', 'p1', 'rob_09', 'var']:
                summary[f'{k}_{st}'] = ci(st, k)
        summary['auc'] = ci('auc')
        summary['nfail'] = ci('n_fail', 'churn')
        results[dname] = {'seeds': seeds, 'summary': summary}
        flush_print(f"\n  {dname} Summary (mean ± 95% CI):")
        for k in ['churn', 'cf']:
            for st in ['avg', 'p1', 'rob_09']:
                c = summary.get(f'{k}_{st}')
                if c: flush_print(f"    {k}_{st}: {c['mean']:.4f} [{c['ci'][0]:.4f}, {c['ci'][1]:.4f}]")
    return results

# ── EXPERIMENT 4: IVF-PQ CODEBOOK ──
def exp_ivfpq():
    flush_print("\n"+"#"*70)
    flush_print("EXP 4: IVF-PQ Codebook Analysis")
    flush_print("#"*70)
    datasets = {'SIFT-128': ('sift-128-euclidean.hdf5', 50000), 'FMNIST': ('fashion-mnist-784-euclidean.hdf5', 50000),
                'GloVe-200': ('glove-200-angular.hdf5', 50000)}
    results = {}
    for dname, (fn, maxv) in datasets.items():
        flush_print(f"\n  === {dname} ===")
        rng = np.random.default_rng(42)
        data, queries = load_ds(fn, maxv, 300)
        n = len(data); dim = data.shape[1]; labels = np.arange(n, dtype=np.int64)
        nlist = min(100, max(10, n//100))
        m_pq = 32
        while dim % m_pq != 0 and m_pq > 1: m_pq -= 1
        flush_print(f"  Building IVF-PQ: n={n}, dim={dim}, nlist={nlist}, m={m_pq}")
        qz = faiss.IndexFlatL2(dim); idx = faiss.IndexIVFPQ(qz, dim, nlist, m_pq, 8)
        idx.train(data); idx.add(data)
        centroids = faiss.rev_swig_ptr(qz.get_xb(), nlist * dim).reshape(nlist, dim).copy()
        _, assign = qz.search(data, 1); assign = assign.ravel()
        qe = np.linalg.norm(data - centroids[assign], axis=1)
        csizes = np.bincount(assign, minlength=nlist)
        flush_print(f"  Pre-churn: QE_mean={qe.mean():.4f}, cluster_std={csizes.std():.1f}")
        gt_data = gt(data, labels, queries, K)
        base = {}
        for np_ in [5, 10, 20]:
            idx.nprobe = np_; _, res = idx.search(queries, K)
            r = np.array([len(set(res[i])&set(gt_data[i]))/K for i in range(len(queries))])
            base[f'nprobe{np_}'] = metrics(r)
            flush_print(f"  Baseline nprobe={np_}: avg={np.mean(r):.4f}")
        churn = {}
        for cp in [10, 20, 30, 40, 50]:
            nc = int(cp/100*n); ci = rng.choice(n, nc, replace=False)
            cd = data.copy(); cd[ci] += rng.normal(0, 0.5, (nc, dim)).astype(np.float32)
            qzc = faiss.IndexFlatL2(dim); ic = faiss.IndexIVFPQ(qzc, dim, nlist, m_pq, 8)
            ic.train(cd); ic.add(cd)
            cc = faiss.rev_swig_ptr(qzc.get_xb(), nlist * dim).reshape(nlist, dim).copy()
            _, ac = qzc.search(cd, 1); ac = ac.ravel()
            qec = np.linalg.norm(cd - cc[ac], axis=1)
            ce_churned = qec[ci]; ce_unchanged = qec[np.ones(n, bool) & ~np.isin(np.arange(n), ci)]
            gt_c = gt(cd, labels, queries, K)
            ic.nprobe = 10; _, res = ic.search(queries, K)
            rc = np.array([len(set(res[i])&set(gt_c[i]))/K for i in range(len(queries))])
            churn[str(cp)] = {'metrics': metrics(rc), 'qe_all': float(qec.mean()), 'qe_churned': float(ce_churned.mean()),
                              'qe_unchanged': float(ce_unchanged.mean()), 'assign_changes': int(np.sum(assign != ac))}
            flush_print(f"  Churn {cp}%: avg={np.mean(rc):.4f}, QE_churned={ce_churned.mean():.4f}, QE_unch={ce_unchanged.mean():.4f}, reassign={np.sum(assign!=ac)}")
        results[dname] = {'config': {'nlist': nlist, 'm': m_pq}, 'pre': {'qe_mean': float(qe.mean()), 'baseline': base}, 'churn': churn}
    return results

# ── MAIN ──
if __name__ == '__main__':
    all_results = {}
    t_start = time.time()
    
    flush_print("Starting optimized experiments...")
    flush_print(f"Seeds: {SEEDS}, K={K}, EF_BASE={EF_BASE}")
    
    all_results['auc_trajectory'] = exp_auc_traj()
    with open(RESULTS_DIR/'partial_auc.json', 'w') as f: json.dump(all_results, f, indent=2, default=str)
    flush_print("\n  [AUC trajectory saved]")
    
    all_results['scale'] = exp_scale()
    with open(RESULTS_DIR/'partial_scale.json', 'w') as f: json.dump(all_results, f, indent=2, default=str)
    flush_print("\n  [Scale saved]")
    
    all_results['multiseed'] = exp_multiseed()
    with open(RESULTS_DIR/'partial_multiseed.json', 'w') as f: json.dump(all_results, f, indent=2, default=str)
    flush_print("\n  [Multi-seed saved]")
    
    all_results['ivfpq'] = exp_ivfpq()
    
    total = time.time() - t_start
    flush_print(f"\n\nTotal time: {total/60:.1f} minutes")
    
    with open(RESULTS_DIR/'experiments_v2.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.float64, np.float32)) else int(x) if isinstance(x, (np.integer, np.int64, np.int32)) else x)
    flush_print(f"Results saved to {RESULTS_DIR/'experiments_v2.json'}")
