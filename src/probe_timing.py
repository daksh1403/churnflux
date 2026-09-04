import time, numpy as np, h5py, hnswlib
t = time.time()
f = h5py.File('/Users/dakshagarwal/dbms_research/sift-128-euclidean.hdf5', 'r')
train = np.array(f['train'], dtype=np.float32)[:10000]
test = np.array(f['test'], dtype=np.float32)[:600]
f.close()
print('load', round(time.time()-t, 2))

t = time.time()
idx = hnswlib.Index(space='l2', dim=128)
idx.init_index(max_elements=20000, ef_construction=200, M=16, allow_replace_deleted=True)
idx.add_items(train, np.arange(10000, dtype=np.int64), num_threads=1)
print('build 10k', round(time.time()-t, 2))

def gt(bv, bl, q, k=10, batch=200):
    g = np.empty((q.shape[0], k), dtype=np.int64)
    for i in range(0, q.shape[0], batch):
        qb = q[i:i+batch]
        d = np.sum(qb**2, axis=1, keepdims=True) + np.sum(bv**2, axis=1) - 2*(qb @ bv.T)
        ix = np.argpartition(d, k, axis=1)[:, :k]
        for r in range(ix.shape[0]):
            g[i+r] = bl[ix[r][np.argsort(d[r, ix[r]])]]
    return g

t = time.time()
gl = gt(train, np.arange(10000, dtype=np.int64), test)
print('gt 600x10k', round(time.time()-t, 2))

t = time.time()
idx.set_ef(50); res, _ = idx.knn_query(test, k=10, num_threads=1)
print('query', round(time.time()-t, 2))

t = time.time()
rec = [len(set(res[i]) & set(gl[i]))/10 for i in range(600)]
print('recall loop', round(time.time()-t, 2))
