import time, numpy as np, h5py, hnswlib
t = time.time()
f = h5py.File('/Users/dakshagarwal/dbms_research/glove-200-angular.hdf5', 'r')
print('keys', list(f.keys()))
train = np.array(f['train'], dtype=np.float32)[:6000]
test = np.array(f['test'], dtype=np.float32)[:300]
print('load 6000x200', round(time.time()-t, 2))
t2 = time.time()
idx = hnswlib.Index(space='l2', dim=200)
idx.init_index(max_elements=12000, ef_construction=200, M=16, allow_replace_deleted=True)
idx.add_items(train, np.arange(6000, dtype=np.int64), num_threads=1)
print('build glove 6000', round(time.time()-t2, 2))
t3 = time.time()
idx.set_ef(50); res, _ = idx.knn_query(test, k=10, num_threads=1)
print('query', round(time.time()-t3, 2))
