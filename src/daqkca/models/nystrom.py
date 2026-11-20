import torch


def nystrom(k_xl, k_ll, ridge=1e-5):
    m = k_ll.shape[0]
    reg = (k_ll+k_ll.T)/2 + ridge*torch.eye(m, dtype=k_ll.dtype, device=k_ll.device)
    # Cholesky from positive semidefinite ideal kernel, positive ridge.
    c = torch.linalg.cholesky(reg)
    return k_xl @ torch.cholesky_solve(k_xl.T, c)


def coordinates(k_xl, k_ll, ridge=1e-5):
    e, v = torch.linalg.eigh((k_ll+k_ll.T)/2)
    inv_root = (v*torch.rsqrt(e.clamp_min(0)+ridge)) @ v.T
    return k_xl @ inv_root


def landmark_indices(x, m, seed=0, method='random', labels=None):
    import numpy as np
    from sklearn.cluster import KMeans
    x = np.asarray(x)
    m = min(m, len(x))
    rng = np.random.default_rng(seed)
    if method == 'random':
        return rng.choice(len(x), m, replace=False).tolist()
    if method == 'stratified':
        if labels is None:
            raise ValueError('stratified requires labels')
        labels=np.asarray(labels)
        out=[]
        for cl in np.unique(labels):
            ids=np.flatnonzero(labels == cl)
            n = min(len(ids), round(m*len(ids)/len(labels)))
            out.extend(rng.choice(ids,n,replace=False).tolist())
        extra=np.setdiff1d(np.arange(len(x)),out)
        if len(out)<m:
            out.extend(rng.choice(extra,m-len(out),replace=False).tolist())
        return out[:m]
    if method=='kmeans':
        centers = KMeans(n_clusters=m, random_state=seed, n_init=5).fit(x).cluster_centers_
        # Unique representative indices; no full quantum kernel computed.
        chosen=set()
        for center in centers:
            for j in np.argsort(((x-center)**2).sum(1)):
                if int(j) not in chosen:
                    chosen.add(int(j));break
        return sorted(chosen)
    raise ValueError(method)
