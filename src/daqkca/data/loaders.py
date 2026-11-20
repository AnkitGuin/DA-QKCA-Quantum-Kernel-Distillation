"""Synthetic mechanistic benchmark and explicit user-provided CSV adapter.

Test is split off FIRST and never inspected by training command; no final
benchmark inference is claimed from this synthetic benchmark.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetParts:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    split_manifest: dict
    raw_train_x: np.ndarray | None = None
    raw_val_x: np.ndarray | None = None
    raw_test_x: np.ndarray | None = None


def synthetic(n=360, d=12, seed=101, task='xor'):
    if d < 4:
        raise ValueError('synthetic requires d >=4')
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,d))
    if task == 'xor':
        signal=x[:,0]*x[:,1] + 0.5*x[:,2] - 0.25*x[:,3]
    elif task == 'linear':
        signal=1.8*x[:,0] - x[:,1] + 0.3*x[:,2]
    else:
        raise ValueError('synthetic task must be xor or linear')
    y=(signal + 0.25*rng.normal(size=n) > 0).astype(np.int64)
    return x,y


def _hash_indices(ids):
    return hashlib.sha256(np.asarray(sorted(ids),dtype='<i8').tobytes()).hexdigest()


def leakage_audit(x, train, val, test):
    groups=[np.asarray(z,dtype=int) for z in (train,val,test)]
    if any(len(z)!=len(set(z.tolist())) for z in groups):
        raise AssertionError('duplicate indices within partition')
    if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i+1,3)):
        raise AssertionError('overlapping indices across partitions')
    # Exact duplicate vectors leak across partitions even if row IDs differ.
    groups_hash=[]
    for ids in groups:
        arr=np.ascontiguousarray(np.asarray(x[ids],dtype=np.float64))
        groups_hash.append({hashlib.sha256(row.tobytes()).digest() for row in arr})
    if any(groups_hash[i] & groups_hash[j] for i in range(3) for j in range(i+1,3)):
        raise AssertionError('duplicate feature vectors across partitions')
    return {'index_disjoint':True, 'exact_features_disjoint':True}


def split_and_scale(x,y, seed=101, test_fraction=.20, val_fraction=.20):
    ids=np.arange(len(y))
    trval, test=train_test_split(ids,test_size=test_fraction,stratify=y,random_state=seed)
    tr, val=train_test_split(trval,test_size=val_fraction/(1-test_fraction),
                             stratify=y[trval],random_state=seed+1)
    audit=leakage_audit(x,tr,val,test)
    scaler=StandardScaler().fit(x[tr])
    xp=scaler.transform(x).astype(np.float64)
    ids_by={'train':tr.tolist(), 'val':val.tolist(),'test':test.tolist()}
    h={k:_hash_indices(v) for k,v in ids_by.items()}
    manifest={'seed':seed, 'n_rows':len(x), 'shape':list(x.shape), 'hashes':h,
              'raw_data_sha256':hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()+np.ascontiguousarray(y,dtype='<i8').tobytes()).hexdigest(),
              'ids':ids_by, 'leakage_audit':audit,
              'preprocessing':'StandardScaler fit on training partition only',
              'test_locked':True}
    return DatasetParts(xp[tr],y[tr],xp[val],y[val],xp[test],y[test],manifest,
                        np.asarray(x[tr]),np.asarray(x[val]),np.asarray(x[test]))


def load_dataset(cfg, seed):
    spec=cfg.get('dataset',{})
    if spec.get('name','synthetic') == 'synthetic':
        x,y=synthetic(spec.get('n',360),spec.get('d',12),seed,spec.get('task','xor'))
    elif spec.get('name') == 'csv':
        src=Path(spec['path'])
        df=pd.read_csv(src)
        label=spec['label']
        if label not in df:
            raise ValueError(f'missing label {label}')
        group_column=spec.get('group_column')
        excluded=[label]+([group_column] if group_column else [])
        if df.drop(columns=excluded).select_dtypes(exclude='number').shape[1]:
            raise ValueError('CSV adapter requires numeric features; add train-fitted encoder')
        y=df[label].to_numpy()
        classes, y=np.unique(y,return_inverse=True)
        if len(classes)!=2:
            raise ValueError('prototype supports binary labels only')
        x=df.drop(columns=excluded).to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('missing/infinite values: add train-only imputer')
        if group_column:
            return split_group_and_scale(x,y,df[group_column].to_numpy(),seed)
    else:
        raise ValueError('unknown dataset')
    return split_and_scale(x,y,seed)


def split_group_and_scale(x,y,groups,seed=101):
    """Leakage-aware alternate for row-linked records; splits entire groups.

    Requires classes in all partitions; no synthetic per-row stratification can
    guarantee class balance in rare-group real datasets.
    """
    from sklearn.model_selection import GroupShuffleSplit
    ids=np.arange(len(y));groups=np.asarray(groups)
    if len(groups)!=len(y):raise ValueError('group vector length mismatch')
    first=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed)
    trainval_rel,test_rel=next(first.split(x,y,groups))
    second=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1)
    train_rel, val_rel=next(second.split(x[trainval_rel],y[trainval_rel],groups[trainval_rel]))
    tr,val,test=ids[trainval_rel][train_rel],ids[trainval_rel][val_rel],ids[test_rel]
    leakage_audit(x,tr,val,test)
    gsets=[set(groups[idx].tolist()) for idx in (tr,val,test)]
    if any(gsets[i]&gsets[j] for i in range(3) for j in range(i+1,3)):
        raise AssertionError('leaked group labels')
    if any(len(np.unique(y[idx]))<2 for idx in (tr,val,test)):
        raise ValueError('group split lacks both classes; choose defensible split')
    scaler=StandardScaler().fit(x[tr]);scaled=scaler.transform(x).astype(np.float64)
    ids_by={'train':tr.tolist(),'val':val.tolist(),'test':test.tolist()}
    manifest={'seed':seed,'n_rows':len(y),'shape':list(x.shape),
              'ids':ids_by,'hashes':{k:_hash_indices(v) for k,v in ids_by.items()},
              'test_locked':True,'group_disjoint':True,
              'raw_data_sha256':hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()+np.ascontiguousarray(y,dtype='<i8').tobytes()).hexdigest(),
              'preprocessing':'StandardScaler fit on train only'}
    return DatasetParts(scaled[tr],y[tr],scaled[val],y[val],scaled[test],y[test],manifest,
                        np.asarray(x[tr]),np.asarray(x[val]),np.asarray(x[test]))
