import pytest
import numpy as np
from daqkca.data.loaders import synthetic,split_and_scale,leakage_audit


def test_split_disjoint():
    x,y=synthetic(100,8,4)
    p=split_and_scale(x,y,4)
    a=p.split_manifest['ids']
    assert not set(a['train'])&set(a['val'])
    assert not set(a['val'])&set(a['test'])
    assert abs(p.train_x.mean())<1e-10


def test_detect_leakage_rows_and_indices():
    x=np.arange(32).reshape(8,4).astype(float)
    x[3]=x[0]
    with pytest.raises(AssertionError):leakage_audit(x,[0,1],[2,3],[4,5])
    with pytest.raises(AssertionError):leakage_audit(x,[0,1],[1,2],[4,5])


def test_group_split_disjoint():
    from daqkca.data.loaders import split_group_and_scale
    x,y=synthetic(n=200,d=5,seed=42)
    # repeat each group, distributed classes probabilistically
    g=np.arange(len(y))//4
    p=split_group_and_scale(x,y,g,seed=42)
    ids=p.split_manifest['ids']
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert set(g[ids[a]]).isdisjoint(set(g[ids[b]]))
