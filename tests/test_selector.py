import torch
from daqkca.models.feature_selector import SparseTopKSelector,DenseProjection


def test_unique_sparse_indices_and_ste_gradient():
    sel=SparseTopKSelector(9,3)
    x=torch.randn(5,9,dtype=torch.float64)
    indices=sel.selected_indices()
    assert len(indices)==len(set(indices))==3
    val=sel(x).square().sum();val.backward()
    assert sel.logits.grad.abs().sum()>0
    assert (sel.logits.grad.abs().sum(0)>0).sum()>3


def test_dense_not_sparse():
    assert DenseProjection(10,3)(torch.randn(4,10)).shape==(4,3)
