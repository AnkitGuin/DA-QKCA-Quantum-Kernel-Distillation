import torch
from daqkca.quantum.measurement import finite_shot_kernel


def test_shot_convergence_and_symmetry():
    k=torch.tensor([[1.,.3],[.3,1.]],dtype=torch.float64)
    a=finite_shot_kernel(k,100,seed=44)
    b=finite_shot_kernel(k,100000,seed=44)
    assert torch.allclose(a,a.T)
    assert abs(b[0,1]-.3)<.02
