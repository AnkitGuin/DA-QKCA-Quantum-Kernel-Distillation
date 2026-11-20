import torch
from daqkca.models.nystrom import nystrom,coordinates
from daqkca.quantum.kernels import KernelTeacher


def test_nystrom_and_coordinates():
    torch.manual_seed(1)
    x=torch.randn(10,5,dtype=torch.float64)
    k=KernelTeacher(5,2)(x)
    approx=nystrom(k,k,ridge=1e-8)
    assert torch.allclose(approx,k,atol=1e-6)
    z=coordinates(k,k)
    assert z.shape==(10,10)
    assert torch.linalg.eigvalsh(approx).min()>-1e-8
