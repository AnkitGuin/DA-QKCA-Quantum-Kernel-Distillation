import torch
from daqkca.utils.seeding import set_seed
from daqkca.quantum.kernels import KernelTeacher


def test_repeatability():
    x=torch.randn(3,5,dtype=torch.float64)
    set_seed(5);k1=KernelTeacher(5,2,1)(x)
    set_seed(5);k2=KernelTeacher(5,2,1)(x)
    assert torch.equal(k1,k2)
