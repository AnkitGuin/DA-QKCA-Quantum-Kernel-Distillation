import numpy as np
import torch
from daqkca.quantum.feature_map import QuantumFeatureMap,apply_one,terminal_shared_rx,fidelity
from daqkca.quantum.kernels import KernelTeacher
from daqkca.quantum.costs import symmetric_unique_pairs


def test_kernel_psd_symmetric_diagonal_and_trainability():
    torch.manual_seed(3)
    x=torch.randn(8,6,dtype=torch.float64)
    t=KernelTeacher(6,3,2)
    k=t(x)
    assert torch.allclose(k,k.T,atol=1e-12)
    assert torch.allclose(k.diag(),torch.ones(8,dtype=torch.float64),atol=1e-12)
    assert k.min()>-1e-12 and k.max()<1+1e-12
    assert torch.linalg.eigvalsh(k).min()>-1e-10
    k[0,1].backward()
    assert t.circuit.theta.grad.norm()>1e-8
    assert t.selector.logits.grad.norm()>1e-8


def test_terminal_cancellation():
    x=torch.randn(5,2,dtype=torch.float64)
    m=QuantumFeatureMap(2,1)
    st=m(x)
    shifted=terminal_shared_rx(st,0.83,1,2)
    assert torch.allclose(fidelity(st),fidelity(shifted),atol=1e-12)


def test_reference_two_qubit_np_unitaries():
    x=torch.tensor([[0.3,-0.8]],dtype=torch.float64)
    m=QuantumFeatureMap(2,1,ansatz='chain')
    angles=x.numpy()[0]
    theta=m.theta.detach().numpy()[0]
    def rot_y(a):return np.array([[np.cos(a/2),-np.sin(a/2)],[np.sin(a/2),np.cos(a/2)]],dtype=complex)
    def rot_x(a):return np.array([[np.cos(a/2),-1j*np.sin(a/2)],[-1j*np.sin(a/2),np.cos(a/2)]],dtype=complex)
    def rot_z(a):return np.diag([np.exp(-.5j*a),np.exp(.5j*a)])
    gates=[]
    for i in range(2):gates.append(rot_z(.6*angles[i])@rot_x(theta[i])@rot_y(.8*angles[i]))
    cnot=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    expected=np.kron(*gates)@np.array([1,0,0,0])
    assert np.allclose(m(x).detach().numpy()[0],expected,atol=1e-12)


def test_pair_count():
    assert symmetric_unique_pairs(45280)==1025161840


def test_alternative_ansatz_preserves_kernel_psd():
    torch.manual_seed(9)
    x=torch.randn(4,4,dtype=torch.float64)
    for ansatz in ['ring','chain']:
        t=KernelTeacher(4,3,2,ansatz=ansatz)
        k=t(x)
        assert torch.linalg.eigvalsh(k).min()>-1e-10
