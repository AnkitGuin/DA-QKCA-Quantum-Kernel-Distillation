"""Differentiable fidelity and classical controls; counters = exact simulated quantities."""
from dataclasses import dataclass
import torch
from torch import nn
from .feature_map import QuantumFeatureMap, fidelity
from ..models.feature_selector import SparseTopKSelector, FixedFeatureSelector, DenseProjection
from ..models.student import DeepKernel


@dataclass
class ResourceCounters:
    states_simulated: int = 0
    kernel_pairs: int = 0  # includes diagonals; actual dense overlap products
    calls: int = 0
    shots: int = 0  # ideal simulation always 0

    def to_dict(self):
        return vars(self).copy()


class KernelTeacher(nn.Module):
    def __init__(self, d, n_qubits, depth=2, selector='sparse', fixed_indices=None,
                 trainable_map=True, ansatz='ring', hidden=24,
                 selector_initialization='random', selector_relaxation='ste',
                 remove_terminal=True):
        super().__init__()
        self.kind = selector
        self.n_qubits, self.depth = n_qubits, depth
        if selector == 'sparse':
            self.selector = SparseTopKSelector(d, n_qubits, initialization=selector_initialization,
                                              relaxation=selector_relaxation)
        elif selector == 'dense':
            self.selector = DenseProjection(d, n_qubits)
        elif selector == 'fixed':
            self.selector = FixedFeatureSelector(fixed_indices or list(range(n_qubits)))
        elif selector in ('rbf', 'matern'):
            self.log_gamma = nn.Parameter(torch.tensor(-__import__("math").log(d), dtype=torch.float64))
        elif selector == 'deep':
            self.deep = DeepKernel(d, hidden=hidden)
        else:
            raise ValueError(selector)
        if selector in ('sparse','dense','fixed'):
            self.circuit = QuantumFeatureMap(n_qubits, depth, trainable_map, ansatz, remove_terminal)
        self.counters = ResourceCounters()
        self.register_buffer('landmarks', None)
        self.nystrom_ridge = 1e-5

    def forward(self, x, y=None):
        if self.landmarks is not None:
            # No N x N quantum Gram is computed. Returned minibatch geometry
            # remains dense, while exact kernel calls are b x m and m x m.
            a=self.exact_kernel(x,self.landmarks)
            b=a if y is None else self.exact_kernel(y,self.landmarks)
            w=self.exact_kernel(self.landmarks)
            reg=w+self.nystrom_ridge*torch.eye(len(w),dtype=w.dtype,device=w.device)
            return a@torch.linalg.solve(reg,b.T)
        return self.exact_kernel(x,y)

    def exact_kernel(self, x, y=None):
        if y is None:
            y = x
            symmetric = True
        else:
            symmetric = False
        self.counters.calls += 1
        self.counters.kernel_pairs += x.shape[0]*y.shape[0]
        if self.kind in ('sparse', 'dense', 'fixed'):
            a = self.circuit(self.selector(x))
            b = a if symmetric else self.circuit(self.selector(y))
            self.counters.states_simulated += x.shape[0] + (0 if symmetric else y.shape[0])
            return fidelity(a, b)
        if self.kind == 'deep':
            a = self.deep(x)
            b = a if symmetric else self.deep(y)
            return (a @ b.T + 1)/2  # valid PSD kernel: offset + linear PSD
        delta = torch.cdist(x.to(torch.float64), y.to(torch.float64))
        gamma = self.log_gamma.exp()
        if self.kind == 'rbf':
            return torch.exp(-gamma*delta.square())
        root3 = 3.**0.5
        return (1 + root3*gamma*delta)*torch.exp(-root3*gamma*delta)

    def selector_penalty(self):
        return self.selector.regularizer() if hasattr(self, 'selector') else torch.tensor(0., dtype=torch.float64)

    @property
    def gate_cost(self):
        return self.circuit.gate_cost if hasattr(self,'circuit') else {'qubits': 0,'depth_layers':0,'one_qubit_gates':0,'two_qubit_gates':0}
