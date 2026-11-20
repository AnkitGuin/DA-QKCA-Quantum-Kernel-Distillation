"""Autodifferentiable exact pure-state simulations; no quantum hardware is used.

Qubit convention: qubit 0 = most significant computational bit.
Order matters: trainable Rx is INTERLEAVED with data encoding to avoid
fidelity-kernel cancellation of any shared terminal unitary.
"""
import math
import torch
from torch import nn


def _ry(a):
    c, s = torch.cos(a / 2), torch.sin(a / 2)
    return torch.stack([torch.stack([c, -s], -1),
                        torch.stack([s, c], -1)], -2).to(torch.complex128)


def _rx(a):
    c, s = torch.cos(a / 2), -1j * torch.sin(a / 2)
    return torch.stack([torch.stack([c, s], -1),
                        torch.stack([s, c], -1)], -2).to(torch.complex128)


def _rz(a):
    p, m = torch.exp(-0.5j * a), torch.exp(0.5j * a)
    z = torch.zeros_like(p)
    return torch.stack([torch.stack([p, z], -1),
                        torch.stack([z, m], -1)], -2).to(torch.complex128)


def apply_one(state: torch.Tensor, gate: torch.Tensor, qubit: int, n: int):
    b = state.shape[0]
    shaped = state.reshape(b, *([2] * n)).movedim(1 + qubit, -1)
    if gate.ndim == 2:
        gate = gate.expand(b, -1, -1)
    changed = torch.einsum('bij,b...j->b...i', gate, shaped)
    return changed.movedim(-1, 1 + qubit).reshape(b, -1)


def apply_cnot(state: torch.Tensor, control: int, target: int, n: int):
    if control == target:
        raise ValueError('control and target must differ')
    index = torch.arange(2**n, device=state.device)
    ctrl_on = ((index >> (n - 1 - control)) & 1).bool()
    src = index ^ (ctrl_on.long() << (n - 1 - target))
    return state[:, src]


class QuantumFeatureMap(nn.Module):
    def __init__(self, n_qubits=3, depth=2, trainable=True, ansatz='ring', remove_terminal=True):
        super().__init__()
        if not 1 <= n_qubits <= 10:
            raise ValueError('exact statevector simulator supports 1..10 qubits')
        if depth < 1:
            raise ValueError('depth must be positive')
        if ansatz not in ('ring', 'chain'):
            raise ValueError('ansatz must be ring or chain')
        self.n_qubits, self.depth, self.ansatz = n_qubits, depth, ansatz
        self.remove_terminal = remove_terminal
        self.theta = nn.Parameter(torch.randn(depth, n_qubits, dtype=torch.float64)*0.15,
                                  requires_grad=trainable)

    def forward(self, angles):
        if angles.ndim != 2 or angles.shape[1] != self.n_qubits:
            raise ValueError('angles must be [batch,n_qubits]')
        angles = angles.to(torch.float64)
        b, n = angles.shape
        state = torch.zeros((b, 2**n), dtype=torch.complex128, device=angles.device)
        # Out-of-place initialization ensures statevector derivation through gates.
        state = torch.cat([torch.ones((b, 1), dtype=torch.complex128, device=angles.device),
                           state[:, 1:]], dim=1)
        for l in range(self.depth):
            for q in range(n):
                state = apply_one(state, _ry(angles[:, q] * (0.8 + 0.1*l)), q, n)
                # Common unitary here is NOT terminal: next Rz reuploads data.
                state = apply_one(state, _rx(self.theta[l, q]), q, n)
                state = apply_one(state, _rz(angles[:, q] * (0.6 + 0.05*l)), q, n)
            if n > 1 and (l < self.depth-1 or not self.remove_terminal):
                for q in range(n - 1):
                    state = apply_cnot(state, q, q+1, n)
                if self.ansatz == 'ring' and n > 2:
                    state = apply_cnot(state, n-1, 0, n)
        return state

    @property
    def gate_cost(self):
        pairs = (self.n_qubits - 1) + int(self.n_qubits > 2 and self.ansatz == 'ring')
        return {'qubits': self.n_qubits, 'depth_layers': self.depth,
                'one_qubit_gates': 3*self.n_qubits*self.depth,
                'two_qubit_gates': pairs*(self.depth-int(self.remove_terminal)),
                'legacy_two_qubit_gates': pairs*self.depth,
                'removed_terminal_two_qubit_gates': pairs*int(self.remove_terminal)}


def fidelity(states_a, states_b=None):
    states_b = states_a if states_b is None else states_b
    return torch.abs(states_a @ states_b.conj().T)**2


def terminal_shared_rx(states, phi, qubit, n):
    """For the cancellation invariant test only."""
    return apply_one(states, _rx(torch.as_tensor(phi, dtype=torch.float64)), qubit, n)
