"""Distinct sparse original-feature selector versus dense feature extraction."""
import torch
from torch import nn


class SparseTopKSelector(nn.Module):
    """Unique hard top-k columns with straight-through softmax gradients.

    Each of q slots has trainable d-way logits. Hard assignment is greedy,
    deterministic, and unique. In training, hard one-hots are forward values;
    backpropagation uses softmax rows. This is a biased STE, not an exact gradient
    through discrete top-k. Outside-set gradient signals are approximate.
    """
    def __init__(self, d, q, temperature=0.8, initialization='random', relaxation='ste'):
        super().__init__()
        if not 1 <= q <= d:
            raise ValueError('q must be in [1,d]')
        if temperature <= 0 or initialization not in ('random', 'identity'):
            raise ValueError('positive temperature and random/identity initialization required')
        if relaxation not in ('ste', 'soft'):
            raise ValueError('relaxation must be ste or soft')
        self.d, self.q, self.temperature = d, q, temperature
        self.relaxation = relaxation
        logits = torch.randn(q, d, dtype=torch.float64)*0.01 if initialization == 'random' else torch.zeros(q, d, dtype=torch.float64)
        if initialization == 'identity':
            logits[torch.arange(q), torch.arange(q)] = 1.
        self.logits = nn.Parameter(logits)

    def selected_indices(self):
        logits = self.logits.detach()
        used = set()
        order = []
        for row in logits:
            scores = row.clone()
            if used:
                scores[list(used)] = -float('inf')
            j = int(scores.argmax().item())
            used.add(j)
            order.append(j)
        return order

    def forward(self, x):
        soft = torch.softmax(self.logits/self.temperature, dim=-1)
        idx = self.selected_indices()
        hard = torch.zeros_like(soft)
        hard[torch.arange(self.q, device=x.device), torch.tensor(idx, device=x.device)] = 1.
        # Soft mode is a dense relaxation; report its hard projection separately.
        assignment = soft if self.relaxation == 'soft' else hard - soft.detach() + soft
        return torch.tanh(x.to(torch.float64) @ assignment.T) * 1.6

    def regularizer(self):
        p = torch.softmax(self.logits/self.temperature, -1)
        return -(p*torch.log(p.clamp_min(1e-12))).sum(-1).mean()


class DenseProjection(nn.Module):
    """Learned dense features; never report as original feature selection."""
    def __init__(self, d, q):
        super().__init__()
        self.linear = nn.Linear(d, q, bias=False, dtype=torch.float64)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        return torch.tanh(self.linear(x.to(torch.float64))) * 1.6

    def regularizer(self):
        return self.linear.weight.square().mean()


class FixedFeatureSelector(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.register_buffer('indices', torch.as_tensor(indices, dtype=torch.long))
        self.q = len(indices)

    def forward(self, x):
        return torch.tanh(x.index_select(-1, self.indices).to(torch.float64)) * 1.6

    def regularizer(self):
        return torch.tensor(0., dtype=torch.float64, device=self.indices.device)

    def selected_indices(self):
        return self.indices.tolist()
