import torch
from torch import nn
import torch.nn.functional as F


class Student(nn.Module):
    """Lightweight full-input student with geometry embedding and task head."""
    def __init__(self, d, hidden=24, embedding_dim=8, classes=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                     nn.Linear(hidden, embedding_dim))
        self.head = nn.Linear(embedding_dim, classes)
        self.double()

    def forward(self, x):
        e = self.encoder(x.to(torch.float64))
        logits = self.head(e)
        return logits, F.normalize(e, p=2, dim=-1)


class DeepKernel(nn.Module):
    def __init__(self, d, hidden=24, embedding_dim=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(),
                                 nn.Linear(hidden, embedding_dim))
        self.double()

    def forward(self, x):
        e = self.net(x.to(torch.float64))
        return F.normalize(e, dim=-1)

class TokenTransformerStudent(nn.Module):
    """Optional full-feature tokenized Transformer emulator, no quantum at inference.

    Feature identity is preserved via trainable position embeddings. This module is
    *not* the exact original-paper tokenizer or architectural reproduction.
    """
    def __init__(self, d, hidden=24, embedding_dim=8, classes=2,
                 heads=2, layers=1, dropout=0.0):
        super().__init__()
        if hidden % heads:
            raise ValueError('hidden must be divisible by heads')
        self.value_embedding=nn.Linear(1,hidden)
        self.feature_embeddings=nn.Parameter(torch.randn(d,hidden)*.03)
        block=nn.TransformerEncoderLayer(d_model=hidden,nhead=heads,
            dim_feedforward=hidden*2,dropout=dropout,activation='gelu',
            batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(block,num_layers=layers,
                                           enable_nested_tensor=False)
        self.proj=nn.Linear(hidden,embedding_dim)
        self.head=nn.Linear(embedding_dim,classes)
        self.double()

    def forward(self,x):
        tokens=self.value_embedding(x.to(torch.float64).unsqueeze(-1))
        tokens=tokens+self.feature_embeddings.unsqueeze(0)
        # Meta-learning needs second derivatives; CPU fused flash SDPA lacks them.
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            pooled=self.encoder(tokens).mean(dim=1)
        e=self.proj(pooled)
        return self.head(e),F.normalize(e,dim=-1)
