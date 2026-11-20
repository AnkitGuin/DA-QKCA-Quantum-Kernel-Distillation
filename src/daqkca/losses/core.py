import torch
import torch.nn.functional as F


def center(k):
    return k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()


def centered_alignment(k, targets, eps=1e-10):
    kc, tc = center(k), center(targets)
    nk, nt = torch.linalg.vector_norm(kc), torch.linalg.vector_norm(tc)
    if float(nk.detach()) < eps or float(nt.detach()) < eps:
        raise ValueError('degenerate centered kernel; CKA undefined')
    return (kc*tc).sum() / (nk*nt + eps)


def label_gram(y, n_classes=2):
    hot = F.one_hot(y.long(), num_classes=n_classes).to(torch.float64)
    return hot @ hot.T


def geometry_loss(quantum_k, embedding, eps=1e-10):
    q = center(quantum_k)
    s = center(embedding @ embedding.T)
    nq, ns = torch.linalg.vector_norm(q), torch.linalg.vector_norm(s)
    if float(nq.detach()) < eps or float(ns.detach()) < eps:
        # The zero-kernel risk is tracked separately by diagnostics.
        return (q-s).square().sum()
    return ((q/(nq+eps))-(s/(ns+eps))).square().sum()


def krr_query_loss(k_train, k_query, y_train, y_query, ridge=0.1, classes=2):
    """Differentiable Cholesky solve; logits on internal query data only."""
    from torch.nn.functional import one_hot
    y_hot = one_hot(y_train.long(), classes).to(torch.float64)
    kernel = (k_train+k_train.T)/2
    mat = kernel + ridge*torch.eye(len(kernel), device=kernel.device, dtype=kernel.dtype)
    coeff = torch.linalg.solve(mat, y_hot)  # linear solve, never explicit inverse
    scores = k_query @ coeff
    return F.cross_entropy(scores, y_query.long())


def effective_rank(k):
    eig = torch.linalg.eigvalsh((k+k.T)*0.5).clamp_min(0)
    weights = eig/(eig.sum()+1e-12)
    entropy = -(weights*weights.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().detach())
