import torch
from daqkca.quantum.kernels import KernelTeacher
from daqkca.models.student import Student,TokenTransformerStudent
from daqkca.training.engine import bilevel_teacher_objective


def test_theta_autograd_agrees_with_central_difference():
    torch.manual_seed(5)
    x=torch.randn(9,5,dtype=torch.float64)
    t=KernelTeacher(5,2,2)
    k=t(x)
    loss=(k[0,1]+.5*k[2,4])
    grad=torch.autograd.grad(loss,t.circuit.theta)[0][0,0].item()
    p=t.circuit.theta
    old=p[0,0].item();eps=1e-5
    with torch.no_grad():p[0,0]=old+eps
    plus=float((t(x)[0,1]+.5*t(x)[2,4]).detach())
    with torch.no_grad():p[0,0]=old-eps
    minus=float((t(x)[0,1]+.5*t(x)[2,4]).detach())
    with torch.no_grad():p[0,0]=old
    numerical=(plus-minus)/(2*eps)
    assert abs(grad-numerical)<3e-6


def test_token_transformer_shape_and_gradients():
    torch.manual_seed(4)
    s=TokenTransformerStudent(d=5,hidden=8,embedding_dim=3)
    logits,emb=s(torch.randn(3,5,dtype=torch.float64))
    assert logits.shape==(3,2) and emb.shape==(3,3)
    logits.square().mean().backward()
    assert s.value_embedding.weight.grad.norm()>0
