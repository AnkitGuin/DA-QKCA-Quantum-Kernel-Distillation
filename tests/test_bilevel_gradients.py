import copy
import torch
from daqkca.models.student import Student
from daqkca.quantum.kernels import KernelTeacher
from daqkca.training.engine import bilevel_teacher_objective

C={'training':{'geometry_weight':0.25,'unroll_steps':2,'inner_lr':0.1,
                'ridge':0.1,'teacher_weight':0.3,'alignment_weight':0.02,
                'beta':1.0,'selector_weight':0.0}}


def test_meta_gradient_reaches_quantum_and_selector():
    torch.manual_seed(42)
    x=torch.randn(20,6,dtype=torch.float64);y=torch.tensor([0,1]*10)
    teacher=KernelTeacher(6,2,2);student=Student(6,10,3)
    total,_=bilevel_teacher_objective(teacher,student,x[:10],y[:10],x[10:],y[10:],C)
    c0=copy.deepcopy(C);c0['training']['beta']=0
    nometa,_=bilevel_teacher_objective(teacher,student,x[:10],y[:10],x[10:],y[10:],c0)
    g=torch.autograd.grad(total-nometa,[teacher.circuit.theta,teacher.selector.logits])
    assert all(float(a.norm())>1e-10 for a in g)


def test_stop_gradient_beta_zero_removes_meta_contribution():
    torch.manual_seed(23)
    x=torch.randn(14,5,dtype=torch.float64);y=torch.tensor([0,1]*7)
    t=KernelTeacher(5,2);s=Student(5,8,3)
    loss1,_=bilevel_teacher_objective(t,s,x[:7],y[:7],x[7:],y[7:],C,'A01')
    c0=copy.deepcopy(C);c0['training']['beta']=0
    loss0,_=bilevel_teacher_objective(t,s,x[:7],y[:7],x[7:],y[7:],c0,'P01')
    assert torch.allclose(loss1,loss0)
