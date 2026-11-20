import argparse
import torch
from daqkca.quantum.kernels import KernelTeacher
from daqkca.models.student import Student
from daqkca.training.engine import bilevel_teacher_objective
from daqkca.utils.config import load_config

p=argparse.ArgumentParser();p.add_argument('--config',default='configs/smoke.yaml');a=p.parse_args()
c=load_config(a.config);torch.manual_seed(101)
d,n=7,c['model']['qubits'],
x=torch.randn(18,d,dtype=torch.float64);y=torch.tensor([0,1]*9)
t=KernelTeacher(d,n);s=Student(d,12,4)
loss,_=bilevel_teacher_objective(t,s,x[:9],y[:9],x[9:],y[9:],c)
g=torch.autograd.grad(loss,[t.selector.logits,t.circuit.theta],retain_graph=True);print('total gradients',[float(z.norm()) for z in g])
# Isolate pure student meta-gradient by subtracting the beta=0 objective.
t0=loss
c0={**c,'training':{**c['training'],'beta':0.0}}
l0,_=bilevel_teacher_objective(t,s,x[:9],y[:9],x[9:],y[9:],c0)
meta=torch.autograd.grad(t0-l0,[t.selector.logits,t.circuit.theta]);print('meta-only gradients',[float(z.norm()) for z in meta])
assert min(float(z.norm()) for z in g)>1e-10
assert min(float(z.norm()) for z in meta)>1e-10
print('PASS: quantum and selector gradients include student-query hypergradient')
