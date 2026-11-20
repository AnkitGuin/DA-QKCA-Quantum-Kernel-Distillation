"""Exact simulation research pilot.

P01 differentiates query classification through T=1..N inner SGD steps;
query is a subset of outer TRAIN, not outer model-selection VAL.
Persistent student update is supervised + geometry on train-only minibatches.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from datetime import datetime,timezone
import torch
import torch.nn.functional as F
from torch.func import functional_call
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from ..quantum.kernels import KernelTeacher
from ..models.student import Student, TokenTransformerStudent
from ..losses.core import centered_alignment, label_gram, geometry_loss, krr_query_loss, effective_rank
from ..evaluation.metrics import evaluate_student
from ..utils.seeding import set_seed
from ..utils.config import config_hash


METHODS={'B00','B01','B02','B03','B04','P01','C01','C02','C03','C04','A01','A02','A03','A04','A05',
         'M01','R01_META','R01_NOMETA','D01_META','D01_NOMETA','ALIGN'}


def fixed_features(x,y,k,seed):
    """Train-only MI+RFE proxy. Not a verified faithful reproduction of R01."""
    mi=mutual_info_classif(x,y,random_state=seed)
    # MI candidates and RFE fit on training partition only.
    top=np.argsort(-mi)[:max(k,min(x.shape[1],2*k))]
    if len(top)==k:
        return top.tolist()
    rfe=RFE(LogisticRegression(max_iter=500,random_state=seed), n_features_to_select=k)
    rfe.fit(x[:,top],y)
    return top[rfe.support_].tolist()


def build_models(cfg,method,xtrain,ytrain,seed):
    d=xtrain.shape[1]
    n=cfg['model']['qubits']
    dep=cfg['model']['depth']
    selector='sparse'
    if method in ('B00','B01','C03'):
        selector='fixed'
    elif method=='B02':
        selector='sparse'
    elif method in ('C01','R01_META','R01_NOMETA'):
        selector='rbf'
    elif method in ('C02','D01_META','D01_NOMETA'):
        selector='deep'
    elif method=='M01':
        selector='matern'
    elif method=='A05':
        selector='dense'
    elif method=='C04':
        selector='rbf'
    fixed=fixed_features(xtrain,ytrain,n,seed) if selector=='fixed' else None
    train_circ=method not in ('B00','B02','C03')
    teacher=KernelTeacher(d,n,dep,selector,fixed,train_circ,
                          ansatz=cfg['model'].get('ansatz','ring'),
                          hidden=cfg['model']['hidden'],
                          selector_initialization=cfg['model'].get('selector_initialization','random'),
                          selector_relaxation=cfg['model'].get('selector_relaxation','ste'),
                          remove_terminal=cfg['model'].get('remove_terminal',True))
    student_arch=cfg['model'].get('student_arch','mlp')
    student_class={'mlp':Student,'transformer':TokenTransformerStudent}.get(student_arch)
    if student_class is None:raise ValueError(f'unknown student_arch {student_arch}')
    # Separate RNG makes initial student tensors identical across teacher choices.
    with torch.random.fork_rng():
        torch.manual_seed(seed+1000003)
        student=student_class(d,cfg['model']['hidden'],cfg['model']['embed'],2)
    if method=='C03':
        for p in teacher.parameters(): p.requires_grad_(False)
    return teacher,student


def inner_minibatches(yt, batch, seed):
    """Returns disjoint inner train/query from training indices."""
    rng=np.random.default_rng(seed)
    labels=np.asarray(yt)
    vals=np.unique(labels)
    if len(vals)<2:
        raise ValueError('binary classes required')
    eligible=rng.permutation(len(yt))
    tr, query=train_test_split(eligible,test_size=.35,random_state=seed,stratify=labels[eligible])
    b=min(batch,len(tr),len(query))
    # Random subset within fold, remains disjoint.
    a=rng.choice(tr,b,replace=False)
    q=rng.choice(query,b,replace=False)
    assert not set(a)&set(q)
    return a,q


def _student_loss(student,x,y,k,geometry_weight,params=None):
    out = functional_call(student,params,(x,)) if params is not None else student(x)
    logits, embedding = out
    task=F.cross_entropy(logits,y)
    geom=geometry_loss(k,embedding) if geometry_weight else task.new_zeros(())
    return task+geometry_weight*geom, task, geom


def bilevel_teacher_objective(teacher,student,xtr,ytr,xq,yq,cfg,method='P01',return_grads=False):
    """Outer meta-gradient flows through STE selector -> quantum kernel -> inner update -> query loss."""
    w=cfg['training']
    K=teacher(xtr)
    kq=teacher(xq,xtr)
    distill=w['geometry_weight'] if method!='A02' else 0.
    params={name:p for name,p in student.named_parameters()}
    inner_losses=[]
    for _ in range(w['unroll_steps']):
        loss,_,_= _student_loss(student,xtr,ytr,K.detach() if method=='A04' else K,distill,params)
        grads=torch.autograd.grad(loss,tuple(params.values()),create_graph=True)
        params={n:p-w['inner_lr']*g for (n,p),g in zip(params.items(),grads)}
        inner_losses.append(loss)
    query_logits,_=functional_call(student,params,(xq,))
    meta=F.cross_entropy(query_logits,yq)
    with torch.no_grad():
        persistent_query=F.cross_entropy(student(xq)[0],yq).item()
    task=krr_query_loss(K,kq,ytr,yq,w['ridge'])
    if method=='A03': task=0.*task
    align=1-centered_alignment(K,label_gram(ytr))
    beta=w['beta'] if method not in ('A01','R01_NOMETA','D01_NOMETA') else 0.
    penalty=w['selector_weight']*teacher.selector_penalty()
    outer=w['teacher_weight']*task+beta*meta+w['alignment_weight']*align+penalty
    return outer, {'meta':meta.item(),'teacher_task':task.item(),'alignment':1-align.item(),
                   'persistent_query_ce_before':persistent_query,
                   'inner_loss':inner_losses[-1].item()}


def train_one(cfg,method,seed,parts,output_dir=None):
    if method not in METHODS:
        raise ValueError(f'unsupported method {method}')
    set_seed(seed)
    st=time.perf_counter()
    teacher,student=build_models(cfg,method,parts.train_x,parts.train_y,seed)
    x=torch.tensor(parts.train_x,dtype=torch.float64)
    y=torch.tensor(parts.train_y,dtype=torch.long)
    if cfg['model'].get('landmarks'):
        from ..models.nystrom import landmark_indices
        ids=landmark_indices(parts.train_x,cfg['model']['landmarks'],seed,
                             cfg['model'].get('landmark_method','random'))
        teacher.landmarks=x[ids].clone()
    w=cfg['training']; batch=w['batch']
    trainable=[p for p in teacher.parameters() if p.requires_grad]
    teacher_optim=torch.optim.Adam(trainable,lr=w['teacher_lr']) if trainable and method!='C03' else None
    opt_class={'adam':torch.optim.Adam,'sgd':torch.optim.SGD}[w.get('student_optimizer','adam')]
    student_optim=opt_class(student.parameters(),lr=w['student_lr'])
    initial_selection=teacher.selector.selected_indices() if hasattr(teacher,'selector') and hasattr(teacher.selector,'selected_indices') else None
    teacher_first=method in ('B00','B01','B02','B03','C01','C02','C03','M01')
    student_only=method=='C04'
    epochs=w['epochs']; traces=[]
    # Same number of quantum/cross-kernel calls per epoch for all teacher-first models,
    # except C04, whose total compute advantage is explicitly reported.
    for epoch in range(epochs):
        tr, query=inner_minibatches(parts.train_y,batch,seed*20000+epoch)
        xt,yt=x[tr],y[tr]; xq,yq=x[query],y[query]
        info={}
        if not student_only and method not in ('B00','C03') and teacher_optim is not None:
            teacher_optim.zero_grad(set_to_none=True)
            if method in ('P01','A01','A02','A03','A04','A05','R01_META','R01_NOMETA','D01_META','D01_NOMETA'):
                outer,info=bilevel_teacher_objective(teacher,student,xt,yt,xq,yq,cfg,method)
            else:
                K=teacher(xt); Kq=teacher(xq,xt)
                task=krr_query_loss(K,Kq,yt,yq,w['ridge'])
                align=1-centered_alignment(K,label_gram(yt))
                outer=w['teacher_weight']*task + w['alignment_weight']*align
                if method=='ALIGN':
                    outer=align
                outer=outer+w['selector_weight']*teacher.selector_penalty()
                if method=='B04':
                    # Joint one-level student training: query task (no adaptation) + geometry.
                    student_loss,_,_= _student_loss(student,xt,yt,K,w['geometry_weight'])
                    outer=outer+student_loss
                info={'teacher_task':task.item(),'alignment':1-align.item()}
            outer.backward()
            teacher_optim.step()
            info['teacher_objective']=float(outer.detach())
            if teacher_first:
                # Stage 1 of teacher-first: train only teacher, no persistent student update.
                info['epoch']=epoch
                info['stage']='teacher_pretrain'
                traces.append(info)
                continue
        # Persistent student trains on train-only subset; teacher is frozen for this update.
        student_optim.zero_grad(set_to_none=True)
        with torch.no_grad():
            K=teacher(xt).detach() if not student_only else None
        g_weight=0. if student_only or method=='A02' else w['geometry_weight']
        loss,ce,geom=_student_loss(student,xt,yt,K,g_weight)
        loss.backward()
        student_optim.step()
        with torch.no_grad():
            info['persistent_query_ce_after']=F.cross_entropy(student(xq)[0],yq).item()
        info.update(epoch=epoch,student_loss=float(loss.detach()),student_ce=float(ce.detach()),
                    geometry=float(geom.detach()))
        info['stage']='student_joint_or_fixed'
        traces.append(info)
    if teacher_first and teacher_optim is not None:
        # Stage 2: frozen teacher guides student for the same persistent-student
        # epoch budget as B04/P01. Query labels never enter teacher or student fitting.
        for epoch in range(epochs):
            tr, _=inner_minibatches(parts.train_y,batch,seed*20000+epoch)
            xt,yt=x[tr],y[tr]
            student_optim.zero_grad(set_to_none=True)
            with torch.no_grad(): K=teacher(xt).detach()
            loss,ce,geom=_student_loss(student,xt,yt,K,w['geometry_weight'])
            loss.backward()
            student_optim.step()
            traces.append({'stage':'student_distillation','epoch':epoch,
                           'student_loss':float(loss.detach()),'student_ce':float(ce.detach()),
                           'geometry':float(geom.detach())})
    # final validation only (final test reserved for separate evaluator).
    val=evaluate_student(student,parts.val_x,parts.val_y)
    elapsed=time.perf_counter()-st
    # Lightweight diagnostics use training data only and no gradients.
    with torch.no_grad():
        if student_only:
            erank=off_std=None
        else:
            diag=teacher(x[:min(30,len(x))])
            erank=effective_rank(diag)
            off=diag[~torch.eye(len(diag),dtype=torch.bool)]
            off_std=float(off.std())
    src_root=Path(__file__).resolve().parents[1]
    h=hashlib.sha256()
    for module_path in sorted(src_root.rglob('*.py')):
        h.update(str(module_path.relative_to(src_root)).encode())
        h.update(module_path.read_bytes())
    source_hash=h.hexdigest()
    result={'source_hash':source_hash,'method':method,'seed':seed,'dataset':cfg.get('dataset',{}).get('name','synthetic'),
            'methodological_status':'research_prototype',
            'validation':val,'test':None,'time_seconds':elapsed,
            'initial_selected_indices':initial_selection,
            'raw_data_sha256':parts.split_manifest.get('raw_data_sha256'),
            'teacher_cost':teacher.counters.to_dict(),'circuit':teacher.gate_cost,
            'student_parameters':sum(p.numel() for p in student.parameters()),
            'effective_rank_train_subset':erank,'kernel_offdiag_std':off_std,
            'selected_indices':teacher.selector.selected_indices() if hasattr(teacher,'selector') and hasattr(teacher.selector,'selected_indices') else None,
            'config_hash':config_hash(cfg),'split_hashes':parts.split_manifest['hashes'],
            'timestamp':datetime.now(timezone.utc).isoformat(),
            'torch_version':torch.__version__,'python':platform.python_version(),
            'cpu':platform.processor() or platform.machine(),
            'epochs':epochs,'batch':batch,'traces':traces}
    if output_dir is not None:
        path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
        # Each ID is deterministic per conditions + measured timestamp; refuse overwrite.
        rid=f'{method}_s{seed}_{config_hash(cfg)[:8]}_{source_hash[:8]}'
        dst=path/rid
        if dst.exists():
            # deliberately do not overwrite original run artifacts
            rid=f'{rid}_{int(time.time_ns())}'
            dst=path/rid
        dst.mkdir()
        result['run_id']=rid
        torch.save({'teacher':teacher.state_dict(),'student':student.state_dict(),
                    'config':cfg,'method':method,'seed':seed},dst/'checkpoint.pt')
        (dst/'result.json').write_text(json.dumps(result,indent=2,default=str))
        (dst/'splits.json').write_text(json.dumps(parts.split_manifest,indent=2))
    else:
        result['run_id']='in_memory'
    return result
