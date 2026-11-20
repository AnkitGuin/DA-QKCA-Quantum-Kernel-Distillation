#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
from daqkca.utils.config import load_config
from daqkca.data.loaders import load_dataset
from daqkca.training.engine import train_one, METHODS
METHODS=METHODS|{'C05_SVM','C05_XGB'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/pilot.yaml')
    p.add_argument('--method',choices=sorted(METHODS),required=True)
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--out',default='experiments/revision_runs')
    p.add_argument('--manifest',default='experiments/manifest_revision.csv',
                   help='new runs must not append to historical development/pilot manifests')
    a=p.parse_args()
    manifest=Path(a.manifest)
    frozen={Path(name).resolve() for name in ['experiments/manifest.csv','experiments/manifest_v1.csv','experiments/manifest_classical_v2.csv']}
    if manifest.resolve() in frozen:
        raise ValueError('Historical manifests are frozen; choose a new manifest')
    cfg=load_config(a.config)
    parts=load_dataset(cfg,a.seed)
    if a.method in ('C05_SVM','C05_XGB'):
        from daqkca.baselines.classical import train_classical
        res=train_classical(cfg,a.method,a.seed,parts,a.out)
    else:
        res=train_one(cfg,a.method,a.seed,parts,a.out)
    manifest.parent.mkdir(parents=True,exist_ok=True)
    row={k:res[k] for k in ('run_id','method','seed','dataset','config_hash','timestamp',
                            'time_seconds')}
    row['validation_macro_f1']=res['validation']['macro_f1']
    row['kernel_pairs']=res['teacher_cost']['kernel_pairs']
    with open(manifest,'a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if f.tell()==0:w.writeheader()
        w.writerow(row)
    print(json.dumps({k:res[k] for k in ('run_id','method','seed','validation','time_seconds','teacher_cost')},indent=2))

if __name__=='__main__': main()
