"""Run 5 paired seeds for train-only-CV conventional controls."""
import csv,argparse
from pathlib import Path
from daqkca.utils.config import load_config
from daqkca.data.loaders import load_dataset
from daqkca.baselines.classical import train_classical

p=argparse.ArgumentParser();p.add_argument('--config',default='configs/pilot.yaml')
p.add_argument('--seeds',default='101,102,103,104,105')
a=p.parse_args()
cfg=load_config(a.config);f=Path('experiments/manifest_classical_v3.csv')
for seed in map(int,a.seeds.split(',')):
    parts=load_dataset(cfg,seed)
    for method in ['C05_SVM','C05_XGB']:
        res=train_classical(cfg,method,seed,parts,'experiments/revision_runs')
        row={'run_id':res['run_id'],'method':method,'seed':seed,
             'validation_macro_f1':res['validation']['macro_f1'],
             'time_seconds':res['time_seconds'],'cv_fits':res['cv_fits'],
             'config_hash':res['config_hash']}
        with open(f,'a',newline='') as file:
            w=csv.DictWriter(file,fieldnames=list(row))
            if file.tell()==0:w.writeheader()
            w.writerow(row)
        print(seed,method,'validation F1',round(row['validation_macro_f1'],4),
              'CV fits',row['cv_fits'],'seconds',round(row['time_seconds'],2),flush=True)
