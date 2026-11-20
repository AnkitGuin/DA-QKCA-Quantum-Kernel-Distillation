#!/usr/bin/env python3
import argparse, json
from daqkca.utils.config import load_config
from daqkca.data.loaders import load_dataset
from daqkca.training.engine import train_one

p=argparse.ArgumentParser();p.add_argument('--config',default='configs/smoke.yaml');a=p.parse_args()
c=load_config(a.config)
for m in ['B00','B04','P01','C01','C02','C04']:
    r=train_one(c,m,101,load_dataset(c,101))
    print(m,'VAL_F1',round(r['validation']['macro_f1'],4),'teacher pairs',r['teacher_cost']['kernel_pairs'])
