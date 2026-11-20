"""Plots are generated only from real run manifests; no placeholder curves."""
import argparse,json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--manifest',default='experiments/manifest.csv');a=p.parse_args()
df=pd.read_csv(a.manifest)
out=Path('experiments/figures');out.mkdir(parents=True,exist_ok=True)
summary=df.groupby('method',as_index=False).agg(n=('validation_macro_f1','size'),
      mean_f1=('validation_macro_f1','mean'),std_f1=('validation_macro_f1','std'),
      mean_pairs=('kernel_pairs','mean'),mean_seconds=('time_seconds','mean'))
summary.to_csv('experiments/aggregates/pilot_summary.csv',index=False)
fig,ax=plt.subplots(figsize=(7,4))
ax.errorbar(range(len(summary)),summary.mean_f1, yerr=summary.std_f1.fillna(0),fmt='o',capsize=4)
ax.set_xticks(range(len(summary)),summary.method,rotation=45)
ax.set_ylabel('Outer-validation macro-F1');ax.set_title('Measured exploratory pilots; not final-test findings')
fig.tight_layout();fig.savefig(out/'validation_f1.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4))
for _,r in df.iterrows():ax.scatter(r.kernel_pairs,r.validation_macro_f1,label=r.method,s=35)
ax.set_xscale('symlog',linthresh=1);ax.set_xlabel('Dense kernel evaluations (including repeated calls)')
ax.set_ylabel('Outer-validation macro-F1');ax.set_title('Validation vs kernel cost (not QPU calls)')
fig.tight_layout();fig.savefig(out/'validation_cost.png',dpi=180);plt.close(fig)
print(summary.to_string(index=False))
