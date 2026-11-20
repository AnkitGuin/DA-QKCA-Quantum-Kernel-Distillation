"""Versioned raw-manifest classical-vs-quantum plot (all measured val data)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

q=pd.read_csv('experiments/manifest_v1.csv')
c=pd.read_csv('experiments/manifest_classical_v2.csv')
methods=['P01','B04','C04','C05_SVM','C05_XGB']
df=pd.concat([q[q.method.isin(methods)],c[c.method.isin(methods)]],ignore_index=True)
summary=df.groupby('method').validation_macro_f1.agg(['mean','std']).reindex(methods)
fig,ax=plt.subplots(figsize=(8.0,4.3))
ax.errorbar(range(len(methods)),summary['mean'],yerr=summary['std'],fmt='o',capsize=5)
ax.set_xticks(range(len(methods)),['P01 bilevel','B04 joint','C04 student','Tuned RBF-SVM','Tuned XGBoost'])
ax.set_ylabel('Outer-validation macro-F1 (mean ± seed SD)')
ax.set_ylim(0.60,0.95)
ax.set_title('Synthetic validation, five seeds; not a final-test comparison')
ax.grid(axis='y',alpha=.2)
fig.tight_layout()
p=Path('experiments/figures/classical_comparison.png');fig.savefig(p,dpi=200)
print(p)
