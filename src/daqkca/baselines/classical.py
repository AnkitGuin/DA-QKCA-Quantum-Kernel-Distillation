"""Real trained classical controls with train-only CV tuning, outer-validation once.

These baselines use the same input data but have a different training/tuning
budget. Report as exploratory, not equal-total-compute quantum comparisons.
"""
import hashlib,json,platform,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer,f1_score
from ..evaluation.metrics import metric_bundle
from ..utils.config import config_hash
from ..utils.seeding import set_seed


def train_classical(cfg,method,seed,parts,outdir=None):
    if method not in {'C05_SVM','C05_XGB'}:raise ValueError(method)
    set_seed(seed)
    start=time.perf_counter()
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed)
    if method=='C05_SVM':
        base=SVC(kernel='rbf')
        grid={'C':[0.1,1.0,10.0,100.0],'gamma':['scale',0.01,0.1,1.]}
    else:
        from xgboost import XGBClassifier
        base=XGBClassifier(n_estimators=150,n_jobs=1,random_state=seed,
                           tree_method='hist',eval_metric='logloss')
        grid={'max_depth':[2,4],'learning_rate':[.05,.15],
              'subsample':[0.8,1.0]}
    if parts.raw_train_x is None or parts.raw_val_x is None:
        raise ValueError('Fold-pure CV requires unscaled raw training and validation inputs')
    pipeline=Pipeline([('impute',SimpleImputer()),('scale',StandardScaler()),('model',base)])
    grid={'model__'+key:value for key,value in grid.items()}
    tuner=GridSearchCV(pipeline,grid,cv=cv,scoring='f1_macro',n_jobs=1,refit=True,
                       error_score='raise')
    tuner.fit(parts.raw_train_x,parts.train_y)
    if method=='C05_SVM':
        score=tuner.decision_function(parts.raw_val_x)
        logits=np.column_stack([np.zeros_like(score),score])
    else:
        p=tuner.predict_proba(parts.raw_val_x)
        logits=np.log(np.maximum(p,1e-12))
    val=metric_bundle(parts.val_y,logits)
    r={'method':method,'seed':seed,'dataset':cfg.get('dataset',{}).get('name','synthetic'),
       'methodological_status':'exploratory_train_only_cv',
       'config_hash':config_hash(cfg),'split_hashes':parts.split_manifest['hashes'],
       'validation':val,'test':None,'inner_train_cv_best_macro_f1':float(tuner.best_score_),
       'best_hyperparameters':tuner.best_params_,'cv_fits':len(tuner.cv_results_['params'])*3,
       'teacher_cost':{'states_simulated':0,'kernel_pairs':0,'calls':0,'shots':0},
       'time_seconds':time.perf_counter()-start,'timestamp':datetime.now(timezone.utc).isoformat(),
       'python':platform.python_version()}
    if outdir:
        folder=Path(outdir);folder.mkdir(parents=True,exist_ok=True)
        rid=f'{method}_s{seed}_{config_hash(cfg)[:8]}'
        dst=folder/rid
        if dst.exists():
            rid+=f'_{time.time_ns()}';dst=folder/rid
        dst.mkdir()
        r['run_id']=rid
        (dst/'result.json').write_text(json.dumps(r,indent=2))
        (dst/'splits.json').write_text(json.dumps(parts.split_manifest,indent=2))
        # Model object is deliberately not pickled. Refit with seed, CV and train
        # data (or extend this baseline to a safe non-pickle checkpoint format).
    else:r['run_id']='in_memory'
    return r
