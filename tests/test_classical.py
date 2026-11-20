from daqkca.baselines.classical import train_classical
from daqkca.data.loaders import split_and_scale,synthetic

def test_svm_control_uses_only_train_cv():
    x,y=synthetic(n=96,d=8,seed=12)
    parts=split_and_scale(x,y,12)
    r=train_classical({'dataset':{'name':'synthetic'}},'C05_SVM',12,parts)
    assert r['cv_fits']==16*3
    assert r['test'] is None
    assert r['validation']['macro_f1']>=0
