import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, log_loss, brier_score_loss
import torch


def metric_bundle(y, logits):
    y=np.asarray(y).astype(int)
    logits=np.asarray(logits)
    pred=logits.argmax(-1)
    exp=np.exp(logits-logits.max(-1,keepdims=True))
    prob=exp[:,1]/exp.sum(-1)
    result={'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred))}
    cm=confusion_matrix(y,pred,labels=[0,1])
    result.update(confusion_matrix=cm.tolist(),
                  per_class_f1=f1_score(y,pred,labels=[0,1],average=None,zero_division=0).tolist(),
                  false_positive_rate=float(cm[0,1]/max(1,cm[0].sum())),
                  brier=float(brier_score_loss(y,prob)),
                  log_loss=float(log_loss(y,np.column_stack([1-prob,prob]),labels=[0,1])))
    if len(np.unique(y))==2:
        result.update(pr_auc=float(average_precision_score(y,prob)),
                      roc_auc=float(roc_auc_score(y,prob)))
    return result


def evaluate_student(student,x,y):
    student.eval()
    with torch.no_grad():
        logits,_=student(torch.tensor(x,dtype=torch.float64))
    return metric_bundle(y,logits.numpy())
