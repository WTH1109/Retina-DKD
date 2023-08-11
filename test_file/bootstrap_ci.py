import numpy as np
from sklearn import metrics
from scipy import stats

def my_bootstrap_ci_model(_pre, _label, n_samples=100, n_range=500):
    _pre = np.array(_pre)
    _label = np.array(_label)
    n_num = len(_pre)
    ACC = []
    SEN = []
    SPEC = []
    F1_SCORE = []
    AUC = []
    for _ in range(n_range):
        indices = np.random.randint(0, n_num, size=n_samples)
        tmp_pre = _pre[indices] > 0.5
        tmp_label = _label[indices]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i_test in range(n_samples):
            if tmp_label[i_test] == 1 and tmp_pre[i_test] == 1:
                tp += 1
            if tmp_label[i_test] == 1 and tmp_pre[i_test] == 0:
                fn += 1
            if tmp_label[i_test] == 0 and tmp_pre[i_test] == 1:
                fp += 1
            if tmp_label[i_test] == 0 and tmp_pre[i_test] == 0:
                tn += 1
        Acc = (tp + tn) / (tp + tn + fp + fn)
        Sen = (tp) / (tp + fn)  # recall
        Spec = (tn) / (tn + fp)
        if tp + fp == 0:
            precision = 0
        else:
            precision = (tp) / (tp + fp)
        recall = Sen
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = (2 * precision * recall) / (precision + recall)
        auc = metrics.roc_auc_score(y_true=np.array(tmp_label), y_score=np.array(_pre[indices]))
        ACC.append(Acc)
        SEN.append(Sen)
        SPEC.append(Spec)
        F1_SCORE.append(f1_score)
        AUC.append(auc)
    ACC_low, ACC_high = CI_CAU(ACC)
    SEN_low, SEN_high = CI_CAU(SEN)
    SPEC_low, SPEC_high = CI_CAU(SPEC)
    F1_SCORE_low, F1_SCORE_high = CI_CAU(F1_SCORE)
    AUC_low, AUC_high = CI_CAU(AUC)
    return ACC_low, ACC_high, SEN_low, SEN_high, SPEC_low, SPEC_high, F1_SCORE_low, F1_SCORE_high, AUC_low, AUC_high


def CI_CAU(valueList):
    averageValue = np.mean(valueList)
    standardError = stats.sem(valueList)
    a = averageValue - 1.96 * standardError
    b = averageValue + 1.96 * standardError
    return a, b