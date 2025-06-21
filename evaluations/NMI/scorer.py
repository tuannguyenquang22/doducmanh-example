import numpy as np
from sklearn import metrics

def scorer(y_true, y_pred):
    score = metrics.cluster.normalized_mutual_info_score(y_true, y_pred)
    return score
