import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def disparate_impact(y_true, y_pred, protected):
    # Disparate Impact = P(pred=1 | protected=0) / P(pred=1 | protected=1)
    grp0 = y_pred[protected==0]
    grp1 = y_pred[protected==1]
    p0 = np.mean(grp0==1)
    p1 = np.mean(grp1==1)
    # avoid division by zero
    if p1 == 0: return np.inf
    return p0 / p1

def equal_opportunity_difference(y_true, y_pred, protected):
    # difference in true positive rates: TPR(protected=0) - TPR(protected=1)
    tprs = {}
    for g in [0,1]:
        mask = (protected==g)
        if mask.sum()==0:
            tprs[g] = 0.0
            continue
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        # true positive rate = TP / (TP + FN) if any positives exist
        tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred_g, labels=[0,1]).ravel()
        denom = tp + fn
        tprs[g] = tp / denom if denom>0 else 0.0
    return tprs[0] - tprs[1]

def compute_basic_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

def reweighing_sample_weights(df, protected_col, label_col):
    # simple reweighing: weight inversely proportional to group-label frequency
    # returns an array of sample weights aligned with df
    groups = df.groupby([protected_col, label_col]).size().unstack(fill_value=0)
    # probability estimate for each (g,y)
    total = groups.sum().sum()
    weights = {}
    for g in groups.index:
        for y in groups.columns:
            count = groups.loc[g,y]
            # avoid divide by zero
            weights[(g,y)] = total / ( (len(groups.index) * len(groups.columns)) * max(1, count) )
    sample_weights = df.apply(lambda row: weights[(row[protected_col], row[label_col])], axis=1).values
    return sample_weights

def oversample_minority(df, protected_col, target_col):
    # simple oversampling: for groups of protected_col, oversample minority group's positive cases to match majority positive rate
    dfs = []
    counts = df.groupby(protected_col)[target_col].sum()
    # compute target positives = max positives among groups
    target_pos = counts.max()
    for g, group in df.groupby(protected_col):
        pos = group[group[target_col]==1]
        neg = group[group[target_col]==0]
        if len(pos)==0:
            dfs.append(group)
            continue
        # oversample positives to reach target_pos
        reps = int(np.ceil(target_pos / len(pos)))
        pos_upsampled = pd.concat([pos]*reps, ignore_index=True).sample(n=target_pos, replace=False)
        new_group = pd.concat([neg, pos_upsampled], ignore_index=True)
        dfs.append(new_group)
    return pd.concat(dfs, ignore_index=True)