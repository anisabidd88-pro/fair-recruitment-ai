import os, joblib
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from utils import disparate_impact, equal_opportunity_difference, compute_basic_metrics, reweighing_sample_weights, oversample_minority

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_candidates.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# features and protected attribute
FEATURES = ["experience","education","skill","interview"]
PROTECTED = "gender"  # we'll evaluate fairness w.r.t. gender (0 male, 1 female)
TARGET = "hire"

X = df[FEATURES].values
y = df[TARGET].values
protected = df[PROTECTED].values

# train/test split stratify on target
X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
    X, y, protected, test_size=0.3, random_state=42, stratify=y
)

# ------- Baseline Logistic Regression -------
log = LogisticRegression(max_iter=1000)
log.fit(X_train, y_train)
y_pred_log = log.predict(X_test)

metrics_log = compute_basic_metrics(y_test, y_pred_log)
di_log = disparate_impact(y_test, y_pred_log, prot_test)
eod_log = equal_opportunity_difference(y_test, y_pred_log, prot_test)

# save baseline results
pd.DataFrame([metrics_log]).to_csv(OUT / "metrics_logistic_baseline.csv", index=False)
with open(OUT / "fairness_logistic_baseline.txt", "w") as f:
    f.write(f"Disparate Impact (gender): {di_log:.4f}\\nEqual Opportunity Difference (gender): {eod_log:.4f}\\n")

# feature importance from logistic (coefficients)
feat_imp_log = dict(zip(FEATURES, log.coef_.ravel()))
pd.Series(feat_imp_log).to_csv(OUT / "feature_importance_logistic.csv")

# plot coefficients
plt.figure(figsize=(6,4))
names = list(feat_imp_log.keys())
vals = [feat_imp_log[n] for n in names]
plt.bar(range(len(vals)), vals)
plt.xticks(range(len(vals)), names, rotation=45)
plt.title("Logistic Regression Coefficients (baseline)")
plt.tight_layout()
plt.savefig(OUT / "coef_logistic.png")
plt.close()

# ------- Random Forest -------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

metrics_rf = compute_basic_metrics(y_test, y_pred_rf)
di_rf = disparate_impact(y_test, y_pred_rf, prot_test)
eod_rf = equal_opportunity_difference(y_test, y_pred_rf, prot_test)

pd.DataFrame([metrics_rf]).to_csv(OUT / "metrics_rf_baseline.csv", index=False)
with open(OUT / "fairness_rf_baseline.txt", "w") as f:
    f.write(f"Disparate Impact (gender): {di_rf:.4f}\\nEqual Opportunity Difference (gender): {eod_rf:.4f}\\n")

pd.Series(dict(zip(FEATURES, rf.feature_importances_))).to_csv(OUT / "feature_importance_rf.csv")

plt.figure(figsize=(6,4))
names = FEATURES
vals = rf.feature_importances_
plt.bar(range(len(vals)), vals)
plt.xticks(range(len(vals)), names, rotation=45)
plt.title("Random Forest Feature Importances (baseline)")
plt.tight_layout()
plt.savefig(OUT / "featimp_rf.png")
plt.close()

# ------- Mitigation: Reweighing during training (sample weights) -------
import pandas as pd
df_train = pd.DataFrame(X_train, columns=FEATURES)
df_train[PROTECTED] = prot_train
df_train[TARGET] = y_train

sample_weights = reweighing_sample_weights(df_train, PROTECTED, TARGET)
# retrain logistic with sample weights
log_rw = LogisticRegression(max_iter=1000)
log_rw.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_log_rw = log_rw.predict(X_test)

metrics_log_rw = compute_basic_metrics(y_test, y_pred_log_rw)
di_log_rw = disparate_impact(y_test, y_pred_log_rw, prot_test)
eod_log_rw = equal_opportunity_difference(y_test, y_pred_log_rw, prot_test)

pd.DataFrame([metrics_log_rw]).to_csv(OUT / "metrics_logistic_reweighing.csv", index=False)
with open(OUT / "fairness_logistic_reweighing.txt", "w") as f:
    f.write(f"Disparate Impact (gender): {di_log_rw:.4f}\\nEqual Opportunity Difference (gender): {eod_log_rw:.4f}\\n")

pd.Series(dict(zip(FEATURES, log_rw.coef_.ravel()))).to_csv(OUT / "feature_importance_logistic_reweighing.csv")

plt.figure(figsize=(6,4))
vals = log_rw.coef_.ravel()
plt.bar(range(len(vals)), vals)
plt.xticks(range(len(vals)), FEATURES, rotation=45)
plt.title("Logistic Regression Coefs (reweighing)")
plt.tight_layout()
plt.savefig(OUT / "coef_logistic_reweighing.png")
plt.close()

# ------- Mitigation: Oversampling minority positives -------
df_full = pd.read_csv(ROOT / "data" / "synthetic_candidates.csv")
from utils import oversample_minority
df_aug = oversample_minority(df_full, PROTECTED, TARGET)
X_aug = df_aug[FEATURES].values
y_aug = df_aug[TARGET].values
prot_aug = df_aug[PROTECTED].values

# split augmented set
X_tr2, X_val2, y_tr2, y_val2, prot_tr2, prot_val2 = train_test_split(X_aug, y_aug, prot_aug, test_size=0.2, random_state=42, stratify=y_aug)

rf_aug = RandomForestClassifier(n_estimators=200, random_state=42)
rf_aug.fit(X_tr2, y_tr2)
y_pred_rf_aug = rf_aug.predict(X_test)  # evaluate on original test set for comparability

metrics_rf_aug = compute_basic_metrics(y_test, y_pred_rf_aug)
di_rf_aug = disparate_impact(y_test, y_pred_rf_aug, prot_test)
eod_rf_aug = equal_opportunity_difference(y_test, y_pred_rf_aug, prot_test)

pd.DataFrame([metrics_rf_aug]).to_csv(OUT / "metrics_rf_oversample.csv", index=False)
with open(OUT / "fairness_rf_oversample.txt", "w") as f:
    f.write(f"Disparate Impact (gender): {di_rf_aug:.4f}\\nEqual Opportunity Difference (gender): {eod_rf_aug:.4f}\\n")

pd.Series(dict(zip(FEATURES, rf_aug.feature_importances_))).to_csv(OUT / "feature_importance_rf_oversample.csv")

plt.figure(figsize=(6,4))
vals = rf_aug.feature_importances_
plt.bar(range(len(vals)), vals)
plt.xticks(range(len(vals)), FEATURES, rotation=45)
plt.title("RF Feature Importances (oversample)")
plt.tight_layout()
plt.savefig(OUT / "featimp_rf_oversample.png")
plt.close()

# ------- Simple local explanation (prototype nearest neighbor) -------
# For a handful of test examples, find nearest training example of same predicted class and show diff
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=1)
nn.fit(X_train)
idxs = np.random.choice(len(X_test), size=6, replace=False)
local_explanations = []
for i in idxs:
    x = X_test[i].reshape(1,-1)
    dist, ind = nn.kneighbors(x)
    train_idx = ind[0][0]
    explanation = {
        "test_index": int(i),
        "pred_logistic": int(y_pred_log[i]),
        "nearest_train_index": int(train_idx),
        "nearest_train_features": X_train[train_idx].tolist(),
        "test_features": X_test[i].tolist()
    }
    local_explanations.append(explanation)

pd.DataFrame(local_explanations).to_csv(OUT / "local_explanations_nearest.csv", index=False)

# ------- Save models -------
joblib.dump(log, OUT / "model_logistic_baseline.joblib")
joblib.dump(log_rw, OUT / "model_logistic_reweighing.joblib")
joblib.dump(rf, OUT / "model_rf_baseline.joblib")
joblib.dump(rf_aug, OUT / "model_rf_oversample.joblib")

# ------- Summary file -------
summary = {
    "logistic_baseline": {"metrics": metrics_log, "disparate_impact": di_log, "eod": eod_log},
    "rf_baseline": {"metrics": metrics_rf, "disparate_impact": di_rf, "eod": eod_rf},
    "logistic_reweighing": {"metrics": metrics_log_rw, "disparate_impact": di_log_rw, "eod": eod_log_rw},
    "rf_oversample": {"metrics": metrics_rf_aug, "disparate_impact": di_rf_aug, "eod": eod_rf_aug}
}
import json
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Training and evaluation complete. Outputs saved to outputs/")