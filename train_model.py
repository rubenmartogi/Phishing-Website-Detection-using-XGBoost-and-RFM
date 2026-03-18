import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE = 12
TEST_SIZE = 0.2
TARGET_COL = "label"

# Samakan dengan notebook
ACTIVE_TRAIN_FEATURES = "hybrid81"  # "url37" atau "hybrid81"

current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, "DataFiles", "data_cleaning.csv")

url_features_37 = [
    "length_url","length_hostname","ip","nb_dots","nb_hyphens","nb_at","nb_qm","nb_and",
    "nb_eq","nb_underscore","nb_tilde","nb_percent","nb_slash","nb_star","nb_colon",
    "nb_comma","nb_semicolumn","nb_dollar","nb_space","nb_www","nb_com","nb_dslash",
    "http_in_path","https_token","ratio_digits_url","ratio_digits_host","punycode","port",
    "tld_in_path","tld_in_subdomain","abnormal_subdomain","nb_subdomains","prefix_suffix",
    "random_domain","shortening_service","path_extension","nb_redirection"
]

if not os.path.exists(data_file_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan: {data_file_path}")

data = pd.read_csv(data_file_path)
data.columns = data.columns.str.strip()

if TARGET_COL not in data.columns:
    raise ValueError(f"Kolom target '{TARGET_COL}' tidak ditemukan di dataset.")

# y numeric
y_all = pd.to_numeric(data[TARGET_COL], errors="coerce")
valid_mask = y_all.notna()
data = data.loc[valid_mask].copy()
y_all = y_all.loc[valid_mask].astype("int64")

id_cols = [c for c in ["url"] if c in data.columns]
all_features = [c for c in data.columns if c not in [TARGET_COL] + id_cols]
webcontent_features_44 = [c for c in all_features if c not in url_features_37]
hybrid_features_81 = url_features_37 + webcontent_features_44

feature_candidates = hybrid_features_81 if ACTIVE_TRAIN_FEATURES == "hybrid81" else url_features_37

def prep_X(df: pd.DataFrame, cols, idx) -> pd.DataFrame:
    X = df.loc[idx].reindex(columns=cols)
    return X.apply(pd.to_numeric, errors="coerce")

train_idx, test_idx = train_test_split(
    data.index,
    test_size=TEST_SIZE,
    stratify=y_all,
    random_state=RANDOM_STATE
)

X_train = prep_X(data, feature_candidates, train_idx)
X_test = prep_X(data, feature_candidates, test_idx)

# drop kolom all-NaN (sesuai praktik notebook)
selected_features = X_train.columns[X_train.notna().any()].tolist()
if not selected_features:
    raise ValueError("Tidak ada fitur valid untuk training setelah cleaning.")

X_train = X_train[selected_features]
X_test = X_test[selected_features]

# median imputasi dari train
train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

y_train = y_all.loc[train_idx]
y_test = y_all.loc[test_idx]

print(f"Mode: {ACTIVE_TRAIN_FEATURES}")
print(f"URL features target: {len(url_features_37)}")
print(f"Web-content features: {len(webcontent_features_44)}")
print(f"Selected actual features: {len(selected_features)}")
print(f"Shape train/test: {X_train.shape} / {X_test.shape}")

def print_metrics(name, y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    print(f"\n{name}")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1       : {f1*100:.2f}%")
    print(f"AUC      : {auc*100:.2f}%" if not np.isnan(auc) else "AUC      : N/A")

# RF
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]
print_metrics("Random Forest", y_test, rf_pred, rf_prob)

# XGB
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    reg_lambda=1.0,
    reg_alpha=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    min_child_weight=2,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
print_metrics("XGBoost", y_test, xgb_pred, xgb_prob)

# Stacking (RF + XGB -> LR)
stack_model = StackingClassifier(
    estimators=[("rf", rf_model), ("xgb", xgb_model)],
    final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    stack_method="predict_proba",
    n_jobs=-1
)
stack_model.fit(X_train, y_train)
stack_pred = stack_model.predict(X_test)
stack_prob = stack_model.predict_proba(X_test)[:, 1]
print_metrics("Stacking", y_test, stack_pred, stack_prob)

# Simpan file standar saja
with open(os.path.join(current_dir, "random_forest_model.pkl"), "wb") as f:
    pickle.dump(rf_model, f)

with open(os.path.join(current_dir, "xgboost_model.pkl"), "wb") as f:
    pickle.dump(xgb_model, f)

with open(os.path.join(current_dir, "rule_lr.pkl"), "wb") as f:
    pickle.dump(stack_model.final_estimator_, f)

with open(os.path.join(current_dir, "feature_columns.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(selected_features))

with open(os.path.join(current_dir, "feature_medians.pkl"), "wb") as f:
    pickle.dump(train_medians.to_dict(), f)

print("\n✅ Model & metadata standar tersimpan.")