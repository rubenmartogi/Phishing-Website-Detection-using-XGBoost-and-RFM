import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, 'DataFiles', 'Dataset89_terbaru.csv')

data = pd.read_csv(data_file_path)

url_features = [
    "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens", "nb_at", "nb_qm", "nb_and",
    "nb_eq", "nb_underscore", "nb_tilde", "nb_percent", "nb_slash", "nb_star", "nb_colon",
    "nb_comma", "nb_semicolumn", "nb_dollar", "nb_space", "nb_www", "nb_com", "nb_dslash",
    "http_in_path", "https_token", "ratio_digits_url", "ratio_digits_host", "punycode", "port",
    "tld_in_path", "tld_in_subdomain", "abnormal_subdomain", "nb_subdomains", "prefix_suffix",
    "random_domain", "shortening_service", "path_extension", "nb_redirection"
]

X = data[url_features]
y = data['label']

# ✅ random_state=12 (sama dengan notebook)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)

# ✅ RANDOM FOREST - parameter sama dengan notebook
rf_model = RandomForestClassifier(
    n_estimators=200,   # ← ubah dari 100 → 200
    max_depth=5,        # ← tambah max_depth=5
    random_state=12,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(f'Random Forest Accuracy : {accuracy_score(y_test, rf_pred)*100:.2f}%')
print(f'Random Forest Precision: {precision_score(y_test, rf_pred)*100:.2f}%')
print(f'Random Forest Recall   : {recall_score(y_test, rf_pred)*100:.2f}%')
print(f'Random Forest F1       : {f1_score(y_test, rf_pred)*100:.2f}%')

with open(os.path.join(current_dir, 'random_forest_model.pkl'), 'wb') as file:
    pickle.dump(rf_model, file)
print('✅ random_forest_model.pkl tersimpan')

# ✅ XGBOOST - parameter sama dengan notebook
xgb_model = XGBClassifier(
    n_estimators=200,       # ← ubah dari 100 → 200
    learning_rate=0.1,
    max_depth=4,            # ← ubah dari 6 → 4
    reg_lambda=1.0,         # ← tambah
    reg_alpha=0.1,          # ← tambah
    subsample=0.8,          # ← tambah
    colsample_bytree=0.8,   # ← tambah
    gamma=0.1,              # ← tambah
    min_child_weight=2,     # ← tambah
    eval_metric='logloss',
    random_state=12,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print(f'XGBoost Accuracy : {accuracy_score(y_test, xgb_pred)*100:.2f}%')
print(f'XGBoost Precision: {precision_score(y_test, xgb_pred)*100:.2f}%')
print(f'XGBoost Recall   : {recall_score(y_test, xgb_pred)*100:.2f}%')
print(f'XGBoost F1       : {f1_score(y_test, xgb_pred)*100:.2f}%')

with open(os.path.join(current_dir, 'xgboost_model.pkl'), 'wb') as file:
    pickle.dump(xgb_model, file)
print('✅ xgboost_model.pkl tersimpan')

# ✅ STACKING ENSEMBLE - sama dengan notebook
stack_model = StackingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model)
    ],
    final_estimator=LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=12
    ),
    stack_method='predict_proba',
    n_jobs=-1
)
stack_model.fit(X_train, y_train)
stack_pred = stack_model.predict(X_test)
print(f'Stacking Accuracy : {accuracy_score(y_test, stack_pred)*100:.2f}%')
print(f'Stacking Precision: {precision_score(y_test, stack_pred)*100:.2f}%')
print(f'Stacking Recall   : {recall_score(y_test, stack_pred)*100:.2f}%')
print(f'Stacking F1       : {f1_score(y_test, stack_pred)*100:.2f}%')

with open(os.path.join(current_dir, 'rule_lr.pkl'), 'wb') as file:
    pickle.dump(stack_model.final_estimator_, file)
print('✅ rule_lr.pkl tersimpan')

# ✅ Simpan feature_columns.txt (dipakai app.py)
with open(os.path.join(current_dir, 'feature_columns.txt'), 'w') as f:
    f.write('\n'.join(url_features))
print('✅ feature_columns.txt tersimpan')