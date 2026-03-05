import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import pickle

# Define the path to the data file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, 'DataFiles', 'Dataset89_terbaru.csv')

# Load the dataset
data = pd.read_csv(data_file_path)

# List fitur URL-based
url_features = [
    "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens", "nb_at", "nb_qm", "nb_and",
    "nb_eq", "nb_underscore", "nb_tilde", "nb_percent", "nb_slash", "nb_star", "nb_colon",
    "nb_comma", "nb_semicolumn", "nb_dollar", "nb_space", "nb_www", "nb_com", "nb_dslash",
    "http_in_path", "https_token", "ratio_digits_url", "ratio_digits_host", "punycode", "port",
    "tld_in_path", "tld_in_subdomain", "abnormal_subdomain", "nb_subdomains", "prefix_suffix",
    "random_domain", "shortening_service", "path_extension", "nb_redirection"
]

# Extract only URL-based features
X = data[url_features]
y = data['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# RANDOM FOREST 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f'Random Forest Accuracy: {rf_accuracy * 100:.2f}%')

with open(os.path.join(current_dir, 'random_forest_model.pkl'), 'wb') as file:
    pickle.dump(rf_model, file)

# XGBOOST 
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f'XGBoost Accuracy: {xgb_accuracy * 100:.2f}%')

with open(os.path.join(current_dir, 'xgboost_model.pkl'), 'wb') as file:
    pickle.dump(xgb_model, file)

# STACKING ENSEMBLE (RF + XGB -> Logistic Regression)
stack_model = StackingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model)
    ],
    final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    stack_method='predict_proba',
    n_jobs=-1
)
stack_model.fit(X_train, y_train)

stack_pred = stack_model.predict(X_test)
stack_accuracy = accuracy_score(y_test, stack_pred)
print(f'Stacking Accuracy: {stack_accuracy * 100:.2f}%')

# Simpan meta-learner (Logistic Regression) ke rule_lr.pkl
with open(os.path.join(current_dir, 'rule_lr.pkl'), 'wb') as file:
    pickle.dump(stack_model.final_estimator_, file)