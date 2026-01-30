import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle

# Define the path to the data file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, 'DataFiles', 'Dataset89_terbaru.csv')

# Load the dataset
data = pd.read_csv(data_file_path)

# Extract features and labels (DROP URL!)
X = data.drop(columns=['url', 'label'])
y = data['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# RANDOM FOREST 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
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
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f'XGBoost Accuracy: {xgb_accuracy * 100:.2f}%')

with open(os.path.join(current_dir, 'xgboost_model.pkl'), 'wb') as file:
    pickle.dump(xgb_model, file)
