import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import json
import os

# Load Data
user_train = pd.read_csv('train/user_train.csv', delimiter=';')
session_train = pd.read_csv('train/session_train.csv', delimiter=',')
user_test = pd.read_csv('test/user_test.csv', delimiter=';')
session_test = pd.read_csv('test/session_test.csv', delimiter=',')

# Data Preprocessing
# Handle missing values
print("User Train Columns:", user_train.columns)
print("Session Train Columns:", session_train.columns)
print("User Test Columns:", user_test.columns)
print("Session Test Columns:", session_test.columns)

# Pre-processing user data
user_train.fillna({'age': user_train['age'].median()}, inplace=True)
session_train.fillna({'page_views': 0, 'session_duration': 0}, inplace=True)

# Aggregating session data
session_agg = session_train.groupby('user_id').agg(
    avg_session_duration=('session_duration', 'mean'),
    total_page_views=('page_views', 'sum'),
    session_count=('session_id', 'count')
).reset_index()

# Merging user data with aggregated session data
train_data = pd.merge(user_train, session_agg, on='user_id', how='left')
train_data.fillna({'avg_session_duration': 0, 'total_page_views': 0, 'session_count': 0}, inplace=True)

# One-hot encoding categorical variables
train_data = pd.get_dummies(train_data, columns=['user_category'], drop_first=True)

# Defining features and target variable
X = train_data.drop(columns=['user_id', 'marketing_target'])
y = train_data['marketing_target']

# Splitting the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
f1 = f1_score(y_val, y_pred, average='weighted')
print(f"F1 Score: {f1}")

# Pre-processing test data
user_test.fillna({'age': user_test['age'].median()}, inplace=True)
session_test.fillna({'page_views': 0, 'session_duration': 0}, inplace=True)

# Aggregating session data for test
session_agg_test = session_test.groupby('user_id').agg(
    avg_session_duration=('session_duration', 'mean'),
    total_page_views=('page_views', 'sum'),
    session_count=('session_id', 'count')
).reset_index()

# Merging user test data with aggregated session data
test_data = pd.merge(user_test, session_agg_test, on='user_id', how='left')
test_data.fillna({'avg_session_duration': 0, 'total_page_views': 0, 'session_count': 0}, inplace=True)

# One-hot encoding categorical variables for test data
test_data = pd.get_dummies(test_data, columns=['user_category'], drop_first=True)

# Store the user_id column separately before it gets dropped during the encoding process
user_ids = test_data['user_id']

# Ensuring that the test data columns match the training data columns
test_data = test_data[X.columns]  # Aligning columns

# Predicting on test data
predictions = model.predict(test_data)

# Preparing the predictions dictionary
prediction_dict = {str(user_id): int(pred) for user_id, pred in zip(user_ids, predictions)}

# Save predictions to a JSON file
output_dir = 'predictions'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output = {"target": prediction_dict}
with open(f'{output_dir}/predictions.json', 'w') as f:
    json.dump(output, f)

print("Predictions saved to predictions/predictions.json")
