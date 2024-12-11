import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import json

# Load Data
def load_data():
    user_train = pd.read_csv('train/user_train.csv')
    session_train = pd.read_csv('train/session_train.csv')
    user_test = pd.read_csv('test/user_test.csv')
    session_test = pd.read_csv('test/session_test.csv')
    return user_train, session_train, user_test, session_test

# Data Preprocessing
def preprocess_data(user_data, session_data):
    # Handle missing values
    user_data.fillna({'age': user_data['age'].median()}, inplace=True)
    session_data.fillna({'page_views': 0, 'session_duration': 0}, inplace=True)

    # Aggregate session data
    session_agg = session_data.groupby('user_id').agg(
        avg_session_duration=('session_duration', 'mean'),
        total_page_views=('page_views', 'sum'),
        session_count=('session_id', 'count')
    ).reset_index()

    # Merge user data with aggregated session data
    merged_data = pd.merge(user_data, session_agg, on='user_id', how='left')
    merged_data.fillna({'avg_session_duration': 0, 'total_page_views': 0, 'session_count': 0}, inplace=True)

    # Encode categorical variables
    merged_data = pd.get_dummies(merged_data, columns=['user_category'], drop_first=True)

    return merged_data

# Train Model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    return model

# Predict and Evaluate
def predict_and_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 Score: {f1}")
    return y_pred

# Generate Predictions File
def generate_predictions(model, user_test, session_test):
    test_data = preprocess_data(user_test, session_test)
    predictions = model.predict(test_data.drop(columns=['user_id']))
    prediction_dict = {str(user_id): int(pred) for user_id, pred in zip(test_data['user_id'], predictions)}

    output = {"target": prediction_dict}
    with open('predictions/predictions.json', 'w') as f:
        json.dump(output, f)
    print("Predictions saved to predictions/predictions.json")

# Main Execution
if __name__ == "__main__":
    user_train, session_train, user_test, session_test = load_data()

    # Preprocess training data
    train_data = preprocess_data(user_train, session_train)
    X = train_data.drop(columns=['user_id', 'marketing_target'])
    y = train_data['marketing_target']

    # Split training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    predict_and_evaluate(model, X_val, y_val)

    # Generate predictions for test data
    generate_predictions(model, user_test, session_test)
