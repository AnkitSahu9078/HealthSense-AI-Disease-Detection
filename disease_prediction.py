# disease_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train_model():
    # Load training data
    print("Loading training data...")
    training_data = pd.read_csv('training.csv')

    print(f"Dataset shape: {training_data.shape}")

    # Display column names for verification
    print("Columns in the dataset:")
    for i, col in enumerate(training_data.columns):
        print(f"{i}: {col}")

    # Identify the target column
    target_col = None
    for col in training_data.columns:
        if 'progn' in col.lower():
            target_col = col
            break

    if target_col is None:
        print("Could not identify the target column. Using the last column...")
        target_col = training_data.columns[-1]

    print(f"Using '{target_col}' as the target column")

    # Drop unnamed columns (like Unnamed: 133)
    unnamed_cols = [col for col in training_data.columns if 'unnamed' in col.lower()]
    if unnamed_cols:
        training_data = training_data.drop(columns=unnamed_cols)
        print(f"Dropped unnamed columns: {unnamed_cols}")

    # Features and target
    X = training_data.drop(columns=[target_col])
    y = training_data[target_col]

    print(f"Found {X.isna().sum().sum()} NaN values in features")
    print(f"Target column preview:\n{y.value_counts().head()}")

    y = y.fillna('Unknown')  # Handle missing values in target

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    disease_names = label_encoder.classes_

    print(f"Number of disease classes: {len(disease_names)}")
    print(f"Sample disease classes: {disease_names[:5]}")

    symptoms = X.columns.tolist()
    print(f"Number of symptoms: {len(symptoms)}")
    print(f"Sample symptoms: {symptoms[:5]}")

    # Impute missing feature values
    print("Imputing missing values in features...")
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = imputer.fit_transform(X)

    # ✅ Convert back to DataFrame with feature names (to avoid warnings later)
    X_df = pd.DataFrame(X_imputed, columns=symptoms)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X_df, y_encoded, test_size=0.2, random_state=42)

    # Train Random Forest model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_val, y_pred))

    # Save model data
    model_data = {
        'model': model,
        'symptoms': symptoms,
        'disease_names': disease_names,
        'imputer': imputer
    }

    # Save to consistent path
    model_path = 'disease_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Model trained and saved to '{model_path}' successfully.")
    return model_data

if __name__ == "__main__":
    train_model()
