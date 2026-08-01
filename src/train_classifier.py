# ==============================
# src/train_classifier.py
# ==============================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import os

from src.preprocess import load_data, build_preprocessor


def create_salary_category(df, salary_column="salary", q=3, labels=("Low", "Medium", "High")):
    """Add a quantile-based salary_category column to df."""
    df = df.copy()
    df["salary_category"] = pd.qcut(df[salary_column], q=q, labels=list(labels))
    return df


def separate_features_target(df, target_column="salary_category", leak_columns=("salary", "salary_category")):
    """Split into X (features) and y (target), dropping all leak-prone columns from X."""
    y = df[target_column]
    X = df.drop(columns=list(leak_columns))
    return X, y


def encode_target(y):
    """Label-encode the categorical target. Returns (y_encoded, fitted label_encoder)."""
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return y_encoded, label_encoder


def split_data(X, y_encoded, test_size=0.2, random_state=42):
    """Stratified train-test split for classification."""
    return train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )


def train_logistic_regression(X_train_processed, y_train, max_iter=1000, random_state=42):
    """Train a Logistic Regression classifier."""
    classifier = LogisticRegression(max_iter=max_iter, random_state=random_state)
    classifier.fit(X_train_processed, y_train)
    return classifier


def evaluate_classifier(classifier, X_test_processed, y_test, label_encoder=None):
    """Evaluate classifier and return metrics + report as a dict."""
    y_pred = classifier.predict(X_test_processed)

    target_names = label_encoder.classes_ if label_encoder is not None else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "classification_report": classification_report(y_test, y_pred, target_names=target_names),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    return metrics


def save_classifier(classifier, label_encoder,
                     model_path="models/logistic_regression_model.pkl",
                     encoder_path="models/label_encoder.pkl"):
    """Save trained classifier and label encoder to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(classifier, model_path)
    joblib.dump(label_encoder, encoder_path)


def run_classification_pipeline(
    data_path="data/raw/salary.csv",
    preprocessor_save_path="models/preprocessor_classifier.pkl",
    model_save_path="models/logistic_regression_model.pkl",
    encoder_save_path="models/label_encoder.pkl"
):
    """Full pipeline: load, categorize, split, preprocess, train, evaluate, save."""
    df = load_data(data_path)
    df = create_salary_category(df)

    X, y = separate_features_target(df)
    y_encoded, label_encoder = encode_target(y)

    X_train, X_test, y_train, y_test = split_data(X, y_encoded)

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    os.makedirs(os.path.dirname(preprocessor_save_path), exist_ok=True)
    joblib.dump(preprocessor, preprocessor_save_path)

    classifier = train_logistic_regression(X_train_processed, y_train)
    metrics = evaluate_classifier(classifier, X_test_processed, y_test, label_encoder)
    save_classifier(classifier, label_encoder, model_save_path, encoder_save_path)

    return classifier, metrics