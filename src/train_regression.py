# ==============================
# src/train_regression.py
# ==============================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

from src.preprocess import (
    load_data,
    separate_features_target,
    split_data,
    load_preprocessor
)


def train_linear_regression(X_train_processed, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train_processed, y_train)
    return model


def evaluate_model(model, X_test_processed, y_test):
    """Evaluate a regression model and return metrics as a dict."""
    y_pred = model.predict(X_test_processed)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {"mae": mae, "rmse": rmse, "r2": r2}


def get_coefficients(model, preprocessor):
    """Return a DataFrame mapping feature names to coefficients, sorted by impact."""
    feature_names = preprocessor.get_feature_names_out()
    coefficients_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_
    })
    coefficients_df["abs_coefficient"] = coefficients_df["coefficient"].abs()
    coefficients_df = coefficients_df.sort_values(by="abs_coefficient", ascending=False)
    return coefficients_df.drop(columns="abs_coefficient")


def save_model(model, save_path="models/linear_regression_model.pkl"):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)


def run_training_pipeline(
    data_path="data/raw/salary.csv",
    preprocessor_path="models/preprocessor.pkl",
    model_save_path="models/linear_regression_model.pkl"
):
    """Full pipeline: load data, preprocess, train, evaluate, save. Returns model + metrics."""
    df = load_data(data_path)
    X, y = separate_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = load_preprocessor(preprocessor_path)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = train_linear_regression(X_train_processed, y_train)
    metrics = evaluate_model(model, X_test_processed, y_test)
    save_model(model, model_save_path)

    return model, metrics