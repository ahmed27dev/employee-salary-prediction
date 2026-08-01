# ==============================
# src/predict.py
# ==============================

import pandas as pd
import joblib

FEATURE_COLUMNS = [
    "job_title", "experience_years", "education_level", "skills_count",
    "industry", "company_size", "location", "remote_work", "certifications"
]


def load_regression_artifacts(
    preprocessor_path="models/preprocessor.pkl",
    model_path="models/linear_regression_model.pkl"
):
    """Load the regression preprocessor and model."""
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    return preprocessor, model


def load_classification_artifacts(
    preprocessor_path="models/preprocessor_classifier.pkl",
    model_path="models/logistic_regression_model.pkl",
    encoder_path="models/label_encoder.pkl"
):
    """Load the classification preprocessor, model, and label encoder."""
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    return preprocessor, model, label_encoder


def build_input_dataframe(form_data: dict):
    """
    Convert raw form input (dict) into a single-row DataFrame
    with correct column order and data types.
    """
    row = {
        "job_title": form_data["job_title"],
        "experience_years": int(form_data["experience_years"]),
        "education_level": form_data["education_level"],
        "skills_count": int(form_data["skills_count"]),
        "industry": form_data["industry"],
        "company_size": form_data["company_size"],
        "location": form_data["location"],
        "remote_work": form_data["remote_work"],
        "certifications": int(form_data["certifications"]),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_salary(input_df, preprocessor, model):
    """Predict continuous salary using the regression pipeline."""
    processed = preprocessor.transform(input_df)
    prediction = model.predict(processed)
    return round(float(prediction[0]), 2)


def predict_salary_category(input_df, preprocessor, model, label_encoder):
    """Predict salary category (Low/Medium/High) using the classification pipeline."""
    processed = preprocessor.transform(input_df)
    prediction_encoded = model.predict(processed)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)
    return prediction_label[0]


def predict_all(form_data: dict):
    """
    Full prediction pipeline: takes raw form input,
    returns both salary number and salary category.
    """
    input_df = build_input_dataframe(form_data)

    reg_preprocessor, reg_model = load_regression_artifacts()
    salary_prediction = predict_salary(input_df, reg_preprocessor, reg_model)

    clf_preprocessor, clf_model, label_encoder = load_classification_artifacts()
    category_prediction = predict_salary_category(input_df, clf_preprocessor, clf_model, label_encoder)

    return {
        "predicted_salary": salary_prediction,
        "predicted_category": category_prediction
    }