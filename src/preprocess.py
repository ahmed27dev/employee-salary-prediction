# ==============================
# src/preprocess.py
# ==============================

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import os


# Fixed category orders (must match training)
EDUCATION_ORDER = ["High School", "Diploma", "Bachelor", "Master", "PhD"]
COMPANY_SIZE_ORDER = ["Startup", "Small", "Medium", "Large", "Enterprise"]
REMOTE_WORK_ORDER = ["No", "Hybrid", "Yes"]

NUMERICAL_FEATURES = ["experience_years", "skills_count", "certifications"]
ORDINAL_FEATURES = ["education_level", "company_size", "remote_work"]
NOMINAL_FEATURES = ["job_title", "industry", "location"]


def load_data(path):
    """Load raw dataset from CSV."""
    return pd.read_csv(path)


def separate_features_target(df, target_column="salary"):
    """Split DataFrame into X (features) and y (target)."""
    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y


def build_preprocessor():
    """Construct the ColumnTransformer (not yet fitted)."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("ord", OrdinalEncoder(
                categories=[EDUCATION_ORDER, COMPANY_SIZE_ORDER, REMOTE_WORK_ORDER]
            ), ORDINAL_FEATURES),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_FEATURES)
        ]
    )
    return preprocessor


def split_data(X, y, test_size=0.2, random_state=42):
    """Train-test split with fixed random_state for reproducibility."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def fit_and_save_preprocessor(X_train, save_path="models/preprocessor.pkl"):
    """Fit preprocessor on training data only, and save it to disk."""
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)

    return preprocessor


def load_preprocessor(path="models/preprocessor.pkl"):
    """Load a previously fitted preprocessor."""
    return joblib.load(path)