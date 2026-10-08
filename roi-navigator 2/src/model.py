"""Predictive maintenance model on the AI4I 2020 dataset (UCI, CC BY 4.0).

Trains two models and returns their predicted failure probabilities on a held-out test set:
  - Logistic regression: simple, explainable baseline
  - Gradient boosting:   stronger model for tabular data

Failure-mode columns (TWF, HDF, PWF, OSF, RNF) are excluded because they describe the
failure itself and would leak the answer.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/ai4i2020.csv"
TARGET = "Machine failure"
RANDOM_STATE = 42


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({
        "air_temp_k": df["Air temperature [K]"],
        "process_temp_k": df["Process temperature [K]"],
        "rpm": df["Rotational speed [rpm]"],
        "torque_nm": df["Torque [Nm]"],
        "tool_wear_min": df["Tool wear [min]"],
    })
    # Engineered features a maintenance engineer would recognize
    X["temp_diff_k"] = X["process_temp_k"] - X["air_temp_k"]          # heat dissipation
    X["power_w"] = X["torque_nm"] * X["rpm"] * 2 * np.pi / 60          # mechanical power
    X["wear_x_torque"] = X["tool_wear_min"] * X["torque_nm"]           # overstrain
    # Product quality variant: L / M / H
    X = pd.concat([X, pd.get_dummies(df["Type"], prefix="type", dtype=int)], axis=1)
    return X


def train_and_score(df: pd.DataFrame | None = None, test_size: float = 0.3) -> dict:
    """Train both models; return test labels, probabilities and AUCs."""
    df = load_data() if df is None else df
    X, y = build_features(df), df[TARGET].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Logistic regression (baseline)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }
    results = {"y_test": y_te, "n_train": len(y_tr), "n_test": len(y_te), "models": {}}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        results["models"][name] = {"proba": proba, "auc": roc_auc_score(y_te, proba)}
    return results
