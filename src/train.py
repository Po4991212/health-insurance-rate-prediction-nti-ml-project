from __future__ import annotations
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data_validation import validate_schema
from .features import FEATURES, TARGET
from .transformers import FeatureAdder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "insurance.csv"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

def build_preprocess():
    numeric_cols = ["age", "bmi", "children", "smoker_yes", "bmi_x_smoker", "age_x_smoker"]
    categorical_cols = ["sex", "smoker", "region", "bmi_category", "age_group"]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop"
    )

def make_pipeline(model):
    preprocess = build_preprocess()
    return Pipeline(steps=[
        ("feature_adder", FeatureAdder()),
        ("preprocess", preprocess),
        ("model", model),
    ])

def eval_regression(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

def main(random_state: int = 42):
    df = pd.read_csv(DATA_PATH)
    validate_schema(df, training=True)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=400, random_state=random_state, n_jobs=-1, min_samples_leaf=2
        )
    }

    results = {}
    best_name, best_r2, best_pipe = None, -1e9, None

    for name, model in models.items():
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        m = eval_regression(y_test, preds)
        results[name] = m

        if m["R2"] > best_r2:
            best_name, best_r2, best_pipe = name, m["R2"], pipe

    joblib.dump(best_pipe, ARTIFACT_DIR / "model.joblib")
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    print("Best model:", best_name)
    print("Metrics:", results[best_name])

if __name__ == "__main__":
    main()
