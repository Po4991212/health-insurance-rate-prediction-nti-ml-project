from __future__ import annotations

"""Training entrypoint.

This module can be run as:
    python -m src.train

It also supports CLI arguments (data path, test size, seed, output directory)
to make experiments reproducible without editing code.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import TrainConfig
from .data_validation import validate_schema
from .features import FEATURES, TARGET
from .transformers import FeatureAdder


def build_preprocess() -> ColumnTransformer:
    """Preprocessing used for all models.

    Note: we intentionally keep all feature engineering inside the sklearn
    Pipeline so training and inference share the exact same logic.
    """

    # Numeric features are raw + engineered numeric columns
    numeric_cols = [
        "age",
        "bmi",
        "children",
        "smoker_yes",
        "bmi_x_smoker",
        "age_x_smoker",
    ]
    # Categorical features include raw and engineered categories
    categorical_cols = ["sex", "smoker", "region", "bmi_category", "age_group"]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_adder", FeatureAdder()),
            ("preprocess", build_preprocess()),
            ("model", model),
        ]
    )


def eval_regression(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train(cfg: TrainConfig) -> dict:
    df = pd.read_csv(cfg.data_path)
    validate_schema(df, training=True)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=cfg.random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            random_state=cfg.random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    }

    results: dict[str, dict[str, float]] = {}
    best_name, best_r2, best_pipe = None, -1e9, None

    for name, model in models.items():
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = eval_regression(y_test, preds)
        results[name] = metrics

        if metrics["R2"] > best_r2:
            best_name, best_r2, best_pipe = name, metrics["R2"], pipe

    cfg.artifact_dir.mkdir(exist_ok=True)
    joblib.dump(best_pipe, cfg.artifact_dir / "model.joblib")
    with open(cfg.artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "best_model": best_name,
                "config": {
                    "data_path": str(cfg.data_path),
                    "test_size": cfg.test_size,
                    "random_state": cfg.random_state,
                },
            },
            f,
            indent=2,
        )

    print(f"Best model: {best_name}")
    print("Metrics:", results[best_name])
    return {"best_model": best_name, "results": results}


def parse_args() -> TrainConfig:
    default = TrainConfig()
    p = argparse.ArgumentParser(description="Train insurance charges regressor")
    p.add_argument("--data", type=Path, default=default.data_path, help="Path to CSV")
    p.add_argument(
        "--artifacts",
        type=Path,
        default=default.artifact_dir,
        help="Where to save model.joblib and metrics.json",
    )
    p.add_argument("--test-size", type=float, default=default.test_size)
    p.add_argument("--seed", type=int, default=default.random_state)
    args = p.parse_args()
    return TrainConfig(
        data_path=args.data,
        artifact_dir=args.artifacts,
        test_size=args.test_size,
        random_state=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
