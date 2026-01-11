from __future__ import annotations

"""Training entrypoint.

Run:
    python -m src.train


"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import TrainConfig
from .data_validation import validate_schema
from .features import FEATURES, TARGET
from .transformers import FeatureAdder


def build_preprocess() -> ColumnTransformer:
    """Preprocessing used for all models.
    """
    numeric_cols = ["age", "bmi", "children", "smoker_yes", "bmi_x_smoker", "age_x_smoker"]
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


def cv_eval(pipe: Pipeline, X_train, y_train, seed: int) -> dict[str, float]:
    """Cross-validation on training set only."""
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    scores = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        n_jobs=-1,
        return_train_score=False,
    )

    cv_mae = -scores["test_mae"]
    cv_r2 = scores["test_r2"]

    return {
        "CV_MAE_mean": float(cv_mae.mean()),
        "CV_MAE_std": float(cv_mae.std()),
        "CV_R2_mean": float(cv_r2.mean()),
        "CV_R2_std": float(cv_r2.std()),
    }


def train(cfg: TrainConfig) -> dict:
    df = pd.read_csv(cfg.data_path)
    validate_schema(df, training=True)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # Models
    models = {
        # Required baseline
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=cfg.random_state),

        # Log-target variants (often helps with right-skewed charges)
        "linear_log": TransformedTargetRegressor(
            regressor=LinearRegression(),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "ridge_log": TransformedTargetRegressor(
            regressor=Ridge(alpha=1.0, random_state=cfg.random_state),
            func=np.log1p,
            inverse_func=np.expm1,
        ),

        # Non-linear model
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            random_state=cfg.random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    }

    results: dict[str, dict[str, float]] = {}

    best_name = None
    best_score = -1e9
    best_pipe: Pipeline | None = None

    for name, model in models.items():
        pipe = make_pipeline(model)

        # CV metrics on train set
        cv_metrics = cv_eval(pipe, X_train, y_train, cfg.random_state)

        # Fit once and evaluate on held-out test
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        test_metrics = eval_regression(y_test, preds)

        results[name] = {**test_metrics, **cv_metrics}

        # Pick best by CV_R2_mean (more robust than a single split)
        # Condition: lower CV_MAE_mean
        score = results[name]["CV_R2_mean"]
        if (score > best_score) or (
            score == best_score and results[name]["CV_MAE_mean"] < results[best_name]["CV_MAE_mean"]  # type: ignore
        ):
            best_score = score
            best_name = name
            best_pipe = pipe

    assert best_name is not None and best_pipe is not None

    # Error analysis on test by smoker for the selected model
    best_preds = best_pipe.predict(X_test)
    test_tmp = X_test.copy()
    test_tmp["y_true"] = y_test.values
    test_tmp["y_pred"] = best_preds
    test_tmp["abs_err"] = (test_tmp["y_true"] - test_tmp["y_pred"]).abs()
    mae_by_smoker = test_tmp.groupby("smoker")["abs_err"].mean().to_dict()

    # Save residual plot (optional, but nice for report)
    cfg.artifact_dir.mkdir(exist_ok=True)
    residuals = y_test.values - best_preds
    plt.figure()
    plt.scatter(best_preds, residuals, s=10, alpha=0.6)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.title(f"Residual plot (best model: {best_name})")
    plt.tight_layout()
    plt.savefig(cfg.artifact_dir / "residual_plot.png", dpi=200)
    plt.close()

    # Persist best model pipeline
    joblib.dump(best_pipe, cfg.artifact_dir / "model.joblib")

    # Save metrics + config + error analysis
    with open(cfg.artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "best_model": best_name,
                "selection_rule": "best CV_R2_mean, tie-breaker lowest CV_MAE_mean",
                "error_analysis": {
                    "MAE_by_smoker": mae_by_smoker
                },
                "config": {
                    "data_path": str(cfg.data_path),
                    "test_size": cfg.test_size,
                    "random_state": cfg.random_state,
                },
            },
            f,
            indent=2,
        )

    # Save metadata 
    with open(cfg.artifact_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "trained_at_utc": datetime.utcnow().isoformat() + "Z",
                "sklearn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
                "features": FEATURES,
                "target": TARGET,
                "best_model": best_name,
            },
            f,
            indent=2,
        )

    print(f"Best model: {best_name}")
    print("Best model metrics:", results[best_name])
    print("MAE by smoker (test):", mae_by_smoker)
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
