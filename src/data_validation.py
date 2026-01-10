from __future__ import annotations
import pandas as pd

COLUMNS_TRAIN = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
COLUMNS_INFER = ["age", "sex", "bmi", "children", "smoker", "region"]

ALLOWED_SEX = {"male", "female"}
ALLOWED_SMOKER = {"yes", "no"}
ALLOWED_REGION = {"northeast", "northwest", "southeast", "southwest"}

def validate_schema(df: pd.DataFrame, training: bool = False) -> None:
    """Basic schema validation. Raises ValueError with a helpful message."""
    expected = COLUMNS_TRAIN if training else COLUMNS_INFER

    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    errors = []
    if missing:
        errors.append(f"Missing columns: {missing}")
    if extra:
        errors.append(f"Unexpected columns: {extra}")

    if errors:
        raise ValueError(" | ".join(errors))

    def in_range(series, lo, hi):
        return series.between(lo, hi, inclusive="both").all()

    if not in_range(df["age"], 0, 120):
        errors.append("age must be in [0, 120]")
    if not in_range(df["bmi"], 10, 70):
        errors.append("bmi must be in [10, 70]")
    if not in_range(df["children"], 0, 20):
        errors.append("children must be in [0, 20]")

    sex_vals = set(df["sex"].astype(str).str.lower().unique())
    smoker_vals = set(df["smoker"].astype(str).str.lower().unique())
    region_vals = set(df["region"].astype(str).str.lower().unique())

    if not sex_vals.issubset(ALLOWED_SEX):
        errors.append(f"sex must be one of {sorted(ALLOWED_SEX)}")
    if not smoker_vals.issubset(ALLOWED_SMOKER):
        errors.append(f"smoker must be one of {sorted(ALLOWED_SMOKER)}")
    if not region_vals.issubset(ALLOWED_REGION):
        errors.append(f"region must be one of {sorted(ALLOWED_REGION)}")

    if training and (df["charges"] <= 0).any():
        errors.append("charges must be > 0")

    if errors:
        raise ValueError(" | ".join(errors))
