from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
TARGET = "charges"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering on a DataFrame with the raw columns."""
    X = df.copy()

    # BMI category
    bins = [0, 18.5, 25, 30, np.inf]
    labels = ["underweight", "normal", "overweight", "obese"]
    X["bmi_category"] = pd.cut(X["bmi"], bins=bins, labels=labels, right=False)

    # Age group
    X["age_group"] = pd.cut(
        X["age"],
        bins=[0, 30, 45, 60, np.inf],
        labels=["<30", "30-44", "45-59", "60+"]
    )

    smoker_yes = (X["smoker"].astype(str).str.lower() == "yes").astype(int)
    X["smoker_yes"] = smoker_yes

    # Interaction terms
    X["bmi_x_smoker"] = X["bmi"] * smoker_yes
    X["age_x_smoker"] = X["age"] * smoker_yes
    return X
