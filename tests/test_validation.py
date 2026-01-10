from __future__ import annotations

import pandas as pd

from src.data_validation import validate_schema


def test_validate_schema_ok_inference() -> None:
    df = pd.DataFrame(
        [
            {
                "age": 30,
                "sex": "male",
                "bmi": 28.0,
                "children": 0,
                "smoker": "no",
                "region": "southeast",
            }
        ]
    )
    validate_schema(df, training=False)


def test_validate_schema_missing_column() -> None:
    df = pd.DataFrame([{"age": 30}])
    try:
        validate_schema(df, training=False)
    except ValueError as e:
        assert "Missing columns" in str(e)
    else:
        raise AssertionError("Expected ValueError")