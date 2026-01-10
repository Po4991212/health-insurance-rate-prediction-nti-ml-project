from __future__ import annotations

"""Local prediction script.

Examples
--------
Inline JSON:
    python -m src.predict --json '{"age":30,"sex":"male","bmi":28.0,"children":0,"smoker":"no","region":"southeast"}'

From file:
    python -m src.predict --json-file sample.json
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import TrainConfig
from .data_validation import validate_schema


def parse_args() -> argparse.Namespace:
    default = TrainConfig()
    p = argparse.ArgumentParser(description="Run local predictions using saved model")
    p.add_argument(
        "--model",
        type=Path,
        default=default.artifact_dir / "model.joblib",
        help="Path to model.joblib",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--json", type=str, help="Single JSON payload as a string")
    g.add_argument("--json-file", type=Path, help="Path to JSON file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}. Train first: python -m src.train")

    if args.json_file:
        payload = json.loads(args.json_file.read_text(encoding="utf-8"))
    else:
        payload = json.loads(args.json)

    # Allow either one object or a list of objects
    rows = payload if isinstance(payload, list) else [payload]

    df = pd.DataFrame(rows)
    validate_schema(df, training=False)

    model = joblib.load(args.model)
    preds = model.predict(df)
    out = [float(x) for x in preds]

    print(json.dumps({"predictions": out} if isinstance(payload, list) else {"prediction": out[0]}, indent=2))


if __name__ == "__main__":
    main()
