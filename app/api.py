from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]

# Tip: run the API from the repo root so `src/...` imports work naturally:
#   uvicorn app.api:app --reload
from src.data_validation import validate_schema

MODEL_PATH = ROOT / "artifacts" / "model.joblib"

app = FastAPI(
    title="Insurance Charges Predictor",
    version="1.0.0",
    description=(
        "Predict medical insurance charges using a trained scikit-learn pipeline. "
        "Use **/predict** for a single record or **/predict_batch** for multiple records."
    ),
)

class InsuranceInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: str
    bmi: float = Field(..., ge=10, le=70)
    children: int = Field(..., ge=0, le=20)
    smoker: str
    region: str

    # Makes Swagger UI show a nice default example.
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 30,
                    "sex": "male",
                    "bmi": 28.0,
                    "children": 0,
                    "smoker": "no",
                    "region": "southeast",
                }
            ]
        }
    }

class PredictionOut(BaseModel):
    prediction: float

class BatchPredictionOut(BaseModel):
    predictions: List[float]

def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = None


@app.on_event("startup")
def _startup():
    global model
    model = load_model()


@app.get("/")
def root():
    return {
        "message": "Insurance Charges Predictor API",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "model_loaded": model is not None,
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: InsuranceInput):
    try:
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded. Train first: python -m src.train",
            )
        df = pd.DataFrame([payload.model_dump()])
        validate_schema(df, training=False)
        pred = float(model.predict(df)[0])
        return {"prediction": pred}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.post("/predict_batch", response_model=BatchPredictionOut)
def predict_batch(payload: List[InsuranceInput]):
    try:
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded. Train first: python -m src.train",
            )
        df = pd.DataFrame([p.model_dump() for p in payload])
        validate_schema(df, training=False)
        preds = [float(x) for x in model.predict(df)]
        return {"predictions": preds}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
