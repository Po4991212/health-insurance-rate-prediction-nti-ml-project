from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_validation import validate_schema

MODEL_PATH = ROOT / "artifacts" / "model.joblib"

app = FastAPI(title="Insurance Charges Predictor", version="1.0.0")

class InsuranceInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: str
    bmi: float = Field(..., ge=10, le=70)
    children: int = Field(..., ge=0, le=20)
    smoker: str
    region: str

class PredictionOut(BaseModel):
    prediction: float

class BatchPredictionOut(BaseModel):
    predictions: List[float]

def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run training first.")
    return joblib.load(MODEL_PATH)

model = load_model()

@app.get("/")
def read_root():
    return {"message": "API is running. Go to /docs for Swagger UI."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: InsuranceInput):
    try:
        df = pd.DataFrame([payload.model_dump()])
        validate_schema(df, training=False)
        pred = float(model.predict(df)[0])
        return {"prediction": pred}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.post("/predict_batch", response_model=BatchPredictionOut)
def predict_batch(payload: List[InsuranceInput]):
    try:
        df = pd.DataFrame([p.model_dump() for p in payload])
        validate_schema(df, training=False)
        preds = [float(x) for x in model.predict(df)]
        return {"predictions": preds}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
