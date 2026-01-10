# Insurance Medical Charges — End‑to‑End ML (Regression)

This repo is a small but complete Machine Learning project: it goes from raw data →
data validation + feature engineering → model training/evaluation → **FastAPI** serving →
a simple **Streamlit** UI.

**Goal:** predict `charges` (medical insurance cost) from personal attributes.

---

## What's inside

```
data/                  Dataset (insurance.csv)
notebooks/             EDA (eda.ipynb)
src/                   Training pipeline + validation + feature engineering
app/                   FastAPI API + Streamlit web app
artifacts/             Saved model (joblib) + metrics.json
reports/               PDF report + figures
tests/                 Quick unit tests (validation + API)
```

---

## Quickstart

Create and activate a virtual env, then install deps:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

Train and save the best model:

```bash
python -m src.train
```

Run the API:

```bash
uvicorn app.api:app --reload
```

Open docs in your browser:

* Swagger UI: `http://127.0.0.1:8000/docs`
* OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

Try a prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":30,"sex":"male","bmi":28.0,"children":0,"smoker":"no","region":"southeast"}'
```

Run the Streamlit UI:

```bash
streamlit run app/web_app.py
```

---

## Training CLI knobs (reproducible experiments)

```bash
python -m src.train --data data/insurance.csv --test-size 0.2 --seed 42 --artifacts artifacts
```

Local prediction without the API:

```bash
python -m src.predict --json '{"age":30,"sex":"male","bmi":28.0,"children":0,"smoker":"no","region":"southeast"}'
```

---

## Model comparison (test split 80/20, seed=42)

| model | MAE | MSE | R2 |
|---|---:|---:|---:|
| random_forest | 2430.35 | 1.975e+07 | 0.8728 |
| ridge | 2767.23 | 2.062e+07 | 0.8671 |
| linear_regression | 2762.71 | 2.070e+07 | 0.8667 |

Selected model: **RandomForestRegressor**.

---

## Helpful shortcuts

If you have `make` installed:

```bash
make train
make api
make web
make test
```

---

## Notes / limitations

* This is a learning project. The model is trained on a small public dataset.
* Predictions are only reliable inside the feature ranges seen in the dataset.
* Do **not** load untrusted `.joblib` model files in production.
