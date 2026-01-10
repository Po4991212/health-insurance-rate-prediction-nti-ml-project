# Insurance Medical Charges - End-to-End ML Project

Bài toán: **Regression** – dự đoán `charges` (chi phí y tế) từ các đặc trưng cá nhân.

## Repo structure
- `data/insurance.csv`: dataset
- `notebooks/`: EDA
- `src/`: training pipeline, validation, feature engineering
- `app/`: FastAPI server + Streamlit UI
- `artifacts/`: model + metrics
- `reports/`: PDF report + figures

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

## Train
```bash
python -m src.train
```

## Run API
```bash
uvicorn app.api:app --reload
# test:
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"age":30,"sex":"male","bmi":28.0,"children":0,"smoker":"no","region":"southeast"}'
```

## Run Streamlit
```bash
streamlit run app/web_app.py
```

## Model comparison (test split 80/20, random_state=42)
            model         MAE          MSE       R2
    random_forest 2430.353436 1.975404e+07 0.872759
            ridge 2767.230443 2.062497e+07 0.867149
linear_regression 2762.709484 2.070044e+07 0.866663

Selected model: **random_forest**
