.PHONY: setup train api web test lint format

setup:
	python -m pip install -r requirements.txt

train:
	python -m src.train

api:
	uvicorn app.api:app --reload

web:
	streamlit run app/web_app.py

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .