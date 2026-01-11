import requests
import streamlit as st

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")
st.title("Medical Insurance Charges Predictor")
st.write("Nhập thông tin và bấm **Dự đoán** để nhận dự đoán chi phí y tế (charges).")

API_URL = st.text_input("API URL", value="http://127.0.0.1:8000/predict")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    sex = st.selectbox("Sex", ["female", "male"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=28.0, step=0.1)
with col2:
    children = st.number_input("Children", min_value=0, max_value=20, value=0, step=1)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Dự đoán"):
    payload = {
        "age": int(age),
        "sex": sex,
        "bmi": float(bmi),
        "children": int(children),
        "smoker": smoker,
        "region": region,
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.text}")
        else:
            pred = r.json()["prediction"]
            st.success(f"Predicted charges: {pred:,.2f}")
    except Exception as e:
        st.error(f"Request failed: {e}")
