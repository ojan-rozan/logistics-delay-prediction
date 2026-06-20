"""Streamlit app — prediksi keterlambatan pengiriman.

Jalankan dari root project:
    streamlit run streamlit_app.py

Butuh model hasil training di output/model_delay_catboost.pkl
(jalankan dulu: python -m test.train_model).
"""
import datetime as dt

import joblib
import pandas as pd
import streamlit as st

from src import config as C
from src.data import data_processing as dp

st.set_page_config(page_title="Logistics Delay Predictor", page_icon="🚚", layout="centered")


@st.cache_resource
def load_model():
    if not C.MODEL_FILE.exists():
        return None
    return joblib.load(C.MODEL_FILE)


@st.cache_data
def category_options():
    """Ambil opsi dropdown dari dataset bila tersedia, kalau tidak pakai default."""
    defaults = {
        "Origin_Warehouse": ["Warehouse_LA", "Warehouse_SF", "Warehouse_NY", "Warehouse_TX"],
        "Destination": ["Chicago", "Phoenix", "New York", "Los Angeles", "Dallas"],
        "Carrier": ["LaserShip", "OnTrac", "FedEx", "UPS", "USPS", "DHL"],
    }
    if C.RAW_FILE.exists():
        try:
            df = pd.read_csv(C.RAW_FILE, usecols=list(defaults))
            return {c: sorted(df[c].dropna().unique().tolist()) for c in defaults}
        except Exception:
            pass
    return defaults


def predict(model, row: dict):
    X = dp.add_features(pd.DataFrame([row]))
    proba = float(model.predict_proba(X)[0, 1])
    return proba


st.title("🚚 Delivery Delay Predictor")
st.caption("Prediksi probabilitas keterlambatan pengiriman (US Logistics).")

model = load_model()
if model is None:
    st.error("Model belum ada. Jalankan dulu: `python -m test.train_model`")
    st.stop()

opts = category_options()
tab_single, tab_batch = st.tabs(["🔹 Prediksi Satuan", "📄 Prediksi Batch (CSV)"])

with tab_single:
    col1, col2 = st.columns(2)
    with col1:
        ship_date = st.date_input("Tanggal Pengiriman", dt.date(2023, 7, 15))
        origin = st.selectbox("Origin Warehouse", opts["Origin_Warehouse"])
        dest = st.selectbox("Destination", opts["Destination"])
        carrier = st.selectbox("Carrier", opts["Carrier"])
    with col2:
        weight = st.number_input("Weight (kg)", min_value=0.0, value=12.5, step=0.5)
        cost = st.number_input("Cost (USD)", min_value=0.0, value=85.0, step=1.0)
        distance = st.number_input("Distance (miles)", min_value=0.0, value=750.0, step=10.0)

    if st.button("Prediksi", type="primary", use_container_width=True):
        row = {
            C.COL_SHIP_DATE: str(ship_date),
            "Origin_Warehouse": origin, "Destination": dest, "Carrier": carrier,
            "Weight_kg": weight, "Cost": cost, "Distance_miles": distance,
        }
        proba = predict(model, row)
        label = "Delayed ⚠️" if proba >= 0.5 else "On-time ✅"
        st.metric("Probabilitas Telat", f"{proba*100:.1f}%")
        st.progress(min(proba, 1.0))
        (st.error if proba >= 0.5 else st.success)(f"Prediksi: **{label}**")

with tab_batch:
    st.write("Upload CSV berisi kolom: "
             "`Shipment_Date, Origin_Warehouse, Destination, Carrier, "
             "Weight_kg, Cost, Distance_miles`")
    up = st.file_uploader("Pilih file CSV", type="csv")
    if up is not None:
        df_in = pd.read_csv(up)
        feats = dp.add_features(df_in)
        df_in["delay_probability"] = model.predict_proba(feats)[:, 1].round(4)
        df_in["prediction"] = (df_in["delay_probability"] >= 0.5).map(
            {True: "Delayed", False: "On-time"})
        st.dataframe(df_in, use_container_width=True)
        st.download_button("Download hasil", df_in.to_csv(index=False),
                           "predictions.csv", "text/csv")
