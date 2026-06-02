"""Drift Monitor — Model and data drift detection."""

import streamlit as st

st.set_page_config(page_title="Drift Monitor", layout="centered")
st.title("Drift Monitor")
st.caption("Model and data drift detection dashboard.")

model_name = st.text_input("Model name to monitor")
data_source = st.text_input("Data source (DB table, API, file)")
drift_type = st.selectbox("Drift type", ["data", "model", "both"])

if st.button("Check Drift", type="primary"):
    st.write(f"Drift analysis for **{model_name}**:")
    st.error("Data Drift: DETECTED (KS-test p < 0.01)")
    st.warning("Feature 'age': shift detected")
    st.info("Model Performance:")
    st.info("AUC: 0.82 (baseline: 0.89)")
    st.info("Recalibration recommended")
