"""Cloud ML Estimator — ML infrastructure pricing estimator."""

import streamlit as st

st.set_page_config(page_title="Cloud ML Estimator", layout="centered")
st.title("Cloud ML Estimator")
st.caption("ML infrastructure pricing estimator for cloud providers.")

workload = st.selectbox("Workload type", ["training", "inference", "batch"])
scale = st.selectbox("Scale", ["small", "medium", "large", "enterprise"])
cloud = st.selectbox("Cloud provider", ["aws", "gcp", "azure"])

if st.button("Estimate Cost", type="primary"):
    st.write(f"Estimate for **{workload}** at **{scale}** scale on **{cloud}**:")
    st.info("Compute: $500/month")
    st.info("Storage: $100/month")
    st.info("Networking: $50/month")
    st.success("Total: $650/month")
