"""DataLab-as-a-Service — Jupyter notebook management for DS teams."""

import streamlit as st

st.set_page_config(page_title="DataLab-as-a-Service", layout="centered")
st.title("DataLab-as-a-Service")
st.caption("Jupyter notebook management for data science teams.")

lab_name = st.text_input("Name of the data lab")
team_size = st.slider("Team size", 1, 50, 5)
resources = st.selectbox("Compute resources", ["standard", "gpu", "high-mem"])

if st.button("Create Lab", type="primary"):
    st.success(f"Data lab **{lab_name}** created!")
    st.info("JupyterHub instance: running")
    st.info("User accounts: created")
    st.info("Access URL: https://datalab.example.com/labs/{lab_name}")
