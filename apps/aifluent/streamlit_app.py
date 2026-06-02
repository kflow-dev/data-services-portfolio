"""AIFluent — Skills acquisition platform with RL recommendation."""

import streamlit as st

st.set_page_config(page_title="AIFluent", layout="centered")
st.title("AIFluent")
st.caption("Skills acquisition platform with RL-based recommendation.")

current_skills = st.text_input("Current skills (comma-separated)")
target_role = st.text_input("Target job role")

if st.button("Create Learning Plan", type="primary"):
    st.write(f"Learning plan for **{target_role}**:")
    st.info("Week 1-2: Python Advanced (online course)")
    st.info("Week 3-4: Machine Learning Fundamentals")
    st.info("Week 5-6: Deep Learning with PyTorch")
    st.info("Week 7-8: MLOps and Deployment")
