"""JobPromis — Job recommender with skills gap analysis."""

import streamlit as st

st.set_page_config(page_title="JobPromis", layout="centered")
st.title("JobPromis")
st.caption("Job recommender with skills gap analysis.")

current_role = st.text_input("Current job title or role")
target_role = st.text_input("Target job title (optional)")
location = st.text_input("Preferred location (optional)")

if st.button("Find Jobs", type="primary"):
    st.write(f"Jobs for **{current_role}**:")
    st.info("Senior Data Scientist — Remote — $150K")
    st.info("ML Engineer — San Francisco — $160K")
    st.info("AI Research Scientist — NYC — $170K")
