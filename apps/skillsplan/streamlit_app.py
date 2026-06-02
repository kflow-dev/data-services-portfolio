"""SkillsPlan — Curriculum optimizer."""

import streamlit as st

st.set_page_config(page_title="SkillsPlan", layout="centered")
st.title("SkillsPlan")
st.caption("Curriculum optimizer and course recommender.")

goals = st.text_input("Learning goals (comma-separated)")
available_time = st.slider("Hours per week available", 1, 40, 10)

if st.button("Optimize Curriculum", type="primary"):
    st.write(f"Optimized curriculum for: **{goals}**")
    st.info("Month 1-2: Foundations (20 hrs)")
    st.info("Month 3-4: Core skills (25 hrs)")
    st.info("Month 5-6: Advanced topics (15 hrs)")
    st.info("Month 7-8: Capstone project (10 hrs)")
