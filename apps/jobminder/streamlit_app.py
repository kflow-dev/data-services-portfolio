"""JobMinder — Job chatbot for career guidance."""

import streamlit as st

st.set_page_config(page_title="JobMinder", layout="centered")
st.title("JobMinder")
st.caption("Job chatbot for career guidance and application assistance.")

query = st.text_area("Your career question")
context = st.text_input("Your background (optional)")

if st.button("Get Advice", type="primary"):
    st.write("**JobMinder response:**")
    st.info("Based on your background, I recommend:")
    st.info("1. Tailor your resume to highlight relevant projects")
    st.info("2. Apply to these 3 positions")
    st.info("3. Prepare for interviews with these topics")
