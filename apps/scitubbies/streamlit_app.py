"""SciTubbies — YouTube content recommender for science/tech."""

import streamlit as st

st.set_page_config(page_title="SciTubbies", layout="centered")
st.title("SciTubbies")
st.caption("YouTube content recommender for science and tech education.")

topic = st.text_input("Topic (e.g., machine learning, physics, biology)")
channel_type = st.selectbox("Channel type", ["all", "educational", "research", "news"])
duration = st.selectbox("Video length", ["short (<10min)", "medium (10-30min)", "long (>30min)"])

if st.button("Find Videos", type="primary"):
    st.write(f"Videos about **{topic}**:")
    st.info("Intro to Transformers — 3Blue1Brown — 25min")
    st.info("Quantum Computing Explained — Veritasium — 18min")
    st.info("The Future of AI — Lex Fridman — 45min")
