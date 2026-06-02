"""E-books/Audiobook RecSys — Content-based recommender."""

import streamlit as st

st.set_page_config(page_title="E-books/Audiobook RecSys", layout="centered")
st.title("E-books/Audiobook RecSys")
st.caption("Content-based recommender for reading materials.")

genre = st.text_input("Genre (e.g., sci-fi, biography, self-help)")
reading_level = st.selectbox("Reading level", ["beginner", "intermediate", "advanced"])
format_type = st.radio("Format", ["book", "audiobook", "both"])

if st.button("Get Recommendations", type="primary"):
    st.write(f"Recommendations for **{genre}** ({reading_level}, {format_type}):")
    st.info("The Quantum Thief (sci-fi) — 4.5 stars")
    st.info("Sapiens (non-fiction) — 4.8 stars")
    st.info("Project Hail Mary (sci-fi) — 4.7 stars")
