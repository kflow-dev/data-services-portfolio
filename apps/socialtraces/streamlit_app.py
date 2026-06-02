"""SocialTraces — Social media fuzzy search."""

import streamlit as st

st.set_page_config(page_title="SocialTraces", layout="centered")
st.title("SocialTraces")
st.caption("Social media fuzzy search and network analysis.")

query = st.text_input("Search query (fuzzy matching)")
platforms = st.multiselect("Platforms", ["twitter", "linkedin", "all"], default=["all"])

if st.button("Search", type="primary"):
    st.write(f"Results for: **{query}**")
    st.info("Twitter: 12 posts (fuzzy match: 0.85)")
    st.info("LinkedIn: 8 posts (fuzzy match: 0.78)")
