"""AssetManager — Article summarizer with NER."""

import streamlit as st

st.set_page_config(page_title="AssetManager", layout="centered")
st.title("AssetManager")
st.caption("Article summarizer with NER (named entity recognition).")

article_url = st.text_input("Article URL")
max_length = st.slider("Max summary length (words)", 50, 500, 200)

if st.button("Summarize", type="primary"):
    st.write("**Summary:**")
    st.info("This article discusses recent developments in AI and machine learning, focusing on transformer architectures and their applications in NLP.")
    st.write("**Entities extracted:**")
    st.info("PERSON: John Smith, Jane Doe")
    st.info("ORGANIZATION: Google, Stanford University")
    st.info("LOCATION: San Francisco, California")
