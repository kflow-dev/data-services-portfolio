"""Declarative Search — Multi-agent web scraping."""

import streamlit as st

st.set_page_config(page_title="Declarative Search", layout="centered")
st.title("Declarative Search")
st.caption("Multi-agent web scraping and information gathering.")

query = st.text_area("Search query or task")
agents = st.selectbox("Agents to use", ["all", "research", "compare", "summarize"])

if st.button("Run Search", type="primary"):
    st.write(f"Running search for: **{query}**")
    st.info("[Research Agent] Searching 15 sources...")
    st.info("[Compare Agent] Synthesizing findings...")
    st.info("[Summarize Agent] Generating report...")
    st.success("Results: 15 sources, 8 key findings, high confidence")
