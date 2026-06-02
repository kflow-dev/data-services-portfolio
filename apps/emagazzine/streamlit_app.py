"""EMagazzine — Price comparison aggregator."""

import streamlit as st

st.set_page_config(page_title="EMagazzine", layout="centered")
st.title("EMagazzine")
st.caption("Price comparison aggregator for e-commerce.")

product = st.text_input("Product name or search query")

if st.button("Compare Prices", type="primary"):
    st.write(f"Price comparison for: **{product}**")
    st.info("SiteA: $149.99 (shipping: free)")
    st.info("SiteB: $159.99 (shipping: $5.99)")
    st.info("SiteC: $145.00 (shipping: $9.99)")
    st.success("Best total: SiteC at $154.99")
