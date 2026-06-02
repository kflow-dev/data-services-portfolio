"""CoolDrinks — Beer SKU recommender for B2B shops."""

import streamlit as st

st.set_page_config(page_title="CoolDrinks", layout="centered")
st.title("CoolDrinks")
st.caption("Beer SKU recommender for B2B shops with geo-location scraping.")

shop_type = st.selectbox("Shop type", ["convenience", "specialty", "bar", "supermarket"])
location = st.text_input("Location (city or coordinates)")
customer_segments = st.text_input("Customer segments (e.g., young, students, professionals)")

if st.button("Recommend SKUs", type="primary"):
    st.write(f"Recommendations for **{shop_type}** in **{location}**:")
    st.info("Craft IPA (local brewery) — 12 cases")
    st.info("Light Lager — 24 cases")
    st.info("Sour Ales (limited) — 6 cases")
    st.info("Pilsner — 18 cases")
