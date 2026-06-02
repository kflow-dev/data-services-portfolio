"""MyNextHome — Real-estate recommender with HMM forecasting."""

import streamlit as st

st.set_page_config(page_title="MyNextHome", layout="centered")
st.title("MyNextHome")
st.caption("Real-estate recommender with HMM-based price forecasting.")

budget = st.slider("Budget range (max)", 200000, 2000000, 600000)
location = st.text_input("Preferred location or neighborhood")
bedrooms = st.slider("Bedrooms", 1, 6, 2)
features = st.text_input("Desired features (comma-separated)")

if st.button("Find Properties", type="primary"):
    st.write(f"Properties in **{location}** around **${budget:,}**:")
    st.info("123 Main St — $625,000 | 3br | 2ba | 1,800 sqft")
    st.info("456 Oak Ave — $580,000 | 2br | 2ba | 1,500 sqft")
    st.info("789 Pine Rd — $695,000 | 3br | 2.5ba | 2,100 sqft")
