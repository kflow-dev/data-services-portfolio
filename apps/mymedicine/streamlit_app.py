"""MyMedicine — Travel medicine lookup and drug availability."""

import streamlit as st

st.set_page_config(page_title="MyMedicine", layout="centered")
st.title("MyMedicine")
st.caption("Travel medicine lookup and drug availability checker.")

medicine = st.text_input("Medicine name or generic")
destination = st.text_input("Destination country/city")

if st.button("Check Availability", type="primary"):
    st.write(f"**{medicine}** availability in **{destination}**:")
    st.error(f"{destination}: Prescription required")
    st.info("Alternatives found:")
    st.info("Generic name: {generic} - Available")
    st.info("Local brand: BrandX - Available at 12 pharmacies")
