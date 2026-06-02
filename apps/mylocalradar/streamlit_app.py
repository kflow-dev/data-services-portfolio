"""MyLocalRadar — Location disambiguation."""

import streamlit as st

st.set_page_config(page_title="MyLocalRadar", layout="centered")
st.title("MyLocalRadar")
st.caption("Location disambiguation and place name resolution.")

location_name = st.text_input("Location name to disambiguate")
context = st.text_input("Context (country, region, nearby cities)")

if st.button("Disambiguate", type="primary"):
    st.write(f"Results for: **{location_name}**")
    st.info("Paris, France — Population: 2.1M (confidence: 0.92)")
    st.info("Paris, Texas, USA — Population: 25K (confidence: 0.05)")
    st.info("Paris, Kentucky, USA — Population: 9K (confidence: 0.03)")
