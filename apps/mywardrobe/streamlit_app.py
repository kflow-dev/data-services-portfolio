"""MyWardrobe — ShopTheLook outfit recommender."""

import streamlit as st

st.set_page_config(page_title="MyWardrobe", layout="centered")
st.title("MyWardrobe")
st.caption("ShopTheLook outfit recommender for occasions and seasons.")

occasion = st.text_input("Occasion", "business meeting")
season = st.selectbox("Season", ["spring", "summer", "fall", "winter"])

if st.button("Generate Outfit Recommendations", type="primary"):
    st.write(f"Outfits for **{occasion}** in **{season}**:")
    st.info("Classic business casual — Blazer + Chinos + Loafers")
    st.info("Smart casual — Cardigan + Oxford Shirt + Dark Jeans")
    st.info("Spring layers — Trench Coat + Sweater + Trousers")
