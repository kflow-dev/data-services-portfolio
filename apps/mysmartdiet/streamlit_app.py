"""MySmartDiet — Personalized diet recommender."""

import streamlit as st

st.set_page_config(page_title="MySmartDiet", layout="centered")
st.title("MySmartDiet")
st.caption("Personalized diet and meal recommender.")

goals = st.selectbox("Goals", ["weight_loss", "muscle_gain", "maintain"])
dietary_restrictions = st.text_input("Dietary restrictions (comma-separated)")

if st.button("Get Diet Plan", type="primary"):
    st.write("**Recommended plan:**")
    st.info("Calories: 1800/day")
    st.info("Macros: Protein 30%, Carbs 40%, Fat 30%")
    st.info("Breakfast: Oatmeal with berries (400 cal)")
    st.info("Lunch: Grilled chicken salad (550 cal)")
    st.info("Dinner: Salmon with vegetables (600 cal)")
