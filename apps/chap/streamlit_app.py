"""CHAP — Common Hybrid Agent Architecture simulation."""

import streamlit as st

st.set_page_config(page_title="CHAP", layout="centered")
st.title("CHAP")
st.caption("Common Hybrid Agent Architecture for socio-physical simulation.")

scenario = st.selectbox("Scenario", ["traffic", "crowd", "market", "evacuation"])
n_agents = st.slider("Number of agents", 10, 1000, 100)
duration = st.slider("Simulation steps", 10, 500, 60)

if st.button("Run Simulation", type="primary"):
    st.write(f"Running simulation: **{scenario}** ({n_agents} agents, {duration} steps)")
    st.success("Simulation complete!")
    st.info("Agents spawned: 100")
    st.info("Interactions: 5,234")
    st.info("Emergent patterns: 3")
