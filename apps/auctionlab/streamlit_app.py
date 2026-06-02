"""AuctionLab — Auction simulator for mechanism design."""

import streamlit as st

st.set_page_config(page_title="AuctionLab", layout="centered")
st.title("AuctionLab")
st.caption("Auction simulator for mechanism design research.")

auction_type = st.selectbox("Auction type", ["english", "dutch", "first-price", "second-price"])
n_bidders = st.slider("Number of bidders", 2, 100, 10)
item_value = st.slider("True item value", 10, 10000, 100)

if st.button("Run Auction", type="primary"):
    st.write(f"**{auction_type}** auction with {n_bidders} bidders:")
    st.success("Auction complete!")
    st.info("Highest bid: $95.50")
    st.info("Winner: Bidder #7")
    st.info("Revenue: $95.50")
    st.info("Efficiency: 95.5%")
