import os
import streamlit as st
import requests

st.set_page_config(page_title="PAPIE", layout="centered")
st.title("PAPIE - Personal AI assistant")

if "msgs" not in st.session_state:
    st.session_state.msgs = []

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask PAPIE..."):
    st.session_state.msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    key = os.environ.get("LOVABLE_API_KEY")
    if not key:
        st.error("Set LOVABLE_API_KEY env var."); st.stop()
    r = requests.post(
        "https://ai.gateway.lovable.dev/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "system", "content": "You are PAPIE, a personal information assistant."}]
                + st.session_state.msgs,
        }, timeout=60,
    )
    reply = r.json()["choices"][0]["message"]["content"]
    st.session_state.msgs.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
