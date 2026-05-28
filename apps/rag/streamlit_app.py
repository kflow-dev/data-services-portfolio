import os
import streamlit as st
import requests

BASE = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = os.environ.get("SUPABASE_ANON_KEY", "")
H = {"Authorization": f"Bearer {KEY}", "apikey": KEY, "Content-Type": "application/json"}

st.set_page_config(page_title="Multi-media RAG", layout="wide")
st.title("Multi-media RAG")

if not (BASE and KEY):
    st.error("Set SUPABASE_URL and SUPABASE_ANON_KEY env vars."); st.stop()

tab1, tab2 = st.tabs(["Ingest", "Ask"])
with tab1:
    src = st.text_input("Source name", "pasted-text")
    txt = st.text_area("Content", height=240)
    if st.button("Embed & index", disabled=not txt):
        r = requests.post(f"{BASE}/functions/v1/rag-ingest", headers=H, json={"source": src, "content": txt})
        st.json(r.json())
with tab2:
    q = st.text_input("Question", "What does the catalog say?")
    if st.button("Ask", disabled=not q):
        r = requests.post(f"{BASE}/functions/v1/rag-query", headers=H, json={"question": q, "k": 5})
        data = r.json()
        st.markdown(data.get("answer", ""))
        with st.expander("Retrieved chunks"):
            for i, m in enumerate(data.get("matches", []), 1):
                st.markdown(f"**[{i}] {m['source']}** (sim {m['similarity']:.2f})")
                st.write(m["content"])
*** Add File: apps/papie/cli.py
import os, sys
import typer, requests

app = typer.Typer()

@app.command()
def chat(message: str):
    key = os.environ.get("LOVABLE_API_KEY")
    if not key:
        typer.echo("Set LOVABLE_API_KEY", err=True); sys.exit(1)
    r = requests.post(
        "https://ai.gateway.lovable.dev/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "google/gemini-2.5-flash",
            "messages": [
                {"role": "system", "content": "You are PAPIE, a personal information assistant."},
                {"role": "user", "content": message},
            ],
        }, timeout=60,
    )
    r.raise_for_status()
    typer.echo(r.json()["choices"][0]["message"]["content"])

if __name__ == "__main__":
    app()
