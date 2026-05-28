import json, os
import streamlit as st
import requests

SAMPLE = [
    {"id": "a1", "title": "Apple announces M5 chip with on-device LLM acceleration", "category": "tech"},
    {"id": "a2", "title": "ECB cuts interest rates by 25bps", "category": "finance"},
    {"id": "a3", "title": "New transformer beats Gemini 2.5 on MMLU", "category": "ai"},
    {"id": "a4", "title": "Real Madrid wins Club World Cup 3-1", "category": "sports"},
    {"id": "a5", "title": "EU AI Act compliance deadline arrives", "category": "policy"},
]

st.set_page_config(page_title="HotNews4U", layout="centered")
st.title("HotNews4U")
st.caption("Personalized news recommender powered by Lovable AI Gateway.")

interests = st.text_area("Your interests", "AI infrastructure, startups, climate policy")
if st.button("Recommend top 5", type="primary"):
    key = os.environ.get("LOVABLE_API_KEY")
    if not key:
        st.error("LOVABLE_API_KEY env var not set."); st.stop()
    with st.spinner("Ranking..."):
        r = requests.post(
            "https://ai.gateway.lovable.dev/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "google/gemini-2.5-flash",
                "messages": [
                    {"role": "system", "content": "Output ONLY valid JSON."},
                    {"role": "user", "content": f"Interests: {interests}\nArticles: {json.dumps(SAMPLE)}\nReturn JSON with key 'ranked'."},
                ],
                "response_format": {"type": "json_object"},
            }, timeout=60,
        )
    out = json.loads(r.json()["choices"][0]["message"]["content"])
    for i, rk in enumerate(out.get("ranked", []), 1):
        art = next((a for a in SAMPLE if a["id"] == rk["id"]), None)
        if not art: continue
        with st.container(border=True):
            st.markdown(f"**#{i} - {art['category']}** - {art['title']}")
            st.caption(f"{rk['reason']} - score {rk['score']:.2f}")
