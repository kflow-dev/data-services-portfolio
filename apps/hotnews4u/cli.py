import json, os, sys
import typer, requests

app = typer.Typer()

SAMPLE = [
    {"id": "a1", "title": "Apple announces M5 chip with on-device LLM acceleration", "category": "tech"},
    {"id": "a2", "title": "ECB cuts interest rates by 25bps", "category": "finance"},
    {"id": "a3", "title": "New transformer beats Gemini 2.5 on MMLU", "category": "ai"},
    {"id": "a4", "title": "Real Madrid wins Club World Cup 3-1", "category": "sports"},
    {"id": "a5", "title": "EU AI Act compliance deadline arrives", "category": "policy"},
]

@app.command()
def recommend(interests: str = typer.Option(..., help="Comma-separated interests")):
    key = os.environ.get("LOVABLE_API_KEY")
    if not key:
        typer.echo("Set LOVABLE_API_KEY", err=True); sys.exit(1)
    prompt = (
        f"User interests: {interests}\n\nArticles:\n{json.dumps(SAMPLE)}\n\n"
        'Rank top 5 as JSON: {"ranked":[{"id":"...","score":0-1,"reason":"..."}]}'
    )
    r = requests.post(
        "https://ai.gateway.lovable.dev/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "google/gemini-2.5-flash",
            "messages": [
                {"role": "system", "content": "Output ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }, timeout=60,
    )
    r.raise_for_status()
    typer.echo(r.json()["choices"][0]["message"]["content"])

if __name__ == "__main__":
    app()
