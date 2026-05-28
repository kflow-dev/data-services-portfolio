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
