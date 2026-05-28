import os, sys
import typer, requests

app = typer.Typer()

def edge(path: str):
    base = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")
    if not base or not key:
        typer.echo("Set SUPABASE_URL and SUPABASE_ANON_KEY", err=True); sys.exit(1)
    return base.rstrip("/") + "/functions/v1/" + path, {
        "Authorization": f"Bearer {key}", "apikey": key, "Content-Type": "application/json",
    }

@app.command()
def ingest(file: str, source: str = "cli-upload"):
    url, h = edge("rag-ingest")
    with open(file) as f:
        content = f.read()
    r = requests.post(url, headers=h, json={"source": source, "content": content}, timeout=120)
    typer.echo(r.text)

@app.command()
def ask(question: str, k: int = 5):
    url, h = edge("rag-query")
    r = requests.post(url, headers=h, json={"question": question, "k": k}, timeout=120)
    typer.echo(r.text)

if __name__ == "__main__":
    app()
