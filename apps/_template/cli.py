import typer

app = typer.Typer(help="Template CLI - replace with real commands.")

@app.command()
def hello(name: str = "world"):
    typer.echo(f"Hello, {name}!")

if __name__ == "__main__":
    app()
