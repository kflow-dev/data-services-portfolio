"""DataLab-as-a-Service — Jupyter notebook management for DS teams."""

import typer

app = typer.Typer(help="DataLab CLI: manage Jupyter notebooks for data science teams.")


@app.command()
def create_lab(
    lab_name: str = typer.Argument(..., help="Name of the data lab"),
    team_size: int = typer.Option(5, "--size", "-s", help="Number of team members"),
    resources: str = typer.Option("standard", "--resources", "-r", help="Compute resources: 'standard', 'gpu', 'high-mem'"):
):
    """Create a new data lab environment with Jupyter."""
    typer.echo(f"Creating data lab: {lab_name}")
    typer.echo(f"Team size: {team_size}")
    typer.echo(f"Resources: {resources}")
    typer.echo("\nLab created:")
    typer.echo("  - JupyterHub instance: running")
    typer.echo("  - User accounts: created")
    typer.echo("  - Shared storage: /data/labs/{lab_name}")
    typer.echo("  - Access URL: https://datalab.example.com/labs/{lab_name}")


@app.command()
def list_labs():
    """List all data labs and their status."""
    typer.echo("Active data labs:")
    typer.echo("  1. hr-analytics — 5 users — GPU — Running")
    typer.echo("  2. fraud-detection — 3 users — Standard — Running")
    typer.echo("  3. forecasting — 4 users — High-Mem — Stopped")


if __name__ == "__main__":
    app()
