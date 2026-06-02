"""Cloud ML Estimator — ML infrastructure pricing estimator."""

import typer

app = typer.Typer(help="CloudML Estimator CLI: estimate ML infrastructure costs.")


@app.command()
def estimate(
    workload: str = typer.Argument(..., help="Workload type: 'training','inference','batch'"),
    scale: str = typer.Argument(..., help="Scale: 'small','medium','large','enterprise'"),
    cloud: str = typer.Option("aws", "--cloud", "-c", help="Cloud provider: 'aws','gcp','azure'"):
    """Estimate cloud ML infrastructure costs."""
    typer.echo(f"Workload: {workload}")
    typer.echo(f"Scale: {scale}")
    typer.echo(f"Cloud: {cloud}")
    typer.echo("\nCost estimate:")
    typer.echo("  Compute: $500/month")
    typer.echo("  Storage: $100/month")
    typer.echo("  Networking: $50/month")
    typer.echo("  Total: $650/month")
    typer.echo("\nRecommended instances:")
    typer.echo("  - 2x GPU instances (training)")
    typer.echo("  - 4x CPU instances (inference)")
    typer.echo("  - Auto-scaling enabled")


@app.command()
def compare_providers(
    workload_spec: str = typer.Argument(..., help="Workload specification JSON"):
    """Compare costs across cloud providers."""
    typer.echo(f"Workload: {workload_spec}")
    typer.echo("\nCost comparison:")
    typer.echo("  AWS: $720/month")
    typer.echo("  GCP: $680/month (best value)")
    typer.echo("  Azure: $750/month")


if __name__ == "__main__":
    app()
