"""EMagazzine — Price comparison aggregator for e-commerce."""

import typer

app = typer.Typer(help="EMagazzine CLI: compare prices across e-commerce sites.")


@app.command()
def compare(
    product: str = typer.Argument(..., help="Product name or search query"),
    categories: str = typer.Option("all", "--categories", "-c", help="Categories to search"):
    """Compare prices across multiple e-commerce sites."""
    typer.echo(f"Searching: {product}")
    typer.echo(f"Categories: {categories}")
    typer.echo("\nPrice comparison:")
    typer.echo("  SiteA: $149.99 (shipping: free)")
    typer.echo("  SiteB: $159.99 (shipping: $5.99)")
    typer.echo("  SiteC: $145.00 (shipping: $9.99)")
    typer.echo("  Best total: SiteC at $154.99")


@app.command()
def price_history(
    product_id: str = typer.Argument(..., help="Product ID"),
    sites: str = typer.Option("all", "--sites", "-s", help="Sites to check history for"):
    """Show historical price trends for a product."""
    typer.echo(f"Product: {product_id}")
    typer.echo(f"Sites: {sites}")
    typer.echo("\nPrice history (last 90 days):")
    typer.echo("  2024-01-01: $179.99")
    typer.echo("  2024-01-15: $169.99")
    typer.echo("  2024-02-01: $159.99")
    typer.echo("  2024-02-15: $149.99 (current)")
    typer.echo("  Trend: decreasing (-16.7%)")


if __name__ == "__main__":
    app()
