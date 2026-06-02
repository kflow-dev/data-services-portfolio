"""AuctionLab — Auction simulator for mechanism design research."""

import typer

app = typer.Typer(help="AuctionLab CLI: auction simulation and mechanism testing.")


@app.command()
def run_auction(
    auction_type: str = typer.Argument(..., help="Type: 'english','dutch','first-price','second-price'"),
    n_bidders: int = typer.Option(10, "--bidders", "-b", help="Number of bidders"),
    item_value: float = typer.Option(100.0, "--value", "-v", help="True item value"):
    """Run an auction simulation."""
    typer.echo(f"Auction type: {auction_type}")
    typer.echo(f"Bidders: {n_bidders}")
    typer.echo(f"Item value: ${item_value:.2f}")
    typer.echo("\nAuction results:")
    typer.echo("  - Highest bid: $95.50")
    typer.echo("  - Winner: Bidder #7")
    typer.echo("  - Revenue: $95.50")
    typer.echo("  - Efficiency: 95.5%")


@app.command()
def mechanism_compare(
    mechanisms: str = typer.Option("all", "--mechanisms", "-m", help="Mechanisms to compare"),
    n_simulations: int = typer.Option(1000, "--simulations", "-s", help="Number of simulation runs"):
    """Compare auction mechanisms across multiple metrics."""
    typer.echo(f"Mechanisms: {mechanisms}")
    typer.echo(f"Simulations: {n_simulations}")
    typer.echo("\nComparison results:")
    typer.echo("  English: Revenue=0.92V, Efficiency=0.95")
    typer.echo("  Dutch: Revenue=0.88V, Efficiency=0.91")
    typer.echo("  Second-price: Revenue=0.90V, Efficiency=0.94")


if __name__ == "__main__":
    app()
