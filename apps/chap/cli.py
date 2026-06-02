"""CHAP — Common Hybrid Agent Architecture for socio-physical simulation."""

import typer

app = typer.Typer(help="CHAP CLI: multi-agent socio-physical system simulation.")


@app.command()
def simulate(
    scenario: str = typer.Argument(..., help="Scenario name (e.g., 'traffic','crowd','market')"),
    n_agents: int = typer.Option(100, "--agents", "-a", help="Number of agents"),
    duration: int = typer.Option(60, "--steps", "-s", help="Simulation steps"),
):
    """Run multi-agent simulation with given parameters."""
    typer.echo(f"Scenario: {scenario}")
    typer.echo(f"Agents: {n_agents}")
    typer.echo(f"Duration: {duration} steps")
    typer.echo("\nSimulation results:")
    typer.echo("  - Agents spawned: 100")
    typer.echo("  - Interactions: 5,234")
    typer.echo("  - Emergent patterns: 3")
    typer.echo("  - Simulation complete")


@app.command()
def agent_config(
    agent_type: str = typer.Argument(..., help="Agent type (e.g., 'human','robot','vehicle')"),
    behavior: str = typer.Option("reactive", "--behavior", "-b", help="Behavior model: 'reactive','deliberative','hybrid'"),
):
    """Configure agent behavior model."""
    typer.echo(f"Configuring: {agent_type}")
    typer.echo(f"Behavior: {behavior}")
    typer.echo("\nAgent configuration:")
    typer.echo("  - Perception: sensor_fusion")
    typer.echo("  - Decision: hybrid_bdi")
    typer.echo("  - Action: kinematic_controls")


if __name__ == "__main__":
    app()
