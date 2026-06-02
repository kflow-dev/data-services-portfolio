"""SkillsPlan — Curriculum optimizer and course recommender."""

import typer

app = typer.Typer(help="SkillsPlan CLI: optimize learning curriculum.")


@app.command()
def optimize(
    goals: str = typer.Argument(..., help="Learning goals (comma-separated)"),
    available_time: int = typer.Option(10, "--hours", "-h", help="Hours per week available"),
    prerequisites: str = typer.Option("", "--prereqs", "-p", help="Existing skills (comma-separated)"):
    """Generate optimized curriculum based on goals and constraints."""
    typer.echo(f"Goals: {goals}")
    typer.echo(f"Available time: {available_time} hrs/week")
    typer.echo(f"Prerequisites: {prerequisites or 'none specified'}")
    typer.echo("\nOptimized curriculum:")
    typer.echo("  Month 1-2: Foundations (20 hrs)")
    typer.echo("  Month 3-4: Core skills (25 hrs)")
    typer.echo("  Month 5-6: Advanced topics (15 hrs)")
    typer.echo("  Month 7-8: Capstone project (10 hrs)")


@app.command()
def recommend_courses(
    skill_area: str = typer.Argument(..., help="Skill area to learn"),
    level: str = typer.Option("beginner", "--level", "-l", help="Starting level"):
    """Recommend courses for a skill area."""
    typer.echo(f"Area: {skill_area}")
    typer.echo(f"Level: {level}")
    typer.echo("\nRecommended courses:")
    typer.echo("  1. Course A — 4 weeks — Rating: 4.8")
    typer.echo("  2. Course B — 6 weeks — Rating: 4.6")
    typer.echo("  3. Course C — 8 weeks — Rating: 4.9")


if __name__ == "__main__":
    app()
