"""MySmartDiet — Personalized diet and meal recommender."""

import typer

app = typer.Typer(help="MySmartDiet CLI: personalized diet recommendations.")


@app.command()
def recommend(
    goals: str = typer.Argument(..., help="Goals: 'weight_loss','muscle_gain','maintain'"),
    dietary_restrictions: str = typer.Option("", "--restrictions", "-d", help="Dietary restrictions (comma-separated)"):
    """Generate personalized diet recommendations."""
    typer.echo(f"Goals: {goals}")
    if dietary_restrictions:
        typer.echo(f"Dietary restrictions: {dietary_restrictions}")
    typer.echo("\nRecommended plan:")
    typer.echo("  Calories: 1800/day")
    typer.echo("  Macros: Protein 30%, Carbs 40%, Fat 30%")
    typer.echo("  Meals:")
    typer.echo("    - Breakfast: Oatmeal with berries (400 cal)")
    typer.echo("    - Lunch: Grilled chicken salad (550 cal)")
    typer.echo("    - Dinner: Salmon with vegetables (600 cal)")


@app.command()
def calorie_counter(
    food_item: str = typer.Argument(..., help="Food item to log"),
    portion: float = typer.Option(1.0, "--portion", "-p", help="Portion size (servings)"):
    """Track calories for a food item."""
    typer.echo(f"Logging: {food_item}")
    typer.echo(f"Portion: {portion} servings")
    typer.echo("\nNutrition info:")
    typer.echo("  Calories: 250")
    typer.echo("  Protein: 15g")
    typer.echo("  Carbs: 30g")
    typer.echo("  Fat: 8g")


if __name__ == "__main__":
    app()
