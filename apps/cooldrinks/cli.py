"""CoolDrinks - Context-aware beverage recommender.

SOTA Techniques:
- Transformer-based sequential recommendation (SASRec)
- Multi-modal fusion (taste + weather + time)
- Bandit-based exploration-exploitation (LinUCB)

Usage:
    CLI:      python cli.py recommend --weather hot --time afternoon --occasion casual
    Streamlit: streamlit run streamlit_app.py
    Notebook: jupyter notebook notebooks/advanced_context_aware_recommender.ipynb
"""

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

app = typer.Typer(help="CoolDrinks: Context-aware beverage recommender with SOTA ML.")

# ============================================================================
# IMPORTS
# ============================================================================

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.data_layer import (
    load_drink_data,
    load_interaction_data,
    generate_drink_catalog,
    generate_context_scenarios,
    generate_interaction_logs,
    get_drink_stats,
    get_context_stats,
)
from models.recommender import HybridRecommenderEngine


# ============================================================================
# GLOBAL STATE
# ============================================================================

_recommender: Optional[HybridRecommenderEngine] = None


def get_recommender() -> HybridRecommenderEngine:
    """Get or create recommender engine (lazy initialization)."""
    global _recommender
    if _recommender is None:
        data_dir = "data/synthetic"
        drink_df = load_drink_data(data_dir)
        interactions_df = load_interaction_data(data_dir)
        _recommender = HybridRecommenderEngine(drink_df, interactions_df)
    return _recommender


# ============================================================================
# CONTEXT ENCODING
# ============================================================================


def encode_context(
    weather: str,
    time_period: str,
    occasion: str,
    bitterness_pref: float = 0.5,
    sweetness_pref: float = 0.5,
    strength_pref: float = 0.5
) -> dict:
    """Encode context into unified representation.

    Args:
        weather: Weather condition (sunny, rainy, cloudy, snowy, stormy)
        time_period: Time period (morning, afternoon, evening)
        occasion: Occasion type (casual, celebration, pairing, recovery, social, business)
        bitterness_pref: User bitterness preference [0, 1]
        sweetness_pref: User sweetness preference [0, 1]
        strength_pref: User strength preference [0, 1]

    Returns:
        Context representation dictionary
    """
    valid_weather = ["sunny", "rainy", "cloudy", "snowy", "stormy"]
    valid_time = ["morning", "afternoon", "evening"]
    valid_occasion = ["casual", "celebration", "pairing", "recovery", "social", "business"]

    return {
        "weather": weather if weather in valid_weather else "sunny",
        "time_period": time_period if time_period in valid_time else "afternoon",
        "occasion": occasion if occasion in valid_occasion else "casual",
        "bitterness_pref": max(0, min(1, bitterness_pref)),
        "sweetness_pref": max(0, min(1, sweetness_pref)),
        "strength_pref": max(0, min(1, strength_pref)),
    }


# ============================================================================
# RECOMMENDATION
# ============================================================================


def recommend_drinks(
    user_id: str,
    weather: str,
    time_period: str,
    occasion: str,
    bitterness_pref: float = 0.5,
    sweetness_pref: float = 0.5,
    strength_pref: float = 0.5,
    top_k: int = 10,
    excluded_items: Optional[list] = None
) -> list:
    """Get top-k recommendations for user and context.

    Args:
        user_id: User identifier
        weather: Weather condition
        time_period: Time period
        occasion: Occasion type
        bitterness_pref: User bitterness preference [0, 1]
        sweetness_pref: User sweetness preference [0, 1]
        strength_pref: User strength preference [0, 1]
        top_k: Number of recommendations
        excluded_items: Items to exclude

    Returns:
        List of recommendation dictionaries
    """
    recommender = get_recommender()

    context = encode_context(
        weather, time_period, occasion,
        bitterness_pref, sweetness_pref, strength_pref
    )

    return recommender.recommend(
        user_id=user_id,
        weather=context["weather"],
        time_period=context["time_period"],
        occasion=context["occasion"],
        bitterness_pref=context["bitterness_pref"],
        sweetness_pref=context["sweetness_pref"],
        strength_pref=context["strength_pref"],
        top_k=top_k,
        excluded_items=excluded_items
    )


# ============================================================================
# CLI COMMANDS
# ============================================================================


@app.command()
def recommend(
    user_id: str = typer.Option(
        "U001",
        "--user",
        "-u",
        help="User identifier (e.g., 'U001')"
    ),
    weather: str = typer.Option(
        "sunny",
        "--weather",
        "-w",
        help="Weather condition (sunny, rainy, cloudy, snowy, stormy)"
    ),
    hour: int = typer.Option(
        14,
        "--hour",
        "-h",
        help="Hour of day (6-23)"
    ),
    occasion: str = typer.Option(
        "casual",
        "--occasion",
        "-o",
        help="Occasion type (casual, celebration, pairing, recovery, social, business)"
    ),
    bitterness: float = typer.Option(
        0.5,
        "--bitterness",
        "-b",
        help="Bitterness preference (0-1, default 0.5)"
    ),
    sweetness: float = typer.Option(
        0.5,
        "--sweetness",
        "-s",
        help="Sweetness preference (0-1, default 0.5)"
    ),
    strength: float = typer.Option(
        0.5,
        "--strength",
        "-t",
        help="Strength preference (0-1, default 0.5)"
    ),
    top_k: int = typer.Option(
        5,
        "--top",
        "-k",
        help="Number of recommendations to return"
    ),
):
    """Get context-aware beverage recommendations.

    Example:
        python cli.py recommend --weather sunny --hour 14 --occasion casual
        python cli.py recommend -u U001 -w rainy -h 20 -o celebration -b 0.7 -s 0.3
    """
    # Determine time period from hour
    if hour < 12:
        time_period = "morning"
    elif hour < 18:
        time_period = "afternoon"
    else:
        time_period = "evening"

    typer.echo(f"Generating recommendations for user {user_id}...")
    typer.echo(f"Context: {weather} weather, {time_period} ({hour:02d}:00), {occasion} occasion")
    typer.echo(f"Taste preferences: bitterness={bitterness:.1f}, sweetness={sweetness:.1f}, strength={strength:.1f}")
    typer.echo()

    try:
        recommendations = recommend_drinks(
            user_id=user_id,
            weather=weather,
            time_period=time_period,
            occasion=occasion,
            bitterness_pref=bitterness,
            sweetness_pref=sweetness,
            strength_pref=strength,
            top_k=top_k
        )

        typer.echo(f"Top {top_k} recommendations:\n")

        for i, rec in enumerate(recommendations, 1):
            typer.echo(f"{i}. {rec['name']}")
            typer.echo(f"   Type: {rec['type']} | Style: {rec['style']} | ABV: {rec['abv']}%")
            typer.echo(f"   Flavor: B={rec['bitterness']:.0f}/100, S={rec['sweetness']:.0f}/100, C={rec['carbonation']:.1f}")
            typer.echo(f"   Overall Score: {rec['overall_score']:.4f}")
            typer.echo(f"   SASRec: {rec['sasrec_score']:.4f} | Fusion: {rec['fusion_score']:.4f} | LinUCB: {rec['linucb_score']:.4f}")
            typer.echo(f"   Seasonality: {rec['seasonality']}")

            # Generate explanation
            context = encode_context(weather, time_period, occasion, bitterness, sweetness, strength)
            explanation = get_recommender().explain_recommendation(rec["drink_id"], context)
            typer.echo(f"   Why: {explanation}")
            typer.echo()

    except Exception as e:
        typer.echo(f"Error generating recommendations: {e}", err=True)
        sys.exit(1)


@app.command()
def list_drinks(
    drink_type: str = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by drink type (beer, wine, coffee, tea, cocktail, non-alcoholic)"
    ),
    min_abv: float = typer.Option(
        None,
        "--min-abv",
        help="Minimum ABV"
    ),
    max_abv: float = typer.Option(
        None,
        "--max-abv",
        help="Maximum ABV"
    ),
    style: str = typer.Option(
        None,
        "--style",
        help="Filter by style"
    ),
    seasonality: str = typer.Option(
        None,
        "--seasonality",
        help="Filter by seasonality (summer, winter, spring, fall, any)"
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of drinks to display"
    ),
):
    """List available drinks with optional filters."""
    drink_df = load_drink_data("data/synthetic")

    # Apply filters
    filtered = drink_df

    if drink_type:
        filtered = filtered[filtered["type"] == drink_type]

    if min_abv is not None:
        filtered = filtered[filtered["abv"] >= min_abv]

    if max_abv is not None:
        filtered = filtered[filtered["abv"] <= max_abv]

    if style:
        filtered = filtered[filtered["style"] == style]

    if seasonality:
        filtered = filtered[filtered["seasonality"] == seasonality]

    # Display results
    typer.echo(f"Found {len(filtered)} drinks")
    typer.echo()

    for _, drink in filtered.head(limit).iterrows():
        typer.echo(f"{drink['drink_id']}: {drink['name']}")
        typer.echo(f"   {drink['type'].capitalize()} - {drink['style'].replace('_', ' ')}")
        typer.echo(f"   ABV: {drink['abv']}% | B: {drink['bitterness']:.0f} | S: {drink['sweetness']:.0f} | C: {drink['carbonation']:.1f}")
        typer.echo(f"   Season: {drink['seasonality']} | Origin: {drink['origin']}")
        typer.echo()


@app.command()
def list_contexts():
    """List available context combinations."""
    typer.echo("Available Context Combinations:")
    typer.echo()

    typer.echo("Weather Conditions:")
    for weather in ["sunny", "rainy", "cloudy", "snowy", "stormy"]:
        typer.echo(f"  - {weather}")

    typer.echo()
    typer.echo("Time Periods:")
    for period in ["morning (6-12)", "afternoon (12-18)", "evening (18-24)"]:
        typer.echo(f"  - {period}")

    typer.echo()
    typer.echo("Occasions:")
    for occasion in ["casual", "celebration", "pairing", "recovery", "social", "business"]:
        typer.echo(f"  - {occasion}")


@app.command(name="generate-data")
def generate_data(
    n_drinks: int = typer.Option(
        120,
        "--drinks",
        "-n",
        help="Number of drinks to generate"
    ),
    n_interactions: int = typer.Option(
        10000,
        "--interactions",
        "-i",
        help="Number of interactions to generate"
    ),
    n_users: int = typer.Option(
        500,
        "--users",
        "-u",
        help="Number of unique users"
    ),
    output_dir: str = typer.Option(
        "data/synthetic",
        "--output-dir",
        "-o",
        help="Output directory"
    ),
):
    """Generate synthetic dataset for demonstration."""
    typer.echo(f"Generating synthetic data...")
    typer.echo(f"  Drinks: {n_drinks}")
    typer.echo(f"  Interactions: {n_interactions}")
    typer.echo(f"  Users: {n_users}")
    typer.echo()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate drink catalog
    typer.echo("Generating drink catalog...")
    drinks_df = generate_drink_catalog(n_drinks)
    drinks_path = output_path / "drinks_catalog.csv"
    drinks_df.to_csv(drinks_path, index=False)
    typer.echo(f"  Saved: {drinks_path}")

    # Generate interaction logs
    typer.echo("Generating interaction logs...")
    scenarios_df = generate_context_scenarios(n_scenarios=50)
    interactions_df = generate_interaction_logs(
        n_interactions=n_interactions,
        n_users=n_users,
        drinks_df=drinks_df,
        scenarios_df=scenarios_df
    )
    interactions_path = output_path / "interaction_logs.csv"
    interactions_df.to_csv(interactions_path, index=False)
    typer.echo(f"  Saved: {interactions_path}")

    # Display statistics
    typer.echo()
    typer.echo("Statistics:")
    typer.echo(f"  Unique drinks: {drinks_df['drink_id'].nunique()}")
    typer.echo(f"  Unique users: {interactions_df['user_id'].nunique()}")
    typer.echo(f"  Interactions: {len(interactions_df)}")
    typer.echo(f"  By type: {drinks_df['type'].value_counts().to_dict()}")

    typer.echo()
    typer.echo("Data generation complete!")


@app.command()
def stats(
    data_dir: str = typer.Option(
        "data/synthetic",
        "--data-dir",
        "-d",
        help="Data directory"
    ),
):
    """Display statistics about the dataset."""
    drink_df = load_drink_data(data_dir)
    interactions_df = load_interaction_data(data_dir)

    typer.echo("Dataset Statistics")
    typer.echo("=" * 50)
    typer.echo()

    typer.echo("Drink Catalog:")
    stats = get_drink_stats(drink_df)
    typer.echo(f"  Total drinks: {stats['total_drinks']}")
    typer.echo(f"  By type: {stats['by_type']}")
    typer.echo(f"  ABV range: {stats['abv_range'][0]:.1f}% - {stats['abv_range'][1]:.1f}%")
    typer.echo(f"  Avg bitterness: {stats['avg_bitterness']:.1f}")
    typer.echo(f"  Avg sweetness: {stats['avg_sweetness']:.1f}")
    typer.echo()

    typer.echo("Interaction Data:")
    ctx_stats = get_context_stats(interactions_df)
    typer.echo(f"  Total interactions: {ctx_stats['total_interactions']}")
    typer.echo(f"  Unique users: {ctx_stats['unique_users']}")
    typer.echo(f"  Unique drinks: {ctx_stats['unique_drinks']}")
    typer.echo(f"  By weather: {ctx_stats['by_weather']}")
    typer.echo(f"  By time period: {ctx_stats['by_time_period']}")
    typer.echo(f"  By occasion: {ctx_stats['by_occasion']}")
    typer.echo(f"  Interaction types: {ctx_stats['interaction_types']}")


if __name__ == "__main__":
    app()
