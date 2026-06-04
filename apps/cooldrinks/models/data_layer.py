"""Data layer for CoolDrinks context-aware recommender.

Provides synthetic data generation for:
- Drink catalog with flavor profiles
- User interaction logs
- Context scenarios
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List


# ============================================================================
# DRINK CATALOG CONSTANTS
# ============================================================================

DRINK_TYPES = ["beer", "wine", "coffee", "tea", "cocktail", "non-alcoholic"]

BEER_STYLES = [
    "pale_ale", "ipa", "stout", "lager", "wheat", "sour", "porter",
    "amber_ale", "pilsner", "brown_ale", "red_ale", "imperial_stout"
]

WINE_STYLES = ["red", "white", "rose", "sparkling", "dessert", "fortified"]

COFFEE_STYLES = ["espresso", "drip", "cold_brew", "french_press", "pour_over", "aero_press"]

TEA_STYLES = ["black", "green", "oolong", "white", "herbal", "pu_erh", "matcha"]

COCKTAIL_STYLES = ["martini", "old_fashioned", "mojito", "margrita", "whiskey_sour",
                   "gin_and_tonic", "negroni", "margarita", "daiquiri", "manhattan"]

NON_ALCOHOLIC_STYLES = ["sparkling_water", "juice", "kombucha", "energy_drink",
                        "lemonade", "iced_tea", "smoothie", "mocktail"]

ORIGINS = ["usa", "germany", "belgium", "ireland", "uk", "france", "spain",
           "italy", "japan", "australia", "mexico", "canada", "netherlands"]

OCCASIONS = ["casual", "celebration", "pairing", "recovery", "social", "business"]

WEATHER_CONDITIONS = ["sunny", "rainy", "cloudy", "snowy", "stormy"]

# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================


def generate_drink_catalog(n_drinks: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic drink catalog with flavor profiles.

    Args:
        n_drinks: Number of drinks to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with drink attributes
    """
    np.random.seed(seed)

    drinks = []
    drink_id = 1

    # Generate beer catalog (50 drinks)
    for i in range(50):
        style = BEER_STYLES[i % len(BEER_STYLES)]
        abv = np.random.uniform(3.0, 12.0)
        bitterness = np.random.uniform(10, 100) if style in ["ipa", "pale_ale", "porter", "stout"] else np.random.uniform(5, 40)
        sweetness = np.random.uniform(10, 60)
        carbonation = np.random.uniform(2, 4.5)
        seasonality = np.random.choice(["any", "summer", "winter", "spring", "fall"],
                                       p=[0.3, 0.25, 0.15, 0.15, 0.15])

        drinks.append({
            "drink_id": f"D{drink_id:03d}",
            "name": f"{style.capitalize().replace('_', ' ')} {drink_id}",
            "type": "beer",
            "style": style,
            "abv": round(abv, 1),
            "bitterness": round(bitterness, 1),
            "sweetness": round(sweetness, 1),
            "carbonation": round(carbonation, 1),
            "strength": round(abv / 12, 2),
            "seasonality": seasonality,
            "origin": np.random.choice(ORIGINS),
        })
        drink_id += 1

    # Generate wine catalog (20 drinks)
    for i in range(20):
        style = WINE_STYLES[i % len(WINE_STYLES)]
        abv = np.random.uniform(8.0, 16.0)
        sweetness = np.random.uniform(5, 70) if style in ["dessert", "fortified"] else np.random.uniform(0, 30)
        bitterness = np.random.uniform(0, 20) if style == "red" else np.random.uniform(0, 5)
        carbonation = 2.0 if style == "sparkling" else np.random.uniform(0, 1)

        drinks.append({
            "drink_id": f"D{drink_id:03d}",
            "name": f"{style.capitalize()} Wine {drink_id}",
            "type": "wine",
            "style": style,
            "abv": round(abv, 1),
            "bitterness": round(bitterness, 1),
            "sweetness": round(sweetness, 1),
            "carbonation": round(carbonation, 1),
            "strength": round(abv / 16, 2),
            "seasonality": "any",
            "origin": np.random.choice(ORIGINS),
        })
        drink_id += 1

    # Generate coffee catalog (15 drinks)
    for i in range(15):
        style = COFFEE_STYLES[i % len(COFFEE_STYLES)]
        abv = 0.0
        bitterness = np.random.uniform(30, 90)
        sweetness = np.random.uniform(0, 80)
        carbonation = np.random.uniform(0, 1)
        strength = np.random.uniform(0.3, 0.9)

        drinks.append({
            "drink_id": f"D{drink_id:03d}",
            "name": f"{style.replace('_', ' ').capitalize()} Coffee {drink_id}",
            "type": "coffee",
            "style": style,
            "abv": abv,
            "bitterness": round(bitterness, 1),
            "sweetness": round(sweetness, 1),
            "carbonation": round(carbonation, 1),
            "strength": round(strength, 2),
            "seasonality": "any",
            "origin": np.random.choice(ORIGINS),
        })
        drink_id += 1

    # Generate tea catalog (15 drinks)
    for i in range(15):
        style = TEA_STYLES[i % len(TEA_STYLES)]
        abv = 0.0
        bitterness = np.random.uniform(5, 50)
        sweetness = np.random.uniform(0, 40)
        carbonation = np.random.uniform(0, 1)
        strength = np.random.uniform(0.1, 0.6)

        drinks.append({
            "drink_id": f"D{drink_id:03d}",
            "name": f"{style.capitalize()} Tea {drink_id}",
            "type": "tea",
            "style": style,
            "abv": abv,
            "bitterness": round(bitterness, 1),
            "sweetness": round(sweetness, 1),
            "carbonation": round(carbonation, 1),
            "strength": round(strength, 2),
            "seasonality": "any",
            "origin": np.random.choice(ORIGINS),
        })
        drink_id += 1

    # Generate cocktail catalog (10 drinks)
    for i in range(10):
        style = COCKTAIL_STYLES[i % len(COCKTAIL_STYLES)]
        abv = np.random.uniform(15, 40)
        bitterness = np.random.uniform(10, 50)
        sweetness = np.random.uniform(10, 60)
        carbonation = np.random.uniform(0, 3) if "tonic" in style else np.random.uniform(0, 1)
        strength = abv / 40

        drinks.append({
            "drink_id": f"D{drink_id:03d}",
            "name": f"{style.replace('_', ' ').capitalize()} {drink_id}",
            "type": "cocktail",
            "style": style,
            "abv": round(abv, 1),
            "bitterness": round(bitterness, 1),
            "sweetness": round(sweetness, 1),
            "carbonation": round(carbonation, 1),
            "strength": round(strength, 2),
            "seasonality": np.random.choice(["any", "summer", "fall"]),
            "origin": np.random.choice(ORIGINS),
        })
        drink_id += 1

    # Generate non-alcoholic catalog (10 drinks)
    for i in range(10):
        style = NON_ALCOHOLIC_STYLES[i % len(NON_ALCOHOLIC_STYLES)]
        abv = 0.0
        bitterness = np.random.uniform(0, 30)
        sweetness = np.random.uniform(10, 90)
        carbonation = np.random.uniform(1, 4) if "sparkling" in style or "kombucha" in style else np.random.uniform(0, 1)
        strength = np.random.uniform(0.1, 0.5)

        drinks.append({
            "drink_id": f"D{drink_id:03d}",
            "name": f"{style.replace('_', ' ').capitalize()} {drink_id}",
            "type": "non-alcoholic",
            "style": style,
            "abv": abv,
            "bitterness": round(bitterness, 1),
            "sweetness": round(sweetness, 1),
            "carbonation": round(carbonation, 1),
            "strength": round(strength, 2),
            "seasonality": "any",
            "origin": np.random.choice(ORIGINS),
        })
        drink_id += 1

    df = pd.DataFrame(drinks)
    return df


def generate_context_scenarios(n_scenarios: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic context scenarios.

    Args:
        n_scenarios: Number of scenarios to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with context attributes
    """
    np.random.seed(seed)

    scenarios = []

    for i in range(n_scenarios):
        # Weather (temperature ranges for context)
        weather = np.random.choice(WEATHER_CONDITIONS)
        if weather == "sunny":
            temp = np.random.uniform(20, 35)
        elif weather == "rainy":
            temp = np.random.uniform(5, 18)
        elif weather == "cloudy":
            temp = np.random.uniform(10, 22)
        elif weather == "snowy":
            temp = np.random.uniform(-5, 5)
        else:  # stormy
            temp = np.random.uniform(10, 20)

        # Time of day
        hour = np.random.randint(6, 24)
        if hour < 12:
            time_period = "morning"
        elif hour < 18:
            time_period = "afternoon"
        else:
            time_period = "evening"

        # Day of week
        dayofweek = np.random.randint(0, 7)
        is_weekend = dayofweek >= 5

        # Occasion
        occasion = np.random.choice(OCCASIONS)

        # Context hash for grouping
        context_hash = f"{weather}_{temp:.0f}_{time_period}_{occasion}"

        scenarios.append({
            "scenario_id": f"S{i+1:03d}",
            "weather": weather,
            "temperature": round(temp, 1),
            "time_period": time_period,
            "hour": hour,
            "dayofweek": dayofweek,
            "is_weekend": is_weekend,
            "occasion": occasion,
            "context_hash": context_hash,
        })

    return pd.DataFrame(scenarios)


def generate_interaction_logs(
    n_interactions: int = 10000,
    n_users: int = 500,
    drinks_df: Optional[pd.DataFrame] = None,
    scenarios_df: Optional[pd.DataFrame] = None,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic user interaction logs.

    Args:
        n_interactions: Number of interactions to generate
        n_users: Number of unique users
        drinks_df: Optional drink catalog
        scenarios_df: Optional context scenarios
        seed: Random seed for reproducibility

    Returns:
        DataFrame with interaction records
    """
    np.random.seed(seed)

    # Load default data if not provided
    if drinks_df is None:
        drinks_df = generate_drink_catalog()
    if scenarios_df is None:
        scenarios_df = generate_context_scenarios()

    interactions = []
    user_ids = [f"U{u+1:03d}" for u in range(n_users)]

    for i in range(n_interactions):
        user_id = np.random.choice(user_ids)
        drink_id = np.random.choice(drinks_df["drink_id"].tolist())
        scenario = np.random.choice(scenarios_df.to_dict("records"))

        # Interaction type: mostly views, some ratings
        interaction_type = np.random.choice(
            ["view", "view", "view", "rate", "purchase"],
            p=[0.6, 0.2, 0.1, 0.08, 0.02]
        )

        # Rating value (if rated)
        if interaction_type == "rate":
            value = round(np.random.uniform(1, 5), 1)
        elif interaction_type == "purchase":
            value = 1
        else:
            value = 0  # view doesn't have meaningful value

        interactions.append({
            "user_id": user_id,
            "drink_id": drink_id,
            "weather": scenario["weather"],
            "temperature": scenario["temperature"],
            "time_period": scenario["time_period"],
            "hour": scenario["hour"],
            "dayofweek": scenario["dayofweek"],
            "occasion": scenario["occasion"],
            "context_hash": scenario["context_hash"],
            "interaction_type": interaction_type,
            "value": value,
        })

    return pd.DataFrame(interactions)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def load_drink_data(data_dir: str = "data/synthetic") -> pd.DataFrame:
    """Load drink catalog or generate synthetic data.

    Args:
        data_dir: Directory containing drink data

    Returns:
        DataFrame with drink catalog
    """
    filepath = Path(data_dir) / "drinks_catalog.csv"

    if filepath.exists():
        return pd.read_csv(filepath)

    # Generate synthetic data
    print("Generating synthetic drink catalog...")
    df = generate_drink_catalog()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return df


def load_interaction_data(data_dir: str = "data/synthetic") -> pd.DataFrame:
    """Load interaction logs or generate synthetic data.

    Args:
        data_dir: Directory containing interaction data

    Returns:
        DataFrame with interaction logs
    """
    filepath = Path(data_dir) / "interaction_logs.csv"

    if filepath.exists():
        return pd.read_csv(filepath)

    # Generate synthetic data
    print("Generating synthetic interaction logs...")
    drinks_df = load_drink_data(data_dir)
    scenarios_df = generate_context_scenarios(n_scenarios=50)
    df = generate_interaction_logs(
        n_interactions=10000,
        n_users=500,
        drinks_df=drinks_df,
        scenarios_df=scenarios_df
    )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return df


def load_context_scenarios(data_dir: str = "data/synthetic") -> pd.DataFrame:
    """Load context scenarios or generate synthetic data.

    Args:
        data_dir: Directory containing scenario data

    Returns:
        DataFrame with context scenarios
    """
    filepath = Path(data_dir) / "context_scenarios.csv"

    if filepath.exists():
        return pd.read_csv(filepath)

    # Generate synthetic data
    print("Generating synthetic context scenarios...")
    df = generate_context_scenarios()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return df


# ============================================================================
# DATA UTILITIES
# ============================================================================


def get_drink_by_id(drink_id: str, drinks_df: pd.DataFrame) -> Optional[Dict]:
    """Get drink by ID.

    Args:
        drink_id: Drink identifier
        drinks_df: Drink catalog

    Returns:
        Drink dictionary or None
    """
    match = drinks_df[drinks_df["drink_id"] == drink_id]
    if len(match) > 0:
        return match.iloc[0].to_dict()
    return None


def get_drink_stats(drinks_df: pd.DataFrame) -> Dict:
    """Get statistics about drink catalog.

    Args:
        drinks_df: Drink catalog

    Returns:
        Dictionary with catalog statistics
    """
    return {
        "total_drinks": len(drinks_df),
        "by_type": drinks_df["type"].value_counts().to_dict(),
        "by_style": drinks_df["style"].value_counts().to_dict(),
        "abv_range": (drinks_df["abv"].min(), drinks_df["abv"].max()),
        "avg_bitterness": drinks_df["bitterness"].mean(),
        "avg_sweetness": drinks_df["sweetness"].mean(),
    }


def get_context_stats(interactions_df: pd.DataFrame) -> Dict:
    """Get statistics about interaction contexts.

    Args:
        interactions_df: Interaction logs

    Returns:
        Dictionary with context statistics
    """
    return {
        "total_interactions": len(interactions_df),
        "unique_users": interactions_df["user_id"].nunique(),
        "unique_drinks": interactions_df["drink_id"].nunique(),
        "by_weather": interactions_df["weather"].value_counts().to_dict(),
        "by_time_period": interactions_df["time_period"].value_counts().to_dict(),
        "by_occasion": interactions_df["occasion"].value_counts().to_dict(),
        "interaction_types": interactions_df["interaction_type"].value_counts().to_dict(),
    }
