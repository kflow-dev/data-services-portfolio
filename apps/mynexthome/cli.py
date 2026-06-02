"""MyNextHome — Hybrid real estate recommender combining geo-distance and content similarity.

Uses a hybrid approach:
1. Geo-distance scoring (Haversine formula) for location proximity
2. Content-based similarity for property features
3. Multi-objective optimization for balancing price, features, and location

Usage:
    CLI:      python cli.py recommend "Lisbon, 500k, 2BR" --preference "neighborhood"
    Streamlit: streamlit run streamlit_app.py
    Notebook:  jupyter notebooks/property_recommender_example.ipynb
"""

import csv
import math
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="MyNextHome: Hybrid real estate recommendations with geo features.")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Property:
    """Represents a real estate property."""
    id: str
    address: str
    city: str
    neighborhood: str
    latitude: float
    longitude: float
    price: int
    bedrooms: int
    bathrooms: float
    sqft: int
    property_type: str  # apartment, house, condo, townhouse
    year_built: int
    features: List[str]
    listing_date: str
    description: str


@dataclass
class UserPreferences:
    """User's property preferences."""
    location: str
    max_budget: int
    min_bedrooms: int
    preferred_neighborhoods: List[str]
    desired_features: List[str]
    property_types: List[str]
    priority: str  # price, location, features, balanced


# ============================================================================
# GEOGRAPHICAL UTILITIES
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points (km)."""
    R = 6371  # Earth's radius in kilometers

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def get_city_coordinates(city: str) -> Tuple[float, float]:
    """Get center coordinates for a city."""
    city_coords = {
        "lisbon": (38.7223, -9.1393),
        "porto": (41.1579, -8.6291),
        "madrid": (40.4168, -3.7038),
        "barcelona": (41.3851, 2.1734),
        "berlin": (52.5200, 13.4050),
        "paris": (48.8566, 2.3522),
        "london": (51.5074, -0.1278),
        "amsterdam": (52.3676, 4.9041),
        "rome": (41.9028, 12.4964),
        "milan": (45.4642, 9.1900),
        "vienna": (48.2082, 16.3738),
        "munich": (48.1351, 11.5820),
        "zurich": (47.3769, 8.5417),
        "zurich": (47.3769, 8.5417),
    }
    return city_coords.get(city.lower(), (48.8566, 2.3522))  # Default to Paris


# ============================================================================
# FEATURE MAPPINGS
# ============================================================================

PROPERTY_TYPE_MAP = {
    "apartment": 0, "house": 1, "condo": 2, "townhouse": 3, "loft": 4, "studio": 5
}

FEATURE_MAP = {
    "balcony": 0, "garage": 1, "pool": 2, "garden": 3,
    "elevator": 4, "parking": 5, "air_conditioning": 6,
    "heating": 7, "dishwasher": 8, "fireplace": 9,
    "gym": 10, "concierge": 11, "renovated": 12, "pet_friendly": 13
}

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def encode_property(prop: Dict) -> np.ndarray:
    """Encode property to feature vector."""
    # Normalize features
    type_idx = PROPERTY_TYPE_MAP.get(prop.get("property_type", "apartment"), 0)
    bedrooms = min(prop.get("bedrooms", 2), 10)  # Cap at 10
    bathrooms = min(prop.get("bathrooms", 1), 5)
    sqft_normalized = min(prop.get("sqft", 800) / 5000, 1.0)
    age_normalized = max(0, 1 - (2024 - prop.get("year_built", 2000)) / 100)
    price_normalized = min(prop.get("price", 300000) / 1000000, 1.0)

    # Feature vector: type(1) + bedrooms(1) + bathrooms(1) + sqft(1) + age(1) + price(1) = 6 dims
    features = np.array([
        type_idx / 5.0,
        bedrooms / 10.0,
        bathrooms / 5.0,
        sqft_normalized,
        age_normalized,
        1.0 - price_normalized,  # Invert: lower price is better
    ], dtype=float)

    return features


def extract_user_preferences(context: str) -> UserPreferences:
    """Parse user context string into preferences object."""
    context_lower = context.lower()

    # Extract location (simplified)
    cities = ["lisbon", "porto", "madrid", "barcelona", "berlin", "paris", "london"]
    location = "lisbon"
    for city in cities:
        if city in context_lower:
            location = city
            break

    # Extract budget
    price_match = re.search(r"(\d{2,4}(?:k)?)", context_lower)
    max_budget = 500000
    if price_match:
        price_str = price_match.group(1)
        if "k" in price_str:
            max_budget = int(price_str.replace("k", "")) * 1000
        else:
            max_budget = int(price_str)

    # Extract bedrooms
    beds_match = re.search(r"(\d+)\s*(?:bed|br)", context_lower)
    min_bedrooms = 2
    if beds_match:
        min_bedrooms = int(beds_match.group(1))

    # Extract priority
    priority = "balanced"
    if "price" in context_lower or "budget" in context_lower:
        priority = "price"
    elif "neighborhood" in context_lower or "location" in context_lower:
        priority = "location"
    elif "features" in context_lower:
        priority = "features"

    return UserPreferences(
        location=location,
        max_budget=max_budget,
        min_bedrooms=min_bedrooms,
        preferred_neighborhoods=[],
        desired_features=[],
        property_types=[],
        priority=priority,
    )


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_synthetic_properties(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic real estate listings."""
    np.random.seed(seed)

    cities = ["lisbon", "porto", "madrid", "barcelona"]
    city_coords = {
        "lisbon": (38.7223, -9.1393),
        "porto": (41.1579, -8.6291),
        "madrid": (40.4168, -3.7038),
        "barcelona": (41.3851, 2.1734),
    }

    neighborhoods = {
        "lisbon": ["alfama", "bairro Alto", "chiado", "prado", "graca", "santa_justa"],
        "porto": ["cedofeita", "frente_river", "vitoria", "massarelos", "bonfim"],
        "madrid": ["malasaña", "chueca", "salamanca", "retiro", "lavapiés"],
        "barcelona": ["gracia", "gothic_quarter", "eixample", "barceloneta", "sent_martí"],
    }

    property_types = ["apartment", "house", "condo", "townhouse"]
    feature_options = ["balcony", "garage", "pool", "garden", "elevator", "parking",
                       "air_conditioning", "heating", "renovated", "pet_friendly"]

    descriptions = [
        "Charming property in historic neighborhood",
        "Modern apartment with city views",
        "Spacious family home with garden",
        "Luxury condo in prime location",
        "Cozy studio perfect for professionals",
        "Renovated home with contemporary finishes",
    ]

    properties = []
    for i in range(n):
        city = np.random.choice(cities)
        coords = city_coords[city]
        neighborhood = np.random.choice(neighborhoods[city])

        # Generate price based on size and city
        base_price = {"lisbon": 3000, "porto": 2500, "madrid": 3500, "barcelona": 4000}[city]
        sqft = int(np.random.uniform(400, 2500))
        price = int(sqft * base_price * np.random.uniform(0.8, 1.5))

        # Add noise around city center
        lat = coords[0] + np.random.normal(0, 0.02)
        lon = coords[1] + np.random.normal(0, 0.02)

        # Generate features (0-4 per property)
        n_features = np.random.randint(1, 5)
        features = np.random.choice(feature_options, size=n_features, replace=False).tolist()

        properties.append({
            "id": f"p{i+1:03d}",
            "address": f"{np.random.randint(1, 999)} {['Main', 'Oak', 'Pine', 'River', 'Hill', 'Central'][np.random.randint(6)]} St",
            "city": city.title(),
            "neighborhood": neighborhood.replace("_", " ").title(),
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "price": price,
            "bedrooms": np.random.choice([1, 2, 3, 4, 5]),
            "bathrooms": round(np.random.uniform(1, 3), 1),
            "sqft": sqft,
            "property_type": np.random.choice(property_types),
            "year_built": np.random.randint(1950, 2024),
            "features": features,
            "listing_date": f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
            "description": np.random.choice(descriptions),
        })

    return pd.DataFrame(properties)


# ============================================================================
# HYBRID RECOMMENDATION ENGINE
# ============================================================================

class HybridPropertyRecommender:
    """Hybrid real estate recommender combining geo and content features."""

    def __init__(self):
        self.properties: List[Property] = []
        self.property_df: pd.DataFrame = None
        self.city_center: Tuple[float, float] = None

    def load_properties(self, properties_df: pd.DataFrame):
        """Load property catalog."""
        self.property_df = properties_df
        self.properties = [Property(**row.to_dict()) for _, row in properties_df.iterrows()]

        # Get city center from first property
        if len(self.properties) > 0:
            self.city_center = (
                self.properties[0].latitude,
                self.properties[0].longitude
            )

    def score_property_geo(
        self,
        prop: Property,
        user_lat: float,
        user_lon: float,
    ) -> float:
        """Score property based on geo proximity (0-1, higher is closer)."""
        if not self.city_center:
            return 0.5

        distance = haversine_distance(
            user_lat, user_lon,
            prop.latitude, prop.longitude
        )

        # Sigmoid decay: closer properties get higher scores
        # Distance in km, max meaningful distance ~50km
        return 1.0 / (1.0 + distance / 20.0)

    def score_property_content(self, prop: Property, prefs: UserPreferences) -> float:
        """Score property based on content match with preferences."""
        score = 0.0
        max_score = 0.0

        # Price match (must be within budget)
        max_score += 0.25
        if prop.price <= prefs.max_budget:
            # Bonus for being well under budget
            score += 0.25 * (1.0 - prop.price / prefs.max_budget)
        else:
            score += 0.05  # Small penalty for over budget

        # Bedroom match
        max_score += 0.25
        if prop.bedrooms >= prefs.min_bedrooms:
            score += 0.25
        elif prop.bedrooms >= prefs.min_bedrooms - 1:
            score += 0.1

        # Neighborhood match
        max_score += 0.25
        if prefs.preferred_neighborhoods:
            for pref_nh in prefs.preferred_neighborhoods:
                if pref_nh.lower() in prop.neighborhood.lower():
                    score += 0.25
                    break
            else:
                score += 0.05
        else:
            score += 0.15  # Neutral score if no preference

        # Feature match
        max_score += 0.25
        if prefs.desired_features:
            feature_match = sum(1 for f in prefs.desired_features if f in prop.features)
            feature_ratio = feature_match / len(prefs.desired_features)
            score += 0.25 * feature_ratio
        else:
            score += 0.1

        return score / max_score if max_score > 0 else 0.0

    def recommend(
        self,
        context: str,
        top_k: int = 10,
        geo_weight: float = 0.4,
        content_weight: float = 0.6,
    ) -> List[Tuple[Property, float, Dict]]:
        """Recommend properties based on context."""
        prefs = extract_user_preferences(context)
        user_coords = get_city_coordinates(prefs.location)

        scores = []
        for prop in self.properties:
            geo_score = self.score_property_geo(prop, user_coords[0], user_coords[1])
            content_score = self.score_property_content(prop, prefs)

            # Hybrid score
            if prefs.priority == "price":
                hybrid_score = content_score * 0.7 + geo_score * 0.3
            elif prefs.priority == "location":
                hybrid_score = geo_score * 0.7 + content_score * 0.3
            else:
                hybrid_score = geo_weight * geo_score + content_weight * content_score

            scores.append((prop, hybrid_score, {
                "geo_score": geo_score,
                "content_score": content_score,
                "priority_score": hybrid_score,
            }))

        # Sort by hybrid score
        scores.sort(key=lambda x: x[1][3], reverse=True)
        return scores[:top_k]


# ============================================================================
# PRICE FORECASTING (HMM-inspired)
# ============================================================================

def simple_price_forecast(
    zip_code: str,
    current_price_per_sqft: float,
    horizon_months: int = 12,
) -> List[Dict]:
    """Simple price forecast using random walk with drift."""
    np.random.seed(42)

    # Base growth rate (2% annually)
    monthly_rate = 0.02 / 12

    forecast = []
    current_price = current_price_per_sqft

    for month in range(1, horizon_months + 1, 3):  # Every 3 months
        # Random walk with drift
        shock = np.random.normal(0, 0.01)
        current_price *= (1 + monthly_rate + shock)

        forecast.append({
            "month": month,
            "price_per_sqft": round(current_price, 1),
            "change_percent": round((current_price / current_price_per_sqft - 1) * 100, 1),
        })

    return forecast


# ============================================================================
# CLI COMMANDS
# ============================================================================

@app.command()
def recommend(
    context: str = typer.Argument(
        ...,
        help="Search criteria (e.g., 'Lisbon, 500k, 2BR, near park')"
    ),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of recommendations"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """Recommend properties based on context."""
    properties_path = Path(data_dir) / "properties.csv"
    if properties_path.exists():
        properties_df = pd.read_csv(properties_path)
    else:
        properties_df = generate_synthetic_properties(200)

    typer.echo(f"Loaded {len(properties_df)} properties")
    typer.echo(f"\nSearch: '{context}'")

    recommender = HybridPropertyRecommender()
    recommender.load_properties(properties_df)

    recs = recommender.recommend(context, top_k)

    typer.echo(f"\nTop {top_k} recommendations:\n")
    for i, (prop, score, metrics) in enumerate(recs, 1):
        typer.echo(f"{i}. {prop.address}")
        typer.echo(f"   {prop.neighborhood}, {prop.city}")
        typer.echo(f"   {prop.bedrooms}BR / {prop.bathrooms}BA | {prop.sqft} sqft | {prop.property_type}")
        typer.echo(f"   Price: ${prop.price:,} (${prop.price / prop.sqft:.0f}/sqft)")
        typer.echo(f"   Built: {prop.year_built}")
        typer.echo(f"   Features: {', '.join(prop.features)}")
        typer.echo(f"   Geo score: {metrics['geo_score']:.2f} | Content score: {metrics['content_score']:.2f}")
        if prop.description:
            typer.echo(f"   {prop.description}")
        typer.echo()


@app.command()
def generate_data(
    output_dir: str = typer.Option("data/synthetic", "--output-dir", "-o", help="Output directory"),
    n_properties: int = typer.Option(200, "--count", "-n", help="Number of properties"),
):
    """Generate synthetic property data."""
    properties_df = generate_synthetic_properties(n_properties)
    filepath = Path(output_dir) / "properties.csv"
    properties_df.to_csv(filepath, index=False)
    typer.echo(f"Generated {n_properties} properties to: {filepath}")


@app.command()
def forecast(
    location: str = typer.Argument(..., help="Location (city)"),
    current_price: float = typer.Argument(..., help="Current price per sqft"),
    months: int = typer.Option(12, "--horizon", "-m", help="Forecast horizon in months"),
):
    """Price forecast for a location."""
    typer.echo(f"Price forecast for {location}:")
    typer.echo(f"Current: ${current_price:.0f}/sqft")
    typer.echo(f"Horizon: {months} months\n")

    forecast = simple_price_forecast("00000", current_price, months)
    for f in forecast:
        typer.echo(f"  Month {f['month']:2d}: ${f['price_per_sqft']:.0f}/sqft ({f['change_percent']:+.1f}%)")


@app.command()
def list_properties(
    city_filter: str = typer.Option("", "--city", help="Filter by city"),
    max_price: int = typer.Option(0, "--max-price", help="Maximum price"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """List available properties."""
    properties_path = Path(data_dir) / "properties.csv"
    if properties_path.exists():
        properties_df = pd.read_csv(properties_path)
    else:
        properties_df = generate_synthetic_properties(200)

    if city_filter:
        properties_df = properties_df[properties_df["city"].str.contains(city_filter, case=False)]
    if max_price > 0:
        properties_df = properties_df[properties_df["price"] <= max_price]

    typer.echo(f"Properties ({len(properties_df)} total):\n")
    for _, prop in properties_df.iterrows():
        typer.echo(f"  {prop['id']}: {prop['address']}")
        typer.echo(f"      {prop['city']}, {prop['neighborhood']}")
        typer.echo(f"      ${prop['price']:,} | {prop['bedrooms']}BR/{prop['bathrooms']}BA | {prop['sqft']} sqft")


if __name__ == "__main__":
    import re
    app()
