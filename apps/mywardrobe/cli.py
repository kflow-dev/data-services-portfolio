"""MyWardrobe — Content-based outfit recommender using cosine similarity.

Uses feature vector encoding on style, color, season, and price attributes
to compute cosine similarity for personalized outfit recommendations.

Usage:
    CLI:      python cli.py recommend --context "smart-casual summer budget 150 EUR"
    Streamlit: streamlit run streamlit_app.py
    Notebook:  jupyter notebook notebooks/outfit_recommendation_example.ipynb
"""

import csv
import os
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="MyWardrobe: Content-based outfit recommendations using cosine similarity.")

# ============================================================================
# FEATURE MAPPINGS
# ============================================================================

STYLE_MAP = {
    "smart-casual": 0, "casual": 1, "formal": 2, "sporty": 3,
    "bohemian": 4, "edgy": 5, "minimalist": 6, "vintage": 7
}

COLOR_MAP = {
    "neutral": 0, "bright": 1, "pastel": 2, "dark": 3, "earth": 4
}

SEASON_MAP = {
    "spring": 0, "summer": 1, "fall": 2, "winter": 3
}

PRICE_BANDS = ["budget", "mid-range", "premium"]

FEATURE_DIM = 20  # 8 styles + 5 colors + 4 seasons + 3 price bands


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Outfit:
    """Represents a wardrobe outfit."""
    id: str
    title: str
    style: str
    color_family: str
    season: str
    price_band: str
    base_price: int
    description: str = ""


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def encode_outfit(outfit: Dict) -> np.ndarray:
    """Encode outfit to feature vector using one-hot encoding.

    Features: style(8) + color(5) + season(4) + price(3) = 20 dimensions
    """
    style_idx = STYLE_MAP.get(outfit.get("style", "casual"), 1)
    color_idx = COLOR_MAP.get(outfit.get("color_family", "neutral"), 0)
    season_idx = SEASON_MAP.get(outfit.get("season", "summer"), 1)
    price_idx = PRICE_BANDS.index(outfit.get("price_band", "mid-range")) if outfit.get("price_band") in PRICE_BANDS else 1

    features = np.zeros(FEATURE_DIM, dtype=float)
    features[style_idx] = 1.0
    features[8 + color_idx] = 1.0
    features[13 + season_idx] = 1.0
    features[17 + price_idx] = 1.0

    return features


def decode_vector(vec: np.ndarray) -> Dict:
    """Decode feature vector back to outfit attributes."""
    style_idx = np.argmax(vec[:8])
    color_idx = np.argmax(vec[8:13])
    season_idx = np.argmax(vec[13:17])
    price_idx = np.argmax(vec[17:20])

    style = list(STYLE_MAP.keys())[style_idx]
    color = list(COLOR_MAP.keys())[color_idx]
    season = list(SEASON_MAP.keys())[season_idx]
    price = PRICE_BANDS[price_idx]

    return {"style": style, "color_family": color, "season": season, "price_band": price}


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return float(dot_product / (norm1 * norm2 + 1e-8))


# ============================================================================
# DATA LOADING
# ============================================================================

def load_outfits(data_dir: str = "data/synthetic") -> pd.DataFrame:
    """Load outfit catalog from CSV or generate synthetic data."""
    filepath = Path(data_dir) / "outfits.csv"

    if filepath.exists():
        return pd.read_csv(filepath)

    return generate_synthetic_outfits(20)


def generate_synthetic_outfits(n: int = 20, seed: int = 42, data_dir: str = "data/synthetic") -> pd.DataFrame:
    """Generate synthetic outfit catalog for testing."""
    np.random.seed(seed)

    styles = list(STYLE_MAP.keys())
    colors = list(COLOR_MAP.keys())
    seasons = list(SEASON_MAP.keys())
    price_bands = PRICE_BANDS

    outfits = []
    descriptions = [
        "Perfect for everyday wear with modern appeal",
        "Elegant choice for special occasions",
        "Comfortable and stylish for any season",
        "Trendy outfit combining classic and contemporary elements",
        "Versatile piece that works for multiple occasions",
        "Seasonal favorite with timeless appeal",
        "Budget-friendly option without compromising style",
        "Premium quality materials and craftsmanship",
    ]

    for i in range(n):
        outfits.append({
            "id": f"w{i+1:02d}",
            "title": f"Outfit {i+1}",
            "style": np.random.choice(styles),
            "color_family": np.random.choice(colors),
            "season": np.random.choice(seasons),
            "price_band": np.random.choice(price_bands),
            "base_price": int(np.random.uniform(50, 350)),
            "description": np.random.choice(descriptions),
        })

    df = pd.DataFrame(outfits)
    filepath = Path(data_dir) / "outfits.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return df


# ============================================================================
# CONTENT-BASED RECOMMENDATION ENGINE
# ============================================================================

class ContentBasedRecommender:
    """Content-based outfit recommender using cosine similarity."""

    def __init__(self):
        self.outfits: List[Outfit] = []
        self.feature_matrix: np.ndarray = None
        self.outfit_ids: List[str] = []

    def add_outfit(self, outfit: Outfit) -> None:
        """Add outfit to catalog."""
        self.outfits.append(outfit)
        self.outfit_ids.append(outfit.id)

    def fit(self) -> None:
        """Compute feature vectors for all outfits."""
        self.feature_matrix = np.array(
            [encode_outfit(asdict(o)) for o in self.outfits],
            dtype=float
        )

    def recommend_from_context(
        self,
        context_style: str = "casual",
        context_season: str = "summer",
        context_budget: str = "mid-range",
        top_k: int = 5,
    ) -> List[Tuple[Outfit, float]]:
        """Recommend outfits based on context parameters."""
        context_vec = encode_outfit({
            "style": context_style,
            "season": context_season,
            "price_band": context_budget,
        })

        similarities = []
        for i, outfit in enumerate(self.outfits):
            sim = cosine_similarity(context_vec, self.feature_matrix[i])
            similarities.append((outfit, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def recommend_from_description(
        self,
        context: str,
        outfits_df: pd.DataFrame,
        top_k: int = 5,
    ) -> List[Dict]:
        """Recommend outfits based on free-text context."""
        context_lower = context.lower()

        # Parse style
        style = "casual"
        for s in STYLE_MAP.keys():
            if s in context_lower:
                style = s
                break

        # Parse season
        season = "summer"
        for se in SEASON_MAP.keys():
            if se in context_lower:
                season = se
                break

        # Parse budget
        price_band = "mid-range"
        if "budget" in context_lower or "<100" in context_lower or "cheap" in context_lower:
            price_band = "budget"
        elif "premium" in context_lower or ">250" in context_lower or "expensive" in context_lower:
            price_band = "premium"

        # Generate synthetic data if DataFrame empty
        if outfits_df.empty:
            outfits_df = generate_synthetic_outfits(20)

        # Score all outfits
        scores = []
        for _, row in outfits_df.iterrows():
            outfit_vec = encode_outfit(row.to_dict())
            context_vec = encode_outfit({
                "style": style,
                "season": season,
                "price_band": price_band,
            })
            similarity = cosine_similarity(context_vec, outfit_vec)
            scores.append({
                "outfit": row.to_dict(),
                "similarity": float(similarity),
            })

        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:top_k]


# ============================================================================
# CLI COMMANDS
# ============================================================================

@app.command()
def recommend(
    context: str = typer.Argument(
        ...,
        help="Context: style, season, budget (e.g., 'smart-casual summer budget 150 EUR')"
    ),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of recommendations"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
    style: str = typer.Option(None, "--style", help="Override style from context"),
    season: str = typer.Option(None, "--season", help="Override season from context"),
    budget: str = typer.Option(None, "--budget", help="Override budget from context"),
):
    """Recommend outfits based on context description or parameters."""
    outfits_df = load_outfits(data_dir)
    typer.echo(f"Loaded {len(outfits_df)} outfits from {data_dir}")

    # Use class-based recommender
    recs = recommender.recommend_from_description(context, outfits_df, top_k)

    typer.echo(f"\nTop {top_k} recommendations for: '{context}'\n")
    for i, rec in enumerate(recs, 1):
        o = rec["outfit"]
        typer.echo(f"{i}. {o['title']}")
        typer.echo(f"   Style: {o['style']}, Season: {o['season']}, Budget: {o['price_band']}")
        typer.echo(f"   Price: {o['base_price']} EUR")
        typer.echo(f"   Similarity: {rec['similarity']:.3f}")
        if o.get("description"):
            typer.echo(f"   {o['description']}")
        typer.echo()


@app.command()
def generate_data(
    output_dir: str = typer.Option("data/synthetic", "--output-dir", "-o", help="Output directory"),
    n_outfits: int = typer.Option(20, "--count", "-n", help="Number of outfits"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
):
    """Generate synthetic outfit data for testing."""
    outfits = generate_synthetic_outfits(n_outfits, seed)
    filepath = Path(output_dir) / "outfits.csv"
    outfits.to_csv(filepath, index=False)
    typer.echo(f"Generated {n_outfits} outfits to: {filepath}")


@app.command()
def list_outfits(
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """List all outfits in catalog."""
    outfits = load_outfits(data_dir)
    typer.echo(f"Outfit catalog ({len(outfits)} total):\n")
    for _, o in outfits.iterrows():
        typer.echo(f"  {o['id']}: {o['title']}")
        typer.echo(f"      {o['style']}, {o['season']}, {o['price_band']} - {o['base_price']} EUR")


@app.command()
def features(
    outfit_id: str = typer.Argument(..., help="Outfit ID to inspect"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """Show feature vector for a specific outfit."""
    outfits = load_outfits(data_dir)
    outfit = outfits[outfits["id"] == outfit_id]

    if outfit.empty:
        typer.echo(f"Outfit {outfit_id} not found!")
        return

    o = outfit.iloc[0].to_dict()
    vec = encode_outfit(o)

    typer.echo(f"Outfit: {o['title']}")
    typer.echo(f"  Style: {o['style']} -> index {np.argmax(vec[:8])}")
    typer.echo(f"  Color: {o['color_family']} -> index {np.argmax(vec[8:13])}")
    typer.echo(f"  Season: {o['season']} -> index {np.argmax(vec[13:17])}")
    typer.echo(f"  Price: {o['price_band']} -> index {np.argmax(vec[17:20])}")
    typer.echo(f"\nFeature vector ({len(vec)} dims):")
    typer.echo(f"  {vec}")


@app.command()
def similarity(
    outfit_id_1: str = typer.Argument(..., help="First outfit ID"),
    outfit_id_2: str = typer.Argument(..., help="Second outfit ID"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """Compute similarity between two outfits."""
    outfits = load_outfits(data_dir)

    o1 = outfits[outfits["id"] == outfit_id_1]
    o2 = outfits[outfits["id"] == outfit_id_2]

    if o1.empty or o2.empty:
        typer.echo("One or both outfits not found!")
        return

    vec1 = encode_outfit(o1.iloc[0].to_dict())
    vec2 = encode_outfit(o2.iloc[0].to_dict())
    sim = cosine_similarity(vec1, vec2)

    typer.echo(f"Similarity between '{o1.iloc[0]['title']}' and '{o2.iloc[0]['title']}': {sim:.3f}")


# ============================================================================
# GLOBAL RECOMMENDER INSTANCE
# ============================================================================

recommender = ContentBasedRecommender()


if __name__ == "__main__":
    app()
