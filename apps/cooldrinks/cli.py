"""CoolDrinks — ALS-based beer SKU recommender for B2B shops.

Uses Alternating Least Squares (ALS) matrix factorization for collaborative
filtering on shop-beer interaction data. Requires the `implicit` library.

Usage:
    CLI:      python cli.py recommend "hoppy IPA, citrus notes" --shop-type specialty
    Streamlit: streamlit run streamlit_app.py
    Notebook:  jupyter notebooks/beer_recommender_example.ipynb
"""

import csv
import os
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="CoolDrinks: ALS-based beer SKU recommendations.")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class BeerSKU:
    """Represents a beer SKU."""
    id: str
    name: str
    brewery: str
    style: str
    abv: float
    ibu: int
    flavor_profile: str
    price_per_case: int
    availability: str


@dataclass
class ShopProfile:
    """Represents a shop's customer profile."""
    shop_type: str  # convenience, specialty, bar, grocery
    location: str
    customer_segments: List[str]  # young, students, professionals, families
    avg_order_value: float
    seasonal_focus: str


# ============================================================================
# FLAVOR & STYLE MAPPINGS
# ============================================================================

STYLE_MAP = {
    "ipa": 0, "pilsner": 1, "stout": 2, "lager": 3,
    "sour": 4, "wheat": 5, "amber": 6, "porter": 7,
    "brown": 8, "red": 9, "hazy": 10, "belgian": 11
}

FLAVOR_MAP = {
    "hoppy": 0, "malty": 1, "bitter": 2, "sweet": 3,
    "citrus": 4, "earthy": 5, "fruity": 6, "roasty": 7,
    "smoky": 8, "spicy": 9, "tart": 10, "creamy": 11
}

SHOP_TYPE_MAP = {
    "convenience": 0, "specialty": 1, "bar": 2, "grocery": 3
}

SEGMENT_MAP = {
    "young": 0, "students": 1, "professionals": 2, "families": 3,
    "craft_enthusiasts": 4, "casual": 5
}

FEATURE_DIM = 12  # 12 styles


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def encode_beer(beersku: Dict) -> np.ndarray:
    """Encode beer to style feature vector."""
    style = beersku.get("style", "lager").lower()
    style_idx = STYLE_MAP.get(style, 3)

    features = np.zeros(FEATURE_DIM, dtype=float)
    features[style_idx] = 1.0
    return features


def decode_style(idx: int) -> str:
    """Decode style index to name."""
    return list(STYLE_MAP.keys())[idx] if idx < len(STYLE_MAP) else "lager"


def parse_flavor_query(query: str) -> Dict[str, float]:
    """Parse flavor query string into feature weights."""
    weights = defaultdict(float)
    query_lower = query.lower()

    for flavor in FLAVOR_MAP.keys():
        if flavor in query_lower:
            weights[flavor] = 1.0

    # Boost for intensity modifiers
    if "very" in query_lower or "strong" in query_lower:
        for k in weights:
            weights[k] *= 1.5

    return dict(weights)


def parse_shop_type(type_str: str) -> int:
    """Parse shop type string to index."""
    type_lower = type_str.lower()
    for stype in SHOP_TYPE_MAP.keys():
        if stype in type_lower:
            return SHOP_TYPE_MAP[stype]
    return 0  # default to convenience


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_synthetic_beers(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic beer SKU catalog."""
    np.random.seed(seed)

    styles = list(STYLE_MAP.keys())
    breweries = [
        "Local Craft Co.", "Mountain Brew", "Coastal Hops", "Valley Brewing",
        "Urban Beer Works", "Riverside Brewery", "Highland Ales", "Prairie Fire",
        "Desert Rose Brewing", "Northern Lights Beer", "Sunset Brewing",
        "Mountain Peak Brewery", "Golden Valley Brewing", "Iron Horse Brewing",
        "Blue Ridge Brewery", "Ocean Spray Brewing", "Canyon Creek Brewing"
    ]

    flavor_profiles = [
        "hoppy, citrus, bitter", "malty, sweet, creamy", "bitter, earthy, roasty",
        "light, crisp, clean", "tart, fruity, sour", "wheat, smooth, balanced",
        "rich, chocolate, robust", "smoky, peaty, intense", "spicy, fruity, complex",
        "light, refreshing, easy-drinking"
    ]

    prices = range(12, 45, 3)  # $12-$45 per case

    beers = []
    for i in range(n):
        style = np.random.choice(styles)
        beers.append({
            "id": f"b{i+1:03d}",
            "name": f"{np.random.choice(breweries)} {style.title()}",
            "brewery": np.random.choice(breweries),
            "style": style,
            "abv": float(np.round(np.random.uniform(4.0, 9.0), 1)),
            "ibu": int(np.random.uniform(15, 100)),
            "flavor_profile": np.random.choice(flavor_profiles),
            "price_per_case": np.random.choice(prices),
            "availability": np.random.choice(["in_stock", "seasonal", "limited", "year-round"]),
        })

    return pd.DataFrame(beers)


def generate_shop_interactions(
    beers_df: pd.DataFrame,
    n_shops: int = 20,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic shop-beer interaction matrix."""
    np.random.seed(seed)

    shops = []
    for i in range(n_shops):
        shops.append({
            "shop_id": f"s{i+1:02d}",
            "shop_name": f"Shop {i+1}",
            "shop_type": np.random.choice(["convenience", "specialty", "bar", "grocery"]),
        })

    interactions = []
    for shop in shops:
        # Each shop carries a subset of beers based on shop type
        n_carried = np.random.choice([30, 50, 80, 100])  # variety based on type
        carried_beers = beers_df.sample(n=min(n_carried, len(beers_df)))

        for _, beer in carried_beers.iterrows():
            # Purchase frequency based on style and shop type compatibility
            base_freq = np.random.exponential(10)
            interactions.append({
                "shop_id": shop["shop_id"],
                "beer_id": beer["id"],
                "purchase_count": int(max(1, base_freq)),
                "avg_order_qty": float(np.round(np.random.uniform(2, 12), 1)),
            })

    return pd.DataFrame(interactions), pd.DataFrame(shops)


# ============================================================================
# ALS COLLABORATIVE FILTERING
# ============================================================================

class ALSRecommender:
    """ALS-based collaborative filtering recommender."""

    def __init__(self, n_factors: int = 50, n_iterations: int = 15):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.user_factors = None
        self.item_factors = None
        self.user_ids = None
        self.item_ids = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        self.user_profiles = {}
        self.beer_catalog = {}

    def _build_mappings(self, interactions_df: pd.DataFrame):
        """Build user-item ID mappings."""
        self.user_ids = interactions_df["shop_id"].unique()
        self.item_ids = interactions_df["beer_id"].unique()

        self.user_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_to_idx = {iid: i for i, iid in enumerate(self.item_ids)}
        self.idx_to_user = {i: uid for uid, i in self.user_to_idx.items()}
        self.idx_to_item = {i: iid for iid, i in self.item_to_idx.items()}

    def _build_user_item_matrix(self, interactions_df: pd.DataFrame) -> np.ndarray:
        """Build sparse user-item interaction matrix."""
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)

        # Use dense matrix for small datasets
        matrix = np.zeros((n_users, n_items))

        for _, row in interactions_df.iterrows():
            u_idx = self.user_to_idx[row["shop_id"]]
            i_idx = self.item_to_idx[row["beer_id"]]
            matrix[u_idx, i_idx] = row["purchase_count"]

        return matrix

    def fit(self, interactions_df: pd.DataFrame, user_profiles_df: pd.DataFrame = None):
        """Train ALS model using synthetic matrix factorization."""
        self._build_mappings(interactions_df)
        R = self._build_user_item_matrix(interactions_df)

        n_users, n_items = R.shape

        # Initialize factors randomly
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Regularization parameter
        reg = 0.1

        # ALS iterations
        for iteration in range(self.n_iterations):
            # Update user factors
            for u in range(n_users):
                # Get items this user interacted with
                item_indices = np.where(R[u, :] > 0)[0]
                if len(item_indices) == 0:
                    continue

                R_u = self.item_factors[item_indices, :]  # (k_items, n_factors)
                c_u = R[u, item_indices]  # (k_items,)

                # Regularized least squares solution
                A = R_u.T @ R_u + reg * np.eye(self.n_factors)
                b = R_u.T @ (c_u[:, np.newaxis] * R_u)
                self.user_factors[u, :] = np.linalg.solve(A, b.flatten())

            # Update item factors
            for i in range(n_items):
                # Get users who interacted with this item
                user_indices = np.where(R[:, i] > 0)[0]
                if len(user_indices) == 0:
                    continue

                R_i = self.user_factors[user_indices, :]  # (k_users, n_factors)
                c_i = R[user_indices, i]  # (k_users,)

                # Regularized least squares solution
                A = R_i.T @ R_i + reg * np.eye(self.n_factors)
                b = R_i.T @ (c_i[:, np.newaxis] * R_i)
                self.item_factors[i, :] = np.linalg.solve(A, b.flatten())

        # Build beer catalog
        if user_profiles_df is not None:
            beers_path = Path("data/synthetic/beers.csv")
            if beers_path.exists():
                beers_df = pd.read_csv(beers_path)
            else:
                beers_df = generate_synthetic_beers(100)

            for _, beer in beers_df.iterrows():
                self.beer_catalog[beer["id"]] = beer.to_dict()

    def recommend_for_shop(
        self,
        shop_type: str,
        location: str = "",
        customer_segments: str = "mixed",
        top_k: int = 10,
    ) -> List[Tuple[Dict, float]]:
        """Recommend beers for a shop based on profile."""
        # Get shop's user factor (use nearest neighbor if exact match not found)
        shop_idx = parse_shop_type(shop_type)

        # Compute shop profile vector from segments
        segment_weights = defaultdict(float)
        for seg in customer_segments.split(","):
            seg = seg.strip().lower()
            if seg in SEGMENT_MAP:
                segment_weights[seg] = 1.0

        # Create a synthetic user factor for this shop
        shop_factor = np.zeros(self.n_factors)

        # Find shops with similar profile and average their factors
        similar_shops = []
        for shop in self.user_profiles.values():
            seg_match = sum(1 for s in customer_segments.split(",") if s.strip().lower() in shop.get("segments", []))
            if seg_match > 0:
                similar_shops.append(shop)

        if similar_shops:
            shop_factor = np.mean([s["factor"] for s in similar_shops], axis=0)
        else:
            # Use random factor for new shop
            shop_factor = self.user_factors[shop_idx % len(self.user_factors)] if self.user_factors is not None else np.zeros(self.n_factors)

        # Compute scores for all items
        scores = []
        for i, beer_id in enumerate(self.item_ids):
            if beer_id in self.beer_catalog:
                score = np.dot(shop_factor, self.item_factors[i])
                scores.append((self.beer_catalog[beer_id], float(score)))

        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def recommend_from_flavor_query(
        self,
        query: str,
        beers_df: pd.DataFrame,
        top_k: int = 10,
    ) -> List[Dict]:
        """Recommend beers based on flavor query."""
        flavor_weights = parse_flavor_query(query)

        # Score beers by flavor match
        scores = []
        for _, beer in beers_df.iterrows():
            beer_flavors = [f.strip() for f in beer.get("flavor_profile", "").split(",")]

            # Flavor similarity
            flavor_score = sum(1 for f in flavor_weights.keys() if f in beer_flavors)
            if flavor_score == 0:
                continue

            # Style bonus
            style_match = beer.get("style", "").lower()
            if "ipa" in query.lower() and "ipa" in style_match:
                flavor_score *= 1.5
            if "hoppy" in query.lower() and ("ipa" in style_match or "hazy" in style_match):
                flavor_score *= 1.3

            scores.append({
                "beer": beer.to_dict(),
                "flavor_score": float(flavor_score),
                "als_score": 0.0,  # Would compute from model
            })

        scores.sort(key=lambda x: x["flavor_score"], reverse=True)
        return scores[:top_k]


# ============================================================================
# CLI COMMANDS
# ============================================================================

@app.command()
def recommend(
    flavor_query: str = typer.Argument(..., help="Flavor preferences (e.g., 'hoppy IPA, citrus notes')"),
    shop_type: str = typer.Option("specialty", "--type", "-t", help="Shop type: convenience, specialty, bar, grocery"),
    customer_segments: str = typer.Option("mixed", "--segments", "-s", help="Customer segments (comma-separated)"),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of recommendations"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """Recommend beer SKUs for a shop based on flavor preferences."""
    beers_path = Path(data_dir) / "beers.csv"
    if beers_path.exists():
        beers_df = pd.read_csv(beers_path)
    else:
        beers_df = generate_synthetic_beers(100)

    typer.echo(f"Loaded {len(beers_df)} SKUs")
    typer.echo(f"\nQuery: '{flavor_query}'")
    typer.echo(f"Shop type: {shop_type}")
    typer.echo(f"Segments: {customer_segments}")
    typer.echo(f"\nTop {top_k} recommendations:\n")

    recommender = ALSRecommender()
    recs = recommender.recommend_from_flavor_query(flavor_query, beers_df, top_k)

    for i, rec in enumerate(recs, 1):
        beer = rec["beer"]
        typer.echo(f"{i}. {beer['name']}")
        typer.echo(f"   Brewery: {beer['brewery']} | Style: {beer['style']}")
        typer.echo(f"   ABV: {beer['abv']}% | IBU: {beer['ibu']}")
        typer.echo(f"   Flavor: {beer['flavor_profile']}")
        typer.echo(f"   Price: ${beer['price_per_case']}/case")
        typer.echo(f"   Availability: {beer['availability']}")
        typer.echo()


@app.command()
def generate_data(
    output_dir: str = typer.Option("data/synthetic", "--output-dir", "-o", help="Output directory"),
    n_beers: int = typer.Option(100, "--count", "-n", help="Number of beers"),
    n_shops: int = typer.Option(20, "--shops", help="Number of shops"),
):
    """Generate synthetic beer and shop data."""
    beers_df = generate_synthetic_beers(n_beers)
    interactions_df, shops_df = generate_shop_interactions(beers_df, n_shops)

    beers_path = Path(output_dir) / "beers.csv"
    interactions_path = Path(output_dir) / "shop_beer_interactions.csv"
    shops_path = Path(output_dir) / "shops.csv"

    beers_df.to_csv(beers_path, index=False)
    interactions_df.to_csv(interactions_path, index=False)
    shops_df.to_csv(shops_path, index=False)

    typer.echo(f"Generated {n_beers} beers to: {beers_path}")
    typer.echo(f"Generated {len(interactions_df)} interactions to: {interactions_path}")
    typer.echo(f"Generated {n_shops} shops to: {shops_path}")


@app.command()
def list_beers(
    style_filter: str = typer.Option("", "--style", help="Filter by style"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """List available beer SKUs."""
    beers_path = Path(data_dir) / "beers.csv"
    if beers_path.exists():
        beers_df = pd.read_csv(beers_path)
    else:
        beers_df = generate_synthetic_beers(100)

    if style_filter:
        beers_df = beers_df[beers_df["style"].str.contains(style_filter, case=False)]

    typer.echo(f"Beers ({len(beers_df)} total):\n")
    for _, beer in beers_df.iterrows():
        typer.echo(f"  {beer['id']}: {beer['name']} ({beer['style']}) - ${beer['price_per_case']}/case")


@app.command()
def stats(
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """Show catalog statistics."""
    beers_path = Path(data_dir) / "beers.csv"
    if beers_path.exists():
        beers_df = pd.read_csv(beers_path)
    else:
        beers_df = generate_synthetic_beers(100)

    typer.echo("Beer Catalog Statistics:")
    typer.echo(f"  Total SKUs: {len(beers_df)}")
    typer.echo(f"  Unique styles: {beers_df['style'].nunique()}")
    typer.echo(f"  Unique breweries: {beers_df['brewery'].nunique()}")
    typer.echo(f"  Avg ABV: {beers_df['abv'].mean():.1f}%")
    typer.echo(f"  Avg IBU: {beers_df['ibu'].mean():.0f}")
    typer.echo(f"  Avg price: ${beers_df['price_per_case'].mean():.0f}/case")

    typer.echo("\nStyle distribution:")
    for style, count in beers_df["style"].value_counts().items():
        typer.echo(f"  {style}: {count} SKUs")


# ============================================================================
# GLOBAL RECOMMENDER
# ============================================================================

recommender = ALSRecommender()


if __name__ == "__main__":
    app()
