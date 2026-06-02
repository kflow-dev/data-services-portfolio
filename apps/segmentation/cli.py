"""Customer Segmentation — Persona creation with KMeans clustering.

MLOps template:
- Uses synthetic data for demonstration
- Implements KMeans clustering with silhouette validation
- Provides persona descriptions and representative customers
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


app = typer.Typer(help="Segmentation CLI: create customer personas via clustering.")


def load_or_generate_data(data_dir: str = "data/synthetic") -> pd.DataFrame:
    """Load customer data or generate synthetic data."""
    filepath = Path(data_dir) / "customer_segmentation.csv"

    if filepath.exists():
        return pd.read_csv(filepath)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500

    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n_samples)],
        "age": np.random.normal(38, 12, n_samples).clip(18, 70).astype(int),
        "income": np.random.lognormal(11, 0.5, n_samples).clip(20000, 200000).astype(int),
        "spending_score": np.random.beta(2, 2, n_samples) * 100,
        "visit_frequency": np.random.poisson(10, n_samples).clip(1, 30),
        "avg_order_value": np.random.lognormal(4.5, 0.8, n_samples).clip(10, 500),
        "tenure_months": np.random.exponential(30, n_samples).clip(1, 72).astype(int),
        "location_type": np.random.choice(["urban", "suburban", "rural"], n_samples),
        "engagement_score": np.random.uniform(0, 100, n_samples),
    })

    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return df


def create_personas(
    df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> dict:
    """Create customer personas using KMeans clustering.

    Args:
        df: Customer data
        n_clusters: Number of personas to create
        random_state: Random seed

    Returns:
        Dictionary with clusters, labels, and persona descriptions
    """
    # Select features for clustering
    feature_cols = ["age", "income", "spending_score", "visit_frequency",
                    "avg_order_value", "tenure_months", "engagement_score"]

    X = df[feature_cols].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Calculate silhouette score
    sil_score = silhouette_score(X_scaled, labels)

    # Create cluster descriptions
    clusters = {}
    for i in range(n_clusters):
        cluster_data = df[labels == i]
        cluster_data_scaled = X_scaled[labels == i]

        # Calculate cluster profile
        profile = {
            "size": len(cluster_data),
            "percentage": round(len(cluster_data) / len(df) * 100, 1),
            "avg_age": round(cluster_data["age"].mean(), 1),
            "avg_income": round(cluster_data["income"].mean(), 0),
            "avg_spending": round(cluster_data["spending_score"].mean(), 1),
            "avg_visits": round(cluster_data["visit_frequency"].mean(), 1),
            "avg_order_value": round(cluster_data["avg_order_value"].mean(), 1),
            "avg_tenure": round(cluster_data["tenure_months"].mean(), 1),
            "top_location": cluster_data["location_type"].mode().iloc[0] if len(cluster_data) > 0 else "unknown",
        }

        # Assign persona name
        if profile["avg_spending"] > 70 and profile["avg_income"] > 100000:
            persona_name = "Affluent Professionals"
        elif profile["avg_spending"] > 60 and profile["avg_age"] < 35:
            persona_name = "Young Enthusiasts"
        elif profile["avg_tenure"] > 40 and profile["avg_income"] > 90000:
            persona_name = "Loyal High-Value"
        elif profile["avg_spending"] < 40:
            persona_name = "Budget Conscious"
        elif profile["avg_visits"] > 12:
            persona_name = "Frequent Shoppers"
        else:
            persona_name = "Average Customers"

        profile["persona_name"] = persona_name

        clusters[i] = profile

    return {
        "labels": labels,
        "clusters": clusters,
        "silhouette_score": sil_score,
        "kmeans": kmeans,
        "scaler": scaler,
    }


def get_representative_customers(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_id: int,
    n_samples: int = 5,
) -> pd.DataFrame:
    """Get representative customers for a persona.

    Args:
        df: Customer data
        labels: Cluster labels
        cluster_id: Target cluster ID
        n_samples: Number of representative customers

    Returns:
        DataFrame with representative customers
    """
    cluster_data = df[labels == cluster_id].copy()

    if len(cluster_data) == 0:
        return pd.DataFrame()

    # Calculate distance from cluster centroid
    feature_cols = ["age", "income", "spending_score", "visit_frequency",
                    "avg_order_value", "tenure_months", "engagement_score"]
    X = cluster_data[feature_cols]
    X_scaled = StandardScaler().fit_transform(X)

    # Get cluster centroid
    centroid = np.mean(X_scaled, axis=0)

    # Calculate distances
    distances = np.sqrt(np.sum((X_scaled - centroid) ** 2, axis=1))

    # Get closest customers
    closest_idx = distances.argsort()[:n_samples]
    representatives = cluster_data.iloc[closest_idx]

    return representatives.reset_index(drop=True)


@app.command()
def create_personas_cmd(
    dataset: str = typer.Argument("data/synthetic", help="Dataset path or name"),
    n_personas: int = typer.Option(4, "--count", "-c", help="Number of personas to create"),
    random_state: int = typer.Option(42, "--seed", "-s", help="Random seed"),
):
    """Create customer personas using clustering."""
    typer.echo(f"Loading data from {dataset}...")

    df = load_or_generate_data(dataset)

    typer.echo(f"Creating {n_personas} personas from {len(df)} customers...")

    result = create_personas(df, n_clusters=n_personas, random_state=random_state)

    typer.echo("\n" + "="*60)
    typer.echo(f"Customer Segmentation: {n_personas} Personas")
    typer.echo(f"Silhouette Score: {result['silhouette_score']:.3f}")
    typer.echo("="*60)

    for cluster_id, profile in result["clusters"].items():
        typer.echo(f"\nPersona {cluster_id + 1}: {profile['persona_name']}")
        typer.echo(f"  Size: {profile['size']} customers ({profile['percentage']}%)")
        typer.echo(f"  Avg Age: {profile['avg_age']} years")
        typer.echo(f"  Avg Income: ${profile['avg_income']:,.0f}")
        typer.echo(f"  Avg Spending Score: {profile['avg_spending']:.1f}")
        typer.echo(f"  Avg Visits/Month: {profile['avg_visits']:.1f}")
        typer.echo(f"  Avg Order Value: ${profile['avg_order_value']:.2f}")
        typer.echo(f"  Avg Tenure: {profile['avg_tenure']:.1f} months")
        typer.echo(f"  Top Location: {profile['top_location']}")


@app.command()
def representative_customers(
    persona_id: int = typer.Argument(..., help="Persona ID (0-indexed)"),
    n_samples: int = typer.Option(5, "--count", "-c", help="Number of representative customers"),
    data_dir: str = typer.Option("data/synthetic", help="Data directory"),
):
    """Select representative customers for a persona."""
    df = load_or_generate_data(data_dir)
    result = create_personas(df, n_clusters=4)

    if persona_id not in result["clusters"]:
        typer.echo(f"Invalid persona ID. Valid IDs: {list(result['clusters'].keys())}")
        sys.exit(1)

    representatives = get_representative_customers(df, result["labels"], persona_id, n_samples)

    typer.echo(f"\nRepresentative Customers for Persona {persona_id + 1}:")
    typer.echo("="*60)

    for _, row in representatives.iterrows():
        typer.echo(f"\nCustomer {row['customer_id']}:")
        typer.echo(f"  Age: {row['age']}")
        typer.echo(f"  Income: ${row['income']:,.0f}")
        typer.echo(f"  Spending Score: {row['spending_score']}")
        typer.echo(f"  Visits/Month: {row['visit_frequency']}")
        typer.echo(f"  Avg Order: ${row['avg_order_value']:.2f}")
        typer.echo(f"  Tenure: {row['tenure_months']} months")
        typer.echo(f"  Location: {row['location_type']}")


@app.command()
def evaluate(
    data_dir: str = typer.Option("data/synthetic", help="Data directory"),
):
    """Evaluate clustering quality."""
    df = load_or_generate_data(data_dir)

    result = create_personas(df, n_clusters=4)

    typer.echo("\nClustering Evaluation:")
    typer.echo("="*40)
    typer.echo(f"Silhouette Score: {result['silhouette_score']:.3f}")
    typer.echo(f"  > 0.25: Weak structure")
    typer.echo(f"  > 0.5: Reasonable structure")
    typer.echo(f"  > 0.7: Strong structure")

    typer.echo("\nCluster Sizes:")
    for cluster_id, profile in result["clusters"].items():
        typer.echo(f"  Persona {cluster_id + 1} ({profile['persona_name']}): {profile['size']} ({profile['percentage']}%)")


if __name__ == "__main__":
    app()
