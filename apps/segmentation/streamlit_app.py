"""Customer Segmentation — Clustering UI."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

st.set_page_config(page_title="Customer Segmentation", layout="wide")

DEPARTMENTS = ["Electronics", "Apparel", "Home"]


def load_or_generate_data(data_dir: str = "data/synthetic") -> pd.DataFrame:
    """Load or generate customer data."""
    filepath = Path(data_dir) / "customer_segmentation.csv"

    if filepath.exists():
        return pd.read_csv(filepath)

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
    """Create personas using KMeans."""
    feature_cols = ["age", "income", "spending_score", "visit_frequency",
                    "avg_order_value", "tenure_months", "engagement_score"]

    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)

    clusters = {}
    for i in range(n_clusters):
        cluster_data = df[labels == i]

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
    }


def get_representative_customers(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_id: int,
    n_samples: int = 5,
) -> pd.DataFrame:
    """Get closest customers to centroid."""
    cluster_data = df[labels == cluster_id].copy()

    if len(cluster_data) == 0:
        return pd.DataFrame()

    feature_cols = ["age", "income", "spending_score", "visit_frequency",
                    "avg_order_value", "tenure_months", "engagement_score"]
    X = cluster_data[feature_cols]
    X_scaled = StandardScaler().fit_transform(X)

    centroid = np.mean(X_scaled, axis=0)
    distances = np.sqrt(np.sum((X_scaled - centroid) ** 2, axis=1))

    closest_idx = distances.argsort()[:n_samples]
    return cluster_data.iloc[closest_idx].reset_index(drop=True)


# Main app
st.title("Customer Segmentation")
st.caption("Create customer personas using KMeans clustering")

col1, col2 = st.columns(2)

with col1:
    n_personas = st.slider("Number of personas", 2, 10, 4)
with col2:
    random_state = st.number_input("Random seed", 42)

if st.button("Create Personas", type="primary"):
    with st.spinner("Running KMeans clustering..."):
        df = load_or_generate_data()
        result = create_personas(df, n_clusters=n_personas, random_state=random_state)

        st.success(f"Created {n_personas} personas!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Silhouette Score", f"{result['silhouette_score']:.3f}")
        with col2:
            st.metric("Total Customers", f"{len(df)}")

        st.subheader("Personas Overview")

        for cluster_id, profile in result["clusters"].items():
            with st.expander(f"Persona {cluster_id + 1}: {profile['persona_name']}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Size", f"{profile['size']} ({profile['percentage']}%)")
                col2.metric("Avg Income", f"${profile['avg_income']:,.0f}")
                col3.metric("Avg Spending", f"{profile['avg_spending']:.1f}")

                col1, col2 = st.columns(2)
                col1.write(f"**Avg Age:** {profile['avg_age']} years")
                col2.write(f"**Avg Visits:** {profile['avg_visits']}/month")
                col1.write(f"**Avg Order:** ${profile['avg_order_value']:.2f}")
                col2.write(f"**Avg Tenure:** {profile['avg_tenure']:.1f} months")
                st.write(f"**Top Location:** {profile['top_location']}")

        # Representative customers
        st.subheader("Representative Customers")
        persona_to_view = st.selectbox(
            "Select persona to view representatives",
            [f"{k+1}: {v['persona_name']}" for k, v in result["clusters"].items()]
        )
        selected_persona = int(persona_to_view.split(":")[0]) - 1

        if selected_persona in result["clusters"]:
            representatives = get_representative_customers(df, result["labels"], selected_persona, 5)
            if len(representatives) > 0:
                st.dataframe(representatives, use_container_width=True)
