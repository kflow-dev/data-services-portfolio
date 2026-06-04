"""Data loader module for loading and validating datasets."""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import numpy as np


class DataLoader:
    """Load and validate datasets from various sources."""

    def __init__(
        self,
        data_dir: str = "data/synthetic",
        random_state: int = 42,
    ):
        """Initialize DataLoader.

        Args:
            data_dir: Path to synthetic data directory
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self._validate_data_dir()

    def _validate_data_dir(self) -> None:
        """Validate that data directory exists."""
        if not self.data_dir.exists():
            warnings.warn(
                f"Data directory {self.data_dir} does not exist. "
                "Creating synthetic data on first load."
            )
            self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """Load a CSV file from the data directory.

        Args:
            filename: Name of the CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame loaded from CSV
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        return pd.read_csv(filepath, **kwargs)

    def load_customer_segmentation(self) -> pd.DataFrame:
        """Load customer segmentation dataset."""
        return self.load_csv("customer_segmentation.csv")

    def load_sku_demand(self) -> pd.DataFrame:
        """Load SKU demand forecasting dataset."""
        return self.load_csv("sku_demand.csv")

    def load_news_articles(self) -> pd.DataFrame:
        """Load news articles dataset."""
        return self.load_csv("news_articles.csv")

    def generate_synthetic_customer_data(
        self,
        n_samples: int = 1000,
        n_clusters: int = 4,
    ) -> pd.DataFrame:
        """Generate synthetic customer segmentation data.

        Args:
            n_samples: Number of samples to generate
            n_clusters: Number of customer segments

        Returns:
            DataFrame with synthetic customer data
        """
        np.random.seed(self.random_state)

        # Generate base features
        age = np.random.normal(38, 12, n_samples).clip(18, 70)
        income = np.random.lognormal(11, 0.5, n_samples).clip(20000, 200000)
        spending_score = np.random.beta(2, 2, n_samples) * 100
        visit_frequency = np.random.poisson(10, n_samples).clip(1, 30)
        avg_order_value = np.random.lognormal(4.5, 0.8, n_samples).clip(10, 500)
        tenure_months = np.random.exponential(30, n_samples).clip(1, 72)

        # Create clusters
        labels = np.random.randint(0, n_clusters, n_samples)
        location_type = np.where(
            labels < n_clusters // 2, "urban", "rural"
        )
        engagement_score = (
            spending_score * 0.6 + visit_frequency * 0.3 + tenure_months / 1.2
        ).clip(0, 100)

        df = pd.DataFrame({
            "customer_id": [f"C{i:04d}" for i in range(n_samples)],
            "age": age.round(0).astype(int),
            "income": income.round(0).astype(int),
            "spending_score": spending_score.round(0).astype(int),
            "visit_frequency": visit_frequency,
            "avg_order_value": avg_order_value.round(2),
            "tenure_months": tenure_months.round(0).astype(int),
            "location_type": location_type,
            "engagement_score": engagement_score.round(0).astype(int),
        })

        # Save to CSV
        filepath = self.data_dir / "customer_segmentation.csv"
        df.to_csv(filepath, index=False)

        return df

    def generate_synthetic_demand_data(
        self,
        n_weeks: int = 52,
        departments: Optional[list] = None,
    ) -> pd.DataFrame:
        """Generate synthetic demand forecasting data.

        Args:
            n_weeks: Number of weeks to generate
            departments: List of departments to include

        Returns:
            DataFrame with synthetic demand data
        """
        if departments is None:
            departments = ["Electronics", "Apparel", "Home"]

        np.random.seed(self.random_state)

        # Product hierarchy
        products = {
            "Electronics": [
                ("Laptops", ["LAP-001", "LAP-002"]),
                ("Tablets", ["TAB-001", "TAB-002"]),
            ],
            "Apparel": [
                ("T-Shirts", ["TSH-BLK-M", "TSH-WHT-L"]),
                ("Jeans", ["JNS-BLU-32", "JNS-BLK-34"]),
            ],
            "Home": [
                ("Kitchen Kits", ["KIT-001", "KIT-002"]),
            ],
        }

        rows = []
        base_price = {
            "LAP-001": 1299.99, "LAP-002": 899.99,
            "TAB-001": 599.99, "TAB-002": 449.99,
            "TSH-BLK-M": 29.99, "TSH-WHT-L": 29.99,
            "JNS-BLU-32": 79.99, "JNS-BLK-34": 79.99,
            "KIT-001": 149.99, "KIT-002": 199.99,
        }

        is_holiday_weeks = [7, 11, 24, 35, 47]

        for week in range(1, n_weeks + 1):
            # Base demand with trend and seasonality
            trend = 1 + 0.02 * week
            seasonality = 1 + 0.1 * np.sin(2 * np.pi * week / 52)
            noise = np.random.normal(1, 0.15)

            for department in departments:
                if department not in products:
                    continue
                for product_group, skus in products[department]:
                    for sku in skus:
                        base_demand = np.random.poisson(50)
                        quantity = int(base_demand * trend * seasonality * noise)

                        price = base_price.get(sku, 50.0)
                        discount = np.random.choice([0, 5, 10], p=[0.5, 0.3, 0.2])
                        revenue = quantity * price * (1 - discount / 100)

                        rows.append({
                            "week": week,
                            "department": department,
                            "product_group": product_group,
                            "sku": sku,
                            "quantity_sold": max(0, quantity),
                            "unit_price": price,
                            "discount_pct": discount,
                            "revenue": round(revenue, 2),
                            "inventory_level": int(np.random.exponential(100)),
                            "is_holiday": week in is_holiday_weeks,
                        })

        df = pd.DataFrame(rows)
        filepath = self.data_dir / "sku_demand.csv"
        df.to_csv(filepath, index=False)

        return df

    def generate_synthetic_news_data(
        self,
        n_articles: int = 100,
    ) -> pd.DataFrame:
        """Generate synthetic news articles data.

        Args:
            n_articles: Number of articles to generate

        Returns:
            DataFrame with synthetic news articles
        """
        np.random.seed(self.random_state)

        categories = ["tech", "finance", "ai", "sports", "policy", "climate"]
        sources = ["techcrunch", "reuters", "wired", "espn", "bloomberg", "bbc"]

        titles = [
            "New AI model achieves breakthrough on benchmark",
            "Market rally continues as inflation cools",
            "Tech company announces revolutionary product",
            "Sports team wins championship after decades",
            "Climate summit reaches historic agreement",
            "Policy changes impact digital economy",
        ]

        rows = []
        base_date = pd.Timestamp("2026-01-01")

        for i in range(n_articles):
            article_date = base_date + pd.Timedelta(days=i * 3)
            category = np.random.choice(categories)
            source = np.random.choice(sources)

            rows.append({
                "article_id": f"A{i:03d}",
                "title": np.random.choice(titles),
                "category": category,
                "published_date": article_date.strftime("%Y-%m-%d"),
                "sentiment_score": round(np.random.beta(2, 1), 2),
                "engagement_score": round(np.random.uniform(3, 10), 1),
                "source": source,
            })

        df = pd.DataFrame(rows)
        filepath = self.data_dir / "news_articles.csv"
        df.to_csv(filepath, index=False)

        return df

    def get_sample_data(self, dataset_type: str = "customer") -> pd.DataFrame:
        """Get sample data based on type.

        Args:
            dataset_type: Type of dataset ('customer', 'demand', 'news')

        Returns:
            Loaded or generated dataset
        """
        if dataset_type == "customer":
            try:
                return self.load_customer_segmentation()
            except FileNotFoundError:
                return self.generate_synthetic_customer_data()
        elif dataset_type == "demand":
            try:
                return self.load_sku_demand()
            except FileNotFoundError:
                return self.generate_synthetic_demand_data()
        elif dataset_type == "news":
            try:
                return self.load_news_articles()
            except FileNotFoundError:
                return self.generate_synthetic_news_data()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
