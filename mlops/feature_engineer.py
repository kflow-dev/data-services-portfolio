"""Feature engineering module for creating and transforming features."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEngineer:
    """Create and transform features for ML models."""

    def __init__(self):
        """Initialize feature engineer."""
        self._scaler: Optional[StandardScaler] = None
        self._encoders: Dict[str, OneHotEncoder] = {}

    def engineer_customer_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Engineer features for customer segmentation.

        Args:
            df: Customer data with columns: age, income, spending_score, etc.

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Age features
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 50, 100],
            labels=["young", "adult", "middle_aged", "senior"],
        )
        df["age_squared"] = df["age"] ** 2
        df["age_log"] = np.log1p(df["age"])

        # Income features
        df["income_log"] = np.log1p(df["income"])
        df["income_per_age"] = df["income"] / (df["age"] + 1)
        df["income_quantile"] = df["income"].rank(pct=True)

        # Spending features
        df["spending_intensity"] = df["spending_score"] * df["visit_frequency"] / 10
        df["value_per_visit"] = df["avg_order_value"] * df["visit_frequency"]
        df["spending_log"] = np.log1p(df["spending_score"])

        # Engagement score (if not present)
        if "engagement_score" not in df.columns:
            df["engagement_score"] = (
                df["spending_score"] * 0.4 +
                df["visit_frequency"] / 30 * 100 * 0.3 +
                df["tenure_months"] / 72 * 100 * 0.3
            )

        # Tenure features
        df["tenure_cat"] = pd.cut(
            df["tenure_months"],
            bins=[0, 12, 36, 72],
            labels=["new", "regular", "loyal"],
        )
        df["tenure_log"] = np.log1p(df["tenure_months"])

        # Composite scores
        df["customer_lifetime_value_proxy"] = (
            df["avg_order_value"] * df["visit_frequency"] * df["tenure_months"]
        )
        df["wealth_score"] = df["income_quantile"] * df["spending_score"]

        return df

    def engineer_demand_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Engineer features for SKU demand forecasting.

        Args:
            df: Demand data with columns: week, department, quantity_sold, etc.

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Time features
        df["week_number"] = df["week"]
        df["week_of_year"] = df["week"] % 52 + 1
        df["month"] = ((df["week"] - 1) // 4) + 1
        df["quarter"] = (df["week"] - 1) // 13 + 1
        df["is_weekend"] = False  # Placeholder

        # Cyclical time encoding
        df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Holiday features
        df["holiday_encoded"] = df["is_holiday"].astype(int)
        df["holiday_sin"] = np.sin(2 * np.pi * df["holiday_encoded"])
        df["holiday_cos"] = np.cos(2 * np.pi * df["holiday_encoded"])

        # Lag features (for time series)
        for lag in [1, 4, 8]:
            df[f"quantity_lag_{lag}"] = df.groupby(
                ["department", "product_group", "sku"]
            )["quantity_sold"].shift(lag)

        # Rolling statistics
        for window in [4, 8, 12]:
            df[f"quantity_rolling_mean_{window}"] = df.groupby(
                ["department", "product_group", "sku"]
            )["quantity_sold"].transform(
                lambda x: x.shift(1).rolling(window=window).mean()
            )
            df[f"quantity_rolling_std_{window}"] = df.groupby(
                ["department", "product_group", "sku"]
            )["quantity_sold"].transform(
                lambda x: x.shift(1).rolling(window=window).std()
            )

        # Price/discount features
        df["discount_flag"] = (df["discount_pct"] > 0).astype(int)
        df["discount_log"] = np.log1p(df["discount_pct"])

        # Revenue per unit
        df["revenue_per_unit"] = df["revenue"] / (df["quantity_sold"] + 1)

        # Inventory turnover proxy
        df["inventory_turnover"] = df["quantity_sold"] / (df["inventory_level"] + 1)

        return df.dropna()

    def engineer_news_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Engineer features for news article analysis.

        Args:
            df: News data with columns: title, category, sentiment_score, etc.

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Title text features
        df["title_length"] = df["title"].str.len()
        df["word_count"] = df["title"].str.split().str.len()
        df["title_upper_ratio"] = df["title"].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x)
        )
        df["title_exclaim_ratio"] = df["title"].apply(
            lambda x: x.count("!") / max(len(x), 1)
        )
        df["title_question_ratio"] = df["title"].apply(
            lambda x: x.count("?") / max(len(x), 1)
        )

        # Category encoding
        df["category_encoded"] = df["category"].map({
            "tech": 0, "finance": 1, "ai": 2,
            "sports": 3, "policy": 4, "climate": 5,
        })

        # Sentiment bins
        df["sentiment_bin"] = pd.cut(
            df["sentiment_score"],
            bins=[-1, 0.3, 0.5, 0.7, 1.0],
            labels=["negative", "neutral", "positive", "very_positive"],
        )

        # Engagement bins
        df["engagement_bin"] = pd.cut(
            df["engagement_score"],
            bins=[0, 5, 7, 9, 10],
            labels=["low", "medium", "high", "very_high"],
        )

        # Combined score
        df["popularity_score"] = (
            df["sentiment_score"] * 0.4 +
            (df["engagement_score"] / 10) * 0.6
        )

        return df

    def engineer_chap_features(
        self,
        n_agents: int,
        scenario: str = "traffic",
    ) -> pd.DataFrame:
        """Engineer features for agent-based simulation.

        Args:
            n_agents: Number of agents
            scenario: Simulation scenario

        Returns:
            DataFrame with agent features
        """
        np.random.seed(42)

        # Agent types
        agent_types = ["human", "robot", "vehicle"]
        behaviors = ["reactive", "deliberative", "hybrid"]

        agents = []
        for i in range(n_agents):
            agent_type = np.random.choice(agent_types)
            behavior = np.random.choice(behaviors)

            # Scenario-specific attributes
            if scenario == "traffic":
                speed = np.random.uniform(20, 120)
                position_x = np.random.uniform(0, 1000)
                position_y = np.random.uniform(0, 1000)
            elif scenario == "crowd":
                speed = np.random.uniform(0.5, 2.0)
                position_x = np.random.uniform(0, 100)
                position_y = np.random.uniform(0, 100)
            else:  # market
                speed = np.random.uniform(0.1, 1.0)
                position_x = np.random.uniform(0, 1000)
                position_y = np.random.uniform(0, 1000)

            agents.append({
                "agent_id": f"AG{i:04d}",
                "agent_type": agent_type,
                "behavior": behavior,
                "speed": round(speed, 2),
                "position_x": round(position_x, 2),
                "position_y": round(position_y, 2),
                "energy": round(np.random.uniform(0, 100), 2),
                "social_range": round(np.random.uniform(5, 50), 2),
                "perception_range": round(np.random.uniform(10, 100), 2),
            })

        return pd.DataFrame(agents)

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        scale: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Convert DataFrame to feature matrix.

        Args:
            df: Input DataFrame
            target_col: Column to exclude (target)
            scale: Whether to scale numerical features

        Returns:
            Tuple of (features, target, feature_names)
        """
        if target_col is not None:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df
            y = None

        # Get feature names
        feature_names = list(X.columns)

        # One-hot encode categorical features
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        numerical_cols = X.select_dtypes(include=["number"]).columns

        X_encoded = X.copy()

        if len(categorical_cols) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = encoder.fit_transform(X[categorical_cols])
            encoded_cols = encoder.get_feature_names_out(categorical_cols)
            X_encoded[categorical_cols] = encoded

        X_final = X_encoded[numerical_cols].values

        if scale and len(numerical_cols) > 0:
            self._scaler = StandardScaler()
            X_final = self._scaler.fit_transform(X_final)

        return X_final, y.values if y is not None else None, list(numerical_cols)
