"""SKU Forecaster - Catalog module.

This module provides the public API for the SKU Forecaster app.
It follows MLOps best practices with proper separation of concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ForecastResult:
    """Forecast result container."""
    department: str
    product_group: str
    horizon: int
    predictions: pd.DataFrame
    model_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "department": self.department,
            "product_group": self.product_group,
            "horizon": self.horizon,
            "predictions": self.predictions.to_dict(orient="records"),
            "model_metrics": self.model_metrics,
        }


class SKUForecaster:
    """Hierarchical demand forecasting model."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        """Initialize forecaster.

        Args:
            n_estimators: Number of boosting stages
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._model: Optional[GradientBoostingRegressor] = None
        self._metrics: Dict[str, float] = {}

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Train the forecasting model.

        Args:
            df: Demand data with columns week, department, product_group, quantity_sold
            test_size: Proportion of data for testing

        Returns:
            Training results with metrics
        """
        # Create features
        df = df.copy()
        df["is_holiday"] = df["week"].isin([7, 11, 24, 35, 47])
        df["lag_1_dept"] = df.groupby("department")["quantity_sold"].shift(1)
        df["lag_4_dept"] = df.groupby("department")["quantity_sold"].shift(4)

        df_clean = df.dropna()

        feature_cols = ["week", "is_holiday", "discount_pct", "lag_1_dept", "lag_4_dept"]
        X = df_clean[feature_cols]
        y = df_clean["quantity_sold"]

        # Train-test split
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Train model
        self._model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self._model.fit(X_train, y_train)

        # Evaluate
        y_pred = self._model.predict(X_test)
        self._metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        }

        return {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "metrics": self._metrics,
        }

    def forecast(
        self,
        df: pd.DataFrame,
        horizon: int = 12,
    ) -> pd.DataFrame:
        """Generate forecast for next N weeks.

        Args:
            df: Historical demand data
            horizon: Number of weeks to forecast

        Returns:
            DataFrame with forecasts and confidence intervals
        """
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")

        last_week = df["week"].max()
        avg_demand = df["quantity_sold"].mean()

        forecasts = []
        for week in range(last_week + 1, last_week + horizon + 1):
            is_holiday = week in [7, 11, 24, 35, 47, 59, 71]
            discount_pct = np.random.choice([0, 5, 10], p=[0.5, 0.3, 0.2])

            features = [[
                week, is_holiday, discount_pct,
                avg_demand * 0.8,
                avg_demand * 0.7,
            ]]

            prediction = self._model.predict(features)[0]

            # Confidence interval (simplified)
            ci_width = prediction * 0.2
            forecasts.append({
                "week": week,
                "forecast": round(prediction, 0),
                "lower_80": round(prediction - ci_width * 0.84, 0),
                "upper_80": round(prediction + ci_width * 0.84, 0),
                "lower_95": round(prediction - ci_width * 1.96, 0),
                "upper_95": round(prediction + ci_width * 1.96, 0),
            })

        return pd.DataFrame(forecasts)

    def predict(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Predict demand for given features.

        Args:
            df: DataFrame with required feature columns

        Returns:
            Predicted demand values
        """
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self._model.predict(df)

    def get_metrics(self) -> Dict[str, float]:
        """Get model performance metrics."""
        return self._metrics.copy()
