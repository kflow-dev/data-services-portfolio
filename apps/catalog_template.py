"""Catalog template for data science portfolio apps.

This template provides a structured approach for creating the public API
of each app following MLOps best practices.

Usage:
    1. Copy this file to your app directory (e.g., apps/myapp/catalog.py)
    2. Replace [AppClassName], [AppFeature], etc. with your app's specifics
    3. Implement the core ML logic in the class methods
    4. Update docstrings and type hints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# RESULT DATACLASSES
# ============================================================================

@dataclass
class PredictionResult:
    """Container for prediction results."""
    predictions: pd.DataFrame
    confidence_intervals: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "predictions": self.predictions.to_dict(orient="records"),
            "confidence_intervals": (
                self.confidence_intervals.to_dict(orient="records")
                if self.confidence_intervals is not None else None
            ),
            "metadata": self.metadata,
        }


@dataclass
class EvaluationResult:
    """Container for model evaluation results."""
    metrics: Dict[str, float]
    feature_importance: Optional[pd.DataFrame] = None
    confusion_matrix: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics,
            "feature_importance": (
                self.feature_importance.to_dict(orient="records")
                if self.feature_importance is not None else None
            ),
        }


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

class DatasetConfig(Protocol):
    """Protocol for dataset configuration."""

    name: str
    description: str
    file_pattern: str
    required_columns: List[str]
    optional_columns: List[str]
    target_column: Optional[str]
    feature_columns: Optional[List[str]]


# ============================================================================
# MAIN APP CLASS
# ============================================================================

class [AppClassName](Protocol):
    """Main class for [App Name] application.

    This class provides the public API for the application following
    MLOps best practices:
    - Separation of concerns (data, model, evaluation)
    - Type hints for better IDE support
    - Comprehensive docstrings
    - Configurable hyperparameters

    Attributes:
        config: Dataset configuration
        model: Trained model instance
        metrics: Last evaluation metrics

    Example:
        >>> app = [AppClassName](n_estimators=100, max_depth=5)
        >>> app.train(data_path="data/customers.csv")
        >>> predictions = app.predict(new_data)
        >>> results = app.evaluate()
    """

    def __init__(
        self,
        # Add your hyperparameters here
        random_state: int = 42,
        **kwargs: Any,
    ):
        """Initialize [App Name].

        Args:
            random_state: Random seed for reproducibility
            **kwargs: Additional configuration parameters
        """
        self.random_state = random_state
        self.config: Optional[DatasetConfig] = None
        self._model: Any = None
        self._metrics: Dict[str, float] = {}

    def load_data(
        self,
        data_path: str,
        **load_kwargs: Any,
    ) -> pd.DataFrame:
        """Load data from file.

        Args:
            data_path: Path to data file
            **load_kwargs: Additional pandas read parameters

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If data file not found
            ValueError: If required columns missing
        """
        import pandas as pd
        from pathlib import Path

        filepath = Path(data_path)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath, **load_kwargs)

        # Validate required columns
        if self.config and self.config.required_columns:
            missing = set(self.config.required_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        return df

    def train(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **train_kwargs: Any,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            X: Feature dataframe
            y: Target series (optional for unsupervised)
            **train_kwargs: Training hyperparameters

        Returns:
            Training results with metrics

        Raises:
            ValueError: If model already trained
        """
        if self._model is not None:
            raise ValueError("Model already trained. Create new instance to retrain.")

        # Implement training logic here
        # Example using scikit-learn:
        # from sklearn.ensemble import RandomForestClassifier
        # self._model = RandomForestClassifier(**self.hyperparams)
        # self._model.fit(X, y)

        return {
            "status": "trained",
            "n_samples": len(X),
            "n_features": len(X.columns),
        }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions.

        Args:
            X: Feature dataframe

        Returns:
            DataFrame with predictions

        Raises:
            ValueError: If model not trained
        """
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Implement prediction logic here
        predictions = self._model.predict(X)

        return pd.DataFrame(predictions, columns=["prediction"])

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **eval_kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate model performance.

        Args:
            X: Feature dataframe for evaluation
            y: True labels
            **eval_kwargs: Evaluation parameters

        Returns:
            EvaluationResult with metrics
        """
        from sklearn.metrics import accuracy_score, f1_score

        y_pred = self.predict(X)

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred, average="weighted"),
        }

        self._metrics = metrics

        return EvaluationResult(metrics=metrics)

    def get_metrics(self) -> Dict[str, float]:
        """Get last evaluation metrics.

        Returns:
            Dictionary of metrics
        """
        return self._metrics.copy()

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path to save model
        """
        import joblib

        joblib.dump(self._model, path)

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path to load model
        """
        import joblib

        self._model = joblib.load(path)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_app(
    app_type: str,
    **kwargs: Any,
) -> [AppClassName]:
    """Factory function for creating app instances.

    Args:
        app_type: Type of app to create
        **kwargs: App-specific parameters

    Returns:
        Configured app instance
    """
    # Implement factory logic if you have multiple app variants
    return [AppClassName](**kwargs)
